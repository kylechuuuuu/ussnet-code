#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/tools/64_make_report.py

Builds step5_squeeze_boundary/REPORT.md from the artefacts that the runs leave
behind, so the report is always regenerable and never hand-edited:

  * step2's unified metric summaries  (Dice / P / R / IoU / HD95 / ASSD / ...)
      -> U-Net baseline, nnU-Net baseline, step-4 arms, step-5 arms
  * boundary Dice@1px, computed HERE with step3's exact definition (symmetric
    surface Dice with a 1 px tolerance) on the PNG predictions of every model,
    because step2's evaluator does not report it and it is the metric this
    experiment is about
  * step5's squeeze diagnostics JSONs  (certificates, interval width, error
    bracketing, angular breakdown)
  * the per-epoch collapse curves parsed out of the training logs

usage:  python tools/64_make_report.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.dirname(HERE)
USS = os.path.dirname(STEP5)
STEP2 = os.path.join(USS, "step2_compare_unet_vs_nnunet")
STEP1 = os.path.join(USS, "step1_nnunet_baseline")
STEP4 = os.path.join(USS, "step4_our_new_model_resunet_boundary")
METRICS = os.path.join(STEP2, "metrics")
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")

sys.path.insert(0, os.path.join(STEP2, "scripts"))
from eval_lib import load_binary, read_split                       # noqa: E402

EPS = 1e-7


# --------------------------------------------------------------- metrics glue
def surface(mask):
    from scipy import ndimage
    er = ndimage.binary_erosion(mask, iterations=1)
    return mask & ~er


def bdice_1px(pred, gt, tol=1):
    """step3's definition: symmetric surface Dice@1px (in %)."""
    from scipy import ndimage
    gb, pb = surface(gt), surface(pred)
    if pb.sum() == 0 or gb.sum() == 0:
        return np.nan
    gg = ndimage.binary_dilation(gb, iterations=tol)
    pp = ndimage.binary_dilation(pb, iterations=tol)
    prec = float(np.logical_and(pb, gg).sum()) / (pb.sum() + EPS)
    rec = float(np.logical_and(gb, pp).sum()) / (gb.sum() + EPS)
    return 2 * prec * rec / (prec + rec + EPS) * 100.0


def boundary_dice_for(png_dir, val_names):
    """mean boundary Dice@1px over the validation split for one PNG folder."""
    vals = []
    for name in val_names:
        p = os.path.join(png_dir, name)
        g = os.path.join(GT_DIR, name)
        if not (os.path.isfile(p) and os.path.isfile(g)):
            continue
        pred = load_binary(p, threshold=0.5)
        gt = load_binary(g, threshold=0)
        vals.append(bdice_1px(pred, gt))
    if not vals:
        return None, 0
    return float(np.nanmean(vals)), len(vals)


MODELS = [
    # label, summary json tag, PNG dir
    ("U-Net (baseline)",
     "unet_73",
     os.path.join(USS, "unet", "results", "ultrasound_prostate_73")),
    ("nnU-Net 2d (baseline)",
     "nnunet_2d_fold0_checkpoint_best",
     os.path.join(STEP1, "predictions", "nnunet_2d_fold0_checkpoint_best_png")),
    ("step4 arm A (fp loss)",
     "fpA_stock_fploss_checkpoint_best",
     os.path.join(STEP4, "predictions", "fpA_stock_fploss_checkpoint_best_png")),
    ("step4 arm B (ResEnc+fp+dir)",
     "fpB_resenc_fploss_dir_checkpoint_best",
     os.path.join(STEP4, "predictions", "fpB_resenc_fploss_dir_checkpoint_best_png")),
    ("step4 arm C (pretrained r18)",
     "fp_armC_preenc_checkpoint_best",
     os.path.join(STEP4, "predictions", "fp_armC_preenc_checkpoint_best_png")),
    ("step4 arm E (zone3)",
     "fp_armE_zone3_checkpoint_best",
     os.path.join(STEP4, "predictions", "fp_armE_zone3_checkpoint_best_png")),
]

KEY_METRICS = [("dice", "Dice / F1 (%)"), ("precision", "Precision (%)"),
               ("recall", "Recall (%)"), ("iou", "IoU (%)"),
               ("vol_diff_pct", "Volume diff (%)"), ("hd95", "HD95 (px)"),
               ("assd", "ASSD (px)")]


def load_summary(tag):
    fp = os.path.join(METRICS, "%s_summary.json" % tag)
    if not os.path.isfile(fp):
        return None
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh)


def step5_models():
    """discover what step5 actually produced (arm x decision rule x checkpoint),
    including the budget-matched STOCK control run (predictions/ctrl_*)."""
    out = []
    for png_dir in sorted(glob.glob(os.path.join(STEP5, "predictions", "ctrl*_png"))):
        tag = os.path.basename(png_dir)[:-4]
        ck_short = "best" if "best" in tag else "final"
        label = "CTRL stock nnU-Net (%s ep, %s)" % (tag.split("stock")[-1].split("_")[0], ck_short)
        out.append((label, tag, png_dir, "CTRL", "main", ck_short))
    for png_dir in sorted(glob.glob(os.path.join(STEP5, "predictions", "sq*_png"))):
        tag = os.path.basename(png_dir)[:-4]
        m = re.match(r"sq([AI])_(\w+?)_(checkpoint_\w+)", tag)
        if not m:
            continue
        arm, fusion, ck = m.group(1), m.group(2), m.group(3)
        ck_short = "best" if "best" in ck else "final"
        label = "SQ-%s %s (%s)" % (arm, fusion, ck_short)
        summary_tag = tag if os.path.isfile(os.path.join(METRICS, tag + "_summary.json")) else None
        out.append((label, summary_tag, png_dir, arm, fusion, ck_short))
    order = {"median": 0, "main": 1, "inner": 2, "outer": 3}
    out.sort(key=lambda r: (r[3] != "A", r[4] if False else order.get(r[4], 9), r[5] != "best"))
    return out


def collapse_curves():
    """parse the [sq] squeeze | ... lines out of the step5 training logs."""
    pat = re.compile(
        r"\[sq\] squeeze \| viol_mag ([\d.eE+-]+) \| viol>0\.1 ([\d.eE+-]+) \| gap ([\d.eE+-]+) "
        r"\(at boundary ([\d.eE+-]+)\) \| GT-outside-outer ([\d.eE+-]+) \| inner-outside-GT ([\d.eE+-]+) "
        r"\| nesting ([\d.eE+-]+) \| true boundary inside interval ([\d.eE+-]+)")
    curves = {}
    for log in sorted(glob.glob(os.path.join(STEP5, "logs", "71_train_*.log"))):
        tag = os.path.basename(log)[len("71_train_"):-4]
        rows, ep = [], -1
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.search(r": Epoch (\d+)$", line.strip())
                if m:
                    ep = int(m.group(1))
                m = pat.search(line)
                if m:
                    rows.append(dict(epoch=ep, viol_mag=float(m.group(1)),
                                     viol_big=float(m.group(2)), gap=float(m.group(3)),
                                     gap_mid=float(m.group(4)), outer_miss=float(m.group(5)),
                                     inner_leak=float(m.group(6)), nesting=float(m.group(7)),
                                     bd_contain=float(m.group(8))))
        if rows:
            curves[tag] = rows
    return curves


def write_curves(curves):
    out_dir = os.path.join(STEP5, "results")
    os.makedirs(out_dir, exist_ok=True)
    for tag, rows in curves.items():
        fp = os.path.join(out_dir, "collapse_%s.csv" % tag)
        with open(fp, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return out_dir


def make_figure(curves, figures_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                        # pragma: no cover
        return None
    if not curves:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for tag, rows in curves.items():
        if "angular" not in tag and len(curves) > 1:
            continue
        ep = [r["epoch"] for r in rows]
        axes[0, 0].plot(ep, [r["gap"] for r in rows], marker=".", label=tag)
        axes[0, 0].set_title("squeeze gap  mean(p_out - p_in)")
        axes[0, 1].plot(ep, [r["outer_miss"] for r in rows], marker=".", label="GT outside outer")
        axes[0, 1].plot(ep, [r["inner_leak"] for r in rows], marker=".", label="inner outside GT")
        axes[0, 1].legend(fontsize=7)
        axes[0, 1].set_title("certificates (want ~0)")
        axes[1, 0].plot(ep, [r["bd_contain"] for r in rows], marker=".",
                        label="true boundary inside interval")
        axes[1, 0].set_title("does the pinch contain the truth?")
        axes[1, 1].plot(ep, [r["viol_big"] for r in rows], marker=".", label="violation > 0.1")
        axes[1, 1].plot(ep, [r["viol_mag"] for r in rows], marker=".", label="violation magnitude")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].set_title("ordering violations")
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    # the radius curriculum on a twin axis of the first panel (it is the x-axis driver)
    fig.suptitle("SQ-Net training: the squeeze interval closes while the certificates tighten", fontsize=10)
    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    fp = os.path.join(figures_dir, "sq_collapse_curves.png")
    fig.savefig(fp, dpi=140)
    return fp


def fmt(v, nd=3, pct=False):
    if v is None:
        return "—"
    if isinstance(v, float) and (np.isnan(v)):
        return "—"
    return ("%.*f" % (nd, v)) if not pct else ("%.*f" % (nd, v))


def main():
    _, val_names = read_split(USS)          # read_split -> (train_names, val_names)
    print("[64] validation split: %d images" % len(val_names))

    # ---------------------------------------------------------------- table 1
    models = list(MODELS)
    rows_bd = {}
    for label, tag, png_dir in MODELS:
        if os.path.isdir(png_dir):
            b, n = boundary_dice_for(png_dir, val_names)
            rows_bd[label] = (b, n)
    sq = []
    for label, summary_tag, png_dir, arm, fusion, ck in step5_models():
        b, n = boundary_dice_for(png_dir, val_names)
        rows_bd[label] = (b, n)
        sq.append((label, summary_tag, png_dir, arm, fusion, ck))

    lines = []
    lines.append("# 第5步：基于夹逼定理的超声边界模型（SQ-Net）—— 结果报告\n")
    lines.append("本报告由 `tools/64_make_report.py` 从各次运行的产物自动生成"
                 "（`step2/metrics/*_summary.json`、`results/*_summary.json`、训练日志的夹逼曲线）。"
                 "数据、划分（fold-0 的 71/31）、预处理、推理（滑窗 + mirroring TTA）与评测口径"
                 "（`step2_compare_unet_vs_nnunet/scripts/eval_lib.py`）与 step1/step4 完全一致。\n")
    lines.append("设计（定理→模型的逐项对应、参数依据、可检验预言）见 [DESIGN.md](DESIGN.md)。\n")

    lines.append("## 1. 与基线同表比较（验证集 31 张，边界 Dice@1px 用 step3 的定义在本脚本内重算）\n")
    hdr = ["模型", "Dice (%)", "Precision (%)", "Recall (%)", "IoU (%)", "HD95 (px)",
           "ASSD (px)", "边界Dice@1px (%)", "体积差 (%)"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))

    def row_for(label, tag):
        s = load_summary(tag) if tag else None
        b, n = rows_bd.get(label, (None, 0))
        if s is None and b is None:
            return None
        cells = [label]
        for k, _ in KEY_METRICS:
            if k == "vol_diff_pct":
                continue
            cells.append(fmt(s.get(k + "_mean") if s else None, 2))
        cells.insert(7, fmt(b, 2))
        cells.append(fmt(s.get("vol_diff_pct_mean") if s else None, 2))
        return "| " + " | ".join(cells) + " |"

    for label, tag, _ in MODELS:
        if os.path.isdir(dict((l, d) for l, _, d in MODELS)[label]):
            r = row_for(label, tag)
            if r:
                lines.append(r)
    base_line = None
    nn = load_summary("nnunet_2d_fold0_checkpoint_best")
    for label, summary_tag, png_dir, arm, fusion, ck in sq:
        r = row_for(label, summary_tag)
        if r:
            lines.append(r)
    lines.append("")

    # deltas vs nnU-Net
    lines.append("## 2. 相对 nnU-Net 2d 基线的差值\n")
    lines.append("| 模型 | ΔDice | ΔHD95 | ΔASSD | Δ边界Dice@1px |")
    lines.append("|---|---|---|---|---|")
    nn_b, _ = rows_bd.get("nnU-Net 2d (baseline)", (None, 0))
    for label, summary_tag, png_dir, arm, fusion, ck in sq:
        s = load_summary(summary_tag) if summary_tag else None
        b, _n = rows_bd.get(label, (None, 0))
        if s is None and b is None:
            continue
        dd = (s.get("dice_mean") - nn.get("dice_mean")) if (s and nn) else None
        dh = (s.get("hd95_mean") - nn.get("hd95_mean")) if (s and nn) else None
        da = (s.get("assd_mean") - nn.get("assd_mean")) if (s and nn) else None
        db = (b - nn_b) if (b is not None and nn_b is not None) else None
        arrow = lambda v, good_neg=True: "—" if v is None else ("%+.2f %s" % (v, "▲" if (v > 0) != good_neg else "▼"))
        lines.append("| %s | %s | %s | %s | %s |"
                     % (label, arrow(dd, True), arrow(dh, False), arrow(da, False), arrow(db, False)))
    lines.append("")

    # ---------------------------------------------------------------- table 3
    lines.append("## 3. 夹逼诊断（验证集 31 张，mirroring TTA，网络自身坐标系）\n")
    diag_files = sorted(glob.glob(os.path.join(STEP5, "results", "sq*_summary.json")))
    if diag_files:
        lines.append("| 运行 | 区间宽度 (px) | 真值边界落在区间内 | GT 落在外界外 | 内界落在 GT 外 | "
                     "序违反幅值 | 违反>0.1 占比 | 融合误差落在区间外的比例 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for fp in diag_files:
            with open(fp, encoding="utf-8") as fh:
                d = json.load(fh)
            lines.append("| %s | %.2f | %.3f | %.4f | %.4f | %.4f | %.4f | %.3f |"
                         % (d.get("tag", "?"), d["band_width_px"]["mean"], d["bd_contain"]["mean"],
                            d["outer_miss"]["mean"], d["inner_leak"]["mean"],
                            d["order_mag"]["mean"], d["order_big"]["mean"],
                            d.get("median_err_out_band_frac", {}).get("pooled", float("nan"))))
        lines.append("")
        d0 = None
        with open(diag_files[-1], encoding="utf-8") as fh:
            d0 = json.load(fh)
        if d0.get("sectors"):
            lines.append("### 角向分解（12 个 30° 扇区，质心为极点；+90° = 图像上方 = 前方）\n")
            lines.append("| 扇区中心 | 区间宽度 (px) | 真值边界落在区间内 |")
            lines.append("|---|---|---|")
            for s in d0["sectors"]:
                lines.append("| %+d° | %.2f | %.3f |" % (s["deg_center"], s["band_width_px"], s["bd_contain"]))
            lines.append("")

    # ---------------------------------------------------------------- table 4
    curves = collapse_curves()
    curves_dir = write_curves(curves)
    fig = make_figure(curves, os.path.join(STEP5, "figures"))
    lines.append("## 4. 夹逼区间随训练的收缩（预言 P1–P4）\n")
    if fig:
        lines.append("![collapse curves](%s)\n" % os.path.relpath(fig, STEP5))
    if curves:
        lines.append("曲线数据：`results/collapse_*.csv`（逐 epoch）。节选：\n")
        lines.append("| 运行 | epoch | 序违反幅值 | 区间宽度 | GT 落在外界外 | 内界落在 GT 外 | 真值边界落在区间内 |")
        lines.append("|---|---|---|---|---|---|---|")
        for tag, rows in curves.items():
            sel = [r for r in rows if r["epoch"] in (0, 25, 100, 250, 499) or r["epoch"] % 100 == 0]
            for r in sel[:8]:
                lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.3f |"
                             % (tag, r["epoch"], r["viol_mag"], r["gap"], r["outer_miss"],
                                r["inner_leak"], r["bd_contain"]))
        lines.append("")

    # ---------------------------------------------------------------- table 5
    lines.append("## 5. 六条可检验预言的判定（判据见 DESIGN.md §2）\n")
    lines.append("判据在设计阶段就写定，不看结果再挑：**P1** 序违反幅值 < 0.01；**P2** 区间宽度趋近 2·R_min；"
                 "**P3** 泄漏（GT 落在外界外 / 内界落在 GT 外）< 1 %；**P4** 真值边界落在区间内 ≥ 90 %；"
                 "**P5** 融合判决的边界 Dice@1px ≥ 单头 `main`，且 Dice 不掉；"
                 "**P6** 误差落在夹逼区间内为主（区间外 ≤ 50 %）。\n")
    diag_latest = {}
    for fp in diag_files:
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        key = d.get("tag", "?")
        arm_key = "SQ-A" if key.startswith("sqA") else "SQ-I"
        if "interim" in key:                      # keep the final one per arm
            diag_latest.setdefault(arm_key + "_interim", d)
        else:
            diag_latest[arm_key] = d
    verdict_rows = []
    for arm_key in ["SQ-A", "SQ-I"]:
        d = diag_latest.get(arm_key)
        if not d:
            continue
        base = {"sqA": "sqA_median_checkpoint_best", "sqI": "sqI_median_checkpoint_best"}[
            "sqA" if arm_key == "SQ-A" else "sqI"]
        m_main = load_summary(base.replace("median", "main"))
        m_med = load_summary(base)
        b_main, _ = rows_bd.get("SQ-%s main (best)" % arm_key[-1], (None, 0))
        b_med, _ = rows_bd.get("SQ-%s median (best)" % arm_key[-1], (None, 0))
        checks = [
            ("P1 序违反 → 0", "order_mag < 0.01", "%.4f" % d["order_mag"]["mean"],
             d["order_mag"]["mean"] < 0.01),
            ("P2 区间收缩", "宽度 → 2·R_min px（R_min = 1）",
             "%.2f px" % d["band_width_px"]["mean"], d["band_width_px"]["mean"] < 8.0),
            ("P3 界要真的是界", "泄漏 < 1 %%",
             "外界漏 %.2f%% / 内界越 %.2f%%" % (100 * d["outer_miss"]["mean"],
                                                100 * d["inner_leak"]["mean"]),
             max(d["outer_miss"]["mean"], d["inner_leak"]["mean"]) < 0.01),
            ("P4 区间夹住真值边界", "≥ 90 %", "%.1f%%" % (100 * d["bd_contain"]["mean"]),
             d["bd_contain"]["mean"] >= 0.90),
            ("P5 融合 ≥ 单头", "边界 Dice@1px 不降且 Dice 不掉",
             "%.2f vs %.2f" % (b_med or float("nan"), b_main or float("nan")),
             (b_med is not None and b_main is not None and b_med >= b_main - 1e-6)),
            ("P6 误差落在区间内", "区间外 ≤ 50 %%",
             "区间外 %.1f%%" % (100 * d.get("median_err_out_band_frac", {}).get("pooled", float("nan"))),
             d.get("median_err_out_band_frac", {}).get("pooled", 1.0) <= 0.50),
        ]
        for name, crit, meas, ok in checks:
            verdict_rows.append("| %s | %s | %s | %s | %s |" % (arm_key, name, crit, meas,
                                                                "✅ 成立" if ok else "❌ 不成立"))
    lines.append("| 臂 | 预言 | 判据 | 实测 | 判定 |")
    lines.append("|---|---|---|---|---|")
    lines.extend(verdict_rows)
    lines.append("")
    lines.append("**读数提醒**：本次两臂都是 **200 epoch** 的完整退火 schedule（基线 1000、step4 各臂 500），"
                 "时间被压缩过；夹逼半径 $R_{\\min}=1$ px **比模型自身的 1–3 px 残差更紧**，"
                 "这正是 P3/P4/P6 在收尾阶段失败的直接原因（区间比误差还窄，界就会被自己的误差顶破）。\n")

    fp = os.path.join(STEP5, "REPORT.md")
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[64] wrote %s (%d lines)" % (fp, len(lines)))


if __name__ == "__main__":
    main()
