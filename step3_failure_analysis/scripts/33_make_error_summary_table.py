#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
33 —— 把 30_error_decomposition.py 输出的若干 *_error_summary.json 拼成一张
中文 markdown 表，直接贴进 step3_failure_analysis/FINDINGS.md 的 §4。
只处理已经存在的 json，所以可以随时重复运行。
"""
import json
import os

STEP3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(STEP3, "results")
OUT_MD = os.path.join(RES, "three_way_error_table.md")

# 展示顺序：tag -> 中文名
ORDER = [
    ("unet_73", "U-Net 基线"),
    ("nnunet_2d_fold0_checkpoint_final", "nnU-Net 2d（final）"),
    ("nnunet_2d_fold0_checkpoint_best", "nnU-Net 2d（best）"),
    ("our_resunet_boundary", "**我们的模型**（final）"),
    ("our_resunet_boundary_best", "**我们的模型**（best）"),
    ("ensemble_ours_and_nnunet", "**我们 + nnU-Net 集成**"),
]

# (键, 中文名, 格式, 越大越好?)  None 表示不比较方向
ROWS = [
    ("n", "验证图数", "%d", None),
    ("dice_mean", "Dice", "%.3f", True),
    ("bdice_1px_mean", "边界 Dice@1px", "%.2f", True),
    ("bdice_2px_mean", "边界 Dice@2px", "%.2f", True),
    ("dice_small_glands", "小腺体 Dice", "%.2f", True),
    ("dice_large_glands", "大腺体 Dice", "%.2f", True),
    ("corr_area_dice", "corr(面积, Dice)", "%+.2f", True),
    ("fn_px_mean", "漏检 FN（像素/图）", "%.0f", False),
    ("fp_px_mean", "误检 FP（像素/图）", "%.0f", False),
    ("fn_share_thin_2px", "FN 中「紧贴边界 ≤2px」占比 %", "%.1f", None),
    ("fn_share_gross_10px", "FN 中「远离边界 >10px」占比 %", "%.1f", None),
    ("fp_share_thin_2px", "FP 中「紧贴边界 ≤2px」占比 %", "%.1f", None),
    ("fp_share_gross_10px", "FP 中「远离边界 >10px」占比 %", "%.1f", None),
    ("n_pred_cc_mean", "预测连通域个数", "%.1f", False),
    ("largest_cc_frac_mean", "最大连通域像素占比 %", "%.1f", None),
    ("holes_px_mean", "空洞像素", "%.0f", False),
    ("fn_near_iface_pct", "127/255 界面附近 FN 率 %", "%.1f", False),
    ("fp_near_iface_pct", "127/255 界面附近 FP 率 %", "%.1f", False),
    ("gt_near_iface_pct", "界面邻域占 GT 像素 %", "%.1f", None),
    ("dice_gain_postproc", "后处理带来的 Dice 增益", "%+.3f", True),
    ("dice_largest_cc_mean", "只留最大连通域后 Dice", "%.3f", True),
    ("dice_filled_mean", "填洞后 Dice", "%.3f", True),
    ("dice_after_postproc_mean", "去小连通域+填洞后 Dice", "%.3f", True),
]


def fmt(v, f):
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:  # NaN
        return "—"
    return f % v


def main():
    data, labels = {}, []
    for tag, label in ORDER:
        p = os.path.join(RES, "%s_error_summary.json" % tag)
        if os.path.exists(p):
            with open(p) as fh:
                data[tag] = json.load(fh)
            labels.append((tag, label))
    if not data:
        print("[33] no *_error_summary.json found in %s" % RES)
        return 1

    base = data.get("unet_73")
    has_base = base is not None

    lines = ["# 三路误差分解对比（同一脚本、同一把尺子、同一 fold 0 的 31 张验证图）", ""]
    head = ["指标"] + [lb for _, lb in labels]
    if has_base and len(labels) > 1:
        head.append("相对 U-Net 基线")
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))

    for key, name, f, higher_better in ROWS:
        cells = [name]
        vals = []
        for tag, _ in labels:
            v = data[tag].get(key)
            vals.append(v)
            cells.append(fmt(v, f))
        if has_base and len(labels) > 1:
            b = base.get(key)
            diffs = []
            for tag, _ in labels:
                if tag == "unet_73":
                    continue
                v = data[tag].get(key)
                if b is None or v is None or (isinstance(v, float) and v != v) \
                        or (isinstance(b, float) and b != b):
                    diffs.append("—")
                    continue
                d = v - b
                if higher_better is None:
                    diffs.append("%+.3g" % d)
                else:
                    better = (d > 0) if higher_better else (d < 0)
                    mark = "▲" if better and abs(d) > 1e-9 else ("▼" if abs(d) > 1e-9 else "=")
                    diffs.append("%s%+.3g" % (mark, d))
            cells.append(" / ".join(diffs))
        lines.append("| " + " | ".join(cells) + " |")

    # FN 率随到边界距离的变化（5 个 band）
    lines.append("")
    lines.append("**漏检率 FN-rate 随「到真值边界的距离」的变化（0-2 / 2-5 / 5-10 / 10-20 / >20 px）**")
    lines.append("")
    lines.append("| 模型 | " + " | ".join(["0-2px", "2-5px", "5-10px", "10-20px", ">20px"]) + " |")
    lines.append("|---|---|---|---|---|---|")
    for tag, label in labels:
        band = data[tag].get("fn_rate_by_band") or []
        cells = ["%.1f%%" % x if x is not None else "—" for x in band[:5]]
        while len(cells) < 5:
            cells.append("—")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")

    # 按分辨率分组
    lines.append("")
    lines.append("**按图像分辨率分组的 Dice**")
    lines.append("")
    lines.append("| 模型 | " + " | ".join(sorted({k for tag, _ in labels
                                              for k in (data[tag].get("dice_by_resolution") or {})})) + " |")
    res_keys = sorted({k for tag, _ in labels for k in (data[tag].get("dice_by_resolution") or {})})
    lines.append("|" + "---|" * (len(res_keys) + 1))
    for tag, label in labels:
        d = data[tag].get("dice_by_resolution") or {}
        lines.append("| " + label + " | " + " | ".join(
            ("%.2f" % d[k]) if k in d else "—" for k in res_keys) + " |")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[33] wrote %s" % OUT_MD)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
