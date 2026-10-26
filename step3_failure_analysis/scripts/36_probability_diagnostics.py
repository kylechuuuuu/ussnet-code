#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36 —— 概率层面的三路诊断（U-Net / nnU-Net / 我们的模型），同一套代码、同一把尺子。

为什么需要它：只看 Dice 无法回答"问题到底在哪"。这里直接读每个模型输出的**概率图**
（U-Net 的 .npy、nnU-Net/我们的 .npz，都已经是原始图像尺寸），算：

  1. threshold sweep        阈值到底是不是杠杆（0.5 附近能榨出多少）
  2. ECE / 可靠性           概率校准（整体 + 只看真值边界附近）
  3. 分区置信                 背景 / 127 区 / 255 区 的平均概率与 >0.5 占比
  4. 边界锐度                真值边界环上的 |∇p|（边界越锐，越不会外溢）
  5. 不确定带宽              p∈[0.4,0.6] 的像素占比
  6. 错判处的置信             FN 像素、FP 像素上的平均概率（是"自信地错"还是"犹豫地错"）

用法（可多组 --source，格式 tag:目录；文件按 <原图名>.npy 或 <case_id>.npz 自动识别）：
  $HOME/anaconda3/envs/zk/bin/python scripts/36_probability_diagnostics.py \
      --source unet_73:results/unet_73_probabilities \
      --source nnunet_2d_fold0_checkpoint_final:../step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_final_nifti
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
STEP3 = os.path.dirname(HERE)
USS = os.path.dirname(STEP3)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import load_binary, read_split, _surface  # noqa: E402

GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
MAP_CSV = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")
RES_DIR = os.path.join(STEP3, "results")
EPS = 1e-7

LABELS = [("unet_73", "U-Net 基线"),
          ("nnunet_2d_fold0_checkpoint_final", "nnU-Net 2d（final）"),
          ("nnunet_2d_fold0_checkpoint_best", "nnU-Net 2d（best）"),
          ("our_resunet_boundary", "**我们的模型**（final）"),
          ("our_resunet_boundary_best", "**我们的模型**（best）"),
          ("ensemble_ours_and_nnunet", "**我们 + nnU-Net 集成**")]


def load_mapping():
    """name(原图) -> case_id, case_id -> name"""
    a, b = {}, {}
    if os.path.exists(MAP_CSV):
        with open(MAP_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                a[row["original_filename"]] = row["case_id"]
                b[row["case_id"]] = row["original_filename"]
    return a, b


def load_prob(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        p = np.load(path).astype(np.float32)
    elif ext == ".npz":
        with np.load(path) as z:
            key = "probabilities" if "probabilities" in z else list(z.keys())[0]
            p = np.asarray(z[key], dtype=np.float32)
    else:
        raise ValueError(path)
    p = np.squeeze(p)
    if p.ndim == 3:                      # (2,H,W) 前景通道
        p = p[1] if p.shape[0] == 2 else p[0]
    elif p.ndim != 2:
        raise ValueError("unexpected shape %s for %s" % (p.shape, path))
    return p


def find_prob(d, name, case_id):
    """在目录里按 原图名.npy / case_id.npz / *.nii.npz 找概率文件"""
    for cand in (os.path.join(d, name.replace(".png", ".npy")),
                 os.path.join(d, name + ".npy"),
                 os.path.join(d, name.replace(".png", "") + ".npz"),
                 os.path.join(d, case_id + ".npz"),
                 os.path.join(d, case_id + ".nii.npz"),
                 os.path.join(d, case_id + ".nii.gz.npz")):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(d, "*%s*.npz" % case_id)) + \
        glob.glob(os.path.join(d, "*%s*.npy" % case_id))
    return hits[0] if hits else None


def dice(a, b):
    return 2.0 * float(np.logical_and(a, b).sum()) / (float(a.sum()) + float(b.sum()) + EPS) * 100


def ece_of(p, g, bins=np.linspace(0, 1, 11), mask=None):
    pf, gf = p.ravel(), g.ravel()
    if mask is not None:
        mf = mask.ravel()
        pf, gf = pf[mf], gf[mf]
    idx = np.clip(np.digitize(pf, bins) - 1, 0, 9)
    conf = np.array([pf[idx == b].mean() if (idx == b).any() else 0.0 for b in range(10)])
    acc = np.array([gf[idx == b].mean() if (idx == b).any() else 0.0 for b in range(10)])
    cnt = np.array([(idx == b).sum() for b in range(10)], dtype=float)
    e = float((np.abs(conf - acc) * cnt).sum() / max(cnt.sum(), 1))
    return e, conf, acc, cnt


def analyze_case(p, gt, gray):
    m = {}
    pr = p >= 0.5
    m["dice"] = dice(pr, gt)
    m["iou"] = float(np.logical_and(pr, gt).sum()) / (float(np.logical_or(pr, gt).sum()) + EPS) * 100

    # 1. 阈值敏感性
    for t in (0.3, 0.4, 0.6, 0.7):
        m["dice@%.1f" % t] = dice(p >= t, gt)

    # 2. 校准
    e, conf, acc, cnt = ece_of(p, gt.astype(np.float32))
    m["ece"] = e
    m["bin_conf"], m["bin_acc"], m["bin_cnt"] = conf.tolist(), acc.tolist(), cnt.tolist()

    # 真值边界环
    er = ndimage.binary_erosion(gt, iterations=1)
    bd = gt & ~er
    ring = ndimage.binary_dilation(bd, iterations=2)
    er2 = ndimage.binary_erosion(ring, iterations=1)
    ring_only = ring & ~er2                       # 环带本身（约 3px 宽）
    m["ring_frac_of_gt"] = float(ring_only.sum()) / max(gt.sum(), 1) * 100
    e_ring, c_ring, a_ring, n_ring = ece_of(p, gt.astype(np.float32), mask=ring_only)
    m["ece_ring"] = e_ring
    m["ring_conf_mean"] = float(p[ring_only].mean()) if ring_only.any() else float("nan")
    m["ring_gt_frac"] = float(gt[ring_only].mean()) if ring_only.any() else float("nan")

    # 3. 分区置信
    for lab, tag in ((0, "bg"), (127, "z127"), (255, "z255")):
        sel = gray == lab
        if sel.any():
            v = p[sel]
            m["prob_%s" % tag] = float(v.mean())
            m["above05_%s" % tag] = float((v >= 0.5).mean()) * 100

    # 4. 边界锐度：真值边界环上概率梯度的模
    gy, gx = np.gradient(p.astype(np.float32))
    gm = np.sqrt(gy ** 2 + gx ** 2)
    m["grad_at_ring"] = float(gm[ring_only].mean()) if ring_only.any() else float("nan")
    m["grad_global"] = float(gm.mean())

    # 5. 不确定带
    m["unc_frac"] = float(((p > 0.4) & (p < 0.6)).mean()) * 100
    if ring_only.any():
        m["unc_frac_ring"] = float(((p[ring_only] > 0.4) & (p[ring_only] < 0.6)).mean()) * 100

    # 6. 错判处的置信
    fn = gt & ~pr
    fp = pr & ~gt
    m["fn_prob_mean"] = float(p[fn].mean()) if fn.any() else float("nan")
    m["fp_prob_mean"] = float(p[fp].mean()) if fp.any() else float("nan")
    m["fn_px"] = int(fn.sum())
    m["fp_px"] = int(fp.sum())
    tp = gt & pr
    m["tp_prob_mean"] = float(p[tp].mean()) if tp.any() else float("nan")
    tn = ~gt & ~pr
    m["tn_prob_mean"] = float(p[tn].mean()) if tn.any() else float("nan")

    # 面积/周长偏差
    m["gt_area"] = int(gt.sum())
    m["pred_area"] = int(pr.sum())
    m["area_ratio"] = float(pr.sum()) / max(int(gt.sum()), 1)
    # 周长口径与 eval_lib 一致 = 边界像素个数（1px 厚表面）
    m["perimeter_pred"] = float(_surface(pr).sum())
    m["perimeter_gt"] = float(_surface(gt).sum())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="tag:概率目录（.npy 或 .npz 自动识别）")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)

    _, val = read_split(USS)
    name2cid, cid2name = load_mapping()

    agg_all = {}
    for spec in args.source:
        tag, d = spec.split(":", 1)
        d = d if os.path.isabs(d) else os.path.abspath(os.path.join(STEP3, d))
        per = {}
        missing = 0
        for n in val:
            cid = name2cid.get(n, os.path.splitext(n)[0])
            pp = find_prob(d, n, cid)
            gt = load_binary(os.path.join(GT_DIR, n))
            gray = np.array(Image.open(os.path.join(GT_DIR, n)).convert("L"))
            if pp is None:
                missing += 1
                continue
            p = load_prob(pp)
            if p.shape != gt.shape:
                from PIL import Image as I
                p = np.asarray(I.fromarray(p).resize((gt.shape[1], gt.shape[0]), I.BILINEAR),
                                dtype=np.float32)
            per[n] = analyze_case(p, gt, gray)
        if not per:
            print("[36] tag %s: 没找到任何概率文件（目录 %s），跳过" % (tag, d))
            continue
        names = list(per)
        agg = {"tag": tag, "dir": d, "n": len(names), "missing": missing}
        keys = set()
        for n in names:
            keys |= set(per[n])
        for k in sorted(keys):
            vals = [per[n][k] for n in names if k in per[n]]
            vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)]
            if vals and not k.startswith("bin_") and k != "name":
                agg[k] = float(np.mean(vals))
        agg["per_image"] = per
        with open(os.path.join(RES_DIR, "%s_probability_diagnostics.json" % tag), "w") as fh:
            json.dump(agg, fh, indent=2)
        agg_all[tag] = agg
        print("=" * 78)
        print("tag %-38s n=%d (missing %d)" % (tag, agg["n"], missing))
        print("  Dice@0.5 %.3f | @0.4 %.3f | @0.6 %.3f | @0.3 %.3f | @0.7 %.3f"
              % (agg.get("dice", float("nan")), agg.get("dice@0.4", float("nan")),
                 agg.get("dice@0.6", float("nan")), agg.get("dice@0.3", float("nan")),
                 agg.get("dice@0.7", float("nan"))))
        print("  ECE 整体 %.4f  只在真值边界环上 %.4f" % (agg.get("ece", float("nan")),
                                                          agg.get("ece_ring", float("nan"))))
        print("  边界环上 |∇p| %.4f (全图平均 %.4f)   不确定带 p∈[0.4,0.6] %.2f%%（环内 %.2f%%）"
              % (agg.get("grad_at_ring", float("nan")), agg.get("grad_global", float("nan")),
                 agg.get("unc_frac", float("nan")), agg.get("unc_frac_ring", float("nan"))))
        print("  错判置信: FN 像素上平均 p=%.3f   FP 像素上平均 p=%.3f   (TP %.3f / TN %.3f)"
              % (agg.get("fn_prob_mean", float("nan")), agg.get("fp_prob_mean", float("nan")),
                 agg.get("tp_prob_mean", float("nan")), agg.get("tn_prob_mean", float("nan"))))
        print("  分区概率: 背景 %.3f (%.1f%%>0.5) | 127 区 %.3f (%.1f%%) | 255 区 %.3f (%.1f%%)"
              % (agg.get("prob_bg", float("nan")), agg.get("above05_bg", float("nan")),
                 agg.get("prob_z127", float("nan")), agg.get("above05_z127", float("nan")),
                 agg.get("prob_z255", float("nan")), agg.get("above05_z255", float("nan"))))
        print("  面积: GT %+.0f px 预测 %+.0f px | 周长 GT %.0f 预测 %.0f | FN %+.0f FP %+.0f"
              % (agg.get("gt_area", float("nan")), agg.get("pred_area", float("nan")),
                 agg.get("perimeter_gt", float("nan")), agg.get("perimeter_pred", float("nan")),
                 agg.get("fn_px", float("nan")), agg.get("fp_px", float("nan"))))

    # ---- 汇总成 md ----
    order = [t for t, _ in LABELS if t in agg_all] + [t for t in agg_all if t not in dict(LABELS)]
    lab = dict(LABELS)
    rows = [("dice", "Dice@0.5", "%.3f"), ("dice@0.4", "Dice@0.4", "%.3f"),
            ("dice@0.6", "Dice@0.6", "%.3f"), ("iou", "IoU@0.5", "%.2f"),
            ("ece", "ECE 整体", "%.4f"), ("ece_ring", "ECE 仅在真值边界环", "%.4f"),
            ("grad_at_ring", "边界环 |∇p|（越大越锐）", "%.4f"),
            ("unc_frac", "不确定带占比 %（p∈[0.4,0.6]）", "%.2f"),
            ("unc_frac_ring", "边界环内不确定带 %", "%.2f"),
            ("fn_prob_mean", "FN 像素平均概率（越低越自信地漏）", "%.3f"),
            ("fp_prob_mean", "FP 像素平均概率（越低越自信地错）", "%.3f"),
            ("prob_bg", "背景区平均概率", "%.3f"),
            ("prob_z127", "127 区平均概率", "%.3f"),
            ("prob_z255", "255 区平均概率", "%.3f"),
            ("above05_bg", "背景区 >0.5 占比 %", "%.3f"),
            ("above05_z127", "127 区 >0.5 占比 %", "%.2f"),
            ("above05_z255", "255 区 >0.5 占比 %", "%.2f"),
            ("perimeter_pred", "预测周长 px（边界像素数）", "%.0f"),
            ("perimeter_gt", "真值周长 px（边界像素数）", "%.0f"),
            ("area_ratio", "预测/真值 面积比", "%.4f"),
            ("pred_area", "预测面积 px", "%.0f"),
            ("gt_area", "真值面积 px", "%.0f")]
    lines = ["# 概率层面三路诊断（同一脚本、同一 fold 0 的 31 张验证图）", "",
             "| 指标 | " + " | ".join(lab.get(t, t) for t in order) + " |",
             "|" + "---|" * (len(order) + 1)]
    for k, name, f in rows:
        cells = []
        for t in order:
            v = agg_all[t].get(k)
            cells.append("—" if v is None or (isinstance(v, float) and v != v) else f % v)
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    out = os.path.join(RES_DIR, "probability_diagnostics_table.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[36] wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
