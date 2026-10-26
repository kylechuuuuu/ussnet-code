#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py

Evaluate one prediction folder (PNG masks) against the ground-truth masks with
the shared metric implementation in eval_lib.py.  Works identically for the
U-Net baseline predictions and for the nnU-Net predictions (exported to PNG by
step1/scripts/05_export_nifti_to_png.py), which is what makes the comparison
fair.

    python scripts/10_evaluate_predictions.py \
        --pred ../../unet/results/ultrasound_prostate_73 --tag unet_73 --split val

Writes  metrics/<tag>_per_image.csv
        metrics/<tag>_summary.json
        metrics/<tag>_summary.txt      (human readable)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_lib import all_metrics, load_binary, load_gray, read_split  # noqa: E402

STEP2 = os.path.dirname(HERE)
USS = os.path.dirname(STEP2)
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")

# columns averaged in the summary table
SUMMARY_KEYS = ["dice", "precision", "recall", "iou", "vol_diff_pct", "hd95", "assd",
                "hd100", "correctness", "completeness", "quality",
                "n_pred_cc", "n_gt_cc", "skel_pred_len", "skel_gt_len",
                "perimeter_pred", "perimeter_gt", "d127_dice", "d255_dice",
                "d127_recall", "d255_recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction folder with PNG masks")
    ap.add_argument("--tag", required=True, help="name of this prediction set")
    ap.add_argument("--out", default=os.path.join(STEP2, "metrics"))
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--threshold", type=int, default=0)
    args = ap.parse_args()

    pred_dir = args.pred if os.path.isabs(args.pred) else os.path.join(STEP2, args.pred)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(STEP2, args.out)
    os.makedirs(out_dir, exist_ok=True)

    train_names, val_names = read_split(USS)
    if args.split == "val":
        names = val_names
    elif args.split == "train":
        names = train_names
    else:
        names = sorted(set(train_names) | set(val_names))

    rows, missing = [], []
    for n in names:
        pp = os.path.join(pred_dir, n)
        gp = os.path.join(GT_DIR, n)
        if not os.path.exists(pp):
            missing.append(n)
            continue
        pred = load_binary(pp, args.threshold)
        gt = load_binary(gp, args.threshold)
        if pred.shape != gt.shape:
            print("!! size mismatch for", n, pred.shape, gt.shape)
            return 3
        gray = load_gray(gp)
        m = all_metrics(pred, gt, gt_gray=gray)
        m["name"] = n
        m["split"] = "val" if n in val_names else "train"
        rows.append(m)

    if not rows:
        print("no predictions found in", pred_dir)
        return 4
    if missing:
        print("WARNING: %d predictions missing (of %d): %s"
              % (len(missing), len(names), missing[:5]))

    fieldnames = ["name", "split"] + [k for k in rows[0].keys() if k not in ("name", "split")]
    csv_path = os.path.join(out_dir, "%s_per_image.csv" % args.tag)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {"tag": args.tag, "pred_dir": pred_dir, "split": args.split,
               "n_images": len(rows), "n_missing": len(missing)}
    for k in SUMMARY_KEYS:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        summary[k + "_mean"] = float(np.nanmean(vals))
        summary[k + "_std"] = float(np.nanstd(vals))
        summary[k + "_median"] = float(np.nanmedian(vals))

    with open(os.path.join(out_dir, "%s_summary.json" % args.tag), "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["%s  (split=%s, n=%d, missing=%d)" % (args.tag, args.split, len(rows), len(missing)),
             "-" * 60]
    for k in SUMMARY_KEYS:
        lines.append("%-16s mean %10.3f   std %8.3f   median %10.3f"
                     % (k, summary[k + "_mean"], summary[k + "_std"], summary[k + "_median"]))
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "%s_summary.txt" % args.tag), "w") as f:
        f.write(txt + "\n")
    print(txt)
    print("\nwrote", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
