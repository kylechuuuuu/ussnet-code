#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_probe_armCshift.py -- armC (ImageNet-pretrained ResNet18 encoder, step4) on
the two decisive domain-shift conditions, with the SAME metric code as
tools/82_clinical_metrics.py (eval_lib.all_metrics + s57.bdice), so the numbers
are directly comparable with results/shift_metrics.csv.

usage: python tools/_probe_armCshift.py [tag] [--out results/_armCshift_probe.csv]
"""
import argparse
import csv
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))

import eval_lib                                                       # noqa: E402
s57 = importlib.import_module("57_shape_prior_ceiling")               # noqa: E402
from scipy import ndimage                                             # noqa: E402

MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")


def read_nii(path):
    import SimpleITK as sitk
    return np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="armCshift")
    ap.add_argument("--pred-root", default=os.path.join(ROOT, "predictions"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "_armCshift_probe.csv"))
    args = ap.parse_args()

    train_names, val_names = eval_lib.read_split(USS)
    names = train_names if args.split == "train" else val_names

    runs = []
    for f in sorted(os.listdir(args.pred_root)):
        if f.endswith("_shiftrun.json"):
            with open(os.path.join(args.pred_root, f)) as fh:
                runs.append(json.load(fh))
    runs = [r for r in runs if r["tag"] == args.tag]
    if not runs:
        raise SystemExit("no run with tag %s" % args.tag)

    rows = []
    for run in runs:
        for key, d in run["outputs"].items():
            if not os.path.isdir(d):
                continue
            cond = key.rsplit("_", 1)[0] if key.rsplit("_", 1)[-1] in (
                "base", "cautious", "amplified") else key
            for n in names:
                stem = os.path.splitext(n)[0]
                fp = os.path.join(d, stem + ".nii.gz")
                if not os.path.exists(fp):
                    continue
                pred = read_nii(fp) > 0.5
                g = eval_lib.load_binary(os.path.join(MASK_DIR, n))
                if pred.shape != g.shape:
                    raise SystemExit("shape mismatch %s %s %s" % (fp, pred.shape, g.shape))
                m = eval_lib.all_metrics(pred, g, eval_lib.load_gray(os.path.join(MASK_DIR, n)))
                # centroid displacement vs GT (the collapse signature of step9 SS2/3)
                pc = np.array(ndimage.center_of_mass(pred)) if pred.any() else np.array([np.nan] * pred.ndim)
                gc = np.array(ndimage.center_of_mass(g)) if g.any() else np.array([np.nan] * g.ndim)
                rows.append({
                    "model": args.tag, "condition": cond, "name": n, "split": args.split,
                    "dice": float(m["dice"]), "bdice1": float(s57.bdice(pred, g, 1)),
                    "hd95": float(m["hd95"]), "assd": float(m["assd"]),
                    "vol_diff_pct": float(m["vol_diff_pct"]),
                    "centroid_shift_px": float(np.linalg.norm(pc - gc)),
                })

    if not rows:
        raise SystemExit("no predictions matched")
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("[probe] %s split=%s n=%d" % (args.tag, args.split, len(names)))
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        dice = [r["dice"] for r in sub]
        print("  %-14s dice %6.2f  min %6.2f  bdice1 %6.2f  HD95 %6.2f  "
              "| <80 %d/%d  <50 %d/%d  centroid_shift %6.1f px"
              % (cond, np.mean(dice), np.min(dice), np.mean([r["bdice1"] for r in sub]),
                 np.mean([r["hd95"] for r in sub]),
                 sum(d < 80 for d in dice), len(dice), sum(d < 50 for d in dice), len(dice),
                 np.mean([r["centroid_shift_px"] for r in sub])))
    print("  wrote %s" % args.out)


if __name__ == "__main__":
    main()
