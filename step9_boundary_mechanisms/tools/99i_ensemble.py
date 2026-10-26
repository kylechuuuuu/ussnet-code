#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99i_ensemble.py

Probability-level ensemble of several fold-0 sandwich models.

Every step-6 arm was predicted with the SAME stock predictor (sliding window +
Gaussian weighting + mirroring TTA + crop-back/resample to the original grid)
and saved its per-case softmax field with --save-probabilities, so the .npz of
different arms are voxel-aligned on the original image grid and can simply be
averaged.  The seven step-6 arms that exist -- w25s / sq2 / w25L in the three
ensemble folders (A/B/C) -- never included the strongest model
(nnUNetTrainerSandwichResencW25), so those ensembles all sit BELOW the single
best arm.  This tool builds the missing combinations.

It writes the PNG prediction set in the exact step-1 convention (single
channel, 0/255, original resolution, original Chinese file name) so that
step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py scores it with
the same implementation and the same val split as every other arm.

usage:
  python 99i_ensemble.py --dirs sq2resencw25_..._base_nifti,sq2w25_..._base_nifti \
                         --tag ens_D_res_w25 [--weights 2,1] [--threshold 0.5]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STEP9 = os.path.dirname(HERE)
USS_ROOT = os.path.dirname(STEP9)
MAPPING = os.path.join(USS_ROOT, "step1_nnunet_baseline", "case_id_mapping.csv")


def load_probs(path):
    """one arm's softmax field for one case -> (2, H, W) float32"""
    with np.load(path) as d:
        p = d["probabilities"]
    p = np.asarray(p, dtype=np.float32)
    while p.ndim > 3 and p.shape[1] == 1:
        p = p[:, 0]
    if p.ndim == 4:                      # (2, 1, H, W) 2d nnU-Net export
        p = np.squeeze(p, axis=1)
    if p.ndim == 2:                      # defensive: hard map only
        p = np.stack([1.0 - p, p], axis=0)
    if p.ndim != 3 or p.shape[0] != 2:
        raise ValueError("unexpected probability shape %r" % (p.shape,))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True,
                    help="comma separated prediction dirs (relative to --pred-root)")
    ap.add_argument("--weights", default="", help="comma separated, default all 1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pred-root",
                    default=os.path.join(USS_ROOT, "step6_sandwich_squeeze", "predictions"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N cases")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    if args.weights:
        w = [float(x) for x in args.weights.split(",")]
        if len(w) != len(dirs):
            raise SystemExit("--weights length != --dirs length")
    else:
        w = [1.0] * len(dirs)
    wsum = float(sum(w))

    with open(MAPPING) as f:
        rows = list(csv.DictReader(f))
    cid2name = {r["case_id"]: r["original_filename"] for r in rows}

    out_dir = os.path.join(args.pred_root, args.tag + "_png")
    os.makedirs(out_dir, exist_ok=True)
    print("[99i] tag=%s dirs=%s weights=%s" % (args.tag, dirs, w))

    done, missing, shapes = 0, [], {}
    for cid, orig in sorted(cid2name.items()):
        if args.limit and done >= args.limit:
            break
        paths = [os.path.join(args.pred_root, d, cid + ".npz") for d in dirs]
        if not all(os.path.exists(p) for p in paths):
            missing.append(cid)
            continue
        acc = None
        for wi, p in zip(w, paths):
            q = load_probs(p)
            acc = q * wi if acc is None else acc + q * wi
        acc /= wsum
        fg = acc[1] >= args.threshold
        shapes[fg.shape] = shapes.get(fg.shape, 0) + 1
        Image.fromarray(np.where(fg, 255, 0).astype(np.uint8), mode="L").save(
            os.path.join(out_dir, orig))
        done += 1

    meta = {"tag": args.tag, "dirs": dirs, "weights": w,
            "pred_root": args.pred_root, "threshold": args.threshold,
            "n_written": done, "n_missing": len(missing),
            "shapes": {str(k): v for k, v in shapes.items()}}
    with open(os.path.join(out_dir, "_ensemble.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[99i] wrote %d PNG -> %s" % (done, out_dir))
    print("[99i] shapes: %s" % (shapes,))
    if missing:
        print("[99i] missing: %s%s" % (missing[:8], " ..." if len(missing) > 8 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
