#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99n_final_artifact.py

Writes the final deliverable: the boundary-oriented ensemble plus the GT-free
post-processing that stacks on top of it.

pipeline (all deterministic, no ground truth)
  1. weighted probability average    res_best x6, w25_best, w25s_best, sq2_best
  2. median filter 3x3 on the fg probability   (kills salt-and-pepper boundary noise)
  3. binary opening 5x5                        (removes protruding spurs)
  4. largest connected component + fill holes
  5. threshold at 0.5 -> 0/255 PNG, original resolution, original Chinese name

usage: python tools/99n_final_artifact.py [--tag ens_K_res6_med3open2fill]
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
PRED = os.path.join(S6, "predictions")
MAPPING = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")

WEIGHTS = {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}
DIRS = {
    "res_best": "sq2resencw25_checkpoint_best_base_nifti",
    "w25_best": "sq2w25_checkpoint_best_base_nifti",
    "w25s_best": "sq2w25s_checkpoint_best_base_nifti",
    "sq2_best": "sq2_checkpoint_best_base_nifti",
}


def largest_cc(m):
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def load_probs(path):
    with np.load(path) as d:
        p = np.asarray(d["probabilities"], dtype=np.float32)
    while p.ndim > 3 and p.shape[1] == 1:
        p = p[:, 0]
    if p.ndim == 2:
        p = np.stack([1.0 - p, p], axis=0)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ens_K_res6_med3open2fill")
    ap.add_argument("--pred-root", default=PRED)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    with open(MAPPING) as f:
        rows = list(csv.DictReader(f))
    out_dir = os.path.join(a.pred_root, a.tag + "_png")
    os.makedirs(out_dir, exist_ok=True)
    wsum = float(sum(WEIGHTS.values()))

    done, shapes = 0, {}
    for r in rows:
        cid, name = r["case_id"], r["original_filename"]
        acc = None
        for lab, w in WEIGHTS.items():
            q = load_probs(os.path.join(PRED, DIRS[lab], cid + ".npz"))
            acc = q * w if acc is None else acc + q * w
        q = acc[1] / wsum

        m = ndi.median_filter(q, size=3) >= 0.5
        m = ndi.binary_opening(m, np.ones((5, 5)))
        m = ndi.binary_fill_holes(largest_cc(m))

        shapes[m.shape] = shapes.get(m.shape, 0) + 1
        fname = name if name.lower().endswith(".png") else name + ".png"
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8), mode="L").save(
            os.path.join(out_dir, fname))
        done += 1
        if a.limit and done >= a.limit:
            break

    with open(os.path.join(out_dir, "_ensemble.json"), "w") as fh:
        json.dump({"tag": a.tag, "members": DIRS, "weights": WEIGHTS,
                   "postprocess": ["median3", "open2", "largest_cc",
                                   "fill_holes"],
                   "threshold": 0.5, "n": done}, fh, indent=2)
    print("[99n] wrote %d PNGs -> %s" % (done, out_dir))
    print("[99n] shapes: %s" % (shapes,))


if __name__ == "__main__":
    main()
