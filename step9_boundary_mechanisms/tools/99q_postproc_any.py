#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99q_postproc_any.py

Same GT-free post-processing as 99n, but for an ARBITRARY member set, so we can
ask the fairness question: does median3+open2+fill_holes help an arbitrary
model (and in particular the plain nnU-Net baseline), or is it specific to the
ensemble field?

  pipeline: weighted prob average (weights normalised) -> median3 -> open2(5x5)
            -> largest_cc -> fill_holes -> threshold 0.5 -> 0/255 PNG

usage:
  python tools/99q_postproc_any.py --tag nnunet_best_med3open2fill \
      --dirs ../step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_best_nifti \
      --weights 1
  python tools/99q_postproc_any.py --tag res_best_med3open2fill \
      --dirs ../step6_sandwich_squeeze/predictions/sq2resencw25_checkpoint_best_base_nifti \
      --weights 1
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dirs", required=True, help="comma separated")
    ap.add_argument("--weights", default=None, help="comma separated, default 1s")
    ap.add_argument("--pred-root", default=PRED)
    a = ap.parse_args()

    dirs = [d.strip() for d in a.dirs.split(",")]
    if a.weights:
        ws = [float(x) for x in a.weights.split(",")]
    else:
        ws = [1.0] * len(dirs)
    assert len(dirs) == len(ws), "dirs/weights length mismatch"
    wsum = float(sum(ws))

    with open(MAPPING) as f:
        rows = list(csv.DictReader(f))
    out_dir = os.path.join(a.pred_root, a.tag + "_png")
    os.makedirs(out_dir, exist_ok=True)

    done, shapes = 0, {}
    for r in rows:
        cid, name = r["case_id"], r["original_filename"]
        acc = None
        for d, w in zip(dirs, ws):
            p = os.path.join(d, cid + ".npz")
            if not os.path.exists(p):
                raise SystemExit("missing %s" % p)
            q = load_probs(p)
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

    with open(os.path.join(out_dir, "_ensemble.json"), "w") as fh:
        json.dump({"tag": a.tag, "dirs": dirs, "weights": ws,
                   "postprocess": ["median3", "open2", "largest_cc",
                                   "fill_holes"],
                   "threshold": 0.5, "n": done}, fh, indent=2)
    print("[99q] wrote %d PNGs -> %s" % (done, out_dir))
    print("[99q] shapes: %s" % (shapes,))


if __name__ == "__main__":
    main()
