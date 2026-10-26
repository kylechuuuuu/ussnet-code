#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
40_probability_smoothing_probe.py

The decisive cheap experiment for "is the remaining boundary error a high-
frequency jitter or a low-frequency (shape) bias?".

Both baselines already saved their probability maps, so WITHOUT any retraining we
threshold a Gaussian-smoothed version and watch the boundary metric:

    prob -> gaussian_filter(prob, sigma) -> >= 0.5

* if a small sigma fixes most of the boundary-Dice gap  => the residual is
  high-frequency jitter  => fix it with output regularisation / smoother
  decoding (TV, curvature, a smoothing term), not with more capacity;
* if smoothing does not help (or hurts)  => the residual is a coherent smooth
  displacement of the whole contour  => fix it with a shape/contour prior or a
  boundary loss able to move the curve coherently.

Also runs the same sweep for the U-Net + nnU-Net probability ensemble, i.e. the
best free baseline available today.

Only Dice / boundary Dice@1px / perimeter are computed (the full metric suite is
far too slow for a 3x6 grid).

usage:
    ~/anaconda3/envs/zk/bin/python scripts/40_probability_smoothing_probe.py
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import ndimage
from skimage import morphology

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "..", "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import load_binary, read_split, _surface  # noqa: E402

STEP3 = os.path.dirname(HERE)
USS = os.path.dirname(STEP3)
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
RES_DIR = os.path.join(STEP3, "results")
UNET_PROB = os.path.join(RES_DIR, "unet_73_probabilities")
NN_PROB = os.path.join(USS, "step1_nnunet_baseline", "predictions",
                       "nnunet_2d_fold0_checkpoint_best_nifti")
MAP_CSV = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")
EPS = 1e-7
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def dice(pred, gt):
    return 2.0 * float(np.logical_and(pred, gt).sum()) / (float(pred.sum()) + float(gt.sum()) + EPS) * 100


def bdice(pred, gt, tol=1):
    gb, pb = _surface(gt), _surface(pred)
    gg = ndimage.binary_dilation(gb, morphology.disk(tol))
    pp = ndimage.binary_dilation(pb, morphology.disk(tol))
    prec = float(np.logical_and(pb, gg).sum()) / (pb.sum() + EPS)
    rec = float(np.logical_and(gb, pp).sum()) / (gb.sum() + EPS)
    return 2 * prec * rec / (prec + rec + EPS) * 100


def main():
    _, val = read_split(USS)
    name2case = {}
    with open(MAP_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name2case[row["original_filename"]] = row["case_id"]

    rows = []
    for n in val:
        gt = load_binary(os.path.join(GT_DIR, n))
        pu = np.load(os.path.join(UNET_PROB, n[:-4] + ".npy")).astype(np.float32)
        with np.load(os.path.join(NN_PROB, name2case[n] + ".npz")) as z:
            pn = np.squeeze(np.asarray(z["probabilities"], dtype=np.float32))
        if pn.ndim == 3:
            pn = pn[1]
        pe = (pu + pn) / 2.0
        r = {"name": n}
        for tag, p in (("u", pu), ("n", pn), ("e", pe)):
            for sg in SIGMAS:
                q = p if sg == 0 else ndimage.gaussian_filter(p, sg)
                pred = q >= 0.5
                r["bd1_%s_s%s" % (tag, sg)] = bdice(pred, gt)
                r["dice_%s_s%s" % (tag, sg)] = dice(pred, gt)
                r["perim_%s_s%s" % (tag, sg)] = int(_surface(pred).sum())
        rows.append(r)
        print("  done", n, flush=True)

    with open(os.path.join(RES_DIR, "smoothing_probe_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    names = {"u": "U-Net", "n": "nnU-Net", "e": "ensemble"}
    summ = {}
    for label, key, fmt in (("boundary Dice@1px", "bd1_%s_s%s", "%12.2f"),
                            ("Dice", "dice_%s_s%s", "%12.3f"),
                            ("perimeter (px)", "perim_%s_s%s", "%12.0f")):
        print()
        print(label + " after smoothing the probability map, then >= 0.5")
        print("  sigma    " + "".join("%12s" % names[t] for t in ("u", "n", "e")))
        for sg in SIGMAS:
            line = "  %-8s" % sg
            for t in ("u", "n", "e"):
                v = float(np.mean([r[key % (t, sg)] for r in rows]))
                line += fmt % v
                summ[key % (t, sg)] = v
            print(line)
    print()
    print("  (ground-truth perimeter = 1078.8 px; nnU-Net unsmoothed perimeter = 1094 px)")

    summ["sigmas"] = SIGMAS
    with open(os.path.join(RES_DIR, "smoothing_probe_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("wrote", os.path.join(RES_DIR, "smoothing_probe_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
