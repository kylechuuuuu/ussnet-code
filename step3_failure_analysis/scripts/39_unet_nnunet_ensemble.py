#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
39_unet_nnunet_ensemble.py

Measures the cheapest possible "what can be done" on this dataset: the
probability-space ensemble of the two EXISTING models (U-Net baseline and stock
nnU-Net 2d).  No training, no new architecture.

Why this is fair: both were trained on the same fold-0 training list (71 cases),
the 31 validation images were never seen by either, and the two probability maps
live in the same geometry (original image size, 0..1).  So averaging them is
variance reduction, not leakage.

Reports, over the 31 fold-0 validation images and with the project's own
`eval_lib.py`:
    U-Net alone | nnU-Net alone | 0.5/0.5 ensemble | per-image best (oracle)
plus the boundary Dice@1px for each, so we can see whether ensembling moves the
metric that actually matters (the boundary), or only the area-based Dice.

usage:
    ~/anaconda3/envs/zk/bin/python scripts/39_unet_nnunet_ensemble.py
"""
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "..", "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import load_binary, load_gray, read_split, all_metrics, _surface  # noqa: E402

STEP3 = os.path.dirname(HERE)
USS = os.path.dirname(STEP3)
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
RES_DIR = os.path.join(STEP3, "results")
PRED_OUT = os.path.join(STEP3, "results", "ensemble_unet_nnunet_png")
MAP_CSV = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")
UNET_PROB = os.path.join(RES_DIR, "unet_73_probabilities")
NN_PROB = os.path.join(USS, "step1_nnunet_baseline", "predictions",
                       "nnunet_2d_fold0_checkpoint_best_nifti")
EPS = 1e-7


def bdice(pred, gt, tol=1):
    gb, pb = _surface(gt), _surface(pred)
    gg = ndimage.binary_dilation(gb, morphology.disk(tol))
    pp = ndimage.binary_dilation(pb, morphology.disk(tol))
    prec = float(np.logical_and(pb, gg).sum()) / (pb.sum() + EPS)
    rec = float(np.logical_and(gb, pp).sum()) / (gb.sum() + EPS)
    return 2 * prec * rec / (prec + rec + EPS) * 100


def main():
    os.makedirs(PRED_OUT, exist_ok=True)
    _, val = read_split(USS)

    name2case = {}
    with open(MAP_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name2case[row["original_filename"]] = row["case_id"]

    rows = []
    for n in val:
        gt = load_binary(os.path.join(GT_DIR, n))
        gray = load_gray(os.path.join(GT_DIR, n))
        pu = np.load(os.path.join(UNET_PROB, n[:-4] + ".npy")).astype(np.float32)
        cid = name2case[n]
        with np.load(os.path.join(NN_PROB, cid + ".npz")) as z:
            pn = np.asarray(z["probabilities"], dtype=np.float32)
        pn = np.squeeze(pn)
        if pn.ndim == 3:
            pn = pn[1]                      # foreground channel
        if pu.shape != pn.shape:
            raise RuntimeError("%s: shape mismatch %s vs %s" % (n, pu.shape, pn.shape))

        pred_u = pu >= 0.5
        pred_n = pn >= 0.5
        pred_e = (pu + pn) / 2.0 >= 0.5

        # sanity: the probability-derived masks must reproduce the stored PNGs
        for tag, p in (("u", pred_u), ("n", pred_n)):
            src = (os.path.join(USS, "unet", "results", "ultrasound_prostate_73", n)
                   if tag == "u" else
                   os.path.join(USS, "step1_nnunet_baseline", "predictions",
                                "nnunet_2d_fold0_checkpoint_best_png", n))
            agree = float(np.logical_and(p, load_binary(src)).sum()) / (p.sum() + EPS)
            if agree < 0.999:
                print("  [warn] %s %s: prob-threshold mask agrees only %.4f with the PNG"
                      % (n, tag, agree))

        mu, mn, me = (all_metrics(pred_u, gt, gray), all_metrics(pred_n, gt, gray),
                      all_metrics(pred_e, gt, gray))

        # save the ensemble mask so the error-decomposition script can consume it
        Image.fromarray((pred_e * 255).astype(np.uint8)).save(os.path.join(PRED_OUT, n))

        rows.append(dict(name=n,
                         dice_u=mu["dice"], dice_n=mn["dice"], dice_e=me["dice"],
                         bd1_u=bdice(pred_u, gt), bd1_n=bdice(pred_n, gt), bd1_e=bdice(pred_e, gt),
                         hd95_u=mu["hd95"], hd95_n=mn["hd95"], hd95_e=me["hd95"],
                         perim_u=mu["perimeter_pred"], perim_n=mn["perimeter_pred"],
                         perim_e=me["perimeter_pred"],
                         cc_u=mu["n_pred_cc"], cc_n=mn["n_pred_cc"], cc_e=me["n_pred_cc"]))

    with open(os.path.join(RES_DIR, "ensemble_unet_nnunet_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    oracle = float(np.mean([max(r["dice_u"], r["dice_n"]) for r in rows]))
    s = dict(n=len(rows),
             dice_unet=m("dice_u"), dice_nnunet=m("dice_n"), dice_ensemble=m("dice_e"),
             dice_oracle_per_image_best=oracle,
             bd1_unet=m("bd1_u"), bd1_nnunet=m("bd1_n"), bd1_ensemble=m("bd1_e"),
             hd95_unet=m("hd95_u"), hd95_nnunet=m("hd95_n"), hd95_ensemble=m("hd95_e"),
             perim_unet=m("perim_u"), perim_nnunet=m("perim_n"), perim_ensemble=m("perim_e"),
             cc_unet=m("cc_u"), cc_nnunet=m("cc_n"), cc_ensemble=m("cc_e"))
    with open(os.path.join(RES_DIR, "ensemble_unet_nnunet_summary.json"), "w") as f:
        json.dump(s, f, indent=2)

    print("=" * 78)
    print("U-Net vs nnU-Net vs probability ensemble, fold 0, n=%d" % s["n"])
    print("  Dice           U-Net %.3f   nnU-Net %.3f   ENSEMBLE %.3f   (oracle per-image best %.3f)"
          % (s["dice_unet"], s["dice_nnunet"], s["dice_ensemble"], oracle))
    print("  boundary@1px   U-Net %.2f%%  nnU-Net %.2f%%  ENSEMBLE %.2f%%"
          % (s["bd1_unet"], s["bd1_nnunet"], s["bd1_ensemble"]))
    print("  HD95 (px)      U-Net %.2f   nnU-Net %.2f   ENSEMBLE %.2f"
          % (s["hd95_unet"], s["hd95_nnunet"], s["hd95_ensemble"]))
    print("  perimeter (px) U-Net %.1f  nnU-Net %.1f  ENSEMBLE %.1f"
          % (s["perim_unet"], s["perim_nnunet"], s["perim_ensemble"]))
    print("  components     U-Net %.2f   nnU-Net %.2f   ENSEMBLE %.2f"
          % (s["cc_unet"], s["cc_nnunet"], s["cc_ensemble"]))
    print("wrote", os.path.join(RES_DIR, "ensemble_unet_nnunet_summary.json"))
    print("wrote PNGs ->", PRED_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
