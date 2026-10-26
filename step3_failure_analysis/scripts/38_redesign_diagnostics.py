#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
38_redesign_diagnostics.py

Measurements that decide HOW to redesign the model, given that both U-Net and
nnU-Net are already at Dice 93.0 / 96.1 but boundary Dice@1px is still only
23% / 35%.

They answer three questions the mean Dice cannot:

  Q1  Is the ground-truth boundary actually visible in the image?
      -> gradient support at the GT boundary vs the whole image vs the model's
         own predicted boundary.  If the prediction is MORE edge-aligned than
         the GT, the remaining error is an annotation/definition mismatch, not a
         capacity problem  =>  regularisers/priors, not more parameters.
      Also corr(per-image edge support, per-image boundary Dice): is the hard
      case the low-contrast case?

  Q2  Is the residual boundary error concentrated in a particular direction?
      -> FN / FP rate per angular sector around the GT centroid, normalised by
         the amount of GT boundary in that sector.  Tells us whether a
         directional / apex-aware mechanism would pay off.

  Q3  Do the models get the COARSE shape right and only miss fine detail?
      -> Dice(gt, opening(gt, disk(r))) = how much of the GT is fine structure;
         Dice(pred, opening(gt, disk(r))) and Dice(opening(pred, disk(r)), gt)
         = coarse-shape agreement.  If coarse shape is right and only fine
         detail is wrong, a boundary-refinement stage is the right lever.

usage:
    ~/anaconda3/envs/zk/bin/python scripts/38_redesign_diagnostics.py
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
from eval_lib import load_binary, read_split, _surface  # noqa: E402

STEP3 = os.path.dirname(HERE)
USS = os.path.dirname(STEP3)
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
RES_DIR = os.path.join(STEP3, "results")
EPS = 1e-7

MODELS = [("unet_73", os.path.join(USS, "unet", "results", "ultrasound_prostate_73")),
          ("nnunet_2d_fold0_checkpoint_best",
           os.path.join(USS, "step1_nnunet_baseline", "predictions",
                        "nnunet_2d_fold0_checkpoint_best_png"))]

N_SECT = 12          # 30-degree sectors
RADII = [3, 5, 10, 15]


def grad_mag(img):
    """Sobel gradient magnitude after a light smoothing (suppresses speckle)."""
    f = ndimage.gaussian_filter(img.astype(np.float32), 1.0)
    gx = ndimage.sobel(f, axis=1)
    gy = ndimage.sobel(f, axis=0)
    return np.hypot(gx, gy)


def dice(a, b):
    return 2.0 * float(np.logical_and(a, b).sum()) / (float(a.sum()) + float(b.sum()) + EPS) * 100


def edge_support(g, mask_boundary, thr):
    """Fraction of boundary pixels whose local gradient exceeds thr."""
    if not mask_boundary.any():
        return float("nan")
    return float((g[mask_boundary] >= thr).mean())


def analyse(gray, gt, pred):
    out = {}
    g = grad_mag(gray)
    gb = _surface(gt)
    pb = _surface(pred)

    p75, p90 = np.percentile(g, 75), np.percentile(g, 90)
    out["grad_mean_image"] = float(g.mean())
    out["grad_mean_gt_boundary"] = float(g[gb].mean())
    out["grad_mean_pred_boundary"] = float(g[pb].mean())
    out["edge_support_gt_p75"] = edge_support(g, gb, p75)
    out["edge_support_pred_p75"] = edge_support(g, pb, p75)
    out["edge_support_gt_p90"] = edge_support(g, gb, p90)
    out["edge_support_pred_p90"] = edge_support(g, pb, p90)

    # ---- where is the residual boundary-band error? ------------------------
    d_gtb = ndimage.distance_transform_edt(~gb)
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(pred, ~gt)
    band = d_gtb <= 2
    fn_b, fp_b = np.logical_and(fn, band), np.logical_and(fp, band)

    ys, xs = np.where(gt)
    cy, cx = float(ys.mean()), float(xs.mean())
    edges = np.linspace(-180, 180, N_SECT + 1)
    # available AREA per sector = the pixels of the 2 px boundary band, so the
    # rate FN_area/per_sector / band_area/per_sector is a real 0..1 rate
    bys, bxs = np.where(band)
    bang = np.rad2deg(np.arctan2(-(bys - cy), bxs - cx))
    gt_hist, _ = np.histogram(bang, bins=edges)
    def sector_hist(mask):
        if not mask.any():
            return np.zeros(N_SECT, int)
        yy, xx = np.where(mask)
        a = np.rad2deg(np.arctan2(-(yy - cy), xx - cx))        # 90 deg = up (anterior)
        return np.histogram(a, bins=edges)[0]

    fn_hist = sector_hist(fn_b)
    fp_hist = sector_hist(fp_b)

    out["sector_centers"] = [float((edges[i] + edges[i + 1]) / 2) for i in range(N_SECT)]
    out["gt_band_px_per_sector"] = [int(v) for v in gt_hist]
    out["fn_rate_per_sector_pct"] = [float(fn_hist[i]) / (gt_hist[i] + EPS) * 100 for i in range(N_SECT)]
    out["fp_rate_per_sector_pct"] = [float(fp_hist[i]) / (gt_hist[i] + EPS) * 100 for i in range(N_SECT)]

    # ---- coarse shape vs fine detail ---------------------------------------
    for r in RADII:
        op_gt = ndimage.binary_opening(gt, morphology.disk(r))
        op_pr = ndimage.binary_opening(pred, morphology.disk(r))
        out["dice_gt_vs_open_gt_r%d" % r] = dice(gt, op_gt)
        out["dice_pred_vs_open_gt_r%d" % r] = dice(pred, op_gt)
        out["dice_open_pred_vs_gt_r%d" % r] = dice(op_pr, gt)

    # boundary Dice@1px (same definition as 30_error_decomposition.py)
    gg, pp = ndimage.binary_dilation(gb, morphology.disk(1)), ndimage.binary_dilation(pb, morphology.disk(1))
    prec = float(np.logical_and(pb, gg).sum()) / (pb.sum() + EPS)
    rec = float(np.logical_and(gb, pp).sum()) / (gb.sum() + EPS)
    out["bdice_1px"] = 2 * prec * rec / (prec + rec + EPS) * 100

    out["dice"] = dice(pred, gt)
    return out


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    _, val = read_split(USS)
    summary = {}

    for tag, pdir in MODELS:
        rows = []
        for n in val:
            pp = os.path.join(pdir, n)
            if not os.path.exists(pp):
                continue
            gray = np.array(Image.open(os.path.join(IMG_DIR, n)).convert("L"))
            gt = load_binary(os.path.join(GT_DIR, n))
            pred = load_binary(pp)
            r = analyse(gray, gt, pred)
            r["name"] = n
            rows.append(r)

        with open(os.path.join(RES_DIR, "%s_redesign_diag.csv" % tag), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        def m(k):
            v = [r[k] for r in rows if r[k] == r[k]]
            return float(np.mean(v)) if v else float("nan")

        sec_fn = np.array([r["fn_rate_per_sector_pct"] for r in rows], dtype=float).mean(0)
        sec_fp = np.array([r["fp_rate_per_sector_pct"] for r in rows], dtype=float).mean(0)
        gtsec = np.array([r["gt_band_px_per_sector"] for r in rows], dtype=float).sum(0)

        es = np.array([r["edge_support_gt_p75"] for r in rows])
        bd = np.array([r["bdice_1px"] for r in rows])
        es_p = np.array([r["edge_support_pred_p75"] for r in rows])

        s = {
            "tag": tag, "n": len(rows),
            "dice_mean": m("dice"), "bdice_1px_mean": m("bdice_1px"),
            "grad_mean_image": m("grad_mean_image"),
            "grad_mean_gt_boundary": m("grad_mean_gt_boundary"),
            "grad_mean_pred_boundary": m("grad_mean_pred_boundary"),
            "edge_support_gt_p75": m("edge_support_gt_p75"),
            "edge_support_pred_p75": m("edge_support_pred_p75"),
            "edge_support_gt_p90": m("edge_support_gt_p90"),
            "edge_support_pred_p90": m("edge_support_pred_p90"),
            "corr_edgesupport_gt_vs_bdice": float(np.corrcoef(es, bd)[0, 1]) if len(es) > 2 else float("nan"),
            "sector_centers": rows[0]["sector_centers"],
            "gt_band_px_per_sector": [int(v) for v in gtsec],
            "fn_rate_per_sector_pct": [float(v) for v in sec_fn],
            "fp_rate_per_sector_pct": [float(v) for v in sec_fp],
        }
        for r in RADII:
            s["dice_gt_vs_open_gt_r%d" % r] = m("dice_gt_vs_open_gt_r%d" % r)
            s["dice_pred_vs_open_gt_r%d" % r] = m("dice_pred_vs_open_gt_r%d" % r)
            s["dice_open_pred_vs_gt_r%d" % r] = m("dice_open_pred_vs_gt_r%d" % r)
        summary[tag] = s

        print("=" * 78)
        print("tag %s  n=%d   Dice %.3f   boundary Dice@1px %.2f%%"
              % (tag, len(rows), s["dice_mean"], s["bdice_1px_mean"]))
        print("  |grad| image %.2f | at GT boundary %.2f | at PRED boundary %.2f   (pred/GT = %.2fx)"
              % (s["grad_mean_image"], s["grad_mean_gt_boundary"], s["grad_mean_pred_boundary"],
                 s["grad_mean_pred_boundary"] / (s["grad_mean_gt_boundary"] + EPS)))
        print("  edge support (grad >= image p75):  GT boundary %.1f%%   pred boundary %.1f%%"
              % (s["edge_support_gt_p75"] * 100, s["edge_support_pred_p75"] * 100))
        print("  edge support (grad >= image p90):  GT boundary %.1f%%   pred boundary %.1f%%"
              % (s["edge_support_gt_p90"] * 100, s["edge_support_pred_p90"] * 100))
        print("  corr(per-image GT edge support, per-image boundary Dice) = %+.3f"
              % s["corr_edgesupport_gt_vs_bdice"])
        print("  coarse vs fine: Dice(gt, open(gt,r))      r=3 %.1f  5 %.1f  10 %.1f  15 %.1f"
              % tuple(s["dice_gt_vs_open_gt_r%d" % r] for r in RADII))
        print("                  Dice(pred, open(gt,r))    r=3 %.1f  5 %.1f  10 %.1f  15 %.1f"
              % tuple(s["dice_pred_vs_open_gt_r%d" % r] for r in RADII))
        print("                  Dice(open(pred,r), gt)    r=3 %.1f  5 %.1f  10 %.1f  15 %.1f"
              % tuple(s["dice_open_pred_vs_gt_r%d" % r] for r in RADII))
        print("  FN/FP rate per 30deg sector, as %% of the boundary-band AREA in that "
              "sector (90=up/anterior, -90=down/posterior):")
        for i in range(N_SECT):
            print("     %+6.0f deg  GT-band px %6d   FN %5.2f%%   FP %5.2f%%"
                  % (s["sector_centers"][i], s["gt_band_px_per_sector"][i],
                     s["fn_rate_per_sector_pct"][i], s["fp_rate_per_sector_pct"][i]))

    with open(os.path.join(RES_DIR, "redesign_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(RES_DIR, "redesign_diagnostics.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
