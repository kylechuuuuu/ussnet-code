#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/tools/65_why_analysis.py

WHY DID THE SQUEEZE MODEL NOT BEAT nnU-Net?  Three quantitative answers, all on
the fold-0 validation split, all on CPU:

  Q1 HOW MUCH INFORMATION DO THE BOUND TARGETS CARRY?
     The squeeze supervises two extra heads with erode(GT, r) and dilate(GT, r).
     If those targets are ~identical to GT, the extra supervision is redundant
     and cannot inject boundary information the stock loss does not already
     have.  Measured: Dice(erode(GT,r), GT) and Dice(dilate(GT,r), GT) for the
     curriculum's r - i.e. exactly how much "new" target there is at each stage.

  Q2 IS THE RESIDUAL THE SAME KIND OF ERROR?
     step 3/4 established that both baselines' residual is a coherent, signed
     contour displacement of 1-3 px (an annotation-convention bias), not random
     jitter.  Measured here for every model: the mean SIGNED displacement of the
     predicted boundary (positive = outside the GT boundary), the share of the
     error that is "thin" (<=2 px) vs "gross" (>10 px), so it can be checked
     whether the squeeze changed the error's *character* or only its size.

  Q3 IS THE GAP SIGNIFICANT AND WHERE DOES IT LIVE?
     Paired per-image comparison of boundary Dice@1px and Dice against the
     nnU-Net baseline (mean delta, Wilcoxon signed-rank p, #images better/worse).

usage:  ~/anaconda3/envs/zk/bin/python tools/65_why_analysis.py
"""
import os
import sys

import numpy as np
from scipy import ndimage, stats
from skimage import morphology

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.dirname(HERE)
USS = os.path.dirname(STEP5)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import load_binary, read_split                     # noqa: E402

EPS = 1e-7
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")

MODELS = {
    "nnU-Net 1000ep": os.path.join(USS, "step1_nnunet_baseline", "predictions",
                                   "nnunet_2d_fold0_checkpoint_best_png"),
    "SQ-A median": os.path.join(STEP5, "predictions", "sqA_median_checkpoint_best_png"),
    "SQ-A main": os.path.join(STEP5, "predictions", "sqA_main_checkpoint_best_png"),
    "SQ-A inner": os.path.join(STEP5, "predictions", "sqA_inner_checkpoint_best_png"),
    "SQ-A outer": os.path.join(STEP5, "predictions", "sqA_outer_checkpoint_best_png"),
    "SQ-I median": os.path.join(STEP5, "predictions", "sqI_median_checkpoint_best_png"),
    "step4 armB 500ep": os.path.join(USS, "step4_our_new_model_resunet_boundary", "predictions",
                                     "fpB_resenc_fploss_dir_checkpoint_best_png"),
}


def surface(m):
    if not m.any():
        return m.copy()
    return m & ~ndimage.binary_erosion(m, ndimage.generate_binary_structure(2, 1), border_value=0)


def bdice_1px(pred, gt, tol=1):
    gb, pb = surface(gt), surface(pred)
    if pb.sum() == 0 or gb.sum() == 0:
        return np.nan
    gg = ndimage.binary_dilation(gb, morphology.disk(tol))
    pp = ndimage.binary_dilation(pb, morphology.disk(tol))
    prec = np.logical_and(pb, gg).sum() / (pb.sum() + EPS)
    rec = np.logical_and(gb, pp).sum() / (gb.sum() + EPS)
    return 2 * prec * rec / (prec + rec + EPS) * 100.0


def dice(pred, gt):
    return 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + EPS) * 100.0


def per_image(pred, gt):
    """boundary Dice, Dice, signed displacement of the predicted boundary, and
    the thin/gross split of the error (step-3 language)."""
    out = {"bdice": bdice_1px(pred, gt), "dice": dice(pred, gt)}
    gb, pb = surface(gt), surface(pred)
    d_to_gt = ndimage.distance_transform_edt(~gb)          # distance of every px to GT boundary
    outside = ~gt                                          # TRUE sign: is the pixel outside GT?
    signed = d_to_gt[pb] * np.where(outside[pb], 1.0, -1.0)   # >0 = boundary sits OUTSIDE GT
    if signed.size:
        out["disp_mean"] = float(np.mean(signed))
        out["disp_abs"] = float(np.mean(np.abs(signed)))
        out["disp_frac_out"] = float(np.mean(signed > 0))
    else:
        out["disp_mean"] = out["disp_abs"] = out["disp_frac_out"] = np.nan
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(pred, ~gt)
    fn_d = d_to_gt[fn] if fn.any() else np.array([0.0])
    fp_d = d_to_gt[fp] if fp.any() else np.array([0.0])
    out["fn_thin2"] = 100.0 * np.mean(fn_d <= 2) if fn.any() else np.nan
    out["fp_thin2"] = 100.0 * np.mean(fp_d <= 2) if fp.any() else np.nan
    out["fn_gross10"] = 100.0 * np.mean(fn_d > 10) if fn.any() else np.nan
    out["fp_gross10"] = 100.0 * np.mean(fp_d > 10) if fp.any() else np.nan
    out["area"] = float(gt.sum())
    return out


def main():
    _, val = read_split(USS)
    gts = {}
    for name in val:
        gts[name] = load_binary(os.path.join(GT_DIR, name), threshold=0)

    # ---------------------------------------------------------------- Q1
    print("=" * 78)
    print("Q1  HOW MUCH DOES THE EXTRA SUPERVISION DIFFER FROM PLAIN GT?")
    print("    (Dice of the bound targets against the true mask, val split)")
    print("      r      Dice(erode(GT,r), GT)   Dice(dilate(GT,r), GT)   'new' pixels")
    for r in [1, 2, 3, 4, 6]:
        de, di, newpx = [], [], []
        for name in val:
            g = gts[name]
            er = ndimage.binary_erosion(g, morphology.disk(r))
            dl = ndimage.binary_dilation(g, morphology.disk(r))
            de.append(dice(er, g))
            di.append(dice(dl, g))
            newpx.append((dl.sum() - er.sum()) / g.sum() * 100)
        print("    %4.1f px      %8.3f %%              %8.3f %%          %6.2f %% of GT area"
              % (r, np.mean(de), np.mean(di), np.mean(newpx)))
    print("    -> at r = 1 px (where the curriculum ends) the two bound targets agree with GT")
    print("       to >99 % Dice: the 'extra' supervision carries almost no new geometry.")

    # ------------------------------------------------------------ Q2 + Q3
    res = {}
    print()
    print("=" * 78)
    print("Q2/Q3  ERROR CHARACTER AND THE PAIRED GAP VS nnU-Net (1000 ep)")
    hdr = ("%-18s %8s %8s %9s %9s %8s %8s %8s" %
           ("model", "bdice", "dice", "disp_mean", "|disp|", "%out", "FN<=2px", "FP<=2px"))
    print(hdr)
    for label, pdir in MODELS.items():
        if not os.path.isdir(pdir):
            print("%-18s (missing: %s)" % (label, pdir))
            continue
        rows = {}
        for name in val:
            p = os.path.join(pdir, name)
            if not os.path.isfile(p):
                continue
            rows[name] = per_image(load_binary(p, threshold=0.5), gts[name])
        res[label] = rows
        if not rows:
            continue
        m = lambda k: np.nanmean([r[k] for r in rows.values()])
        print("%-18s %8.2f %8.2f %9.3f %9.3f %7.1f%% %7.1f%% %7.1f%%" %
              (label, m("bdice"), m("dice"), m("disp_mean"), m("disp_abs"),
               100 * m("disp_frac_out"), m("fn_thin2"), m("fp_thin2")))

    base = res.get("nnU-Net 1000ep")
    if base:
        print()
        print("paired vs nnU-Net 1000ep (same 31 images, Wilcoxon signed-rank):")
        print("  %-18s %10s %10s %8s %8s %10s" %
              ("model", "d_bdice", "d_dice", "better", "worse", "p(bdice)"))
        for label, rows in res.items():
            if label == "nnU-Net 1000ep" or not rows:
                continue
            common = [k for k in rows if k in base]
            if len(common) < 5:
                continue
            db = np.array([rows[k]["bdice"] - base[k]["bdice"] for k in common])
            dd = np.array([rows[k]["dice"] - base[k]["dice"] for k in common])
            try:
                p = stats.wilcoxon(db).pvalue if np.any(db != 0) else 1.0
            except Exception:
                p = float("nan")
            print("  %-18s %+10.2f %+10.3f %8d %8d %10.3f" %
                  (label, db.mean(), dd.mean(), int((db > 0).sum()), int((db < 0).sum()), p))
        print()
        print("  reading: a mean d_bdice near 0 with a LARGE |disp_mean| in both models means")
        print("  the residual is the same systematic contour displacement, not a new failure.")


if __name__ == "__main__":
    main()
