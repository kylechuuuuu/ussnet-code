#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_failure_analysis/scripts/30_error_decomposition.py

Where exactly do the two baselines fail?  For every validation image and every
method this script decomposes the error into

  * a DISTANCE-TO-BOUNDARY profile  - is the error a thin boundary band (shape
    is right, edge is off by a pixel or two) or a gross miss / spurious blob?
  * the effect of trivial POST-PROCESSING (largest connected component, hole
    filling, small-object removal)   - how much of the loss is just speckle?
  * boundary Dice at 1 px and 2 px tolerance
  * behaviour at the INTERNAL 127|255 interface of the annotation
  * stratification by gland size and by image resolution

usage:
    python scripts/30_error_decomposition.py --tags unet_73:<pred dir> ...
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "..", "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import load_binary, load_gray, read_split  # noqa: E402

STEP3 = os.path.dirname(HERE)
USS = os.path.dirname(STEP3)
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
RES_DIR = os.path.join(STEP3, "results")
FIG_DIR = os.path.join(STEP3, "figures")

BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 10 ** 9)]
EPS = 1e-7


def surface(m):
    if not m.any():
        return np.zeros_like(m)
    return m & ~ndimage.binary_erosion(m, ndimage.generate_binary_structure(2, 1),
                                       border_value=0)


def dice(a, b):
    return 2.0 * float(np.logical_and(a, b).sum()) / (float(a.sum()) + float(b.sum()) + EPS) * 100


def dilate(m, r):
    return ndimage.binary_dilation(m, morphology.disk(r))


def analyze(pred, gt, gray):
    out = {}
    npix = gt.size
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(pred, ~gt)
    gb = surface(gt)
    d_gt_b = ndimage.distance_transform_edt(~gb)

    out["dice"] = dice(pred, gt)
    out["fn_px"] = int(fn.sum())
    out["fp_px"] = int(fp.sum())
    out["gt_px"] = int(gt.sum())

    # ---- distance-to-GT-boundary profile of the error -----------------------
    fn_hist, fp_hist, gt_hist = [], [], []
    for lo, hi in BANDS:
        sel = (d_gt_b >= lo) & (d_gt_b < hi)
        fn_hist.append(int(np.logical_and(fn, sel).sum()))
        fp_hist.append(int(np.logical_and(fp, d_gt_b >= lo).sum()
                           - np.logical_and(fp, d_gt_b >= hi).sum()))
        gt_hist.append(int(np.logical_and(gt, sel).sum()))
    out["fn_hist"] = fn_hist
    out["fp_hist"] = fp_hist
    out["gt_hist"] = gt_hist
    out["fn_share_thin_2px"] = fn_hist[0] / (fn.sum() + EPS) * 100
    out["fn_share_gross_10px"] = sum(fn_hist[3:]) / (fn.sum() + EPS) * 100
    out["fp_share_thin_2px"] = fp_hist[0] / (fp.sum() + EPS) * 100
    out["fp_share_gross_10px"] = sum(fp_hist[3:]) / (fp.sum() + EPS) * 100

    # distance profile of the error, normalised by the area available in that band
    out["fn_rate_by_band"] = [fn_hist[i] / (gt_hist[i] + EPS) * 100 for i in range(len(BANDS))]

    # ---- post-processing ----------------------------------------------------
    lab, n_cc = ndimage.label(pred)
    out["n_pred_cc"] = int(n_cc)
    if n_cc > 1:
        sizes = ndimage.sum(pred, lab, range(1, n_cc + 1))
        out["largest_cc_frac"] = float(sizes.max()) / (pred.sum() + EPS) * 100
        keep = np.isin(lab, [int(np.argmax(sizes)) + 1])
    else:
        out["largest_cc_frac"] = 100.0 if pred.any() else 0.0
        keep = pred
    out["dice_largest_cc"] = dice(keep, gt)
    out["dice_filled"] = dice(ndimage.binary_fill_holes(pred), gt)
    post = ndimage.binary_fill_holes(keep)
    out["dice_post_lcc_fill"] = dice(post, gt)
    out["dice_post_all"] = dice(ndimage.binary_fill_holes(
        morphology.remove_small_objects(pred, 64)), gt)
    out["holes_px"] = int(np.logical_and(pred, ndimage.binary_fill_holes(pred) & ~pred).sum())

    # ---- boundary Dice at tolerance -----------------------------------------
    pb = surface(pred)
    for tol in (1, 2):
        gg = dilate(gb, tol)
        pp = dilate(pb, tol)
        prec = float(np.logical_and(pb, gg).sum()) / (pb.sum() + EPS)
        rec = float(np.logical_and(gb, pp).sum()) / (gb.sum() + EPS)
        out["bdice_%dpx" % tol] = 2 * prec * rec / (prec + rec + EPS) * 100

    # ---- internal 127|255 interface -----------------------------------------
    z127, z255 = gray == 127, gray == 255
    iface = np.logical_and(z127, dilate(z255, 1))
    if iface.any():
        d_iface = ndimage.distance_transform_edt(~iface)
        near = d_iface < 5
        out["fn_near_iface_pct"] = float(np.logical_and(fn, near).sum()) / (fn.sum() + EPS) * 100
        out["fp_near_iface_pct"] = float(np.logical_and(fp, near).sum()) / (fp.sum() + EPS) * 100
        out["gt_near_iface_pct"] = float(np.logical_and(gt, near).sum()) / (gt.sum() + EPS) * 100
    else:
        out["fn_near_iface_pct"] = out["fp_near_iface_pct"] = out["gt_near_iface_pct"] = float("nan")

    # ---- size / resolution stratification ------------------------------------
    out["gt_area"] = int(gt.sum())
    out["image_h"], out["image_w"] = gt.shape
    out["fg_ratio"] = float(gt.sum()) / npix * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="entries of the form name:prediction_dir")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    _, val_names = read_split(USS)
    names = val_names

    for spec in args.tags:
        tag, pdir = spec.split(":", 1)
        pdir = pdir if os.path.isabs(pdir) else os.path.abspath(os.path.join(STEP3, pdir))
        rows = []
        for n in names:
            pp = os.path.join(pdir, n)
            if not os.path.exists(pp):
                continue
            pred = load_binary(pp)
            gt = load_binary(os.path.join(GT_DIR, n))
            gray = load_gray(os.path.join(GT_DIR, n))
            r = analyze(pred, gt, gray)
            r["name"] = n
            rows.append(r)

        csv_path = os.path.join(RES_DIR, "%s_error_decomposition.csv" % tag)
        keys = [k for k in rows[0] if k != "name"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name"] + keys)
            w.writeheader()
            w.writerows(rows)

        def mean(k):
            v = [r[k] for r in rows if r[k] == r[k]]
            return float(np.mean(v)) if v else float("nan")

        dl = np.array([r["dice"] for r in rows])
        areas = np.array([r["gt_area"] for r in rows], dtype=float)
        small = areas < np.median(areas)
        summ = {
            "tag": tag, "pred_dir": pdir, "n": len(rows),
            "dice_mean": float(dl.mean()),
            "dice_after_postproc_mean": mean("dice_post_all"),
            "dice_gain_postproc": mean("dice_post_all") - float(dl.mean()),
            "dice_largest_cc_mean": mean("dice_largest_cc"),
            "dice_filled_mean": mean("dice_filled"),
            "bdice_1px_mean": mean("bdice_1px"), "bdice_2px_mean": mean("bdice_2px"),
            "fn_px_mean": mean("fn_px"), "fp_px_mean": mean("fp_px"),
            "fn_share_thin_2px": mean("fn_share_thin_2px"),
            "fn_share_gross_10px": mean("fn_share_gross_10px"),
            "fp_share_thin_2px": mean("fp_share_thin_2px"),
            "fp_share_gross_10px": mean("fp_share_gross_10px"),
            "fn_rate_by_band": [mean("fn_rate_by_band"), None],
            "n_pred_cc_mean": mean("n_pred_cc"),
            "largest_cc_frac_mean": mean("largest_cc_frac"),
            "holes_px_mean": mean("holes_px"),
            "fn_near_iface_pct": mean("fn_near_iface_pct"),
            "fp_near_iface_pct": mean("fp_near_iface_pct"),
            "gt_near_iface_pct": mean("gt_near_iface_pct"),
            "dice_small_glands": float(dl[small].mean()) if small.any() else float("nan"),
            "dice_large_glands": float(dl[~small].mean()) if (~small).any() else float("nan"),
            "corr_area_dice": float(np.corrcoef(areas, dl)[0, 1]),
        }
        # per-band means of the FN rate
        br = np.array([r["fn_rate_by_band"] for r in rows], dtype=float)
        summ["fn_rate_by_band"] = [float(x) for x in br.mean(0)]
        # resolution groups
        res = {}
        for r in rows:
            k = "%dx%d" % (r["image_w"], r["image_h"])
            res.setdefault(k, []).append(r["dice"])
        summ["dice_by_resolution"] = {k: float(np.mean(v)) for k, v in res.items()}

        with open(os.path.join(RES_DIR, "%s_error_summary.json" % tag), "w") as f:
            json.dump(summ, f, indent=2)

        print("=" * 78)
        print("tag %s  (%d val images)   Dice %.3f" % (tag, len(rows), summ["dice_mean"]))
        print("  Dice after post-processing (small-obj removal + hole fill): %.3f  (gain %+.3f)"
              % (summ["dice_after_postproc_mean"], summ["dice_gain_postproc"]))
        print("  Dice keeping only largest CC: %.3f   filling holes: %.3f"
              % (summ["dice_largest_cc_mean"], summ["dice_filled_mean"]))
        print("  boundary Dice @1px %.2f  @2px %.2f" % (summ["bdice_1px_mean"], summ["bdice_2px_mean"]))
        print("  error budget: FN %d px / FP %d px per image" % (summ["fn_px_mean"], summ["fp_px_mean"]))
        print("  FN within 2 px of GT boundary: %.1f%%   FN farther than 10 px: %.1f%%"
              % (summ["fn_share_thin_2px"], summ["fn_share_gross_10px"]))
        print("  FP within 2 px of GT boundary: %.1f%%   FP farther than 10 px: %.1f%%"
              % (summ["fp_share_thin_2px"], summ["fp_share_gross_10px"]))
        print("  FN rate by distance band (0-2,2-5,5-10,10-20,>20 px): "
              + ", ".join("%.1f%%" % x for x in summ["fn_rate_by_band"][:5]))
        print("  predicted connected components: %.1f (largest holds %.1f%%)   holes %d px"
              % (summ["n_pred_cc_mean"], summ["largest_cc_frac_mean"], summ["holes_px_mean"]))
        print("  error near the internal 127|255 interface: FN %.1f%%  FP %.1f%%  "
              "(interface neighbourhood holds %.1f%% of GT)"
              % (summ["fn_near_iface_pct"], summ["fp_near_iface_pct"], summ["gt_near_iface_pct"]))
        print("  Dice small glands %.2f vs large glands %.2f  (corr area-Dice %+.2f)"
              % (summ["dice_small_glands"], summ["dice_large_glands"], summ["corr_area_dice"]))
        print("  Dice by resolution: " + ", ".join("%s %.2f" % kv for kv in summ["dice_by_resolution"].items()))
        print("  -> %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
