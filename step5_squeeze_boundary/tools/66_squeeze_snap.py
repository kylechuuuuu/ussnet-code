#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/tools/66_squeeze_snap.py

RETHOUGHT USE OF THE SQUEEZE MODEL: the interval is not a voting scheme, it is
a LEGAL SEARCH RANGE for the boundary.  The theorem's premise says

        true boundary  inside  [inner, outer]

so inference stops asking "which head wins the vote" and starts asking "which
point inside the interval does the image evidence support".  Every predicted
boundary pixel is moved along its local image-edge normal to the strongest edge
response, limited to

  (a) an unconstrained window of +-kappa px          -> classical snapping
      (reproduces step4 section 5.5, the only route that ever scored +1.3)
  (b) the SAME window but clipped to the squeeze interval (outer & ~inner)
      -> the pinch ARBITRATES how far image evidence may move the annotation:
      a wide interval means the model is unsure, the image may speak;
      a narrow one means stay on the annotation convention.

(kappa, gamma) are tuned on the fold-0 TRAIN split only, then applied once to
the 31 validation images.  The same snapping is also applied to the nnU-Net
baseline PNGs (no interval exists there: only the free variant), separating
        evidence gain     (the snap itself)
        arbitration gain  (the interval constraint on top)

usage:  python tools/66_squeeze_snap.py --pred-dir ... --inner-dir ... --outer-dir ... --label ...
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.dirname(HERE)
USS = os.path.dirname(STEP5)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import load_binary, read_split                      # noqa: E402

GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
EPS = 1e-7
DS_SIGMA = 2.0        # edge-field smoothing
STEP = 0.5            # search step along the normal (px)


def surface(m):
    if not m.any():
        return m.copy()
    return m & ~ndimage.binary_erosion(m, ndimage.generate_binary_structure(2, 1),
                                       border_value=0)


def bdice_1px(pred, gt, tol=1):
    from skimage import morphology
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


def edge_field(img):
    """Per-image normalised edge strength + its unit normal (nx, ny)."""
    blur = ndimage.gaussian_filter(img, DS_SIGMA)
    gy, gx = np.gradient(blur)
    n = np.maximum(np.hypot(gx, gy), 1e-9)
    mag = np.clip(np.hypot(gx, gy) / np.percentile(n * 0 + np.hypot(gx, gy), 99.5), 0.0, 1.0)
    return mag, gx / n, gy / n


def snap(pred, img_f, kmax, gamma, interval=None):
    """Move each boundary pixel along its normal to the strongest edge response.

    score(d) = mag(p + d*n) - gamma * d^2 ;  if `interval` is given, a candidate
    must lie inside  outer & ~inner  (or stay within +-0.5 px of the prediction).
    Returns (snapped_mask, mean_|d|).
    """
    mag, nx, ny = img_f
    pb = surface(pred)
    ys, xs = np.nonzero(pb)
    if ys.size == 0:
        return pred.copy(), 0.0
    h, w = pred.shape
    offsets = np.arange(-kmax, kmax + 1e-9, STEP)

    nxp = nx[ys, xs][:, None]
    nyp = ny[ys, xs][:, None]
    yy = np.clip(ys[:, None] + nyp * offsets[None, :], 0, h - 1)
    xx = np.clip(xs[:, None] + nxp * offsets[None, :], 0, w - 1)
    yi = yy.astype(np.int32)
    xi = xx.astype(np.int32)
    vals = mag[yi, xi]
    score = vals - gamma * (offsets[None, :] ** 2)
    if interval is not None:
        inside = interval[yi, xi] > 0
        legal = inside | (np.abs(offsets)[None, :] <= 0.5001)
        score = np.where(legal, score, -1e9)
    dbest = offsets[np.argmax(score, axis=1)]

    out = pred.copy()
    for (y, x, d) in zip(ys, xs, dbest):
        if d == 0:
            continue
        out[y, x] = False
        y2 = int(np.clip(round(y + ny[y, x] * d), 0, h - 1))
        x2 = int(np.clip(round(x + nx[y, x] * d), 0, w - 1))
        out[y2, x2] = True
    return out, float(np.mean(np.abs(dbest)))


VARIANTS = []                                  # key -> (kmax, gamma)
for kmax in (1.0, 1.5, 2.0, 3.0):
    for gamma in (0.0, 0.02):
        VARIANTS.append(("k%.1f_g%.2f" % (kmax, gamma), kmax, gamma))


def load_pred(pred_dir, name):
    p = os.path.join(pred_dir, name)
    return load_binary(p, threshold=0.5) if os.path.isfile(p) else None


def one_case(name, pred_dir, inner_dir, outer_dir, want_interval, kappa, gamma):
    """-> dict with raw / snap / snap_iv bdice (and dice) for one image."""
    gt = load_binary(os.path.join(GT_DIR, name), threshold=0)
    pred0 = load_pred(pred_dir, name)
    if gt is None or pred0 is None or gt.shape != pred0.shape:
        return None
    img_path = os.path.join(IMG_DIR, name)
    if not os.path.isfile(img_path):
        return None
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    if img.shape != gt.shape:
        return None
    f = edge_field(img)

    row = {"raw_bd": bdice_1px(pred0, gt), "raw_dice": dice(pred0, gt),
           "n_bd_px": int(surface(pred0).sum())}
    s, md = snap(pred0, f, kappa, gamma)
    row["snap_bd"] = bdice_1px(s, gt)
    row["snap_dice"] = dice(s, gt)
    row["snap_mean_d"] = md
    row["snap_iv_bd"] = None
    if want_interval and inner_dir and outer_dir:
        ip = os.path.join(inner_dir, name)
        op = os.path.join(outer_dir, name)
        if os.path.isfile(ip) and os.path.isfile(op):
            mi = load_binary(ip, threshold=0.5)
            mo = load_binary(op, threshold=0.5)
            interval = mo & ~mi
            s2, md2 = snap(pred0, f, kappa, gamma, interval=interval)
            row["snap_iv_bd"] = bdice_1px(s2, gt)
            row["snap_iv_dice"] = dice(s2, gt)
            row["snap_iv_mean_d"] = md2
            row["band_px"] = float(interval.sum() / max(1e-6, surface(pred0).sum()))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--inner-dir", default=None)
    ap.add_argument("--outer-dir", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    want_interval = args.inner_dir and args.outer_dir \
        and os.path.isdir(args.inner_dir) and os.path.isdir(args.outer_dir)

    train_names, val_names = read_split(USS)

    # ---- 1. tune (kappa, gamma) on TRAIN, constraint-free -------------------
    print("[66] %s | tuning kappa/gamma on TRAIN (%d) ..." % (args.label, len(train_names)))
    best = None                                    # (key, mean bdice)
    for key, kmax, gamma in VARIANTS:
        vals = []
        for name in train_names:
            r = one_case(name, args.pred_dir, None, None, False, kmax, gamma)
            if r:
                vals.append(r["snap_bd"])
        m = float(np.nanmean(vals)) if vals else float("nan")
        print("[66]   %-14s train bdice %.2f" % (key, m))
        if best is None or (m == m and m > best[1]):
            best = (key, m)
    key0, kmax, gamma = [v for v in VARIANTS if v[0] == best[0]][0]
    print("[66]   chosen: kmax=%.1f gamma=%.2f (train %.2f)" % (kmax, gamma, best[1]))

    # ---- 2. apply to VAL once ----------------------------------------------
    rows = {}
    for name in val_names:
        r = one_case(name, args.pred_dir, args.inner_dir, args.outer_dir,
                     want_interval, kmax, gamma)
        if r:
            rows[name] = r

    def col(k):
        v = [r[k] for r in rows.values() if r.get(k) is not None]
        return (float(np.nanmean(v)) if v else None), len(v)

    raw_bd, _ = col("raw_bd")
    snap_bd, _ = col("snap_bd")
    iv_bd, n_iv = col("snap_iv_bd")
    raw_dice, _ = col("raw_dice")
    snap_dice, _ = col("snap_dice")

    out = OrderedDict(label=args.label, pred_dir=args.pred_dir,
                      kappa=kmax, gamma=gamma,
                      val={"n": len(rows), "raw_bdice": raw_bd, "snap_free_bdice": snap_bd,
                           "raw_dice": raw_dice, "snap_dice": snap_dice,
                           "snap_interval_bdice": iv_bd, "n_interval": n_iv})
    print("\n[66] ==== %s ==== (VAL %d images, mirroring NOT involved)" % (args.label, len(rows)))
    print("[66]   raw bdice@1px        %.2f | dice %.2f" % (raw_bd, raw_dice))
    print("[66]   snap (free +-k)      %.2f | dice %.2f  [evidence gain %+0.2f]"
          % (snap_bd, snap_dice, snap_bd - raw_bd))
    if iv_bd is not None:
        print("[66]   snap IN interval     %.2f              [arbitration delta %+0.2f]"
              % (iv_bd, iv_bd - snap_bd))
    fp = args.out or os.path.join(STEP5, "results", "snap_%s.json"
                                  % args.label.replace(" ", "_"))
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[66] written %s" % fp)


if __name__ == "__main__":
    main()
