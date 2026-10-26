#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DESIGN_NEXT.md item A -- can a *directional* normal-direction snap recover the
boundary from the image alone, with NO learning at all?

Why this probe
--------------
§5.3 showed probability smoothing does nothing (residual is not high-frequency
jitter).  §5.4 showed ensembling does nothing (it is not variance).  §5.1 showed
a shape subspace can only compress (it is not "unsmooth shape").
§5.2 oracle showed the residual IS a directional ~2.6 px (median) normal
displacement, and moving along the normal recovers +49.7 boundary-Dice points.

This probe asks the cheap question in between: is the *image* informative enough
at the right scale that a purely classical, direction-aware search -- walk along
the contour normal, stop at the nearest strong |dI/dt| ridge -- already beats the
model?  If yes, item A is worth a learned refiner and the refiner only has to do
better than a 5-line baseline.  If no, the residual is not in the raw speckle and
item A needs the model's own features (a second stage), not the image.

Protocol (identical to §5.1 / §5.2 so the numbers sit side by side)
------------------------------------------------------------------
* 31 fold-0 validation images, shared split, cc1 (largest connected component)
  of each prediction so topology cleanup is never credited to the snap.
* contour -> resample 400 pts -> outward normals -> sample I along the normal in
  [-16, +16] px at 0.5 px -> |dI/dt| ridge search.
* candidate selection = the ridge NEAREST to t=0 among those >= frac * max ridge
  (a global argmax jumps to unrelated echogenic edges; report it as a variant to
  show the difference).
* re-rasterise with the exact even-odd fill, score Dice / boundary-Dice@1px,
  @2px / boundary length.
* `ctrl` (t=0, round-tripped through resample+rasterise) controls for any effect
  of the resampling itself.

Baselines to beat (FINDINGS §4): U-Net 93.01 / 23.25 / 33.95 / 1956.5 px,
nnU-Net 96.11 / 35.16 / 50.15 / 1094.2 px.  GT boundary length 1094.2 px.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS = os.path.dirname(STEP4)

spec = importlib.util.spec_from_file_location(
    "shape57", os.path.join(HERE, "57_shape_prior_ceiling.py"))
s57 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s57)
spec58 = importlib.util.spec_from_file_location(
    "refine58", os.path.join(HERE, "58_normal_refine_oracle.py"))
s58 = importlib.util.module_from_spec(spec58)
spec58.loader.exec_module(s58)

IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
NPT = 400
TS = np.arange(-16.0, 16.0001, 0.5)


def load_image(name):
    base = name if name.lower().endswith(".png") else name + ".png"
    p = os.path.join(IMG_DIR, base)
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p).convert("L"), dtype=np.float32)
    return a


def sample_along(img, pts, nrm, ts):
    """(N,2) pts, (N,2) normals, (T,) offsets -> (N,T) bilinear samples."""
    q = pts[:, None, :] + ts[None, :, None] * nrm[:, None, :]
    r = q[..., 0].ravel()
    c = q[..., 1].ravel()
    v = ndimage.map_coordinates(img, [r, c], order=1, mode="nearest")
    return v.reshape(len(pts), len(ts))


def ridge_profile(img, pts, nrm, ts, smooth_t=1.0):
    P = sample_along(img, pts, nrm, ts)
    G = np.abs(np.gradient(P, ts, axis=1))
    if smooth_t > 0:
        G = ndimage.gaussian_filter1d(G, smooth_t, axis=1, mode="nearest")
    return G


def pick_nearest(G, ts, frac, maxstep):
    thr = frac * G.max(axis=1, keepdims=True)
    ok = (G >= thr) & (np.abs(ts)[None, :] <= maxstep)
    cost = np.where(ok, np.abs(ts)[None, :], 1e18)
    j = np.argmin(cost, axis=1)
    t = ts[j].astype(float).copy()
    t[~ok.any(axis=1)] = 0.0
    return t


def pick_argmax(G, ts, maxstep):
    Gm = np.where(np.abs(ts)[None, :] <= maxstep, G, -1.0)
    return ts[np.argmax(Gm, axis=1)].astype(float)


def blen(m):
    return float(s57.surface(m).sum())


def snap_one(img, cc1, ts, frac, maxstep, mode="nearest", smooth_t=1.0):
    c = s57.contour_of(cc1)
    if c is None:
        return None, None
    pts = s57.resample_closed(c, NPT, 0)
    nrm = s58.normals_of(pts, cc1)
    if mode == "ctrl":
        t = np.zeros(len(pts))
    else:
        G = ridge_profile(img, pts, nrm, ts, smooth_t)
        t = (pick_nearest(G, ts, frac, maxstep) if mode == "nearest"
             else pick_argmax(G, ts, maxstep))
    new = pts + t[:, None] * nrm
    m = s57.rasterize(new, cc1.shape)
    a0, a1 = float(cc1.sum()), float(m.sum())
    if a1 <= 0 or not (0.5 * a0 <= a1 <= 2.0 * a0):
        m = cc1.copy()
    return m, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(STEP4, "results",
                                                  "edge_snap_probe.json"))
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="Gaussian pre-smoothing of the image")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    _, val = s57.read_split(USS, exp="ultrasound_prostate_73")
    if args.max:
        val = val[:args.max]

    models = {
        "unet": os.path.join(USS, "unet", "results", "ultrasound_prostate_73"),
        "nnunet_best": os.path.join(USS, "step1_nnunet_baseline", "predictions",
                                    "nnunet_2d_fold0_checkpoint_best_png"),
    }

    variants = [("ctrl", None, None, "ctrl")]
    for frac in (0.3, 0.5, 0.7):
        for ms in (4.0, 8.0, 16.0):
            variants.append(("snap_f%02d_s%02d" % (frac * 100, ms), frac, ms,
                             "nearest"))
    variants.append(("argmax_s16", None, 16.0, "argmax"))

    rows = []
    for nm in val:
        gt = s57.load_gt(nm)
        img0 = load_image(nm)
        if gt is None or img0 is None:
            print("[snap] WARN missing %s" % nm)
            continue
        shape = gt.shape
        img = (ndimage.gaussian_filter(img0, args.sigma) if args.sigma > 0
               else img0)
        for tag, d in models.items():
            pred = s57.load_pred(os.path.join(d, nm if nm.lower().endswith(".png")
                                              else nm + ".png"))
            if pred is None:
                print("[snap] WARN no pred %s %s" % (tag, nm))
                continue
            if pred.shape != shape:
                print("[snap] WARN shape %s %s %s vs %s" % (tag, nm, pred.shape, shape))
                continue
            cc1 = s57.largest_cc(pred)
            rec = dict(name=nm, model=tag,
                       dice_raw=s57.dice(pred, gt), bd1_raw=s57.bdice(pred, gt, 1),
                       blen_raw=blen(pred))
            for vname, frac, ms, mode in variants:
                m, t = snap_one(img, cc1, TS, frac, ms, mode)
                if m is None:
                    continue
                rec["dice_" + vname] = s57.dice(m, gt)
                rec["bd1_" + vname] = s57.bdice(m, gt, 1)
                rec["bd2_" + vname] = s57.bdice(m, gt, 2)
                rec["blen_" + vname] = blen(m)
                if t is not None:
                    rec["tmove_" + vname] = float(np.mean(np.abs(t)))
                    rec["tfrac_" + vname] = float(np.mean(np.abs(t) > 0.5))
            rows.append(rec)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)

    names = [v[0] for v in variants]
    print("\n[snap] n=%d val images, image sigma=%.2f, NPT=%d\n" % (len(val), args.sigma, NPT))
    for tag in models:
        sel = [r for r in rows if r["model"] == tag]
        if not sel:
            continue
        print("== %s (n=%d) ==" % (tag, len(sel)))
        print("%-16s %8s %8s %8s %9s %7s %7s" %
              ("variant", "Dice", "bd1px", "bd2px", "bnd_len", "|t|", ">0.5px"))
        print("%-16s %8.2f %8.2f %8.2f %9.1f" %
              ("raw", np.mean([r["dice_raw"] for r in sel]),
               np.mean([r["bd1_raw"] for r in sel]),
               np.nan, np.mean([r["blen_raw"] for r in sel])))
        for vn in names:
            k = "dice_" + vn
            if not any(k in r for r in sel):
                continue
            print("%-16s %8.2f %8.2f %8.2f %9.1f %7.2f %7.3f" %
                  (vn,
                   np.mean([r[k] for r in sel]),
                   np.mean([r["bd1_" + vn] for r in sel]),
                   np.mean([r["bd2_" + vn] for r in sel]),
                   np.mean([r["blen_" + vn] for r in sel]),
                   np.mean([r.get("tmove_" + vn, 0.0) for r in sel]),
                   np.mean([r.get("tfrac_" + vn, 0.0) for r in sel])))
        print()
    print("[snap] wrote %s" % args.out)


if __name__ == "__main__":
    main()
