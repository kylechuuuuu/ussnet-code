#!/usr/bin/env python3
"""95_shape_relax.py -- mechanism 2 candidate: evidence-gated shape relaxation.

step9 measurements (90/91/94) say:
  * the whole HD95 is made of one or two coherent arcs of |t|>10 px (14.6% of
    points), 31/31 cases, 64% of the tail on the anterior TZ-outer arc;
  * overshoot (prediction too far out) beats undershoot 1037:718;
  * in that region the TRUE boundary has no image edge (GT is the strongest edge
    within +-20 px at only 16.3% of tail points; step across it 11.3 vs 23.2 at
    points the model already gets right).
So the anterior boundary is not hidden behind a wrong choice of edge -- it is
mostly NOT MARKED.  If it is at least a smooth continuation of the part the
model does get right, then the fix is a shape prior applied ONLY where the image
evidence is absent (never touching the 46.7% of points already within 2 px).

This script tests that, training-free, in the radial parameterisation r(theta)
around the mask centroid:
  w_i = clip(G(0)/Gmax, .05, 1) ** p     -> how much this point counts as evidence
  fit a PERIODIC smoothing spline r(theta) with these weights (S ~ scale^2 px^2)
  r_new = r_fit            (replace)   or   min(r, r_fit)   (shrink only)
  p=0 is the control: uniform smoothing, i.e. shape prior with no evidence gate.
Also the ORACLE of this family: put the GT ray radius at every unsupported point
and leave the supported ones alone -> the ceiling of "fill in the unsupported
arcs perfectly".

CPU only.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.interpolate import splrep, splev

HERE = os.path.dirname(os.path.abspath(__file__))
STEP9 = os.path.dirname(HERE)
USS = os.path.dirname(STEP9)
STEP6 = os.path.join(USS, "step6_sandwich_squeeze")
STEP2 = os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts")
sys.path.insert(0, os.path.join(STEP6, "tools"))
sys.path.insert(0, STEP2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = _load("arb", os.path.join(STEP6, "tools", "63_arbitrate.py"))
s57, s58 = arb.s57, arb.s58
ev = _load("eval_lib", os.path.join(STEP2, "eval_lib.py"))

NPZ = os.path.join(STEP6, "predictions", "sq2resencw25_checkpoint_best_base_nifti")
NPT = 400
TS = np.arange(-24.0, 24.0001, 0.5)
SIGMA = 1.5


def ray_r(mask, cx, cy, th, rmax=560.0, step=0.25):
    """outermost mask radius along each ray from (cx, cy)."""
    rr = np.arange(0.0, rmax + step, step)
    x = cx + np.cos(th)[:, None] * rr[None, :]
    y = cy + np.sin(th)[:, None] * rr[None, :]
    v = ndimage.map_coordinates(mask.astype(np.float32), [y.ravel(), x.ravel()],
                                order=0, mode="nearest").reshape(len(th), len(rr)) > 0.5
    any_ = v.any(axis=1)
    idx = np.where(any_, len(rr) - 1 - np.argmax(v[:, ::-1], axis=1), 0)
    return rr[idx], any_


def fit_periodic(th, r, w, S):
    o = np.argsort(th)
    th_s, r_s, w_s = th[o], r[o], w[o]
    th_e = np.concatenate([th_s, th_s[:1] + 2 * np.pi])
    r_e = np.concatenate([r_s, r_s[:1]])
    w_e = np.concatenate([w_s, w_s[:1]])
    tck = splrep(th_e, r_e, w=w_e, per=True, s=S)
    return splev(th, tck), o


def rebuild(cx, cy, th, r, shape, dense=1400):
    """r(theta) -> binary mask (largest cc)."""
    tt = np.linspace(th.min(), th.min() + 2 * np.pi, dense)
    rr = np.interp(tt, np.concatenate([th, th[:1] + 2 * np.pi]),
                   np.concatenate([r, r[:1]]))
    x = cx + rr * np.cos(tt)
    y = cy + rr * np.sin(tt)
    poly = np.stack([y, x], axis=1)
    m = s57.rasterize(poly, shape)
    return s57.largest_cc(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=NPZ)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scales", default="0.75,1.5,3.0,6.0")
    ap.add_argument("--powers", default="0,1,3,10")
    ap.add_argument("--out-dir", default=os.path.join(STEP9, "results"))
    ap.add_argument("--out", default="shape_relax_sq2resencw25.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scales = [float(x) for x in args.scales.split(",")]
    powers = [float(x) for x in args.powers.split(",")]

    train, val = arb.read_split(USS)
    names = train if args.split == "train" else val
    if args.limit:
        names = names[:args.limit]

    per_case, acc = [], {}
    for name in names:
        stem = arb.stem_of(name)
        gt = s57.load_gt(stem)
        arr = arb.load_prob(args.npz_dir, stem)
        if gt is None or arr is None:
            continue
        img = arb.load_image(stem)
        p = arb.align_prob(arr, gt.shape)
        M0 = s57.largest_cc(p > 0.5)
        c = s57.contour_of(M0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, M0)

        ys, xs = np.nonzero(M0)
        cy, cx = ys.mean(), xs.mean()
        th = np.arctan2(pts[:, 0] - cy, pts[:, 1] - cx)
        r = np.hypot(pts[:, 0] - cy, pts[:, 1] - cx)
        # keep theta monotone for the periodic fit
        order = np.argsort(th)
        th, r, pts, nrm = th[order], r[order], pts[order], nrm[order]

        imgf = ndimage.gaussian_filter(img, SIGMA)
        G = arb.ridge_profile(imgf, pts, nrm, TS, 1.0)
        gmax = np.maximum(G.max(axis=1), 1e-9)
        g0 = G[:, int(np.argmin(np.abs(TS)))]
        sup = np.clip(g0 / gmax, 0.05, 1.0)
        unsup = g0 / gmax < 0.25

        r_gt_ray, ok_ray = ray_r(gt, cx, cy, th)

        cands = {"raw": M0}
        # identity self-test of the rebuild path
        cands["rebuilt_identity"] = rebuild(cx, cy, th, r, gt.shape)
        for S in scales:
            for pw in powers:
                w = sup ** pw
                rf, _ = fit_periodic(th, r, w, S * S * len(r))
                cands["spl_p%g_S%g" % (pw, S)] = rebuild(cx, cy, th, rf, gt.shape)
                cands["shr_p%g_S%g" % (pw, S)] = rebuild(cx, cy, th, np.minimum(r, rf), gt.shape)
        # oracle of this family: GT radius at the unsupported points only
        r_or = r.copy()
        r_or[unsup & ok_ray] = r_gt_ray[unsup & ok_ray]
        cands["oracle_unsup"] = rebuild(cx, cy, th, r_or, gt.shape)
        # pure-GT-radius control (upper bound of the whole radial family)
        r_full = np.where(ok_ray, r_gt_ray, r)
        cands["oracle_full_ray"] = rebuild(cx, cy, th, r_full, gt.shape)

        row = {"name": stem, "n_unsup": int(unsup.sum()), "n": int(len(r)),
               "dice_gt_roundtrip": s57.dice(cands["rebuilt_identity"], gt)}
        for k, m in cands.items():
            hd, assd, _ = ev.surface_metrics(m, gt)
            row["bdice_" + k] = s57.bdice(m, gt, tol=1)
            row["dice_" + k] = s57.dice(m, gt)
            if k in ("raw", "oracle_unsup", "oracle_full_ray") or k.startswith("spl_p1_") or k.startswith("spl_p10_"):
                row["hd95_" + k] = hd
                row["assd_" + k] = assd
        per_case.append(row)
        print("%-22s unsup %3d/%d  raw %5.2f  id %5.2f  orac_unsup %5.2f  orac_full %5.2f" %
              (stem, row["n_unsup"], row["n"], row["bdice_raw"], row["dice_gt_roundtrip"],
               row["bdice_oracle_unsup"], row["bdice_oracle_full_ray"]))
        sys.stdout.flush()

    keys = [k[6:] for k in per_case[0] if k.startswith("bdice_")]
    for k in ["raw"] + sorted(x for x in keys if x != "raw"):
        v = np.array([r["bdice_" + k] for r in per_case], dtype=float)
        dv = np.array([r["bdice_" + k] - r["bdice_raw"] for r in per_case])
        hk = "hd95_" + k
        hv = np.array([r.get(hk, np.nan) for r in per_case], dtype=float)
        acc[k] = {"bdice1_mean": float(np.nanmean(v)), "delta": float(np.nanmean(dv)),
                  "win": int((dv > 0).sum()), "loss": int((dv < 0).sum()),
                  "hd95_mean": float(np.nanmean(hv)) if np.isfinite(hv).any() else None,
                  "hd95_median": float(np.nanmedian(hv)) if np.isfinite(hv).any() else None}
    out = {"split": args.split, "n_cases": len(per_case), "scales": scales,
           "powers": powers, "per_case": per_case, "summary": acc}
    with open(os.path.join(args.out_dir, args.out), "w") as fh:
        json.dump(out, fh, indent=1)

    print("=" * 92)
    print("SHAPE RELAX  split=%s  n=%d  (bdice@1px, sorted by mean)" % (args.split, len(per_case)))
    for k, d in sorted(acc.items(), key=lambda kv: -kv[1]["bdice1_mean"]):
        print("  %-22s %7.3f  d%+7.3f  win/loss %2d/%2d  HD95 %s" %
              (k, d["bdice1_mean"], d["delta"], d["win"], d["loss"],
               ("%.2f" % d["hd95_mean"]) if d["hd95_mean"] is not None else "--"))


if __name__ == "__main__":
    main()
