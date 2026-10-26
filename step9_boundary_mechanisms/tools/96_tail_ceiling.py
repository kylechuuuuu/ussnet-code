#!/usr/bin/env python3
"""96_tail_ceiling.py -- where exactly is the recoverable boundary error?

95 showed the learned-free shape relaxation loses (-0.4 best). But its oracle
(repair the radius at every image-unsupported point with the GT radius) was only
+10.6, while repairing ALL points was +69.6. That does not add up with 94 (which
said the tail points have no image edge): it means my "unsupported" gate does not
coincide with the tail.

This script decomposes the ceiling by ERROR MAGNITUDE: repair the radius with the
GT radius only at points with |t_gt| > tau, and see how much bdice@1px comes back.
It also asks whether the GT boundary itself is radial-smooth (round-trip the GT
ray radius through the periodic spline) -- if that fails, the radial
parameterisation is the wrong shape space and any radial shape prior is dead on
arrival.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage

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
sr = _load("sr95", os.path.join(HERE, "95_shape_relax.py"))
ev = _load("eval_lib", os.path.join(STEP2, "eval_lib.py"))
s57, s58 = arb.s57, arb.s58

TG = np.arange(-90.0, 90.0001, 0.5)
TS = sr.TS
NPT = sr.NPT
NPZ = sr.NPZ


def gt_pos(sdf, pts, nrm, tg=TG):
    q = pts[:, None, :] + tg[None, :, None] * nrm[:, None, :]
    v = ndimage.map_coordinates(sdf, [q[..., 0].ravel(), q[..., 1].ravel()],
                                order=1, mode="nearest").reshape(len(pts), len(tg))
    out = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        s = np.sign(v[i])
        idx = np.flatnonzero(s[:-1] * s[1:] < 0)
        if idx.size == 0:
            continue
        j = idx[np.argmin(np.abs(tg[idx]))]
        v0, v1 = v[i, j], v[i, j + 1]
        out[i] = tg[j] + (tg[j + 1] - tg[j]) * (-v0) / (v1 - v0) if v1 != v0 else tg[j]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=NPZ)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--taus", default="2,5,10,15,20,30")
    ap.add_argument("--out-dir", default=os.path.join(STEP9, "results"))
    ap.add_argument("--out", default="tail_ceiling_sq2resencw25.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    taus = [float(x) for x in args.taus.split(",")]

    train, val = arb.read_split(USS)
    names = train if args.split == "train" else val
    if args.limit:
        names = names[:args.limit]

    per_case = []
    for name in names:
        stem = arb.stem_of(name)
        gt = s57.load_gt(stem)
        arr = arb.load_prob(args.npz_dir, stem)
        if gt is None or arr is None:
            continue
        img = arb.load_image(stem)
        M0 = s57.largest_cc(arb.align_prob(arr, gt.shape) > 0.5)
        c = s57.contour_of(M0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, M0)
        ys, xs = np.nonzero(M0)
        cy, cx = ys.mean(), xs.mean()
        th = np.arctan2(pts[:, 0] - cy, pts[:, 1] - cx)
        r = np.hypot(pts[:, 0] - cy, pts[:, 1] - cx)
        o = np.argsort(th)
        th, r, pts, nrm = th[o], r[o], pts[o], nrm[o]

        t_gt = gt_pos(s58.sdf_of(gt), pts, nrm)
        imgf = ndimage.gaussian_filter(img, sr.SIGMA)
        G = arb.ridge_profile(imgf, pts, nrm, TS, 1.0)
        sup = np.clip(G[:, int(np.argmin(np.abs(TS)))] / np.maximum(G.max(axis=1), 1e-9), 0.05, 1.0)
        r_gt_ray, ok_ray = sr.ray_r(gt, cx, cy, th)

        row = {"name": stem, "n": len(r)}
        base = sr.rebuild(cx, cy, th, r, gt.shape)
        row["bdice_base"] = s57.bdice(base, gt, tol=1)
        hd, assd, _ = ev.surface_metrics(base, gt)
        row["hd95_base"], row["assd_base"] = hd, assd
        row["dice_base"] = s57.dice(base, gt)
        for tau in taus:
            sel = (np.abs(t_gt) > tau) & ok_ray
            rn = np.where(sel, r_gt_ray, r)
            m = sr.rebuild(cx, cy, th, rn, gt.shape)
            row["bdice_fixt%g" % tau] = s57.bdice(m, gt, tol=1)
            row["frac_fixt%g" % tau] = float(np.nanmean(sel))
            if tau in (10.0, 30.0):
                row["hd95_fixt%g" % tau] = ev.surface_metrics(m, gt)[0]
        # unsupported-gate oracle (as in 95) for reference
        unsup = sup < 0.25
        m = sr.rebuild(cx, cy, th, np.where(unsup & ok_ray, r_gt_ray, r), gt.shape)
        row["bdice_fixunsup"], row["frac_fixunsup"] = s57.bdice(m, gt, tol=1), float(unsup.mean())
        # full repair
        m = sr.rebuild(cx, cy, th, r_gt_ray, gt.shape)
        row["bdice_fixall"] = s57.bdice(m, gt, tol=1)
        row["hd95_fixall"], row["assd_fixall"] = ev.surface_metrics(m, gt)[:2]
        row["dice_fixall"] = s57.dice(m, gt)
        # is the GT boundary itself radial-smooth?
        for S in (1.5, 3.0, 6.0):
            rf, _ = sr.fit_periodic(th, r_gt_ray, np.ones_like(r), S * S * len(r))
            m = sr.rebuild(cx, cy, th, rf, gt.shape)
            row["bdice_gt_spl%g" % S] = s57.bdice(m, gt, tol=1)
        per_case.append(row)
        print("%-22s base %5.2f  fixt10 %5.2f (%.0f%%)  fixt30 %5.2f  fixall %5.2f  gt_spl1.5 %5.2f" %
              (stem, row["bdice_base"], row["bdice_fixt10"], 100 * row["frac_fixt10"],
               row["bdice_fixt30"], row["bdice_fixall"], row["bdice_gt_spl1.5"]))
        sys.stdout.flush()

    keys = [k[6:] for k in per_case[0] if k.startswith("bdice_")]
    acc = {}
    for k in keys:
        v = np.array([r["bdice_" + k] for r in per_case], float)
        dv = v - np.array([r["bdice_base"] for r in per_case], float)
        acc[k] = {"bdice1_mean": float(np.nanmean(v)), "delta": float(np.nanmean(dv)),
                  "win": int((dv > 0).sum()), "loss": int((dv < 0).sum())}
        hk = "hd95_" + k
        if hk in per_case[0]:
            hv = np.array([r[hk] for r in per_case], float)
            acc[k]["hd95_mean"] = float(np.nanmean(hv))
        fk = "frac_" + k
        if fk in per_case[0]:
            acc[k]["frac_repaired"] = float(np.mean([r[fk] for r in per_case]))
    out = {"split": args.split, "n_cases": len(per_case), "taus": taus,
           "per_case": per_case, "summary": acc}
    with open(os.path.join(args.out_dir, args.out), "w") as fh:
        json.dump(out, fh, indent=1)

    print("=" * 90)
    print("TAIL CEILING  split=%s n=%d   (repair radius with GT radius at selected points)" %
          (args.split, len(per_case)))
    for k, d in sorted(acc.items(), key=lambda kv: -kv[1]["bdice1_mean"]):
        fr = ("  frac %.3f" % d["frac_repaired"]) if "frac_repaired" in d else ""
        hd = ("  HD95 %.2f" % d["hd95_mean"]) if "hd95_mean" in d else ""
        print("  %-14s %7.3f  d%+7.3f  win/loss %2d/%2d%s%s" %
              (k, d["bdice1_mean"], d["delta"], d["win"], d["loss"], fr, hd))


if __name__ == "__main__":
    main()
