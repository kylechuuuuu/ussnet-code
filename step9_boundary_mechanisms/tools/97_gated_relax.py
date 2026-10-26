#!/usr/bin/env python3
"""97_gated_relax.py -- mechanism 2, done right: replace the radius ONLY where the
image gives no evidence, keep every supported point exactly as the model had it.

95 fitted a spline to the model radius and substituted it EVERYWHERE, so it also
smoothed the 78% of points that were already right (best variant -0.4). 96 then
showed the repair is worth +10.2 (unsupported gate) / +11.3 (|t_gt|>10, 14.3% of
points) / +23.0 (|t_gt|>5).  So the substitution must be gated.

Variants
  gate_p{p}_S{s}   r <- spline_fit(r, weights=sup**p)   at unsupported points only
  gshr_p{p}_S{s}   ... and only where that shrinks (dominant error is overshoot)
  ora_p{p}_S{s}    same machinery but gated on the ORACLE  |t_gt|>10   (ceiling)
  gtfit_S{s}       fill unsupported points with a spline through the GT radius at
                   the SUPPORTED points -> "can the arc be inferred from its
                   supported neighbours at all?"  the real ceiling of a shape prior
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
    ap.add_argument("--scales", default="0.75,1.5,3.0")
    ap.add_argument("--powers", default="1,3")
    ap.add_argument("--gate", type=float, default=0.25)
    ap.add_argument("--tail", type=float, default=10.0)
    ap.add_argument("--out-dir", default=os.path.join(STEP9, "results"))
    ap.add_argument("--out", default="gated_relax_sq2resencw25.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scales = [float(x) for x in args.scales.split(",")]
    powers = [float(x) for x in args.powers.split(",")]

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
        unsup = sup < args.gate
        tail = np.abs(t_gt) > args.tail
        r_gt_ray, ok_ray = sr.ray_r(gt, cx, cy, th)

        row = {"name": stem, "frac_unsup": float(unsup.mean()), "frac_tail": float(tail.mean())}
        base = sr.rebuild(cx, cy, th, r, gt.shape)
        row["bdice_base"] = s57.bdice(base, gt, tol=1)
        row["hd95_base"], row["assd_base"] = ev.surface_metrics(base, gt)[:2]

        for S in scales:
            for pw in powers:
                w = sup ** pw
                rf_img, _ = sr.fit_periodic(th, r, w, S * S * len(r))
                for gname, gate in (("gate", unsup), ("ora", tail)):
                    rn = np.where(gate, rf_img, r)
                    row["bdice_%s_p%g_S%g" % (gname, pw, S)] = s57.bdice(sr.rebuild(cx, cy, th, rn, gt.shape), gt, tol=1)
                    if gname == "gate":
                        rs = np.where(gate, np.minimum(r, rf_img), r)
                        row["bdice_gshr_p%g_S%g" % (pw, S)] = s57.bdice(sr.rebuild(cx, cy, th, rs, gt.shape), gt, tol=1)
                # key variant: shrink-only at the oracle tail (overshoot is dominant)
                rs = np.where(tail, np.minimum(r, rf_img), r)
                row["bdice_orashr_p%g_S%g" % (pw, S)] = s57.bdice(sr.rebuild(cx, cy, th, rs, gt.shape), gt, tol=1)
            # real ceiling: infer the unsupported arc from the GT's SUPPORTED part
            rf_gt, _ = sr.fit_periodic(th, r_gt_ray, np.ones_like(r), S * S * len(r))
            rr = np.where(unsup & ok_ray, rf_gt, r)
            row["bdice_gtfit_S%g" % S] = s57.bdice(sr.rebuild(cx, cy, th, rr, gt.shape), gt, tol=1)
            rf_gt2, _ = sr.fit_periodic(th, r_gt_ray, sup, S * S * len(r))
            rr2 = np.where(unsup & ok_ray, rf_gt2, r)
            row["bdice_gtfitw_S%g" % S] = s57.bdice(sr.rebuild(cx, cy, th, rr2, gt.shape), gt, tol=1)
        per_case.append(row)
        print("%-22s base %5.2f  unsup %.2f tail %.2f  gate_p1_S1.5 %5.2f  gtfit_S1.5 %5.2f" %
              (stem, row["bdice_base"], row["frac_unsup"], row["frac_tail"],
               row.get("bdice_gate_p1_S1.5", np.nan), row["bdice_gtfit_S1.5"]))
        sys.stdout.flush()

    keys = [k[6:] for k in per_case[0] if k.startswith("bdice_")]
    acc = {}
    for k in keys:
        v = np.array([r["bdice_" + k] for r in per_case], float)
        dv = v - np.array([r["bdice_base"] for r in per_case], float)
        acc[k] = {"bdice1_mean": float(np.nanmean(v)), "delta": float(np.nanmean(dv)),
                  "win": int((dv > 0).sum()), "loss": int((dv < 0).sum())}
    out = {"split": args.split, "n_cases": len(per_case), "gate": args.gate, "tail": args.tail,
           "per_case": per_case, "summary": acc}
    with open(os.path.join(args.out_dir, args.out), "w") as fh:
        json.dump(out, fh, indent=1)
    print("=" * 88)
    print("GATED RELAX  split=%s n=%d  (base %.3f = radial rebuild of the base prediction)" %
          (args.split, len(per_case), acc["base"]["bdice1_mean"]))
    for k, d in sorted(acc.items(), key=lambda kv: -kv[1]["bdice1_mean"])[:24]:
        print("  %-20s %7.3f  d%+7.3f  win/loss %2d/%2d" %
              (k, d["bdice1_mean"], d["delta"], d["win"], d["loss"]))


if __name__ == "__main__":
    main()
