#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99aa_viewspread_band.py

"Sandwich by measurement" vs "sandwich by architecture" -- on identical
geometry, identical cases, identical code path.

step7/73 and step9/92 already answered the architectural question: the interval
produced by the three tied heads covers the GT boundary on ~0.67-0.70 of the
normal samples, while an EQUAL-AREA ISOTROPIC band covers 0.87-0.94; on 31/31
cases the blob wins.  Reason (step6 trainer docstring): the three heads are one
function plus a constant logit offset, so the "band" is a level-set offset, not
an uncertainty estimate.

This tool asks the one question that has never been asked:

    if the interval comes from REAL input perturbations instead of a bias offset
    on a shared feature map -- t_v = the 0.5-crossing of the model's own
    probability field along the normal, for each appearance view v of step8's
    pre-registered shift bank -- does [min_v t_v, max_v t_v] bracket the GT
    better than a constant band of the SAME mean width?

Every row is measured on the same 400 normals of the same in-domain contour, at
the same points, with step7/73's own GT locator, so the rows are directly
comparable:

  arch3h            three tied heads (cautious/base/amplified) = step7's interval
  view_17_edge      spread over all 17 conditions; a view with NO crossing in
                    +/-30 px counts as "the boundary is >=30 px away" (edge)
  view_17_near      same, but a missing view is dropped (>=2 views required)
  view_art3         spread over the shipped recipe's group (in_domain + jpegs)
  view_nonlin4      gamma_060/080/125 + in_domain (augmentation-covered axis)
  view_speckle3     speckle_15/35 + in_domain
  view_aff3         step8's "affine" group (z-score absorbed) = NEGATIVE CONTROL

Per row: coverage of t_gt, coverage of a constant band with the same mean width
(isolates angular variation from mean width), miss split (inward/outward), and
the fraction of normals where at least one pre-registered view has no nearby
crossing (a GT-free out-of-distribution indicator).

usage:
  python tools/99aa_viewspread_band.py [--split val] [--npt 400] [--limit 0]
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage

USS = "/media/72TB_disk/personal/zk/USSnet"
STEP7 = os.path.join(USS, "step7_band_arbitration")
STEP6 = os.path.join(USS, "step6_sandwich_squeeze")
STEP8P = os.path.join(USS, "step8_domain_shift", "predictions")
STEP9 = os.path.join(USS, "step9_boundary_mechanisms")

sys.path.insert(0, os.path.join(STEP7, "tools"))
sys.path.insert(0, os.path.join(STEP6, "tools"))
sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))

arb = importlib.import_module("63_arbitrate")            # noqa: E402
s57, s58 = arb.s57, arb.s58                              # noqa: E402
read_split = arb.read_split                              # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sh", os.path.join(USS, "step8_domain_shift", "tools", "shift_lib.py"))
sh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sh)

ALL = list(sh.SHIFTS.keys())                             # 17 pre-registered
ART = ["in_domain"] + list(sh.GROUPS["artifact"])
NONLIN = ["in_domain"] + list(sh.GROUPS["nonlinear"])
SPECK = ["in_domain"] + list(sh.GROUPS["speckle"])
AFF = ["in_domain"] + list(sh.GROUPS["affine"])
EDGE = 30.0
TS = np.arange(-EDGE, EDGE + 0.0001, 0.25)


def crossings(pstack, pts, nrm, ts=TS):
    """0.5-crossing position nearest to 0 for every view, along every normal.

    pstack: (V, H, W) probability maps of one case, aligned to the GT shape.
    returns (V, P) signed positions; nan = no crossing inside the window.
    """
    V = pstack.shape[0]
    P = pts.shape[0]
    y = pts[:, 0][:, None] + nrm[:, 0][:, None] * ts[None, :]
    x = pts[:, 1][:, None] + nrm[:, 1][:, None] * ts[None, :]
    vi = np.repeat(np.arange(V), P * ts.size)
    yy = np.tile(np.broadcast_to(y[None], (V, P, ts.size)).ravel(), 1)
    xx = np.tile(np.broadcast_to(x[None], (V, P, ts.size)).ravel(), 1)
    coords = np.stack([vi, yy, xx], axis=0)
    vals = ndimage.map_coordinates(pstack, coords, order=1,
                                   mode="nearest").reshape(V, P, ts.size) - 0.5
    sgn = np.sign(vals)
    cross = (sgn[:, :, :-1] * sgn[:, :, 1:]) <= 0
    out = np.full((V, P), np.nan)
    for v in range(V):
        for i in range(P):
            js = np.flatnonzero(cross[v, i])
            if js.size == 0:
                continue
            j = js[np.argmin(np.abs(ts[js]))]
            v0, v1 = vals[v, i, j], vals[v, i, j + 1]
            f = 0.0 if abs(v1 - v0) < 1e-9 else v0 / (v0 - v1)
            out[v, i] = ts[j] + f * (ts[j + 1] - ts[j])
    return out


def gt_position(sdf, pts, nrm, ts=TS):
    """step7/73's GT-along-normal locator: nearest crossing of the GT SDF."""
    y = pts[:, 0][:, None] + nrm[:, 0][:, None] * ts[None, :]
    x = pts[:, 1][:, None] + nrm[:, 1][:, None] * ts[None, :]
    vals = ndimage.map_coordinates(sdf, np.stack([y.ravel(), x.ravel()], 0),
                                   order=1, mode="nearest").reshape(pts.shape[0], ts.size)
    sgn = np.sign(vals)
    cross = (sgn[:, :-1] * sgn[:, 1:]) <= 0
    out = np.full(pts.shape[0], np.nan)
    for i in range(pts.shape[0]):
        js = np.flatnonzero(cross[i])
        if js.size == 0:
            continue
        j = js[np.argmin(np.abs(ts[js]))]
        v0, v1 = vals[i, j], vals[i, j + 1]
        f = 0.0 if abs(v1 - v0) < 1e-9 else v0 / (v0 - v1)
        out[i] = ts[j] + f * (ts[j + 1] - ts[j])
    return out


def band_stats(lo, hi, t_gt):
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    w = hi - lo
    half = float(np.mean(w)) / 2.0
    return {"coverage": float(((t_gt >= lo) & (t_gt <= hi)).mean()),
            "const_same_width": float((np.abs(t_gt) <= half).mean()),
            "r_iso": half, "mean_width": float(np.mean(w)),
            "miss_out": float((t_gt > hi).mean()),
            "miss_in": float((t_gt < lo).mean()),
            "mean_abs_tgt": float(np.mean(np.abs(t_gt)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--npt", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(STEP9, "results", "viewspread_band_val.json"))
    args = ap.parse_args()

    train_names, val_names = read_split(USS)
    names = train_names if args.split == "train" else val_names
    if args.limit:
        names = names[:args.limit]
    stems = [os.path.splitext(n)[0] for n in names]
    print("[99aa] split=%s n=%d npt=%d views=%d" % (args.split, len(names), args.npt, len(ALL)))

    acc, per_case = {}, []

    def push(row, st):
        d = acc.setdefault(row, {"cov": [], "const": [], "w": [], "mo": [], "mi": [],
                                 "tgt": [], "nanfrac": []})
        d["cov"].append(st["coverage"]); d["const"].append(st["const_same_width"])
        d["w"].append(st["mean_width"]); d["mo"].append(st["miss_out"])
        d["mi"].append(st["miss_in"]); d["tgt"].append(st["mean_abs_tgt"])

    for k, (n, stem) in enumerate(zip(names, stems)):
        gt = s57.load_gt(stem)
        base = arb.load_prob(os.path.join(STEP8P, "sq2shift_in_domain_base_nifti"), stem)
        if base is None:
            print("[99aa] skip %s (no in_domain npz)" % stem)
            continue
        pb = arb.align_prob(base, gt.shape)
        M0 = s57.largest_cc(pb > 0.5)
        pts = s57.resample_closed(s57.contour_of(M0), args.npt, 0)
        nrm = s58.normals_of(pts, M0)
        t_gt = gt_position(s58.sdf_of(gt), pts, nrm)
        okc = ~np.isnan(t_gt)                       # the point set EVERY row uses
        if okc.sum() < args.npt // 2:
            print("[99aa] skip %s (only %d/%d normals reach the GT)" % (stem, okc.sum(), args.npt))
            continue

        P = {}
        for c in ALL:
            r = arb.load_prob(os.path.join(STEP8P, "sq2shift_%s_base_nifti" % c), stem)
            P[c] = None if r is None else arb.align_prob(r, gt.shape)
        if any(v is None for v in P.values()):
            print("[99aa] skip %s (missing views)" % stem)
            continue
        stack = np.stack([P[c] for c in ALL], axis=0)
        t_v = crossings(stack, pts, nrm)            # (V, P), nan = collapsed view
        tgt = t_gt[okc]

        def build(conds, mode):
            sub = t_v[[ALL.index(c) for c in conds]][:, okc]
            valid = ~np.isnan(sub)
            nvalid = valid.sum(axis=0)
            if mode == "edge":
                s = np.where(valid, sub, EDGE * np.sign(sub[np.arange(sub.shape[1])[
                    np.argmax(valid, axis=0)], np.arange(sub.shape[1])] + 1e-12))
                lo, hi = np.nanmin(s, axis=0), np.nanmax(s, axis=0)
            else:
                keep = nvalid >= 2
                lo = np.array([np.nanmin(sub[:, i]) if keep[i] else np.nan
                               for i in range(sub.shape[1])])
                hi = np.array([np.nanmax(sub[:, i]) if keep[i] else np.nan
                               for i in range(sub.shape[1])])
            good = ~np.isnan(lo) & ~np.isnan(hi)
            return band_stats(lo[good], hi[good], tgt[good]), float((nvalid < len(conds)).mean())

        this = {"name": n}
        for label, conds, mode in (("view_17_edge", ALL, "edge"),
                                   ("view_17_near", ALL, "near"),
                                   ("view_art3", ART, "near"),
                                   ("view_nonlin4", NONLIN, "near"),
                                   ("view_speckle3", SPECK, "near"),
                                   ("view_aff3", AFF, "near")):
            st, nanfrac = build(conds, mode)
            st["nan_frac"] = nanfrac
            push(label, st)
            this[label] = st

        # the architectural interval, same normals, same point set
        rb = arb.load_prob(os.path.join(STEP8P, "sq23h_in_domain_base_nifti"), stem)
        rc = arb.load_prob(os.path.join(STEP8P, "sq23h_in_domain_cautious_nifti"), stem)
        ra = arb.load_prob(os.path.join(STEP8P, "sq23h_in_domain_amplified_nifti"), stem)
        if rb is not None and rc is not None and ra is not None:
            Min = M0 & (arb.align_prob(rc, gt.shape) > 0.5)
            Mout = M0 | (arb.align_prob(ra, gt.shape) > 0.5)
            t_in_mag, t_out_mag = arb.limits(pts, nrm, Min, Mout)
            tin = -np.clip(t_in_mag, 0.0, float(arb.OFFS[-1]))
            tout = np.clip(t_out_mag, 0.0, float(arb.OFFS[-1]))
            st = band_stats(tin[okc], tout[okc], tgt)
            st["r_iso"] = float((Mout & ~Min).sum()) / (2.0 * float(max(1, s57.surface(M0).sum())))
            st["mean_width"] = 2.0 * st["r_iso"]
            st["nan_frac"] = float((~okc).mean())
            push("arch3h", st)
            this["arch3h"] = st
        per_case.append(this)
        if (k + 1) % 5 == 0:
            print("[99aa] %d/%d" % (k + 1, len(names)))

    if not acc:
        raise SystemExit("nothing processed")
    order = ["arch3h", "view_17_edge", "view_17_near", "view_art3", "view_nonlin4",
             "view_speckle3", "view_aff3"]
    print("\n[99aa] cases=%d   coverage over %d normals/case" % (len(per_case), args.npt))
    print("[99aa] %-14s %8s %8s %9s %8s %8s %8s %8s"
          % ("row", "cover", "const_w", "d_cover", "width", "|t_gt|", "miss_out", "miss_in"))
    out = {"split": args.split, "n": len(per_case), "npt": args.npt, "rows": {}}
    for r in order:
        if r not in acc:
            continue
        d = acc[r]
        C, K = np.array(d["cov"]), np.array(d["const"])
        o = {"coverage": float(C.mean()), "const_same_width": float(K.mean()),
             "delta_vs_const": float((C - K).mean()),
             "delta_win": int((C > K).sum()), "delta_loss": int((C < K).sum()),
             "mean_width": float(np.mean(d["w"])), "mean_abs_tgt": float(np.mean(d["tgt"])),
             "miss_out": float(np.mean(d["mo"])), "miss_in": float(np.mean(d["mi"]))}
        out["rows"][r] = o
        print("[99aa] %-14s %8.4f %8.4f %+9.4f %8.2f %8.2f %8.4f %8.4f"
              % (r, o["coverage"], o["const_same_width"], o["delta_vs_const"],
                 o["mean_width"], o["mean_abs_tgt"], o["miss_out"], o["miss_in"]))
    out["per_case"] = per_case

    for r in ("view_17_edge", "view_17_near"):
        if r in acc:
            C = np.array(acc[r]["cov"])
            if "arch3h" in acc:
                A = np.array(acc["arch3h"]["cov"])
                print("\n[99aa] per-case %s vs arch3h: mean %+.4f  win %d / loss %d"
                      % (r, (C - A).mean(), int((C > A).sum()), int((C < A).sum())))
                try:
                    from scipy.stats import wilcoxon
                    nz = (C - A)[np.abs(C - A) > 1e-9]
                    if len(nz) >= 5:
                        print("[99aa]   Wilcoxon p = %.4f" % float(wilcoxon(nz).pvalue))
                except Exception:
                    pass
    # GT-free OOD indicator: does "a pre-registered view has no nearby crossing"
    # identify the cases that collapse under compound_B?
    if per_case:
        nf = np.array([c["view_17_near"]["nan_frac"] for c in per_case])
        print("\n[99aa] normals with >=1 missing view: mean %.4f  max %.4f"
              % (nf.mean(), nf.max()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[99aa] wrote %s" % args.out)


if __name__ == "__main__":
    main()
