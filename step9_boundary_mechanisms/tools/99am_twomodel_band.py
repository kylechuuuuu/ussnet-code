#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99am_twomodel_band.py

"Two models, one inner contour one outer contour" -- does a sandwich built from
TWO INDEPENDENT MODELS carry information that the three tied heads do not?

The three sandwich heads are one function plus a constant logit offset, so their
interval is a level-set offset (step7/73, step9/92, 99aa).  Two separately
trained models are a genuinely different construction: their boundary errors can
decorrelate, and then [min(t_A), max(t_B)] along the normal is a real
uncertainty interval.

Everything is measured exactly like 99aa: same 400 normals of the same
in-domain contour of the same case, same GT locator, same band_statistic
(coverage of t_gt vs a CONSTANT band of the same mean width).  Rows are
therefore directly comparable with 99aa's view_* rows and with arch3h.

Per pair it also reports the point-wise error correlation, which is the number
that decides whether the construction can work at all:
  r_AB       Pearson r between t_A and t_B over the shared normals
  r_signed   Pearson r of the SIGNED errors with the GT (t_A - t_gt, t_B - t_gt)
  same_dir   fraction of points where both models are on the same side of the GT
  rho_w_err  Spearman(band width, |t_gt|) pooled -- does the band widen where
             the error is large?  (the property the sandwich is supposed to have)

usage: python tools/99am_twomodel_band.py [--split val] [--npt 400] [--limit 0]
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np

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
from scipy import stats as sps                            # noqa: E402

sys.path.insert(0, os.path.join(STEP9, "tools"))
_a = importlib.import_module("99aa_viewspread_band")      # noqa: E402
crossings, gt_position, band_stats = _a.crossings, _a.gt_position, _a.band_stats
TS, EDGE = _a.TS, _a.EDGE

# model label -> (probability-dir template, is it a three-head arm)
MODELS = {
    "sq2":   ("sq2shift_%s_base_nifti", False),   # ResEnc + sandwich 500ep (shipped)
    "fpB":   ("fpB_%s_nifti", False),             # ResEnc, NO sandwich 500ep
    "fpA":   ("fpA_%s_nifti", False),             # stock, NO sandwich 500ep
    "sq2a":  ("sq2a_%s_base_nifti", False),       # stock + sandwich 500ep
    "stock": ("stockshift_%s_nifti", False),      # stock 1000ep
}
PAIRS = [("sq2", "stock"), ("sq2", "fpB"), ("sq2", "fpA"),
         ("sq2", "sq2a"), ("stock", "fpA"), ("stock", "sq2a")]


def load(model, cond, stem, shape):
    tpl = MODELS[model][0]
    r = arb.load_prob(os.path.join(STEP8P, tpl % cond), stem)
    return None if r is None else arb.align_prob(r, shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--npt", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(STEP9, "results", "twomodel_band_val.json"))
    args = ap.parse_args()

    _, val_names = read_split(USS)
    names = (val_names if args.split == "val" else read_split(USS)[0])
    if args.limit:
        names = names[:args.limit]
    stems = [os.path.splitext(n)[0] for n in names]
    print("[99am] split=%s n=%d npt=%d" % (args.split, len(names), args.npt))

    acc = {}
    corr = {}
    per_case = []

    def push(row, st):
        d = acc.setdefault(row, {"cov": [], "const": [], "w": [], "mo": [], "mi": [], "tgt": []})
        d["cov"].append(st["coverage"]); d["const"].append(st["const_same_width"])
        d["w"].append(st["mean_width"]); d["mo"].append(st["miss_out"])
        d["mi"].append(st["miss_in"]); d["tgt"].append(st["mean_abs_tgt"])

    def pushc(row, key, v):
        corr.setdefault(row, {}).setdefault(key, []).append(v)

    for n, stem in zip(names, stems):
        gt = s57.load_gt(stem)
        pb = load("sq2", "in_domain", stem, gt.shape)
        if pb is None:
            print("[99am] skip %s" % stem); continue
        M0 = s57.largest_cc(pb > 0.5)
        pts = s57.resample_closed(s57.contour_of(M0), args.npt, 0)
        nrm = s58.normals_of(pts, M0)
        t_gt = gt_position(s58.sdf_of(gt), pts, nrm)
        okc = ~np.isnan(t_gt)
        if okc.sum() < args.npt // 2:
            print("[99am] skip %s (only %d normals reach GT)" % (stem, okc.sum())); continue

        stack, have = [], []
        for m in MODELS:
            q = load(m, "in_domain", stem, gt.shape)
            if q is None:
                print("[99am] skip %s (missing %s)" % (stem, m)); have = None; break
            stack.append(q); have.append(m)
        if have is None:
            continue
        t_all = crossings(np.stack(stack, axis=0), pts, nrm)     # (M, P)
        idx = {m: i for i, m in enumerate(have)}
        tgt = t_gt[okc]

        # reference: the architectural three-head interval, same points
        rb = arb.load_prob(os.path.join(STEP8P, "sq23h_in_domain_base_nifti"), stem)
        rc = arb.load_prob(os.path.join(STEP8P, "sq23h_in_domain_cautious_nifti"), stem)
        ra = arb.load_prob(os.path.join(STEP8P, "sq23h_in_domain_amplified_nifti"), stem)
        if rb is not None and rc is not None and ra is not None:
            Min = M0 & (arb.align_prob(rc, gt.shape) > 0.5)
            Mout = M0 | (arb.align_prob(ra, gt.shape) > 0.5)
            ti, to = arb.limits(pts, nrm, Min, Mout)
            lo3 = -np.clip(ti, 0.0, float(arb.OFFS[-1]))[okc]
            hi3 = np.clip(to, 0.0, float(arb.OFFS[-1]))[okc]
            st = band_stats(lo3, hi3, tgt)
            st["nan_frac"] = float((~okc).mean())
            st["rho_w_err"] = float(sps.spearmanr(hi3 - lo3, np.abs(tgt)).statistic)
            push("arch3h", st)
            pushc("arch3h", "rho_w_err", st["rho_w_err"])
            per_case.append({"name": stem, "row": "arch3h", **st})

        for A, B in PAIRS:
            ta = t_all[idx[A]][okc]; tb = t_all[idx[B]][okc]
            good = ~np.isnan(ta) & ~np.isnan(tb)
            if good.sum() < 50:
                continue
            row = "pair_%s_%s" % (A, B)
            st = band_stats(ta[good], tb[good], tgt[good])
            w = np.abs(ta[good] - tb[good])
            st["r_AB"] = float(np.corrcoef(ta[good], tb[good])[0, 1])
            ea, eb = ta[good] - tgt[good], tb[good] - tgt[good]
            st["r_err"] = float(np.corrcoef(ea, eb)[0, 1])
            st["same_dir"] = float((np.sign(ea) == np.sign(eb)).mean())
            st["both_gt2px"] = float(((np.abs(ea) > 2) & (np.abs(eb) > 2)).mean())
            st["rho_w_err"] = float(sps.spearmanr(w, np.abs(tgt[good])).statistic)
            st["nan_frac"] = float((~okc).mean())
            push(row, st)
            pushc(row, "r_AB", st["r_AB"]); pushc(row, "r_err", st["r_err"])
            pushc(row, "rho_w_err", st["rho_w_err"])
            per_case.append({"name": stem, "row": row, **st})

    print("\n  %-20s %7s %7s %8s %7s %7s %7s %7s %8s" %
          ("row", "cover", "const", "delta", "width", "miss_o", "miss_i", "|t_gt|", "rho(w,e)"))
    out = {}
    for row, d in sorted(acc.items()):
        cov, con, w = np.mean(d["cov"]), np.mean(d["const"]), np.mean(d["w"])
        rec = {"coverage": cov, "const_same_width": con, "delta": cov - con,
               "mean_width": w, "miss_out": np.mean(d["mo"]), "miss_in": np.mean(d["mi"]),
               "mean_abs_tgt": np.mean(d["tgt"])}
        if row in corr:
            rec.update({k: float(np.mean(v)) for k, v in corr[row].items()})
        out[row] = rec
        print("  %-20s %7.4f %7.4f %+8.4f %7.2f %7.4f %7.4f %7.3f %8s" %
              (row, rec["coverage"], rec["const_same_width"], rec["delta"], rec["mean_width"],
               rec["miss_out"], rec["miss_in"], rec["mean_abs_tgt"],
               ("%+.3f" % rec["rho_w_err"]) if "rho_w_err" in rec else "-"))

    print("\n  per-case point-wise error correlation (mean over 31 cases):")
    for row in sorted(corr):
        g = lambda k: np.mean(corr[row][k]) if k in corr[row] else float("nan")
        print("    %-20s r_AB %+.3f   r_err %+.3f   rho(w,|err|) %+.3f" %
              (row, g("r_AB"), g("r_err"), g("rho_w_err")))
        out[row].update({k + "_percase": float(np.mean(v)) for k, v in corr[row].items()})

    json.dump({"rows": out, "per_case": per_case}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n[99am] wrote %s" % args.out)


if __name__ == "__main__":
    main()
