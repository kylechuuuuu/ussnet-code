#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_band_arbitration/tools/70_dp_arbitrate.py

MECHANISM 1, deepened.  The squeeze interval becomes a HARD TRUST REGION and the
boundary inside it is DETERMINED by a globally optimal dynamic program.

Two independent changes with respect to step6 tools/63_arbitrate.py:

  (A) CANDIDATE SET -- crest instead of flank (`--cand crest`).
      step6 snaps a contour point to the *nearest* offset whose image-ridge
      response |dI/dt| merely exceeds frac * max_t |dI/dt|.  That sample sits on
      the FLANK of the edge response, not at its crest, so the executed move is
      systematically smaller than the true displacement.  Physically the
      boundary is at the crest, so the candidate set can be restricted to local
      maxima of the ridge profile.  A fallback to the step6 gate keeps the rule
      from starving where a crest is unavailable.

  (B) DECISION RULE -- global instead of per-point.
      The whole closed contour is decided at once as the exact minimiser of

        min_{t_0..t_{N-1}}  sum_i C_i(t_i) + lam * sum_i |t_i - t_{i+1}|  (cyclic)

      with the squeeze interval [lo_i, hi_i] as a HARD trust region (offsets
      outside it are not candidates).  Two cost models are available:

        gate cost   C_i(t) = |t|   if t is a candidate
                    C_i(0) = PEN   if the point has a candidate but prefers 0
                    C_i(0) = 0     if the point has no candidate
                    C_i(t) = inf   otherwise
        reward      C_i(t) = -R_i(t) + gamma*(t/sigma_p)^2   (elastic snake)

      solved exactly by cyclic Viterbi over all start states (state = offset
      index, transition price = lam*|dt|).  For the gate cost, lam = 0 makes the
      DP separable and it then reproduces step6's greedy rule EXACTLY -- this is
      verified numerically at run time (`dp_g0_lrn`), the control that isolates
      "global consistency" from "different evidence".

lam is the one new degree of freedom: the price of a jump between neighbouring
contour samples (2.4 px apart at NPT=400 on a ~1000 px perimeter).  `--lam-list`
sweeps it in a single pass so the sensitivity is visible rather than hidden.

Variants per case (all share cc1(base): topology cleanup is never credited to
the arbitration):

    raw, free, band_lrn, band_thr, veto_lrn     step6 reproduction (baselines)
    dp_g0_lrn / dp_c0_lrn                       lam=0 controls
    dp_g_* / dp_c_*                             gate cost, gate / crest candidates
    dp_r_* / dp_rb_*                            elastic-snake reward cost
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
STEP7 = os.path.dirname(HERE)
USS = os.path.dirname(STEP7)
STEP6_TOOLS = os.path.join(USS, "step6_sandwich_squeeze", "tools")

sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import read_split                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "arb63", os.path.join(STEP6_TOOLS, "63_arbitrate.py"))
arb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arb)

s57 = arb.s57
s58 = arb.s58

TSF = np.arange(-16.0, 16.0001, 0.5)      # step6's grid: evidence + baseline
NPT = 400
BIG = 1e12

# set in main(): the DP candidate grid (a sub-window of TSF)
TS = None
SEL = None
K0 = None


def set_grid(win):
    global TS, SEL, K0
    SEL = np.abs(TSF) <= win + 1e-9
    TS = TSF[SEL]
    K0 = int(np.argmin(np.abs(TS)))


# ------------------------------------------------------------------ evidence --
def crest_mask(G):
    """Local maxima of the ridge response along t (3-point strict maximum)."""
    left = np.roll(G, 1, axis=1)
    right = np.roll(G, -1, axis=1)
    left[:, 0] = -1.0
    right[:, -1] = -1.0
    return (G >= left) & (G > right)


def _R(G, mode):
    rmax = G.max(axis=1, keepdims=True) + 1e-9
    Rrel = G / rmax
    if mode == "rel":
        return Rrel
    Rabs = G / (np.percentile(G, 90) + 1e-9)
    if mode == "abs":
        return Rabs
    return 0.5 * Rrel + 0.5 * Rabs


def evidence(G, frac, cand, mode="rel"):
    """Boolean candidate mask on the DP grid TS."""
    gated = _R(G, mode) >= frac
    if cand == "crest":
        cr = crest_mask(G) & gated
        use = np.where(cr.any(axis=1, keepdims=True), cr, gated)
    else:
        use = gated
    return use[:, SEL]


def cost_map_gate(cand_mask, lo, hi, pen):
    inwin = (TS[None, :] >= lo[:, None] - 1e-9) & (TS[None, :] <= hi[:, None] + 1e-9)
    cand = cand_mask & inwin
    any_c = cand.any(axis=1)
    C = np.where(cand, np.abs(TS)[None, :], BIG)
    C[:, K0] = np.where(any_c, pen, 0.0)
    C[:, K0] = np.where(cand[:, K0], 0.0, C[:, K0])
    return np.ascontiguousarray(C)


def cost_map_reward(G, frac, gamma, sig_p, mode="rel"):
    Rg = _R(G, mode) * (_R(G, mode) >= frac)
    return -(Rg[:, SEL] - gamma * (TS[None, :] / max(sig_p, 1e-6)) ** 2)


# ----------------------------------------------------------------------- DP --
def dp_solve(C, lam):
    """Exact cyclic Viterbi over all K start states (minimisation)."""
    N, K = C.shape
    cost = lam * np.abs(TS[None, :] - TS[:, None])       # cost[prev, cur]
    dp = np.full((K, K), BIG)
    dp[np.arange(K), np.arange(K)] = C[0]
    bk = np.zeros((N, K, K), dtype=np.int16)
    for i in range(1, N):
        tot = dp[:, :, None] + cost[None, :, :]          # (s, prev, cur)
        k = np.argmin(tot, axis=1)                       # (s, cur) best prev
        bk[i] = k.astype(np.int16)
        dp = np.take_along_axis(tot, k[:, None, :], axis=1)[:, 0, :] + C[i][None, :]
    close = dp + cost
    s_best, j_best = np.unravel_index(np.argmin(close), close.shape)
    t = np.zeros(N)
    j = int(j_best)
    for i in range(N - 1, 0, -1):
        t[i] = TS[j]
        j = int(bk[i, s_best, j])
    t[0] = TS[int(s_best)]
    return t


def solve_cand(G, lo, hi, args, cand, mode="rel", lam=None):
    C = cost_map_gate(evidence(G, args.frac, cand, mode), lo, hi, args.pen)
    t = dp_solve(C, args.lam if lam is None else lam)
    j = np.abs(t[:, None] - TS[None, :]).argmin(axis=1)
    bad = C[np.arange(len(t)), j] >= BIG / 2
    t[bad] = 0.0
    return np.clip(t, lo, hi)


def solve_reward(G, lo, hi, args, mode="rel", lam=None):
    C = cost_map_reward(G, args.frac, args.gamma, args.sigma_p, mode)
    inwin = (TS[None, :] >= lo[:, None] - 1e-9) & (TS[None, :] <= hi[:, None] + 1e-9)
    t = dp_solve(np.where(inwin, C, BIG), args.lam if lam is None else lam)
    j = np.abs(t[:, None] - TS[None, :]).argmin(axis=1)
    bad = ~inwin[np.arange(len(t)), j]
    t[bad] = 0.0
    return np.clip(t, lo, hi)


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-base-dir", default=None)
    ap.add_argument("--npz-cautious-dir", default=None)
    ap.add_argument("--npz-amplified-dir", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="dp")
    ap.add_argument("--results-dir", default=os.path.join(STEP7, "results"))
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--maxstep", type=float, default=4.0)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--dp-win", type=float, default=6.0)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--lam-list", default=None)
    ap.add_argument("--pen", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.30)
    ap.add_argument("--sigma-p", type=float, default=4.0)
    ap.add_argument("--thr-hi", type=float, default=0.85)
    ap.add_argument("--thr-lo", type=float, default=0.15)
    ap.add_argument("--scale-lrn", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-dp", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    set_grid(args.dp_win)

    train_names, val_names = read_split(USS)
    names = train_names if args.split == "train" else val_names
    if args.limit:
        names = names[:args.limit]
    lams = [float(x) for x in args.lam_list.split(",")] if args.lam_list else [args.lam]
    multi = len(lams) > 1

    rows = []
    t00 = time.time()
    ctrl_g = []
    for k, name in enumerate(names):
        stem = arb.stem_of(name)
        gt = s57.load_gt(stem)
        if gt is None:
            continue
        try:
            img = arb.load_image(stem)
        except Exception:
            continue
        if img.shape != gt.shape:
            continue

        p_b = arb.align_prob(arb.load_prob(args.npz_base_dir, stem), gt.shape) \
            if args.npz_base_dir else None
        if p_b is None:
            continue
        M0 = s57.largest_cc(p_b > 0.5)
        if not M0.any():
            continue
        Min = Mout = None
        if args.npz_cautious_dir and args.npz_amplified_dir:
            pc = arb.load_prob(args.npz_cautious_dir, stem)
            pa = arb.load_prob(args.npz_amplified_dir, stem)
            if pc is not None and pa is not None:
                Min = M0 & (arb.align_prob(pc, gt.shape) > 0.5)
                Mout = M0 | (arb.align_prob(pa, gt.shape) > 0.5)
        Min_t = M0 & (p_b > args.thr_hi)
        Mout_t = M0 | (p_b > args.thr_lo)

        c = s57.contour_of(M0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, M0)
        imgf = ndimage.gaussian_filter(img, args.sigma)
        G = arb.ridge_profile(imgf, pts, nrm, TSF, 1.0)

        ms = args.maxstep
        win = {}
        if Min is not None:
            t_in, t_out = arb.limits(pts, nrm, Min, Mout)
            win["lrn"] = (-np.clip(t_in * args.scale_lrn, 0, ms),
                          np.clip(t_out * args.scale_lrn, 0, ms))
        t_in_t, t_out_t = arb.limits(pts, nrm, Min_t, Mout_t)
        win["thr"] = (-np.clip(t_in_t, 0, ms), np.clip(t_out_t, 0, ms))

        gb = s57.surface(gt)
        row = {"name": stem}
        if "lrn" in win:
            row["contain_lrn"] = float((gb & (Mout & ~Min)).sum() / max(1, gb.sum()))
            row["band_lrn_px"] = float(np.mean(-win["lrn"][0] + win["lrn"][1]))
        row["contain_thr"] = float((gb & (Mout_t & ~Min_t)).sum() / max(1, gb.sum()))
        row["band_thr_px"] = float(np.mean(-win["thr"][0] + win["thr"][1]))

        lo_f, hi_f = -np.full(NPT, ms), np.full(NPT, ms)
        lo, hi = win.get("lrn", (lo_f, hi_f))
        res = {"raw": (M0, None)}
        if not args.no_baselines:
            t_free = arb.pick(G, TSF, args.frac, lo_f, hi_f)
            res["free"] = arb.apply_t(M0, pts, nrm, t_free)
            for tag, (wl, wh) in win.items():
                res["band_" + tag] = arb.apply_t(M0, pts, nrm,
                                                 arb.pick(G, TSF, args.frac, wl, wh))
            if "lrn" in win:
                ok = (t_free <= hi + 1e-9) & (t_free >= lo - 1e-9)
                res["veto_lrn"] = arb.apply_t(M0, pts, nrm, np.where(ok, t_free, 0.0))

        if not args.no_dp:
            # controls: lam = 0 must reproduce the corresponding greedy rule
            tg = arb.pick(G, TSF, args.frac, lo, hi)
            t0 = solve_cand(G, lo, hi, args, "gate", lam=0.0)
            ctrl_g.append(float(np.max(np.abs(t0 - tg))))
            res["dp_g0_lrn"] = arb.apply_t(M0, pts, nrm, t0)
            res["dp_c0_lrn"] = arb.apply_t(
                M0, pts, nrm, solve_cand(G, lo, hi, args, "crest", lam=0.0))

            for lam in lams:
                sfx = "" if not multi else "_l%g" % lam
                res["dp_g_free" + sfx] = arb.apply_t(
                    M0, pts, nrm, solve_cand(G, lo_f, hi_f, args, "gate", lam=lam))
                res["dp_c_free" + sfx] = arb.apply_t(
                    M0, pts, nrm, solve_cand(G, lo_f, hi_f, args, "crest", lam=lam))
                for tag, (wl, wh) in win.items():
                    res["dp_g_" + tag + sfx] = arb.apply_t(
                        M0, pts, nrm, solve_cand(G, wl, wh, args, "gate", lam=lam))
                    res["dp_c_" + tag + sfx] = arb.apply_t(
                        M0, pts, nrm, solve_cand(G, wl, wh, args, "crest", lam=lam))
                res["dp_r_lrn" + sfx] = arb.apply_t(
                    M0, pts, nrm, solve_reward(G, lo, hi, args, "rel", lam=lam))
                res["dp_rb_lrn" + sfx] = arb.apply_t(
                    M0, pts, nrm, solve_reward(G, lo, hi, args, "both", lam=lam))
                res["dp_ra_lrn" + sfx] = arb.apply_t(
                    M0, pts, nrm, solve_reward(G, lo, hi, args, "abs", lam=lam))
                # does the LEARNED band still matter inside the reward (snake)
                # formulation?  same energy, only the trust region changes.
                res["dp_r_free" + sfx] = arb.apply_t(
                    M0, pts, nrm, solve_reward(G, lo_f, hi_f, args, "rel", lam=lam))
                if "thr" in win:
                    wtl, wth = win["thr"]
                    res["dp_r_thr" + sfx] = arb.apply_t(
                        M0, pts, nrm, solve_reward(G, wtl, wth, args, "rel", lam=lam))

        for v, (m, t) in res.items():
            row["bdice_" + v] = s57.bdice(m, gt, 1)
            row["dice_" + v] = s57.dice(m, gt)
            if t is not None:
                row["move_" + v] = float(np.mean(np.abs(t)))
                row["tvmove_" + v] = float(np.mean(np.abs(np.roll(t, -1) - t)))
            if args.write and args.out_dir:
                arb.write_png(os.path.join(args.out_dir, v, stem + ".png"), m)
        rows.append(row)
        if (k + 1) % 5 == 0:
            print("[70] %d/%d  (%.1fs)" % (k + 1, len(names), time.time() - t00))

    variants = [v[len("bdice_"):] for v in sorted({q for r in rows
                                                   for q in r if q.startswith("bdice_")})]
    head = ["raw", "free", "band_lrn", "band_thr", "veto_lrn", "dp_g0_lrn", "dp_c0_lrn"]
    variants = [v for v in head if v in variants] + \
               sorted(v for v in variants if v not in head)

    print("\n[70] ==== %s | split %s | n=%d | frac %.2f pen %.1f maxstep %.1f "
          "sigma %.1f dp_win %.1f scale_lrn %.2f | lams %s ===="
          % (args.tag, args.split, len(rows), args.frac, args.pen, args.maxstep,
             args.sigma, args.dp_win, args.scale_lrn, lams))
    summary = {"tag": args.tag, "split": args.split, "n": len(rows),
               "params": {q: getattr(args, q) for q in
                          ("frac", "maxstep", "sigma", "dp_win", "lam", "pen",
                           "gamma", "sigma_p", "thr_hi", "thr_lo", "scale_lrn",
                           "lam_list", "scale_lrn")},
               "lams": lams, "mean": {}, "median": {}, "wins": {},
               "control_gate_l0_vs_greedy": float(np.max(ctrl_g)) if ctrl_g else None}
    base_b = np.array([r.get("bdice_raw", np.nan) for r in rows])
    for v in variants:
        vals = np.array([r.get("bdice_" + v, np.nan) for r in rows], dtype=float)
        summary["mean"][v] = float(np.nanmean(vals))
        summary["median"][v] = float(np.nanmedian(vals))
        dice_m = float(np.nanmean([r.get("dice_" + v, np.nan) for r in rows]))
        if v != "raw":
            d = vals - base_b
            summary["wins"][v] = {"vs_raw": float(np.nanmean(d)),
                                  "n_better": int(np.nansum(d > 0.1)),
                                  "n_worse": int(np.nansum(d < -0.1))}
            mv = float(np.nanmean([r.get("move_" + v, np.nan) for r in rows]))
            tv = float(np.nanmean([r.get("tvmove_" + v, np.nan) for r in rows]))
            print("[70]   %-16s bdice %6.2f  dice %6.2f  vs-raw %+6.2f  "
                  "(+%d/-%d)  |t| %.2f  TV %.3f"
                  % (v, summary["mean"][v], dice_m, summary["wins"][v]["vs_raw"],
                     summary["wins"][v]["n_better"], summary["wins"][v]["n_worse"],
                     mv, tv))
        else:
            print("[70]   %-16s bdice %6.2f  dice %6.2f"
                  % (v, summary["mean"][v], dice_m))
    if summary["control_gate_l0_vs_greedy"] is not None:
        print("[70]   control gate lam=0 vs step6 greedy: max px diff = %.3g"
              % summary["control_gate_l0_vs_greedy"])

    os.makedirs(args.results_dir, exist_ok=True)
    jp = os.path.join(args.results_dir, "dp_arbitrate_%s.json" % args.tag)
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_image": rows}, fh, indent=1)
    cp = os.path.join(args.results_dir, "dp_arbitrate_%s.csv" % args.tag)
    cols = ["name"] + sorted({q for r in rows for q in r if q != "name"})
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[70] wrote %s" % jp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
