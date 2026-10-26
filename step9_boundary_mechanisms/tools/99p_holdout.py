#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99p_holdout.py

Honest, out-of-sample check of the ensemble gain.

The weight sweep (99k) selects the best candidate on the same 31 val cases it
is then scored on, so its +0.42 bdice@1px is an upper bound.  Here the ranking
is done on one half of the val cases and the chosen candidate is scored on the
other half, repeated for many random splits:

  round r:  shuffle val -> select on the first `nsel`, score on the rest
            candidate_r = argmax bdice1 on the selection half
            delta_r     = bdice1(candidate_r) - bdice1(res_best)  on the holdout

Also reported, on the same holdout halves, the fixed candidates best4_res6 /
best4_res8 (no selection) so the two numbers are comparable.

usage: python tools/99p_holdout.py [--ref res_best] [--rounds 200] [--seed 0]
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

S9 = "/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms"
USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
PRED = os.path.join(S6, "predictions")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


k99 = load_mod("k99", os.path.join(S9, "tools", "99k_weight_sweep.py"))
arb = load_mod("arb", os.path.join(S6, "tools", "63_arbitrate.py"))
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary",
                                   "tools", "57_shape_prior_ceiling.py"))


def name2cid_map():
    import csv
    p = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            out[r["original_filename"]] = r["case_id"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="res_best")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n2c = name2cid_map()
    _train, val = arb.read_split(USS)
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}

    have = [(l, d) for l, d in k99.MEMBERS
            if os.path.isfile(os.path.join(PRED, d, n2c[val[0]] + ".npz"))]
    if not any(l == args.ref for l, _ in have):
        raise SystemExit("reference member %s missing" % args.ref)
    print("members present: %s" % [l for l, _ in have])

    P = {}
    for lab, d in have:
        P[lab] = {n: k99.load_probs(os.path.join(PRED, d, n2c[n] + ".npz"))
                  for n in val}

    cands = [(nm, w) for nm, w in k99.CANDIDATES
             if all(lab in P for lab in w) and nm != args.ref]
    print("candidates: %d" % len(cands))

    def score_case(weights, n):
        acc = None
        for lab, w in weights.items():
            q = P[lab][n]
            acc = q * w if acc is None else acc + q * w
        acc = acc / float(sum(weights.values()))
        pred = s57.largest_cc(acc[1] >= 0.5)
        return s57.bdice(pred, gts[n], tol=1)

    B = np.zeros((len(cands), len(val)), dtype=np.float64)
    for ci, (nm, w) in enumerate(cands):
        for vi, n in enumerate(val):
            B[ci, vi] = score_case(w, n)
        sys.stdout.write("\r  scored %2d/%d %-22s" % (ci + 1, len(cands), nm))
        sys.stdout.flush()
    print()

    ref_ci = [i for i, (l, _) in enumerate(have) if l == args.ref]
    ref = [nm for nm, _ in cands]
    # reference row per case
    ref_b = np.array([score_case({args.ref: 1}, n) for n in val])

    idx_fix = {}
    for want in ("best4_res2", "best4_res4", "best4_res6", "best4_res8",
                 "best4_equal"):
        if want in ref:
            idx_fix[want] = ref.index(want)

    rng = np.random.default_rng(args.seed)
    n = len(val)
    nsel = n // 2
    picks = {nm: 0 for nm, _ in cands}
    sel_rows, fix_rows = [], {k: [] for k in idx_fix}
    full_rows = []
    for r in range(args.rounds):
        perm = rng.permutation(n)
        sel, hold = perm[:nsel], perm[nsel:]
        m = B[:, sel].mean(axis=1)
        best = int(np.argmax(m))
        picks[cands[best][0]] += 1
        sel_rows.append(B[best, hold].mean() - ref_b[hold].mean())
        for want, ci in idx_fix.items():
            fix_rows[want].append(B[ci, hold].mean() - ref_b[hold].mean())
    for ci, (nm, _) in enumerate(cands):
        full_rows.append((nm, float(B[ci].mean()), float(B[ci].mean() - ref_b.mean())))

    print("\nrounds=%d  selection half=%d  holdout half=%d"
          % (args.rounds, nsel, n - nsel))
    print("\n%-26s %10s %8s %10s" % ("strategy", "holdout_d", "se", "win_frac"))
    arr = np.array(sel_rows)
    print("%-26s %+10.3f %8.3f %10.2f" %
          ("select-on-half", arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr)),
           float((arr > 0).mean())))
    for want, v in fix_rows.items():
        v = np.array(v)
        print("%-26s %+10.3f %8.3f %10.2f" %
              ("fixed:" + want, v.mean(), v.std(ddof=1) / np.sqrt(len(v)),
               float((v > 0).mean())))

    print("\nchosen-on-selection-half histogram:")
    for nm, c in sorted(picks.items(), key=lambda t: -t[1]):
        if c:
            print("   %-24s %4d" % (nm, c))

    print("\nfull-val (in-sample, upper bound) deltas vs %s:" % args.ref)
    for nm, bd, d in sorted(full_rows, key=lambda t: -t[2]):
        print("   %-24s bdice1 %8.3f  d %+7.3f" % (nm, bd, d))

    out = os.path.join(S9, "results", "holdout_sweep_val.json")
    with open(out, "w") as f:
        json.dump({"ref": args.ref, "rounds": args.rounds, "seed": args.seed,
                   "select_on_half": {"mean": float(arr.mean()),
                                      "se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
                                      "win_frac": float((arr > 0).mean())},
                   "fixed": {w: {"mean": float(np.mean(v)),
                                 "se": float(np.std(v, ddof=1) / np.sqrt(len(v))),
                                 "win_frac": float((np.array(v) > 0).mean())}
                             for w, v in fix_rows.items()},
                   "picks": {nm: int(c) for nm, c in picks.items()},
                   "full_val": [{"name": nm, "bdice1": bd, "delta": d}
                                for nm, bd, d in full_rows]}, f, indent=2)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
