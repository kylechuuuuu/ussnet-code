#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99ao_pin_gate.py

Does the GT-free guard from 99ai generalise to the two "pin" conditions, or is it
a compound_B-specific trick?

The pin experiment (99ak/99al/99an) showed the compound_B recipe factorises as
    texture destroyed only (abC_texture_only)   -> dice 92.96, 2/31 gross failures
    smooth gain only     (abD_gain_only)        -> dice 96.13, 0/31
    both (= compound_B)                         -> dice 78.99, 8/31
i.e. texture loss is the enabling condition and the gain multiplies it.

Here the same conf_margin gate (tau = min over in_domain val, fall back to the
stock nnU-Net arm / to a pooled probability field) is applied to abC and abD.
If it fires on the failures and leaves the healthy cases alone, the guard is a
general OOD device, not a per-condition hack.

usage: python tools/99ao_pin_gate.py
"""
import importlib
import json
import os
import sys

import numpy as np

S9 = "/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms"
sys.path.insert(0, os.path.join(S9, "tools"))
s9ai = importlib.import_module("99ai_robust_fallback")
eval_lib, s57 = s9ai.eval_lib, s9ai.s57
load_fg, conf_margin, dice = s9ai.load_fg, s9ai.conf_margin, s9ai.dice
USS, MASK_DIR = s9ai.USS, s9ai.MASK_DIR

CONDS = [("in_domain", "sq2shift", "stockshift"),
         ("compound_B", "sq2shift", "stockshift"),
         ("abC_texture_only", "ab", "stkab"),
         ("abD_gain_only", "ab", "stkab")]
A_POOL = 0.4


def main():
    _, names = eval_lib.read_split(USS)
    stems = [os.path.splitext(n)[0] for n in names]
    gt = {n: eval_lib.load_binary(os.path.join(MASK_DIR, n)) for n in names}

    P = {}
    for cond, tq, ts in CONDS:
        P[cond] = {}
        for n, st in zip(names, stems):
            q = load_fg(tq, st, cond)
            s = load_fg(ts, st, cond)
            if q is None or s is None:
                print("[99ao] missing %s %s (sq=%s st=%s)" % (cond, st, q is not None, s is not None))
                continue
            P[cond][n] = dict(sq=q, st=s, cm=conf_margin(q),
                              d_sq=dice(q > 0.5, gt[n]), d_st=dice(s > 0.5, gt[n]))
        print("[99ao] %-18s n=%d" % (cond, len(P[cond])))

    tau = float(np.min([v["cm"] for v in P["in_domain"].values()]))
    print("\n[99ao] tau (min conf_margin over in_domain) = %.4f" % tau)

    res = {"tau": tau, "a_pool": A_POOL, "rows": {}}
    print("\n  %-18s %6s %7s %8s %8s %8s %7s" %
          ("condition", "fire", "sq2", "fallback", "pooled", "oracle", "best_arm"))
    for cond, _, _ in CONDS:
        if cond not in P:
            continue
        v = P[cond]
        fire = [n for n in v if v[n]["cm"] < tau]
        d_sq = np.mean([x["d_sq"] for x in v.values()])
        d_st = np.mean([x["d_st"] for x in v.values()])
        gated = np.mean([(v[n]["d_st"] if n in fire else v[n]["d_sq"]) for n in v])
        pooled = np.mean([dice((A_POOL * v[n]["sq"] + (1 - A_POOL) * v[n]["st"]) > 0.5, gt[n])
                          for n in v])
        gpool = np.mean([(dice((A_POOL * v[n]["sq"] + (1 - A_POOL) * v[n]["st"]) > 0.5, gt[n])
                          if n in fire else v[n]["d_sq"]) for n in v])
        oracle = np.mean([max(v[n]["d_sq"], v[n]["d_st"]) for n in v])
        row = {"n": len(v), "n_fire": len(fire), "fire_cases": sorted(fire),
               "dice_sq2": float(d_sq), "dice_fallback_arm": float(d_st),
               "dice_gated": float(gated), "dice_gated_pool": float(gpool),
               "dice_pool_uncond": float(pooled), "dice_oracle": float(oracle),
               "cm_min": float(min(x["cm"] for x in v.values())),
               "cm_max": float(max(x["cm"] for x in v.values()))}
        res["rows"][cond] = row
        print("  %-18s %2d/%-3d %7.2f %8.2f %8.2f %8.2f %7.2f" %
              (cond, len(fire), len(v), d_sq, d_st, gated, oracle,
               max(d_sq, d_st)))
        if fire:
            print("       fired: %s" % ", ".join(sorted(fire)))

    print("\n  per-case conf_margin vs sq2 dice (the two pin conditions):")
    for cond in ["abC_texture_only", "abD_gain_only"]:
        if cond not in P:
            continue
        v = P[cond]
        print("    %-18s" % cond, " ".join("%s:%.0f/%.2f" % (n.split("_")[0], v[n]["d_sq"], v[n]["cm"])
                                           for n in sorted(v, key=lambda k: v[k]["cm"])[:8]))

    out = os.path.join(S9, "results", "pin_gate.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print("\n[99ao] wrote %s" % out)


if __name__ == "__main__":
    main()
