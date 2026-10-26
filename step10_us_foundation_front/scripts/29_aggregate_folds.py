#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/29_aggregate_folds.py

Aggregate the uniform 5-split protocol (runs/l2vit_u_fold0..4) into the tables the
project reports: mean +- std per condition, plus the boundary-arbitration
consistency (raw -> free -> band_lrn / veto_lrn and the interval containment).

Protocol note (must be stated in any write-up): splits_final.json holds five
*independent random 71/31 splits* of the same 102 patients -- the val sets overlap
(7-14 cases pairwise, union 85/102), so this is "5 repeated 71/31 splits", NOT a
5-fold partition.  Every fold: 200 epochs, warmup 20, lr 1e-3, batch 2, teacher =
UltraSAM3 24-block trunk on shifted views (1,2,3), and the reported model is the
BEST-VAL checkpoint (ckpt_best.pt, val Dice scored every 25 epochs on 12 val
patients) because the plain "final epoch" rule failed on fold 2 (bad basin).

usage: ~/anaconda3/envs/zk/bin/python scripts/29_aggregate_folds.py [--folds 0,1,2,3,4]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
ARB = "/media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze/results"
CONDS = ["in_domain", "affine_gain_m25", "affine_gain_p25", "blur_20", "blur_35",
         "compound_A", "compound_B", "gamma_060", "gamma_080", "gamma_125",
         "jpeg_15", "jpeg_30", "shadow", "speckle_15", "speckle_35", "tgc_m60", "tgc_p60"]
HELD = ["blur_45", "speckle_50", "compound_C"]


def load(tag):
    p = os.path.join(RES, "l2vit_eval_%s.json" % tag)
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", default=os.path.join(RES, "FIVE_SPLIT_RESULTS.md"))
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    main_tab, held_tab, arb_tab, missing = {}, {}, {}, []
    for k in folds:
        tag = "l2vit_u_fold%d" % k
        s = load(tag)
        h = load(tag + "_heldout")
        if s is None:
            missing.append(k)
            continue
        main_tab[k] = {c: s[c] for c in CONDS if c in s}
        if h:
            held_tab[k] = {c: h[c] for c in HELD if c in h}
        ap_ = os.path.join(ARB, "arbitrate_%s_val.json" % tag)
        if os.path.exists(ap_):
            arb_tab[k] = json.load(open(ap_))["summary"]["mean"]

    if not main_tab:
        print("no completed folds yet; run the queues first"); return 1

    lines = []
    P = lambda s: (print(s), lines.append(s))
    P("# 5-split protocol results (narrow ViT student, 13.33 M)\n")
    P("Protocol: 5 *repeated random 71/31 splits* from step1 splits_final.json "
      "(val sets overlap 7-14 cases, union 85/102 -> NOT a 5-fold partition). "
      "Each fold: 200 ep, warmup 20, lr 1e-3, batch 2, rc6/ra8, teacher = UltraSAM3 "
      "24-block trunk, mirror TTA at inference, reported model = best-val checkpoint.\n")
    P("## Per-condition mean +- std (folds %s)\n" % ",".join(str(k) for k in sorted(main_tab)))
    P("| condition | mean Dice | std | min-fold | max-fold | mean bdice@1 |")
    P("|---|---|---|---|---|---|")
    for c in CONDS:
        ds = [main_tab[k][c]["dice"] for k in sorted(main_tab) if c in main_tab[k]]
        bd = [main_tab[k][c]["bdice1"] for k in sorted(main_tab) if c in main_tab[k]]
        if not ds:
            continue
        m, sd = ms(ds)
        P("| %s | **%.2f** | %.2f | %.2f | %.2f | %.2f |" % (c, m, sd, min(ds), max(ds), float(np.mean(bd))))
    P("")
    P("## Held-out conditions (never used for training)\n")
    P("| condition | mean Dice | std | per-fold |")
    P("|---|---|---|---|")
    for c in HELD:
        ds = [held_tab[k][c]["dice"] for k in sorted(held_tab) if c in held_tab[k]]
        if not ds:
            continue
        m, sd = ms(ds)
        P("| %s | **%.2f** | %.2f | %s |" % (c, m, sd, ", ".join("%.2f" % d for d in ds)))
    P("")
    if arb_tab:
        P("## Boundary arbitration consistency (63_arbitrate.py, same implementation as step6)\n")
        P("| fold | raw | free | band_lrn | veto_lrn | contain_lrn | band px |")
        P("|---|---|---|---|---|---|---|")
        for k in sorted(arb_tab):
            a = arb_tab[k]
            P("| %d | %.2f | %.2f | %.2f | %.2f | %.3f | %.2f |"
              % (k, a["raw"], a["free"], a["band_lrn"], a["veto_lrn"], a["contain_lrn"], a["band_lrn_px"]))
        vals = {key: [arb_tab[k][key] for k in sorted(arb_tab)]
                for key in ["raw", "free", "band_lrn", "veto_lrn", "contain_lrn"]}
        cells = []
        for key in ["raw", "free", "band_lrn", "veto_lrn"]:
            m, sd = ms(vals[key])
            cells.append("%.2f+-%.2f" % (m, sd))
        cm, cs = ms(vals["contain_lrn"])
        P("| **mean+-std** | %s | %.3f+-%.3f | |" % (" | ".join(cells), cm, cs))
        P("")
    if missing:
        P("Folds still missing: %s" % missing)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n[29] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
