#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/30_grid_compare.py

Recipe-stability grid: compare training recipes on the weakest splits
(0, 3, 4) and report, per config, the mean +- std across those splits.

Reference config ("u") = warmup 20 / lr 1e-3  (the 5-split protocol run).
Grid configs are launched with:
    WARMUP=50 LR=1e-3 PREFIX=l2vit_gC bash scripts/run_fold_queue.sh 6 0 3 4   # C
    WARMUP=20 LR=5e-4 PREFIX=l2vit_gD bash scripts/run_fold_queue.sh 7 0 3 4   # D

usage: ~/anaconda3/envs/zk/bin/python scripts/30_grid_compare.py
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
FOLDS = [0, 3, 4]
KEYS = ["in_domain", "compound_B", "blur_35"]
HELD = ["blur_45", "speckle_50", "compound_C"]


def load(tag):
    p = os.path.join(RES, "l2vit_eval_%s.json" % tag)
    h = os.path.join(RES, "l2vit_eval_%s_heldout.json" % tag)
    a = json.load(open(p))["summary"] if os.path.exists(p) else None
    b = json.load(open(h))["summary"] if os.path.exists(h) else None
    return a, b


def main():
    configs = []                       # (label, tag-prefix, note)
    for pref, lbl, note in [("l2vit_u", "B  ref  warmup20 lr1e-3", "reference (5-split protocol)"),
                            ("l2vit_gC", "C      warmup50 lr1e-3", "slower warmup"),
                            ("l2vit_gD", "D      warmup20 lr5e-4", "lower lr")]:
        configs.append((lbl, pref, note))

    table = {}
    for lbl, pref, note in configs:
        per_fold = {}
        for k in FOLDS:
            a, b = load("%s_fold%d_final" % (pref, k))
            if a:
                per_fold[k] = (a, b)
        if per_fold:
            table[lbl] = (note, per_fold)

    if not table:
        print("no results yet"); return 1

    print("## 配置 × 折（Dice；留出条件用于验证不是过拟合到所见扰动）\n")
    hdr = "| 配置 | " + " | ".join("fold %d" % k for k in FOLDS) + " | 均值±标准差 | in_domain bdice@1 |"
    print(hdr); print("|" + "---|" * (len(FOLDS) + 3))
    for lbl in sorted(table):
        note, pf = table[lbl]
        vals = [pf[k][0]["compound_B"]["dice"] for k in FOLDS if k in pf]
        bd = [pf[k][0]["in_domain"]["bdice1"] for k in FOLDS if k in pf]
        cells = ["%.2f" % pf[k][0]["compound_B"]["dice"] if k in pf else "—" for k in FOLDS]
        ms = "%.2f±%.2f" % (st.mean(vals), st.stdev(vals)) if len(vals) > 1 else "—"
        print("| %s | %s | **%s** | %.2f |" % (lbl, " | ".join(cells), ms, st.mean(bd) if bd else 0))

    print("\n## 每配置/折的关键条件\n")
    for lbl in sorted(table):
        note, pf = table[lbl]
        print("### %s (%s)" % (lbl, note))
        print("| fold | in_domain | compound_B | blur_35 | bdice@1 | held-out blur_45 / speckle_50 / compound_C |")
        print("|---|---|---|---|---|---|")
        for k in FOLDS:
            if k not in pf:
                print("| %d | — | — | — | — | — |" % k); continue
            a, b = pf[k]
            hs = " / ".join("%.2f" % b[c]["dice"] for c in HELD) if b else "—"
            print("| %d | %.2f | %.2f | %.2f | %.2f | %s |" % (
                k, a["in_domain"]["dice"], a["compound_B"]["dice"], a["blur_35"]["dice"], a["in_domain"]["bdice1"], hs))
        done = [k for k in FOLDS if k in pf]
        if len(done) > 1:
            flat = [a["in_domain"]["dice"] - min(a[c]["dice"] for c in a) for k, (a, b) in pf.items()]
            print("折内最大落差: %s（均值 %.2f）" % (["%.2f" % x for x in flat], st.mean(flat)))
        print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
