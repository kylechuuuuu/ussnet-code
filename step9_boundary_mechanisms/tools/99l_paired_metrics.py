#!/usr/bin/env python3
"""99l_paired_metrics.py -- paired significance tests on the official per-image CSVs.

The bdice@1px sweep (99k) answers "is the ensemble better at the 1px boundary?";
this answers the same question for the three metrics the project actually
reports (Dice / HD95 / ASSD), which live in step2's `metrics/*_per_image.csv`.

Compare a candidate tag against one or more reference tags, pairing rows on the
image name (so only images present in both are used).

usage:
  python tools/99l_paired_metrics.py --cand ens_J_res6_w25_w25s_sq2 \
      --refs sq2resencw25_checkpoint_best_base nnunet_2d_fold0_checkpoint_best
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

METRICS_DIR = "/media/72TB_disk/personal/zk/USSnet/step2_compare_unet_vs_nnunet/metrics"
# for HD95/ASSD lower is better; for Dice higher is better
FIELDS = [("dice", +1), ("hd95", -1), ("assd", -1)]


def load(tag):
    path = os.path.join(METRICS_DIR, f"{tag}_per_image.csv")
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = r.get("name") or r.get("image") or r.get("case")
            rows[key] = r
    return rows


def paired(cand, ref, field, n_boot=20000, seed=0):
    keys = [k for k in cand if k in ref]
    keys.sort()
    a = np.array([float(cand[k][field]) for k in keys])
    b = np.array([float(ref[k][field]) for k in keys])
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nwin = int((d > 0).sum())
    nloss = int((d < 0).sum())
    # exact two-sided sign test
    from math import comb
    n = nwin + nloss
    k = min(nwin, nloss)
    p = 2.0 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) if n else 1.0
    p = min(p, 1.0)
    return dict(n=len(d), cand_mean=float(a.mean()), ref_mean=float(b.mean()),
                delta=float(d.mean()), sd=float(d.std(ddof=1)),
                ci95=[float(lo), float(hi)], win=nwin, loss=nloss, p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cand = load(a.cand)
    out = {}
    print(f"candidate: {a.cand}  (n={len(cand)} images)")
    for ref_tag in a.refs:
        ref = load(ref_tag)
        keys = [k for k in cand if k in ref]
        print(f"\nvs {ref_tag}   paired n={len(keys)}")
        print(f"{'metric':6} {'cand':>9} {'ref':>9} {'delta':>8} "
              f"{'CI95':>20} {'sign p':>8} {'win/loss':>9}")
        out[ref_tag] = {}
        for field, good in FIELDS:
            r = paired(cand, ref, field)
            out[ref_tag][field] = r
            # report the improvement in the "better" direction
            imp = r["delta"] * good
            lo, hi = r["ci95"]
            if good < 0:            # mirror into "improvement" space, keep order
                lo, hi = -hi, -lo
            print(f"{field:6} {r['cand_mean']:9.4f} {r['ref_mean']:9.4f} "
                  f"{imp:+8.4f} [{lo:+7.4f},{hi:+7.4f}] "
                  f"{r['p']:8.4f} {r['win']:4d}/{r['loss']:<4d}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
