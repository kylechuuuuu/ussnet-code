#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_band_arbitration/tools/71_paired_stats.py

Paired significance analysis of the arbitration variants from a
results/dp_arbitrate_<tag>.csv produced by 70_dp_arbitrate.py.

For each variant vs a reference (default band_lrn, step 6's best rule) it prints
the mean delta, the win/loss/tie count, a paired bootstrap 95% CI of the mean
delta and a two-sided Wilcoxon signed-rank p-value.  Nothing is trained here --
this only asks whether an observed delta on n=31 val cases is bigger than the
noise of the split.

usage:
  python tools/71_paired_stats.py --csv results/dp_arbitrate_dpval_sweep.csv \
      [--ref band_lrn] [--variants dp_r_lrn_l0.4,...] [--metric bdice] [--n-boot 20000]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEP7 = os.path.abspath(os.path.join(HERE, ".."))


def read_csv(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref", default="band_lrn")
    ap.add_argument("--variants", default="")
    ap.add_argument("--metric", default="bdice")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    rows = read_csv(args.csv)
    n = len(rows)
    pref = args.metric + "_"

    def col(v):
        k = pref + v
        if k not in rows[0]:
            return None
        return np.array([float(r[k]) for r in rows])

    allv = sorted({k[len(pref):] for k in rows[0] if k.startswith(pref)})
    ref = col(args.ref)
    if ref is None:
        raise SystemExit("reference %s not in csv" % args.ref)

    if args.variants:
        cands = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        cands = sorted(allv, key=lambda v: -np.nanmean(col(v)))[:args.top]

    rng = np.random.default_rng(12345)
    idx = rng.integers(0, n, size=(args.n_boot, n))

    print("[71] %s | n=%d | metric=%s | ref=%s (mean %.3f)"
          % (os.path.basename(args.csv), n, args.metric, args.ref, ref.mean()))
    print("[71] %-20s %8s %8s %6s %6s %6s   %-18s %8s" %
          ("variant", "mean", "delta", "win", "loss", "tie", "95% CI of delta", "p(wilcox)"))
    for v in cands:
        a = col(v)
        if a is None:
            print("[71]   %-20s (missing)" % v)
            continue
        d = a - ref
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        win = int((d > 1e-9).sum())
        loss = int((d < -1e-9).sum())
        tie = n - win - loss
        p = float("nan")
        try:
            from scipy.stats import wilcoxon
            nz = d[np.abs(d) > 1e-9]
            if len(nz) >= 5:
                p = float(wilcoxon(nz).pvalue)
        except Exception:
            pass
        print("[71] %-20s %8.3f %+8.3f %6d %6d %6d   [%+7.3f, %+7.3f] %8.4f"
              % (v, a.mean(), d.mean(), win, loss, tie, lo, hi, p))

    print("")
    print("[71] top-10 variants by mean %s:" % args.metric)
    for v in sorted(allv, key=lambda v: -np.nanmean(col(v)))[:10]:
        print("[71]   %-22s %.3f" % (v, np.nanmean(col(v))))


if __name__ == "__main__":
    main()
