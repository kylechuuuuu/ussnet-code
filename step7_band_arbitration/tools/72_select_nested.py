#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_band_arbitration/tools/72_select_nested.py

The step-7 arbitration has free parameters (the snake's elasticity gamma, its
length scale sigma_p, the field smoothness lam).  Picking them by looking at the
same 31 validation cases we then report is exactly the mistake step 4 made, so
this tool estimates the gain the way it would actually be used:

    nested selection -- for every held-out case, choose the parameter setting
    that is best on the OTHER cases only, then apply it to the held-out one.

Reported numbers are therefore honest (no parameter has seen the case it is
scored on) and are directly comparable to the fixed-parameter step-6 rules.

usage:
  python tools/72_select_nested.py --results-dir results \
      [--pattern '*dpgrid_*.csv,dp_arbitrate_dpval_sweep.csv'] \
      [--ref band_lrn] [--metric bdice]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEP7 = os.path.abspath(os.path.join(HERE, ".."))

GRID_RE = re.compile(r"dpgrid_g(?P<g>[\d.]+)_s(?P<s>[\d.]+)\.csv$")
LAM_RE = re.compile(r"_l(?P<lam>[\d.]+)$")


def load_csv(path, metric="bdice"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    pref = metric + "_"
    out = {}
    for k in rows[0]:
        if k.startswith(pref):
            out[k[len(pref):]] = np.array([float(r[k]) for r in rows])
    return out, [r["name"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(STEP7, "results"))
    ap.add_argument("--pattern", default="*dpgrid_*.csv,dp_arbitrate_dpval_sweep.csv")
    ap.add_argument("--ref", default="band_lrn")
    ap.add_argument("--metric", default="bdice")
    ap.add_argument("--out", default=os.path.join(STEP7, "results", "nested_selection.json"))
    args = ap.parse_args()

    files = []
    for pat in args.pattern.split(","):
        files += sorted(glob.glob(os.path.join(args.results_dir, pat.strip())))
    if not files:
        raise SystemExit("no result csv matched")

    # ------------------------------------------------------------- gather grid
    cases = None
    grid = {}       # key -> per-case array
    base = {}       # variant name -> per-case array (from the plain sweep)
    params_of = {}
    for path in files:
        d, names = load_csv(path, args.metric)
        if cases is None:
            cases = names
        elif names != cases:
            print("[72] WARNING case order differs in %s - skipped" % path)
            continue
        m = GRID_RE.search(path)
        if m:
            g, s = float(m.group("g")), float(m.group("s"))
            jf = path[:-4] + ".json"
            if os.path.exists(jf):
                with open(jf) as f:
                    pj = json.load(f).get("summary", {}).get("params", {})
                g = float(pj.get("gamma", g))
                s = float(pj.get("sigma_p", s))
            for v, arr in d.items():
                if not v.startswith("dp_"):
                    base.setdefault(v, arr)
                    continue
                lm = LAM_RE.search(v)
                if not lm:
                    base.setdefault(v, arr)
                    continue
                fam = v[:lm.start()]           # e.g. dp_r_lrn
                key = "%s|g%.2f|s%.1f|l%.2f" % (fam, g, s, float(lm.group("lam")))
                grid[key] = arr
                params_of[key] = {"family": fam, "gamma": g, "sigma_p": s,
                                  "lam": float(lm.group("lam"))}
        else:
            for v, arr in d.items():
                base.setdefault(v, arr)

    n = len(cases)
    print("[72] %d cases | %d grid settings | %d baseline variants" % (n, len(grid), len(base)))
    ref = base.get(args.ref)
    if ref is None:
        raise SystemExit("reference %s missing" % args.ref)

    keys = sorted(grid)
    X = np.vstack([grid[k] for k in keys])          # (K, n)

    def nested(selector):
        """selector(train_idx) -> key index; returns held-out predictions."""
        pred = np.zeros(n)
        chosen = []
        for i in range(n):
            tr = np.array([j for j in range(n) if j != i])
            ki = selector(tr)
            chosen.append(keys[ki])
            pred[i] = X[ki, i]
        return pred, chosen

    def best_on(tr, pool=None):
        pool = range(len(keys)) if pool is None else pool
        means = [X[k][tr].mean() for k in pool]
        return list(pool)[int(np.argmax(means))]

    pred_loo, chosen_loo = nested(best_on)

    rng = np.random.default_rng(2024)
    idx = rng.integers(0, n, size=(20000, n))

    def report(name, arr):
        d = arr - ref
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = float("nan")
        try:
            from scipy.stats import wilcoxon
            nz = d[np.abs(d) > 1e-9]
            if len(nz) >= 5:
                p = float(wilcoxon(nz).pvalue)
        except Exception:
            pass
        print("[72] %-24s %7.3f  (%+.3f vs %s)  win %2d loss %2d  CI [%+.3f,%+.3f]  p=%.4f"
              % (name, arr.mean(), d.mean(), args.ref,
                 int((d > 1e-9).sum()), int((d < -1e-9).sum()), lo, hi, p))
        return {"mean": float(arr.mean()), "delta": float(d.mean()),
                "ci": [float(lo), float(hi)], "p": p,
                "win": int((d > 1e-9).sum()), "loss": int((d < -1e-9).sum())}

    print("")
    res = {}
    for v in ("raw", "free", "band_lrn", "band_thr", "veto_lrn", "dp_g_free_l0.15",
              "dp_g_lrn_l0.4", "dp_r_lrn_l0.4"):
        if v in base:
            res[v] = report(v + " (fixed)", base[v])
    res["nested_loo"] = report("nested-LOO snake", pred_loo)

    # oracle: the single best setting chosen with knowledge of all 31 cases
    kbest = best_on(np.arange(n))
    res["oracle_best_fixed"] = report("oracle-best fixed (%s)" % keys[kbest], X[kbest])

    from collections import Counter
    c = Counter(chosen_loo)
    print("\n[72] nested-LOO parameter stability (how often each setting was chosen):")
    for k, cnt in c.most_common(8):
        print("[72]   %-28s %2d/%d" % (k, cnt, n))

    res["params_of"] = params_of
    res["chosen_loo"] = chosen_loo
    res["cases"] = cases
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n[72] wrote %s" % args.out)


if __name__ == "__main__":
    main()
