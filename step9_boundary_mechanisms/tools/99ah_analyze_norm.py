#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99ah_analyze_norm.py

Evaluate the test-time normalization battery (99ag) AND normalization-view
probability ensembling, against the raw (in-domain-preprocessing) prediction of
the SAME shipped checkpoint.

reads  step8_domain_shift/predictions/tn_<variant>__<cond>_base_nifti/*.npz
       step8_domain_shift/predictions/sq2shift_<cond>_base_nifti/*.npz   (raw)

writes step9_boundary_mechanisms/results/testtime_norm.json
"""
import argparse
import json
import os
import sys

import numpy as np

USS = "/media/72TB_disk/personal/zk/USSnet"
S8 = os.path.join(USS, "step8_domain_shift")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
sys.path.insert(0, os.path.join(S8, "tools"))
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
import eval_lib  # noqa: E402
import importlib.util  # noqa: E402

sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))
import importlib  # noqa: E402
m82 = importlib.import_module("82_clinical_metrics")
s57 = importlib.import_module("57_shape_prior_ceiling")

MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")

VARIANTS = ["flat64", "flat128", "tgcrow", "pwl", "histmatch"]
CONDS = ["in_domain", "compound_B"]
COLLAPSE = ["丁全土_001_161201", "李明_001_160415", "何朝良_002_173039",
            "陈靖平_001_110332", "崔志和_001_162508"]
PRED = os.path.join(S8, "predictions")


def centroid(m):
    ys, xs = np.nonzero(m)
    return (float(ys.mean()), float(xs.mean())) if ys.size else None


def load_fg(fp):
    """nnU-Net saves probabilities with shape (2, 1, H, W): squeeze the
    singleton, then take channel 1 (foreground).  82_clinical_metrics'
    read_npz_prob takes a[0] on 4-D input, i.e. the BACKGROUND channel."""
    with np.load(fp) as d:
        p = np.asarray(d["probabilities"], dtype=np.float32)
    while p.ndim > 3 and p.shape[1] == 1:
        p = p[:, 0]
    if p.ndim == 3 and p.shape[0] == 2:
        p = p[1]
    return p


def load(tag, stem, cond):
    d = os.path.join(PRED, "%s_%s_base_nifti" % (tag, cond))
    fp = os.path.join(d, stem + ".npz")
    if not os.path.exists(fp):
        return None
    return load_fg(fp)


def metrics(p, g):
    pred = p > 0.5
    pl, gl = s57.largest_cc(pred), s57.largest_cc(g)
    cp, cg = centroid(pl), centroid(gl)
    disp = float(np.hypot(cp[0] - cg[0], cp[1] - cg[1])) if (cp and cg) else float("nan")
    return dict(dice=float(s57.dice(pred, g)), bdice1=float(s57.bdice(pred, g, 1)),
                centroid_disp=disp)


def agg(rows):
    d = np.array([r["dice"] for r in rows], float)
    b = np.array([r["bdice1"] for r in rows], float)
    c = np.array([r["centroid_disp"] for r in rows], float)
    coll = [r["dice"] for r in rows if r["name"] in COLLAPSE]
    return dict(n=len(rows), dice=float(d.mean()), bdice1=float(b.mean()),
                min_dice=float(d.min()), n_below80=int((d < 80).sum()),
                n_below50=int((d < 50).sum()), disp=float(np.nanmean(c)),
                disp_p90=float(np.nanpercentile(c, 90)),
                collapse5_dice=[round(float(x), 2) for x in coll])


ENSEMBLES = {
    "ens_raw_flat64": ["raw", "flat64"],
    "ens_raw_pwl": ["raw", "pwl"],
    "ens_raw_histmatch": ["raw", "histmatch"],
    "ens_raw_flat64_pwl_hist": ["raw", "flat64", "pwl", "histmatch"],
    "ens_all6": ["raw"] + VARIANTS,
    "ens_flat64_pwl_hist": ["flat64", "pwl", "histmatch"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(S9, "results", "testtime_norm.json"))
    args = ap.parse_args()

    _, val_names = eval_lib.read_split(USS)
    names = [n for n in val_names]
    stems = [os.path.splitext(n)[0] for n in names]
    gt = {n: eval_lib.load_binary(os.path.join(MASK_DIR, n)) for n in names}

    allout = {}
    for cond in CONDS:
        P = {"raw": {}}
        for v in VARIANTS:
            P[v] = {}
        for n, st in zip(names, stems):
            a = load("sq2shift", st, cond)
            if a is None:
                print("[99ah] missing raw", cond, st)
                continue
            P["raw"][n] = a
            for v in VARIANTS:
                b = load("tn", st, "%s__%s" % (v, cond))
                if b is not None:
                    P[v][n] = b
        rows = {}
        for src in ["raw"] + VARIANTS:
            rr = []
            for n in names:
                if n not in P[src]:
                    continue
                m = metrics(P[src][n], gt[n])
                m["name"] = n
                rr.append(m)
            rows[src] = rr
        for en, srcs in ENSEMBLES.items():
            rr = []
            for n in names:
                if any(n not in P[s] for s in srcs):
                    continue
                q = np.mean([P[s][n] for s in srcs], axis=0)
                m = metrics(q, gt[n])
                m["name"] = n
                rr.append(m)
            rows[en] = rr
        print("\n=== %s (val 31, shipped SQ2res-w25, base head)" % cond)
        print("  %-26s %7s %7s %7s %6s %6s %7s %7s" %
              ("source", "dice", "bdice1", "min", "<80", "<50", "disp", "disp90"))
        base = None
        for k in ["raw"] + VARIANTS + list(ENSEMBLES):
            s = agg(rows[k])
            if k == "raw":
                base = s
            print("  %-26s %7.2f %7.2f %7.2f %6d %6d %7.1f %7.1f  %s" %
                  (k, s["dice"], s["bdice1"], s["min_dice"], s["n_below80"],
                   s["n_below50"], s["disp"], s["disp_p90"],
                   ("Δdice %+.2f" % (s["dice"] - base["dice"])) if base else ""))
        allout[cond] = {k: agg(v) for k, v in rows.items()}
        print("  collapsed-5 dice: raw %s" % allout[cond]["raw"]["collapse5_dice"])
        for k in list(ENSEMBLES) + VARIANTS:
            print("    %-24s %s" % (k, allout[cond][k]["collapse5_dice"]))

    json.dump(allout, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n[99ah] wrote %s" % args.out)


if __name__ == "__main__":
    main()
