#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99ai_robust_fallback.py

The attribution run (99ab/99ae) showed that the compound_B localization collapse
follows the BACKBONE, not the sandwich objective:

    ResEnc backbone, no sandwich   fpB       compound_B 78.82  (min 0.00)
    ResEnc backbone + sandwich     SQ2res-w25          78.99  (min 0.00)
    stock backbone + sandwich      sq2a                91.90  (min 47.19)
    stock backbone, fp loss        fpA                 90.90  (min 17.80)
    stock backbone 1000ep          stock               93.24  (min 46.79)

and in_domain the ordering is reversed (ResEnc 96.38 / 35.26  vs  stock 95.91 /
34.15).  So the two backbones are complementary: one is sharper in domain, the
other survives the acquisition shift.

Two GT-FREE ways to get both, both measurable without training:

  POOL     q = a * p_sq2 + (1-a) * p_stock, threshold 0.5.  Sweeps the
           accuracy/robustness trade-off.  No gate, no threshold to tune.

  FALLBACK step8_report section 3 already found that base-head-only uncertainty
           scores detect gross failure with AUROC 0.990 (unc_area_px) / 0.989
           (conf_margin).  So: use the sharp arm by default and fall back to the
           robust arm when that score is anomalous w.r.t. IN-DOMAIN.  The gate
           never sees the ground truth and never sees compound_B.

writes step9_boundary_mechanisms/results/robust_fallback.json
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
sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))
import eval_lib  # noqa: E402
import importlib  # noqa: E402
s57 = importlib.import_module("57_shape_prior_ceiling")

PRED = os.path.join(S8, "predictions")
MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
SQUAD, STOCK, FPA = "sq2shift", "stockshift", "fpA"


def load_fg(tag, stem, cond):
    fp = os.path.join(PRED, "%s_%s_nifti" % (tag, cond), stem + ".npz")
    if not os.path.exists(fp):
        fp2 = os.path.join(PRED, "%s_%s_base_nifti" % (tag, cond), stem + ".npz")
        if not os.path.exists(fp2):
            return None
        fp = fp2
    with np.load(fp) as d:
        p = np.asarray(d["probabilities"], dtype=np.float32)
    while p.ndim > 3 and p.shape[1] == 1:
        p = p[:, 0]
    if p.ndim == 3 and p.shape[0] == 2:
        p = p[1]
    return p


def unc_area(p):
    return int(((p > 0.05) & (p < 0.95)).sum())


def conf_margin(p):
    m = p > 0.5
    if not m.any():
        return 0.0
    surf = m & ~eval_lib.ndimage.binary_erosion(m, np.ones((3, 3), bool), border_value=0)
    return float(np.abs(2.0 * p[surf] - 1.0).mean()) if surf.any() else 0.0


def dice(pred, g):
    return float(s57.dice(pred, g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(S9, "results", "robust_fallback.json"))
    args = ap.parse_args()

    _, val_names = eval_lib.read_split(USS)
    names = val_names
    stems = [os.path.splitext(n)[0] for n in names]
    gt = {n: eval_lib.load_binary(os.path.join(MASK_DIR, n)) for n in names}

    P, POOL = {}, {}
    for cond in ["in_domain", "compound_B"]:
        P[cond] = {}
        for n, st in zip(names, stems):
            q = load_fg(SQUAD, st, cond)
            s = load_fg(STOCK, st, cond)
            f = load_fg(FPA, st, cond)
            if q is None or s is None:
                print("[99ai] missing", cond, st)
                continue
            P[cond][n] = dict(sq=q, st=s, fp=f, ua=unc_area(q), cm=conf_margin(q),
                              d_sq=dice(q > 0.5, gt[n]), d_st=dice(s > 0.5, gt[n]),
                              d_fp=dice(f > 0.5, gt[n]) if f is not None else float("nan"))
        n = len(P[cond])
        print("\n=== %s  (n=%d)" % (cond, n))
        print("  sq2 (ResEnc sandwich) dice %.2f   stock (stock 1000ep) dice %.2f   "
              "fpA (stock+fploss 500ep) dice %.2f"
              % (np.mean([v["d_sq"] for v in P[cond].values()]),
                 np.mean([v["d_st"] for v in P[cond].values()]),
                 np.nanmean([v["d_fp"] for v in P[cond].values()])))
        ua = np.array([v["ua"] for v in P[cond].values()], float)
        print("  unc_area_px  min %d  p25 %d  median %d  p75 %d  max %d"
              % (ua.min(), np.percentile(ua, 25), np.median(ua), np.percentile(ua, 75), ua.max()))

        # ---- POOL sweep: a on sq2
        print("\n  POOL  a*p_sq2 + (1-a)*p_stock  ->  dice / min / n<80")
        pool = []
        for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            d = []
            for n in names:
                if n not in P[cond]:
                    continue
                q = a * P[cond][n]["sq"] + (1.0 - a) * P[cond][n]["st"]
                d.append(dice(q > 0.5, gt[n]))
            d = np.array(d)
            pool.append(dict(a=a, dice=float(d.mean()), min=float(d.min()),
                             n80=int((d < 80).sum())))
            print("    a=%.1f  dice %6.2f  min %6.2f  n<80 %2d" % (a, d.mean(), d.min(), (d < 80).sum()))
        POOL[cond] = pool

    # ---- FALLBACK: gate calibrated on IN-DOMAIN only
    print("\n=== FALLBACK: threshold calibrated on in_domain, applied to both ===")
    ua_in = np.array([v["ua"] for v in P["in_domain"].values()], float)
    cm_in = np.array([v["cm"] for v in P["in_domain"].values()], float)
    sweep = []
    for tag, ref, mode in [("unc_area_px", ua_in, "hi"), ("conf_margin", cm_in, "lo")]:
        print("\n  gate on %s (in_domain range %.3f .. %.3f)" % (tag, ref.min(), ref.max()))
        print("    %-22s %-28s %-28s" % ("rule", "in_domain dice (fires)", "compound_B dice (fires)"))
        for mult in [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]:
            tau = ref.max() * mult if mode == "hi" else ref.min() / mult
            row = {"score": tag, "mult": mult, "tau": float(tau)}
            for cond in ["in_domain", "compound_B"]:
                dd, nf = [], 0
                for n in names:
                    if n not in P[cond]:
                        continue
                    v = P[cond][n]
                    fire = v["ua"] > tau if mode == "hi" else v["cm"] < tau
                    dd.append(v["d_st"] if fire else v["d_sq"])
                    nf += int(fire)
                dd = np.array(dd)
                row[cond] = dict(dice=float(dd.mean()), min=float(dd.min()),
                                 n80=int((dd < 80).sum()), fires=nf)
            print("    %-22s dice %6.2f min %6.2f n<80 %2d (%2d)  dice %6.2f min %6.2f n<80 %2d (%2d)"
                  % ("tau=%.1fx in_max" % mult,
                     row["in_domain"]["dice"], row["in_domain"]["min"], row["in_domain"]["n80"], row["in_domain"]["fires"],
                     row["compound_B"]["dice"], row["compound_B"]["min"], row["compound_B"]["n80"], row["compound_B"]["fires"]))
            sweep.append(row)
    # oracle gate ceiling
    dd = np.array([min(v["d_sq"], v["d_st"]) if False else (v["d_st"] if v["d_sq"] < v["d_st"] else v["d_sq"])
                   for v in P["compound_B"].values()])
    print("\n  ORACLE gate on compound_B (perfect knowledge): dice %.2f  n<80 %d"
          % (dd.mean(), (dd < 80).sum()))
    dd = np.array([(v["d_st"] if v["d_sq"] < v["d_st"] else v["d_sq"]) for v in P["in_domain"].values()])
    print("  ORACLE gate on in_domain:                       dice %.2f" % dd.mean())

    per = {c: {n: {k: float(v) for k, v in d.items() if k not in ("sq", "st", "fp")}
               for n, d in P[c].items()} for c in P}
    out = {"per_cond": per, "pool": POOL, "gate_sweep": sweep}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n[99ai] wrote %s" % args.out)


if __name__ == "__main__":
    main()
