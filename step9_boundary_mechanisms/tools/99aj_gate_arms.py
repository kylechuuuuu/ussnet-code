#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99aj_gate_arms.py

Which robust arm should the GT-free gate route to, and can the sandwich
framework itself supply it?

gate (pre-registered, GT-free, one-class):
    tau = min conf_margin over the IN-DOMAIN val cases of the shipped model
    if conf_margin(p_sq2) < tau:  answer with the fallback arm's prediction
    else:                         answer with p_sq2

conf_margin(p) = mean |2p-1| over the 0.5-surface of p  (base head only,
no ground truth, no knowledge of compound_B).

writes step9_boundary_mechanisms/results/gate_arms.json
"""
import argparse
import json
import os
import sys

import numpy as np

USS = "/media/72TB_disk/personal/zk/USSnet"
S8 = os.path.join(USS, "step8_domain_shift")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))
import eval_lib  # noqa: E402
import importlib  # noqa: E402
s57 = importlib.import_module("57_shape_prior_ceiling")

PRED = os.path.join(S8, "predictions")
MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
CONDS = ["in_domain", "compound_B"]
# label -> (tag, subdir suffix)
ARMS = {
    "stock1000": ("stockshift", ""),
    "fpA_500": ("fpA", ""),
    "s200": ("s200", ""),
    "sq2a_stock_sandwich": ("sq2a", "base"),
    "sq2w25": ("sq2w25", "base"),
    "sq2": ("sq2", "base"),
}


def load_fg(tag, stem, cond, suffix):
    for pat in ("%s_%s%s_nifti" % (tag, cond, "_" + suffix if suffix else ""),
                "%s_%s_nifti" % (tag, cond)):
        fp = os.path.join(PRED, pat, stem + ".npz")
        if os.path.exists(fp):
            with np.load(fp) as d:
                p = np.asarray(d["probabilities"], dtype=np.float32)
            while p.ndim > 3 and p.shape[1] == 1:
                p = p[:, 0]
            if p.ndim == 3 and p.shape[0] == 2:
                p = p[1]
            return p
    return None


def conf_margin(p):
    m = p > 0.5
    if not m.any():
        return 0.0
    surf = m & ~eval_lib.ndimage.binary_erosion(m, np.ones((3, 3), bool), border_value=0)
    return float(np.abs(2.0 * p[surf] - 1.0).mean()) if surf.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(S9, "results", "gate_arms.json"))
    args = ap.parse_args()
    _, val_names = eval_lib.read_split(USS)
    stems = [os.path.splitext(n)[0] for n in val_names]
    gt = {n: eval_lib.load_binary(os.path.join(MASK_DIR, n)) for n in val_names}

    D = {}
    for cond in CONDS:
        D[cond] = {}
        for n, st in zip(val_names, stems):
            q = load_fg("sq2shift", st, cond, "base")
            if q is None:
                print("[99aj] missing sq2shift", cond, st)
                continue
            rec = dict(sq=q, cm=conf_margin(q), d_sq=float(s57.dice(q > 0.5, gt[n])))
            for lbl, (tag, suf) in ARMS.items():
                a = load_fg(tag, st, cond, suf)
                rec[lbl] = a
                rec["d_" + lbl] = float(s57.dice(a > 0.5, gt[n])) if a is not None else float("nan")
            D[cond][n] = rec

    cm_in = np.array([v["cm"] for v in D["in_domain"].values()], float)
    tau = float(cm_in.min())
    print("[99aj] tau = in_domain min conf_margin = %.4f  (in_domain range %.3f..%.3f)"
          % (tau, cm_in.min(), cm_in.max()))
    print("       compound_B fires: %d/31\n" % sum(1 for v in D["compound_B"].values() if v["cm"] < tau))

    res = {"tau": tau, "arms": {}}
    hdr = "%-22s %-26s %-26s" % ("fallback arm", "in_domain (always | gated)", "compound_B (always | gated)")
    print(hdr)
    print("-" * len(hdr))
    for lbl in ARMS:
        row = {}
        for cond in CONDS:
            al = np.array([v["d_" + lbl] for v in D[cond].values()], float)
            ga = np.array([(v["d_" + lbl] if v["cm"] < tau else v["d_sq"]) for v in D[cond].values()], float)
            row[cond] = dict(always_dice=float(al.mean()), always_n80=int((al < 80).sum()),
                             gated_dice=float(ga.mean()), gated_n80=int((ga < 80).sum()),
                             gated_min=float(ga.min()))
        res["arms"][lbl] = row
        print("%-22s %6.2f n<80 %2d | %6.2f n<80 %2d   %6.2f n<80 %2d | %6.2f n<80 %2d"
              % (lbl, row["in_domain"]["always_dice"], row["in_domain"]["always_n80"],
                 row["in_domain"]["gated_dice"], row["in_domain"]["gated_n80"],
                 row["compound_B"]["always_dice"], row["compound_B"]["always_n80"],
                 row["compound_B"]["gated_dice"], row["compound_B"]["gated_n80"]))

    # pool of sq2 + best arm, gated
    print("\npooled fallback (gate -> mean of sq2 and stock1000, weight on sq2 = a):")
    pres = []
    for a in [0.0, 0.2, 0.3, 0.4, 0.5]:
        row = {}
        for cond in CONDS:
            gd, ad = [], []
            for v in D[cond].values():
                q = a * v["sq"] + (1.0 - a) * v["stock1000"]
                ad.append(float(s57.dice(q > 0.5, gt[[k for k, w in D[cond].items() if w is v][0]])))
            row[cond] = None
        pres.append(row)
    # (skip: per-case gt lookup by identity is awkward; report the plain pool instead)
    for a in [0.0, 0.2, 0.3, 0.4, 0.5]:
        line = []
        for cond in CONDS:
            dd = []
            for n, v in D[cond].items():
                q = a * v["sq"] + (1.0 - a) * v["stock1000"]
                gated = (1.0 - a) * v["stock1000"] + a * v["sq"]
                use = gated if v["cm"] < tau else v["sq"]
                dd.append(float(s57.dice(use > 0.5, gt[n])))
            dd = np.array(dd)
            line.append("%6.2f (n<80 %2d)" % (dd.mean(), (dd < 80).sum()))
        print("  a=%.1f  in_domain %s   compound_B %s" % (a, line[0], line[1]))

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n[99aj] wrote %s" % args.out)


if __name__ == "__main__":
    main()
