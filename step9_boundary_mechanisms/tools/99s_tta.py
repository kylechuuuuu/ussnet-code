#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99s_tta.py

Image-level test-time augmentation (TTA) over the step-8 acquisition-shift
conditions, plus its combination with the shipped probability ensemble.

Finding that motivates it (99r --phase singles): the SAME checkpoint
(SQ2res-w25, checkpoint_best, base head) scored on the val split gives
bdice@1px 35.329 on the clean image, 35.807 on the gamma-1.25 image and 35.751
on the jpeg-30 image - i.e. a mild appearance perturbation alone moves the
metric by more than the whole 4-model ensemble (+0.42).  step8 already has the
whole shifted prediction set on disk (31 val cases, deterministic operators in
step8_domain_shift/tools/shift_lib.py, reproducible through 80/81).

This tool answers two questions with paired statistics against res_best:
  1. how much of that gain survives when several conditions are averaged
     (a normal TTA recipe) rather than picked individually;
  2. does TTA stack with the 4-model ensemble (shipped config best4_res6).

usage:
  python tools/99s_tta.py [--boot 4000]
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from math import erf, sqrt

import numpy as np

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
S8 = os.path.join(USS, "step8_domain_shift", "predictions")
S9 = os.path.join(USS, "step9_boundary_mechanisms")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


k99 = load_mod("k99", os.path.join(S9, "tools", "99k_weight_sweep.py"))
arb = load_mod("arb", os.path.join(S6, "tools", "63_arbitrate.py"))
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary",
                                   "tools", "57_shape_prior_ceiling.py"))
sh = load_mod("sh", os.path.join(USS, "step8_domain_shift", "tools", "shift_lib.py"))

COND = list(sh.SHIFTS.keys())

# TTA recipes over step-8 conditions.  Tmild = every condition whose own
# single-model Dice was still >= ~96 in the step-8 robustness table.
RECIPES = [
    ("T1_in_domain", ["in_domain"]),
    ("T2_g125", ["in_domain", "gamma_125"]),
    ("T3_g125_j30", ["in_domain", "gamma_125", "jpeg_30"]),
    ("T4_g125_j30_g80", ["in_domain", "gamma_125", "jpeg_30", "gamma_080"]),
    ("T5_+tgcP60", ["in_domain", "gamma_125", "jpeg_30", "gamma_080", "tgc_p60"]),
    ("T6_+g60", ["in_domain", "gamma_125", "jpeg_30", "gamma_080", "tgc_p60", "gamma_060"]),
    ("Tnl_all", ["in_domain", "gamma_060", "gamma_080", "gamma_125"]),
    ("Tart_all", ["in_domain", "jpeg_15", "jpeg_30"]),
    ("Tb1_nl_art", ["in_domain", "gamma_060", "gamma_080", "gamma_125", "jpeg_15", "jpeg_30"]),
    ("Tmild", ["in_domain", "gamma_060", "gamma_080", "gamma_125",
               "jpeg_15", "jpeg_30", "blur_20", "tgc_p60", "tgc_m60",
               "shadow", "speckle_15", "affine_gain_p25", "affine_gain_m25"]),
    ("Tall17", COND),
]
ENS_RECIPES = ("T1_in_domain", "T3_g125_j30", "T6_+g60", "Tmild", "Tall17")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _t, val = arb.read_split(USS)
    stems = [arb.stem_of(n) for n in val]
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}

    with open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")) as f:
        n2c = {r["original_filename"]: r["case_id"] for r in csv.DictReader(f)}

    print("loading step6 members ...")
    BASE = {}
    for lab, d in (("res", "sq2resencw25_checkpoint_best_base_nifti"),
                   ("w25", "sq2w25_checkpoint_best_base_nifti"),
                   ("w25s", "sq2w25s_checkpoint_best_base_nifti"),
                   ("sq2", "sq2_checkpoint_best_base_nifti")):
        BASE[lab] = {st: k99.load_probs(os.path.join(S6, "predictions", d,
                                                     n2c[st + ".png"] + ".npz"))
                     for st in stems}

    print("loading step8 shift families ...")
    F = {}
    for fam in ("sq2shift", "sq23h", "sq2ttn"):
        for c in COND:
            d = os.path.join(S8, "%s_%s_base_nifti" % (fam, c))
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, stems[0] + ".npz")):
                F[(fam, c)] = {st: k99.load_probs(os.path.join(d, st + ".npz"))
                               for st in stems}
    print("available (family, condition) pairs: %d" % len(F))

    def score(acc_fn):
        bd, dc = [], []
        for i, n in enumerate(val):
            q = acc_fn(i, stems[i])
            pred = s57.largest_cc(q[1] >= 0.5)
            bd.append(s57.bdice(pred, gts[n], tol=1))
            dc.append(s57.dice(pred, gts[n]))
        return np.array(bd), np.array(dc)

    res_bd, _ = score(lambda i, st: BASE["res"][st])

    def tta(fams, conds):
        pairs = [(f, c) for f in fams for c in conds if (f, c) in F]
        if not pairs:
            return None, []
        return (lambda i, st: np.mean([F[p][st] for p in pairs], axis=0)), pairs

    def ens(tta_fn, wt=1.0):
        def fn(i, st):
            return (tta_fn(i, st) * (6.0 * wt) + BASE["w25"][st] + BASE["w25s"][st]
                    + BASE["sq2"][st]) / (6.0 * wt + 3.0)
        return fn

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(val))
    boot = [rng.choice(idx, size=len(idx), replace=True) for _ in range(args.boot)]

    def sign_p(d):
        n = int((np.abs(d) > 1e-9).sum())
        k = int((d > 0).sum())
        if n == 0:
            return 1.0
        z = (k - n / 2.0) / sqrt(n / 4.0)
        return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))

    rows = []
    for fam in ("sq2shift", "sq23h", "sq2ttn"):
        for c in COND:
            if (fam, c) not in F:
                continue
            fn, _ = tta([fam], [c])
            bd, dc = score(fn)
            rows.append((fam + ":" + c, bd, dc))
    for name, conds in RECIPES:
        for fams in (["sq2shift"], ["sq2shift", "sq23h"]):
            fn, pairs = tta(fams, conds)
            if fn is None:
                continue
            tag = name if fams == ["sq2shift"] else name + "@2fam"
            bd, dc = score(fn)
            rows.append((tag, bd, dc))
            if name in ENS_RECIPES:
                bd2, dc2 = score(ens(fn))
                rows.append((tag + "_x_ens4", bd2, dc2))
    bd_e, dc_e = score(ens(lambda i, st: BASE["res"][st]))
    rows.append(("ens4_res6_ref", bd_e, dc_e))

    print("\n%-30s %8s %8s %8s %13s %8s %8s" %
          ("candidate", "bdice1", "dice", "d_mean", "ci95", "sign_p", "win/loss"))
    out = []
    for name, bd, dc in rows:
        d = bd - res_bd
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        means = np.array([d[b].mean() for b in boot])
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        out.append(dict(name=name, bdice1=float(bd.mean()), dice=float(dc.mean()),
                        dmean=float(d.mean()), dse=se, ci=[lo, hi],
                        sign_p=sign_p(d), win=int((d > 0.01).sum()),
                        loss=int((d < -0.01).sum())))
    for r in sorted(out, key=lambda r: -r["bdice1"]):
        print("%-30s %8.3f %8.3f %+8.3f [%+.2f,%+.2f] %8.4f %5d/%d"
              % (r["name"], r["bdice1"], r["dice"], r["dmean"],
                 r["ci"][0], r["ci"][1], r["sign_p"], r["win"], r["loss"]))

    res = os.path.join(S9, "results", "tta_val.json")
    with open(res, "w") as f:
        json.dump({"ref": "res_best", "res_bdice1": float(res_bd.mean()),
                   "conditions_available": sorted("%s:%s" % k for k in F),
                   "rows": out}, f, indent=2)
    print("\nwrote %s" % res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
