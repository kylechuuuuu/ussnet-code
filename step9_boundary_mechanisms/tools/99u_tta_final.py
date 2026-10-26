#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99u_tta_final.py

Final shoot-out between the shipped delivery (4-model probability ensemble +
spike-removal post-processing = ens_K) and the appearance-domain TTA recipes
found by 99r/99s.  Everything is scored on the 31 fold-0 val cases with the
project boundary metric (bdice@1px) and Dice, raw and with the shipped
post-processing, and validated with a 200-round 15/16 split (fixed recipes, so
this measures the recipe, not a val-fitted choice).

The TTA recipes are taken from step8's own pre-registered GROUPS (shift_lib.py):
"artifact" = jpeg_30 + jpeg_15, "nonlinear" = gamma_060/080/125, and the
reference condition in_domain - no condition is picked by its val score.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

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

ART = ["in_domain", "jpeg_30", "jpeg_15"]
NL = ["in_domain", "gamma_060", "gamma_080", "gamma_125"]
MILD = ["in_domain", "gamma_060", "gamma_080", "gamma_125", "jpeg_15", "jpeg_30",
        "blur_20", "tgc_p60", "tgc_m60", "shadow", "speckle_15",
        "affine_gain_p25", "affine_gain_m25"]
ALL17 = None  # filled from shift_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sh = load_mod("sh", os.path.join(USS, "step8_domain_shift", "tools", "shift_lib.py"))
    global ALL17
    ALL17 = list(sh.SHIFTS.keys())

    _t, val = arb.read_split(USS)
    stems = [arb.stem_of(n) for n in val]
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}
    with open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")) as f:
        n2c = {r["original_filename"]: r["case_id"] for r in csv.DictReader(f)}

    print("loading step6 members ...")
    B = {}
    for lab, d in (("res", "sq2resencw25_checkpoint_best_base_nifti"),
                   ("w25", "sq2w25_checkpoint_best_base_nifti"),
                   ("w25s", "sq2w25s_checkpoint_best_base_nifti"),
                   ("sq2", "sq2_checkpoint_best_base_nifti")):
        B[lab] = {st: k99.load_probs(os.path.join(S6, "predictions", d, n2c[st + ".png"] + ".npz"))
                  for st in stems}
    print("loading step8 conditions ...")
    need = sorted({c for rec in (ART, NL, MILD, list(sh.SHIFTS.keys())) for c in rec})
    S = {}
    for c in need:
        d = os.path.join(S8, "sq2shift_%s_base_nifti" % c)
        if os.path.isdir(d):
            S[c] = {st: k99.load_probs(os.path.join(d, st + ".npz")) for st in stems}
    print("conditions loaded: %d" % len(S))

    def tta(conds):
        cs = [c for c in conds if c in S]

        def fn(st):
            return np.mean([S[c][st] for c in cs], axis=0)
        return fn

    def ens(tta_fn):
        def fn(st):
            return (tta_fn(st) * 6.0 + B["w25"][st] + B["w25s"][st] + B["sq2"][st]) / 9.0
        return fn

    CANDS = {
        "res_clean": lambda st: B["res"][st],
        "ens4": lambda st: (B["res"][st] * 6 + B["w25"][st] + B["w25s"][st] + B["sq2"][st]) / 9.0,
        "tta_art": tta(ART),
        "tta_nl": tta(NL),
        "tta_mild": tta(MILD),
        "tta_all17": tta(ALL17),
        "tta_art_x_ens4": ens(tta(ART)),
        "tta_nl_x_ens4": ens(tta(NL)),
        "tta_mild_x_ens4": ens(tta(MILD)),
        "tta_all17_x_ens4": ens(tta(ALL17)),
    }

    def pp(m):
        return ndi.binary_fill_holes(s57.largest_cc(
            ndi.binary_opening(m, np.ones((5, 5)))))

    def mask_of(fn, st, post):
        q = fn(st)[1]
        if post:
            m = ndi.median_filter(q, size=3) >= 0.5
            m = pp(m)
        else:
            m = s57.largest_cc(q >= 0.5)
        return m

    def score(fn, post):
        bd, dc = [], []
        for n, st in zip(val, stems):
            m = mask_of(fn, st, post)
            bd.append(s57.bdice(m, gts[n], tol=1))
            dc.append(s57.dice(m, gts[n]))
        return np.array(bd), np.array(dc)

    res_bd, _ = score(CANDS["res_clean"], False)
    ref_bd, ref_dc = score(CANDS["ens4"], True)   # == shipped ens_K

    rows = []
    for name, fn in CANDS.items():
        for post in (False, True):
            bd, dc = score(fn, post)
            d = bd - ref_bd
            rows.append(dict(name=name + ("+pp" if post else ""), bdice1=float(bd.mean()),
                             dice=float(dc.mean()), dmean=float(d.mean()),
                             dse=float(d.std(ddof=1) / np.sqrt(len(d))),
                             win=int((d > 0.01).sum()), loss=int((d < -0.01).sum()),
                             dvs_res=float((bd - res_bd).mean()), bd=bd))
    rows.sort(key=lambda r: -r["bdice1"])
    print("\n%-22s %8s %8s %8s %8s %10s" %
          ("candidate", "bdice1", "dice", "d vs ensK", "d vs res", "win/loss"))
    for r in rows:
        print("%-22s %8.3f %8.4f %+8.3f %+8.3f %5d/%d" %
              (r["name"], r["bdice1"], r["dice"], r["dmean"], r["dvs_res"],
               r["win"], r["loss"]))

    # 200-round 15/16 split: fixed recipes, no selection inside
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(val))
    splits = [rng.choice(idx, size=15, replace=False) for _ in range(args.rounds)]
    print("\nhold-out (%d rounds of 15/16, delta of bdice1 vs shipped ens_K):" % args.rounds)
    for r in rows:
        if r["name"] in ("ens4+pp", "res_clean", "res_clean+pp"):
            continue
        dm, wins = [], 0
        for half in splits:
            rest = np.setdiff1d(idx, half)
            dd = float(r["bd"][rest].mean() - ref_bd[rest].mean())
            dm.append(dd)
            wins += dd > 0
        print("  %-22s mean %+7.3f  SE %.3f  win_frac %.3f" %
              (r["name"], float(np.mean(dm)), float(np.std(dm) / np.sqrt(len(dm))),
               wins / args.rounds))

    out = [{k: v for k, v in r.items() if k != "bd"} for r in rows]
    p = os.path.join(S9, "results", "tta_final_val.json")
    with open(p, "w") as f:
        json.dump({"ref": "ens4+pp (shipped ens_K)", "res_bdice1": float(res_bd.mean()),
                   "rows": out}, f, indent=2)
    print("\nwrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
