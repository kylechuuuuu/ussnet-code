#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99k_weight_sweep.py

Offline weight sweep for the probability-level ensemble, evaluated with the
project metric (bdice@1px) on the fold-0 val split.

Loading the per-case softmax fields once lets every candidate weight vector be
scored without touching the GPU again, so the whole ensemble design space (all
8 members = 4 step-6 arms x {checkpoint_best, checkpoint_final}) can be mapped
in one go.  The dirs hold case-id files (USP001.npz), so names go through
step1's case_id_mapping.csv.

Reported per candidate:
  bdice1 / dice  mean over the 31 val cases
  d_mean, d_se    paired delta of bdice1 against the reference member, with the
                  standard error, plus a paired bootstrap 95% CI and the
                  sign-test p-value (n=31 is small; a +0.5 bdice shift is NOT
                  significant on its own)

usage:
  python 99k_weight_sweep.py [--ref res_best] [--boot 4000]
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
PRED = os.path.join(S6, "predictions")
MAPPING = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = load_mod("arb", os.path.join(S6, "tools", "63_arbitrate.py"))
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary",
                                   "tools", "57_shape_prior_ceiling.py"))

with open(MAPPING) as f:
    name2cid = {r["original_filename"]: r["case_id"] for r in csv.DictReader(f)}

MEMBERS = [
    ("res_best", "sq2resencw25_checkpoint_best_base_nifti"),
    ("res_final", "sq2resencw25_checkpoint_final_base_nifti"),
    ("w25_best", "sq2w25_checkpoint_best_base_nifti"),
    ("w25_final", "sq2w25_checkpoint_final_base_nifti"),
    ("w25s_best", "sq2w25s_checkpoint_best_base_nifti"),
    ("w25s_final", "sq2w25s_checkpoint_final_base_nifti"),
    ("sq2_best", "sq2_checkpoint_best_base_nifti"),
    ("sq2_final", "sq2_checkpoint_final_base_nifti"),
    ("nnunet_best", "/media/72TB_disk/personal/zk/USSnet/step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_best_nifti"),
    ("res_step075", "sq2resencw25_best_step075_base_nifti"),
    ("res_nostta", "sq2resencw25_best_nostta_base_nifti"),
    ("band_final", "sq2res_w25_band_final_base_nifti"),
    ("ctl_final", "sq2res_w25_ctl_final_base_nifti"),
]

CANDIDATES = [
    ("res_best", {"res_best": 1}),
    ("res_best+res_final", {"res_best": 1, "res_final": 1}),
    ("best4_equal", {"res_best": 1, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res2", {"res_best": 2, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res3", {"res_best": 3, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res4", {"res_best": 4, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res6", {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("nnunet_best", {"nnunet_best": 1}),
    ("best4_res5", {"res_best": 5, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res8", {"res_best": 8, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res10", {"res_best": 10, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res12", {"res_best": 12, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res16", {"res_best": 16, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("res6_plus_step075", {"res_best": 6, "res_step075": 1, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("res6_plus_nostta", {"res_best": 6, "res_nostta": 1, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("res6_plus_both", {"res_best": 6, "res_step075": 1, "res_nostta": 1, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("res8_plus_both", {"res_best": 8, "res_step075": 1, "res_nostta": 1, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("res2_plus_both", {"res_best": 2, "res_step075": 1, "res_nostta": 1, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("res_step075_only", {"res_step075": 1}),
    ("res_nostta_only", {"res_nostta": 1}),
    ("res1_step075_1", {"res_best": 1, "res_step075": 1}),
    ("all8_equal", {m: 1 for m, _ in MEMBERS}),
    ("all8_res2", {"res_best": 2, "res_final": 2, "w25_best": 1, "w25_final": 1,
                   "w25s_best": 1, "w25s_final": 1, "sq2_best": 1, "sq2_final": 1}),
    ("all8_res3", {"res_best": 3, "res_final": 3, "w25_best": 1, "w25_final": 1,
                   "w25s_best": 1, "w25s_final": 1, "sq2_best": 1, "sq2_final": 1}),
    ("all8_res6", {"res_best": 6, "res_final": 6, "w25_best": 1, "w25_final": 1,
                   "w25s_best": 1, "w25s_final": 1, "sq2_best": 1, "sq2_final": 1}),
    ("no_sq2_equal", {"res_best": 1, "res_final": 1, "w25_best": 1, "w25_final": 1,
                      "w25s_best": 1, "w25s_final": 1}),
    ("no_w25s_equal", {"res_best": 1, "res_final": 1, "w25_best": 1, "w25_final": 1,
                       "sq2_best": 1, "sq2_final": 1}),
    ("band_only", {"band_final": 1}),
    ("ctl_only", {"ctl_final": 1}),
    ("arms_equal", {"band_final": 1, "ctl_final": 1}),
    ("best4_plus_band", {"res_best": 6, "band_final": 1, "w25_best": 1,
                         "w25s_best": 1, "sq2_best": 1}),
    ("best4_plus_ctl", {"res_best": 6, "ctl_final": 1, "w25_best": 1,
                        "w25s_best": 1, "sq2_best": 1}),
    ("best4_plus_both_arms", {"res_best": 6, "band_final": 1, "ctl_final": 1,
                              "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best5_res3_arms", {"res_best": 3, "band_final": 1, "ctl_final": 1,
                         "w25_best": 1, "w25s_best": 1, "sq2_best": 1}),
    ("best4_res6_band1", {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1, "band_final": 1}),
    ("best4_res6_band2", {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1, "band_final": 2}),
    ("best4_res6_band4", {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1, "band_final": 4}),
    ("best4_res6_ctl2", {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1, "ctl_final": 2}),
    ("best4_res6_band2_ctl2", {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1,
                               "band_final": 2, "ctl_final": 2}),
]


def load_probs(path):
    with np.load(path) as d:
        p = np.asarray(d["probabilities"], dtype=np.float32)
    while p.ndim > 3 and p.shape[1] == 1:
        p = p[:, 0]
    if p.ndim == 2:
        p = np.stack([1.0 - p, p], axis=0)
    return p


def sign_test_p(d):
    """two-sided sign test, normal approximation"""
    d = np.asarray(d)
    n = int((np.abs(d) > 1e-9).sum())
    k = int((d > 0).sum())
    if n == 0:
        return 1.0
    from math import erf, sqrt
    z = (k - n / 2.0) / np.sqrt(n / 4.0)
    return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="res_best")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, val = arb.read_split(USS)
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}

    have = [(l, d) for l, d in MEMBERS
            if os.path.isfile(os.path.join(PRED, d, name2cid[val[0]] + ".npz"))]
    print("members present: %s" % [l for l, _ in have])
    if len(have) < 1:
        raise SystemExit("no members found")

    P = {}
    for lab, d in have:
        P[lab] = {n: load_probs(os.path.join(PRED, d, name2cid[n] + ".npz"))
                  for n in val}

    def score(weights):
        bd, dc = [], []
        for n in val:
            acc = None
            for lab, w in weights.items():
                q = P[lab][n]
                acc = q * w if acc is None else acc + q * w
            acc = acc / float(sum(weights.values()))
            pred = s57.largest_cc(acc[1] >= 0.5)
            bd.append(s57.bdice(pred, gts[n], tol=1))
            dc.append(s57.dice(pred, gts[n]))
        return np.array(bd), np.array(dc)

    rows = []
    for name, w in CANDIDATES:
        if any(lab not in P for lab in w):
            continue
        bd, dc = score(w)
        rows.append((name, bd, dc))

    ref_bd = None
    for name, bd, dc in rows:
        if name == args.ref:
            ref_bd = bd
    if ref_bd is None:
        bd, dc = score({args.ref: 1})
        rows.insert(0, (args.ref, bd, dc))
        ref_bd = bd

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(val))
    boot = [rng.choice(idx, size=len(idx), replace=True) for _ in range(args.boot)]

    print("\n%-22s %8s %8s %8s %8s %13s %8s %10s" %
          ("candidate", "bdice1", "dice", "d_mean", "d_se", "ci95", "sign_p",
           "win/loss"))
    out = []
    for name, bd, dc in rows:
        d = bd - ref_bd
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        means = np.array([d[b].mean() for b in boot])
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        out.append(dict(name=name, bdice1=float(bd.mean()), dice=float(dc.mean()),
                        dmean=float(d.mean()), dse=se, ci=[lo, hi],
                        sign_p=sign_test_p(d),
                        win=int((d > 0.01).sum()), loss=int((d < -0.01).sum())))
    for r in sorted(out, key=lambda r: -r["bdice1"]):
        print("%-22s %8.3f %8.3f %+8.3f %8.3f [%+.2f,%+.2f] %8.3f %5d/%d"
              % (r["name"], r["bdice1"], r["dice"], r["dmean"], r["dse"],
                 r["ci"][0], r["ci"][1], r["sign_p"], r["win"], r["loss"]))

    res = os.path.join(USS, "step9_boundary_mechanisms", "results",
                       "ensemble_sweep_val.json")
    os.makedirs(os.path.dirname(res), exist_ok=True)
    with open(res, "w") as f:
        json.dump({"ref": args.ref, "rows": out, "members": [l for l, _ in have]},
                  f, indent=2)
    print("\nwrote %s" % res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
