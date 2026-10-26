#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99m_ens_postproc.py

Does any GT-free post-processing on the ENSEMBLE probability field still pay?

The ensemble `best4_res6` is the current best artifact (bdice@1px 35.753 /
Dice 96.394).  §4/§8 already showed post-processing on the SINGLE model is
worth <= +0.2 bdice.  This re-asks the question where it now matters: on the
ensemble field, with the project metric.

Every variant here is deterministic and needs no ground truth; the GT is used
only to score.  Reported: bdice@1px, Dice, paired delta vs `raw` (sign test).

usage: python tools/99m_ens_postproc.py
"""
import csv
import importlib.util
import json
import os

import numpy as np
from scipy import ndimage as ndi

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
PRED = os.path.join(S6, "predictions")
MAPPING = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")
OUT = os.path.join(USS, "step9_boundary_mechanisms", "results",
                   "ens_postproc_val.json")

# the winning weight vector from 99k
WEIGHTS = {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}
DIRS = {
    "res_best": "sq2resencw25_checkpoint_best_base_nifti",
    "w25_best": "sq2w25_checkpoint_best_base_nifti",
    "w25s_best": "sq2w25s_checkpoint_best_base_nifti",
    "sq2_best": "sq2_checkpoint_best_base_nifti",
}


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


def load_probs(path):
    with np.load(path) as d:
        p = np.asarray(d["probabilities"], dtype=np.float32)
    while p.ndim > 3 and p.shape[1] == 1:
        p = p[:, 0]
    if p.ndim == 2:
        p = np.stack([1.0 - p, p], axis=0)
    return p


def sign_p(d):
    from math import erf, sqrt
    d = np.asarray(d)
    n = int((np.abs(d) > 1e-9).sum())
    if n == 0:
        return 1.0
    k = int((d > 0).sum())
    z = (k - n / 2.0) / np.sqrt(n / 4.0)
    return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))


# --------------------------------------------------------------------------
# variants: q is the ensemble foreground probability, thr is the decision level
def v_raw(q, thr):
    return s57.largest_cc(q >= thr)


def v_thr_lo(q, thr):
    return s57.largest_cc(q >= thr - 0.05)


def v_thr_hi(q, thr):
    return s57.largest_cc(q >= thr + 0.05)


def v_fill(q, thr):
    return ndi.binary_fill_holes(s57.largest_cc(q >= thr))


def v_open1(q, thr):
    return s57.largest_cc(ndi.binary_opening(q >= thr, np.ones((3, 3))))


def v_close1(q, thr):
    return s57.largest_cc(ndi.binary_closing(q >= thr, np.ones((3, 3))))


def v_open2(q, thr):
    return s57.largest_cc(ndi.binary_opening(q >= thr, np.ones((5, 5))))


def v_close2(q, thr):
    return s57.largest_cc(ndi.binary_closing(q >= thr, np.ones((5, 5))))


def v_med3(q, thr):
    return s57.largest_cc(ndi.median_filter(q, size=3) >= thr)


def v_med5(q, thr):
    return s57.largest_cc(ndi.median_filter(q, size=5) >= thr)


def v_gauss1(q, thr):
    return s57.largest_cc(ndi.gaussian_filter(q, 1.0) >= thr)


def v_gauss2(q, thr):
    return s57.largest_cc(ndi.gaussian_filter(q, 2.0) >= thr)


def v_gauss1_fill(q, thr):
    return ndi.binary_fill_holes(s57.largest_cc(ndi.gaussian_filter(q, 1.0) >= thr))


def v_med3_open2(q, thr):
    return s57.largest_cc(ndi.binary_opening(
        ndi.median_filter(q, size=3) >= thr, np.ones((5, 5))))


def v_open2_fill(q, thr):
    return ndi.binary_fill_holes(
        s57.largest_cc(ndi.binary_opening(q >= thr, np.ones((5, 5)))))


def v_open2_close1(q, thr):
    m = ndi.binary_opening(q >= thr, np.ones((5, 5)))
    return s57.largest_cc(ndi.binary_closing(m, np.ones((3, 3))))


def v_open2_close1_fill(q, thr):
    m = ndi.binary_opening(q >= thr, np.ones((5, 5)))
    m = ndi.binary_closing(m, np.ones((3, 3)))
    return ndi.binary_fill_holes(s57.largest_cc(m))


def v_med3_open2_fill(q, thr):
    m = ndi.binary_opening(ndi.median_filter(q, size=3) >= thr,
                           np.ones((5, 5)))
    return ndi.binary_fill_holes(s57.largest_cc(m))


VARIANTS = [
    ("raw", v_raw), ("thr-0.05", v_thr_lo), ("thr+0.05", v_thr_hi),
    ("fill_holes", v_fill), ("open1", v_open1), ("close1", v_close1),
    ("open2", v_open2), ("close2", v_close2), ("median3", v_med3),
    ("median5", v_med5), ("gauss1", v_gauss1), ("gauss2", v_gauss2),
    ("gauss1+fill", v_gauss1_fill),
    ("median3+open2", v_med3_open2),
    ("open2+fill", v_open2_fill),
    ("open2+close1", v_open2_close1),
    ("open2+close1+fill", v_open2_close1_fill),
    ("median3+open2+fill", v_med3_open2_fill),
]


def main():
    _, val = arb.read_split(USS)
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}
    wsum = float(sum(WEIGHTS.values()))

    accs = {}
    for n in val:
        a = None
        for lab, w in WEIGHTS.items():
            q = load_probs(os.path.join(PRED, DIRS[lab], name2cid[n] + ".npz"))
            a = q * w if a is None else a + q * w
        accs[n] = a / wsum

    res = {}
    for name, fn in VARIANTS:
        bd, dc = [], []
        for n in val:
            pred = fn(accs[n][1], 0.5)
            bd.append(s57.bdice(pred, gts[n], tol=1))
            dc.append(s57.dice(pred, gts[n]))
        res[name] = (np.array(bd), np.array(dc))

    ref = res["raw"][0]
    print(f"ensemble best4_res6, val n={len(val)}")
    print(f"{'variant':12} {'bdice1':>8} {'dice':>8} {'delta':>8} "
          f"{'sign p':>8} {'win/loss':>10}")
    rows = []
    for name, _ in VARIANTS:
        bd, dc = res[name]
        d = bd - ref
        rows.append(dict(name=name, bdice1=float(bd.mean()), dice=float(dc.mean()),
                         delta=float(d.mean()), p=sign_p(d),
                         win=int((d > 0).sum()), loss=int((d < 0).sum())))
        print(f"{name:12} {bd.mean():8.4f} {dc.mean():8.4f} {d.mean():+8.4f} "
              f"{sign_p(d):8.4f} {int((d > 0).sum()):5d}/{int((d < 0).sum()):<4d}")

    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
