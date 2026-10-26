#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99t_tta_mechanism.py

WHY does a mild appearance perturbation move bdice@1px by more than the whole
4-model ensemble?  For each candidate field this prints, over the 31 val cases:

  area_err%   signed relative area error (pred - gt)/gt * 100   (+ = too big)
  bpre/bprec  boundary precision / recall at 1 px
  dice, bdice1

and then the per-case relation between the change in signed area error and the
change in bdice1 (Pearson r over the 31 cases) - if TTA helps because the clean
model over-segments, the two must be strongly anti-correlated.

usage:  python tools/99t_tta_mechanism.py
"""
import csv
import importlib.util
import json
import os

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

ST = np.ones((3, 3), bool)


def bnd_pr(pred, gt, tol=1):
    sp = s57.surface(pred)
    sg = s57.surface(gt)
    dp = ndi.binary_dilation(sp, np.ones((2 * tol + 1,) * 2, bool))
    dg = ndi.binary_dilation(sg, np.ones((2 * tol + 1,) * 2, bool))
    pre = float((sp & dg).sum()) / max(int(sp.sum()), 1)
    rec = float((sg & dp).sum()) / max(int(sg.sum()), 1)
    return 100 * pre, 100 * rec


def main():
    _t, val = arb.read_split(USS)
    stems = [arb.stem_of(n) for n in val]
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}
    with open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")) as f:
        n2c = {r["original_filename"]: r["case_id"] for r in csv.DictReader(f)}

    def s6(d):
        return {st: k99.load_probs(os.path.join(S6, "predictions", d, n2c[st + ".png"] + ".npz"))
                for st in stems}

    def s8(c):
        return {st: k99.load_probs(os.path.join(S8, "sq2shift_%s_base_nifti" % c, st + ".npz"))
                for st in stems}

    res = s6("sq2resencw25_checkpoint_best_base_nifti")
    w25 = s6("sq2w25_checkpoint_best_base_nifti")
    w25s = s6("sq2w25s_checkpoint_best_base_nifti")
    sq2 = s6("sq2_checkpoint_best_base_nifti")
    g125 = s8("gamma_125")
    j30 = s8("jpeg_30")
    ind = s8("in_domain")
    g080 = s8("gamma_080")
    g060 = s8("gamma_060")
    tgc = s8("tgc_p60")

    cands = {
        "res_clean": lambda st: res[st],
        "gamma125": lambda st: g125[st],
        "jpeg30": lambda st: j30[st],
        "T2_(ind+g125)": lambda st: (ind[st] + g125[st]) / 2.0,
        "T3_(ind+g125+j30)": lambda st: (ind[st] + g125[st] + j30[st]) / 3.0,
        "T6": lambda st: (ind[st] + g125[st] + j30[st] + g080[st]
                          + tgc[st] + g060[st]) / 6.0,
        "ens4_clean": lambda st: (res[st] * 6 + w25[st] + w25s[st] + sq2[st]) / 9.0,
        "ens4_T3": lambda st: ((ind[st] + g125[st] + j30[st]) / 3.0 * 6
                               + w25[st] + w25s[st] + sq2[st]) / 9.0,
    }

    rows = {k: dict(area=[], pre=[], rec=[], dice=[], bd=[]) for k in cands}
    for n, st in zip(val, stems):
        gt = gts[n]
        gtp = int(gt.sum())
        for k, fn in cands.items():
            m = s57.largest_cc(fn(st)[1] >= 0.5)
            p, r = bnd_pr(m, gt)
            rows[k]["area"].append(100.0 * (int(m.sum()) - gtp) / gtp)
            rows[k]["pre"].append(p)
            rows[k]["rec"].append(r)
            rows[k]["dice"].append(s57.dice(m, gt))
            rows[k]["bd"].append(s57.bdice(m, gt, tol=1))

    print("%-18s %9s %8s %8s %8s %8s" %
          ("candidate", "area_err%", "bd_prec", "bd_rec", "dice", "bdice1"))
    out = {}
    for k in cands:
        r = rows[k]
        out[k] = {kk: float(np.mean(vv)) for kk, vv in r.items()}
        print("%-18s %+9.3f %8.2f %8.2f %8.4f %8.3f" %
              (k, out[k]["area"], out[k]["pre"], out[k]["rec"],
               out[k]["dice"], out[k]["bd"]))

    print("\nper-case coupling (vs res_clean, n=%d):" % len(val))
    base_area = np.array(rows["res_clean"]["area"])
    base_bd = np.array(rows["res_clean"]["bd"])
    for k in cands:
        if k == "res_clean":
            continue
        da = np.array(rows[k]["area"]) - base_area
        db = np.array(rows[k]["bd"]) - base_bd
        r = float(np.corrcoef(da, db)[0, 1]) if da.std() > 0 and db.std() > 0 else float("nan")
        print("  %-18s d_area%+7.3f  d_bdice%+7.3f  r(d_area,d_bdice)=%+.3f" %
              (k, da.mean(), db.mean(), r))

    p = os.path.join(S9, "results", "tta_mechanism_val.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % p)


if __name__ == "__main__":
    raise SystemExit(main())
