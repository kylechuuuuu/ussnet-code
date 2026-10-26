#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99v_deliver.py

Final delivery tool.  Builds the finalist configs on the 31 fold-0 val cases,
prints the pairwise paired statistics (bootstrap CI95 + exact sign test) and can
emit the winner as step-1-convention 0/255 PNGs so the project's own evaluator
(step2/scripts/10_evaluate_predictions.py) can score it.

configs
  nnunet+pp        plain nnU-Net 2d fold-0 checkpoint_best + shipped postproc
  res+pp           step6 SQ2res-w25 base + postproc
  ens4+pp          shipped delivery ens_K: (res*6 + w25 + w25s + sq2)/9 + postproc
  tta_art+pp       SQ2res-w25 with TTA over step8's "artifact" group
                   (in_domain, jpeg_30, jpeg_15) + postproc
  tta_art_x_ens4+pp  the same TTA view applied to the res member of ens4 + postproc

usage
  python tools/99v_deliver.py [--emit tta_art+pp] [--boot 4000]
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from math import erf, sqrt

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
S8 = os.path.join(USS, "step8_domain_shift", "predictions")
S1 = os.path.join(USS, "step1_nnunet_baseline", "predictions")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default="")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    _t, val = arb.read_split(USS)
    stems = [arb.stem_of(n) for n in val]
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}
    with open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")) as f:
        n2c = {r["original_filename"]: r["case_id"] for r in csv.DictReader(f)}

    print("loading ...")
    B = {}
    for lab, d in (("res", "sq2resencw25_checkpoint_best_base_nifti"),
                   ("w25", "sq2w25_checkpoint_best_base_nifti"),
                   ("w25s", "sq2w25s_checkpoint_best_base_nifti"),
                   ("sq2", "sq2_checkpoint_best_base_nifti")):
        B[lab] = {st: k99.load_probs(os.path.join(S6, "predictions", d, n2c[st + ".png"] + ".npz"))
                  for st in stems}
    B["nnunet"] = {st: k99.load_probs(os.path.join(S1, "nnunet_2d_fold0_checkpoint_best_nifti",
                                                   n2c[st + ".png"] + ".npz")) for st in stems}
    S = {c: {st: k99.load_probs(os.path.join(S8, "sq2shift_%s_base_nifti" % c, st + ".npz"))
             for st in stems} for c in ART}

    def fg_tta(st):
        return np.mean([S[c][st] for c in ART], axis=0)

    def fg_ens4(st):
        return (B["res"][st] * 6 + B["w25"][st] + B["w25s"][st] + B["sq2"][st]) / 9.0

    def fg_tta_ens(st):
        return (fg_tta(st) * 6 + B["w25"][st] + B["w25s"][st] + B["sq2"][st]) / 9.0

    FIELDS = {
        "nnunet": lambda st: B["nnunet"][st],
        "res": lambda st: B["res"][st],
        "ens4": fg_ens4,
        "tta_art": fg_tta,
        "tta_art_x_ens4": fg_tta_ens,
    }
    CONFIGS = [k + "+pp" for k in FIELDS]

    def make(cfg, st):
        base, post = cfg.split("+")
        q = FIELDS[base](st)[1]
        if post == "pp":
            m = ndi.median_filter(q, size=3) >= 0.5
            m = ndi.binary_fill_holes(s57.largest_cc(ndi.binary_opening(m, np.ones((5, 5)))))
        else:
            m = s57.largest_cc(q >= 0.5)
        return m

    BD, DC = {}, {}
    for cfg in CONFIGS:
        bd, dc = [], []
        for n, st in zip(val, stems):
            m = make(cfg, st)
            bd.append(s57.bdice(m, gts[n], tol=1))
            dc.append(s57.dice(m, gts[n]))
        BD[cfg], DC[cfg] = np.array(bd), np.array(dc)

    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(val))
    boot = [rng.choice(idx, size=len(idx), replace=True) for _ in range(a.boot)]

    def sign_p(d):
        n = int((np.abs(d) > 1e-9).sum())
        k = int((d > 0).sum())
        if n == 0:
            return 1.0
        z = (k - n / 2.0) / sqrt(n / 4.0)
        return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))

    print("\n%-20s %8s %8s" % ("config", "bdice1", "dice"))
    for cfg in CONFIGS:
        print("%-20s %8.3f %8.4f" % (cfg, BD[cfg].mean(), DC[cfg].mean()))

    print("\npaired (bootstrap %d, exact sign test):" % a.boot)
    for cfg in CONFIGS:
        for ref in ("res+pp", "ens4+pp", "nnunet+pp"):
            if cfg == ref:
                continue
            d = BD[cfg] - BD[ref]
            means = np.array([d[b].mean() for b in boot])
            lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
            print("  %-20s - %-20s %+7.3f  CI[%+.2f,%+.2f]  p=%.4f  %2d/%2d"
                  % (cfg, ref, d.mean(), lo, hi, sign_p(d),
                     int((d > 0.01).sum()), int((d < -0.01).sum())))

    np.savez(os.path.join(S9, "results", "final_val_bd.npz"),
             **{c: BD[c] for c in CONFIGS}, **{"dice_" + c: DC[c] for c in CONFIGS})

    if a.emit:
        assert a.emit in CONFIGS, "unknown config %s" % a.emit
        out_dir = os.path.join(S6, "predictions", "final_%s_png" % a.emit.split("+")[0])
        os.makedirs(out_dir, exist_ok=True)
        for n, st in zip(val, stems):
            m = make(a.emit, st)
            fname = n if n.lower().endswith(".png") else n + ".png"
            Image.fromarray(np.where(m, 255, 0).astype(np.uint8), mode="L").save(
                os.path.join(out_dir, fname))
        print("\nemitted %d val PNGs -> %s" % (len(val), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
