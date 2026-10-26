#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99j_bdice_table.py

Score a list of PNG prediction folders with the project's boundary metric
(bdice@1px, step4 tools/57_shape_prior_ceiling.py) on the fold-0 val split, and
report paired wins/losses against a reference folder.

This is the metric the whole analysis is expressed in (SQ2res-w25 base = 35.33
bdice@1px at Dice 96.38); the step-2 evaluator reports Dice/HD95/ASSD but NOT
bdice@1px, so ensemble claims were unverifiable without this table.

usage:
  python 99j_bdice_table.py --dirs A,B,C --labels res,ensD,ensI [--ref 0]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
from PIL import Image

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = load_mod("arb", os.path.join(S6, "tools", "63_arbitrate.py"))
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary",
                                   "tools", "57_shape_prior_ceiling.py"))


def load_png_bin(path):
    a = np.array(Image.open(path).convert("L"))
    return s57.largest_cc(a > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ref", type=int, default=0, help="index used for paired diff")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dirs.split(",")]
    labels = [l.strip() for l in args.labels.split(",")]
    assert len(dirs) == len(labels), "dirs/labels length mismatch"

    train, val = arb.read_split(USS)
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}
    preds = {}
    for lab, d in zip(labels, dirs):
        preds[lab] = {}
        for n in val:
            p = os.path.join(d, n)
            if not os.path.exists(p):
                raise SystemExit("missing prediction %s" % p)
            preds[lab][n] = load_png_bin(p)

    ref = labels[args.ref]
    print("val n=%d   boundary Dice @1px tolerance   ref=%s" % (len(val), ref))
    print("%-32s %8s %8s %8s %8s" % ("pred", "bdice1", "dice", "win", "loss"))
    for lab in labels:
        bd, dc, win, loss = [], [], 0, 0
        for n in val:
            b = s57.bdice(preds[lab][n], gts[n], tol=1)
            bd.append(b)
            dc.append(s57.dice(preds[lab][n], gts[n]))
            if lab != ref:
                d = b - s57.bdice(preds[ref][n], gts[n], tol=1)
                win += int(d > 0.01)
                loss += int(d < -0.01)
        print("%-32s %8.3f %8.3f %8d %8d" % (lab, float(np.mean(bd)),
                                             float(np.mean(dc)), win, loss))

    if len(labels) > 1:
        print("\npaired per-case bdice1 delta vs %s" % ref)
        for lab in labels:
            if lab == ref:
                continue
            d = np.array([s57.bdice(preds[lab][n], gts[n], tol=1)
                          - s57.bdice(preds[ref][n], gts[n], tol=1) for n in val])
            print("  %-32s mean %+7.3f  median %+7.3f  p10 %+7.3f  p90 %+7.3f"
                  % (lab, d.mean(), np.median(d), np.percentile(d, 10),
                     np.percentile(d, 90)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
