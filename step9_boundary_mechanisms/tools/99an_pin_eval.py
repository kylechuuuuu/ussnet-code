#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99an_pin_eval.py

The "pin" experiment: compound_B's damage is super-additive (99ae/99ad).  Which
half of the recipe is the load-bearing one?

  abC_texture_only = gamma^1.00 -> blur 2.2 -> speckle 0.30   (texture destroyed,
                     NO tgc, NO shadow)          -> is texture loss enough?
  abD_gain_only    = gamma^0.75 -> tgc(-0.5) -> shadow 0.30   (smooth multiplicative
                     gain only, NO blur/speckle) -> is gain alone enough?

Measures the two new prediction dirs with 99ae's own measure_dir/summarise (so the
numbers are on exactly the same scale as the 99ad leave-one-out table) and MERGES
them into results/attribution_shift.json without dropping the earlier ablation
rows (99al overwrote ab_shiftrun.json, so the old rows are read back from JSON).

usage: python tools/99an_pin_eval.py
"""
import importlib
import json
import os
import sys

import numpy as np

S9 = "/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms"
PREDS = "/media/72TB_disk/personal/zk/USSnet/step8_domain_shift/predictions"
MASK_DIR = "/media/72TB_disk/personal/zk/USSnet/UltrasoundProstate/masks"
sys.path.insert(0, os.path.join(S9, "tools"))

s9ae = importlib.import_module("99ae_domain_shift_attribution")
eval_lib = s9ae.eval_lib
USS = s9ae.USS

NEW = [("abC_texture_only", "ab_abC_texture_only_base_nifti"),
       ("abD_gain_only", "ab_abD_gain_only_base_nifti")]


def main():
    _, val_names = eval_lib.read_split(USS)
    stems = [os.path.splitext(n)[0] for n in val_names]
    gt = {n: eval_lib.load_binary(os.path.join(MASK_DIR, n)) for n in val_names}

    path = os.path.join(S9, "results", "attribution_shift.json")
    res = json.load(open(path)) if os.path.exists(path) else {"new": {}, "ablation": {}}
    abl = res.setdefault("ablation", {})

    for cond, d in NEW:
        dd = os.path.join(PREDS, d)
        if not os.path.isdir(dd):
            print("[99an] missing %s" % dd); continue
        rows = s9ae.measure_dir(dd, val_names, stems, gt)
        st = s9ae.summarise(rows)
        abl["ab|" + cond] = st
        print("[99an] %-18s dice %.2f  min %.2f  <80 %d  <50 %d  disp %.1f  collapse5 %s"
              % (cond, st["dice"], st["min_dice"], st["n_below80"], st["n_below50"],
                 st["centroid_disp_mean"], st["collapsed5_dice"]))

    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)

    print("\n  compound_B ablation / pin table (shipped SQ2res-w25, base head, val 31)")
    print("  %-24s %7s %7s %5s %5s %8s" % ("condition", "dice", "min", "<80", "<50", "disp"))
    order = ["ab|abB_rep", "ab|abA_fix", "ab|abC_texture_only", "ab|abD_gain_only"]
    order += [k for k in sorted(abl) if k.startswith("ab|abB_") and k not in order]
    for k in order:
        if k not in abl:
            continue
        v = abl[k]
        print("  %-24s %7.2f %7.2f %5d %5d %8.1f" %
              (k.split("|")[1], v["dice"], v["min_dice"], v["n_below80"],
               v["n_below50"], v["centroid_disp_mean"]))
    print("\n[99an] merged into %s" % path)


if __name__ == "__main__":
    main()
