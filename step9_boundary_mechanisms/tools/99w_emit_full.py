#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99w_emit_full.py

Emit the delivered configuration for ALL 102 cases (fold-0 train 71 + val 31),
not just the 31 val cases.  Only the val cases have an honest metric; the train
cases are included so the delivery is a complete prediction set for the modality.

config = the shipped `tta_art+pp`:
    field = (in_domain + jpeg_30 + jpeg_15) / 3        # step8 "reference"+"artifact"
    mask  = median3 -> binary_opening(5x5) -> largest_cc -> fill_holes -> >=0.5

  in_domain : sq2resencw25_checkpoint_best_base_nifti/<case_id>.npz   (102)
  jpeg_*    : step8_domain_shift/predictions/sq2shift_jpeg_*_base_nifti/<name>.npz

usage: python tools/99w_emit_full.py [--tag final_tta_art_full_png]
"""
import argparse
import csv
import importlib.util
import os
import sys

import numpy as np
from PIL import Image
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


s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools",
                                   "57_shape_prior_ceiling.py"))
k99 = load_mod("k99", os.path.join(S9, "tools", "99k_weight_sweep.py"))

ART = ["jpeg_30", "jpeg_15"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final_tta_art_full_png")
    a = ap.parse_args()

    with open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")) as f:
        rows = list(csv.DictReader(f))
    out = os.path.join(S6, "predictions", a.tag)
    os.makedirs(out, exist_ok=True)

    n_ok = 0
    for r in rows:
        cid = r["case_id"]
        name = r["original_filename"]              # already carries the .png suffix
        stem = name[:-4] if name.lower().endswith(".png") else name
        # prefer the step-8 in_domain field (same model, same settings) so that
        # the val part of this folder is bit-identical to final_tta_art_png;
        # for train cases it does not exist and the step-6 prediction is used
        p_in8 = os.path.join(S8, "sq2shift_in_domain_base_nifti", stem + ".npz")
        p_ind = p_in8 if os.path.exists(p_in8) else os.path.join(
            S6, "predictions", "sq2resencw25_checkpoint_best_base_nifti", cid + ".npz")
        ps = [os.path.join(S8, "sq2shift_%s_base_nifti" % c, stem + ".npz") for c in ART]
        if not os.path.exists(p_ind) or not all(os.path.exists(p) for p in ps):
            print("skip (missing npz): %s" % name)
            continue
        q = np.mean([k99.load_probs(p) for p in [p_ind] + ps], axis=0)[1]
        m = ndi.median_filter(q, size=3) >= 0.5
        m = ndi.binary_fill_holes(s57.largest_cc(ndi.binary_opening(m, np.ones((5, 5)))))
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8), mode="L").save(
            os.path.join(out, name if name.lower().endswith(".png") else name + ".png"))
        n_ok += 1
    print("emitted %d/102 PNGs -> %s" % (n_ok, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
