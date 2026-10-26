#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_nnunet_baseline/scripts/05_export_nifti_to_png.py

Convert nnU-Net prediction NIfTI files into the same PNG convention that the
U-Net baseline writes (`unet/results/...`): a single-channel PNG, 0 or 255,
at the ORIGINAL image resolution, with the ORIGINAL (Chinese) file name.

That makes the two prediction sets directly comparable and lets step 2 apply
one single evaluation implementation to both.

usage:
    python scripts/05_export_nifti_to_png.py \
        --pred predictions/nnunet_2d_fold0_checkpoint_best_nifti \
        --out  predictions/nnunet_2d_fold0_checkpoint_best_png
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
MAPPING = os.path.join(BASE, "case_id_mapping.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="folder with nnU-Net output .nii.gz")
    ap.add_argument("--out", required=True, help="folder to write PNGs into")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    pred_dir = args.pred if os.path.isabs(args.pred) else os.path.join(BASE, args.pred)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(BASE, args.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(MAPPING) as f:
        rows = list(csv.DictReader(f))
    cid2name = {r["case_id"]: r["original_filename"] for r in rows}

    n_done, n_missing, shapes = 0, [], {}
    for cid, orig in sorted(cid2name.items()):
        p = os.path.join(pred_dir, cid + ".nii.gz")
        if not os.path.exists(p):
            n_missing.append(cid)
            continue
        itk = sitk.ReadImage(p)
        arr = sitk.GetArrayFromImage(itk)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            print("!! unexpected shape", cid, arr.shape)
            return 3
        # nnU-Net exports the raw label map (argmax); be tolerant to soft input
        if arr.dtype.kind == "f":
            arr = (arr >= args.threshold).astype(np.uint8)
        binary = np.where(arr > 0, 255, 0).astype(np.uint8)
        Image.fromarray(binary, mode="L").save(os.path.join(out_dir, orig))
        shapes[arr.shape] = shapes.get(arr.shape, 0) + 1
        n_done += 1

    # sanity: shapes must match the ground truth PNGs
    gt_dir = os.path.join(os.path.dirname(BASE), "UltrasoundProstate", "masks")
    mism = 0
    for cid, orig in cid2name.items():
        if not os.path.exists(os.path.join(out_dir, orig)):
            continue
        a = np.array(Image.open(os.path.join(out_dir, orig)).convert("L")).shape
        b = np.array(Image.open(os.path.join(gt_dir, orig)).convert("L")).shape
        if a != b:
            mism += 1
            if mism <= 5:
                print("!! size mismatch", orig, a, b)

    print("exported %d/%d predictions -> %s" % (n_done, len(cid2name), out_dir))
    print("shapes:", shapes)
    if n_missing:
        print("missing:", n_missing[:10], "..." if len(n_missing) > 10 else "")
    print("ground-truth size mismatches:", mism)
    return 0


if __name__ == "__main__":
    sys.exit(main())
