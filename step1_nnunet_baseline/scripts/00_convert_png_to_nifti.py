#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_nnunet_baseline/scripts/00_convert_png_to_nifti.py

Convert the raw PNG dataset

    USSnet/UltrasoundProstate/{images,masks}/*.png        (102 pairs)

into the nnU-Net v2 raw format

    USSnet/step1_nnunet_baseline/data/nnUNet_raw/
        Dataset001_UltrasoundProstate/
            imagesTr/USP001_0000.nii.gz      (2-D uint8, spacing 1.0 x 1.0 mm)
            labelsTr/USP001.nii.gz           (2-D uint8, 0 = background, 1 = prostate)
            dataset.json

Notes
-----
* The original file names are Chinese; nnU-Net identifiers must be ASCII, so
  every case gets a stable ASCII id (USP001 ... USP102) and the mapping is
  written to `case_id_mapping.csv` in step1_nnunet_baseline/.
* nnU-Net's 2-D convention (nnunetv2/imageio/base_reader_writer.py) is a
  **2-D NIfTI** which the SimpleITK reader turns into (c=1, 1, H, W) with
  spacing (999*max, sp_y, sp_x).  So we write plain 2-D files.
* The masks contain two abutting contour labels (127 and 255).  For the
  head-to-head comparison with the existing U-Net baseline - which binarises
  both to foreground - the label map here is the binarised union (0/1).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                 # .../USSnet/step1_nnunet_baseline
USS = os.path.dirname(BASE)                  # .../USSnet

SRC_IMG = os.path.join(USS, "UltrasoundProstate", "images")
SRC_MSK = os.path.join(USS, "UltrasoundProstate", "masks")
DST = os.path.join(BASE, "data", "nnUNet_raw", "Dataset001_UltrasoundProstate")
MAPPING_CSV = os.path.join(BASE, "case_id_mapping.csv")

SPACING_MM = (1.0, 1.0)      # unknown real pixel pitch -> unit spacing


def write_2d_nifti(arr2d, path):
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr2d))
    img.SetSpacing(SPACING_MM)
    img.SetOrigin((0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(img, path, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for d in (SRC_IMG, SRC_MSK):
        if not os.path.isdir(d):
            print("missing source folder:", d)
            return 2
    if os.path.isdir(DST) and os.path.isdir(os.path.join(DST, "imagesTr")) \
            and os.listdir(os.path.join(DST, "imagesTr")) and not args.overwrite:
        print("destination already populated, use --overwrite:", DST)
        return 2

    img_dir = os.path.join(DST, "imagesTr")
    lbl_dir = os.path.join(DST, "labelsTr")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    names = sorted(n for n in os.listdir(SRC_IMG) if n.lower().endswith(".png"))
    rows, n_rgb, n_sizes = [], [], {}
    for i, name in enumerate(names, start=1):
        stem = os.path.splitext(name)[0]
        mpath = os.path.join(SRC_MSK, stem + ".png")
        if not os.path.exists(mpath):
            print("!! no mask for", name)
            return 3

        im = Image.open(os.path.join(SRC_IMG, name))
        if im.mode == "RGB":
            a = np.array(im)
            if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])):
                n_rgb.append(name)
            gray = a[..., 0]
        else:
            gray = np.array(im.convert("L"))
        gray = np.ascontiguousarray(gray.astype(np.uint8))

        msk = np.array(Image.open(mpath).convert("L"))
        if gray.shape != msk.shape:
            print("!! shape mismatch", name, gray.shape, msk.shape)
            return 4
        lab = np.where(msk > 0, 1, 0).astype(np.uint8)

        cid = "USP%03d" % i
        write_2d_nifti(gray, os.path.join(img_dir, cid + "_0000.nii.gz"))
        write_2d_nifti(lab, os.path.join(lbl_dir, cid + ".nii.gz"))

        key = "%dx%d" % (gray.shape[1], gray.shape[0])
        n_sizes[key] = n_sizes.get(key, 0) + 1
        rows.append(dict(case_id=cid, original_filename=name,
                         height=int(gray.shape[0]), width=int(gray.shape[1]),
                         fg_pixels=int(lab.sum()),
                         source_mask_values="0;127;255"))

    with open(MAPPING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dataset_json = {
        "channel_names": {"0": "US"},
        "labels": {"background": 0, "prostate": 1},
        "numTraining": len(rows),
        "file_ending": ".nii.gz",
        "name": "UltrasoundProstate",
        "description": "2-D transrectal ultrasound prostate segmentation, 102 annotated "
                       "frames (101 patients). Label = union of the two source contour "
                       "labels (127 and 255) binarised to foreground, matching the "
                       "existing U-Net baseline.",
        "licence": "research use only",
        "reference": "internal clinical dataset, USSnet/UltrasoundProstate",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(os.path.join(DST, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("converted %d cases -> %s" % (len(rows), DST))
    print("image sizes (WxH):", n_sizes)
    if n_rgb:
        print("WARNING: %d images are not R==G==B (kept channel 0): %s"
              % (len(n_rgb), n_rgb[:5]))
    print("case-id mapping written to", MAPPING_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
