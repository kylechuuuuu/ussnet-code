#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build `Dataset002_UltrasoundProstate3` -- the SAME 102 frames and the SAME
fold-0 split as Dataset001, but with the two source contour labels kept apart
instead of binarised:

    0 = background      (source value 0)
    1 = zone A          (source value 127)
    2 = zone B          (source value 255)

Why bother: every one of the 102 masks contains BOTH source values
(`tools/71_zone_split_stats.py`: 102/102, zone areas 48.8 % / 51.2 % of the
union), so the internal zone boundary is a real, consistently annotated
structure.  Dataset001 throws it away -- a binary model never sees it.  A model
that has to explain it as well gets an extra source of information about gland
shape, and its union `argmax != 0` is a *differently parameterised* estimate of
the same outer contour, which is exactly what an ensemble wants.

Everything else is held fixed so the only difference versus every other arm is
the output parameterisation:

  * the images are symlinks to Dataset001's `imagesTr` (identical pixels),
  * `splits_final.json` is copied verbatim (byte-identical fold 0),
  * the same fingerprint JSON drives the loss.

The dataset is written into a STEP4-LOCAL nnU-Net raw root
(`data/nnUNet_raw`) so that the shared, read-only step1 tree is never touched,
and so that pointing `nnUNet_raw` here cannot disturb a running job.

Run:  python tools/72_build_zone3_dataset.py [--out <dir>] [--dry-run]
"""
from __future__ import print_function

import argparse
import csv
import io
import json
import os
import shutil
import sys

import numpy as np
import SimpleITK as sitk
from PIL import Image

TOOLS = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(TOOLS)
USS = os.path.dirname(STEP4)

SRC_RAW = os.path.join(USS, "step1_nnunet_baseline", "data", "nnUNet_raw",
                       "Dataset001_UltrasoundProstate")
SRC_PRE = os.path.join(USS, "step1_nnunet_baseline", "data", "nnUNet_preprocessed",
                       "Dataset001_UltrasoundProstate")
MAPPING = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")
MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
DST_NAME = "Dataset002_UltrasoundProstate3"

DATASET_JSON = {
    "channel_names": {"0": "US"},
    "labels": {"background": 0, "zone_low": 1, "zone_high": 2},
    "numTraining": 102,
    "file_ending": ".nii.gz",
    "name": "UltrasoundProstate3",
    "description": ("Same 102 transrectal ultrasound frames as Dataset001, with "
                    "the two source contour labels kept as separate classes "
                    "(1 = source 127, 2 = source 255) instead of being unioned. "
                    "The union of classes 1 and 2 is byte-identical to "
                    "Dataset001's foreground, so models trained here are "
                    "directly comparable after argmax != 0."),
    "licence": "research use only",
    "reference": "internal clinical dataset, USSnet/UltrasoundProstate",
    "release": "1.0",
    "overwrite_image_reader_writer": "SimpleITKIO",
}


def read_mapping():
    rows = {}
    with io.open(MAPPING, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["case_id"]] = r
    return rows


def zone_array(png_path):
    """(H, W) uint8 with values {0, 1, 2}."""
    a = np.array(Image.open(png_path))
    bad = set(np.unique(a).tolist()) - {0, 127, 255}
    if bad:
        raise SystemExit("unexpected mask values %s in %s" % (sorted(bad), png_path))
    out = np.zeros(a.shape, dtype=np.uint8)
    out[a == 127] = 1
    out[a == 255] = 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(STEP4, "data", "nnUNet_raw"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dst = os.path.join(args.out, DST_NAME)
    img_dst = os.path.join(dst, "imagesTr")
    lab_dst = os.path.join(dst, "labelsTr")

    mapping = read_mapping()
    case_ids = sorted(mapping)
    print("[72] %d cases in %s" % (len(case_ids), os.path.basename(MAPPING)))
    if not args.dry_run:
        for d in (img_dst, lab_dst):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)

    n_ok = 0
    for cid in case_ids:
        src_img = os.path.join(SRC_RAW, "imagesTr", cid + "_0000.nii.gz")
        src_lab = os.path.join(SRC_RAW, "labelsTr", cid + ".nii.gz")
        png = os.path.join(MASK_DIR, mapping[cid]["original_filename"])
        for p in (src_img, src_lab, png):
            if not os.path.exists(p):
                raise SystemExit("[72] MISSING %s" % p)

        z = zone_array(png)                                   # (H, W) {0,1,2}
        ref = sitk.ReadImage(src_lab)
        a = sitk.GetArrayFromImage(ref)                        # (H, W) for 2-D
        if a.shape != z.shape:
            if a.T.shape == z.shape:
                raise SystemExit("[72] %s: label is transposed vs the PNG "
                                 "(%s vs %s) -- handle orientation first"
                                 % (cid, a.shape, z.shape))
            raise SystemExit("[72] %s: shape mismatch %s vs %s"
                             % (cid, a.shape, z.shape))

        old_fg = (np.asarray(a) > 0)
        new_fg = (z > 0)
        if not np.array_equal(old_fg, new_fg):
            d = int(np.logical_xor(old_fg, new_fg).sum())
            raise SystemExit("[72] %s: the reconstructed union differs from "
                             "Dataset001's foreground by %d px" % (cid, d))
        fg = int(new_fg.sum())
        if fg != int(mapping[cid]["fg_pixels"]):
            print("[72]   note %s: fg %d vs mapping %s"
                  % (cid, fg, mapping[cid]["fg_pixels"]))

        if not args.dry_run:
            os.symlink(src_img, os.path.join(img_dst, cid + "_0000.nii.gz"))
            out = sitk.GetImageFromArray(z)
            out.CopyInformation(ref)
            sitk.WriteImage(out, os.path.join(lab_dst, cid + ".nii.gz"))
        n_ok += 1

    print("[72] verified %d/%d cases: union == Dataset001 foreground, "
          "in every image both zones are present" % (n_ok, len(case_ids)))

    if not args.dry_run:
        with io.open(os.path.join(dst, "dataset.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(DATASET_JSON, indent=4, ensure_ascii=False) + "\n")
        shutil.copy2(os.path.join(SRC_PRE, "splits_final.json"),
                     os.path.join(dst, "splits_final.json"))
        print("[72] wrote %s (%d images, %d labels, dataset.json, splits_final.json)"
              % (dst, len(os.listdir(img_dst)), len(os.listdir(lab_dst))))
        with io.open(os.path.join(SRC_PRE, "splits_final.json")) as fh:
            sp = json.load(fh)
        print("[72] fold 0: %d train / %d val (identical to Dataset001)"
              % (len(sp[0]["train"]), len(sp[0]["val"])))
    else:
        print("[72] dry run -- nothing written")


if __name__ == "__main__":
    sys.exit(main())
