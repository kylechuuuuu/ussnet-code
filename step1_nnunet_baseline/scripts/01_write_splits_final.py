#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_nnunet_baseline/scripts/01_write_splits_final.py

Write `splits_final.json` into the nnU-Net preprocessed folder **before**
training, so nnU-Net uses our split instead of generating a random 5-fold.

Fold 0 is an exact copy of the split used by the existing U-Net baseline in
`USSnet/unet` (7:3, seed 42 -> 71 train / 31 val, see
`unet/datasets/ultrasound_prostate_73`).  Comparing nnU-Net fold 0 against the
U-Net result is therefore a like-for-like comparison on identical images.

Folds 1-4 are extra seeded 7:3 splits of the same kind.  They are not used by
the default fold-0 run; they exist so the same protocol can be repeated for
error bars.
"""
import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
USS = os.path.dirname(BASE)

MAPPING = os.path.join(BASE, "case_id_mapping.csv")
UNET_TRAIN_DIR = os.path.join(USS, "unet", "datasets", "ultrasound_prostate_73", "train", "images")
UNET_VAL_DIR = os.path.join(USS, "unet", "datasets", "ultrasound_prostate_73", "val", "images")
OUT = os.path.join(BASE, "data", "nnUNet_preprocessed", "Dataset001_UltrasoundProstate",
                   "splits_final.json")

VAL_RATIO = 0.3
EXTRA_SEEDS = [1234, 2024, 7, 99]


def main():
    with open(MAPPING) as f:
        m = {r["original_filename"]: r["case_id"] for r in csv.DictReader(f)}

    def pngs(d):
        return sorted(n for n in os.listdir(d) if n.lower().endswith(".png"))

    tr = [m[n] for n in pngs(UNET_TRAIN_DIR)]
    va = [m[n] for n in pngs(UNET_VAL_DIR)]
    print("U-Net split -> nnU-Net fold 0: train %d / val %d" % (len(tr), len(va)))
    assert not (set(tr) & set(va)), "train/val overlap in the U-Net split!"
    assert len(tr) + len(va) == len(m), "split does not cover all cases"

    folds = [{"train": sorted(tr), "val": sorted(va)}]
    all_cases = sorted(m.values())
    for seed in EXTRA_SEEDS:
        ids = list(all_cases)
        random.Random(seed).shuffle(ids)
        n_val = int(round(len(ids) * VAL_RATIO))
        folds.append({"train": sorted(ids[n_val:]), "val": sorted(ids[:n_val])})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(folds, f, indent=2)

    print("wrote %d folds -> %s" % (len(folds), OUT))
    for i, fd in enumerate(folds):
        tag = "<-- == U-Net baseline split" if i == 0 else "(extra seed)"
        print("  fold %d: train %3d  val %3d  %s" % (i, len(fd["train"]), len(fd["val"]), tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
