#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/fold_split.py

Per-fold train/val splits in the ORIGINAL PNG naming used everywhere else in
this project (images/, teacher caches, shift suite, arbitration).

* fold k -> step1's nnUNet preprocessed splits_final.json (5 folds, USPxxx stems)
  mapped through step1's case_id_mapping.csv (USPxxx -> 原始中文文件名.png).
* fold 0 is verified against the materialised directories
  (unet/datasets/ultrasound_prostate_73/{train,val}/images) that step2's
  eval_lib.read_split uses, so every previously produced number keeps referring
  to the same 71/31 split.  `--fold 0` therefore returns the directory listing
  (authoritative), all other folds go through the mapping.

Self-check:  python fold_split.py
"""
import csv
import json
import os

USS = "/media/72TB_disk/personal/zk/USSnet"
SPLITS = os.path.join(USS, "step1_nnunet_baseline", "data", "nnUNet_preprocessed",
                      "Dataset001_UltrasoundProstate", "splits_final.json")
MAPPING = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")
EXP = "ultrasound_prostate_73"


def _dirs(uss_root=USS):
    base = os.path.join(uss_root, "unet", "datasets", EXP)

    def names(sub):
        d = os.path.join(base, sub, "images")
        return sorted(n for n in os.listdir(d) if n.lower().endswith(".png"))

    return names("train"), names("val")


def _mapping():
    m = {}
    with open(MAPPING, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("case_id") or "").strip()
            orig = (row.get("original_filename") or "").strip()
            if cid and orig:
                m[cid] = orig if orig.lower().endswith(".png") else orig + ".png"
    return m


def read_split(fold=0, uss_root=USS):
    """-> (train_names, val_names) as PNG filenames of the original cohort."""
    if int(fold) == 0:
        return _dirs(uss_root)
    m = _mapping()
    d = json.load(open(SPLITS))[int(fold)]
    conv = lambda ns: sorted(m[c] for c in ns)
    return conv(d["train"]), conv(d["val"])


def main():
    dtr, dva = _dirs()
    m = _mapping()
    splits = json.load(open(SPLITS))
    for k in range(len(splits)):
        tr, va = read_split(k)
        ok = ""
        if k == 0:
            ok = " | matches dirs: %s" % (sorted(tr) == sorted(dtr) and sorted(va) == sorted(dva))
        print("fold %d: train %d / val %d%s" % (k, len(tr), len(va), ok))
    allv = [n for k in range(len(splits)) for n in read_split(k)[1]]
    print("union of val folds: %d unique of %d cases (%s)"
          % (len(set(allv)), len(m), "OK" if len(set(allv)) == len(m) else "MISMATCH"))


if __name__ == "__main__":
    main()
