#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/24_stage_extra_conditions.py

Stage HELD-OUT acquisition conditions for the val split: operators that appear
NOWHERE in training, so "the student got more robust" can be separated from
"the student memorised blur_35".

    blur_45     gaussian sigma 4.5                        (much stronger than any
                                                           training view's 2.2/3.5)
    speckle_50  lognormal sigma 0.50
    compound_C  gamma 0.7 -> depth ramp + shadow -> blur 3.0 -> speckle 0.45
                -> jpeg q=25                            (compound, unseen recipe)

Writes the same layout step8 uses, so 22_eval_student.py can consume it with
--image-root:
    data/shift_extra/<cond>/images/<name>.png

usage: ~/anaconda3/envs/zk/bin/python scripts/24_stage_extra_conditions.py
"""
import argparse
import io
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
OUT = os.path.join(ROOT, "data", "shift_extra")

sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("l0", os.path.join(HERE, "10_l0_probe.py"))
l0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l0)


def blur45(a, rng):
    return gaussian_filter(a, 4.5)


def speckle50(a, rng):
    return a * rng.lognormal(mean=0.0, sigma=0.50, size=a.shape).astype(np.float32)


def compound_c(a, rng):
    H, W = a.shape
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    a = np.clip(a, 1e-3, 1.0) ** 0.7
    a = a * (1.0 - 0.5 * yy)
    a = a * (1.0 + 0.3 * np.sin(2 * np.pi * xx / max(4.0, W / 6.0)) *
             np.exp(-((xx - W * 0.5) / (0.15 * W)) ** 2))
    a = gaussian_filter(a, 3.0)
    a = a * rng.lognormal(mean=0.0, sigma=0.45, size=a.shape).astype(np.float32)
    a = np.clip(a, 0, 1)
    buf = io.BytesIO()
    Image.fromarray((a * 255).astype(np.uint8)).save(buf, format="JPEG", quality=25)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L"),
                      dtype=np.float32) / 255.0


CONDS = {"blur_45": blur45, "speckle_50": speckle50, "compound_C": compound_c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    if args.fold:
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location("fold_split", os.path.join(HERE, "fold_split.py"))
        _fs = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_fs)
        _, val_names = _fs.read_split(args.fold)
        out_root = os.path.join(ROOT, "data", "shift_extra_fold%d" % args.fold)
    else:
        _, val_names = l0.eval_lib.read_split(USS)
        out_root = OUT
    names = val_names[:args.limit] if args.limit else val_names
    print("[extra] fold=%d n=%d -> %s" % (args.fold, len(names), out_root), flush=True)
    for cond, fn in CONDS.items():
        d = os.path.join(out_root, cond, "images")
        os.makedirs(d, exist_ok=True)
        for n in names:
            src = os.path.join(USS, "UltrasoundProstate", "images", n)
            a = np.asarray(Image.open(src).convert("L"), dtype=np.float32) / 255.0
            rng = np.random.default_rng(abs(hash((n, cond))) % (2 ** 31))
            b = np.clip(fn(a, rng), 0, 1)
            Image.fromarray((b * 255).astype(np.uint8)).save(os.path.join(d, n))
        print("[extra] %-11s %d images -> %s" % (cond, len(names), d), flush=True)
    print("[extra] done")


if __name__ == "__main__":
    sys.exit(main())
