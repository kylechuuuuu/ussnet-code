#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/20_cache_teacher.py

Cache the UltraSAM3 (SAM3 ViT-Det, 841.7 M) trunk features for the 71 fold-0
TRAIN images.  These are the distillation targets for the L2 prototype; caching
them once means the 10 GB teacher never has to be resident during student
training.

Stored per image: the trunk output cropped to the VALID image area (the padded
rows are dropped), fp16, plus the geometry so the student can resample.
    results/teacher/ultrasam3/<name>.npy   (C=1024, gh, gw) fp16
    results/teacher/ultrasam3/_meta.json

usage: ~/anaconda3/envs/zk/bin/python scripts/20_cache_teacher.py [--device cuda:0]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "teacher", "ultrasam3")
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location("l0", os.path.join(HERE, "10_l0_probe.py"))
l0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l0)

CANVAS = (1008, 1008)          # SAM3 pretrain img_size / patch 14 (=72x72 tokens)
PATCH = 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    l0.CANVAS = CANVAS
    import torch
    dev = torch.device(args.device)
    enc = l0.build_encoder("ultrasam3", dev, CANVAS)

    train_names, val_names = l0.eval_lib.read_split(l0.USS)
    names = train_names[:args.limit] if args.limit else train_names
    meta = {}
    t0 = time.time()
    for i, n in enumerate(names):
        img, _ = l0.load_case(n, "in_domain")
        f = enc.encode(img)                                   # (1024, 72, 72)
        nw, nh = l0.VALID
        gh = nh // PATCH
        f = f[:, :gh, :]                                      # drop the padded rows
        np.save(os.path.join(OUT, n.replace(".png", ".npy")),
                f.half().cpu().numpy())
        meta[n] = {"canvas": CANVAS, "valid": [int(nw), int(nh)], "patch": PATCH,
                   "feat_shape": list(f.shape)}
        if (i + 1) % 10 == 0 or i == len(names) - 1:
            print("  %d/%d  (%.1f min elapsed)" % (i + 1, len(names), (time.time() - t0) / 60), flush=True)
    with open(os.path.join(OUT, "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("[cache] %d images -> %s  (%.1f MB)"
          % (len(names), OUT, sum(os.path.getsize(os.path.join(OUT, f))
                                  for f in os.listdir(OUT) if f.endswith(".npy")) / 1e6))


if __name__ == "__main__":
    sys.exit(main())
