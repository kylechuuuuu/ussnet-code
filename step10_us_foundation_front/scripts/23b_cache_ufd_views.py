#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/23b_cache_ufd_views.py

Cache UltraFedFM (MAE ViT-B/16, 111.9 M) last-block features on the SAME shifted
views as the UltraSAM3 t24 cache (`23_cache_teacher_views.py`), so the
"teacher-swap" experiment (E5) can be trained with:

    python scripts/27_student_narrow_vit.py --views-dir results/teacher_views_ufd_fold3 \
        --teacher-dim 768 --fold 3 --tag l2vit_e5ufd_fold3 ...

Differences vs 23 (all deliberate):
* seeds are CRC32-stable (`%s|%d`, recorded in _meta.json) -- 27 regenerates the
  training view from the recorded seed, so student input == teacher realisation.
* teacher feature = UltraFedFM token grid (patch 16, canvas 1024 -> 64x64), rows
  cropped to the valid region, then BILINEARLY resampled to the STUDENT's token
  grid (72 cols, rows = valid rows at patch 14) -- the student architecture is
  untouched (single-variable teacher swap).
* image.npy saved in 23's convention (zscore, 1024x768, fp16) for completeness;
  training reads the raw image + regenerates the view, never this file.

usage:
  ~/anaconda3/envs/zk/bin/python scripts/23b_cache_ufd_views.py --fold 3 --device cuda:0
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
sys.path.insert(0, HERE)


def _load(mod_name, fname):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


l0 = _load("l0u", "10_l0_probe.py")
tv = _load("tvu", "23_cache_teacher_views.py")
fs = _load("fsu", "fold_split.py")

TCANVAS = (1024, 1024)     # UltraFedFM canvas (L0 probe convention)
SPATCH = 16                # UltraFedFM patch
TPATCH = 14                # student token patch (27_student_narrow_vit.py)
SCOLS = 72                 # student token cols on the 1008-wide canvas
S_CANVAS_W = 1008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--views", default="0,1,2,3,4,5")
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "results", "teacher_views_ufd_fold%d" % args.fold)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    enc = l0.build_encoder("ultrafedfm", dev, TCANVAS)
    train_names, _ = fs.read_split(args.fold)
    want = {int(v) for v in args.views.split(",") if v.strip()}
    meta = {"views": {str(k): tv.VIEWS[k] for k in tv.VIEWS},
            "teacher": "ultrafedfm MAE ViT-B/16, last block, resampled to student grid",
            "images": {}}
    t0 = time.time()
    for i, n in enumerate(train_names):
        raw = np.asarray(Image.open(os.path.join(USS, "UltrasoundProstate", "images", n))
                         .convert("L"), dtype=np.float32) / 255.0
        H0, W0 = raw.shape
        nh_s = min(S_CANVAS_W, max(1, int(round(H0 * S_CANVAS_W / float(W0)))))
        rows_s = nh_s // TPATCH                    # student valid token rows (54 or 52)
        nh_t = min(TCANVAS[1], max(1, int(round(H0 * TCANVAS[0] / float(W0)))))
        meta["images"][n] = []
        for k, kind in tv.VIEWS.items():
            if k not in want:
                continue
            seed = zlib.crc32(("%s|%d" % (n, k)).encode()) % (2 ** 31)
            a = tv.shift_image(raw, kind, seed, tv.RECIPE.get(k))
            pil = Image.fromarray((a * 255).astype(np.uint8))
            # student-side image cache (23's convention; not read by training)
            a_img = np.asarray(pil.resize((1024, 768), Image.BILINEAR), dtype=np.float32) / 255.0
            np.save(os.path.join(out, "%s__v%d.image.npy" % (n[:-4], k)),
                    tv.zscore(a_img).astype(np.float16))
            # teacher feature: canvas 1024 -> valid rows at patch 16 -> resample
            # to the student token grid (rows_s x 72), bilinear
            f = enc.encode(pil)                    # (768, 64, 64) float tensor
            f = f[:, :nh_t // SPATCH, :]
            fr = F.interpolate(f[None].float(), size=(rows_s, SCOLS),
                               mode="bilinear", align_corners=False)[0]
            np.save(os.path.join(out, "%s__v%d.feat.npy" % (n[:-4], k)),
                    fr.half().cpu().numpy())
            meta["images"][n].append({"view": k, "kind": kind, "seed": int(seed),
                                      "feat_shape": list(fr.shape)})
        if (i + 1) % 10 == 0 or i == len(train_names) - 1:
            print("  %d/%d (%.1f min)" % (i + 1, len(train_names), (time.time() - t0) / 60), flush=True)
    with open(os.path.join(out, "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("[ufd] %d images x %d views -> %s" % (len(train_names), len(want), out), flush=True)


if __name__ == "__main__":
    sys.exit(main())
