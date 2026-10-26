#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/23_cache_teacher_views.py

Cache UltraSAM3 trunk features on SHIFTED views of the 71 fold-0 train images.

WHY: the first L2 prototype distilled the teacher on CLEAN images only, and it
collapsed on blur/speckle exactly like the ImageNet-pretrained front does
(compound_B 27.08 vs armC's 55.26, teacher probe 91.31, eval in
results/l2_eval_l2_student_200ep.json).  Matching a shift-invariant teacher on
clean images does not transfer that invariance to the student -- the student's
own conv trunk still anchors on texture.  So the teacher is run on the same
shifted views the student will see, and the distillation target becomes
"what the teacher's representation looks like UNDER the shift".

Views (deterministic per (image, view), mirroring step8/step9's registered axes):
    v0 clean
    v1 texture+gain   gamma 0.75 -> depth ramp + shadow stripes -> blur 2.2 -> speckle 0.30
    v2 texture only   blur 2.2 -> speckle 0.30
    v3 gain only      gamma 0.75 -> depth ramp + shadow stripes

Cached per (image, view):
    results/teacher_views/<name>__v<k>.image.npy   student input, z-scored, fp16
    results/teacher_views/<name>__v<k>.feat.npy    teacher feature map, cropped, fp16
    results/teacher_views/_meta.json

usage: ~/anaconda3/envs/zk/bin/python scripts/23_cache_teacher_views.py [--device cuda:0]
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
OUT = os.path.join(ROOT, "results", "teacher_views")
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location("l0", os.path.join(HERE, "10_l0_probe.py"))
l0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l0)

_s = importlib.util.spec_from_file_location("fs", os.path.join(HERE, "fold_split.py"))
fs = importlib.util.module_from_spec(_s)
_s.loader.exec_module(fs)

CW, CH = 1024, 768                 # student canvas
TCANVAS = (1008, 1008)             # teacher canvas (SAM3 patch-14 img_size)
PATCH = 14
VIEWS = {0: "clean", 1: "texture+gain", 2: "texture", 3: "gain",
         4: "heavy_blur", 5: "compound_full"}
# per-view recipe overrides (kind, blur_sigma, speckle_sigma, gamma)
RECIPE = {4: dict(blur=3.5, speckle=0.35, gamma=1.0),
          5: dict(blur=3.0, speckle=0.40, gamma=0.75, shadow=True, ramp=True)}


def shift_image(a, kind, seed, recipe=None):
    """a: float32 (H,W) in [0,1] -> shifted float32 (H,W) in [0,1].

    `recipe` (dict) overrides the axis strengths for the extra views v4/v5, so
    one code path serves both the step8-matching views and the stronger ones.
    """
    if kind == "clean":
        return a
    recipe = recipe or {}
    blur = float(recipe.get("blur", 2.2))
    speckle = float(recipe.get("speckle", 0.30))
    gamma = float(recipe.get("gamma", 0.75))
    want_gain = ("gain" in kind) or recipe.get("shadow") or recipe.get("ramp")
    want_texture = ("texture" in kind) or "blur" in recipe
    rng = np.random.default_rng(seed)
    H, W = a.shape
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    if want_gain:
        a = np.clip(a, 1e-3, 1.0) ** gamma
        ramp = 1.0 - 0.5 * yy
        xx = np.arange(W, dtype=np.float32)[None, :]
        stripe = 1.0 + 0.3 * np.sin(2 * np.pi * xx / max(4.0, W / 6.0)) * \
            np.exp(-((xx - W * 0.5) / (0.15 * W)) ** 2)
        a = a * ramp * stripe
    if want_texture:
        a = gaussian_filter(a, blur)
        a = a * rng.lognormal(mean=0.0, sigma=speckle, size=a.shape).astype(np.float32)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def zscore(a):
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", default="", help="e.g. 4,5 to add only the extra views")
    ap.add_argument("--blocks", type=int, default=32,
                    help="truncate the SAM3 trunk to the first K blocks (24 keeps full robustness)")
    args = ap.parse_args()

    global OUT
    if args.blocks != 32:
        OUT = OUT + "_t%d" % args.blocks
    if args.fold:
        OUT = OUT + "_fold%d" % args.fold
    os.makedirs(OUT, exist_ok=True)
    import torch
    dev = torch.device(args.device)
    l0.CANVAS = TCANVAS
    if args.blocks != 32:
        t25 = importlib.import_module("25_probe_truncated_teacher")
        ck = os.path.join(ROOT, "weights", "UltraSAM3.pt")
        l0.build_encoder = (lambda kk: (lambda name, device, canvas:
                                        t25.TruncatedSAM3(device, ck, canvas, kk)))(args.blocks)
    enc = l0.build_encoder("ultrasam3", dev, TCANVAS)

    train_names, _ = fs.read_split(args.fold)
    names = train_names[:args.limit] if args.limit else train_names
    meta = {"views": VIEWS, "teacher_canvas": TCANVAS, "student_canvas": [CW, CH],
            "images": {}}
    t0 = time.time()
    for i, n in enumerate(names):
        raw = np.asarray(Image.open(os.path.join(USS, "UltrasoundProstate", "images", n))
                         .convert("L"), dtype=np.float32) / 255.0
        meta["images"][n] = []
        want = {int(v) for v in args.views.split(",") if v.strip()} if args.views else None
        for k, kind in VIEWS.items():
            if want is not None and k not in want:
                continue
            seed = abs(hash((n, k))) % (2 ** 31)
            a = shift_image(raw, kind, seed, RECIPE.get(k))
            # --- student input: resized + z-scored, cached fp16 ---
            a_img = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
                (CW, CH), Image.BILINEAR), dtype=np.float32) / 255.0
            np.save(os.path.join(OUT, "%s__v%d.image.npy" % (n[:-4], k)),
                    zscore(a_img).astype(np.float16))
            # --- teacher feature: same view through the SAM3 canvas ---
            f = enc.encode(Image.fromarray((a * 255).astype(np.uint8)))  # SAM3 encoder returns a tensor
            # Do NOT read l0.VALID here: with --blocks the encoder comes from a
            # separately-loaded copy of 10_l0_probe (importlib), so it writes
            # VALID into THAT module's namespace and l0.VALID stays None.
            # Derive the valid rows from the source geometry (same rule as prepare()).
            H0, W0 = raw.shape
            nh = min(TCANVAS[1], max(1, int(round(H0 * TCANVAS[0] / float(W0)))))
            f = f[:, :nh // PATCH, :]
            np.save(os.path.join(OUT, "%s__v%d.feat.npy" % (n[:-4], k)),
                    f.half().cpu().numpy())
            meta["images"][n].append({"view": k, "kind": kind, "seed": int(seed),
                                      "feat_shape": list(f.shape)})
        if (i + 1) % 10 == 0 or i == len(names) - 1:
            print("  %d/%d (%.1f min)" % (i + 1, len(names), (time.time() - t0) / 60), flush=True)
    with open(os.path.join(OUT, "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".npy"))
    print("[views] %d images x %d views -> %s (%.0f MB)" % (len(names), len(VIEWS), OUT, size / 1e6))


if __name__ == "__main__":
    sys.exit(main())
