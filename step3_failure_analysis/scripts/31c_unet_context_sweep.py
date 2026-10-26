#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_failure_analysis/scripts/31c_unet_context_sweep.py

Follow-up to 31b.  31b showed that running the U-Net on 256x256 tiles taken at
native resolution (i.e. exactly the patch size it was trained on) is MUCH worse
(Dice 84.7) than running it on the whole 768x1024 frame (Dice 93.0).  That
suggests the U-Net benefits from a large spatial CONTEXT, not from the resolution.

This script sweeps the inference context size (tile side 256/384/512 and the
whole frame) at native resolution, so the claim "context, not resolution, is
what the U-Net is missing" can be made quantitatively.  Nothing here changes
the trained weights: inference only.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

HERE = os.path.dirname(os.path.abspath(__file__))
STEP3 = os.path.dirname(HERE)
USS = os.path.dirname(STEP3)
UNET = os.path.join(USS, "unet")
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
sys.path.insert(0, UNET)
from eval_lib import load_binary, read_split, surface_metrics  # noqa: E402
from model.unet import Ubase  # noqa: E402

GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
RES_DIR = os.path.join(STEP3, "results")
TILES = [256, 384, 512]


def dice(a, b):
    return 2.0 * float(np.logical_and(a, b).sum()) / (float(a.sum()) + float(b.sum()) + 1e-7) * 100


def infer_full(net, x):
    """One forward pass on the whole padded frame (the way unet/test.py does it)."""
    H, W = x.shape[2], x.shape[3]
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    p = torch.sigmoid(net(F.pad(x, (0, pw, 0, ph), mode="reflect")))
    return p[:, :, :H, :W][0, 0].cpu().numpy()


def infer_tiled(net, x, t):
    """Non-overlapping t x t tiles at native scale, all tiles in one batch."""
    H, W = x.shape[2], x.shape[3]
    ph, pw = (t - H % t) % t, (t - W % t) % t
    xt = F.pad(x, (0, pw, 0, ph), mode="reflect")
    tiles = xt.unfold(2, t, t).unfold(3, t, t)            # 1,3,ny,nx,t,t
    ny, nx = tiles.shape[2], tiles.shape[3]
    flat = tiles.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, t, t)
    out = torch.sigmoid(net(flat)).view(ny, nx, t, t)
    return out.permute(0, 2, 1, 3).reshape(ny * t, nx * t)[:H, :W].cpu().numpy()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Ubase().to(dev)
    net.load_state_dict(torch.load(os.path.join(UNET, "weight", "ultrasound_prostate_73",
                                               "best_model.pth"), map_location=dev))
    net.eval()
    tt = ToTensor()
    _, val = read_split(USS)

    keys = ["full"] + ["tile%d" % t for t in TILES]
    acc = {k: {"dice": [], "hd95": []} for k in keys}
    rows = []
    with torch.no_grad():
        for n in val:
            x = tt(Image.open(os.path.join(IMG_DIR, n)).convert("RGB")).unsqueeze(0).to(dev)
            gt = load_binary(os.path.join(GT_DIR, n))
            r = {"name": n}
            for k in keys:
                p = infer_full(net, x) if k == "full" else infer_tiled(net, x, int(k[4:]))
                b = p >= 0.5
                d, h = dice(b, gt), surface_metrics(b, gt)[0]
                acc[k]["dice"].append(d)
                acc[k]["hd95"].append(h)
                r[k + "_dice"], r[k + "_hd95"] = d, h
            rows.append(r)
            print("%-30s " % n[:30] + " | ".join("%s %6.2f/%6.1f" % (k, r[k + "_dice"], r[k + "_hd95"])
                                                 for k in keys))

    out = {"keys": keys, "per_image": rows,
           "summary": {k: {"dice_mean": float(np.mean(acc[k]["dice"])),
                           "dice_std": float(np.std(acc[k]["dice"])),
                           "hd95_mean": float(np.mean(acc[k]["hd95"]))} for k in keys}}
    with open(os.path.join(RES_DIR, "unet_context_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    print("U-Net inference-context sweep on the 31 validation images (native resolution)")
    for k in keys:
        print("  %-8s context: Dice %6.3f +/- %.3f   HD95 %6.1f px"
              % (k, out["summary"][k]["dice_mean"], out["summary"][k]["dice_std"],
                 out["summary"][k]["hd95_mean"]))
    print("-> results/unet_context_sweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
