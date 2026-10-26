#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_failure_analysis/scripts/31b_unet_tiled_vs_fullframe.py

Closes the "train/test mismatch" question for the U-Net baseline properly.

The U-Net was trained on random 256x256 crops taken at NATIVE resolution, but
it is applied to a whole 768x1024 frame.  Resizing the whole frame to 256x256
(as an earlier draft of this analysis did) is NOT a fair reconstruction of the
training condition, so this script instead tiles the native-resolution frame
into 256x256 tiles, runs the network on each tile (the network maps 256 -> 256
exactly), stitches the tiles and compares the result with the full-frame
inference.

Reports Dice/HD95 for full-frame vs tiled inference, and how many pixels change
their 0/1 decision.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from scipy import ndimage

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
TILE = 256


def dice(a, b):
    return 2.0 * float(np.logical_and(a, b).sum()) / (float(a.sum()) + float(b.sum()) + 1e-7) * 100


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Ubase().to(dev)
    net.load_state_dict(torch.load(os.path.join(UNET, "weight", "ultrasound_prostate_73",
                                               "best_model.pth"), map_location=dev))
    net.eval()
    tt = ToTensor()
    _, val = read_split(USS)

    rows = []
    with torch.no_grad():
        for n in val:
            img = Image.open(os.path.join(IMG_DIR, n)).convert("RGB")
            x = tt(img).unsqueeze(0).to(dev)
            H, W = x.shape[2], x.shape[3]
            gt = load_binary(os.path.join(GT_DIR, n))

            # ---- full frame -------------------------------------------------
            ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
            p_full = torch.sigmoid(net(F.pad(x, (0, pw, 0, ph), mode="reflect")))
            p_full = p_full[:, :, :H, :W][0, 0].cpu().numpy()

            # ---- 256x256 native-scale tiles ---------------------------------
            ph_t, pw_t = (TILE - H % TILE) % TILE, (TILE - W % TILE) % TILE
            xt = F.pad(x, (0, pw_t, 0, ph_t), mode="reflect")
            tiles = xt.unfold(2, TILE, TILE).unfold(3, TILE, TILE)   # 1,3,ny,nx,256,256
            ny, nx = tiles.shape[2], tiles.shape[3]
            flat = tiles.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, TILE, TILE)
            out = torch.sigmoid(net(flat)).view(ny, nx, TILE, TILE)
            p_tile = out.view(ny, nx, TILE, TILE).permute(0, 2, 1, 3).reshape(ny * TILE, nx * TILE)[:H, :W].cpu().numpy()

            bf, bt = p_full >= 0.5, p_tile >= 0.5
            hf = surface_metrics(bf, gt)[0]
            ht = surface_metrics(bt, gt)[0]
            rows.append(dict(
                name=n,
                dice_full=dice(bf, gt), dice_tiled=dice(bt, gt),
                hd95_full=hf, hd95_tiled=ht,
                decision_flip_pct=float((bf != bt).mean()) * 100,
                corr=float(np.corrcoef(p_full.ravel(), p_tile.ravel())[0, 1]),
            ))
            print("%-30s full %.2f | tiled %.2f | flip %5.2f%% | HD95 %6.1f vs %6.1f | r=%.3f"
                  % (n[:30], rows[-1]["dice_full"], rows[-1]["dice_tiled"],
                     rows[-1]["decision_flip_pct"], hf, ht, rows[-1]["corr"]))

    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("dice_full", "dice_tiled", "hd95_full", "hd95_tiled",
                     "decision_flip_pct", "corr")}
    agg["per_image"] = rows
    with open(os.path.join(RES_DIR, "unet_tiled_vs_fullframe.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print("\n" + "=" * 74)
    print("full-frame  : Dice %.3f  HD95 %.1f px" % (agg["dice_full"], agg["hd95_full"]))
    print("tiled 256px : Dice %.3f  HD95 %.1f px" % (agg["dice_tiled"], agg["hd95_tiled"]))
    print("decision flips between the two input regimes: %.2f%% of pixels" % agg["decision_flip_pct"])
    print("mean correlation of the two probability maps: %.4f" % agg["corr"])
    print("-> results/unet_tiled_vs_fullframe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
