#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_failure_analysis/scripts/31_unet_probability_diagnostics.py

Re-run the existing U-Net baseline itself (weights only, no retraining) so that
its failure modes can be measured from PROBABILITIES rather than from the saved
binary PNGs:

  A. threshold sensitivity      - how much does the 0.5 threshold cost?
  B. test-time augmentation     - does averaging the 4 dihedral flips help?
  C. resolution mismatch        - the model was trained on 256x256 crops but is
                                  applied to full 768x1024 frames; what does that
                                  cost? (measured by also running it the way it
                                  was trained: resize -> 256x256 -> forward)
  D. shift equivariance        - the encoder/decoder use unpadded convolutions and
                                  asymmetric F.pad, so a 1-pixel input shift may
                                  not produce a 1-pixel output shift
  E. calibration               - reliability / ECE of the sigmoid output
  F. per-region confidence     - predicted probability inside the 127 zone, the
                                  255 zone and the background

usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/31_unet_probability_diagnostics.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
STEP3 = os.path.dirname(HERE)
USS = os.path.dirname(STEP3)
UNET = os.path.join(USS, "unet")
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
sys.path.insert(0, UNET)
from eval_lib import load_binary, read_split  # noqa: E402
from model.unet import Ubase  # noqa: E402

GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
RES_DIR = os.path.join(STEP3, "results")
EPS = 1e-7


def pad16(t):
    _, _, h, w = t.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    return F.pad(t, (0, pw, 0, ph), mode="reflect"), ph, pw


def unpad(t, ph, pw):
    return t[:, :, : t.shape[2] - ph if ph else None, : t.shape[3] - pw if pw else None]


def dice(a, b):
    return 2.0 * float(np.logical_and(a, b).sum()) / (float(a.sum()) + float(b.sum()) + EPS) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(UNET, "weight",
                                                      "ultrasound_prostate_73", "best_model.pth"))
    ap.add_argument("--out", default=os.path.join(RES_DIR, "unet_probability_diagnostics.json"))
    ap.add_argument("--save-probs", default=os.path.join(RES_DIR, "unet_73_probabilities"))
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(args.save_probs, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Ubase().to(dev)
    sd = torch.load(args.weights, map_location=dev)
    net.load_state_dict(sd)
    net.eval()
    print("loaded", args.weights, "on", dev)

    _, val = read_split(USS)
    tt = ToTensor()
    per_img = {}
    probs_all, labels_all = [], []

    for n in val:
        img = Image.open(os.path.join(IMG_DIR, n)).convert("RGB")
        x = tt(img).unsqueeze(0).to(dev)
        xp, ph, pw = pad16(x)
        gt = load_binary(os.path.join(GT_DIR, n))
        gray = np.array(Image.open(os.path.join(GT_DIR, n)).convert("L"))

        with torch.no_grad():
            p_full = torch.sigmoid(unpad(net(xp), ph, pw))[0, 0].cpu().numpy()

            # ---- B. dihedral TTA ------------------------------------------------
            acc = p_full.copy()
            for k in (1, 2, 3):                      # hflip, vflip, hvflip
                xf = x.clone()
                if k & 1:
                    xf = torch.flip(xf, dims=[3])
                if k & 2:
                    xf = torch.flip(xf, dims=[2])
                xfp, phf, pwf = pad16(xf)
                pf = torch.sigmoid(unpad(net(xfp), phf, pwf))[0, 0].cpu().numpy()
                if k & 1:
                    pf = pf[:, ::-1]
                if k & 2:
                    pf = pf[::-1, :]
                acc += pf
            p_tta = acc / 4.0

            # ---- C. run it the way it was trained: 256x256 ----------------------
            xr = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            xrp, phr, pwr = pad16(xr)
            pr = torch.sigmoid(unpad(net(xrp), phr, pwr))
            p_256 = F.interpolate(pr, size=gt.shape, mode="bilinear",
                                  align_corners=False)[0, 0].cpu().numpy()

            # ---- D. 1-pixel shift equivariance ---------------------------------
            xs = torch.roll(xp, shifts=1, dims=3)
            ps = torch.sigmoid(unpad(net(xs), ph, pw))[0, 0].cpu().numpy()
            ps = np.roll(ps, shift=-1, axis=1)

        m = {}
        for tag, p in (("full", p_full), ("tta", p_tta), ("res256", p_256), ("shift", ps)):
            pr_bin = p >= 0.5
            m["dice_" + tag] = dice(pr_bin, gt)
            m["iou_" + tag] = float(np.logical_and(pr_bin, gt).sum()) / \
                (float(np.logical_or(pr_bin, gt).sum()) + EPS) * 100

        m["shift_absdiff_mean"] = float(np.abs(p_full - ps).mean())
        m["shift_flip_frac"] = float((np.round(p_full) != np.round(ps)).mean()) * 100
        m["full_vs_256_flip_frac"] = float((np.round(p_full) != np.round(p_256)).mean()) * 100

        # ---- F. per-region confidence ------------------------------------------
        for lab, tag in ((0, "bg"), (127, "z127"), (255, "z255")):
            sel = gray == lab
            if sel.any():
                v = p_full[sel]
                m["prob_%s_mean" % tag] = float(v.mean())
                m["prob_%s_std" % tag] = float(v.std())
                m["prob_%s_above05" % tag] = float((v >= 0.5).mean()) * 100

        # ---- E. calibration ----------------------------------------------------
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p_full.ravel(), bins) - 1, 0, 9)
        conf = np.array([p_full.ravel()[idx == b].mean() if (idx == b).any() else 0.0
                         for b in range(10)])
        acc_b = np.array([gt.ravel()[idx == b].mean() if (idx == b).any() else 0.0
                          for b in range(10)])
        cnt = np.array([(idx == b).sum() for b in range(10)], dtype=float)
        ece = float((np.abs(conf - acc_b) * cnt).sum() / max(cnt.sum(), 1))
        m["ece"] = ece
        m["bin_conf"] = conf.tolist()
        m["bin_acc"] = acc_b.tolist()
        m["bin_cnt"] = cnt.tolist()
        per_img[n] = m

        np.save(os.path.join(args.save_probs, n.replace(".png", ".npy")), p_full.astype(np.float16))
        probs_all.append(p_full.ravel()[::37])
        labels_all.append(gt.ravel()[::37])

    # ---------------- threshold sweep ----------------
    starts = float(np.mean([v["dice_full"] for v in per_img.values()]))
    sweep = {}
    for t in np.arange(0.05, 0.96, 0.05):
        ds = []
        for n in val:
            p = np.load(os.path.join(args.save_probs, n.replace(".png", ".npy"))).astype(np.float32)
            gt = load_binary(os.path.join(GT_DIR, n))
            ds.append(dice(p >= t, gt))
        sweep["%.2f" % t] = float(np.mean(ds))
    best_t = max(sweep, key=sweep.get)

    agg = {
        "n": len(val),
        "dice_full_0.5": starts,
        "dice_tta_0.5": float(np.mean([v["dice_tta"] for v in per_img.values()])),
        "dice_res256_0.5": float(np.mean([v["dice_res256"] for v in per_img.values()])),
        "dice_shift_0.5": float(np.mean([v["dice_shift"] for v in per_img.values()])),
        "threshold_sweep": sweep,
        "best_threshold": best_t,
        "dice_at_best_threshold": sweep[best_t],
        "threshold_gain": sweep[best_t] - starts,
        "ece_mean": float(np.mean([v["ece"] for v in per_img.values()])),
        "shift_absdiff_mean": float(np.mean([v["shift_absdiff_mean"] for v in per_img.values()])),
        "shift_flip_frac": float(np.mean([v["shift_flip_frac"] for v in per_img.values()])),
        "full_vs_256_flip_frac": float(np.mean([v["full_vs_256_flip_frac"] for v in per_img.values()])),
        "per_image": per_img,
    }
    for lab in ("bg", "z127", "z255"):
        agg["prob_%s_mean" % lab] = float(np.mean([per_img[n]["prob_%s_mean" % lab] for n in val
                                                   if "prob_%s_mean" % lab in per_img[n]]))
        agg["prob_%s_above05" % lab] = float(np.mean([per_img[n]["prob_%s_above05" % lab] for n in val
                                                      if "prob_%s_above05" % lab in per_img[n]]))
    # global reliability diagram
    P = np.concatenate(probs_all)
    G = np.concatenate(labels_all).astype(np.float32)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(P, bins) - 1, 0, 9)
    agg["global_bin_conf"] = [float(P[idx == b].mean()) if (idx == b).any() else 0.0 for b in range(10)]
    agg["global_bin_acc"] = [float(G[idx == b].mean()) if (idx == b).any() else 0.0 for b in range(10)]
    agg["global_bin_cnt"] = [int((idx == b).sum()) for b in range(10)]

    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)

    print("=" * 74)
    print("U-Net probability diagnostics on the %d validation images" % len(val))
    print("  Dice @0.5                  : %.3f" % agg["dice_full_0.5"])
    print("  Dice @0.5 with 4x TTA      : %.3f  (%+.3f)" % (agg["dice_tta_0.5"], agg["dice_tta_0.5"] - starts))
    print("  Dice @0.5 resized to 256   : %.3f  (%+.3f)  <- how it was trained"
          % (agg["dice_res256_0.5"], agg["dice_res256_0.5"] - starts))
    print("  Dice@best threshold %.2f    : %.3f  (%+.3f)"
          % (float(best_t), sweep[best_t], sweep[best_t] - starts))
    print("  ECE (10 bins)              : %.4f" % agg["ece_mean"])
    print("  mean |p - p(shifted)|      : %.4f   pixels whose 0/1 decision flips: %.2f%%"
          % (agg["shift_absdiff_mean"], agg["shift_flip_frac"]))
    print("  pixels changing 0/1 decision between full res and 256: %.2f%%"
          % agg["full_vs_256_flip_frac"])
    print("  mean predicted probability  : background %.3f | 127 zone %.3f | 255 zone %.3f"
          % (agg["prob_bg_mean"], agg["prob_z127_mean"], agg["prob_z255_mean"]))
    print("  fraction >0.5               : background %.2f%% | 127 zone %.2f%% | 255 zone %.2f%%"
          % (agg["prob_bg_above05"], agg["prob_z127_above05"], agg["prob_z255_above05"]))
    print("  threshold sweep             : "
          + " ".join("%s:%.1f" % (k, v) for k, v in list(sweep.items())[::4]))
    print("  -> %s" % args.out)
    print("  -> per-image probability maps in %s" % args.save_probs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
