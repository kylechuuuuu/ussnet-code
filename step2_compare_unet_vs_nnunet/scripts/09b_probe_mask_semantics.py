#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: what values do the masks actually have, and what does ToTensor do to them?"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

USS = "/media/72TB_disk/personal/zk/USSnet"
RAW = os.path.join(USS, "UltrasoundProstate", "masks")
SPLIT = os.path.join(USS, "unet", "datasets", "ultrasound_prostate_73", "val", "masks")
PRED = os.path.join(USS, "unet", "results", "ultrasound_prostate_73")

name = sorted(n for n in os.listdir(SPLIT) if n.endswith(".png"))[0]
print("image:", name)
for tag, p in (("raw", os.path.join(RAW, name)), ("split", os.path.join(SPLIT, name)),
               ("pred", os.path.join(PRED, name))):
    a = np.array(Image.open(p).convert("L"))
    print("%-6s unique=%s  shape=%s  >127 count=%d  >0 count=%d"
          % (tag, np.unique(a)[:6], a.shape, int((a > 127).sum()), int((a > 0).sum())))
    t = ToTensor()(Image.open(p).convert("L"))
    print("        ToTensor dtype=%s min=%.4f max=%.4f uniq=%s"
          % (t.dtype, float(t.min()), float(t.max()),
             np.unique(t.numpy())[:6]))

# --- replicate test.py F1 exactly for this image -------------------------------
gt = ToTensor()(Image.open(os.path.join(SPLIT, name)).convert("L")).unsqueeze(0)
pr = ToTensor()(Image.open(os.path.join(PRED, name)).convert("L")).unsqueeze(0)
prb = (pr >= 0.5).float()
gtb = (gt >= 0.5).float()


def f1(pred, g):
    tp = (g * pred).sum()
    fp = ((1 - g) * pred).sum()
    fn = (g * (1 - pred)).sum()
    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)
    return float(2 * p * r / (p + r + 1e-7) * 100), float(p * 100), float(r * 100)


print("test.py-style  F1 with GT binarised >=0.5 :", f1(prb, gtb))
print("train.py-style F1 with raw soft GT       :", f1(prb, gt))

from skimage import morphology
for tag, g in (("binarised", gtb[0].numpy() > 0.5), ("soft>0", gt[0].numpy() > 0)):
    sk_g = morphology.skeletonize(g)
    sk_p = morphology.skeletonize(prb[0].numpy() > 0.5)
    inter = np.logical_and(sk_g, sk_p).sum()
    print("topo vs %-10s: |skel_gt|=%d |skel_pred|=%d overlap=%d corr=%.2f%% com=%.2f%% qual=%.2f%%"
          % (tag, sk_g.sum(), sk_p.sum(), inter,
             inter / max(sk_p.sum(), 1) * 100, inter / max(sk_g.sum(), 1) * 100,
             inter / max(sk_p.sum() + sk_g.sum() - inter, 1) * 100))
