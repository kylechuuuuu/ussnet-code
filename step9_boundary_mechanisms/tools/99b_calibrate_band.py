#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate the boundary-band loss magnitude on the REAL val predictions.

Why: the band term must be comparable to L_base + 0.25(L_c + L_a) (~0.1-0.3)
at the START of the fine-tune, otherwise `weight` is meaningless.  Then report
the gradient share it would have.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze/trainer")
sys.path.insert(0, "/media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze/tools")
sys.path.insert(0, "/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/trainer")

from boundary_losses import band_terms   # noqa: E402

USS = "/media/72TB_disk/personal/zk/USSnet"
NPZ = os.path.join(USS, "step6_sandwich_squeeze", "predictions",
                   "sq2resencw25_checkpoint_best_base_nifti")
GTD = os.path.join(USS, "UltrasoundProstate", "masks")
sys.path.insert(0, os.path.join(USS, "step6_sandwich_squeeze", "tools"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "arb", os.path.join(USS, "step6_sandwich_squeeze", "tools", "63_arbitrate.py"))
arb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arb)

train, val = arb.read_split(USS)
print("val cases:", len(val))

rows = []
for name in val:
    stem = arb.stem_of(name)
    gt = np.array(Image.open(os.path.join(GTD, name if name.endswith(".png") else name + ".png")).convert("L")) > 0
    p = arb.align_prob(arb.load_prob(NPZ, stem), gt.shape)
    rows.append((stem, gt, p))
print("loaded", len(rows), "cases; gt shape", rows[0][1].shape)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", dev)


def as_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(dev)


print("\n%-8s %8s %8s %8s | %8s %8s | %8s %8s"
      % ("D", "L_out", "L_in", "sum", "out@.9", "in@.9", "gt+1px", "gt+2px"))
for D in (8.0, 12.0, 16.0, 20.0):
    acc = {k: [] for k in ("out", "in", "out9", "in9")}
    for stem, gt, p in rows:
        g = as_t(gt)
        q = as_t(p)
        with torch.no_grad():
            lo, li = band_terms(q, g, D)
            h = (q > 0.9).float()
            lo9, li9 = band_terms(h, g, D)
        acc["out"].append(float(lo))
        acc["in"].append(float(li))
        acc["out9"].append(float(lo9))
        acc["in9"].append(float(li9))
    # what a nearly-perfect model would score: GT shifted by 1 and 2 px
    qs = [as_t(rows[0][1])]
    g0 = as_t(rows[0][1])
    s1 = F.max_pool2d(g0, 3, 1, 1)
    s2 = F.max_pool2d(g0, 5, 1, 2)
    with torch.no_grad():
        b1 = sum(float(v) for v in band_terms(s1, g0, D))
        b2 = sum(float(v) for v in band_terms(s2, g0, D))
    print("%-8g %8.3f %8.3f %8.3f | %8.3f %8.3f | %8.3f %8.3f"
          % (D, np.mean(acc["out"]), np.mean(acc["in"]),
             np.mean(acc["out"]) + np.mean(acc["in"]),
             np.mean(acc["out9"]), np.mean(acc["in9"]), b1, b2))

# ---------------------------------------------------------------- GPU timing
print("\n--- GPU timing of the band term at training resolution ---")
big = torch.randn(4, 6, 640, 1024, device=dev, requires_grad=True)
tgt = torch.zeros(4, 1, 640, 1024, device=dev)
tgt[:, :, 200:500, 300:700] = 1.0
from boundary_losses import BoundaryBandLoss
m = BoundaryBandLoss(D=12.0, weight=1.0, scales=(0, 1, 2)).to(dev)
outs = [big, torch.randn(4, 6, 320, 512, device=dev, requires_grad=True),
        torch.randn(4, 6, 160, 256, device=dev, requires_grad=True)]
tgts = [tgt, F.max_pool2d(tgt, 2, 2), F.max_pool2d(tgt, 4, 4)]
for _ in range(3):
    l, _d = m(outs, tgts, [0.5, 0.25, 0.125])
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    l, _d = m(outs, tgts, [0.5, 0.25, 0.125])
    l.backward()
torch.cuda.synchronize()
print("3 scales + backward: %.1f ms / iteration" % ((time.time() - t0) / 10 * 1e3))
print("band loss value on the fake batch: %.4f" % float(l))
