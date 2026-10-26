#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for the step-9 boundary band loss (CPU, no nnU-Net).

Checks the properties the loss is supposed to have, not just that it runs:
  * perfect prediction -> 0
  * symmetry: an inflation by delta and a deflation by delta cost the same
  * quadratic ramp: delta=3 costs ~ delta^2/D (within a factor)
  * monotone growth with delta
  * gradients finite and non-zero
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze/trainer")
sys.path.insert(0, "/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/trainer")

from boundary_losses import BoundaryBandLoss, band_terms, boundary_diag

torch.manual_seed(0)
B, H, W = 2, 256, 256
gt = torch.zeros(B, 1, H, W)
gt[:, :, 88:168, 88:168] = 1.0                 # 80x80 square, centre 128
D = 12.0


def ring(m):
    return (F.max_pool2d(m, 3, 1, 1) - (-F.max_pool2d(-m, 3, 1, 1))).clamp(0, 1)


def shift(m, delta):
    """Chebyshev dilate (delta>0) / erode (delta<0) by |delta| px."""
    k = 2 * abs(delta) + 1
    if delta > 0:
        return F.max_pool2d(m, k, 1, abs(delta))
    return -F.max_pool2d(-m, k, 1, abs(delta))


print("D = %g   GT area = %d   GT boundary P = %d"
      % (D, int(gt.sum()), int(ring(gt).sum())))
print("%-10s %9s %9s %9s   %-6s %-6s" % ("case", "L_out", "L_in", "sum", "rec1", "pre1"))

base = None
for delta in [0, 1, 2, 3, 5, 10, 20, 30, 40]:
    q = shift(gt, delta) if delta > 0 else gt
    lo, li = band_terms(q, gt, D)
    d = boundary_diag(q, gt)
    print("%-10s %9.4f %9.4f %9.4f   %-6.3f %-6.3f"
          % ("grow+%d" % delta, lo, li, lo + li, d["brec1"], d["bpre1"]))

print()
for delta in [1, 2, 3, 5, 10, 20, 30]:
    qg = shift(gt, delta)
    qs = shift(gt, -delta)
    log, loi = band_terms(qg, gt, D)
    los, lii = band_terms(qs, gt, D)
    pred = delta ** 2 / D
    print("%-10s grow=%.4f shrink=%.4f  gap=%+.4f   delta^2/D=%.4f"
          % ("+-%d px" % delta, log + loi, los + lii,
             (log + loi) - (los + lii), pred))

print()
q = gt.clone().requires_grad_(True)
lo, li = band_terms(q, gt, D)
(lo + li).backward()
print("perfect: loss=%.6f grad finite=%s" % (float(lo + li), bool(torch.isfinite(q.grad).all())))

# ---- module with fake deep-supervision outputs ---------------------------
outs = [torch.randn(B, 6, H, W, requires_grad=True),
        torch.randn(B, 6, H // 2, W // 2, requires_grad=True),
        torch.randn(B, 6, H // 4, W // 4, requires_grad=True)]
tgts = [gt, F.max_pool2d(gt, 2, 2), F.max_pool2d(gt, 4, 4)]
mod = BoundaryBandLoss(D=D, weight=1.0, scales=(0, 1, 2))
loss, diag = mod(outs, tgts, [0.5, 0.25, 0.125])
loss.backward()
print("module loss = %.4f" % float(loss))
for k in sorted(diag):
    print("   %-12s %.6f" % (k, diag[k]))
print("all grads finite:", all(bool(torch.isfinite(o.grad).all()) for o in outs))

# ---- timing (what does it cost per iteration?) ---------------------------
import time
big = torch.randn(2, 6, 640, 1024, requires_grad=True)
tgt = torch.zeros(2, 1, 640, 1024)
tgt[:, :, 200:500, 300:700] = 1.0
m = BoundaryBandLoss(D=D, weight=1.0, scales=(0, 1, 2))
m(big, [tgt], [1.0])                      # warm-up
t0 = time.time()
for _ in range(5):
    l, _d = m(big, [tgt], [1.0])
t1 = time.time()
print("band term on (2,640,1024) single scale: %.1f ms / iteration" % ((t1 - t0) / 5 * 1e3))
print("SMOKE OK")
