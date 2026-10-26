#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is there TRAINING SIGNAL for the boundary-band loss?

The fine-tune can only learn from what it sees on fold-0's 71 training images.
step6's REPORT gives train bdice@1px 99.65 -- if that is also true pointwise,
the band loss is already ~0 on every training patch and the fine-tune has
nothing to optimise.  Measure it, do not assume it.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

USS = "/media/72TB_disk/personal/zk/USSnet"
sys.path.insert(0, os.path.join(USS, "step6_sandwich_squeeze", "trainer"))
sys.path.insert(0, os.path.join(USS, "step9_boundary_mechanisms", "trainer"))
import importlib.util                                                   # noqa: E402
from boundary_losses import band_terms                                  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "arb", os.path.join(USS, "step6_sandwich_squeeze", "tools", "63_arbitrate.py"))
arb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arb)

NPZ = os.path.join(USS, "step6_sandwich_squeeze", "predictions",
                   "sq2resencw25_checkpoint_best_base_nifti")
GTD = os.path.join(USS, "UltrasoundProstate", "masks")
train, val = arb.read_split(USS)
dev = torch.device("cuda")
D = 12.0


def as_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(dev)


def collect(names, tag, limit=None):
    outs = []
    for name in names[:limit]:
        stem = arb.stem_of(name)
        gt = np.array(Image.open(os.path.join(GTD, name)).convert("L")) > 0
        p = arb.align_prob(arb.load_prob(NPZ, stem), gt.shape)
        g = as_t(gt)
        q = as_t(p)
        with torch.no_grad():
            lo, li = band_terms(q, g, D)
        # pointwise boundary displacement: p50 / p90 / mean of |t_gt|
        pr = (q > 0.5).float()
        bd_g = (F.max_pool2d(g, 3, 1, 1) - (-F.max_pool2d(-g, 3, 1, 1))).clamp(0, 1)
        bd_p = (F.max_pool2d(pr, 3, 1, 1) - (-F.max_pool2d(-pr, 3, 1, 1))).clamp(0, 1)
        # fraction of GT boundary pixels with a predicted boundary pixel within tol
        rec = {}
        for tol in (1, 2, 3, 5, 10):
            rec[tol] = float((bd_g * F.max_pool2d(bd_p, 2 * tol + 1, 1, tol)).sum()
                             / bd_g.sum().clamp_min(1.0))
        dice = float(2 * (pr * g).sum() / (pr.sum() + g.sum() + 1e-6))
        outs.append((stem, float(lo), float(li), dice, rec))
    print("\n== %s  (n=%d, D=%g) ==" % (tag, len(outs), D))
    print("%-26s %8s %8s %8s | %6s %6s %6s %6s %6s | %6s"
          % ("case", "L_out", "L_in", "sum", "rec1", "rec2", "rec3", "rec5", "rec10", "dice"))
    for stem, lo, li, dice, rec in outs:
        print("%-26s %8.3f %8.3f %8.3f | %6.3f %6.3f %6.3f %6.3f %6.3f | %6.4f"
              % (stem, lo, li, lo + li, rec[1], rec[2], rec[3], rec[5], rec[10], dice))
    s = np.array([r[1] + r[2] for r in outs])
    r1 = np.array([r[4][1] for r in outs])
    print("MEAN  L_band=%.3f (median %.3f, p10 %.3f, p90 %.3f)   rec@1px=%.3f"
          % (s.mean(), np.median(s), np.percentile(s, 10), np.percentile(s, 90), r1.mean()))
    return s


collect(val, "VAL (the failure)")
collect(train, "TRAIN fold-0 (what the fine-tune would see)", limit=12)
