#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent recomputation of Dice/F1 directly from the saved prediction PNGs.

Deliberately does NOT reuse test.py's code path, so any bug there shows up as a mismatch.
Also reports IoU, per-image size bias, and a ground-truth-vs-prediction area regression.
"""
import os

import numpy as np
from PIL import Image


VAL_IMG = "datasets/ultrasound_prostate_73/val/images"
VAL_MSK = "datasets/ultrasound_prostate_73/val/masks"
PRED = "results/ultrasound_prostate_73"


def main():
    names = sorted(n for n in os.listdir(PRED) if n.endswith(".png"))
    rows = []
    for n in names:
        gt_p = os.path.join(VAL_MSK, n)
        if not os.path.exists(gt_p):
            print("skip (no gt)", n)
            continue
        gt = np.array(Image.open(gt_p).convert("L")) > 127
        pr = np.array(Image.open(os.path.join(PRED, n)).convert("L")) > 127
        if gt.shape != pr.shape:
            print("!! shape mismatch", n, gt.shape, pr.shape)
            continue
        tp = int((gt & pr).sum())
        fp = int((~gt & pr).sum())
        fn = int((gt & ~pr).sum())
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        rows.append((n, dice * 100, iou * 100, prec * 100, rec * 100,
                     int(gt.sum()), int(pr.sum())))

    d = np.array([r[1] for r in rows])
    i = np.array([r[2] for r in rows])
    p = np.array([r[3] for r in rows])
    r_ = np.array([r[4] for r in rows])
    gta = np.array([r[5] for r in rows], dtype=float)
    pra = np.array([r[6] for r in rows], dtype=float)

    print("n =", len(rows))
    print("recomputed  Dice/F1 : mean %.2f  median %.2f  min %.2f  max %.2f"
          % (d.mean(), np.median(d), d.min(), d.max()))
    print("recomputed  IoU     : mean %.2f" % i.mean())
    print("recomputed  Prec    : mean %.2f" % p.mean())
    print("recomputed  Rec     : mean %.2f" % r_.mean())
    print()
    ratio = pra / np.maximum(gta, 1)
    print("pred/gt area ratio  : mean %.3f  median %.3f  min %.3f  max %.3f"
          % (ratio.mean(), np.median(ratio), ratio.min(), ratio.max()))
    corr = np.corrcoef(gta, pra)[0, 1]
    print("corr(gt_area, pred_area) = %.3f" % corr)
    print()
    print("smallest-GT images (hardest):")
    for n, dd, ii, pp, rr, ga, pa in sorted(rows, key=lambda x: x[5])[:5]:
        print("  gt_area %6d  pred_area %6d  Dice %5.2f  P %5.2f  R %5.2f  %s"
              % (ga, pa, dd, pp, rr, n))


if __name__ == "__main__":
    main()
