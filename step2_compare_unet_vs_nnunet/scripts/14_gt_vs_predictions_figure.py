#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_gt_vs_predictions_figure.py

"Ground truth vs the two models" for USSnet (U-Net baseline vs nnU-Net 2d),
re-worked for CLARITY: no zoomed crops, no speckle-noise error fills.

Writes into ../figures/:
  fig5_gt_vs_models_overlay.png
      8 cases, 2 cases per row, each case = 3 big panels:
        GT       : image dimmed + GT area filled green + crisp green outline
        U-Net    : image dimmed + U-Net area filled magenta + magenta outline,
                   GT outline drawn on top in green for instant comparison
        nnU-Net  : same, cyan
      Only the areas are coloured (no per-pixel FN/FP speckle), so the shape is
      what the eye sees.  Cases = 3 worst for U-Net + 3 where U-Net wins +
      median + best.
  fig6_gt_vs_models_difference.png
      the same cases, pure segmentation-difference maps on a WHITE background
      (no ultrasound at all):  green = true positive, red = FN (missed),
      blue = FP (over-segmented).  Two columns per case (U-Net, nnU-Net).

Usage:
    ~/anaconda3/envs/zk/bin/python scripts/14_gt_vs_predictions_figure.py
"""
import os
import sys
import csv

import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_lib import load_binary, read_split  # noqa: E402

STEP2 = os.path.dirname(HERE)
USS = os.path.dirname(STEP2)
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
FIG_DIR = os.path.join(STEP2, "figures")
MET_DIR = os.path.join(STEP2, "metrics")
STEP3_RES = os.path.join(USS, "step3_failure_analysis", "results")

UNET_DIR = os.path.join(USS, "unet", "results", "ultrasound_prostate_73")
NN_DIR = os.path.join(USS, "step1_nnunet_baseline", "predictions",
                      "nnunet_2d_fold0_checkpoint_best_png")

C_GT_OUT = "#00b050"      # GT outline (dark green, readable on grayscale)
C_GT_FILL = np.array([60, 220, 90])
C_UNET_OUT = "#e000a8"
C_UNET_FILL = np.array([255, 60, 200])
C_NN_OUT = "#0080ff"
C_NN_FILL = np.array([60, 170, 255])


def read_col(path, key):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if key in r:
                try:
                    out[r["name"]] = float(r[key])
                except ValueError:
                    pass
    return out


def panel_filled(ax, gray, fill_mask, fill_rgb, title, gt_for_outline=None, lw=1.6):
    """Dimmed grayscale + one solid coloured area + crisp outlines."""
    base = (gray.astype(np.float32) * 0.55)[..., None] * np.ones((1, 1, 3), np.float32)
    m = fill_mask
    base[m] = 0.45 * base[m] + 0.55 * fill_rgb
    ax.imshow(base.astype(np.uint8))
    ax.contour(fill_mask, levels=[.5], colors=[fill_rgb / 255.0], linewidths=lw)
    if gt_for_outline is not None:
        ax.contour(gt_for_outline, levels=[.5], colors=C_GT_OUT, linewidths=lw)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def panel_diff(ax, gt, pred, title):
    """Clean difference map on white: TP green, FN red, FP blue, TN light gray."""
    img = np.ones(gt.shape + (3,), np.float32) * 0.95
    tp = np.logical_and(gt, pred)
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(pred, ~gt)
    img[fn] = np.array([235, 60, 60]) / 255.0
    img[fp] = np.array([60, 120, 245]) / 255.0
    img[tp] = np.array([40, 190, 90]) / 255.0
    ax.imshow(img)
    ax.contour(gt, levels=[.5], colors="#1a1a1a", linewidths=1.1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _, val = read_split(USS)

    d_u = read_col(os.path.join(MET_DIR, "unet_73_per_image.csv"), "dice")
    d_n = read_col(os.path.join(MET_DIR, "nnunet_2d_fold0_checkpoint_best_per_image.csv"), "dice")
    b_u = read_col(os.path.join(STEP3_RES, "unet_73_error_decomposition.csv"), "bdice_1px")
    b_n = read_col(os.path.join(STEP3_RES,
                                "nnunet_2d_fold0_checkpoint_best_error_decomposition.csv"), "bdice_1px")

    common = [n for n in val if n in d_u and n in d_n]
    by_u = sorted(common, key=lambda n: d_u[n])
    worst3 = by_u[:3]
    unet_wins = sorted([n for n in common if d_u[n] > d_n[n]],
                       key=lambda n: d_n[n] - d_u[n])[:3]
    order, seen = [], set()
    for n in worst3 + unet_wins + [by_u[len(by_u) // 2], by_u[-1]]:
        if n not in seen:
            seen.add(n)
            order.append(n)

    def tag_of(n):
        if n in worst3:
            return "U-Net 最差"
        if n in unet_wins:
            return "U-Net 胜出"
        return "中位" if n == by_u[len(by_u) // 2] else "最好"

    cache = {}
    for n in order:
        cache[n] = (np.array(Image.open(os.path.join(IMG_DIR, n)).convert("L")),
                    load_binary(os.path.join(GT_DIR, n)),
                    load_binary(os.path.join(UNET_DIR, n)),
                    load_binary(os.path.join(NN_DIR, n)))

    print("cases:", ", ".join("%s(%s)" % (n[:-4], tag_of(n)) for n in order))

    # ------------------------------------------------- fig5: filled overlay
    nrow = (len(order) + 1) // 2
    fig, axes = plt.subplots(nrow, 6, figsize=(19.5, 3.9 * nrow))
    axes = np.atleast_2d(axes)
    for i, n in enumerate(order):
        r, c0 = divmod(i, 2)
        c0 *= 3
        gray, gt, pu, pn = cache[n]
        panel_filled(axes[r, c0], gray, gt, C_GT_FILL,
                     "%s  (%s)\nGROUND TRUTH" % (n[:-4], tag_of(n)))
        panel_filled(axes[r, c0 + 1], gray, pu, C_UNET_FILL,
                     "U-Net   Dice %.2f\n边界Dice@1px %.1f%%"
                     % (d_u[n], b_u.get(n, np.nan)), gt_for_outline=gt)
        panel_filled(axes[r, c0 + 2], gray, pn, C_NN_FILL,
                     "nnU-Net   Dice %.2f\n边界Dice@1px %.1f%%"
                     % (d_n[n], b_n.get(n, np.nan)), gt_for_outline=gt)
    for j in range(len(order), nrow * 2):
        r, c0 = divmod(j, 2)
        for c in range(3):
            axes[r, c0 * 3 + c].axis("off")

    handles = [Patch(facecolor=C_GT_FILL / 255.0, edgecolor=C_GT_OUT, label="ground truth"),
               Patch(facecolor=C_UNET_FILL / 255.0, edgecolor=C_UNET_OUT, label="U-Net prediction"),
               Patch(facecolor=C_NN_FILL / 255.0, edgecolor=C_NN_OUT, label="nnU-Net prediction"),
               Patch(facecolor="none", edgecolor=C_GT_OUT, lw=2,
                     label="green line = ground-truth outline (drawn on every panel)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.972))
    fig.suptitle("USSnet — U-Net vs nnU-Net against the ground truth, one case per block "
                 "(validation fold 0)", fontsize=14, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f5 = os.path.join(FIG_DIR, "fig5_gt_vs_models_overlay.png")
    fig.savefig(f5, dpi=125)
    plt.close(fig)

    # ------------------------------------------------- fig6: difference maps
    nrow = (len(order) + 1) // 2
    fig, axes = plt.subplots(nrow, 4, figsize=(13.0, 3.5 * nrow))
    axes = np.atleast_2d(axes)
    for i, n in enumerate(order):
        r, c0 = divmod(i, 2)
        c0 *= 2
        _, gt, pu, pn = cache[n]
        panel_diff(axes[r, c0], gt, pu,
                   "%s  (%s)\nU-Net   Dice %.2f" % (n[:-4], tag_of(n), d_u[n]))
        panel_diff(axes[r, c0 + 1], gt, pn,
                   "nnU-Net   Dice %.2f" % d_n[n])
    for j in range(len(order), nrow * 2):
        r, c0 = divmod(j, 2)
        for c in range(2):
            axes[r, c0 * 2 + c].axis("off")
    handles = [Patch(facecolor=np.array([40, 190, 90]) / 255.0, label="correct (TP)"),
               Patch(facecolor=np.array([235, 60, 60]) / 255.0, label="missed (FN)"),
               Patch(facecolor=np.array([60, 120, 245]) / 255.0, label="over-segmented (FP)"),
               Patch(facecolor="none", edgecolor="#1a1a1a", lw=2, label="black line = ground-truth outline")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.968))
    fig.suptitle("USSnet — difference against the ground truth, no ultrasound background "
                 "(validation fold 0)", fontsize=14, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    f6 = os.path.join(FIG_DIR, "fig6_gt_vs_models_difference.png")
    fig.savefig(f6, dpi=125)
    plt.close(fig)

    print("wrote", f5)
    print("wrote", f6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
