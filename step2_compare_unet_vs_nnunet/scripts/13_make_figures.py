#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_compare_unet_vs_nnunet/scripts/13_make_figures.py

Figures for the U-Net vs nnU-Net comparison:
  fig1_metric_boxplots.png    per-image metric distributions
  fig2_dice_scatter.png       image-by-image Dice, method A vs method B
  fig3_qualitative.png        best / median / worst case overlays
  fig4_error_maps.png         FN/FP maps of the same cases
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the patient filenames / titles are Chinese, so a CJK-capable font is required
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_lib import load_binary, read_split  # noqa: E402

STEP2 = os.path.dirname(HERE)
USS = os.path.dirname(STEP2)
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="prediction folder per tag")
    ap.add_argument("--metrics-dir", default=os.path.join(STEP2, "metrics"))
    ap.add_argument("--fig-dir", default=os.path.join(STEP2, "figures"))
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    per, vals = {}, {}
    for t in args.tags:
        rows = list(csv.DictReader(open(os.path.join(args.metrics_dir, "%s_per_image.csv" % t))))
        per[t] = {r["name"]: r for r in rows}
        vals[t] = {k: np.array([float(r[k]) for r in rows]) for k in
                   ("dice", "precision", "recall", "iou", "hd95", "assd",
                    "correctness", "completeness", "quality")}
    common = sorted(set.intersection(*[set(per[t]) for t in args.tags]))

    # ---------------- fig 1: boxplots ----------------
    keys = ["dice", "precision", "recall", "iou", "hd95", "assd", "quality"]
    titles = ["Dice (%)", "Precision (%)", "Recall (%)", "IoU (%)",
              "HD95 (px)", "ASSD (px)", "Topo quality (%)"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, k, ti in zip(axes.ravel(), keys, titles):
        data = [[per[t][n][k] and float(per[t][n][k]) for n in common] for t in args.tags]
        bp = ax.boxplot(data, tick_labels=args.labels, patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cfe3f7")
        for i, t in enumerate(args.tags):
            ax.scatter(np.full(len(common), i + 1) + np.random.RandomState(0).uniform(-.08, .08, len(common)),
                       [float(per[t][n][k]) for n in common], s=8, alpha=.5, color="#1f3b57")
        ax.set_title(ti)
        ax.grid(alpha=.25)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Per-image metrics on the shared validation split (n=%d)" % len(common))
    fig.tight_layout()
    f1 = os.path.join(args.fig_dir, "fig1_metric_boxplots.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---------------- fig 2: scatter ----------------
    f2 = None
    if len(args.tags) >= 2:
        a = np.array([float(per[args.tags[0]][n]["dice"]) for n in common])
        b = np.array([float(per[args.tags[1]][n]["dice"]) for n in common])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(a, b, s=22, color="#1f3b57")
        lo, hi = min(a.min(), b.min()) - 1, 100
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(args.labels[0] + " Dice (%)")
        ax.set_ylabel(args.labels[1] + " Dice (%)")
        n_win = int((b > a).sum())
        ax.set_title("per-image Dice — %s wins %d / %d" % (args.labels[1], n_win, len(common)))
        ax.grid(alpha=.25)
        fig.tight_layout()
        f2 = os.path.join(args.fig_dir, "fig2_dice_scatter.png")
        fig.savefig(f2, dpi=150)
        plt.close(fig)

    # ---------------- fig 3+4: qualitative, error maps ----------------
    order = sorted(common, key=lambda n: float(per[args.tags[0]][n]["dice"]))
    picks = [order[0], order[len(order) // 2], order[-1]]
    pick_names = ["worst", "median", "best"]

    ncol = 1 + len(args.tags)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(4.2 * ncol, 4.0 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, (n, tag) in enumerate(zip(picks, pick_names)):
        gray = np.array(Image.open(os.path.join(IMG_DIR, n)).convert("L"))
        gt = load_binary(os.path.join(GT_DIR, n))
        ax = axes[r, 0]
        ax.imshow(gray, cmap="gray")
        ax.contour(gt, levels=[.5], colors="lime", linewidths=1.2)
        ax.set_title("%s (%s)\nGT Dice %.2f" % (n, tag, float(per[args.tags[0]][n]["dice"])))
        ax.axis("off")
        for c, t in enumerate(args.tags, start=1):
            pdir = args.dirs[c - 1]
            pdir = pdir if os.path.isabs(pdir) else os.path.join(STEP2, pdir)
            pred = load_binary(os.path.join(pdir, n))
            ax = axes[r, c]
            ax.imshow(gray, cmap="gray")
            ax.contour(gt, levels=[.5], colors="lime", linewidths=1.2)
            ax.contour(pred, levels=[.5], colors="magenta", linewidths=1.2)
            ax.set_title("%s — Dice %.2f" % (args.labels[c - 1], float(per[t][n]["dice"])))
            ax.axis("off")
    fig.suptitle("green = ground truth, magenta = prediction")
    fig.tight_layout()
    f3 = os.path.join(args.fig_dir, "fig3_qualitative_overlays.png")
    fig.savefig(f3, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(picks), len(args.tags), figsize=(4.2 * len(args.tags), 4.0 * len(picks)), squeeze=False)
    for r, n in enumerate(picks):
        gray = np.array(Image.open(os.path.join(IMG_DIR, n)).convert("L"))
        gt = load_binary(os.path.join(GT_DIR, n))
        for c, t in enumerate(args.tags):
            pdir = args.dirs[c]
            pdir = pdir if os.path.isabs(pdir) else os.path.join(STEP2, pdir)
            pred = load_binary(os.path.join(pdir, n))
            rgb = np.stack([gray] * 3, -1).astype(np.float32)
            fn = np.logical_and(gt, ~pred)
            fp = np.logical_and(pred, ~gt)
            rgb[fn] = [255, 40, 40]        # missed gland  -> red
            rgb[fp] = [60, 120, 255]       # over-segmented -> blue
            ax = axes[r, c]
            ax.imshow(rgb.astype(np.uint8))
            ax.set_title("%s\n%s  FN %d px / FP %d px"
                         % (n, args.labels[c], int(fn.sum()), int(fp.sum())), fontsize=8)
            ax.axis("off")
    fig.suptitle("red = false negative (missed), blue = false positive (over-segmented)")
    fig.tight_layout()
    f4 = os.path.join(args.fig_dir, "fig4_error_maps.png")
    fig.savefig(f4, dpi=150)
    plt.close(fig)

    print("figures ->", args.fig_dir)
    for p in (f1, f2, f3, f4):
        if p:
            print("  ", os.path.basename(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
