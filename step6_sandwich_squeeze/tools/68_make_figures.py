#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/tools/68_make_figures.py

Two comparison figures for the step-6 conclusion:

  figures/step6_overlays.png : three validation cases (largest / median / smallest
      boundary-Dice gain over nnU-Net+snap) as image + contours
      (GT / nnU-Net / ours) + the learned interval boundaries (cautious/amplified).
  figures/step6_bars.png     : Dice / HD95 / ASSD / bdice@1px across all models.

usage:  python tools/68_make_figures.py            # defaults to the resenc winner
        STEP6_FIG_TAG=sq2w25s_checkpoint_best python tools/68_make_figures.py
"""
import importlib.util
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STEP6 = os.path.dirname(HERE)
USS = os.path.dirname(STEP6)

spec = importlib.util.spec_from_file_location("arb63", os.path.join(HERE, "63_arbitrate.py"))
arb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arb)

TAG = os.environ.get("STEP6_FIG_TAG", "sq2resencw25_checkpoint_best")
OURS_DIR = os.path.join(STEP6, "predictions", TAG + "_arb_val", "band_lrn")
NNT_DIR = os.path.join(USS, "step1_nnunet_baseline", "predictions",
                       "nnunet_2d_fold0_checkpoint_best_png")
NPZ = {k: os.path.join(STEP6, "predictions", TAG + "_%s_nifti" % k)
       for k in ("base", "cautious", "amplified")}
FIGS = os.path.join(STEP6, "figures")
os.makedirs(FIGS, exist_ok=True)


def load_mask_png(path):
    return np.array(Image.open(path).convert("L")) > 127


def overlays():
    a_o = json.load(open(os.path.join(STEP6, "results", "arbitrate_%s_val.json" % TAG)))["rows"]
    a_n = json.load(open(os.path.join(STEP6, "results", "arbitrate_calib_nnunet_val.json")))["rows"]
    d_o = {r["name"]: r for r in a_o}
    d_n = {r["name"]: r for r in a_n}
    common = sorted(set(d_o) & set(d_n))
    gains = {n: d_o[n]["bdice_band_lrn"] - d_n[n]["bdice_free"] for n in common}
    order = sorted(common, key=lambda n: gains[n], reverse=True)
    pick = [order[0], order[len(order) // 2], order[-1]]

    fig, axes = plt.subplots(len(pick), 2, figsize=(11.5, 4.8 * len(pick)), dpi=115)
    for i, stem in enumerate(pick):
        img = arb.load_image(stem)
        gt = arb.s57.load_gt(stem)
        nnt = load_mask_png(os.path.join(NNT_DIR, stem + ".png"))
        ours = load_mask_png(os.path.join(OURS_DIR, stem + ".png"))
        pb = arb.align_prob(arb.load_prob(NPZ["base"], stem), gt.shape)
        pc = arb.align_prob(arb.load_prob(NPZ["cautious"], stem), gt.shape)
        pa = arb.align_prob(arb.load_prob(NPZ["amplified"], stem), gt.shape)
        base = arb.s57.largest_cc(pb > 0.5)
        Min = base & (pc > 0.5)
        Mout = base | (pa > 0.5)
        ys, xs = np.nonzero(gt)
        y0, y1 = max(0, ys.min() - 30), min(img.shape[0], ys.max() + 30)
        x0, x1 = max(0, xs.min() - 30), min(img.shape[1], xs.max() + 30)
        crops = [(0, img.shape[0], 0, img.shape[1]), (y0, y1, x0, x1)]
        for j, (cy0, cy1, cx0, cx1) in enumerate(crops):
            ax = axes[i, j]
            ax.imshow(img[cy0:cy1, cx0:cx1], cmap="gray")
            for m, col, ls, lw in ((gt, "lime", "solid", 1.3),
                                   (nnt, "deepskyblue", "solid", 1.3),
                                   (ours, "red", "solid", 1.6),
                                   (Min, "orange", "dashed", 0.9),
                                   (Mout, "orange", "dashed", 0.9)):
                sub = m[cy0:cy1, cx0:cx1]
                if sub.any():
                    ax.contour(sub.astype(float), levels=[0.5], colors=[col],
                               linewidths=lw, linestyles=ls)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("full" if j == 0 else "zoom", fontsize=10)
        axes[i, 0].set_ylabel("case %s\nbdice %.1f -> %.1f"
                              % (stem.split("_")[-1], d_n[stem]["bdice_free"],
                                 d_o[stem]["bdice_band_lrn"]), fontsize=8)
    h = [plt.Line2D([0], [0], color=c, lw=2, ls="-" if c != "orange" else (0, (5, 3)))
         for c in ("lime", "deepskyblue", "red", "orange")]
    fig.legend(h, ["GT", "nnU-Net 1000ep (+ same snap)", "ours: SQ2-res-w25 + interval arbitration",
                   "learned interval (cautious / amplified)"],
               loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    out = os.path.join(FIGS, "step6_overlays.png")
    fig.savefig(out)
    plt.close(fig)
    print("[68] wrote", out)


def bars():
    labels = ["U-Net", "nnU-Net\n1000ep", "step4 armB\n500ep", "SQ2\n500ep",
              "SQ2w25\n500ep", "SQ2w25\n1000ep", "SQ2-res-w25\n500ep", "ens B\n(w25s+w25L)"]
    dice = [93.006, 96.106, 96.159, 95.92, 96.08, 96.13, 96.377, 96.154]
    hd95 = [52.26, 30.95, 30.62, 31.24, 28.87, 30.24, 28.20, 29.50]
    assd = [11.25, 5.60, 5.60, 5.71, 5.39, 5.51, 5.15, 5.39]
    bdice = [23.25, 35.23, 35.17, 35.89, 36.50, 36.27, 36.88, 36.57]
    colors = ["#9e9e9e", "#607d8b", "#546e7a", "#ffb74d", "#ff9800", "#f57c00", "#d32f2f", "#e57373"]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.8), dpi=115)
    data = [("Dice (%, higher better)", dice), ("HD95 (px, lower better)", hd95),
            ("ASSD (px, lower better)", assd), ("boundary Dice@1px (%, higher better)", bdice)]
    for ax, (title, vals) in zip(axs, data):
        bars_ = ax.bar(range(len(vals)), vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        top = max(vals)
        for b, v in zip(bars_, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.012, "%.2f" % v,
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(min(vals) - top * 0.05, top * 1.10)
        ax.grid(axis="y", alpha=0.25)
    axs[3].axhline(36.59, color="deepskyblue", ls="--", lw=1)
    axs[3].axhline(36.67, color="#546e7a", ls="--", lw=1)
    axs[3].text(0.02, 36.59, " nnU-Net+snap 36.59", color="deepskyblue", fontsize=7.5, va="bottom")
    axs[3].text(0.02, 36.67, " armB+snap 36.67", color="#546e7a", fontsize=7.5, va="bottom")
    axs[1].axhline(30.95, color="deepskyblue", ls="--", lw=1)
    axs[0].axhline(96.106, color="deepskyblue", ls="--", lw=1)
    fig.suptitle("step6 SQ2 sandwich-squeeze, fold-0 val (n=31): ours (red/orange) vs baselines; "
                 "bars = each model's best configuration (ours incl. interval arbitration)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGS, "step6_bars.png")
    fig.savefig(out)
    plt.close(fig)
    print("[68] wrote", out)


if __name__ == "__main__":
    overlays()
    bars()
