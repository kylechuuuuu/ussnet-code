#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
56_make_fp_figures.py

"Ground truth vs U-Net vs nnU-Net vs OUR fingerprint-driven model" for USSnet.

Same visual language as step 2's 14_gt_vs_predictions_figure.py (which was
re-worked for clarity: no zoom crops, no per-pixel speckle error fills), but
generalised to N models so the redesigned model can be put next to both
baselines:

    fig7_fp_vs_baselines_overlay.png
        one row per case; column 1 = ground truth, then one column per model.
        Each panel: dimmed ultrasound + the segmentation AREA filled with one
        colour + a crisp outline; the GT outline is drawn on top of every model
        panel so the displacement is visible without any zoom.
    fig8_fp_vs_baselines_difference.png
        the same cases, pure difference maps on a WHITE background (no
        ultrasound): green = correct, red = missed (FN), blue = over-segmented
        (FP), thin black GT outline.

Cases are chosen from OUR model's per-image metrics: the worst, the best, the
median, and the cases where our model loses to the U-Net baseline.

Usage:
    ~/anaconda3/envs/zk/bin/python scripts/56_make_fp_figures.py \
        --ours-tag  fpB_resenc_fploss_dir_checkpoint_best \
        --ours-dir  predictions/fpB_resenc_fploss_dir_checkpoint_best_png \
        --ours-label "Ours (ResEnc + fp loss + dir)"
    # optional extra model:  --extra tag:label:dir
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS = os.path.dirname(STEP4)
STEP2 = os.path.join(USS, "step2_compare_unet_vs_nnunet")
sys.path.insert(0, os.path.join(STEP2, "scripts"))
from eval_lib import load_binary, read_split  # noqa: E402

IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
MET_DIR = os.path.join(STEP2, "metrics")
FIG_DIR = os.path.join(STEP4, "figures")

UNET_DIR = os.path.join(USS, "unet", "results", "ultrasound_prostate_73")
NN_DIR = os.path.join(USS, "step1_nnunet_baseline", "predictions",
                      "nnunet_2d_fold0_checkpoint_best_png")

C_GT_OUT = "#00b050"
C_GT_FILL = np.array([60, 220, 90])
C_UNET_OUT = "#e000a8"
C_UNET_FILL = np.array([255, 60, 200])
C_NN_OUT = "#0080ff"
C_NN_FILL = np.array([60, 170, 255])
C_OURS_OUT = "#c2680a"
C_OURS_FILL = np.array([255, 175, 45])


def read_col(path, key):
    out = {}
    if not os.path.exists(path):
        return out
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
    ax.contour(fill_mask, levels=[.5], colors=[np.asarray(fill_rgb) / 255.0],
               linewidths=lw)
    if gt_for_outline is not None:
        ax.contour(gt_for_outline, levels=[.5], colors=C_GT_OUT, linewidths=lw)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def panel_diff(ax, gt, pred, title):
    """Clean difference map on white: TP green, FN red, FP blue."""
    img = np.ones(gt.shape + (3,), np.float32) * 0.95
    img[np.logical_and(gt, ~pred)] = np.array([235, 60, 60]) / 255.0
    img[np.logical_and(pred, ~gt)] = np.array([60, 120, 245]) / 255.0
    img[np.logical_and(gt, pred)] = np.array([40, 190, 90]) / 255.0
    ax.imshow(img)
    ax.contour(gt, levels=[.5], colors="#1a1a1a", linewidths=1.1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def pick_cases(ours_dice, unet_dice, n=8):
    """worst / best / median for ours + the cases where ours loses to U-Net."""
    names = [k for k in ours_dice if k in unet_dice]
    names.sort(key=lambda k: ours_dice[k])
    worst = names[:3]
    best = names[-2:][::-1]
    mid = [names[len(names) // 2]]
    loses = sorted(names, key=lambda k: ours_dice[k] - unet_dice[k])[:2]
    out = []
    for k in worst + loses + mid + best:
        if k not in out:
            out.append(k)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-tag", required=True,
                    help="e.g. fpB_resenc_fploss_dir_checkpoint_best")
    ap.add_argument("--ours-dir", required=True,
                    help="dir of our PNG masks (absolute or relative to step4)")
    ap.add_argument("--ours-label", default="Ours (fingerprint-driven)")
    ap.add_argument("--extra", action="append", default=[],
                    help="tag:label:dir (repeatable) - extra model column")
    ap.add_argument("--out-prefix", default="fp")
    args = ap.parse_args()

    ours_dir = args.ours_dir
    if not os.path.isabs(ours_dir):
        ours_dir = os.path.join(STEP4, ours_dir)

    _, val = read_split(USS)
    val = [n for n in val if n.lower().endswith(".png")]

    models = [
        ("unet_73", "U-Net (baseline)", UNET_DIR, C_UNET_FILL, C_UNET_OUT),
        ("nnunet_2d_fold0_checkpoint_best", "nnU-Net 2d (best)", NN_DIR,
         C_NN_FILL, C_NN_OUT),
        (args.ours_tag, args.ours_label, ours_dir, C_OURS_FILL, C_OURS_OUT),
    ]
    for spec in args.extra:
        tag, label, d = spec.split(":", 2)
        if not os.path.isabs(d):
            d = os.path.join(STEP4, d)
        models.append((tag, label, d, np.array([150, 90, 220]), "#6b2fa0"))

    # ---- per-image metrics + case selection --------------------------------
    dice = {tag: read_col(os.path.join(MET_DIR, "%s_per_image.csv" % tag), "dice")
            for tag, _, _, _, _ in models}
    step3_res = os.path.join(USS, "step3_failure_analysis", "results")
    bdice = {tag: read_col(os.path.join(step3_res, "%s_error_decomposition.csv" % tag),
                           "bdice_1px")
             for tag, _, _, _, _ in models}
    ours_tag = args.ours_tag
    if not dice[ours_tag]:
        print("[56] ERROR: no per-image metrics for %s in %s" % (ours_tag, MET_DIR))
        return 2

    cases = pick_cases(dice[ours_tag], dice.get("unet_73", {}))
    print("[56] cases (%d):" % len(cases))
    for c in cases:
        print("      %-34s ours %6.2f  unet %6.2f  nn %6.2f"
              % (c, dice[ours_tag].get(c, float("nan")),
                 dice.get("unet_73", {}).get(c, float("nan")),
                 dice.get("nnunet_2d_fold0_checkpoint_best", {}).get(c, float("nan"))))

    ncol = 1 + len(models)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- figure 7: overlay ------------------------------------------------
    fig, axes = plt.subplots(len(cases), ncol,
                             figsize=(3.35 * ncol, 3.0 * len(cases)),
                             squeeze=False)
    for r, name in enumerate(cases):
        gray = np.asarray(plt.imread(os.path.join(IMG_DIR, name)))
        if gray.ndim == 3:
            gray = gray[..., :3].mean(-1)
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.01 else gray.astype(np.uint8)
        gt = load_binary(os.path.join(GT_DIR, name))

        panel_filled(axes[r][0], gray, gt, C_GT_FILL,
                     "%s\nGROUND TRUTH" % name.replace(".png", ""))
        for c, (tag, label, d, fill, out) in enumerate(models, start=1):
            ppath = os.path.join(d, name)
            if not os.path.exists(ppath):
                axes[r][c].axis("off")
                continue
            p = load_binary(ppath)
            dsc = dice[tag].get(name)
            bd = bdice[tag].get(name)
            t = "%s\nDice %s" % (label, "%.2f" % dsc if dsc is not None else "n/a")
            if bd is not None:
                t += " | bdice@1px %.1f%%" % bd
            panel_filled(axes[r][c], gray, p, fill, t, gt_for_outline=gt)
    handles = [Patch(facecolor=C_GT_FILL / 255.0, edgecolor=C_GT_OUT, label="ground truth")]
    for tag, label, d, fill, out in models:
        handles.append(Patch(facecolor=fill / 255.0, edgecolor=out, label=label))
    handles.append(Patch(facecolor="white", edgecolor=C_GT_OUT,
                         label="green line = ground-truth outline (on every panel)"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5),
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("USSnet - ground truth vs both baselines vs the fingerprint-driven model "
                 "(validation fold 0)", y=1.035, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out7 = os.path.join(FIG_DIR, "fig7_%s_vs_baselines_overlay.png" % args.out_prefix)
    fig.savefig(out7, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[56] ->", out7)

    # ---- figure 8: difference maps ---------------------------------------
    fig, axes = plt.subplots(len(cases), len(models),
                             figsize=(3.35 * len(models), 3.0 * len(cases)),
                             squeeze=False)
    for r, name in enumerate(cases):
        gt = load_binary(os.path.join(GT_DIR, name))
        for c, (tag, label, d, fill, out) in enumerate(models):
            ppath = os.path.join(d, name)
            if not os.path.exists(ppath):
                axes[r][c].axis("off")
                continue
            p = load_binary(ppath)
            dsc = dice[tag].get(name)
            panel_diff(axes[r][c], gt, p,
                       "%s\nDice %s" % (label, "%.2f" % dsc if dsc is not None else "n/a"))
    handles = [Patch(facecolor=np.array([40, 190, 90]) / 255.0, label="correct (TP)"),
               Patch(facecolor=np.array([235, 60, 60]) / 255.0, label="missed (FN)"),
               Patch(facecolor=np.array([60, 120, 245]) / 255.0, label="over-segmented (FP)"),
               Patch(facecolor="white", edgecolor="#1a1a1a",
                     label="black line = ground-truth outline")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("USSnet - difference against the ground truth, no ultrasound background "
                 "(validation fold 0)", y=1.035, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out8 = os.path.join(FIG_DIR, "fig8_%s_vs_baselines_difference.png" % args.out_prefix)
    fig.savefig(out8, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[56] ->", out8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
