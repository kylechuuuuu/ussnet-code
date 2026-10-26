#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99o_final_figure.py

Visual check of the final artifact: same image, three contours
  green  = ground truth
  red    = step-6 strongest single model (SQ2res-w25, base head)
  cyan   = ens_K (weighted ensemble + median3/open2/fill_holes)

Cases are chosen from the official per-image Dice: the cases where the final
pipeline gained most over the single model, plus a median and a worst case, so
the figure cannot cherry-pick silently.

usage: python tools/99o_final_figure.py
"""
import csv
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
PRED = os.path.join(S6, "predictions")
MET = os.path.join(USS, "step2_compare_unet_vs_nnunet", "metrics")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
RES_PNG = os.path.join(PRED, "sq2resencw25_checkpoint_best_base_png")
K_PNG = os.path.join(PRED, "ens_K_res6_med3open2fill_png")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = load_mod("arb", os.path.join(S6, "tools", "63_arbitrate.py"))
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary",
                                   "tools", "57_shape_prior_ceiling.py"))


def read_csv(tag):
    with open(os.path.join(MET, tag + "_per_image.csv")) as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def png_mask(path, shape):
    m = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    if m.shape != shape:
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
            (shape[1], shape[0]), Image.NEAREST), dtype=np.uint8) > 127
    return m


def main():
    res = read_csv("sq2resencw25_checkpoint_best_base")
    k = read_csv("ens_K_res6_med3open2fill")
    names = [n for n in k if n in res]
    names.sort(key=lambda n: float(k[n]["dice"]) - float(res[n]["dice"]),
               reverse=True)
    picks = [names[0], names[1], names[2], names[len(names) // 2], names[-1]]
    titles = ["gain #1", "gain #2", "gain #3", "median gain", "worst gain"]

    fig, axes = plt.subplots(1, len(picks), figsize=(6.0 * len(picks), 6.4))
    for ax, name, ttl in zip(axes, picks, titles):
        stem = arb.stem_of(name)
        gt = np.asarray(s57.load_gt(name)).astype(bool)
        img = np.asarray(Image.open(os.path.join(
            USS, "UltrasoundProstate", "images", stem + ".png")).convert("L"))
        mres = png_mask(os.path.join(RES_PNG, name if name.endswith(".png") else name + ".png"), gt.shape)
        mk = png_mask(os.path.join(K_PNG, name if name.endswith(".png") else name + ".png"), gt.shape)

        ax.imshow(img, cmap="gray")
        for m, col, lw in [(gt, "#00ff44", 1.6), (mres, "#ff2d2d", 1.6),
                           (mk, "#00d5ff", 1.6)]:
            ax.contour(m.astype(float), levels=[0.5], colors=[col],
                       linewidths=lw, linestyles="solid")
        ys, xs = np.nonzero(gt)
        pad = 70
        ax.set_xlim(max(0, xs.min() - pad), xs.max() + pad)
        ax.set_ylim(min(gt.shape[0], ys.max() + pad), max(0, ys.min() - pad))
        d_res = float(res[name]["dice"])
        d_k = float(k[name]["dice"])
        ax.set_title("%s\n%s\ndice res %.2f -> ens_K %.2f (%+.2f)"
                     % (name, ttl, d_res, d_k, d_k - d_res), fontsize=10)
        ax.axis("off")

    fig.suptitle("green = GT · red = step6 strongest single (SQ2res-w25) · "
                 "cyan = ens_K (ensemble + median3+open2+fill)", fontsize=13)
    out = os.path.join(S9, "figures", "final_ens_K.png")
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print("wrote", out)
    print("picks:", [(n, round(float(k[n]["dice"]) - float(res[n]["dice"]), 3)) for n in picks])


if __name__ == "__main__":
    main()
