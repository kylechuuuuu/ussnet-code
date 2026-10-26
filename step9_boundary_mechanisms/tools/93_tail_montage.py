#!/usr/bin/env python3
"""93_tail_montage.py -- visual diagnosis of the gross anterior excursions.

For the worst cases (by HD95 / tail run length) draw:
  image | GT union contour (green) | GT TZ=255 contour (yellow)
        | base prediction contour (red) | tail points |t_gt|>10 (blue dots,
        with a blue line to where the GT boundary actually is)
so we can see WHAT the model includes anteriorly.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STEP9 = os.path.dirname(HERE)
USS = os.path.dirname(STEP9)
STEP6_TOOLS = os.path.join(USS, "step6_sandwich_squeeze", "tools")
sys.path.insert(0, STEP6_TOOLS)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = _load("arb", os.path.join(STEP6_TOOLS, "63_arbitrate.py"))
s57, s58 = arb.s57, arb.s58
NPZ = os.path.join(USS, "step6_sandwich_squeeze", "predictions",
                   "sq2resencw25_checkpoint_best_base_nifti")
MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
TG = np.arange(-90.0, 90.0001, 0.5)
NPT = 400


def gt_pos(sdf, pts, nrm):
    q = pts[:, None, :] + TG[None, :, None] * nrm[:, None, :]
    v = ndimage.map_coordinates(sdf, [q[..., 0].ravel(), q[..., 1].ravel()],
                                order=1, mode="nearest").reshape(len(pts), len(TG))
    out = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        s = np.sign(v[i])
        idx = np.flatnonzero(s[:-1] * s[1:] < 0)
        if idx.size == 0:
            continue
        j = idx[np.argmin(np.abs(TG[idx]))]
        v0, v1 = v[i, j], v[i, j + 1]
        out[i] = TG[j] + (TG[j + 1] - TG[j]) * (0.0 - v0) / (v1 - v0) if v1 != v0 else TG[j]
    return out


def panel(ax, stem):
    gt = s57.load_gt(stem)
    raw = np.array(Image.open(arb.find_file(MASK_DIR, stem, ".png")).convert("L"))
    p = arb.align_prob(arb.load_prob(NPZ, stem), gt.shape)
    img = arb.load_image(stem)
    M0 = s57.largest_cc(p > 0.5)
    c = s57.contour_of(M0)
    pts = s57.resample_closed(c, NPT, 0)
    nrm = s58.normals_of(pts, M0)
    t = gt_pos(s58.sdf_of(gt), pts, nrm)
    ok = np.isfinite(t)
    tail = ok & (np.abs(t) > 10.0)

    yy, xx = np.nonzero(gt | M0)
    r0, r1 = max(0, yy.min() - 70), min(gt.shape[0], yy.max() + 70)
    c0, c1 = max(0, xx.min() - 70), min(gt.shape[1], xx.max() + 70)
    ax.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=255)
    for m, col, lw in [(gt, "#20ff20", 1.6), (M0, "#ff3030", 1.6)]:
        cs = s57.contour_of(m)
        if cs is not None:
            ax.plot(cs[:, 1] - c0, cs[:, 0] - r0, color=col, lw=lw)
    tz = raw == 255
    ctz = s57.contour_of(tz)
    if ctz is not None:
        ax.plot(ctz[:, 1] - c0, ctz[:, 0] - r0,
                color="#ffe000", lw=1.0, alpha=0.85)
    if tail.any():
        q = pts[tail] + t[tail, None] * nrm[tail]
        ax.scatter(pts[tail, 1] - c0, pts[tail, 0] - r0, s=7, c="#00d0ff", zorder=5)
        for a, b in zip(pts[tail], q):
            ax.plot([a[1] - c0, b[1] - c0], [a[0] - r0, b[0] - r0],
                    color="#00d0ff", lw=0.7, alpha=0.8, zorder=4)
    ax.set_title("%s  dice %.2f bdice1 %.2f  tail %d" %
                 (stem, s57.dice(M0, gt), s57.bdice(M0, gt, 1), int(tail.sum())),
                 fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    ap.add_argument("--from-json", default=os.path.join(STEP9, "results",
                                                       "tail_anatomy_val.json"))
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--sort", default="hd95")
    ap.add_argument("--out", default=os.path.join(STEP9, "figures", "tail_montage.png"))
    args = ap.parse_args()

    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        import json
        d = json.load(open(args.from_json))
        cs = sorted(d["cases"], key=lambda r: -r[args.sort])[:args.top]
        names = [c["name"] for c in cs]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = len(names)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, stem in zip(axes, names):
        panel(ax, stem)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=105)
    print("[93] wrote %s  (%d cases)" % (args.out, n))


if __name__ == "__main__":
    main()
