#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""99g: figure for the GT-free boundary risk map.

Left of each pair : contour coloured by the GT-FREE score (entropy within +-2px)
Right of each pair: the same contour coloured by the TRUE |t_gt|
Only the base probability field and the image are used to make the left panel.
"""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image
from scipy import ndimage as ndi

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
NPZ = os.path.join(S6, "predictions", "sq2resencw25_checkpoint_best_%s_nifti")
TS = np.arange(-40.0, 40.0001, 0.5)
NPT = 400
CASES = ["黄信华_005_160912", "娄启安_001_102325", "周平伟_001_161543"]


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = load_mod("arb", os.path.join(S6, "tools", "63_arbitrate.py"))
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools",
                                   "57_shape_prior_ceiling.py"))
s58 = load_mod("s58", os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools",
                                   "58_normal_refine_oracle.py"))


def sdf_of(gt):
    st = ndi.generate_binary_structure(2, 1)
    s = gt & ~ndi.binary_erosion(gt, st, border_value=0)
    d = ndi.distance_transform_edt(~s)
    return np.where(gt, -d, d).astype(np.float32)


def probes(arr, pts, nrm, ts):
    return ndi.map_coordinates(arr, np.stack([
        pts[:, 0][None, :] + nrm[:, 0][None, :] * ts[:, None],
        pts[:, 1][None, :] + nrm[:, 1][None, :] * ts[:, None]]), order=1, mode="nearest")


def t_gt_of(sdf, pts, nrm, ts):
    v = probes(sdf, pts, nrm, ts)
    out = np.full(v.shape[1], np.nan)
    for j in range(v.shape[1]):
        col = np.sign(v[:, j])
        zi = np.nonzero(np.diff(col) != 0)[0]
        if len(zi) == 0:
            continue
        t0, t1 = ts[zi], ts[zi + 1]
        v0, v1 = v[zi, j], v[zi + 1, j]
        tt = t0 - v0 * (t1 - t0) / (v1 - v0)
        out[j] = tt[np.argmin(np.abs(tt))]
    return out


fig, axes = plt.subplots(2, 3, figsize=(19, 11))
for col, name in enumerate(CASES):
    stem = arb.stem_of(name)
    gt = s57.load_gt(name)
    img = np.asarray(Image.open(os.path.join(USS, "UltrasoundProstate", "images",
                                             stem + ".png")).convert("L"), dtype=np.float32)
    pb = arb.align_prob(arb.load_prob(NPZ % "base", stem), gt.shape)
    M0 = s57.largest_cc(pb > 0.5)
    pts = s57.resample_closed(s57.contour_of(M0), NPT, 0)
    nrm = s58.normals_of(pts, M0)
    p = np.clip(pb, 1e-6, 1 - 1e-6)
    ent = (-(p * np.log(p) + (1 - p) * np.log(1 - p))).astype(np.float32)
    ent_loc = probes(ent, pts, nrm, TS[np.abs(TS) <= 2.0]).mean(axis=0)
    tgt = np.abs(t_gt_of(sdf_of(gt), pts, nrm, TS))

    for row, (vals, ttl, cmap, vmax) in enumerate([
            (ent_loc, "GT-FREE risk:  mean entropy within $\\pm$2px", "magma", np.nanpercentile(ent_loc, 98)),
            (tgt, "TRUE $|t_{gt}|$  (px)", "magma", 25.0)]):
        ax = axes[row, col]
        ax.imshow(img, cmap="gray")
        g = s57.surface(gt)
        ax.imshow(np.ma.masked_where(~g, g), cmap="winter", alpha=0.45)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, linewidth=2.6)
        lc.set_array(np.nan_to_num(vals[:-1]))
        lc.set_clim(0, vmax)
        ax.add_collection(lc)
        ax.set_xlim(80, 950); ax.set_ylim(700, 60)
        ax.set_title("%s\n%s" % (name, ttl), fontsize=9)
        ax.axis("off")
        plt.colorbar(lc, ax=ax, fraction=0.035, pad=0.01)
out = os.path.join(S9, "figures", "localization.png")
plt.tight_layout()
plt.savefig(out, dpi=95)
print("wrote", out)
