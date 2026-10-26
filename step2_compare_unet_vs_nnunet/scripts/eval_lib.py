#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_compare_unet_vs_nnunet/scripts/eval_lib.py

One single metric implementation, used for BOTH the U-Net baseline and the
nnU-Net baseline, so the comparison cannot be biased by two different metric
codes.

Ground truth convention
-----------------------
The raw annotation PNGs hold two abutting labels (127 = outer zone,
255 = inner zone).  Both the U-Net baseline and the nnU-Net baseline are
trained on the UNION of the two, so all headline metrics below are computed
against the union (`gt = mask > 0`).  Metrics against the individual source
labels are reported as extra columns for the failure analysis.

Metrics per image (binary foreground):
    dice / f1       2|P^G| / (|P|+|G|)                       x100
    precision       |P^G| / |P|                              x100
    recall          |P^G| / |G|                              x100
    iou             |P^G| / |P u G|                          x100
    hd95            symmetric 95th percentile surface distance (pixels)
    assd            average symmetric surface distance (pixels)
    vol_diff_pct    (|P|-|G|)/|G|                            x100
    correctness     skeleton overlap / |skel(P)|              x100   (as in unet/metric/metric.py)
    completeness    skeleton overlap / |skel(G)|              x100   (as in unet/metric/metric.py)
    quality         skeleton overlap / |skel(P) u skel(G)|    x100   (as in unet/metric/metric.py)
"""
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology

EPS = 1e-7


# --------------------------------------------------------------------------- io
def load_binary(path, threshold=0):
    """Read a PNG and return a bool mask: foreground = pixel > threshold.

    Default threshold=0 treats the raw two-label annotation (0/127/255) as the
    UNION of the two source labels, which is the target both baselines are
    trained on; prediction PNGs only contain 0/255 for which it is exact.
    """
    a = np.array(Image.open(path).convert("L"))
    return a > threshold


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


# ---------------------------------------------------------------------- metrics
def _surface(mask):
    """Boundary pixels of a binary mask."""
    if not mask.any():
        return np.zeros_like(mask)
    struct = ndimage.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=struct, border_value=0)


def _directed_distances(a_surface, b_surface):
    """Distance (in px) from every pixel of a_surface to the nearest b_surface pixel."""
    if not a_surface.any() or not b_surface.any():
        return np.array([], dtype=np.float64)
    dt = ndimage.distance_transform_edt(~b_surface)
    return dt[a_surface].astype(np.float64)


def surface_metrics(pred, gt):
    """hd95 / assd / hd100 in pixels; NaN when one side is empty."""
    ps, gs = _surface(pred), _surface(gt)
    d_pg = _directed_distances(ps, gs)
    d_gp = _directed_distances(gs, ps)
    if d_pg.size == 0 or d_gp.size == 0:
        return float("nan"), float("nan"), float("nan")
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    assd = 0.5 * (d_pg.mean() + d_gp.mean())
    hd100 = max(d_pg.max(), d_gp.max())
    return float(hd95), float(assd), float(hd100)


def _dice(a, b):
    inter = float(np.logical_and(a, b).sum())
    return 2.0 * inter / (float(a.sum()) + float(b.sum()) + EPS) * 100.0


def all_metrics(pred, gt, gt_gray=None):
    """Predicate/ground-truth bool arrays -> dict of metrics."""
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn

    dice = 2 * tp / (2 * tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    iou = tp / (union + EPS)
    vol_diff = (tp + fp - (tp + fn)) / (tp + fn + EPS) * 100.0

    sk_p = morphology.skeletonize(pred)
    sk_g = morphology.skeletonize(gt)
    inter = float(np.logical_and(sk_p, sk_g).sum())
    sp, sg = float(sk_p.sum()), float(sk_g.sum())
    correctness = inter / (sp + EPS) * 100.0
    completeness = inter / (sg + EPS) * 100.0
    quality = inter / (sp + sg - inter + EPS) * 100.0

    hd95, assd, hd100 = surface_metrics(pred, gt)

    out = dict(
        dice=dice * 100.0, precision=precision * 100.0, recall=recall * 100.0,
        iou=iou * 100.0, vol_diff_pct=vol_diff,
        hd95=hd95, assd=assd, hd100=hd100,
        correctness=correctness, completeness=completeness, quality=quality,
        pred_pixels=int(pred.sum()), gt_pixels=int(gt.sum()),
        n_pred_cc=int(ndimage.label(pred)[1]), n_gt_cc=int(ndimage.label(gt)[1]),
        skel_pred_len=int(sp), skel_gt_len=int(sg),
        perimeter_pred=int(_surface(pred).sum()), perimeter_gt=int(_surface(gt).sum()),
    )

    if gt_gray is not None:
        for lab, tag in ((127, "d127"), (255, "d255")):
            reg = gt_gray == lab
            inter_l = float(np.logical_and(pred, reg).sum())
            out[tag + "_dice"] = _dice(pred, reg)
            out[tag + "_pixels"] = int(reg.sum())
            out[tag + "_recall"] = inter_l / (float(reg.sum()) + EPS) * 100.0
            out[tag + "_precision"] = inter_l / (float(pred.sum()) + EPS) * 100.0
    return out


def region_metrics(pred, region):
    """Legacy helper: coverage of `region` by `pred`."""
    inter = float(np.logical_and(pred, region).sum())
    r = float(region.sum())
    return dict(region_pixels=int(r),
                recall_in_region=inter / (r + EPS) * 100.0,
                precision_in_region=inter / (float(pred.sum()) + EPS) * 100.0)


# ------------------------------------------------------------------ split helpers
def read_split(uss_root, exp="ultrasound_prostate_73"):
    """Return (train_names, val_names) of the U-Net baseline split (original PNG names)."""
    base = os.path.join(uss_root, "unet", "datasets", exp)

    def names(sub):
        d = os.path.join(base, sub, "images")
        return sorted(n for n in os.listdir(d) if n.lower().endswith(".png"))

    return names("train"), names("val")
