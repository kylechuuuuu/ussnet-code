#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/tools/50_extract_boundary_fingerprint.py

WHY THIS FILE EXISTS
--------------------
nnU-Net's dataset fingerprint (dataset_fingerprint.json) measures *area* and
*intensity*: foreground intensity percentiles, median shape after cropping,
median relative size.  The planner turns those into patch size, batch size,
normalization and the training recipe.  It measures **nothing about boundary
geometry** -- yet the measured bottleneck of both the U-Net baseline and the
stock nnU-Net on this dataset is exactly boundary geometry:

    boundary Dice@1px   : U-Net 23.25 %      nnU-Net 35.16 %
    predicted perimeter : U-Net 1.81x GT     nnU-Net 1.01x GT
    FN rate by sector   : 13.4 % posterior .. 28.9 % anterior  -> 2.2x

So this tool extracts the *missing* half of the fingerprint.  It is the
"fingerprint upgrade" of the redesign: instead of a human guessing the band
width, the boundary weight and the sector weights, the dataset declares them.

Everything is computed on the **fold-0 TRAINING split only** (71 images), so
there is no leakage into the 31 validation images used for the comparison.

It also acts as the *planner* for the boundary loss: it converts the measured
descriptors into concrete loss hyper-parameters (`derived_loss_config`) that
`trainer/fp_config.py` loads at training time.

USAGE
    ~/anaconda3/envs/zk/bin/python tools/50_extract_boundary_fingerprint.py

OUTPUT
    data/boundary_fingerprint.json      <- descriptors + derived loss config
    logs/50_boundary_fingerprint.log    <- human readable summary
"""
import os
import sys
import csv
import json
import math
from datetime import datetime

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import perimeter as sk_perimeter

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS = os.path.dirname(STEP4)

IMAGES = os.path.join(USS, "UltrasoundProstate", "images")
MASKS = os.path.join(USS, "UltrasoundProstate", "masks")
STEP1 = os.path.join(USS, "step1_nnunet_baseline")
SPLITS = os.path.join(STEP1, "data", "nnUNet_preprocessed",
                      "Dataset001_UltrasoundProstate", "splits_final.json")
MAPPING = os.path.join(STEP1, "case_id_mapping.csv")

OUT_JSON = os.path.join(STEP4, "data", "boundary_fingerprint.json")
OUT_LOG = os.path.join(STEP4, "logs", "50_boundary_fingerprint.log")

N_SECTORS = 12
OPENING_RADII = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30]


# --------------------------------------------------------------------------- io
def load_mapping():
    """case_id (USPxxx) -> original PNG file name."""
    m = {}
    with open(MAPPING, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m[row["case_id"]] = row["original_filename"]
    return m


def train_split(fold=0):
    with open(SPLITS, encoding="utf-8") as fh:
        splits = json.load(fh)
    return list(splits[fold]["train"]), list(splits[fold]["val"])


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def load_mask(path):
    a = np.asarray(Image.open(path))
    return (a > 0)              # 127 and 255 abut -> union is the whole prostate


# ------------------------------------------------------------------- geometry
def grad_magnitude(img):
    gy, gx = np.gradient(img.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def boundary_ring(mask, width=1):
    """Ring of `width` px centred on the mask boundary."""
    st = ndimage.generate_binary_structure(2, 2)
    dil = ndimage.binary_dilation(mask, st, iterations=width)
    ero = ndimage.binary_erosion(mask, st, iterations=width)
    return np.logical_xor(dil, ero)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def dice(a, b):
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / s


def sector_index(shape, mask, n_sectors=N_SECTORS):
    """12 equal 30-degree sectors around the mask centroid.

    Angle convention: 90 deg = image UP = anterior (these are transrectal
    ultrasound images, the top of the image is the anterior side).
    """
    h, w = shape
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.degrees(np.arctan2(-(yy - cy), xx - cx))        # -180..180, up = +90
    idx = np.floor((ang + 180.0) / (360.0 / n_sectors)).astype(np.int32)
    np.clip(idx, 0, n_sectors - 1, out=idx)
    return idx


# ------------------------------------------------------------------ extraction
def extract(names, mapping):
    per_image = []
    for i, cid in enumerate(names):
        fn = mapping[cid]
        img = load_gray(os.path.join(IMAGES, fn))
        gt = load_mask(os.path.join(MASKS, fn))
        g = grad_magnitude(img)
        rings = boundary_ring(gt, 1)
        gimg = float(g.mean())
        gbnd = float(g[rings].mean()) if rings.any() else 0.0
        p75, p90 = np.percentile(g, [75, 90])
        es75 = float((g[rings] >= p75).mean()) if rings.any() else 0.0
        es90 = float((g[rings] >= p90).mean()) if rings.any() else 0.0
        area = float(gt.sum())
        per = float(sk_perimeter(gt))
        # opening invariance: largest r whose opening still reproduces the mask
        r_star = 0
        for r in OPENING_RADII:
            if dice(gt, ndimage.binary_opening(gt, disk(r))) >= 0.995:
                r_star = r
            else:
                break
        ncc = int(ndimage.label(gt)[1])
        sec = sector_index(gt.shape, gt)
        sec_g, sec_len = np.zeros(N_SECTORS), np.zeros(N_SECTORS)
        for s in range(N_SECTORS):
            m = np.logical_and(sec == s, rings)
            sec_len[s] = float(m.sum())
            if m.any():
                sec_g[s] = float(g[m].mean())
        per_image.append(dict(case=cid, file=fn, h=int(gt.shape[0]), w=int(gt.shape[1]),
                              area=area, perimeter=per, r_star=r_star, n_cc=ncc,
                              grad_mean=gimg, grad_boundary=gbnd,
                              grad_ratio=(gbnd / gimg if gimg > 0 else 0.0),
                              edge_support_p75=es75, edge_support_p90=es90,
                              fg_frac=float(area / gt.size),
                              dir_grad=sec_g.tolist(), dir_len=sec_len.tolist()))
        if (i + 1) % 10 == 0 or i == len(names) - 1:
            print("  %3d/%d  %s" % (i + 1, len(names), cid), flush=True)
    return per_image


def summarise(per_image):
    arr = lambda k: np.array([p[k] for p in per_image], dtype=np.float64)
    area = arr("area")
    per = arr("perimeter")
    ratio = arr("grad_ratio")
    es75 = arr("edge_support_p75")
    es90 = arr("edge_support_p90")
    r_star = arr("r_star")

    boundary_contrast = dict(
        grad_ratio_mean=float(ratio.mean()), grad_ratio_p10=float(np.percentile(ratio, 10)),
        grad_ratio_p90=float(np.percentile(ratio, 90)),
        edge_support_p75_mean=float(es75.mean()), edge_support_p90_mean=float(es90.mean()),
        boundary_is_strong_edge=bool(ratio.mean() >= 2.0 and es75.mean() >= 0.5),
        boundary_fully_visible=bool(es75.mean() >= 0.75 and es90.mean() >= 0.45),
    )

    shape_statistics = dict(
        perimeter_over_sqrt_area_mean=float((per / np.sqrt(area)).mean()),
        perimeter_over_sqrt_area_std=float((per / np.sqrt(area)).std()),
        opening_invariance_radius_median=float(np.median(r_star)),
        is_smooth=bool(np.median(r_star) >= 8),
        n_cc_mean=float(arr("n_cc").mean()), n_cc_max=int(arr("n_cc").max()),
        mean_area_px=float(area.mean()), mean_linear_scale_px=float(math.sqrt(area.mean())),
    )

    dg = np.array([p["dir_grad"] for p in per_image])          # (N, 12)
    dl = np.array([p["dir_len"] for p in per_image])
    tot_len = dl.sum(axis=1, keepdims=True)
    tot_len[tot_len == 0] = 1.0
    valid = dg > 0
    dg_mean = np.where(valid.any(axis=0),
                       np.divide((dg * valid).sum(axis=0),
                                 np.maximum(valid.sum(axis=0), 1)), 0.0)
    pos = dg_mean[dg_mean > 0]
    rel = (dg_mean / pos.mean()).tolist() if pos.size else [1.0] * N_SECTORS
    angular_profile = dict(
        sector_deg_centers=[-165 + 30 * s for s in range(N_SECTORS)],
        sector_deg_convention="90 = image up = anterior; -90 = image down = posterior (rectal wall)",
        edge_strength=dg_mean.tolist(),
        edge_strength_rel=rel,
        boundary_length_frac=(dl / tot_len).mean(axis=0).tolist(),
    )

    scale_spread = dict(
        area_cv=float(area.std() / area.mean()),
        h_median=float(np.median(arr("h"))), w_median=float(np.median(arr("w"))),
        fg_frac_mean=float(arr("fg_frac").mean()), fg_frac_std=float(arr("fg_frac").std()),
        fg_frac_min=float(arr("fg_frac").min()), fg_frac_max=float(arr("fg_frac").max()),
    )
    return boundary_contrast, shape_statistics, angular_profile, scale_spread


def derive_loss_config(bc, ss, ap, sc):
    """The planner: fingerprint descriptors -> concrete loss hyper-parameters.

    Every number below is traceable to a measured descriptor, which is the whole
    point -- the previous design hard-coded band=3, beta=4, lambda=0.25 by hand.
    """
    lin = ss["mean_linear_scale_px"]

    # Tolerance: the residual contour error of both models is a coherent
    # displacement of ~1-3 px; 1 % of the gland's linear scale gives 3 px here,
    # so the tolerant term is a +-3 px band.
    tol_px = int(np.clip(round(0.01 * lin), 2, 6))
    tolerant_band = 2 * tol_px + 1                       # odd kernel for max-pool

    # Sharp pressure is only meaningful where the boundary really is an edge.
    bce_band = 3 if bc["grad_ratio_mean"] >= 3.0 else 5
    beta = float(np.clip(round(bc["grad_ratio_mean"]), 2.0, 6.0))

    # Perimeter / length term: justified because the ground truth is smooth.
    smooth = 1.0 if ss["is_smooth"] else 0.4

    # Direction weights: sectors where the boundary is WEAKER get a higher
    # weight (they are where both models fail -- anterior FN is 2.2x posterior).
    rel = np.array(ap["edge_strength_rel"], dtype=np.float64)
    rel = np.where(rel > 0, rel, 1.0)
    gamma = 1.0
    w = np.clip(rel ** (-gamma), 0.5, 3.0)
    w = w / w.mean()

    return dict(
        tolerant_band=int(tolerant_band),
        tolerance_px=int(tol_px),
        bce_band=int(bce_band),
        bce_beta=float(beta),
        weight_tolerant_bdice=0.5,
        weight_boundary_bce=round(0.25 * float(bc["edge_support_p75_mean"]), 4),
        weight_perimeter=round(0.05 * float(smooth), 4),
        weight_direction=1.0,
        dir_sector_weights=[round(float(v), 4) for v in w],
        dir_gamma=float(gamma),
        warmup_epochs=100,
    )


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 78)
    say("[50] boundary/geometry fingerprint for Dataset001_UltrasoundProstate")
    say("     computed on the fold-0 TRAINING split only (no leakage)")
    say("=" * 78)

    mapping = load_mapping()
    tr, va = train_split(0)
    say("[50] fold 0 : train %d / val %d  (val NOT used for the fingerprint)"
        % (len(tr), len(va)))

    per_image = extract(tr, mapping)
    bc, ss, ap, sc = summarise(per_image)
    cfg = derive_loss_config(bc, ss, ap, sc)

    say("")
    say("--- boundary_contrast ---")
    say("  |gradI| at GT boundary / whole image : %.2fx  (p10 %.2f, p90 %.2f)"
        % (bc["grad_ratio_mean"], bc["grad_ratio_p10"], bc["grad_ratio_p90"]))
    say("  boundary pixels with |gradI| >= image p75 : %.1f %%"
        % (100 * bc["edge_support_p75_mean"]))
    say("  boundary pixels with |gradI| >= image p90 : %.1f %%"
        % (100 * bc["edge_support_p90_mean"]))
    say("  -> strong edge: %s | fully visible: %s"
        % (bc["boundary_is_strong_edge"], bc["boundary_fully_visible"]))
    say("--- shape_statistics ---")
    say("  perimeter / sqrt(area) : %.3f +- %.3f"
        % (ss["perimeter_over_sqrt_area_mean"], ss["perimeter_over_sqrt_area_std"]))
    say("  opening-invariance radius (median) : %.0f px  -> smooth: %s"
        % (ss["opening_invariance_radius_median"], ss["is_smooth"]))
    say("  mean area %.0f px  -> mean linear scale %.1f px"
        % (ss["mean_area_px"], ss["mean_linear_scale_px"]))
    say("  connected components mean %.2f / max %d"
        % (ss["n_cc_mean"], ss["n_cc_max"]))
    say("--- angular_profile ---")
    say("  sector(deg)   edge strength rel   boundary length frac")
    for s in range(N_SECTORS):
        say("   %6d          %6.3f              %6.4f"
            % (ap["sector_deg_centers"][s], ap["edge_strength_rel"][s],
               ap["boundary_length_frac"][s]))
    say("--- scale_spread ---")
    say("  area CV %.3f | median size %.0f x %.0f | fg frac %.4f +- %.4f (%.4f .. %.4f)"
        % (sc["area_cv"], sc["h_median"], sc["w_median"], sc["fg_frac_mean"],
           sc["fg_frac_std"], sc["fg_frac_min"], sc["fg_frac_max"]))
    say("")
    say("--- derived_loss_config (the planner output the trainer reads) ---")
    for k, v in cfg.items():
        say("  %-28s %s" % (k, v))

    payload = dict(
        meta=dict(dataset="Dataset001_UltrasoundProstate", fold=0,
                  n_train=len(tr), n_val=len(va),
                  created=datetime.now().isoformat(timespec="seconds"),
                  source="original PNG masks (union of 127|255), fold-0 train split only",
                  angle_convention="90 deg = image up = anterior"),
        boundary_contrast=bc,
        shape_statistics=ss,
        angular_profile=ap,
        scale_spread=sc,
        derived_loss_config=cfg,
        per_image=per_image,
    )
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with open(OUT_LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote", OUT_JSON)
    print("wrote", OUT_LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
