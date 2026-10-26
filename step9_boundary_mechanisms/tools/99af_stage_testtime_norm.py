#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99af_stage_testtime_norm.py

STAGE ONLY (writes PNGs; no model, no GPU).

compound_B is, structurally, a low-frequency MULTIPLICATIVE GAIN applied to the
image (linear-in-depth TGC ramp + a smooth vertical shadow stripe) on top of a
gamma change, blur and speckle.  nnU-Net normalises with a per-image z-score,
which can only remove a GLOBAL affine map -- a smooth spatial gain survives it
and looks exactly like anatomy.  That is the mechanism behind the 5/31 cases
that lose the organ under compound_B (prediction centroid 200-320 px away).

Urology/ultrasound practice has standard remedies for exactly this, and none of
them are in the pipeline.  All are GT-FREE and test-time only, so they can be
measured with the SHIPPED checkpoint and no retraining:

  flat64 / flat128  flat-field (a.k.a. attenuation / homomorphic) correction:
                    I / blur_sigma(I), the smooth gain is divided out
  tgcrow            row-mean (depth) gain removal: divide by the smoothed
                    per-row mean profile, i.e. an explicit TGC correction
  pwl               piecewise-linear intensity standardisation
                    (Nyul-Udupa decile landmark matching) to a reference
                    built from the 71 TRAIN images only -- fixes dynamic range /
                    log-compression (gamma) differences, and is NOT affine, so
                    the z-score cannot absorb it
  histmatch         full histogram matching to the same train reference CDF

output: step8_domain_shift/data/shift_norm/<variant>__<cond>/images/<stem>.png
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

USS = "/media/72TB_disk/personal/zk/USSnet"
S8 = os.path.join(USS, "step8_domain_shift")
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
import eval_lib  # noqa: E402

SRC = os.path.join(S8, "data", "shift")
IMGDIR = os.path.join(USS, "UltrasoundProstate", "images")
OUT = os.path.join(S8, "data", "shift_norm")
CONDS = ["in_domain", "compound_B", "compound_A"]
VARIANTS = ["flat64", "flat128", "tgcrow", "pwl", "histmatch"]


def _u8(a):
    a = np.asarray(a, np.float32)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def flat(im, sigma):
    g = ndimage.gaussian_filter(im.astype(np.float32), sigma)
    return _u8(im.astype(np.float32) / (g + 1.0))


def tgcrow(im):
    x = im.astype(np.float32)
    prof = ndimage.gaussian_filter(x, (16, 1)).mean(axis=1)      # per-row level
    prof = ndimage.gaussian_filter1d(prof, 8)
    prof = np.maximum(prof, 1.0)
    return _u8(x / (prof / prof.mean())[:, None])


def _deciles(v, n=11):
    return np.percentile(v, np.linspace(0, 100, n))


def build_reference():
    """landmark / CDF reference from the 71 TRAIN images only."""
    train_names, _ = eval_lib.read_split(USS)
    dec, hist_cdf, xg = [], [], np.linspace(0, 255, 256)
    for n in train_names:
        a = np.array(Image.open(os.path.join(IMGDIR, n)).convert("L"), dtype=np.uint8)
        dec.append(_deciles(a.ravel()))
        h = np.bincount(a.ravel(), minlength=256).astype(np.float64)
        c = np.cumsum(h)
        hist_cdf.append(c / c[-1])
    return np.mean(dec, axis=0), np.mean(np.stack(hist_cdf), axis=0), xg


def pwl(im, lm_ref):
    lm = _deciles(im.ravel())
    # remove ties so np.interp stays monotone
    for i in range(1, len(lm)):
        if lm[i] <= lm[i - 1]:
            lm[i] = lm[i - 1] + 1e-3
    out = np.interp(im.ravel().astype(np.float32), lm, lm_ref)
    return _u8(out.reshape(im.shape))


def histmatch(im, ref_cdf):
    h = np.bincount(im.ravel(), minlength=256).astype(np.float64)
    c = np.cumsum(h) / h.sum()
    lut = np.interp(c, ref_cdf, np.arange(256, dtype=np.float64))
    return lut[im.ravel()].reshape(im.shape).astype(np.uint8)


def main():
    lm_ref, ref_cdf, _ = build_reference()
    print("[99af] reference built from 71 train images")
    for cond in CONDS:
        for v in VARIANTS:
            d = os.path.join(OUT, "%s__%s" % (v, cond), "images")
            os.makedirs(d, exist_ok=True)
        cdir = os.path.join(SRC, cond, "images")
        if not os.path.isdir(cdir):
            print("[99af] missing %s" % cdir)
            continue
        for name in sorted(os.listdir(cdir)):
            im = np.array(Image.open(os.path.join(cdir, name)).convert("L"), dtype=np.uint8)
            outs = {"flat64": flat(im, 64.0), "flat128": flat(im, 128.0),
                    "tgcrow": tgcrow(im), "pwl": pwl(im, lm_ref),
                    "histmatch": histmatch(im, ref_cdf)}
            for v, a in outs.items():
                Image.fromarray(a).save(
                    os.path.join(OUT, "%s__%s" % (v, cond), "images", name))
        print("[99af] staged %s: %d cases x %d variants" % (cond, len(os.listdir(cdir)), len(VARIANTS)))
    print("[99af] wrote under %s" % OUT)


if __name__ == "__main__":
    main()
