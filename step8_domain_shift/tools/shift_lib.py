#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_domain_shift/tools/shift_lib.py

Physically motivated ACQUISITION-SHIFT operators for transrectal ultrasound
(TRUS) prostate imaging, plus the appearance-standardisation transforms used to
test whether a shift is recoverable at inference time.

Why these and not "augmentations"
---------------------------------
The pipeline normalises every image with nnU-Net's per-image ZScoreNormalization.
Any shift that is a GLOBAL AFFINE intensity map (gain, offset) is therefore
mathematically invisible to the network -- including it would only manufacture
the illusion of robustness.  The operators below are grouped by whether that
invariance can absorb them:

    affine (invariant)      gain, offset          -> control, must show NO drop
    pointwise nonlinear     gamma / log-compress  -> NOT absorbed by z-score
    spatially varying       tgc ramp, shadow      -> NOT absorbed
    stochastic              speckle (multiplicative Rayleigh-like)
    resolution              gaussian blur
    acquisition artifact    jpeg-30
    compound                "other machine": gamma + tgc + speckle + blur

Every operator leaves the ANATOMY untouched: the ground-truth mask is unchanged,
so this measures the robustness of the segmentation function under an
appearance shift, not a geometric one.  This is a model of INTRA-vendor
re-parameterisation (time-gain compensation, dynamic range / log compression,
speckle statistics, focal zone), not of true cross-site anatomy variation --
that limitation is stated in the report.

All functions take and return uint8 grayscale (H, W) and an explicit numpy
Generator so every condition is reproducible bit-for-bit.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


# --------------------------------------------------------------- utilities --
def _u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def _speckle_lognormal(img, rng, s):
    """Multiplicative speckle with unit mean: exp(s*N - s^2/2)."""
    n = rng.standard_normal(img.shape) * s - 0.5 * s * s
    return _u8(img.astype(np.float32) * np.exp(n))


def _tgc(img, alpha):
    """Depth-dependent gain (time-gain compensation mis-set)."""
    h = img.shape[0]
    ramp = np.linspace(1.0 - alpha / 2.0, 1.0 + alpha / 2.0, h, dtype=np.float32)
    return _u8(img.astype(np.float32) * ramp[:, None])


def _shadow(img, depth=0.45, width_frac=0.22, pos=0.42):
    """Acoustic shadow: a smooth vertical dark band (attenuating structure)."""
    w = img.shape[1]
    x = np.arange(w, dtype=np.float32)
    c = pos * w
    sig = max(width_frac * w / 2.355, 1.0)
    band = 1.0 - depth * np.exp(-0.5 * ((x - c) / sig) ** 2)
    return _u8(img.astype(np.float32) * band[None, :])


# ------------------------------------------------------------ the registry --
def _mk(name, fn):
    return name, fn


SHIFTS = {
    # --- reference row: the unshifted image (identity operator) -------
    "in_domain": lambda im, r: im,
    # --- global affine: invariant under ZScoreNormalization (control) --------
    "affine_gain_p25": lambda im, r: _u8(np.clip(im.astype(np.float32) * 1.25 + 8, 0, 255)),
    "affine_gain_m25": lambda im, r: _u8(np.clip(im.astype(np.float32) * 0.75 - 8, 0, 255)),
    # --- pointwise nonlinear (dynamic range / log compression) --------------
    "gamma_080": lambda im, r: _u8(255.0 * (im.astype(np.float32) / 255.0) ** 0.80),
    "gamma_125": lambda im, r: _u8(255.0 * (im.astype(np.float32) / 255.0) ** 1.25),
    "gamma_060": lambda im, r: _u8(255.0 * (im.astype(np.float32) / 255.0) ** 0.60),
    # --- spatially varying gain --------------------------------------------
    "tgc_p60": lambda im, r: _tgc(im, 0.60),
    "tgc_m60": lambda im, r: _tgc(im, -0.60),
    "shadow": lambda im, r: _shadow(im),
    # --- stochastic speckle -------------------------------------------------
    "speckle_15": lambda im, r: _speckle_lognormal(im, r, 0.15),
    "speckle_35": lambda im, r: _speckle_lognormal(im, r, 0.35),
    # --- resolution ----------------------------------------------------------
    "blur_20": lambda im, r: _u8(ndimage.gaussian_filter(im.astype(np.float32), 2.0)),
    "blur_35": lambda im, r: _u8(ndimage.gaussian_filter(im.astype(np.float32), 3.5)),
    # --- acquisition artifact -----------------------------------------------
    "jpeg_30": lambda im, r: _jpeg(im, 30),
    "jpeg_15": lambda im, r: _jpeg(im, 15),
    # --- compound "different machine" ---------------------------------------
    "compound_A": lambda im, r: _speckle_lognormal(
        _u8(ndimage.gaussian_filter(_tgc(
            _u8(255.0 * (im.astype(np.float32) / 255.0) ** 1.20), 0.5)
            .astype(np.float32), 1.6)), r, 0.25),
    "compound_B": lambda im, r: _shadow(
        _speckle_lognormal(
            _u8(ndimage.gaussian_filter(_tgc(
                _u8(255.0 * (im.astype(np.float32) / 255.0) ** 0.75), -0.5)
                .astype(np.float32), 2.2)), r, 0.30), depth=0.30),
}

GROUPS = {
    "reference": ["in_domain"],
    "affine": ["affine_gain_p25", "affine_gain_m25"],
    "nonlinear": ["gamma_060", "gamma_080", "gamma_125"],
    "spatial": ["tgc_p60", "tgc_m60", "shadow"],
    "speckle": ["speckle_15", "speckle_35"],
    "resolution": ["blur_20", "blur_35"],
    "artifact": ["jpeg_30", "jpeg_15"],
    "compound": ["compound_A", "compound_B"],
}


def _jpeg(im, q):
    import io
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(im)).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.uint8)


def cond_seed(cond):
    """Deterministic per-condition seed so every run is bit-identical."""
    h = 2166136261
    for ch in cond:
        h = (h ^ ord(ch)) * 16777619
    return h % (2 ** 31 - 1)


def apply_shift(img_u8, cond):
    rng = np.random.default_rng(cond_seed(cond))
    return SHIFTS[cond](img_u8, rng)


# ------------------------------------------- appearance standardisation (TTN) --
def reference_histogram(paths, nbins=256):
    """Mean normalised intensity histogram of a reference (training) set.

    Only the intensity MARGINAL is used, so no spatial/anatomical information
    from the training set can leak into a test prediction.
    """
    acc = np.zeros(nbins, dtype=np.float64)
    n = 0
    for p in paths:
        a = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        acc += np.bincount(a.ravel(), minlength=nbins)
        n += a.size
    return acc / max(acc.sum(), 1.0)


def hist_match_to_ref(img_u8, ref_hist):
    """Monotone remap that carries the image's intensity CDF onto ref_hist."""
    nbins = len(ref_hist)
    src = np.bincount(img_u8.ravel(), minlength=nbins).astype(np.float64)
    src_cdf = np.cumsum(src) / max(src.sum(), 1.0)
    ref_cdf = np.cumsum(ref_hist)
    lut = np.interp(src_cdf, ref_cdf, np.arange(nbins, dtype=np.float64))
    return _u8(lut[img_u8])
