#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99ac_ablate_compound.py

STAGE ONLY (writes PNGs; no model, no GPU).

compound_A and compound_B are both four/five-operator "other machine" chains, yet
step8 measured 95.54 vs 78.99 mean Dice on the same 31 val cases -- while every
individual ingredient of compound_B is nearly harmless on its own
(tgc_m60 96.35, shadow 96.13, blur_35 94.38, speckle_35 96.04, gamma_060 96.35).
That gap is either composition (the model has never seen several nuisance
structures at once) or one specific ingredient.

This stages leave-one-out variants of compound_B, all with the SAME speckle
noise realisation (seeded from the case name, not the condition name) so that
differences come from the structural operators and not from a different noise
draw.

compound_B (step8_domain_shift/tools/shift_lib.py, verbatim order) =
    shadow( speckle( blur2.2( tgc(-0.5)( gamma^0.75 ) ) ), depth=0.30 )

output: step8_domain_shift/data/shift_abl/<cond>/images/<stem>.png
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

USS = "/media/72TB_disk/personal/zk/USSnet"
S8 = os.path.join(USS, "step8_domain_shift")
sys.path.insert(0, os.path.join(S8, "tools"))
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))

from shift_lib import _u8, _tgc, _shadow, _speckle_lognormal, cond_seed  # noqa: E402
import eval_lib  # noqa: E402


def _gamma(im, g):
    return _u8(255.0 * (im.astype(np.float32) / 255.0) ** g)


def _blur(im, s):
    return _u8(ndimage.gaussian_filter(im.astype(np.float32), s))


def _compose(im, r, gamma=0.75, tgc=-0.5, blur=2.2, speck=0.30, shadow=0.30):
    x = im
    if gamma is not None:
        x = _gamma(x, gamma)
    if tgc is not None:
        x = _tgc(x, tgc)
    if blur is not None:
        x = _blur(x, blur)
    if speck is not None:
        x = _speckle_lognormal(x, r, speck)
    if shadow is not None:
        x = _shadow(x, depth=shadow)
    return x


# name -> kwargs of _compose   (None = operator removed)
VARIANTS = {
    "abA_fix":       dict(gamma=1.20, tgc=+0.5, blur=1.6, speck=0.25, shadow=None),
    "abB_rep":       dict(),                                   # must match compound_B
    "abB_noshadow":  dict(shadow=None),
    "abB_notgc":     dict(tgc=None),
    "abB_nogamma":   dict(gamma=None),
    "abB_notspeckle": dict(speck=None),
    "abB_noblur":    dict(blur=None),
    "abB_tgcsign":   dict(tgc=+0.5),
}

SRC = os.path.join(S8, "data", "shift", "in_domain", "images")
REF = os.path.join(S8, "data", "shift", "compound_B", "images")
OUT = os.path.join(S8, "data", "shift_abl")


def main():
    _, val_names = eval_lib.read_split(USS)
    print("[99ac] %d val cases, %d variants -> %s" % (len(val_names), len(VARIANTS), OUT))
    ident, diff = 0, 0
    for name in val_names:
        stem = os.path.splitext(name)[0]
        im = np.array(Image.open(os.path.join(SRC, name)).convert("L"), dtype=np.uint8)
        # one noise realisation per CASE, shared by every variant
        r = np.random.default_rng(cond_seed("abl" + stem))
        rngs = {k: np.random.default_rng(cond_seed("compound_B" if k == "abB_rep"
                                         else "abl" + stem)) for k in VARIANTS}
        for cond, kw in VARIANTS.items():
            d = os.path.join(OUT, cond, "images")
            os.makedirs(d, exist_ok=True)
            out = _compose(im, rngs[cond], **kw)
            Image.fromarray(out).save(os.path.join(d, name))
        # validate the reproduction variant against step8's own staged image
        rep = np.array(Image.open(os.path.join(OUT, "abB_rep", "images", name)))
        ref = np.array(Image.open(os.path.join(REF, name)))
        same = True
        rf = os.path.join(REF, name)
        if os.path.exists(rf):
            same = bool(np.array_equal(rep, ref))
            if same:
                ident += 1
            else:
                diff += 1
    print("[99ac] abB_rep vs step8 compound_B staged image: identical %d / %d (diff %d)"
          % (ident, ident + diff, diff))
    print("[99ac] (abB_rep uses the per-case shared seed, so a diff is expected unless"
          " the draw coincides; the ablation differences are what matter)")
    print("[99ac] staged under %s" % OUT)


if __name__ == "__main__":
    main()
