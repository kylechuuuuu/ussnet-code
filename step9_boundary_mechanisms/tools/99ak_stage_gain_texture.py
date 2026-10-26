#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99ak_stage_gain_texture.py

Two extra leave-one-*in* conditions that settle the mechanism behind the
compound_B collapse (the existing 99ac ablation only removed ingredients):

  abC_texture_only   blur(sigma=2.2) + speckle(0.30), NO gamma / tgc / shadow
                     -> is destroying the high-frequency texture enough?
  abD_gain_only      gamma^0.75 + tgc(-0.5) + shadow(0.30), NO blur / speckle
                     -> is the smooth multiplicative gain enough on its own?

staged into step8_domain_shift/data/shift_abl/<cond>/images/ so the existing
99ad image-root works unchanged.  Same per-case speckle realisation convention
(seed = cond_seed("abl"+stem)) as the 99ac variants.
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

USS = "/media/72TB_disk/personal/zk/USSnet"
S8 = os.path.join(USS, "step8_domain_shift")
sys.path.insert(0, os.path.join(S8, "tools"))
import shift_lib as SL  # noqa: E402

SRC = os.path.join(S8, "data", "shift", "in_domain", "images")
OUT = os.path.join(S8, "data", "shift_abl")
CONDS = ["abC_texture_only", "abD_gain_only"]


def stage(name, im, rng):
    """mirror the exact compound_B call chain, with identity ops where noted."""
    if name == "abC_texture_only":
        # gamma 1.00 (identity) -> blur 2.2 -> speckle 0.30, no tgc, no shadow
        a = SL._u8(255.0 * (im / 255.0) ** 1.00)
        a = SL._u8(ndimage.gaussian_filter(a.astype(np.float32), 2.2))
        return SL._speckle_lognormal(a, rng, 0.30)
    # abD_gain_only: gamma^0.75 -> tgc(-0.5) -> shadow(0.30), no blur/speckle
    a = SL._u8(255.0 * (im / 255.0) ** 0.75)
    a = SL._tgc(a, -0.5)
    return SL._shadow(a, depth=0.30)


def main():
    for c in CONDS:
        os.makedirs(os.path.join(OUT, c, "images"), exist_ok=True)
    for f in sorted(os.listdir(SRC)):
        im = np.array(Image.open(os.path.join(SRC, f)).convert("L"), dtype=np.uint8)
        rng = np.random.default_rng(SL.cond_seed("abl" + os.path.splitext(f)[0]))
        for c in CONDS:
            Image.fromarray(stage(c, im, rng)).save(os.path.join(OUT, c, "images", f))
    print("[99ak] staged %d cases x %d conditions into %s"
          % (len(os.listdir(SRC)), len(CONDS), OUT))


if __name__ == "__main__":
    main()
