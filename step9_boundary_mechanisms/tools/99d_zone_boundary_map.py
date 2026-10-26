#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which zone OWNS the union boundary arc, and where?

Decides whether the anterior union boundary is the TZ(255) anterior edge or the
PZ(127) anterior edge.  This matters because the two have very different image
support (90_error_anatomy Part B: PZ outer |gradI| 8.78 vs TZ outer 6.32) and
because, if the anterior union edge is a zone edge, that edge is a visible
structure the model could anchor on, whereas a fascia outside both zones is not.

GT only, no model, CPU.
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

USS = "/media/72TB_disk/personal/zk/USSnet"
MASK = os.path.join(USS, "UltrasoundProstate", "masks")
IMG = os.path.join(USS, "UltrasoundProstate", "images")
sys.path.insert(0, os.path.join(USS, "step6_sandwich_squeeze", "tools"))
import importlib.util                                                    # noqa: E402

spec = importlib.util.spec_from_file_location(
    "arb", os.path.join(USS, "step6_sandwich_squeeze", "tools", "63_arbitrate.py"))
arb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arb)
spec57 = importlib.util.spec_from_file_location(
    "s57", os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools",
                        "57_shape_prior_ceiling.py"))
s57 = importlib.util.module_from_spec(spec57)
spec57.loader.exec_module(s57)

train, val = arb.read_split(USS)
names = val
print("cases:", len(names))


def surface(m):
    st = ndi.generate_binary_structure(2, 1)
    return m & ~ndi.binary_erosion(m, st, border_value=0)


NBINS = 24
edges = np.linspace(-180, 180, NBINS + 1)
cnt = {"pz": np.zeros(NBINS), "tz": np.zeros(NBINS)}
gsum = {"pz": np.zeros(NBINS), "tz": np.zeros(NBINS)}
area_frac = []
step_pz, step_tz = [], []

for name in names:
    stem = arb.stem_of(name)
    m = np.array(Image.open(os.path.join(MASK, name)).convert("L"))
    img = np.asarray(Image.open(os.path.join(IMG, stem + ".png")).convert("L"), dtype=np.float32)
    u = m > 0
    pz = m == 127
    tz = m == 255
    area_frac.append(pz.sum() / max(u.sum(), 1))
    bu = surface(u)
    # the zone label of the pixel just INSIDE the union boundary
    owner_pz = pz & ndi.binary_dilation(bu, iterations=1)
    owner_tz = tz & ndi.binary_dilation(bu, iterations=1)
    interior_pz = ndi.binary_dilation(owner_pz, iterations=2) & bu
    interior_tz = ndi.binary_dilation(owner_tz, iterations=2) & bu
    amp = np.abs(ndi.gaussian_filter(img, 1.5) - ndi.gaussian_filter(img, 5.0))
    yy, xx = np.nonzero(bu)
    cy, cx = yy.mean(), xx.mean()
    ang = np.degrees(np.arctan2(-(yy - cy), xx - cx))
    b = np.clip(((ang + 180) / 360 * NBINS).astype(int), 0, NBINS - 1)
    for i in range(len(yy)):
        key = "pz" if interior_pz[yy[i], xx[i]] else ("tz" if interior_tz[yy[i], xx[i]] else None)
        if key is None:
            continue
        cnt[key][b[i]] += 1
        gsum[key][b[i]] += amp[yy[i], xx[i]]

print("\npz share of union area: mean %.3f (min %.3f max %.3f)"
      % (np.mean(area_frac), np.min(area_frac), np.max(area_frac)))

lab = ["%d..%d" % (edges[i], edges[i + 1]) for i in range(NBINS)]
print("\n%-12s %8s %8s | %8s %8s" % ("angle bin", "n(PZ)", "n(TZ)", "grad(PZ)", "grad(TZ)"))
for i in range(NBINS):
    print("%-12s %8d %8d | %8.2f %8.2f"
          % (lab[i], cnt["pz"][i], cnt["tz"][i],
             gsum["pz"][i] / max(cnt["pz"][i], 1), gsum["tz"][i] / max(cnt["tz"][i], 1)))

print("\n%-12s %8s %8s" % ("region", "PZ share", "TZ share"))
for lo, hi, nm in [(-180, -120, "posterior-lat"), (-120, -90, "postero-ant?"),
                   (-90, -60, "anterior-mid"), (-60, 0, "anterior-lat"),
                   (0, 60, "lat"), (60, 120, "postero"), (120, 180, "posterior")]:
    m = (edges[:-1] >= lo) & (edges[:-1] < hi)
    np_, nt = cnt["pz"][m].sum(), cnt["tz"][m].sum()
    tot = np_ + nt
    print("%-12s %8.3f %8.3f   (n=%d)" % (nm, np_ / max(tot, 1), nt / max(tot, 1), tot))
