#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How usable is the 127/255 zone split as an auxiliary target?

Prints, over the 102 masks, how many contain value 127 only / 255 only / both,
and the per-image area fraction of each zone.  A zone split that is present in
every image is a real internal boundary (usable auxiliary supervision); one that
appears in a handful of images is an annotation artefact."""
import glob
import os

import numpy as np
from PIL import Image

# tools/ -> step4_our_new_model_resunet_boundary/ -> USSnet/
USS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
MS = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))
print("MASK_DIR =", MASK_DIR)

both = only127 = only255 = neither = 0
frac127, frac255 = [], []
weird = {}
for m in MS:
    a = np.array(Image.open(m))
    u = set(np.unique(a).tolist())
    f127 = float((a == 127).sum()) / a.size
    f255 = float((a == 255).sum()) / a.size
    if u - {0, 127, 255}:
        weird[os.path.basename(m)] = sorted(u)
    if 127 in u and 255 in u:
        both += 1
        frac127.append(f127)
        frac255.append(f255)
    elif 127 in u:
        only127 += 1
    elif 255 in u:
        only255 += 1
    else:
        neither += 1

print("masks                     : %d" % len(MS))
print("has 127 and 255           : %d" % both)
print("has 127 only              : %d" % only127)
print("has 255 only              : %d" % only255)
print("has neither               : %d" % neither)
if weird:
    print("unexpected values in      : %d file(s), e.g. %s"
          % (len(weird), list(weird.items())[:3]))
if both:
    f127 = np.array(frac127)
    f255 = np.array(frac255)
    print("zone 127 area fraction    : mean %.4f  median %.4f  min %.4f  max %.4f"
          % (f127.mean(), np.median(f127), f127.min(), f127.max()))
    print("zone 255 area fraction    : mean %.4f  median %.4f  min %.4f  max %.4f"
          % (f255.mean(), np.median(f255), f255.min(), f255.max()))
    r = f127 / (f127 + f255)
    print("127 share of the union    : mean %.3f  median %.3f  min %.3f  max %.3f"
          % (r.mean(), np.median(r), r.min(), r.max()))
