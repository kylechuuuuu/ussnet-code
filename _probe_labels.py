#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check spatial relation between mask label 127 and 255 (nested? disjoint? ring?)."""
import os
import numpy as np
from PIL import Image

ROOT = "/media/72TB_disk/personal/zk/USSnet/UltrasoundProstate"
MSK = os.path.join(ROOT, "masks")
names = sorted(os.listdir(MSK))

inner127_in_255 = 0
inner255_in_127 = 0
rows = []
for n in names:
    a = np.array(Image.open(os.path.join(MSK, n)).convert("L"))
    m127 = a == 127
    m255 = a == 255
    # bounding boxes
    def bbox(m):
        ys, xs = np.where(m)
        return ys.min(), ys.max(), xs.min(), xs.max()
    b1, b2 = bbox(m127), bbox(m255)
    # is the 255 bbox contained in the 127 bbox?
    c255in127 = (b2[0] >= b1[0] and b2[1] <= b1[1] and b2[2] >= b1[2] and b2[3] <= b1[3])
    c127in255 = (b1[0] >= b2[0] and b1[1] <= b2[1] and b1[2] >= b2[2] and b1[3] <= b2[3])
    inner127_in_255 += c127in255
    inner255_in_127 += c255in127
    # per-row/col: which comes first, and is one always surrounded
    rows.append((n, b1, b2, c255in127, c127in255))

print("255 bbox inside 127 bbox : %d/%d" % (inner255_in_127, len(names)))
print("127 bbox inside 255 bbox : %d/%d" % (inner127_in_255, len(names)))
for r in rows[:6]:
    print(r)

# For one image, print a coarse ASCII map to see the nesting
a = np.array(Image.open(os.path.join(MSK, names[0])).convert("L"))
step = 24
h, w = a.shape
print("\nASCII map (P=255, i=127, .=0), image", names[0], "step", step)
for y in range(0, h, step):
    line = ""
    for x in range(0, w, step):
        blk = a[y:y + step, x:x + step]
        if (blk == 255).sum() > (blk == 127).sum() and (blk == 255).sum() > 0:
            line += "P"
        elif (blk == 127).sum() > 0:
            line += "i"
        else:
            line += "."
    print(line)
