#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe UltrasoundProstate dataset: sizes, mask label semantics, patient grouping."""
import os
import collections
import numpy as np
from PIL import Image

ROOT = "/media/72TB_disk/personal/zk/USSnet/UltrasoundProstate"
IMG = os.path.join(ROOT, "images")
MSK = os.path.join(ROOT, "masks")

imgs = sorted(os.listdir(IMG))
msks = sorted(os.listdir(MSK))
print("n_img=%d  n_mask=%d" % (len(imgs), len(msks)))

sizes = collections.Counter()
immodes = collections.Counter()
labelcombos = collections.Counter()
areas = {0: [], 127: [], 255: []}
tot127 = tot255 = tot0 = 0
shapes = collections.Counter()

for n in msks:
    ip = os.path.join(IMG, n)
    mp = os.path.join(MSK, n)
    if not os.path.exists(ip):
        print("!! no image for", n)
        continue
    im = Image.open(ip)
    m = Image.open(mp)
    immodes[im.mode] += 1
    sizes[im.size] += 1
    a = np.array(m.convert("L"))
    shapes[a.shape] += 1
    u, c = np.unique(a, return_counts=True)
    d = dict(zip(u.tolist(), c.tolist()))
    labelcombos[tuple(sorted(d.keys()))] += 1
    for v in (0, 127, 255):
        areas[v].append(d.get(v, 0))
    tot0 += d.get(0, 0)
    tot127 += d.get(127, 0)
    tot255 += d.get(255, 0)

print("image modes:", dict(immodes))
print("image sizes:", dict(sizes))
print("mask value-sets:", dict(labelcombos))
print("pixel totals  bg0=%d  v127=%d  v255=%d" % (tot0, tot127, tot255))
print("fraction of non-zero that is 127: %.3f" % (tot127 / max(tot127 + tot255, 1)))
for v in (0, 127, 255):
    arr = np.array(areas[v], dtype=float)
    print("  value %3d : mean_area=%.0f median=%.0f min=%.0f max=%.0f  nonzero_imgs=%d/%d"
          % (v, arr.mean(), np.median(arr), arr.min(), arr.max(), int((arr > 0).sum()), len(arr)))

pat = collections.Counter(n.rsplit(".", 1)[0].split("_")[0] for n in msks)
print("n_patients=%d  slices/patient=%s" % (len(pat), sorted(collections.Counter(pat.values()).items())))

# background crop of ultrasound sector (black side bands?)
im = np.array(Image.open(os.path.join(IMG, imgs[0])).convert("L"))
print("sample image shape", im.shape, "min", im.min(), "max", im.max(),
      "mean", round(float(im.mean()), 1))
