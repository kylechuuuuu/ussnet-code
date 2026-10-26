#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-off patcher for step2/scripts/13_make_figures.py:
   (1) use a CJK-capable font so the Chinese patient filenames render,
   (2) fix the fig4 subplot indexing bug (squeeze=False) that crashed when a
       single method was passed (axes shape (1,n) -> axes[r, c] IndexError)."""
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "13_make_figures.py")
s = open(P).read()

if "Noto Sans CJK" not in s:
    anchor = "import matplotlib.pyplot as plt"
    assert s.count(anchor) == 1, s.count(anchor)
    s = s.replace(anchor, anchor +
                  "\n\n# the patient filenames / titles are Chinese, so a CJK-capable font is required\n"
                  "plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']\n"
                  "plt.rcParams['axes.unicode_minus'] = False", 1)

old = ("fig, axes = plt.subplots(len(picks), len(args.tags), "
       "figsize=(4.2 * len(args.tags), 4.0 * len(picks)))")
new = ("fig, axes = plt.subplots(len(picks), len(args.tags), "
       "figsize=(4.2 * len(args.tags), 4.0 * len(picks)), squeeze=False)")
if old in s:
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, new)
    old2 = "squeeze=False)\n    axes = np.atleast_2d(axes)\n    for r, n in enumerate(picks):"
    new2 = "squeeze=False)\n    for r, n in enumerate(picks):"
    assert s.count(old2) == 1, s.count(old2)
    s = s.replace(old2, new2)
    print("fig4 layout patched")

open(P, "w").write(s)
print("patched", P)
