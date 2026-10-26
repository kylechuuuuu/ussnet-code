#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_compare_unet_vs_nnunet/scripts/09_probe_gt_definition.py

Quick probe: what exactly is the U-Net baseline being scored against?

The raw masks hold two abutting labels (127 = "outer" zone, 255 = "inner"
zone).  We therefore measure the existing U-Net predictions against three
possible ground truths (union / 255-only / 127-only) and inspect how the
prediction relates to the internal 127|255 interface.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_lib import load_binary, read_split  # noqa: E402

STEP2 = os.path.dirname(HERE)
USS = os.path.dirname(STEP2)
GT = os.path.join(USS, "UltrasoundProstate", "masks")
PRED = os.path.join(USS, "unet", "results", "ultrasound_prostate_73")


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    return 2.0 * inter / (a.sum() + b.sum() + 1e-7) * 100.0


def main():
    _, val = read_split(USS)
    rows = []
    for n in val:
        p = load_binary(os.path.join(PRED, n))
        g = np.array(Image.open(os.path.join(GT, n)).convert("L"))
        union, g255, g127 = g > 0, g == 255, g == 127
        rows.append(dict(
            name=n, pred=int(p.sum()), union=int(union.sum()),
            l127=int(g127.sum()), l255=int(g255.sum()),
            d_union=dice(p, union), d_255=dice(p, g255), d_127=dice(p, g127),
            frac_pred_in_127=float(np.logical_and(p, g127).sum()) / (p.sum() + 1e-7) * 100,
            frac_pred_in_255=float(np.logical_and(p, g255).sum()) / (p.sum() + 1e-7) * 100,
            union_cov=float(np.logical_and(p, union).sum()) / (union.sum() + 1e-7) * 100,
        ))

    print("%-28s %9s %9s %9s %9s | %7s %7s %7s | %6s %6s %6s" %
          ("image", "|pred|", "|union|", "|127|", "|255|",
           "D_union", "D_255", "D_127", "%in127", "%in255", "cov%"))
    for r in rows:
        print("%-28s %9d %9d %9d %9d | %7.2f %7.2f %7.2f | %6.1f %6.1f %6.1f" %
              (r["name"][:28], r["pred"], r["union"], r["l127"], r["l255"],
               r["d_union"], r["d_255"], r["d_127"],
               r["frac_pred_in_127"], r["frac_pred_in_255"], r["union_cov"]))
    print("-" * 130)
    for k in ("d_union", "d_255", "d_127", "frac_pred_in_127", "frac_pred_in_255", "union_cov"):
        print("mean %-18s %8.3f" % (k, float(np.mean([r[k] for r in rows]))))
    tot_pred = sum(r["pred"] for r in rows)
    print("\npooled |pred| %d  |union| %d  |127| %d  |255| %d" %
          (tot_pred, sum(r["union"] for r in rows), sum(r["l127"] for r in rows),
           sum(r["l255"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
