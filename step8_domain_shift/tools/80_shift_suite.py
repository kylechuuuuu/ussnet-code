#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_domain_shift/tools/80_shift_suite.py

Build the acquisition-shift suite for step 8 and quantify, per condition, how
much of the appearance change actually survives nnU-Net's per-image
ZScoreNormalization.

For every condition the anatomy is untouched (the GT mask is reused verbatim),
so a Dice drop measures the sensitivity of the segmentation function to
appearance, not to geometry.

Outputs
-------
  data/shift/<cond>/images/<name>.png       shifted image
  data/shift_ttn/<cond>/images/<name>.png   shifted + histogram-matched back to
                                            the training intensity marginal (TTN)
  results/shift_suite_<split>.json          per-condition statistics
  figures/step8_shifts_montage.png          visual sanity check

usage:
  python tools/80_shift_suite.py --split val [--conditions all|a,b,c] \
      [--limit N] [--no-ttn]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))

from shift_lib import SHIFTS, GROUPS, apply_shift, reference_histogram, hist_match_to_ref  # noqa: E402
import eval_lib  # noqa: E402
import importlib.util as _ilu  # noqa: E402

_fs_spec = _ilu.spec_from_file_location(
    "fold_split", os.path.join(USS, "step10_us_foundation_front", "scripts", "fold_split.py"))
fold_split = _ilu.module_from_spec(_fs_spec)
_fs_spec.loader.exec_module(fold_split)

IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")


def znorm(a):
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--conditions", default="all",
                    help="'all', a group name, or a comma-separated list")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "data", "shift"))
    ap.add_argument("--ttn-root", default=os.path.join(ROOT, "data", "shift_ttn"))
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0, help="use the k-th stored 71/31 split (0 = the project's materialised fold 0)")
    ap.add_argument("--no-ttn", action="store_true")
    args = ap.parse_args()

    if args.conditions == "all":
        conds = list(SHIFTS.keys())
    elif args.conditions in GROUPS:
        conds = list(GROUPS[args.conditions])
    else:
        conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in SHIFTS:
            raise SystemExit("unknown condition: %s" % c)

    if args.fold:      # --fold k>0 -> the k-th stored 71/31 split (PNG 原始命名)
        train_names, val_names = fold_split.read_split(args.fold)
    else:
        train_names, val_names = eval_lib.read_split(USS)
    names = train_names if args.split == "train" else val_names
    if args.limit:
        names = names[:args.limit]
    print("[80] split=%s  n=%d  conditions=%d" % (args.split, len(names), len(conds)))

    # reference intensity marginal for the TTN transform: TRAIN images only
    ref_hist = None
    if not args.no_ttn:
        ref_paths = [os.path.join(IMG_DIR, n) for n in train_names]
        ref_hist = reference_histogram(ref_paths)
        print("[80] TTN reference histogram from %d training images" % len(ref_paths))

    report = {"split": args.split, "n": len(names), "conditions": {}, "groups": GROUPS}
    examples = {}

    for cond in conds:
        d_shift = os.path.join(args.out_root, cond, "images")
        d_ttn = os.path.join(args.ttn_root, cond, "images")
        os.makedirs(d_shift, exist_ok=True)
        if ref_hist is not None:
            os.makedirs(d_ttn, exist_ok=True)

        z_dist, raw_mu, raw_sd, tt_z = [], [], [], []
        for i, name in enumerate(names):
            src = os.path.join(IMG_DIR, name)
            im = np.array(Image.open(src).convert("L"), dtype=np.uint8)
            sh = apply_shift(im, cond)
            Image.fromarray(sh).save(os.path.join(d_shift, name))
            raw_mu.append(float(sh.mean()))
            raw_sd.append(float(sh.std()))
            z_dist.append(float(np.abs(znorm(sh) - znorm(im)).mean()))
            if ref_hist is not None:
                tt = hist_match_to_ref(sh, ref_hist)
                Image.fromarray(tt).save(os.path.join(d_ttn, name))
                tt_z.append(float(np.abs(znorm(tt) - znorm(im)).mean()))
            if i == 0:
                examples[cond] = sh

        rec = {
            "n": len(names),
            "raw_mean": float(np.mean(raw_mu)),
            "raw_std": float(np.mean(raw_sd)),
            "zscore_residual_px": float(np.mean(z_dist)),
        }
        if tt_z:
            rec["ttn_zscore_residual_px"] = float(np.mean(tt_z))
        report["conditions"][cond] = rec
        print("[80] %-16s raw mu=%6.2f sd=%6.2f  z-resid=%.4f%s"
              % (cond, rec["raw_mean"], rec["raw_std"], rec["zscore_residual_px"],
                 ("  ttn-resid=%.4f" % rec["ttn_zscore_residual_px"]) if tt_z else ""))

    # baseline (unshifted) reference statistics
    base_im = np.array(Image.open(os.path.join(IMG_DIR, names[0])).convert("L"), dtype=np.uint8)
    report["in_domain"] = {"n": len(names),
                           "raw_mean": float(np.mean([np.array(Image.open(
                               os.path.join(IMG_DIR, n)).convert("L")).mean() for n in names]))}

    os.makedirs(args.results_dir, exist_ok=True)
    out_json = os.path.join(args.results_dir, "shift_suite_%s.json" % args.split)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print("[80] wrote %s" % out_json)

    # ---------------------------------------------------------------- montage
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = ["in_domain"] + [c for c in conds if c in examples]
        show = show[:8]
        ncol = 4
        nrow = int(np.ceil(len(show) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, cond in zip(axes, show):
            ax.imshow(base_im if cond == "in_domain" else examples[cond], cmap="gray")
            extra = ""
            if cond in report["conditions"]:
                extra = "\nz-resid %.3f" % report["conditions"][cond]["zscore_residual_px"]
            ax.set_title(cond + extra, fontsize=9)
            ax.axis("off")
        for ax in axes[len(show):]:
            ax.axis("off")
        fig.tight_layout()
        fig_dir = os.path.join(ROOT, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        out_png = os.path.join(fig_dir, "step8_shifts_montage.png")
        fig.savefig(out_png, dpi=110)
        print("[80] wrote %s" % out_png)
    except Exception as e:                                   # pragma: no cover
        print("[80] montage skipped: %r" % (e,))


if __name__ == "__main__":
    main()
