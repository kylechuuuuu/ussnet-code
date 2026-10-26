#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/11_cka_channels.py

Two zero-training measurements on the frozen US-foundation-model encoders, both
needed before any compression work (L1 of the design):

(A) CKA CROSS-SIMILARITY MATRIX
    Linear CKA between the patch-feature matrices of every pair of encoders on
    the same images.  This answers "are these teachers a single bias seen twice,
    or genuinely complementary representations?" -- the same question step9
    answered with r_err = 0.84-0.90 for two *segmentation* models (which is why
    two-model interval squeezing failed).  High CKA everywhere => multi-teacher
    distillation has little to add; low CKA between two teachers => the pair is
    worth keeping.

(B) CHANNEL IMPORTANCE FOR THE BOUNDARY
    For every channel of every encoder: how well does it separate the boundary
    band (+-2 px of the GT contour) from the gland interior?  Reported as the
    mean per-image AUC, sorted, plus how many channels carry 90 % of the total
    |2*AUC-1| mass.  This is the prior for the channel-gate that will perform
    the actual operator extraction, and for mixed-precision assignment later.

usage:
  ~/anaconda3/envs/zk/bin/python scripts/11_cka_channels.py \
      --models ultrasam,ultrafedfm,ultrasam3 --device cuda:0 [--limit 31] [--dump]

--dump stores the per-image feature grids under results/feats/<model>/ so the
analysis can be re-run without touching the GPU again.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
FEATS = os.path.join(RESULTS, "feats")
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location("l0", os.path.join(HERE, "10_l0_probe.py"))
l0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l0)

# per-model canvas: SAM-family trunks need a patch-aligned square canvas
CANVAS_OF = {"ultrasam": (1024, 1024), "ultrasam3": (1008, 1008),
             "ultrafedfm": (1024, 1024), "fetalclip": (1024, 1024), "armc_imagenet": (1024, 1024)}
GRID = 16                       # common CKA grid (16x16 cells over the canvas)


def extract(model, names, device, cond="in_domain", dump=False):
    """-> (feats (N,C,g,g), labels (N,g,g)) on a common GRID, or from cache."""
    cache = os.path.join(FEATS, "%s.npz" % model)
    if os.path.exists(cache):
        z = np.load(cache)
        print("[cka] %s: loaded cache %s" % (model, cache), flush=True)
        return z["feats"], z["labels"], z["valid"]
    cw, ch = CANVAS_OF[model]
    l0.CANVAS = (cw, ch)
    enc = l0.build_encoder(model, device, (cw, ch))
    feats, labels = [], []
    for n in names:
        img, gt_c = l0.load_case(n, cond)
        f = enc.encode(img)                                    # (C, gh, gw)
        g = F.interpolate(f[None], size=(GRID, GRID), mode="bilinear", align_corners=False)[0]
        feats.append(g.cpu().numpy().astype(np.float16))
        lab = l0.grid_labels(gt_c, cw // GRID)                 # (GRID, GRID) bool
        labels.append(lab)
    feats = np.stack(feats)
    labels = np.stack(labels)
    valid = l0.valid_cells(cw // GRID)
    if dump:
        os.makedirs(FEATS, exist_ok=True)
        np.savez_compressed(cache, feats=feats, labels=labels, valid=valid)
        print("[cka] %s: cached -> %s" % (model, cache), flush=True)
    return feats, labels, valid


def linear_cka(X, Y):
    """Linear CKA between two (n_samples, n_features) matrices."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y) ** 2
    xtx = np.linalg.norm(X.T @ X)
    yty = np.linalg.norm(Y.T @ Y)
    return float(xty / (xtx * yty + 1e-12))


def boundary_band(lab, radius_cells=1):
    """Cells whose neighbourhood contains both foreground and background."""
    fg = torch.from_numpy(lab.astype(np.float32))[None, None]
    k = 2 * radius_cells + 1
    dil = F.max_pool2d(fg, k, 1, radius_cells) > 0.5
    ero = F.max_pool2d(1 - fg, k, 1, radius_cells) < 0.5
    return (dil & ero)[0, 0].numpy()


def auc_of_channel(x, pos, neg):
    """Rank AUC of channel x separating pos cells from neg cells."""
    p, q = x[pos], x[neg]
    if p.size == 0 or q.size == 0:
        return np.nan
    allv = np.concatenate([p, q])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    rp = ranks[:p.size].sum()
    return float((rp - p.size * (p.size + 1) / 2.0) / (p.size * q.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ultrasam,ultrafedfm,ultrasam3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=31)
    ap.add_argument("--condition", default="in_domain")
    ap.add_argument("--dump", action="store_true", default=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    train_names, val_names = l0.eval_lib.read_split(l0.USS)
    names = val_names[:args.limit] if args.limit else val_names
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    store = {}
    for m in models:
        feats, labels, valid = extract(m, names, dev, args.condition, args.dump)
        store[m] = (feats, labels, valid)
        print("[cka] %s: feats %s labels %s" % (m, feats.shape, labels.shape), flush=True)

    # ---------------- (A) CKA matrix ----------------
    print("\n[cka] linear CKA between encoders (same %d images, %dx%d cell grid):"
          % (len(names), GRID, GRID), flush=True)
    keys = list(store.keys())
    hdr = "%-12s" % "" + "".join("%12s" % k for k in keys)
    print(hdr, flush=True)
    cka_mat = {}
    for a in keys:
        fa, _, va = store[a]
        Xa = fa.reshape(len(fa), fa.shape[1], -1)[:, :, va.reshape(-1)].transpose(0, 2, 1)
        Xa = Xa.reshape(-1, Xa.shape[-1]).astype(np.float32)
        row = []
        for b in keys:
            fb, _, vb = store[b]
            Xb = fb.reshape(len(fb), fb.shape[1], -1)[:, :, vb.reshape(-1)].transpose(0, 2, 1)
            Xb = Xb.reshape(-1, Xb.shape[-1]).astype(np.float32)
            c = linear_cka(Xa, Xb) if a != b else 1.0
            row.append(c)
            cka_mat["%s|%s" % (a, b)] = c
        print("%-12s" % a + "".join("%12.3f" % v for v in row), flush=True)

    # ---------------- (B) channel importance ----------------
    print("\n[cka] boundary-band channel importance (mean per-image AUC of "
          "|2*AUC-1|, top channels carry the boundary signal):", flush=True)
    chan = {}
    for m in keys:
        feats, labels, valid = store[m]
        aucs = []
        for i in range(len(feats)):
            band = boundary_band(labels[i])
            inside = labels[i] & ~band
            pos, neg = (band & valid), (inside & valid)
            if pos.sum() < 4 or neg.sum() < 4:
                continue
            x = feats[i].reshape(feats[i].shape[0], -1)
            a = np.array([auc_of_channel(x[c], pos.reshape(-1), neg.reshape(-1))
                          for c in range(x.shape[0])])
            aucs.append(a)
        if not aucs:
            continue
        a = np.nanmean(np.stack(aucs), axis=0)
        strength = np.abs(2 * a - 1)
        ok = ~np.isnan(strength)
        strength = np.nan_to_num(strength)
        order = strength.argsort()[::-1]
        total = strength.sum()
        k90 = int(np.searchsorted(np.cumsum(strength[order]) / total, 0.9) + 1)
        chan[m] = {"n_channels": int(len(a)), "mean_auc": float(np.nanmean(a[ok])),
                   "top_auc": float(a[order][:10].mean()),
                   "k_for_90pct": k90, "frac_for_90pct": float(k90 / len(a)),
                   "top_idx": order[:32].tolist()}
        print("   %-12s C=%-5d mean|2AUC-1| %.3f  top10 mean AUC %.3f  "
              "channels for 90%% of the boundary signal: %d/%d (%.1f%%)"
              % (m, len(a), float(strength.mean()), float(np.nanmean(a[order][:10])),
                 k90, len(a), 100.0 * k90 / len(a)), flush=True)

    out = os.path.join(RESULTS, "cka_channels.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"condition": args.condition, "n_images": len(names), "grid": GRID,
                   "models": keys, "cka": cka_mat, "channels": chan}, fh, indent=2)
    print("\n[cka] wrote %s" % out)


if __name__ == "__main__":
    sys.exit(main())
