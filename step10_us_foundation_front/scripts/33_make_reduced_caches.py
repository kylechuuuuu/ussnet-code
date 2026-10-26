#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/33_make_reduced_caches.py

Build reduced teacher-feature caches for the "core content retention" sweep.
Each cache mirrors results/teacher_views_t24_fold{K} (same _meta.json seeds, same
file names) but stores only k dimensions per token:

    pca{k}     (x - mean) @ W_k        -- W_k from 32_pca_teacher.py (data-driven)
    rand{k}    (x - mean) @ R_k/sqrt(k) -- Gaussian random projection (JL control)
    chan{k}    x[:, :k]                -- first-k channels, UNCENTERED (L0 control)
    chanc{k}   (x - mean)[:, :k]       -- first-k channels, CENTERED
    mean1024   (x - mean)              -- centering only, no reduction
    noise{k}   iid N(0,1) per token    -- zero-mean target with NO teacher content

usage: python scripts/33_make_reduced_caches.py --fold 3 --methods pca4,mean1024,...
"""
import argparse
import glob
import json
import os
import shutil
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--methods", default="pca64,pca16,pca4,pca256,rand64,chan64")
    args = ap.parse_args()

    src = os.path.join(ROOT, "results", "teacher_views_t24_fold%d" % args.fold)
    pca = os.path.join(ROOT, "results", "pca_teacher_fold%d.npz" % args.fold)
    z = np.load(pca)
    mu = z["mean"].astype(np.float32)             # (1024,)
    W = z["components"].astype(np.float32)        # (1024, 1024)
    feats = sorted(glob.glob(os.path.join(src, "*.feat.npy")))
    print("[red] fold %d: %d source feature files" % (args.fold, len(feats)), flush=True)

    for spec in [s.strip() for s in args.methods.split(",") if s.strip()]:
        method = "".join(ch for ch in spec if ch.isalpha())
        k = int("".join(ch for ch in spec if ch.isdigit()) or 0)
        out = os.path.join(ROOT, "results", "teacher_views_%s_fold%d" % (spec, args.fold))
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(src, "_meta.json"), os.path.join(out, "_meta.json"))
        if method == "pca":
            P = W[:, :k]
        elif method == "rand":
            rng = np.random.default_rng(0)
            P = (rng.standard_normal((1024, k)) / np.sqrt(k)).astype(np.float32)
        elif method in ("chan", "chanc", "mean"):
            P = None
        elif method == "noise":
            P = None
        else:
            raise SystemExit("unknown method %s" % method)

        t0 = time.time()
        for i, f in enumerate(feats):
            a = np.load(f).astype(np.float32)       # (1024, h, w)
            C, h, w = a.shape
            X = a.transpose(1, 2, 0).reshape(-1, C)  # (tokens, 1024)
            if method == "chan":
                Y = X[:, :k]
            elif method == "chanc":
                Y = (X - mu)[:, :k]
            elif method == "mean":
                Y = (X - mu)
            elif method == "noise":
                rng = np.random.default_rng(zlib.crc32(os.path.basename(f).encode()) % (2 ** 31))
                Y = rng.standard_normal((X.shape[0], k)).astype(np.float32)
            else:
                Y = (X - mu) @ P
            np.save(os.path.join(out, os.path.basename(f)),
                    Y.reshape(h, w, Y.shape[1]).transpose(2, 0, 1).astype(np.float16))
        print("[red] %s: k=%d  %d files  (%.1f min)" % (spec, Y.shape[1], len(feats), (time.time() - t0) / 60), flush=True)
    print("[red] done", flush=True)


if __name__ == "__main__":
    main()
