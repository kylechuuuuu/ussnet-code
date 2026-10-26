#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/32_pca_teacher.py

Fit PCA on the cached UltraSAM3 t24 teacher features (fold3 train cache) and
report the variance spectrum.  Exact PCA via chunked Gram accumulation
(1024x1024), so no memory blow-up and no subsampling.

Output: results/pca_teacher_fold3.npz  (mean, components 1024xK, evals)
Prints cumulative explained variance at k = 1..1024 (selected points) so the
reduced-cache sweep can pick k values.
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "results", "teacher_views_t24_fold3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--views", default="0,1,2,3")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pca_teacher_fold3.npz"))
    ap.add_argument("--kmax", type=int, default=1024)
    args = ap.parse_args()

    want = {"__v%d.feat.npy" % int(v) for v in args.views.split(",")}
    files = [f for f in sorted(glob.glob(os.path.join(args.cache, "*.feat.npy")))
             if any(f.endswith(w) for w in want)]
    print("[pca] %d feature files from %s" % (len(files), args.cache), flush=True)

    C = 1024
    gram = np.zeros((C, C), dtype=np.float64)
    s1 = np.zeros(C, dtype=np.float64)
    n = 0
    for i, f in enumerate(files):
        a = np.load(f).astype(np.float32)            # (C, h, 72)
        X = a.transpose(1, 2, 0).reshape(-1, C)      # (tokens, C)
        gram += X.T.astype(np.float64) @ X.astype(np.float64)
        s1 += X.sum(axis=0, dtype=np.float64)
        n += X.shape[0]
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(files)), flush=True)

    mean = (s1 / n).astype(np.float64)
    cov = gram / n - np.outer(mean, mean)
    evals, evecs = np.linalg.eigh(cov)               # ascending
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0, None)
    evecs = evecs[:, order]
    ratio = evals / evals.sum()
    cum = np.cumsum(ratio)

    print("[pca] tokens=%d | top-1 %.1f%% | cumulative:" % (n, 100 * ratio[0]), flush=True)
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        print("   k=%4d  %.3f" % (k, cum[k - 1]), flush=True)

    K = min(args.kmax, C)
    np.savez(args.out,
             mean=mean.astype(np.float32),
             components=evecs[:, :K].astype(np.float32),   # (1024, K)
             evals=evals.astype(np.float32),
             n_tokens=np.int64(n))
    print("[pca] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
