#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DESIGN_NEXT.md section 3.A, built the one way that needs NO clean stage-1.

Why synthetic perturbations
---------------------------
§5.2 proved the ceiling is huge (oracle normal relocation = +49.7 boundary-Dice
points on nnU-Net).  §5.5 proved the *image alone* is not enough locally: a
hand-written ridge search gets +1.3 while the global ridge gets -13.6.
The 2x2 conclusion is that a refiner needs BOTH direction-awareness AND learned
context -- i.e. it must be a learned second stage.

Training it on fold-0 train is normally impossible: the only available stage-1
weights were trained there, so their train-set predictions have ~0 displacement
and teach nothing.  This script sidesteps that completely: it never asks a model
for anything.  It builds training pairs from

    (real image, GT contour + a SYNTHETIC smooth normal perturbation)

and the target is the oracle displacement ``-sdf(p)`` (exactly the move §5.2
measured), so a perfect refiner == the §5.2 oracle.  Nothing is read from any
trained model, so there is no fold-0 contamination and no extra stage-1 run.

Inputs per contour point (normal-aligned frame, v = normal, u = tangential)
    ch0  image intensity, per-image z-scored
    ch1  |grad I| of the z-scored image
    scalars: curvature, normalised radius, arc fraction, centre intensity,
             centre |grad I|, contour scale
The patch is 2-D in (v, u), so the network sees real 2-D edge/speckle structure
-- strictly more than the 1-D profile that failed in §5.5 -- and the scalars
carry the shape context that §5.5 lacked entirely.

Output: one signed displacement per point, tanh-bounded to +-16 px.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from PIL import Image

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS = os.path.dirname(STEP4)

spec = importlib.util.spec_from_file_location(
    "shape57", os.path.join(HERE, "57_shape_prior_ceiling.py"))
s57 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s57)
spec58 = importlib.util.spec_from_file_location(
    "refine58", os.path.join(HERE, "58_normal_refine_oracle.py"))
s58 = importlib.util.module_from_spec(spec58)
spec58.loader.exec_module(s58)

IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
NPT = 400
MAXD = 16.0
VH, VW = 16, 8          # half extents: normal +-16 px, tangential +-8 px
NV, NU = 2 * VH + 1, 2 * VW + 1      # 33 x 17
NSCALAR = 6
NREP = 4                # 1 clean anchor pass + 3 perturbed passes


# ------------------------------------------------------------------ helpers --
def load_image(name):
    base = name if name.lower().endswith(".png") else name + ".png"
    p = os.path.join(IMG_DIR, base)
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p).convert("L"), dtype=np.float32)
    return a


def grad_mag(img):
    gy, gx = np.gradient(img)
    return np.hypot(gy, gx)


def bil(img, pts):
    return ndimage.map_coordinates(img, [pts[:, 0], pts[:, 1]], order=1,
                                   mode="nearest")


def prep_image(img):
    im = ndimage.gaussian_filter(img, 1.0)
    mu, sd = float(im.mean()), float(im.std() + 1e-6)
    z = (im - mu) / sd
    g = grad_mag(z)
    gs = g.std() + 1e-6
    return z, g / gs


def make_targets(gt, pts):
    """Oracle target: move along the normal by -sdf(p), clamped to +-MAXD."""
    gb = s57.surface(gt)
    d = ndimage.distance_transform_edt(~gb)
    sdf = np.where(gt, -d, d)
    t = -bil(sdf.astype(np.float32), pts)
    return np.clip(t, -MAXD, MAXD), sdf


def perturb_field(n, rng):
    """Smooth random normal displacement, amplitude heavy-tailed, matching the
    measured stage-1 error scale (median ~2.6 px, p90 ~11 px)."""
    sig = rng.uniform(0.02, 0.20) * n
    raw = rng.standard_normal(n)
    d = ndimage.gaussian_filter1d(raw, max(sig, 1.0), mode="wrap")
    d = d / (d.std() + 1e-9)
    amp = float(np.abs(rng.normal(0.0, 1.0)) * rng.choice([0.5, 0.9, 1.5, 2.5, 4.0, 7.0, 11.0]))
    return np.clip(d * amp, -MAXD, MAXD)


def extract(pts, nrm, z, g, gt, rng=None):
    """-> (patches (N,2,NV,NU), scalars (N,6), target (N,)) at `pts`."""
    n = len(pts)
    # tangential direction (rotate the normal back)
    tan = np.stack([nrm[:, 1], -nrm[:, 0]], axis=1)
    vs = np.arange(-VH, VH + 1, dtype=float)
    us = np.arange(-VW, VW + 1, dtype=float)
    V, U = np.meshgrid(vs, us, indexing="ij")           # (NV, NU)
    q = (pts[:, None, None, :]
         + V[None, :, :, None] * nrm[:, None, None, :]
         + U[None, :, :, None] * tan[:, None, None, :])
    rr = q[..., 0].reshape(n, -1)
    cc = q[..., 1].reshape(n, -1)
    pz = ndimage.map_coordinates(z, [rr.ravel(), cc.ravel()], order=1,
                                 mode="nearest").reshape(n, NV, NU)
    pg = ndimage.map_coordinates(g, [rr.ravel(), cc.ravel()], order=1,
                                 mode="nearest").reshape(n, NV, NU)
    patches = np.stack([pz, pg], axis=1).astype(np.float32)     # (N,2,NV,NU)

    # scalars: curvature, normalised radius, arc fraction, centre I, centre |g|,
    # contour scale
    d1 = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    d2 = np.roll(pts, -2, axis=0) - 2 * pts + np.roll(pts, 2, axis=0)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    kappa = np.abs(cross) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-6)
    cen = pts.mean(axis=0)
    R = float(np.sqrt(((pts - cen) ** 2).sum(axis=1).mean()) + 1e-6)
    rnorm = np.linalg.norm(pts - cen, axis=1) / R
    arc = np.arange(n) / float(n) * 2.0 - 1.0
    cen_i = bil(z, pts)
    cen_g = bil(g, pts)
    scal = np.stack([np.clip(kappa * 40.0, 0, 5), rnorm, arc, cen_i, cen_g,
                     np.full(n, R / 280.0)], axis=1).astype(np.float32)
    tgt, _ = make_targets(gt, pts)
    return patches, scal, tgt.astype(np.float32)


# -------------------------------------------------------------------- model --
class Refiner(nn.Module):
    def __init__(self, nscalar=NSCALAR):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
        self.c1 = blk(2, 16)
        self.c2 = blk(16, 24)
        self.c3 = blk(24, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sf = nn.Sequential(nn.Linear(nscalar, 16), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(32 + 16, 48), nn.ReLU(inplace=True),
                                  nn.Linear(48, 1))

    def forward(self, x, s):
        h = self.c3(self.c2(self.c1(x)))
        h = self.pool(h).flatten(1)
        h = torch.cat([h, self.sf(s)], dim=1)
        return torch.tanh(self.head(h)).squeeze(1) * MAXD


# ------------------------------------------------------------------ dataset --
def build(names, rng, n_per_img=NPT, verbose=True):
    P, S, T = [], [], []
    for nm in names:
        gt = s57.load_gt(nm)
        img = load_image(nm)
        if gt is None or img is None:
            continue
        z, g = prep_image(img)
        c = s57.contour_of(s57.largest_cc(gt))
        if c is None:
            continue
        pts = s57.resample_closed(c, n_per_img, 0)
        nrm = s58.normals_of(pts, gt)
        for rep in range(NREP):     # 1 clean anchor + NREP-1 perturbed passes
            if rep == 0:
                p2, nnrm = pts, nrm
            else:
                d = perturb_field(len(pts), rng)
                p2 = pts + d[:, None] * nrm
                nnrm = s58.normals_of(p2, gt)
            p, s, t = extract(p2, nnrm, z, g, gt)
            P.append(p); S.append(s); T.append(t)
        if verbose:
            print("[refiner] %s ok" % nm)
    if not P:
        raise SystemExit("no data")
    return (np.concatenate(P), np.concatenate(S), np.concatenate(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max", type=int, default=0, help="use only N train images")
    ap.add_argument("--out", default=os.path.join(STEP4, "results", "normal_refiner.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", type=int, default=11,
                    help="images held out of the refiner fit set for diagnostics")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val = s57.read_split(USS, exp="ultrasound_prostate_73")
    if args.max:
        train = train[:args.max]
    # hold out 11 training images to check the refiner's own generalisation
    if args.hold > 0 and len(train) > args.hold + 8:
        fit_names, hold_names = train[:-args.hold], train[-args.hold:]
    else:
        fit_names, hold_names = train, []

    t0 = time.time()
    print("[refiner] building fit set from %d images ..." % len(fit_names))
    Pf, Sf, Tf = build(fit_names, rng)
    print("[refiner] fit  X=%s scal=%s y=%s  (%.1f s)"
          % (Pf.shape, Sf.shape, Tf.shape, time.time() - t0))
    Ph = Sh = Th = None
    if hold_names:
        Ph, Sh, Th = build(hold_names, rng, verbose=False)
        print("[refiner] hold X=%s" % (Ph.shape,))

    dev = "cpu"
    net = Refiner().to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    print("[refiner] params = %d, device = %s" % (nparam, dev))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.SmoothL1Loss(beta=2.0)

    Xf = torch.from_numpy(Pf); SfT = torch.from_numpy(Sf); Yf = torch.from_numpy(Tf)
    n = len(Xf)
    hist = []
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            opt.zero_grad()
            out = net(Xf[idx], SfT[idx])
            loss = lossf(out, Yf[idx])
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()
        msg = "[refiner] ep %3d/%d  fit_loss %.4f" % (ep + 1, args.epochs, tot / max(nb, 1))
        if Ph is not None:
            net.eval()
            with torch.no_grad():
                o = net(torch.from_numpy(Ph), torch.from_numpy(Sh))
                hl = float(lossf(o, torch.from_numpy(Th)).item())
                mae = float((o - torch.from_numpy(Th)).abs().mean().item())
                base_mae = float(torch.from_numpy(Th).abs().mean().item())
            msg += "  hold_loss %.4f  hold_MAE %.3f px (predict-0 MAE %.3f)" % (hl, mae, base_mae)
        print(msg, flush=True)
        hist.append(msg)

    torch.save({"state": net.state_dict(), "nparam": nparam,
                "VH": VH, "VW": VW, "NSCALAR": NSCALAR,
                "history": hist}, args.out)
    print("[refiner] saved %s" % args.out)


if __name__ == "__main__":
    main()
