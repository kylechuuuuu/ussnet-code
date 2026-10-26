#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 64：阶段二精修器 —— 上下文来自 stage-1 的概率场，而不是图像。

§5.6 证明了"只看图像"的精修器是负收益（nnU-Net 边界 Dice@1px 35.28 -> 32.46）。
原因是 §5.5 量到的：预测边界处的 |∇I| 已经是 GT 边界处的 1.03 倍，
模型本来就坐在图像强脊上，局部斑点读不出"该往哪边挪多少"。

本实验把输入换成 **stage-1 自己的看法**：
  ch0 = stage-1 前景概率 p
  ch1 = 到"当前轮廓"的有符号距离（截断到 +-MAXD 并归一化）
  ch2 = 图像 z-score
  ch3 = |∇I| / std
（`--inp img` 只用 ch2/ch3，用来复现 §5.6 的负结果做对照。）

**这是一个不需要 GPU、也不需要训练 stage-1 的合法留出实验**：
fold-0 的 31 张验证图对两个基线（unet_73 / nnunet_2d_fold0_checkpoint_best）都是干净的，
在其中切 20 训 / 11 测 → 在 20 张上训精修器、在 11 张上测。

用法：
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python tools/64_stage2_refiner_context.py --inp ctx
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python tools/64_stage2_refiner_context.py --inp img
"""

import argparse
import csv
import importlib.util
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS = os.path.dirname(STEP4)


def _load(mod, fn):
    p = os.path.join(HERE, fn)
    spec = importlib.util.spec_from_file_location(mod, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


s57 = _load("s57c", "57_shape_prior_ceiling.py")
s58 = _load("s58c", "58_normal_refine_oracle.py")
s60 = _load("s60c", "60_train_normal_refiner.py")

NPT = s57.NPT
MAXD = s60.MAXD
VH, VW = s60.VH, s60.VW
NV, NU = s60.NV, s60.NU
NSCALAR = 8
NIN = {"ctx": 4, "img": 2}

S1 = os.path.join(USS, "step1_nnunet_baseline")
S1_NPZ = os.path.join(S1, "predictions", "nnunet_2d_fold0_checkpoint_best_nifti")
S1_PNG = os.path.join(S1, "predictions", "nnunet_2d_fold0_checkpoint_best_png")
MAP_CSV = os.path.join(S1, "case_id_mapping.csv")


# ------------------------------------------------------------------ loading --
def name2cid():
    out = {}
    with open(MAP_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["original_filename"]] = row["case_id"]
    return out


def load_stage1(name, n2c):
    cid = n2c.get(name)
    if cid is None:
        return None, None
    npz = os.path.join(S1_NPZ, cid + ".npz")
    png = os.path.join(S1_PNG, name)
    if not (os.path.exists(npz) and os.path.exists(png)):
        return None, None
    prob = np.load(npz)["probabilities"][1, 0].astype(np.float32)
    m = np.array(Image.open(png).convert("L")) > 127
    return prob, m


def pscal_of(prob, m0):
    """(mean p well inside) - (mean p just outside) -- per-image context scalar."""
    inn = ndimage.binary_erosion(m0, iterations=4)
    out = s57.dilate(m0, 6) & ~m0
    pi = float(prob[inn].mean()) if inn.any() else 0.5
    po = float(prob[out].mean()) if out.any() else 0.5
    return np.clip(pi - po, -1.0, 1.0)


# ----------------------------------------------------------------- features --
def extract2(pts, nrm, sdf, prob, z, g, pscal, inp, gt=None):
    """-> (patches (N,nin,NV,NU), scalars (N,8), target (N,) or None)."""
    n = len(pts)
    tan = np.stack([nrm[:, 1], -nrm[:, 0]], axis=1)
    vs = np.arange(-VH, VH + 1, dtype=float)
    us = np.arange(-VW, VW + 1, dtype=float)
    V, U = np.meshgrid(vs, us, indexing="ij")
    q = (pts[:, None, None, :]
         + V[None, :, :, None] * nrm[:, None, None, :]
         + U[None, :, :, None] * tan[:, None, None, :])
    rr = q[..., 0].reshape(n, -1)
    cc = q[..., 1].reshape(n, -1)

    def samp(f):
        return ndimage.map_coordinates(f, [rr.ravel(), cc.ravel()], order=1,
                                       mode="nearest").reshape(n, NV, NU)

    pz = samp(z)
    pg = samp(g)
    if inp == "ctx":
        pp = samp(prob)
        pd = np.clip(samp(sdf), -MAXD, MAXD) / MAXD
        patches = np.stack([pp, pd, pz, pg], axis=1).astype(np.float32)
    else:
        patches = np.stack([pz, pg], axis=1).astype(np.float32)

    d1 = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    d2 = np.roll(pts, -2, axis=0) - 2 * pts + np.roll(pts, 2, axis=0)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    kappa = np.abs(cross) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-6)
    cen = pts.mean(axis=0)
    R = float(np.sqrt(((pts - cen) ** 2).sum(axis=1).mean()) + 1e-6)
    rnorm = np.linalg.norm(pts - cen, axis=1) / R
    arc = np.arange(n) / float(n) * 2.0 - 1.0
    scal = np.stack([np.clip(kappa * 40.0, 0, 5), rnorm, arc,
                     s60.bil(z, pts), s60.bil(g, pts),
                     np.full(n, R / 280.0),
                     np.full(n, pscal),
                     np.clip(s60.bil(sdf, pts), -MAXD, MAXD) / MAXD],
                    axis=1).astype(np.float32)
    tgt = None
    if gt is not None:
        tgt, _ = s60.make_targets(gt, pts)
        tgt = tgt.astype(np.float32)
    return patches, scal, tgt


# -------------------------------------------------------------------- model --
class Refiner2(nn.Module):
    def __init__(self, nin, nscalar=NSCALAR):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))

        self.c1 = blk(nin, 16)
        self.c2 = blk(16, 24)
        self.c3 = blk(24, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sf = nn.Sequential(nn.Linear(nscalar, 16), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(48, 48), nn.ReLU(inplace=True),
                                  nn.Linear(48, 1))

    def forward(self, x, s):
        h = self.pool(self.c3(self.c2(self.c1(x)))).flatten(1)
        h = torch.cat([h, self.sf(s)], dim=1)
        return torch.tanh(self.head(h)).squeeze(1) * MAXD


# ------------------------------------------------------------------ pipeline --
def start_contour(prob, m0):
    c = s57.contour_of(s57.largest_cc(m0))
    if c is None:
        return None, None
    pts = s57.resample_closed(c, NPT, 0)
    return pts, s58.normals_of(pts, s57.largest_cc(m0))


def build(names, prob_of, img_of, pscal_of_map, inp, rng, nrep=4, verbose=True):
    P, S, T = [], [], []
    for nm in names:
        gt = s57.load_gt(nm)
        img = img_of(nm)
        prob, m0 = prob_of(nm)
        if gt is None or img is None or prob is None:
            continue
        z, g = s60.prep_image(img)
        pts, nrm = start_contour(prob, m0)
        if pts is None:
            continue
        geo0 = s57.largest_cc(m0)
        ps = pscal_of_map(nm)
        for rep in range(nrep):
            if rep == 0:
                p2, geo = pts, geo0
            else:
                d = s60.perturb_field(len(pts), rng)
                p2 = pts + d[:, None] * nrm
                geo = s57.rasterize(p2, geo0.shape)
            nrm2 = s58.normals_of(p2, geo)
            sdf = s58.sdf_of(geo)
            pa, sc, tg = extract2(p2, nrm2, sdf, prob, z, g, ps, inp, gt)
            if tg is None:
                continue
            P.append(pa)
            S.append(sc)
            T.append(tg)
        if verbose:
            print("[64] %s ok" % nm)
    if not P:
        raise SystemExit("no training data")
    return np.concatenate(P), np.concatenate(S), np.concatenate(T)


def apply_refiner(model, prob, z, g, m0, ps, inp, iters, smooth=2.0):
    m = s57.largest_cc(m0)
    a0 = float(m.sum())
    for _ in range(iters):
        pts, nrm = start_contour(prob, m)
        if pts is None:
            return m
        sdf = s58.sdf_of(m)
        pa, sc, _ = extract2(pts, nrm, sdf, prob, z, g, ps, inp, None)
        with torch.no_grad():
            d = model(torch.from_numpy(pa), torch.from_numpy(sc)).numpy()
        if smooth > 0:
            d = ndimage.gaussian_filter1d(d, smooth, mode="wrap")
        m2 = s57.rasterize(pts + d[:, None] * nrm, m.shape)
        a1 = float(m2.sum())
        if a0 > 0 and 0.5 * a0 <= a1 <= 2.0 * a0:
            m = m2
    return m


def oracle_mask(gt, m0):
    m = s57.largest_cc(m0)
    c = s57.contour_of(m)
    if c is None:
        return m
    pts = s57.resample_closed(c, NPT, 0)
    nrm = s58.normals_of(pts, m)
    sdf = s58.sdf_of(gt)
    d = np.clip(-s60.bil(sdf, pts), -MAXD, MAXD)
    return s57.rasterize(pts + d[:, None] * nrm, m.shape)


def score(m, gt):
    # s57.dice / s57.bdice already return percent (0-100)
    return (s57.dice(m, gt), s57.bdice(m, gt, 1),
            s57.bdice(m, gt, 2), float(s57.surface(m).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", choices=["ctx", "img"], default="ctx")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--ntrain", type=int, default=20)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    n2c = name2cid()
    _, val_names = s57.read_split(USS, exp="ultrasound_prostate_73")
    rng_split = np.random.default_rng(a.split_seed)
    order = list(val_names)
    rng_split.shuffle(order)
    tr_names = order[:a.ntrain]
    te_names = order[a.ntrain:]
    print("[64] inp=%s  split-seed=%d  train %d / test %d" %
          (a.inp, a.split_seed, len(tr_names), len(te_names)))

    img_cache, prob_cache, ps_cache = {}, {}, {}

    def img_of(nm):
        if nm not in img_cache:
            img_cache[nm] = s60.load_image(nm)
        return img_cache[nm]

    def prob_of(nm):
        if nm not in prob_cache:
            prob_cache[nm] = load_stage1(nm, n2c)
        return prob_cache[nm]

    def ps_of(nm):
        if nm not in ps_cache:
            pr, m0 = prob_of(nm)
            ps_cache[nm] = pscal_of(pr, s57.largest_cc(m0))
        return ps_cache[nm]

    P, S, T = build(tr_names, prob_of, img_of, ps_of, a.inp, rng)
    print("[64] X=%s scal=%s  build %.1fs" % (P.shape, S.shape, time.time() - t0))

    model = Refiner2(NIN[a.inp])
    nparam = sum(p.numel() for p in model.parameters())
    print("[64] params = %d" % nparam)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lf = nn.SmoothL1Loss(beta=2.0)
    X = torch.from_numpy(P)
    Ss = torch.from_numpy(S)
    Y = torch.from_numpy(T)
    n = len(Y)
    for ep in range(1, a.epochs + 1):
        perm = torch.from_numpy(rng.permutation(n))
        tot, nb = 0.0, 0
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            opt.zero_grad()
            out = model(X[idx], Ss[idx])
            loss = lf(out, Y[idx])
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        if ep == 1 or ep % 10 == 0 or ep == a.epochs:
            print("[64] ep %3d/%d  fit_loss %.4f  (%.0fs)" %
                  (ep, a.epochs, tot / max(nb, 1), time.time() - t0))

    # ------------------------------------------------------------- evaluate --
    keys = ["raw", "cc1", "ctrl"] + ["r%d" % i for i in range(1, a.iters + 1)] + ["oracle1"]

    def eval_set(names, label):
        rows = []
        for nm in names:
            gt = s57.load_gt(nm)
            img = img_of(nm)
            prob, m0 = prob_of(nm)
            if gt is None or prob is None:
                continue
            z, g = s60.prep_image(img)
            ps = ps_of(nm)
            rec = {"name": nm}
            rec["raw"] = score(m0, gt)
            rec["cc1"] = score(s57.largest_cc(m0), gt)
            cpts, _ = start_contour(prob, s57.largest_cc(m0))
            ctrl = s57.rasterize(cpts, gt.shape) if cpts is not None else s57.largest_cc(m0)
            rec["ctrl"] = score(ctrl, gt)
            m = ctrl
            for it in range(1, a.iters + 1):
                m = apply_refiner(model, prob, z, g, m, ps, a.inp, 1, a.smooth)
                rec["r%d" % it] = score(m, gt)
            rec["oracle1"] = score(oracle_mask(gt, m0), gt)
            rows.append(rec)
        print("\n== %s | %s (n=%d) ==" % (a.inp, label, len(rows)))
        print("%-10s %8s %8s %8s %10s" % ("variant", "Dice", "bd1px", "bd2px", "bnd_len"))
        summ = {}
        for k in keys:
            v = np.array([r[k] for r in rows], dtype=float)
            summ[k] = v.mean(axis=0).tolist()
            print("%-10s %8.2f %8.2f %8.2f %10.1f" %
                  (k, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean()))
        return rows, summ

    tr_rows, tr_summ = eval_set(tr_names, "TRAIN (seen)")
    rows, summ = eval_set(te_names, "HELD-OUT")

    print("\nper-image Dice (held-out):")
    for r in rows:
        print("  %-28s raw %6.2f  ctrl %6.2f  %s  oracle %6.2f" %
              (r["name"], r["raw"][0], r["ctrl"][0],
               "  ".join("r%d %6.2f" % (i, r["r%d" % i][0]) for i in range(1, a.iters + 1)),
               r["oracle1"][0]))

    if a.out:
        import json
        with open(a.out, "w") as fh:
            json.dump({"inp": a.inp, "nparam": nparam, "ntrain": len(tr_names),
                       "ntest": len(rows), "summary": summ,
                       "per_image": [{k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in r.items()} for r in rows]}, fh, indent=1)
        print("[64] wrote %s" % a.out)
    print("[64] total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
