#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step12_lowrank_decomp/train_refiner.py

Pilot: low-rank boundary-decomposition refiner (LRDR) on top of the delivered
SQ2res-w25 big model (fold-0, 71/31 split).

Design (the "decomposition" part):
    shared head      : global anatomy logits from the full-res decoder
    low-rank code    : 1x1 conv to k channels at 1/2 resolution (k=8)
    boundary head    : 1x1 conv k->1, zero-init, added to the shared logits
    aux boundary head: 1x1 conv k->1, supervised by a soft boundary band
                       (this forces the low-rank code to carry boundary evidence)

Training input : image + perturbed big-model train mask + SDF
Training target: GT union (mask > 0)
Perturbations  : smooth elastic field + dilate/erode + local blobs, calibrated to
                 the observed val boundary error (|err| mean ~6.5 px, p90 ~15 px)

Eval input     : image + big-model val mask (raw / free / veto / TTA) + SDF
Eval target    : GT union at the ORIGINAL resolution.

This is deliberately a *pilot*: it tests whether a learned low-rank boundary
decomposition can beat the hand-written arbitration on the same big model.
The teacher (UltraSAM3) low-rank branch is the next ablation; the hook is the
`--teacher` flag (not implemented yet) so the code stays honest.
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
USS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import all_metrics, _surface  # noqa: E402


GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
PRED = os.path.join(USS, "step6_sandwich_squeeze", "predictions")
TRAIN_PRED = os.path.join(PRED, "sq2resencw25_checkpoint_best_arb_train", "raw")
VAL_PRED = os.path.join(PRED, "sq2resencw25_checkpoint_best_arb_val", "raw")
VAL_FREE = os.path.join(PRED, "sq2resencw25_checkpoint_best_arb_val", "free")
VAL_VETO = os.path.join(PRED, "sq2resencw25_checkpoint_best_arb_val", "veto_lrn")
VAL_TTA = os.path.join(PRED, "final_tta_art_x_ens4_png")

CANVAS = 1024
EPS = 1e-7


# ------------------------------------------------------------------ canvas io
def _canvas(arr, is_mask=False):
    """(H,W) -> (CANVAS,CANVAS), image at top-left, returns (canvas, (nh,nw))."""
    H, W = arr.shape
    nw = CANVAS
    nh = min(CANVAS, max(1, int(round(H * nw / float(W)))))
    im = Image.fromarray(arr.astype(np.uint8) if is_mask else arr)
    im = im.resize((nw, nh), Image.NEAREST if is_mask else Image.BILINEAR)
    a = np.asarray(im)
    if is_mask:
        out = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    else:
        out = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    out[:nh, :nw] = a
    return out, (nh, nw)


def load_case(name, pred_dir=None, tta=False):
    raw = np.asarray(Image.open(os.path.join(IMG_DIR, name)).convert("L"), dtype=np.float32) / 255.0
    gt = np.asarray(Image.open(os.path.join(GT_DIR, name)).convert("L")) > 0
    if pred_dir is not None:
        pm = np.asarray(Image.open(os.path.join(pred_dir, name)).convert("L")) > 0
    else:
        pm = None
    return raw, gt, pm


def make_sdf(mask):
    out = ndimage.distance_transform_edt(~mask)
    inn = ndimage.distance_transform_edt(mask)
    return np.clip((out - inn) / 20.0, -2.0, 2.0).astype(np.float32)


def soft_band(sdf):
    """Boundary-band target in [0,1], peak at the zero level set."""
    return np.exp(-np.abs(sdf) * 2.0).astype(np.float32)


# -------------------------------------------------------------- perturbations
def elastic_perturb(mask, rng, amp_max=12.0):
    """Smooth random displacement field, calibrated to the val error scale."""
    gh, gw = 8, 8
    dx = rng.normal(size=(gh, gw)).astype(np.float32)
    dy = rng.normal(size=(gh, gw)).astype(np.float32)
    dx = ndimage.zoom(dx, (CANVAS / gh, CANVAS / gw), order=3)[:CANVAS, :CANVAS]
    dy = ndimage.zoom(dy, (CANVAS / gh, CANVAS / gw), order=3)[:CANVAS, :CANVAS]
    amp = rng.uniform(1.5, amp_max)
    dx = dx / (np.abs(dx).max() + 1e-6) * amp
    dy = dy / (np.abs(dy).max() + 1e-6) * amp
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32)
    coords = np.stack([yy + dy, xx + dx], axis=0)
    w = ndimage.map_coordinates(mask.astype(np.float32), coords, order=1, mode="nearest")
    return w > 0.5


def perturb_mask(mask, rng):
    """Identity 12%, else elastic + dilate/erode + occasional blob."""
    if rng.random() < 0.12:
        return mask.copy()
    m = mask
    if rng.random() < 0.85:
        m = elastic_perturb(m, rng, amp_max=rng.uniform(5.0, 18.0))
    if rng.random() < 0.65:
        r = rng.integers(1, 9)
        m = ndimage.binary_dilation(m, iterations=int(r)) if rng.random() < 0.5 \
            else ndimage.binary_erosion(m, iterations=int(r))
    if rng.random() < 0.25:
        # local blob near the boundary: add or remove a disk
        b = _surface(m)
        ys, xs = np.where(b)
        if len(ys):
            i = rng.integers(0, len(ys))
            cy, cx = int(ys[i]), int(xs[i])
            rr = int(rng.integers(5, 16))
            yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
            disk = (yy * yy + xx * xx) <= rr * rr
            y0, y1 = max(0, cy - rr), min(CANVAS, cy + rr + 1)
            x0, x1 = max(0, cx - rr), min(CANVAS, cx + rr + 1)
            d = disk[(y0 - (cy - rr)):(y0 - (cy - rr)) + (y1 - y0),
                     (x0 - (cx - rr)):(x0 - (cx - rr)) + (x1 - x0)]
            if rng.random() < 0.5:
                m[y0:y1, x0:x1] |= d
            else:
                m[y0:y1, x0:x1] &= ~d
    return m


# ---------------------------------------------------------------------- model
class Conv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class LRDR(nn.Module):
    """Low-rank boundary-decomposition refiner."""

    def __init__(self, base=24, k=8):
        super().__init__()
        self.k = k
        self.e1 = Conv(4, base)
        self.e2 = Conv(base, base * 2)
        self.e3 = Conv(base * 2, base * 4)
        self.e4 = Conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = Conv(base * 8 + base * 4, base * 4)
        self.u2 = Conv(base * 4 + base * 2, base * 2)
        self.u1 = Conv(base * 2 + base, base)
        self.shared = nn.Conv2d(base, 1, 1)
        self.code = nn.Conv2d(base * 2, k, 1)          # 1/2 resolution
        self.boundary = nn.Conv2d(k, 1, 1)             # k -> residual logits
        self.aux = nn.Conv2d(k, 1, 1)                  # boundary-band auxiliary
        nn.init.normal_(self.shared.weight, std=0.01)
        nn.init.zeros_(self.shared.bias)
        nn.init.normal_(self.boundary.weight, std=0.01)
        nn.init.zeros_(self.boundary.bias)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        e1 = self.e1(x)                               # full
        e2 = self.e2(self.pool(e1))                   # 1/2
        e3 = self.e3(self.pool(e2))                   # 1/4
        e4 = self.e4(self.pool(e3))                   # 1/8
        d3 = self.u3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.u2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.u1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        shared = self.shared(d1)
        code = self.code(d2)                          # (B,k,H/2,W/2)
        code_up = F.interpolate(code, size=x.shape[-2:], mode="bilinear", align_corners=False)
        bnd = self.boundary(code_up)
        aux = self.aux(code_up)
        delta = shared + torch.tanh(self.alpha) * bnd
        return delta, shared, code, aux


# --------------------------------------------------------------------- dataset
class RefinerDS(Dataset):
    def __init__(self, names, train=True, fixed_seed=None):
        self.names = names
        self.train = train
        self.fixed_seed = fixed_seed
        self.items = []
        for n in names:
            raw, gt, pm = load_case(n, TRAIN_PRED if train else VAL_PRED)
            img_c, hw = _canvas(raw)
            gt_c, _ = _canvas(gt.astype(np.uint8), True)
            pm_c, _ = _canvas(pm.astype(np.uint8), True)
            gt_c = gt_c > 0
            pm_c = pm_c > 0
            sdf = make_sdf(pm_c)
            gt_sdf = make_sdf(gt_c)
            gt_band = soft_band(gt_sdf)
            # GT boundary distance weight (for the boundary-focused BCE)
            gb = _surface(gt_c)
            dist = ndimage.distance_transform_edt(~gb).astype(np.float32)
            wgt = (1.0 + 5.0 * np.exp(-dist / 3.0)).astype(np.float32)
            valid = np.zeros((CANVAS, CANVAS), dtype=np.float32)
            nh, nw = hw
            valid[:nh, :nw] = 1.0
            self.items.append(dict(name=n, img=img_c.astype(np.float32), gt=gt_c.astype(np.float32),
                                   pm=pm_c.astype(np.float32), sdf=sdf, gt_sdf=gt_sdf,
                                   gt_band=gt_band, wgt=wgt, valid=valid, hw=hw))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        seed = (self.fixed_seed + i) if self.fixed_seed is not None else random.randint(0, 2 ** 31 - 1)
        rng = np.random.default_rng(seed)
        if self.train:
            m = perturb_mask(it["pm"] > 0.5, rng)
            sdf = make_sdf(m)
            flip = rng.random() < 0.5
        else:
            m = it["pm"] > 0.5
            sdf = it["sdf"]
            flip = False
        img = it["img"]
        gt = it["gt"]
        gt_sdf = it["gt_sdf"]
        gt_band = it["gt_band"]
        wgt = it["wgt"]
        valid = it["valid"]
        dsdf = gt_sdf - sdf
        if flip:
            img = img[:, ::-1].copy(); m = m[:, ::-1].copy(); sdf = sdf[:, ::-1].copy()
            gt = gt[:, ::-1].copy(); gt_band = gt_band[:, ::-1].copy(); dsdf = dsdf[:, ::-1].copy()
            wgt = wgt[:, ::-1].copy(); valid = valid[:, ::-1].copy()
        # mild intensity jitter (not geometry)
        if self.train:
            img = np.clip(img * rng.uniform(0.9, 1.1) + rng.uniform(-0.03, 0.03), 0, 1)
        x = np.stack([img, m.astype(np.float32), sdf, soft_band(sdf)], 0).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(gt[None].astype(np.float32)),
                torch.from_numpy(gt_band[None].astype(np.float32)),
                torch.from_numpy(dsdf[None].astype(np.float32)),
                torch.from_numpy(wgt[None].astype(np.float32)),
                torch.from_numpy(valid[None].astype(np.float32)), it["name"])


# ---------------------------------------------------------------------- losses
def soft_dice(logits, target, valid=None, eps=1e-6):
    p = torch.sigmoid(logits)
    if valid is not None:
        p = p * valid
        target = target * valid
    tp = (p * target).sum((2, 3))
    return (1.0 - (2 * tp + eps) / (p.sum((2, 3)) + target.sum((2, 3)) + eps)).mean()


def bce_valid(logits, target, valid, weight=None):
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        l = l * weight
    l = l * valid
    return l.sum() / (valid.sum() + 1e-6)


# ----------------------------------------------------------------------- train
def run_epoch(model, loader, opt, scaler, dev, train=True):
    model.train(train)
    tot = 0.0
    n = 0
    for x, gt, gtb, dsdf, wgt, valid, _ in loader:
        x, gt, gtb, dsdf, wgt, valid = (x.to(dev), gt.to(dev), gtb.to(dev),
                                        dsdf.to(dev), wgt.to(dev), valid.to(dev))
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                delta, shared, code, aux = model(x)
                pred_sdf = x[:, 2:3] + delta
                pred_logits = -pred_sdf * 5.0
                gt_sdf = dsdf + x[:, 2:3]
                band_mask = (torch.abs(gt_sdf) < 0.5).float()
                denom = (band_mask * valid).sum() + 1e-6
                l_sdf = (F.smooth_l1_loss(delta, dsdf, reduction="none", beta=0.1)
                         * band_mask * valid).sum() / denom
                l_dice = soft_dice(pred_logits, gt, valid)
                l_aux = bce_valid(aux, gtb, valid)
                l_code = 1e-4 * code.pow(2).mean()
                loss = 10.0 * l_sdf + l_dice + 0.1 * l_aux + l_code
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        tot += float(loss.detach()) * x.shape[0]
        n += x.shape[0]
    return tot / max(1, n)


@torch.no_grad()
def predict_case(model, name, pred_dir, dev, tta=False):
    raw, gt, pm = load_case(name, pred_dir)
    img_c, (nh, nw) = _canvas(raw)
    pm_c, _ = _canvas(pm.astype(np.uint8), True)
    pm_c = pm_c > 0
    sdf = make_sdf(pm_c)
    band = soft_band(sdf)
    x = np.stack([img_c, pm_c.astype(np.float32), sdf, band], 0)[None].astype(np.float32)
    delta, _, _, _ = model(torch.from_numpy(x).to(dev))
    pred_sdf = sdf + delta[0, 0].float().cpu().numpy()
    pred_c = pred_sdf[:nh, :nw] <= 0
    # back to original resolution
    pred = np.asarray(Image.fromarray((pred_c * 255).astype(np.uint8)).resize(
        (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
    return pred, gt, pred_sdf[:nh, :nw]


def bdice(pred, gt, tol=1):
    gb, pb = _surface(gt), _surface(pred)
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    gg = ndimage.binary_dilation(gb, ndimage.generate_binary_structure(2, 1), iterations=tol)
    pp = ndimage.binary_dilation(pb, ndimage.generate_binary_structure(2, 1), iterations=tol)
    prec = float(np.logical_and(pb, gg).sum()) / (pb.sum() + EPS)
    rec = float(np.logical_and(gb, pp).sum()) / (gb.sum() + EPS)
    return 2 * prec * rec / (prec + rec + EPS) * 100.0


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "refiner_fold0"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pred-dir", default=VAL_PRED, help="val prediction dir for eval")
    ap.add_argument("--tag", default="refiner")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    train_names = sorted(os.listdir(TRAIN_PRED))
    val_names = sorted(os.listdir(VAL_PRED))
    if args.smoke:
        train_names = train_names[:6]
        val_names = val_names[:4]
    # selection split: every 7th train case held out for checkpoint selection
    sel_names = train_names[::7]
    fit_names = [n for n in train_names if n not in set(sel_names)]
    print("[lrdr] train %d  sel %d  val %d" % (len(fit_names), len(sel_names), len(val_names)), flush=True)

    ds_fit = RefinerDS(fit_names, train=True)
    ds_sel = RefinerDS(sel_names, train=True, fixed_seed=12345)
    dl_fit = DataLoader(ds_fit, batch_size=args.batch, shuffle=True, num_workers=3,
                        pin_memory=True, drop_last=False)
    dl_sel = DataLoader(ds_sel, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = LRDR(base=args.base, k=args.k).to(dev)
    print("[lrdr] params %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    if not args.eval_only:
        best = (-1e9, -1)
        for ep in range(args.epochs):
            t0 = time.time()
            tr = run_epoch(model, dl_fit, opt, scaler, dev, train=True)
            sched.step()
            msg = "ep %3d/%d loss %.4f lr %.2e %.1f min" % (ep + 1, args.epochs, tr,
                                                             opt.param_groups[0]["lr"], (time.time() - t0) / 60)
            if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
                # selection: synthetic perturbations on held-out train cases
                model.eval()
                dices, bds = [], []
                for x, gt, gtb, dsdf, wgt, valid, _ in dl_sel:
                    x, gt, valid = x.to(dev), gt.to(dev), valid.to(dev)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        delta, _, _, _ = model(x)
                    p = ((x[:, 2:3] + delta) <= 0) & (valid > 0.5)
                    g = (gt > 0.5) & (valid > 0.5)
                    for i in range(p.shape[0]):
                        pi, gi = p[i, 0].cpu().numpy(), g[i, 0].cpu().numpy()
                        dices.append(2 * (pi & gi).sum() / max(1, pi.sum() + gi.sum()) * 100)
                        bds.append(bdice(pi, gi))
                sc = float(np.mean(bds))
                msg += " | sel dice %.2f bdice %.2f" % (np.mean(dices), np.mean(bds))
                if sc > best[0]:
                    best = (sc, ep)
                    torch.save({"net": model.state_dict(), "args": vars(args), "epoch": ep},
                               os.path.join(args.out, "ckpt_best.pt"))
                model.train()
            print(msg, flush=True)
            if ep % 20 == 0 or ep == args.epochs - 1:
                torch.save({"net": model.state_dict(), "args": vars(args), "epoch": ep},
                           os.path.join(args.out, "ckpt.pt"))

    # ------------------------------------------------------------- evaluation
    ck = args.ckpt or os.path.join(args.out, "ckpt_best.pt")
    if not os.path.exists(ck):
        ck = os.path.join(args.out, "ckpt.pt")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["net"])
    model.eval()
    print("[lrdr] eval ckpt %s (ep %s)" % (ck, blob.get("epoch")), flush=True)
    rows = []
    for n in val_names:
        pred, gt, _ = predict_case(model, n, args.pred_dir, dev)
        pm = np.asarray(Image.open(os.path.join(args.pred_dir, n)).convert("L")) > 0
        mr = all_metrics(pred, gt)
        mi = all_metrics(pm, gt)
        rows.append(dict(name=n, dice_in=mi["dice"], dice_out=mr["dice"],
                         bd_in=bdice(pm, gt), bd_out=bdice(pred, gt),
                         hd95_in=mi["hd95"], hd95_out=mr["hd95"],
                         assd_in=mi["assd"], assd_out=mr["assd"],
                         vd_in=mi["vol_diff_pct"], vd_out=mr["vol_diff_pct"]))
        print("  %-28s dice %.3f -> %.3f | bd1 %.2f -> %.2f | hd95 %.2f -> %.2f" %
              (n, mi["dice"], mr["dice"], rows[-1]["bd_in"], rows[-1]["bd_out"],
               mi["hd95"], mr["hd95"]), flush=True)
    csv_path = os.path.join(args.out, "eval_%s.csv" % args.tag)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    from scipy import stats as st
    summary = {}
    for key in ["dice", "bd", "hd95", "assd", "vd"]:
        a = np.array([r[key + "_in"] for r in rows], float)
        b = np.array([r[key + "_out"] for r in rows], float)
        t, p = st.ttest_rel(b, a)
        try:
            _, pw = st.wilcoxon(b, a)
        except Exception:
            pw = float("nan")
        summary[key] = dict(in_mean=float(a.mean()), out_mean=float(b.mean()),
                            delta=float(b.mean() - a.mean()), p_t=float(p), p_w=float(pw),
                            wins=int((b > a).sum()) if key != "hd95" and key != "assd" and key != "vd" else int((b < a).sum()))
    with open(os.path.join(args.out, "eval_%s.json" % args.tag), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
