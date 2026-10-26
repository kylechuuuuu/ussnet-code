#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step12_lowrank_decomp/train_deform.py

Low-rank boundary-deformation refiner (LRDR-v2).

Why v2: v1 (dense SDF correction) could not beat the input mask -- the output
space was too free and the model learned a global bias.  v2 predicts a *smooth,
low-rank displacement field* and warps the input mask:

    image + mask + SDF -> encoder -> k-dim code (k=8) -> (dx,dy) at 1/4 res
    -> upsample -> grid_sample(mask order=1) -> refined mask

The low-rank code is the "decomposition": a k-dim basis of coherent boundary
deformations (the error mode observed in step9 is a coherent arc, not pixel
noise).  The displacement field is smooth by construction, so it cannot fit
high-frequency noise; identity is zero displacement.

Train : big-model train masks (near-perfect) + synthetic smooth deformations
Eval  : big-model val masks (raw / free / veto / TTA), GT at original resolution.
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
sys.path.insert(0, HERE)
import train_refiner as R  # reuse canvas/io/metrics/perturbations

from eval_lib import all_metrics  # noqa: E402

CANVAS = R.CANVAS
TRAIN_PRED, VAL_PRED = R.TRAIN_PRED, R.VAL_PRED


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


class DeformNet(nn.Module):
    def __init__(self, base=32, k=8, max_disp=25.0):
        super().__init__()
        self.k = k
        self.max_disp = max_disp
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, 2, 1, bias=False), nn.GroupNorm(8, base), nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, 2, 1, bias=False), nn.GroupNorm(8, base), nn.SiLU(inplace=True),
        )  # 1/4
        self.e2 = Conv(base, base * 2)       # 1/8
        self.e3 = Conv(base * 2, base * 4)   # 1/16
        self.u2 = Conv(base * 4 + base * 2, base * 2)
        self.u1 = Conv(base * 2 + base, base)
        self.code = nn.Conv2d(base, k, 1)    # low-rank deformation basis
        self.disp = nn.Conv2d(k, 2, 1)       # k -> (dx,dy)
        nn.init.normal_(self.disp.weight, std=0.01)
        nn.init.zeros_(self.disp.bias)
        # start with a small identity-ish deformation
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        h = self.stem(x)                      # 1/4
        e2 = self.e2(F.avg_pool2d(h, 2))      # 1/8
        e3 = self.e3(F.avg_pool2d(e2, 2))     # 1/16
        d2 = self.u2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.u1(torch.cat([F.interpolate(d2, size=h.shape[-2:], mode="bilinear", align_corners=False), h], 1))
        code = self.code(d1)                  # (B,k,1/4,1/4)
        disp = torch.tanh(self.disp(code)) * self.max_disp * torch.tanh(self.scale)
        return disp, code


def warp(mask, disp):
    """mask (B,1,H,W) in [0,1], disp (B,2,H/4,W/4) in pixels -> warped mask."""
    B, _, H, W = mask.shape
    du = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=True)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=mask.device),
                            torch.linspace(-1, 1, W, device=mask.device), indexing="ij")
    gx = xx + du[:, 0] / max(1, W - 1) * 2.0
    gy = yy + du[:, 1] / max(1, H - 1) * 2.0
    grid = torch.stack([gx, gy], -1)
    return F.grid_sample(mask, grid, mode="bilinear", padding_mode="border", align_corners=True)


class DeformDS(Dataset):
    def __init__(self, names, train=True, fixed_seed=None, pred_dir=None):
        self.names = names
        self.train = train
        self.fixed_seed = fixed_seed
        self.items = []
        pd = pred_dir or (TRAIN_PRED if train else VAL_PRED)
        for n in names:
            raw, gt, pm = R.load_case(n, pd)
            img_c, hw = R._canvas(raw)
            gt_c, _ = R._canvas(gt.astype(np.uint8), True)
            pm_c, _ = R._canvas(pm.astype(np.uint8), True)
            gt_c = gt_c > 0
            pm_c = pm_c > 0
            valid = np.zeros((CANVAS, CANVAS), dtype=np.float32)
            nh, nw = hw
            valid[:nh, :nw] = 1.0
            self.items.append(dict(name=n, img=img_c.astype(np.float32), gt=gt_c.astype(np.float32),
                                   pm=pm_c.astype(np.float32), valid=valid, hw=hw))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        seed = (self.fixed_seed + i) if self.fixed_seed is not None else random.randint(0, 2 ** 31 - 1)
        rng = np.random.default_rng(seed)
        if self.train:
            m = R.perturb_mask(it["pm"] > 0.5, rng)
            flip = False
        else:
            m = it["pm"] > 0.5
            flip = False
        img = it["img"]; gt = it["gt"]; valid = it["valid"]
        if flip:
            img = img[:, ::-1].copy(); m = m[:, ::-1].copy()
            gt = gt[:, ::-1].copy(); valid = valid[:, ::-1].copy()
        sdf = R.make_sdf(m)
        # only image + mask + sdf; the network predicts a smooth deformation
        x = np.stack([img, m.astype(np.float32), sdf], 0).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(gt[None].astype(np.float32)),
                torch.from_numpy(valid[None].astype(np.float32)), it["name"])


def soft_dice(p, t, eps=1e-6):
    tp = (p * t).sum((2, 3))
    return (1 - (2 * tp + eps) / (p.sum((2, 3)) + t.sum((2, 3)) + eps)).mean()


def run_epoch(model, loader, opt, scaler, dev, train=True):
    model.train(train)
    tot, n = 0.0, 0
    for x, gt, valid, _ in loader:
        x, gt, valid = x.to(dev), gt.to(dev), valid.to(dev)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                disp, code = model(x)
                warped = warp(x[:, 1:2], disp)
                logits = torch.logit(warped.float().clamp(1e-3, 1 - 1e-3))
                l_bce = (F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
                         * valid).sum() / (valid.sum() + 1e-6)
                l_dice = soft_dice(warped * valid, gt * valid)
                # smoothness of the displacement field (low-res)
                dx = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1]).mean()
                dy = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :]).mean()
                l_smooth = 0.05 * (dx + dy)
                l_code = 1e-5 * code.pow(2).mean()
                loss = l_bce + l_dice + l_smooth + l_code
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        tot += float(loss.detach()) * x.shape[0]; n += x.shape[0]
    return tot / max(1, n)


@torch.no_grad()
def predict(model, name, pred_dir, dev):
    raw, gt, pm = R.load_case(name, pred_dir)
    img_c, (nh, nw) = R._canvas(raw)
    pm_c, _ = R._canvas(pm.astype(np.uint8), True)
    pm_c = pm_c > 0
    sdf = R.make_sdf(pm_c)
    x = np.stack([img_c, pm_c.astype(np.float32), sdf], 0)[None].astype(np.float32)
    disp, _ = model(torch.from_numpy(x).to(dev))
    warped = warp(torch.from_numpy(pm_c.astype(np.float32))[None, None].to(dev), disp)[0, 0].cpu().numpy()
    pred_c = warped[:nh, :nw] >= 0.5
    pred = np.asarray(Image.fromarray((pred_c * 255).astype(np.uint8)).resize(
        (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
    return pred, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-disp", type=float, default=25.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "deform_fold0"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pred-dir", default=VAL_PRED)
    ap.add_argument("--tag", default="deform")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    train_names = sorted(os.listdir(TRAIN_PRED))
    val_names = sorted(os.listdir(VAL_PRED))
    if args.smoke:
        train_names = train_names[:8]
        val_names = val_names[:4]
    sel_names = train_names[::7]
    fit_names = [n for n in train_names if n not in set(sel_names)]
    print("[deform] train %d sel %d val %d" % (len(fit_names), len(sel_names), len(val_names)), flush=True)

    ds_fit = DeformDS(fit_names, train=True)
    ds_sel = DeformDS(sel_names, train=True, fixed_seed=999)
    dl_fit = DataLoader(ds_fit, batch_size=args.batch, shuffle=True, num_workers=3, pin_memory=True)
    dl_sel = DataLoader(ds_sel, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = DeformNet(base=args.base, k=args.k, max_disp=args.max_disp).to(dev)
    print("[deform] params %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    if not args.eval_only:
        best = (-1e9, -1)
        for ep in range(args.epochs):
            t0 = time.time()
            tr = run_epoch(model, dl_fit, opt, scaler, dev, True)
            sched.step()
            msg = "ep %3d/%d loss %.4f lr %.2e %.1f min" % (ep + 1, args.epochs, tr,
                                                             opt.param_groups[0]["lr"], (time.time() - t0) / 60)
            if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
                model.eval()
                ds_, bs_ = [], []
                for x, gt, valid, _ in dl_sel:
                    x, gt, valid = x.to(dev), gt.to(dev), valid.to(dev)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        disp, _ = model(x)
                        warped = warp(x[:, 1:2], disp)
                    p = (warped > 0.5) & (valid > 0.5)
                    g = (gt > 0.5) & (valid > 0.5)
                    for i in range(p.shape[0]):
                        pi, gi = p[i, 0].cpu().numpy(), g[i, 0].cpu().numpy()
                        ds_.append(2 * (pi & gi).sum() / max(1, pi.sum() + gi.sum()) * 100)
                        bs_.append(R.bdice(pi, gi))
                sc = float(np.mean(bs_))
                msg += " | sel dice %.2f bdice %.2f" % (np.mean(ds_), np.mean(bs_))
                if sc > best[0]:
                    best = (sc, ep)
                    torch.save({"net": model.state_dict(), "args": vars(args), "epoch": ep},
                               os.path.join(args.out, "ckpt_best.pt"))
                model.train()
            print(msg, flush=True)
            if ep % 25 == 0 or ep == args.epochs - 1:
                torch.save({"net": model.state_dict(), "args": vars(args), "epoch": ep},
                           os.path.join(args.out, "ckpt.pt"))

    ck = args.ckpt or os.path.join(args.out, "ckpt_best.pt")
    if not os.path.exists(ck):
        ck = os.path.join(args.out, "ckpt.pt")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["net"]); model.eval()
    print("[deform] eval %s (ep %s)" % (ck, blob.get("epoch")), flush=True)
    rows = []
    for n in val_names:
        pred, gt = predict(model, n, args.pred_dir, dev)
        pm = np.asarray(Image.open(os.path.join(args.pred_dir, n)).convert("L")) > 0
        mr = all_metrics(pred, gt); mi = all_metrics(pm, gt)
        rows.append(dict(name=n, dice_in=mi["dice"], dice_out=mr["dice"],
                         bd_in=R.bdice(pm, gt), bd_out=R.bdice(pred, gt),
                         hd95_in=mi["hd95"], hd95_out=mr["hd95"],
                         assd_in=mi["assd"], assd_out=mr["assd"],
                         vd_in=mi["vol_diff_pct"], vd_out=mr["vol_diff_pct"]))
        print("  %-28s dice %.3f -> %.3f | bd1 %.2f -> %.2f | hd95 %.2f -> %.2f" %
              (n, mi["dice"], mr["dice"], rows[-1]["bd_in"], rows[-1]["bd_out"],
               mi["hd95"], mr["hd95"]), flush=True)
    with open(os.path.join(args.out, "eval_%s.csv" % args.tag), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    from scipy import stats as st
    summary = {}
    for key in ["dice", "bd", "hd95", "assd", "vd"]:
        a = np.array([r[key + "_in"] for r in rows]); b = np.array([r[key + "_out"] for r in rows])
        t, p = st.ttest_rel(b, a)
        try: _, pw = st.wilcoxon(b, a)
        except Exception: pw = float("nan")
        summary[key] = dict(in_mean=float(a.mean()), out_mean=float(b.mean()),
                            delta=float(b.mean() - a.mean()), p_t=float(p), p_w=float(pw),
                            wins=int((b > a).sum()) if key in ("dice", "bd") else int((b < a).sum()))
    with open(os.path.join(args.out, "eval_%s.json" % args.tag), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
