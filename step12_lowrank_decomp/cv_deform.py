#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step12_lowrank_decomp/cv_deform.py

5-fold cross-validation of the low-rank deformation refiner on the 31 val cases
of the delivered SQ2res-w25 big model (fold-0).

Why CV instead of synthetic perturbations: the big model is near-perfect on its
71 training cases, so a refiner trained on synthetic perturbations of those
masks learns "do nothing" and does not transfer to the val errors.  Here the
refiner sees the *actual* big-model errors on 25 val cases and is evaluated on
the 6 held-out cases; aggregating the 5 folds gives an out-of-fold prediction
for all 31 cases.

This is a pilot: it tells us whether a constrained low-rank deformation field
can correct the real error mode (coherent boundary arcs) at all.
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_refiner as R
import train_deform as D
from eval_lib import all_metrics

VAL_PRED = R.VAL_PRED


def train_one(names, dev, epochs, batch, lr, base, k, max_disp, pred_dir=None):
    ds = D.DeformDS(names, train=False, pred_dir=pred_dir)  # actual big-model masks
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=3, pin_memory=True)
    model = D.DeformNet(base=base, k=k, max_disp=max_disp).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, gt, valid, _ in dl:
            x, gt, valid = x.to(dev), gt.to(dev), valid.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                disp, code = model(x)
                warped = D.warp(x[:, 1:2], disp)
                logits = torch.logit(warped.float().clamp(1e-3, 1 - 1e-3))
                l_bce = (F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
                         * valid).sum() / (valid.sum() + 1e-6)
                l_dice = D.soft_dice(warped * valid, gt * valid)
                dx = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1]).mean()
                dy = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :]).mean()
                loss = l_bce + l_dice + 0.05 * (dx + dy) + 1e-5 * code.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * x.shape[0]; n += x.shape[0]
        sched.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print("    ep %3d/%d loss %.4f" % (ep + 1, epochs, tot / max(1, n)), flush=True)
    return model


@torch.no_grad()
def predict_model(model, names, pred_dir, dev, out_dir):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for n in names:
        raw, gt, pm = R.load_case(n, pred_dir)
        img_c, (nh, nw) = R._canvas(raw)
        pm_c, _ = R._canvas(pm.astype(np.uint8), True)
        pm_c = pm_c > 0
        sdf = R.make_sdf(pm_c)
        x = np.stack([img_c, pm_c.astype(np.float32), sdf], 0)[None].astype(np.float32)
        disp, _ = model(torch.from_numpy(x).to(dev))
        warped = D.warp(torch.from_numpy(pm_c.astype(np.float32))[None, None].to(dev), disp)[0, 0].cpu().numpy()
        pred_c = warped[:nh, :nw] >= 0.5
        pred = np.asarray(Image.fromarray((pred_c * 255).astype(np.uint8)).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
        Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(out_dir, n))
        mr = all_metrics(pred, gt); mi = all_metrics(pm, gt)
        rows.append(dict(name=n, dice_in=mi["dice"], dice_out=mr["dice"],
                         bd_in=R.bdice(pm, gt), bd_out=R.bdice(pred, gt),
                         hd95_in=mi["hd95"], hd95_out=mr["hd95"],
                         assd_in=mi["assd"], assd_out=mr["assd"]))
    return rows


def summarize(rows, tag):
    from scipy import stats as st
    s = {}
    for key in ["dice", "bd", "hd95", "assd"]:
        a = np.array([r[key + "_in"] for r in rows]); b = np.array([r[key + "_out"] for r in rows])
        t, p = st.ttest_rel(b, a)
        try: _, pw = st.wilcoxon(b, a)
        except Exception: pw = float("nan")
        s[key] = dict(in_mean=float(a.mean()), out_mean=float(b.mean()),
                      delta=float(b.mean() - a.mean()), p_t=float(p), p_w=float(pw),
                      wins=int((b > a).sum()) if key in ("dice", "bd") else int((b < a).sum()))
    print(tag, json.dumps(s, indent=2), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-disp", type=float, default=25.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pred-dir", default=VAL_PRED)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "cv_deform"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    names = sorted(os.listdir(args.pred_dir))
    rng = np.random.default_rng(args.seed)
    perm = list(rng.permutation(len(names)))
    folds = [ [names[i] for i in perm[f::args.folds]] for f in range(args.folds) ]
    print("[cv] n=%d folds=%s" % (len(names), [len(f) for f in folds]), flush=True)

    all_rows = []
    for fi, held in enumerate(folds):
        tr = [n for n in names if n not in set(held)]
        print("[cv] fold %d/%d train %d held %d" % (fi + 1, args.folds, len(tr), len(held)), flush=True)
        model = train_one(tr, dev, args.epochs, args.batch, args.lr,
                          args.base, args.k, args.max_disp, pred_dir=args.pred_dir)
        out_dir = os.path.join(args.out, "fold%d" % fi)
        rows = predict_model(model, held, args.pred_dir, dev, out_dir)
        all_rows.extend(rows)
        torch.save({"net": model.state_dict()}, os.path.join(args.out, "fold%d.pt" % fi))
        del model
        torch.cuda.empty_cache()

    all_rows.sort(key=lambda r: r["name"])
    with open(os.path.join(args.out, "cv_per_case.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    s = summarize(all_rows, "[cv] out-of-fold")
    with open(os.path.join(args.out, "cv_summary.json"), "w") as f:
        json.dump(s, f, indent=2)


if __name__ == "__main__":
    main()
