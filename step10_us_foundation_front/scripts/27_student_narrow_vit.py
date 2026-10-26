#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/27_student_narrow_vit.py

L2-v3: a NARROW ViT student (same architecture family as the teacher) distilled
from the 24-block UltraSAM3 trunk, with the sandwich head.

WHY A ViT STUDENT (measured evidence, all in results/)
-----------------------------------------------------
* depth frontier (teacher_truncation.json): 24 of 32 blocks keep the full
  robustness (compound_B 91.02 vs 91.31) -- 16 blocks 85.12, 8 blocks 34.18.
* width frontier (feature_subset_24.json): the first 128 of 1024 feature
  channels lose only 1.7 Dice on compound_B (89.31 vs 91.02).
* the from-scratch CONV student (runs/l2_student_views_200ep) reached only
  compound_B 63.02 even with view distillation, and adding stronger blur views
  did not help (run A) -> the failure is ARCHITECTURAL: a stride-8 conv
  bottleneck cannot recover a boundary that blur destroyed, while a
  window-attention + RoPE ViT (the teacher's family) can.

So the student keeps the family and gives up width/depth:
    ViT: patch 14, embed 256, depth 16, window 24, global blocks (3,7,11,15),
         RoPE + tiled abs-pos  ~= 13 M params
    + decoder to stride 2 + 6-channel sandwich head  ~= 0.5 M
    (vs the teacher's 841.7 M -> ~60x smaller, and ~12x smaller in the parts
     the depth/width frontiers say are redundant)

Distillation: cosine on the 24-block trunk feature map (cached), on the SAME
72x54 valid token grid -> no resampling.  Training views: 40 % clean + 60 % of
the cached shifted views (the ingredient that produced the only robustness gain
so far).

usage:
  ~/anaconda3/envs/zk/bin/python scripts/27_student_narrow_vit.py --epochs 200
  ~/anaconda3/envs/zk/bin/python scripts/27_student_narrow_vit.py --eval-only --tag <run>
"""
import argparse
import csv
import importlib
import importlib.util
import json
import os
import random
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
OUT = os.path.join(ROOT, "runs")
VIEWS_DIR = os.path.join(ROOT, "results", "teacher_views_t24")      # 24-block teacher
RES = os.path.join(ROOT, "results")
META = {}                  # per-cache _meta.json: {image: [{view, seed, ...}]}
TDIM = 1024                # teacher feature channels (SAM3 trunk 1024, UltraFedFM 768)
SCRAMBLE = False           # R2 control: per-sample spatial permutation of the target
sys.path.insert(0, HERE)


def _load(mod_name, fname):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


l0 = _load("l0n", "10_l0_probe.py")
fs = _load("fsn", "fold_split.py")
tr = _load("trn", "21_train_student.py")
tv = _load("tvn", "23_cache_teacher_views.py")

TC = (1008, 1008)          # teacher/student canvas (SAM3 img_size, patch 14)
VALID_H = 54               # 756/14 valid token rows for our 1024x768 images
PATCH = 14
LOSS_STRIDE = 2            # DC+CE grid: 504x504 canvas
SHIFT_DIR = os.path.join(USS, "step8_domain_shift", "data", "shift")
CONDITIONS = ["in_domain", "affine_gain_m25", "affine_gain_p25", "blur_20", "blur_35",
              "compound_A", "compound_B", "gamma_060", "gamma_080", "gamma_125",
              "jpeg_15", "jpeg_30", "shadow", "speckle_15", "speckle_35", "tgc_m60", "tgc_p60"]


# --------------------------------------------------------------------- network
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class ViTStudent(nn.Module):
    def __init__(self, embed=256, depth=16, head_bias=1.0):
        super().__init__()
        from sam3.model.vitdet import ViT
        self.embed, self.depth = embed, depth
        self.trunk = ViT(img_size=TC[0], pretrain_img_size=336, patch_size=PATCH,
                         embed_dim=embed, depth=depth, num_heads=max(1, embed // 32),
                         mlp_ratio=4.0, qkv_bias=True, use_abs_pos=True, tile_abs_pos=True,
                         global_att_blocks=tuple(range(3, depth, 4)), rel_pos_blocks=(),
                         use_rope=True, use_interp_rope=True, window_size=24,
                         pretrain_use_cls_token=True, retain_cls_token=False,
                         ln_pre=True, ln_post=True, return_interm_layers=False,
                         bias_patch_embed=False, use_act_checkpoint=False)
        self.d1 = ConvBlock(embed, 128)
        self.d2 = ConvBlock(128, 64)
        self.d3 = ConvBlock(64, 32)
        self.d4 = ConvBlock(32, 32)
        # Multi-level decoder + deep supervision: the shallow 2-level decoder of
        # the first version (72 -> t/2 -> t, i.e. d3 was an identity-resolution
        # step) is the likely reason its in-domain boundary metric sat at 13.8
        # while the conv student with a real 4-level decoder reached 21.3.
        self.seg = nn.Conv2d(32, 6, 1, 1)
        self.aux4 = nn.Conv2d(32, 6, 1, 1)      # stride 4 of the canvas
        self.aux8 = nn.Conv2d(64, 6, 1, 1)      # stride 8 of the canvas
        with torch.no_grad():
            w, b = self.seg.weight.data, self.seg.bias.data
            w[2:4].copy_(w[0:2]); w[4:6].copy_(w[0:2])
            b[2:4].copy_(b[0:2]); b[4:6].copy_(b[0:2])
            b[3] = b[1] - head_bias
            b[5] = b[1] + head_bias

    def forward(self, x):
        feat = self.trunk(x)[-1]                                   # (B,embed,72,72)
        t = (TC[0] // LOSS_STRIDE, TC[1] // LOSS_STRIDE)      # 504 = stride 2
        g4 = (t[0] // 2, t[1] // 2)                            # stride 4
        g8 = (t[0] // 4, t[1] // 4)                            # stride 8
        d1 = self.d1(feat)                                     # token grid (stride 14)
        d2 = self.d2(F.interpolate(d1, size=g8, mode="nearest"))
        d3 = self.d3(F.interpolate(d2, size=g4, mode="nearest"))
        d4 = self.d4(F.interpolate(d3, size=t, mode="nearest"))
        return self.seg(d4), feat, self.aux4(d3), self.aux8(d2)


# ------------------------------------------------------------------------ data
def canvas_input(raw01):
    """(H,W) in [0,1] -> (1,3,1008,1008) z-scored canvas, image at top-left."""
    H, W = raw01.shape
    nw = TC[0]
    nh = min(TC[1], max(1, int(round(H * nw / float(W)))))
    small = np.asarray(Image.fromarray((raw01 * 255).astype(np.uint8)).resize(
        (nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    c = np.zeros(TC, dtype=np.float32)
    c[:nh, :nw] = small
    c = (c - c.mean()) / (c.std() + 1e-6)
    return torch.from_numpy(c)[None, None].repeat(1, 3, 1, 1)


def gt_canvas(name):
    gt = l0.eval_lib.load_binary(os.path.join(USS, "UltrasoundProstate", "masks", name)) > 0
    H, W = gt.shape
    nw = TC[0]
    nh = min(TC[1], max(1, int(round(H * nw / float(W)))))
    g = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)) > 127
    c = np.zeros(TC, dtype=bool)
    c[:nh, :nw] = g
    return c


def grid(mask, stride):
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (TC[1] // stride, TC[0] // stride), Image.NEAREST)) > 127


def raw_of(name, view):
    """raw [0,1] image for a training view.

    2026-10-19 FIX: v>0 used to be REGENERATED with seed=hash((name,view)) --
    CPython string hashing is randomised per process, so the student's view
    never matched the realisation the cached teacher feature was computed on
    (23 wrote image+feat in ONE process).  We now regenerate from the seed
    recorded in the cache's _meta.json, so student input and teacher target
    are the SAME realisation."""
    raw = np.asarray(Image.open(os.path.join(USS, "UltrasoundProstate", "images", name))
                     .convert("L"), dtype=np.float32) / 255.0
    if view == 0:
        return raw
    seed = None
    for r in META.get(name, ()):
        if int(r["view"]) == view:
            seed = int(r["seed"])
            break
    if seed is None:                       # cache without meta: stable fallback
        seed = zlib.crc32(("%s|%d" % (name, view)).encode()) % (2 ** 31)
    return tv.shift_image(raw, tv.VIEWS[view], seed, tv.RECIPE.get(view))


def batch_iter(names, rng, batch, targets, views, clean_prob, tf):
    while True:
        idx = rng.sample(range(len(names)), min(batch, len(names)))
        xs, fts, yb, yc, ya = [], [], [], [], []
        for i in idx:
            nm = names[i]
            v = 0 if (not views or rng.random() <= clean_prob) else int(rng.choice(views))
            raw = raw_of(nm, v)
            flip = rng.random() < 0.5
            if flip:                       # the GEOMETRY must be mirrored on the
                raw = raw[:, ::-1]         # teacher map too, or the distillation
            xs.append(canvas_input(raw))   # target is misaligned
            g = gt_canvas(nm)
            if flip:
                g = g[:, ::-1]
            inner, outer = targets(g)
            yb.append(grid(g, LOSS_STRIDE).astype(np.int64))
            yc.append(grid(inner, LOSS_STRIDE).astype(np.int64))
            ya.append(grid(outer, LOSS_STRIDE).astype(np.int64))
            p = os.path.join(VIEWS_DIR, "%s__v%d.feat.npy" % (nm[:-4], v))
            # The cache holds each image cropped to ITS OWN valid token rows
            # (54 for 1024x768, 52 for the 1260x910 cases), so pad to the common
            # grid.  The missing rows are the canvas padding, where the teacher
            # itself saw zeros -> zero padding is faithful, and the student's
            # own feature rows there are pad-driven too.
            f = np.zeros((TDIM, VALID_H, TC[0] // PATCH), dtype=np.float32)
            if os.path.exists(p):
                a = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32)
                gh = min(a.shape[1], VALID_H)
                f[:a.shape[0], :gh, :] = a[:a.shape[0], :gh, :]
            if flip:
                f = np.ascontiguousarray(f[:, :, ::-1])
            if SCRAMBLE:
                # R2 control: destroy the spatial content of the target with a
                # per-(image,view) permutation (stable across epochs, random
                # across samples): the student still matches "a real FM feature
                # map of the right size", but none of its spatial information.
                r = np.random.default_rng(zlib.crc32(("%s|s|%d" % (nm, v)).encode()) % (2 ** 31))
                perm = r.permutation(f.shape[1] * f.shape[2])
                f = f.reshape(f.shape[0], -1)[:, perm].reshape(f.shape)
            fts.append(torch.from_numpy(f))
        yield (torch.cat(xs).float(),
               tuple(torch.from_numpy(np.stack(v)).long() for v in (yb, yc, ya)),
               torch.stack(fts).float())


# ----------------------------------------------------------------------- train
@torch.no_grad()
def quick_val_dice(net, names, dev, n=12):
    """Cheap in-domain val Dice on n val patients (no TTA) -> checkpoint selector.
    Batches of 10 epochs did land in a bad basin for one split (fold 2: base loss
    plateaued at 0.75 while folds 0/1 were at 0.10), so the reported number must be
    a *selected* checkpoint, exactly like nnU-Net's checkpoint_best."""
    net.eval()
    ds = []
    for nm in names[:n]:
        raw = np.asarray(Image.open(os.path.join(USS, "UltrasoundProstate", "images", nm))
                         .convert("L"), dtype=np.float32) / 255.0
        x = canvas_input(raw).to(dev)
        lg = F.interpolate(net(x)[0].float(), size=TC, mode="bilinear", align_corners=False)
        H, W = raw.shape
        nw = TC[0]
        nh = min(TC[1], max(1, int(round(H * nw / float(W)))))
        pb = torch.softmax(lg[:, 0:2], 1)[:, 1][0].cpu().numpy()[:nh, :nw] > 0.5
        g = gt_canvas(nm)[:nh, :nw]
        ds.append(2.0 * (pb & g).sum() / max(1.0, float(pb.sum() + g.sum())))
    net.train()
    return float(np.mean(ds))


def train(args):
    dev = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    run = os.path.join(OUT, args.tag)
    os.makedirs(run, exist_ok=True)
    log = open(os.path.join(run, "train.log"), "w", buffering=1)

    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n")

    global VIEWS_DIR, META, TDIM, SCRAMBLE
    if args.views_dir:
        VIEWS_DIR = args.views_dir if os.path.isabs(args.views_dir) else os.path.join(ROOT, args.views_dir)
        print("[vit-student] custom teacher cache %s" % VIEWS_DIR, flush=True)
    elif args.fold:
        VIEWS_DIR = VIEWS_DIR + "_fold%d" % args.fold
        print("[vit-student] fold %d -> teacher cache %s" % (args.fold, VIEWS_DIR), flush=True)
    TDIM = args.teacher_dim
    SCRAMBLE = args.scramble_feats
    _mp = os.path.join(VIEWS_DIR, "_meta.json")
    if os.path.exists(_mp):
        META = json.load(open(_mp))["images"]
        print("[vit-student] _meta.json loaded: %d images (aligned view seeds)" % len(META), flush=True)
    else:
        print("[vit-student] WARNING: no _meta.json in %s -> crc32 fallback seeds" % VIEWS_DIR, flush=True)
    train_names, val_names = fs.read_split(args.fold)
    if args.limit:
        train_names = train_names[:args.limit]
    angles, scale = tr.fingerprint()
    targets = tr.Targets(args.rc, args.ra, angles, scale)
    net = ViTStudent(embed=args.embed, depth=args.depth).to(dev)
    n_net = sum(p.numel() for p in net.parameters())
    proj = nn.Conv2d(args.embed, args.teacher_dim, 1, bias=False).to(dev)
    n_proj = sum(p.numel() for p in proj.parameters())
    views = tuple(int(v) for v in args.views.split(",") if v.strip()) if args.views else ()
    P("[vit-student] tag=%s embed=%d depth=%d epochs=%d batch=%d lr=%g" %
      (args.tag, args.embed, args.depth, args.epochs, args.batch, args.lr))
    _tparam = {1024: 841.7e6, 768: 111.9e6}.get(args.teacher_dim, float(args.teacher_dim))
    P("[vit-student] params %.2f M deployable (+%.2f M training-only projector) | teacher dim %d -> %.0fx"
      % (n_net / 1e6, n_proj / 1e6, args.teacher_dim, _tparam / n_net))
    P("[vit-student] views=%s clean_prob=%.2f | teacher target = cached trunk features over the SAME views"
      % (views or "none", args.clean_prob))
    P("[vit-student] fingerprint weakest sectors -> scale %s"
      % ([round(s, 2) for s in scale] if scale else None))

    opt = torch.optim.AdamW(list(net.parameters()) + list(proj.parameters()), lr=args.lr, weight_decay=1e-4)
    def _lr(e, epochs=args.epochs, warm=args.warmup):
        if e < warm:                      # linear warmup, then cosine-ish decay
            return (e + 1) / float(max(1, warm))
        return (1 - (e - warm) / max(1, epochs - warm)) ** 0.9

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(args.seed)
    steps = max(1, len(train_names) // args.batch)
    it = batch_iter(train_names, rng, args.batch, targets, views, args.clean_prob, None)
    t0 = time.time()
    best = (0.0, -1)
    for ep in range(args.epochs):
        net.train()
        acc = {}
        for _ in range(steps):
            x, (yb, yc, ya), tf = next(it)
            x = x.to(dev); yb, yc, ya = (t.to(dev) for t in (yb, yc, ya)); tf = tf.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, feat, a4, a8 = net(x)
                lg = logits.float()
                l_base = tr.dc_ce(lg[:, 0:2], yb)
                l_c = tr.dc_ce(lg[:, 2:4], yc)
                l_a = tr.dc_ce(lg[:, 4:6], ya)
                p_b = torch.softmax(lg[:, 0:2], 1)[:, 1]
                p_c = torch.softmax(lg[:, 2:4], 1)[:, 1]
                p_a = torch.softmax(lg[:, 4:6], 1)[:, 1]
                l_nest = (F.relu(p_c - p_b) + F.relu(p_b - p_a)).mean()
                loss = (l_base + args.w_cautious * l_c + args.w_amplified * l_a
                        + args.w_nest * l_nest) / (1 + args.w_cautious + args.w_amplified)
                if args.w_deep > 0:      # deep supervision on the base head
                    y4 = F.interpolate(yb[:, None].float(), size=a4.shape[-2:], mode="nearest")[:, 0].long()
                    y8 = F.interpolate(yb[:, None].float(), size=a8.shape[-2:], mode="nearest")[:, 0].long()
                    l_ds = tr.dc_ce(a4.float()[:, 0:2], y4) + tr.dc_ce(a8.float()[:, 0:2], y8)
                    loss = loss + args.w_deep * l_ds
                l_d = tf.new_zeros(())
                if args.w_distill > 0:
                    s = proj(feat.float()[:, :, :VALID_H, :])
                    l_d = tr.distill_cosine(s, tf)
                    loss = loss + args.w_distill * l_d
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            for k, v in (("base", l_base), ("caut", l_c), ("amp", l_a), ("nest", l_nest), ("dist", l_d)):
                acc[k] = acc.get(k, 0.0) + float(v.detach())
        sched.step()
        if args.sel_every and (ep % args.sel_every == 0 or ep == args.epochs - 1):
            blob = {"net": net.state_dict(), "embed": args.embed, "depth": args.depth,
                    "args": vars(args), "epoch": ep}
            if ep == args.epochs - 1:
                torch.save(blob, os.path.join(run, "ckpt.pt"))
            d = quick_val_dice(net, val_names, dev, n=args.sel_n)
            if d > best[0]:
                best = (d, ep)
                torch.save(blob, os.path.join(run, "ckpt_best.pt"))
            P("[vit-student]   sel ep %3d val_dice %.4f best %.4f @ep%d" % (ep, d, best[0], best[1]))
        if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs - 1:
            P("[vit-student] ep %3d/%d base %.4f caut %.4f amp %.4f nest %.5f dist %.4f lr %.2e %.1f min"
              % (ep, args.epochs, acc["base"] / steps, acc["caut"] / steps, acc["amp"] / steps,
                 acc["nest"] / steps, acc["dist"] / steps, opt.param_groups[0]["lr"], (time.time() - t0) / 60))
        if ep % 25 == 0 or ep == args.epochs - 1:
            blob = {"net": net.state_dict(), "embed": args.embed, "depth": args.depth,
                    "args": vars(args), "epoch": ep}
            torch.save(blob, os.path.join(run, "ckpt.pt"))
            if args.snapshots:
                torch.save(blob, os.path.join(run, "ckpt_ep%03d.pt" % ep))
    P("[vit-student] done %.1f min -> %s" % ((time.time() - t0) / 60, run))


# ------------------------------------------------------------------------ eval
@torch.no_grad()
def eval_run(args):
    dev = torch.device(args.device)
    ck_path = os.path.join(OUT, args.tag, args.ckpt or "ckpt.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net = ViTStudent(embed=ck["embed"], depth=ck["depth"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    n_net = sum(p.numel() for p in net.parameters())
    _, val_names = fs.read_split(args.fold)
    rows = []
    conds = CONDITIONS if args.cond == "all" else [c.strip() for c in args.cond.split(",") if c.strip()]
    for cond in conds:
        for n in val_names:
            ip = os.path.join(args.image_root, cond, "images", n)
            if not os.path.exists(ip):
                # NEVER fall back silently: a missing shift directory made fold 0
                # report the clean-image number under every condition name.
                if cond != "in_domain":
                    raise SystemExit("[eval] missing shift image for cond=%s: %s "
                                     "(wrong --image-root?)" % (cond, ip))
                ip = os.path.join(USS, "UltrasoundProstate", "images", n)
            raw = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
            x = canvas_input(raw).to(dev)
            logits = net(x)[0]
            if args.tta:
                l2 = net(torch.flip(x, dims=[3]))[0]
                logits = 0.5 * (logits + torch.flip(l2, dims=[3]))
            lg = F.interpolate(logits.float(), size=(TC[1], TC[0]), mode="bilinear", align_corners=False)
            pb = (torch.softmax(lg[:, 0:2], 1)[:, 1] > 0.5)[0].cpu().numpy()
            g = gt_canvas(n)
            m = l0.eval_lib.all_metrics(pb, g, None)
            rows.append({"condition": cond, "name": n, "dice": float(m["dice"]), "iou": float(m["iou"]),
                         "bdice1": float(l0.s57.bdice(pb, g, 1))})
        sub = [r for r in rows if r["condition"] == cond]
        d = [r["dice"] for r in sub]
        print("   %-16s Dice %6.2f (min %6.2f) | bdice@1 %5.2f | <80 %2d/%d"
              % (cond, np.mean(d), np.min(d), np.mean([r["bdice1"] for r in sub]),
                 sum(x < 80 for x in d), len(d)), flush=True)
    tag = args.tag_out or args.tag
    with open(os.path.join(RES, "l2vit_eval_%s.csv" % tag), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {c: {"dice": float(np.mean([r["dice"] for r in rows if r["condition"] == c])),
                   "dice_min": float(np.min([r["dice"] for r in rows if r["condition"] == c])),
                   "bdice1": float(np.mean([r["bdice1"] for r in rows if r["condition"] == c])),
                   "n_lt80": int(sum(r["dice"] < 80 for r in rows if r["condition"] == c))} for c in conds}
    with open(os.path.join(RES, "l2vit_eval_%s.json" % tag), "w") as fh:
        json.dump({"tag": args.tag, "params": n_net, "summary": summary}, fh, indent=2)
    print("[vit-student] params %.2f M | wrote results/l2vit_eval_%s.{json,csv}" % (n_net / 1e6, tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=0, help="linear LR warmup epochs")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--rc", type=float, default=3.0)
    ap.add_argument("--ra", type=float, default=4.0)
    ap.add_argument("--w-distill", type=float, default=1.0)
    ap.add_argument("--views-dir", type=str, default="",
                    help="override teacher cache dir (default: results/teacher_views_t24[_foldK])")
    ap.add_argument("--teacher-dim", type=int, default=1024,
                    help="teacher feature channels (SAM3 trunk 1024; UltraFedFM 768)")
    ap.add_argument("--scramble-feats", action="store_true",
                    help="R2 control: per-sample spatial permutation of teacher targets")
    ap.add_argument("--w-nest", type=float, default=0.5)
    ap.add_argument("--w-deep", type=float, default=0.5, help="deep-supervision weight (0 = off)")
    ap.add_argument("--w-cautious", type=float, default=0.25)
    ap.add_argument("--w-amplified", type=float, default=0.25)
    ap.add_argument("--views", default="1,2,3")
    ap.add_argument("--clean-prob", type=float, default=0.4)
    ap.add_argument("--tag", default="l2vit_200ep")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train images (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sel-every", type=int, default=0,
                    help="every N epochs, score a quick val Dice and keep ckpt_best.pt (0 = off)")
    ap.add_argument("--sel-n", type=int, default=12, help="val patients used by the selector")
    ap.add_argument("--snapshots", action="store_true",
                    help="also keep ckpt_ep<NNN>.pt every 25 epochs (for val-based best selection)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--cond", default="all")
    ap.add_argument("--image-root", default=SHIFT_DIR)
    ap.add_argument("--tag-out", default="")
    ap.add_argument("--ckpt", default="", help="checkpoint filename inside the run dir (default ckpt.pt)")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    if args.eval_only:
        eval_run(args)
    else:
        train(args)


if __name__ == "__main__":
    sys.exit(main())
