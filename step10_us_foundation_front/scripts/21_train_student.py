#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/21_train_student.py

L2 PROTOTYPE: a small "US core extractor" + the sandwich head, distilled from
the cached UltraSAM3 trunk features.

WHY THIS SHAPE (all measured in this project)
---------------------------------------------
* L0 probe: UltraSAM3's frozen features are the best AND the shift-robustest
  (92.80 -> 91.31 Dice under compound_B, 0/31 cases < 80) but cost 841.7 M
  params / 10 GB.
* CKA: UltraSAM3 is near-orthogonal to UltraFedFM/UltraSam (0.13/0.17) while
  those two are redundant (0.72) => ONE teacher, and it is this one.
* Channel importance: 90 % of the boundary signal is spread over 64-67 % of all
  channels => there is no sparse "boundary channel" subset to keep, so the
  compression must be STRUCTURAL (fewer/narrower stages) + distillation, not
  top-k channel pruning.

Student: 4-stage conv encoder (32/64/128/256), 3-level decoder, ~0.5 M params,
6-channel head tied at init with a logit offset exactly like step6's sandwich.

Three loss groups:
  (1) FEATURE DISTILLATION (cosine, at stride 8 and stride 4 taps) against the
      cached UltraSAM3 map -- the teacher's "core content", not its weights.
      Geometric augmentation is applied to the teacher map with the same
      parameters as the image, while the intensity augmentation is applied to
      the image only => the distillation term explicitly asks the student to be
      intensity-shift invariant (which is what step8/step9 showed matters).
  (2) SANDWICH losses: stock DC+CE for base (vs GT), cautious (vs erode 3 px),
      amplified (vs dilate 4 px).
  (3) INTERVAL term: soft nesting (cautious <= base <= amplified) + the radii
      modulated by the step-4 angular fingerprint (wider interval where the
      image evidence is weak).

usage:
  ~/anaconda3/envs/zk/bin/python scripts/21_train_student.py --epochs 200 --device cuda:0
  ~/anaconda3/envs/zk/bin/python scripts/21_train_student.py --epochs 2 --limit 6 --tag smoke
"""
import argparse
import importlib.util
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
OUT = os.path.join(ROOT, "runs")
TEACHER = os.path.join(ROOT, "results", "teacher", "ultrasam3")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location("l0", os.path.join(HERE, "10_l0_probe.py"))
l0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l0)

CW, CH = 1024, 768
LOSS_STRIDE = 2                    # DC+CE is computed on a stride-2 grid
TAP_STRIDES = (8, 4)               # distillation taps (e3, e2)
TAP_CH = (256, 128)


# --------------------------------------------------------------------- network
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class StudentNet(nn.Module):
    WIDTHS = (32, 64, 128, 256)          # default (1.95 M deployable)

    def __init__(self, head_bias=1.0, widths=None):
        super().__init__()
        w = tuple(widths) if widths else self.WIDTHS
        self.widths = w
        self.stem = ConvBlock(1, w[0])                  # stride 1
        self.s1 = ConvBlock(w[0], w[1], stride=2)       # /2
        self.s2 = ConvBlock(w[1], w[2], stride=2)       # /4
        self.s3 = ConvBlock(w[2], w[3], stride=2)       # /8
        self.u2 = ConvBlock(w[3] + w[2], w[2])
        self.u1 = ConvBlock(w[2] + w[1], w[1])
        self.u0 = ConvBlock(w[1] + w[0], w[0])
        self.seg = nn.Conv2d(w[0], 6, 1, 1)
        with torch.no_grad():                            # tie the three versions
            wt, bt = self.seg.weight.data, self.seg.bias.data
            wt[2:4].copy_(wt[0:2]); wt[4:6].copy_(wt[0:2])
            bt[2:4].copy_(bt[0:2]); bt[4:6].copy_(bt[0:2])
            bt[3] = bt[1] - head_bias                    # cautious fg logit lower
            bt[5] = bt[1] + head_bias                    # amplified fg logit higher

    def forward(self, x):
        e0 = self.stem(x)
        e1 = self.s1(e0)
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        d2 = self.u2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.u1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        d0 = self.u0(torch.cat([F.interpolate(d1, size=e0.shape[-2:], mode="nearest"), e0], 1))
        return self.seg(d0), {"e3": e3, "e2": e2}


# --------------------------------------------------------------------- losses
def dc_ce(logits, target, eps=1e-5):
    ce = F.cross_entropy(logits, target, reduction="mean")
    p = torch.softmax(logits, 1)[:, 1]
    t = target.float()
    tp = (p * t).sum(1)
    dice = 1.0 - (2 * tp + eps) / (p.sum(1) + t.sum(1) + eps)
    return ce + dice.mean()


def distill_cosine(tap, teacher, eps=1e-6):
    s = F.normalize(tap.float(), dim=1, eps=eps)
    t = F.normalize(teacher.float(), dim=1, eps=eps)
    return (1.0 - (s * t).sum(1)).mean()


# -------------------------------------------------------------------- targets
class Targets(object):
    def __init__(self, r_c=3.0, r_a=4.0, angles=None, sector_scale=None):
        self.r_c, self.r_a = r_c, r_a
        self.angles, self.scale = angles, sector_scale
        self.st = ndimage.generate_binary_structure(2, 1)

    def scale_map(self, gt):
        if not self.scale or self.angles is None:
            return None
        H, W = gt.shape
        cy, cx = ndimage.center_of_mass(gt)
        yy, xx = np.mgrid[0:H, 0:W]
        ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0
        return np.take(np.asarray(self.scale, dtype=np.float32),
                       np.digitize(ang, self.angles) % len(self.scale))

    def __call__(self, gt):
        it = max(1, int(round(self.r_c)))
        dt = max(1, int(round(self.r_a)))
        inner = ndimage.binary_erosion(gt, self.st, iterations=it, border_value=0)
        outer = ndimage.binary_dilation(gt, self.st, iterations=dt)
        sm = self.scale_map(gt)
        if sm is not None:                    # weak-evidence sectors: wider interval
            weak = sm > 1.2
            inner = inner & ~(ndimage.binary_erosion(gt, self.st, iterations=1) & weak)
            outer = outer | (ndimage.binary_dilation(gt, self.st, iterations=dt + 1) & weak)
        return inner, outer


# ------------------------------------------------------------------------ data
def load_gt(name):
    gt = l0.eval_lib.load_binary(os.path.join(USS, "UltrasoundProstate", "masks", name)) > 0
    return np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((CW, CH), Image.NEAREST)) > 127


def load_image(name):
    img = Image.open(os.path.join(USS, "UltrasoundProstate", "images", name)).convert("L")
    img = img.resize((CW, CH), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a, load_gt(name)


def sample_view(name, rng, views, clean_prob, tfeat, views_root):
    """Pick a training view.

    clean      -> the PNG + intensity augmentation, teacher = cached clean map
    shifted vK -> the cached pre-shifted image AND the teacher map computed on
                  that same shift (this is what transfers the teacher's
                  invariance rather than only its clean-image representation)
    Returns (image, gt, teacher_map_or_None, is_shifted).
    """
    if views and rng.random() > clean_prob:
        k = int(rng.choice(views))
        ip = os.path.join(views_root, "%s__v%d.image.npy" % (name[:-4], k))
        fp = os.path.join(views_root, "%s__v%d.feat.npy" % (name[:-4], k))
        if os.path.exists(ip) and os.path.exists(fp):
            a = np.load(ip).astype(np.float32)
            th = np.asarray(np.load(fp, mmap_mode="r"), dtype=np.float32)
            return a, load_gt(name), th, True
    a, gt = load_image(name)
    th = tfeat.get(name)
    return a, gt, (None if th is None else np.asarray(th, dtype=np.float32)), False


def geom_augment(a, gt, rng, teacher=None, intensity=True):
    """Geometric ops are mirrored onto the teacher map; intensity ops only on the
    image (that is what makes the distillation term an invariance objective)."""
    # Flips only: a 90-degree rotation swaps H and W (1024x768 -> 768x1024),
    # which breaks batching, and ultrasound has a fixed depth/lateral geometry
    # so 90-degree rotations are not a meaningful invariance here.
    if rng.random() < 0.5:                       # left-right
        a, gt = a[:, ::-1], gt[:, ::-1]
        if teacher is not None:
            # ascontiguousarray: a reversed slice has negative strides and
            # torch.from_numpy refuses it
            teacher = np.ascontiguousarray(teacher[:, :, ::-1])
    if rng.random() < 0.3:                       # up-down
        a, gt = a[::-1], gt[::-1]
        if teacher is not None:
            teacher = np.ascontiguousarray(teacher[:, ::-1, :])
    if not intensity:
        return np.ascontiguousarray(a), np.ascontiguousarray(gt), teacher
    if rng.random() < 0.5:                                  # intensity gamma
        g = rng.uniform(0.7, 1.4)
        a = np.sign(a) * (np.abs(np.clip(a, -3, 3)) ** g)
    if rng.random() < 0.3:                                  # speckle (step9 axis)
        noise = np.asarray(np.random.default_rng(rng.getrandbits(32)).standard_normal(a.shape),
                           dtype=np.float32)
        a = a * (1.0 + rng.uniform(0.05, 0.3) * noise)
    return np.ascontiguousarray(a), np.ascontiguousarray(gt), teacher


def to_grid(mask, stride):
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (CW // stride, CH // stride), Image.NEAREST)) > 127


def batch_iter(names, rng, batch, targets, tfeat, views=(), clean_prob=0.4,
               views_root=None):
    while True:
        idx = rng.sample(range(len(names)), min(batch, len(names)))
        xs, yb, yc, ya, t8, t4 = [], [], [], [], [], []
        for i in idx:
            nm = names[i]
            a, gt, th, is_shifted = sample_view(nm, rng, views, clean_prob, tfeat, views_root)
            if is_shifted:
                a, gt, th = geom_augment(a, gt, rng, th, intensity=False)
            else:
                a, gt, th = geom_augment(a, gt, rng, th)
            inner, outer = targets(gt)
            xs.append(a[None])
            yb.append(to_grid(gt, LOSS_STRIDE).astype(np.int64))
            yc.append(to_grid(inner, LOSS_STRIDE).astype(np.int64))
            ya.append(to_grid(outer, LOSS_STRIDE).astype(np.int64))
            if th is not None:
                g8 = torch.from_numpy(th)[None]
                g8 = F.interpolate(g8, size=(CH // 8, CW // 8), mode="bilinear",
                                   align_corners=False)[0]
                g4 = F.interpolate(g8[None], size=(CH // 4, CW // 4), mode="bilinear",
                                   align_corners=False)[0]
                t8.append(g8.half()); t4.append(g4.half())
        yield (torch.from_numpy(np.stack(xs)).float(),
               tuple(torch.from_numpy(np.stack(v)).long() for v in (yb, yc, ya)),
               (torch.stack(t8).float() if t8 else None,
                torch.stack(t4).float() if t4 else None))


def fingerprint():
    p = os.path.join(USS, "step4_our_new_model_resunet_boundary", "data",
                     "boundary_fingerprint.json")
    try:
        with open(p, encoding="utf-8") as fh:
            ap = json.load(fh).get("angular_profile", {})
        rel = [float(x) for x in (ap.get("edge_strength_rel") or [])]
        if not rel:
            return None, None
        m = sum(rel) / len(rel)
        inv = [m / max(1e-6, r) for r in rel]
        mi = sum(inv) / len(inv)
        return (sorted(float(x) for x in (ap.get("sector_deg_centers") or [])),
                [min(1.5, max(0.5, v / mi)) for v in inv])
    except Exception as e:
        print("[student] fingerprint unavailable (%s) - isotropic radii" % e)
        return None, None


# ----------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rc", type=float, default=3.0)
    ap.add_argument("--ra", type=float, default=4.0)
    ap.add_argument("--w-distill", type=float, default=1.0)
    ap.add_argument("--w-nest", type=float, default=0.5)
    ap.add_argument("--w-cautious", type=float, default=0.25)
    ap.add_argument("--w-amplified", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--widths", default="", help="e.g. 48,96,192,384 for the capacity run")
    ap.add_argument("--views", default="", help="e.g. 1,2,3 to train on the cached shifted views")
    ap.add_argument("--clean-prob", type=float, default=0.4)
    ap.add_argument("--views-root", default=os.path.join(ROOT, "results", "teacher_views"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    tag = args.tag or ("l2_student_%dep" % args.epochs)
    run = os.path.join(OUT, tag)
    os.makedirs(run, exist_ok=True)
    log = open(os.path.join(run, "train.log"), "w", buffering=1)

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); log.write(s + "\n")

    train_names, val_names = l0.eval_lib.read_split(l0.USS)
    if args.limit:
        train_names = train_names[:args.limit]

    angles, scale = fingerprint()
    targets = Targets(args.rc, args.ra, angles, scale)
    widths = tuple(int(v) for v in args.widths.split(",")) if args.widths else None
    net = StudentNet(widths=widths).to(dev)
    n_param = sum(p.numel() for p in net.parameters())
    # Training-only 1x1 projectors so cosine distillation can compare the
    # student's narrow taps with the teacher's 1024-d map.  They are NOT part of
    # the deployed student (the segmentation path never uses them).
    projs = nn.ModuleDict({"e3": nn.Conv2d(net.widths[3], 1024, 1, bias=False),
                           "e2": nn.Conv2d(net.widths[2], 1024, 1, bias=False)}).to(dev)
    n_proj = sum(p.numel() for p in projs.parameters())

    tfeat = {}
    for n in train_names:
        p = os.path.join(TEACHER, n.replace(".png", ".npy"))
        if os.path.exists(p):
            tfeat[n] = np.load(p, mmap_mode="r")
    P("[student] tag=%s epochs=%d batch=%d lr=%g device=%s" % (tag, args.epochs, args.batch, args.lr, args.device))
    P("[student] train=%d val=%d | teacher features cached for %d/%d train images"
      % (len(train_names), len(val_names), len(tfeat), len(train_names)))
    P("[student] fingerprint sectors=%s scale=%s"
      % (angles, [round(s, 2) for s in scale] if scale else None))
    P("[student] radii cautious=%.1f amplified=%.1f | w_distill=%g w_nest=%g w_caut=%.2f w_amp=%.2f"
      % (args.rc, args.ra, args.w_distill, args.w_nest, args.w_cautious, args.w_amplified))
    P("[student] params %.2f M deployable (+ %.2f M training-only projectors)  "
      "(UltraSAM3 teacher 841.7 M -> %.0fx smaller)"
      % (n_param / 1e6, n_proj / 1e6, 841.7e6 / n_param))
    if not tfeat:
        P("[student] WARNING: no cached teacher features -> distillation disabled; run 20_cache_teacher.py")

    opt = torch.optim.AdamW(list(net.parameters()) + list(projs.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (1 - e / max(1, args.epochs)) ** 0.9)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(args.seed)
    steps = max(1, len(train_names) // args.batch)
    t0 = time.time()
    views = tuple(int(v) for v in args.views.split(",") if v.strip()) if args.views else ()
    P("[student] views=%s clean_prob=%.2f (shifted views carry their OWN teacher map)"
      % (views or "none", args.clean_prob))
    it = batch_iter(train_names, rng, args.batch, targets, tfeat, views, args.clean_prob,
                    args.views_root)

    for ep in range(args.epochs):
        net.train()
        acc = {}
        for _ in range(steps):
            x, (yb, yc, ya), (t8, t4) = next(it)
            x = x.to(dev); yb, yc, ya = (t.to(dev) for t in (yb, yc, ya))
            if t8 is not None:
                t8, t4 = t8.to(dev), t4.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, taps = net(x)
                lg = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear", align_corners=False)
                l_base = dc_ce(lg[:, 0:2], yb)
                l_c = dc_ce(lg[:, 2:4], yc)
                l_a = dc_ce(lg[:, 4:6], ya)
                p_b = torch.softmax(lg[:, 0:2], 1)[:, 1]
                p_c = torch.softmax(lg[:, 2:4], 1)[:, 1]
                p_a = torch.softmax(lg[:, 4:6], 1)[:, 1]
                l_nest = (F.relu(p_c - p_b) + F.relu(p_b - p_a)).mean()
                loss = (l_base + args.w_cautious * l_c + args.w_amplified * l_a
                        + args.w_nest * l_nest) / (1 + args.w_cautious + args.w_amplified)
                l_d = x.new_zeros(())
                if t8 is not None and args.w_distill > 0:
                    l_d = 0.5 * (distill_cosine(projs["e3"](taps["e3"]), t8)
                                 + distill_cosine(projs["e2"](taps["e2"]), t4))
                    loss = loss + args.w_distill * l_d
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            for k, v in (("base", l_base), ("caut", l_c), ("amp", l_a), ("nest", l_nest), ("dist", l_d)):
                acc[k] = acc.get(k, 0.0) + float(v.detach())
        sched.step()
        if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs - 1:
            P("[student] ep %3d/%d  base %.4f caut %.4f amp %.4f nest %.5f dist %.4f  lr %.2e  %.1f min"
              % (ep, args.epochs, acc["base"] / steps, acc["caut"] / steps, acc["amp"] / steps,
                 acc["nest"] / steps, acc["dist"] / steps, opt.param_groups[0]["lr"],
                 (time.time() - t0) / 60))
        if ep % 25 == 0 or ep == args.epochs - 1:
            torch.save({"net": net.state_dict(), "projs": projs.state_dict(),
                        "widths": list(net.widths), "args": vars(args), "epoch": ep},
                       os.path.join(run, "ckpt.pt"))
    P("[student] done in %.1f min -> %s" % ((time.time() - t0) / 60, run))


if __name__ == "__main__":
    sys.exit(main())
