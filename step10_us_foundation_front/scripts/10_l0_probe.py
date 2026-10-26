#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/10_l0_probe.py

L0 "anchor health-check" for ultrasound foundation-model encoders.

WHAT IT ANSWERS (and what it deliberately does NOT)
---------------------------------------------------
A *relative* question between frozen encoders and acquisition conditions: does a
US-foundation-model encoder keep its features anchored on anatomy when the
acquisition changes (blur + speckle + gain, i.e. the pre-registered
`compound_B`), or does it drift onto the low-frequency gain field like the
ImageNet-pretrained front did (step4 armC: 95.83 in-domain -> 55.26 compound_B,
27/31 cases < 80)?

It is a ZERO-TRAINING probe: 5-NN over frozen features, reference set = the 71
fold-0 TRAIN images, queries = the 31 VAL images (in_domain + shift
conditions).  Absolute numbers are therefore NOT comparable with the step6 /
step8 segmentation table (no decoder, no fine-tuning, no TTA); they ARE
comparable across encoders and across conditions, which is the decision the
project needs now.

GEOMETRY (this is where the first version was wrong)
----------------------------------------------------
Every image (and its GT) is resized to one fixed CANVAS, default 1024x768 --
no padding, no per-image scale bookkeeping.  Cell (i,j) of the probe grid maps
to canvas pixel (i*stride+stride/2, j*stride+stride/2), and the prediction is
resampled from the grid back to the full canvas, so Dice / bdice@1px are
computed in canvas pixels by the SAME eval_lib + 57_shape_prior_ceiling code
as every other number in this project.  91 of the 102 cases are natively
1024x768, i.e. for them the canvas IS the original resolution; the remaining 11
are rescaled by 0.79-0.81.

usage:
  ~/anaconda3/envs/zk/bin/python scripts/10_l0_probe.py --model ultrafedfm \
      --device cuda:0 --conditions in_domain,compound_B [--stride 8] [--limit N]
"""
import argparse
import csv
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
WEIGHTS = os.path.join(ROOT, "weights")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))
import eval_lib                                                       # noqa: E402
s57 = importlib.import_module("57_shape_prior_ceiling")               # noqa: E402
from scipy import ndimage                                             # noqa: E402

IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
MSK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
SHIFT_DIR = os.path.join(USS, "step8_domain_shift", "data", "shift")

CANVAS = (1024, 768)                      # (W, H); overridable with --canvas
VALID = None                              # (nw, nh) image area inside the canvas; set per case


# --------------------------------------------------------------------- encoders
class EncoderBase(object):
    """Frozen encoder: PIL grayscale -> (C, gH, gW) feature grid on device."""

    def __init__(self, device, canvas=CANVAS):
        self.device = device
        self.canvas = canvas
        self.patch = 16

    norm = "zscore"                       # or "imagenet" (SAM-family encoders)

    def prepare(self, img):
        """Image -> (1,3,ch,cw) canvas tensor.  The image is resized to the
        canvas WIDTH and placed top-left; any remaining rows are zero padding
        (needed by SAM-family encoders, whose abs pos-embed is 1024x1024)."""
        global VALID
        cw, ch = self.canvas
        W, H = img.size
        nw, nh = cw, min(ch, max(1, int(round(H * cw / float(W)))))
        VALID = (nw, nh)
        small = img.resize((nw, nh), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        canvas = np.zeros((ch, cw), dtype=np.float32)
        canvas[:nh, :nw] = arr
        if self.norm == "imagenet":
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            rgb = np.repeat(canvas[..., None], 3, axis=2)
            rgb = (rgb - mean) / std
            x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        else:
            m = canvas[:nh, :nw]
            canvas[:nh, :nw] = (m - m.mean()) / (m.std() + 1e-6)       # per-image z-score
            x = torch.from_numpy(canvas)[None, None].repeat(1, 3, 1, 1)[0]
        return x[None].to(self.device)

    def encode(self, img):
        raise NotImplementedError


class TimmViTEncoder(EncoderBase):
    """MAE / DINO-style plain ViT through timm (patch 16 or 14)."""

    def __init__(self, device, ckpt, canvas=CANVAS, keyprefix="", tag="vit"):
        super().__init__(device, canvas)
        import re
        import timm
        self.tag = tag
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        for k in ("model", "state_dict", "module"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
        if keyprefix:
            sd = {k[len(keyprefix):] if k.startswith(keyprefix) else k: v for k, v in sd.items()}

        pe = sd.get("pos_embed")
        if pe is None:                                                 # CLIP naming
            pe = sd.get("visual.positional_embedding")
        if pe is None:
            raise SystemExit("no pos_embed in checkpoint - not a plain ViT")
        patch = sd["patch_embed.proj.weight"].shape[-1] if "patch_embed.proj.weight" in sd else \
            sd["visual.conv1.weight"].shape[-1]
        width = pe.shape[-1]

        def _depth(prefix):
            idx = {int(m.group(1)) for k in sd
                   for m in [re.match(re.escape(prefix) + r"(\d+)\.", k)] if m}
            return (max(idx) + 1) if idx else 0

        depth = _depth("blocks.") or _depth("visual.transformer.resblocks.")
        arch = {(12, 768): "vit_base", (24, 1024): "vit_large", (32, 1280): "vit_huge"}.get((depth, width))
        if arch is None:
            raise SystemExit("unrecognised ViT geometry: depth=%d width=%d" % (depth, width))
        n_pre = int(round((pe.shape[1] - 1) ** 0.5))
        self.patch = patch

        have = set(timm.list_models())
        name = next((c for c in ("%s_patch%d_224" % (arch, patch), "%s_patch%d" % (arch, patch)) if c in have), None)
        if name is None:
            raise SystemExit("timm has no %s_patch%d" % (arch, patch))
        # build at the checkpoint's own pretrain resolution so pos_embed matches,
        # then let dynamic_img_size interpolate it for our (non-square) canvas
        model = timm.create_model(name, pretrained=False, num_classes=0,
                                  img_size=n_pre * patch, dynamic_img_size=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(unexpected)
        print("[enc] %s: timm %s, %d/%d tensors loaded (%d missing, %d unexpected)"
              % (tag, name, n_loaded, len(sd), len(missing), len(unexpected)), flush=True)
        if n_loaded < 0.5 * len(sd):
            raise SystemExit("too few tensors matched (%d/%d) - wrong arch guess" % (n_loaded, len(sd)))
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, img):
        x = self.prepare(img)
        f = self.model.forward_features(x)
        if isinstance(f, dict):
            f = f.get("x") or f.get("last_hidden_state")
        f = f[:, 1:]                                                   # drop cls token
        cw, ch = self.canvas
        gh, gw = ch // self.patch, cw // self.patch
        f = f.transpose(1, 2).reshape(f.shape[0], -1, gh, gw)
        return f[0].float()


# path of the package's PARENT dir (inserting the package dir itself makes
# `import segment_anything` fail - the first version did exactly that)
SAM1_PATH = "/media/72TB_disk/personal/zk/SACM-SAM3"


class SAM1ViTEncoder(EncoderBase):
    """SAM ViT-B image encoder (SAM1 architecture, 1024x1024 abs pos-embed).

    Used for the ultrasound-adapted SAM checkpoints (UltraSam).  Their mmpretrain
    key naming is remapped onto the plain segment_anything module names; the
    checkpoint's own neck / decode heads are not needed for a feature probe, so
    the last transformer block's 768-d tokens are returned.
    """

    def __init__(self, device, ckpt, canvas, prefix="backbone.", tag=""):
        super().__init__(device, canvas)
        import re
        from functools import partial
        import torch.nn as nn
        if SAM1_PATH not in sys.path:
            sys.path.insert(0, SAM1_PATH)
        from segment_anything.modeling.image_encoder import ImageEncoderViT
        self.norm = "imagenet"
        self.patch = 16
        self.embed_dim = 768
        self.tag = tag
        vit = ImageEncoderViT(depth=12, embed_dim=768, img_size=canvas[1], mlp_ratio=4,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              global_attn_indexes=[2, 5, 8, 11], window_size=14,
                              out_chans=256, use_rel_pos=True, qkv_bias=True)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        mapped = {}
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            kk = k[len(prefix):]
            m = re.match(r"layers\.(\d+)\.(.*)", kk)
            if m:
                rest = (m.group(2).replace("ln1.", "norm1.").replace("ln2.", "norm2.")
                        .replace("ffn.layers.0.", "mlp.lin1.").replace("ffn.layers.1.", "mlp.lin2."))
                kk = "blocks.%s.%s" % (m.group(1), rest)
            kk = kk.replace("patch_embed.projection.", "patch_embed.proj.")
            mapped[kk] = v
        missing, unexpected = vit.load_state_dict(mapped, strict=False)
        n_ok = len(mapped) - len(unexpected)
        print("[enc] %s: SAM1 ViT-B, %d/%d tensors loaded, %d missing, %d unexpected"
              % (tag, n_ok, len(mapped), len(missing), len(unexpected)), flush=True)
        if n_ok < 100:
            raise SystemExit("SAM1 remap matched too few tensors (%d)" % n_ok)
        self.model = vit.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, img):
        x = self.prepare(img)
        m = self.model
        y = m.patch_embed(x)
        if m.pos_embed is not None:
            if hasattr(m, "get_abs_pos"):
                y = y + m.get_abs_pos(m.pos_embed, y.shape[1], y.shape[2])
            else:
                y = y + m.pos_embed
        for blk in m.blocks:
            y = blk(y)
        # SAM's blocks are channels-LAST (B,H,W,C) -- its own neck does the
        # permute before the conv neck, so we must do it too (otherwise the
        # "64" spatial grid is mistaken for 64 feature channels)
        C = self.embed_dim
        if y.ndim == 4 and y.shape[-1] == C and y.shape[1] != C:
            y = y.permute(0, 3, 1, 2)
        return y[0].float()




class SAM3ViTEncoder(EncoderBase):
    """SAM3 perception-encoder ViT (UltraSAM3's backbone).

    SAM3's ViT is a ViTDet trunk: patch 14, width 1024, depth 32, window 24 with
    global attention on blocks (7,15,23,31), RoPE (use_interp_rope) + tiled
    absolute pos-embed.  It is instantiated through sam3's own class with the
    exact kwargs its model_builder uses, and the checkpoint's
    `backbone.vision_backbone.trunk.*` tensors are loaded into it.
    """

    def __init__(self, device, ckpt, canvas, prefix="backbone.vision_backbone.trunk.",
                 tag="sam3"):
        super().__init__(device, canvas)
        import sys as _sys
        _sys.path.insert(0, "/home/ta/anaconda3/envs/zk/lib/python3.11/site-packages")
        from sam3.model.vitdet import ViT
        self.norm = "imagenet"
        self.patch = 14
        self.embed_dim = 1024
        self.tag = tag
        vit = ViT(img_size=canvas[1], pretrain_img_size=336, patch_size=14, embed_dim=1024,
                  depth=32, num_heads=16, mlp_ratio=4.625, norm_layer="LayerNorm",
                  drop_path_rate=0.0, qkv_bias=True, use_abs_pos=True, tile_abs_pos=True,
                  global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(), use_rope=True,
                  use_interp_rope=True, window_size=24, pretrain_use_cls_token=True,
                  retain_cls_token=False, ln_pre=True, ln_post=False,
                  return_interm_layers=False, bias_patch_embed=False,
                  use_act_checkpoint=False)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in ck["model"].items() if k.startswith(prefix)}
        sd = {k[len(prefix):]: v for k, v in sd.items()}
        missing, unexpected = vit.load_state_dict(sd, strict=False)
        n_ok = len(sd) - len(unexpected)
        print("[enc] %s: SAM3 ViT-Det (1024w/32L/p14), %d/%d tensors loaded, %d missing, %d unexpected"
              % (tag, n_ok, len(sd), len(missing), len(unexpected)), flush=True)
        if n_ok < 100:
            raise SystemExit("SAM3 remap matched too few tensors (%d)" % n_ok)
        self.model = vit.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, img):
        x = self.prepare(img)
        out = self.model(x)
        if isinstance(out, dict):
            out = out.get("x", list(out.values())[-1])
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if out.ndim == 4 and out.shape[-1] == self.embed_dim and out.shape[1] != self.embed_dim:
            out = out.permute(0, 3, 1, 2)          # channels-last -> channels-first
        return out[0].float()


def build_encoder(name, device, canvas):
    if name == "ultrasam":
        return SAM1ViTEncoder(device, os.path.join(WEIGHTS, "UltraSam.pth"), canvas,
                              prefix="backbone.", tag="ultrasam-SAM1-ViT-B/16 (US-43d)")
    if name == "ultrafedfm":
        return TimmViTEncoder(device, os.path.join(WEIGHTS, "UltraFedFM_checkpoint.pth"),
                              canvas, keyprefix="", tag="ultrafedfm-MAE-ViT-B/16")
    if name == "ultrasam3":
        return SAM3ViTEncoder(device, os.path.join(WEIGHTS, "UltraSAM3.pt"), canvas,
                              tag="ultrasam3-SAM3-ViT-Det (US-37d)")
    if name == "fetalclip":
        return TimmViTEncoder(device, os.path.join(WEIGHTS, "FetalCLIP_weights.pt"),
                              canvas, keyprefix="module.visual.", tag="fetalclip-CLIP-ViT-L/14")
    raise SystemExit("encoder %s not implemented yet" % name)


# -------------------------------------------------------------------- data + grid
def load_case(name, condition):
    """Image + GT, both resized to the canvas.  in_domain is read from the same
    shift tree that produced every step8 reference number, so probe and
    baselines see byte-identical inputs."""
    cand = [os.path.join(SHIFT_DIR, condition, "images", name),
            os.path.join(IMG_DIR, name)]
    ip = next((p for p in cand if os.path.exists(p)), None)
    if ip is None:
        raise SystemExit("no image for %s under condition %s" % (name, condition))
    img = Image.open(ip).convert("L")
    gt = eval_lib.load_binary(os.path.join(MSK_DIR, name)) > 0
    cw, ch = CANVAS
    W, H = img.size
    nw, nh = cw, min(ch, max(1, int(round(H * cw / float(W)))))
    gt_small = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)) > 127
    gt_c = np.zeros((ch, cw), dtype=bool)
    gt_c[:nh, :nw] = gt_small
    return img, gt_c


def grid_labels(gt_canvas, stride):
    """GT label at every probe-cell centre (canvas coordinates)."""
    cw, ch = CANVAS
    gh, gw = ch // stride, cw // stride
    yy = np.clip((np.arange(gh) * stride + stride // 2), 0, ch - 1)
    xx = np.clip((np.arange(gw) * stride + stride // 2), 0, cw - 1)
    return gt_canvas[np.ix_(yy, xx)]                                   # (gh, gw) bool


def valid_cells(stride):
    """Boolean mask of probe cells that lie inside the image (not the pad)."""
    cw, ch = CANVAS
    nw, nh = VALID if VALID else CANVAS
    gh, gw = ch // stride, cw // stride
    yy = np.arange(gh) * stride + stride // 2
    xx = np.arange(gw) * stride + stride // 2
    return (yy[:, None] < nh) & (xx[None, :] < nw)


def predict_mask(feat_ref, lab_ref, feat_q, stride, k=5, chunk=512):
    """5-NN in frozen-feature space -> binary mask at canvas resolution.

    feat_ref is (C, M) and lab_ref (M,) -- only cells that lie inside the image
    (not in the SAM-family padding) are in the bank.  Cells outside the image in
    the query are forced to background, so padding can never be predicted as
    foreground.  Similarity runs in fp16 in chunks (the bank is ~0.9 M cells).
    """
    cw, ch = CANVAS
    C = feat_q.shape[0]
    gh, gw = ch // stride, cw // stride
    q = F.interpolate(feat_q[None], size=(gh, gw), mode="bilinear", align_corners=False)[0]
    qf = q.reshape(C, -1).t()                                     # (gh*gw, C)
    vm = valid_cells(stride).reshape(-1)
    qv = F.normalize(qf[vm].to(feat_ref.device).float(), dim=1).half()
    R = F.normalize(feat_ref.t().float(), dim=1).half()           # (M, C)
    labs = lab_ref.to(qv.device)
    out = torch.empty(qv.shape[0], dtype=torch.bool, device=qv.device)
    for i in range(0, qv.shape[0], chunk):
        idx = (qv[i:i + chunk] @ R.t()).topk(k, dim=1).indices
        out[i:i + chunk] = labs[idx].float().mean(dim=1) > 0.5
    flat = np.zeros(gh * gw, dtype=bool)
    flat[vm] = out.cpu().numpy()
    grid = flat.reshape(gh, gw).astype(np.uint8) * 255
    return np.array(Image.fromarray(grid).resize((cw, ch), Image.NEAREST)) > 127


# ------------------------------------------------------------------------- main
def main():
    global CANVAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["ultrafedfm", "fetalclip", "ultrasam", "ultrasam3", "armc_imagenet"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conditions", default="in_domain,compound_B")
    ap.add_argument("--canvas", default="1024x768", help="WxH canvas, default 1024x768")
    ap.add_argument("--stride", type=int, default=8, help="probe grid stride (canvas px)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--layers", default="last", help="last | all  (reserved: multi-layer concat)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cw, ch = (int(v) for v in args.canvas.lower().split("x"))
    CANVAS = (cw, ch)
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    tag = args.tag or args.model
    print("[l0] model=%s tag=%s device=%s canvas=%dx%d stride=%d k=%d"
          % (args.model, tag, args.device, cw, ch, args.stride, args.k), flush=True)

    train_names, val_names = eval_lib.read_split(USS)
    if args.limit:
        val_names = val_names[:args.limit]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    enc = build_encoder(args.model, dev, CANVAS)

    # ---- reference bank: the 71 train images (unshifted) -------------------
    print("[l0] encoding %d reference (train) images ..." % len(train_names), flush=True)
    gh, gw = ch // args.stride, cw // args.stride
    feats, labs = [], []
    for i, n in enumerate(train_names):
        img, gt_c = load_case(n, "in_domain")
        f = enc.encode(img)                                       # prepare() sets VALID
        vm = valid_cells(args.stride)                              # (gh,gw) inside the image
        fi = F.interpolate(f[None], size=(gh, gw), mode="bilinear", align_corners=False)[0]
        feats.append(fi[:, vm])
        labs.append(torch.from_numpy(grid_labels(gt_c, args.stride)[vm].astype(np.float32)))
        if (i + 1) % 20 == 0:
            print("   %d/%d" % (i + 1, len(train_names)), flush=True)
    feat_ref = torch.cat(feats, dim=1).to(dev)                    # (C, M)
    lab_ref = torch.cat(labs).to(dev)                             # (M,)
    print("[l0] bank %s  labels %s  (train fg fraction %.4f)"
          % (tuple(feat_ref.shape), tuple(lab_ref.shape), float(lab_ref.mean())), flush=True)

    # ---- queries ----------------------------------------------------------
    rows = []
    for cond in conds:
        print("[l0] condition=%s queries=%d" % (cond, len(val_names)), flush=True)
        for n in val_names:
            img, gt_c = load_case(n, cond)
            pred = predict_mask(feat_ref, lab_ref, enc.encode(img), args.stride, k=args.k)
            m = eval_lib.all_metrics(pred, gt_c, None)
            pc = np.array(ndimage.center_of_mass(pred)) if pred.any() else np.array([np.nan, np.nan])
            gc = np.array(ndimage.center_of_mass(gt_c)) if gt_c.any() else np.array([np.nan, np.nan])
            rows.append({"model": tag, "condition": cond, "name": n,
                         "dice": float(m["dice"]), "iou": float(m["iou"]),
                         "recall": float(m["recall"]), "precision": float(m["precision"]),
                         "bdice1": float(s57.bdice(pred, gt_c, 1)),
                         "bdice4": float(s57.bdice(pred, gt_c, 4)),
                         "centroid_shift_px": float(np.linalg.norm(pc - gc))})
        sub = [r for r in rows if r["condition"] == cond]
        d = [r["dice"] for r in sub]
        print("   %-12s Dice %6.2f (min %6.2f) | bdice@1 %5.2f | bdice@4 %5.2f | centroid %5.1f px | <80 %d/%d"
              % (cond, np.mean(d), np.min(d), np.mean([r["bdice1"] for r in sub]),
                 np.mean([r["bdice4"] for r in sub]), np.mean([r["centroid_shift_px"] for r in sub]),
                 sum(x < 80 for x in d), len(d)), flush=True)

    out_csv = os.path.join(RESULTS, "l0_probe_%s.csv" % tag)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, "l0_probe_%s.json" % tag), "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "tag": tag, "canvas": list(CANVAS), "stride": args.stride,
                   "k": args.k, "n_ref": len(train_names),
                   "summary": {c: {kk: float(np.mean([r[kk] for r in rows if r["condition"] == c]))
                                   for kk in ("dice", "bdice1", "bdice4", "centroid_shift_px")}
                               for c in conds},
                   "rows": rows}, fh, indent=2, ensure_ascii=False)
    print("[l0] wrote %s" % out_csv)


if __name__ == "__main__":
    sys.exit(main())
