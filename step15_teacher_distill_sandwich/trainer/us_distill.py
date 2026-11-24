#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step15_teacher_distill_sandwich/trainer/us_distill.py

Feature-level distillation from a FROZEN ultrasound foundation model into the
sandwich U-Net.  There is no student network: the sandwich itself is the only
trainable model, and the teacher only supplies dense features.

Two uses of the teacher, both inside the sandwich forward pass:

  1. FiLM modulation at encoder stage 4 (stride 16):
         h <- h * (1 + tanh(gamma(proj(f)))) + beta(proj(f))
     gamma/beta are zero-initialised, so the model starts as the baseline.

  2. An auxiliary 1x1 head predicts the teacher's PCA-16 feature map from the
     same h; the trainer adds  1 - cosine(pred, target)  to the sandwich loss.

The teacher runs online (`torch.no_grad`, eval mode).  Nothing is cached, no
second network is trained, nothing is concatenated to the input.

environment:
    US_FM_CHECKPOINT   path to the UltraFedFM MAE ViT-B/16 checkpoint (required)
    US_DISTILL_STATS   path to prior_stats.npz fitted by 01_fit_prior_stats.py
                       (default: ./prior_stats.npz)
"""
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

FM_PATH = os.environ.get("US_FM_CHECKPOINT", "")
STATS_PATH = os.environ.get("US_DISTILL_STATS", "prior_stats.npz")


class FrozenTeacher(nn.Module):
    """UltraFedFM MAE ViT-B/16, frozen; returns a stride-16 feature map."""

    def __init__(self, path: str = None, img_size: int = 224):
        super().__init__()
        path = path or FM_PATH
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                "UltraFedFM checkpoint not found. Set US_FM_CHECKPOINT to the "
                "UltraFedFM_checkpoint.pth path (it is not part of this repo).")

        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0,
            img_size=img_size, dynamic_img_size=True, class_token=True, global_pool="")
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        sd = {k: v for k, v in sd.items()
              if not k.startswith("decoder") and k != "mask_token"}
        msg = self.vit.load_state_dict(sd, strict=False)
        print("[distill] teacher load:", msg, flush=True)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        self.dim = 768

    def train(self, mode: bool = True):
        # stay frozen/eval no matter what the trainer does
        super().train(False)
        self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w = x.shape[0], x.shape[-2], x.shape[-1]
        ph = (16 - h % 16) % 16                      # ViT-B/16 needs /16 sizes
        pw = (16 - w % 16) % 16
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        with torch.no_grad():
            xf = xp.reshape(b, -1)
            mn = xf.min(1, keepdim=True)[0].reshape(b, 1, 1, 1)
            mx = xf.max(1, keepdim=True)[0].reshape(b, 1, 1, 1)
            xn = (xp - mn) / (mx - mn + 1e-6)        # per-image min-max
            xn = xn.repeat(1, 3, 1, 1)
            xn = (xn - 0.5) / 0.5                    # [-1, 1]
            f = self.vit.forward_features(xn)
            if isinstance(f, (list, tuple)):
                f = f[0]
            f = f[:, 1:]                             # drop cls token
            gh, gw = xp.shape[2] // 16, xp.shape[3] // 16
            if f.shape[1] != gh * gw:                # safety for odd sizes
                s = int(round(f.shape[1] ** 0.5))
                gh = gw = s
            f = f.reshape(b, gh, gw, -1).permute(0, 3, 1, 2).contiguous()
        return f


def load_pca_stats(path: str = None, dim: int = 16):
    """Load the frozen PCA basis / std fitted on the training split."""
    path = path or STATS_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "prior_stats.npz not found: %s\n"
            "run scripts/01_fit_prior_stats.py on the TRAIN split first." % path)
    z = np.load(path)
    return (torch.from_numpy(z["mean"]).float(),
            torch.from_numpy(z["components"][:dim]).float(),
            torch.from_numpy(z["std"][:dim]).float().clamp_min(1e-6))


class DistillUNet(nn.Module):
    """Stock nnU-Net encoder/decoder + frozen teacher FiLM + aux PCA-16 head."""

    def __init__(self, base: nn.Module, inject_stage: int = 4, aux_dim: int = 16,
                 stats_path: str = None):
        super().__init__()
        self.base = base
        self.inject_stage = inject_stage
        self.teacher = FrozenTeacher()

        mean, comp, std = load_pca_stats(stats_path, aux_dim)
        self.register_buffer("pca_mean", mean)
        self.register_buffer("pca_comp", comp)
        self.register_buffer("pca_std", std)

        c = int(base.encoder.output_channels[inject_stage])
        self.proj = nn.Conv2d(self.teacher.dim, c, 1)
        self.film_g = nn.Conv2d(c, c, 1)
        self.film_b = nn.Conv2d(c, c, 1)
        for m in (self.film_g, self.film_b):         # identity at init
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.aux_head = nn.Conv2d(c, int(aux_dim), 1)
        self._aux = None

    # nnU-Net helpers reach into `network.decoder`; delegate to the base net
    @property
    def decoder(self):
        return self.base.decoder

    def compute_conv_feature_map_size(self, input_size):
        return self.base.compute_conv_feature_map_size(input_size)

    def teacher_pca(self, f: torch.Tensor) -> torch.Tensor:
        fv = f.permute(0, 2, 3, 1)
        p = torch.matmul(fv - self.pca_mean, self.pca_comp.t()) / self.pca_std
        return p.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor):
        self._aux = None
        skips = []
        h = x
        for i, stage in enumerate(self.base.encoder.stages):
            h = stage(h)
            if i == self.inject_stage:
                f = self.teacher(x)
                if f.shape[-2:] != h.shape[-2:]:
                    f = F.interpolate(f, size=h.shape[-2:], mode="bilinear",
                                      align_corners=False)
                p = self.proj(f)
                h = h * (1.0 + torch.tanh(self.film_g(p))) + self.film_b(p)
                if self.training:                    # aux loss is training-only
                    tgt = self.teacher_pca(f)
                    if tgt.shape[-2:] != h.shape[-2:]:
                        tgt = F.interpolate(tgt, size=h.shape[-2:], mode="bilinear",
                                            align_corners=False)
                    self._aux = (self.aux_head(h), tgt.detach())
            skips.append(h)
        return self.base.decoder(skips)
