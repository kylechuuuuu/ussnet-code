#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/trainer/squeeze_net.py

The three-headed network of the squeeze-theorem (夹逼) boundary model, and the
inference-time "squeeze fusion" that turns the three heads into one boundary.

Network
-------
The architecture is UNTOUCHED stock nnU-Net (whatever the plans built for this
dataset: PlainConvUNet, 8 stages, 32..512, 2 convs/stage).  The only change is
the final segmentation convolution: instead of 2 output channels it emits 6,
i.e. three *stock binary heads* next to each other:

    channels [0:2]  main   (中值头: the estimate that is being pinched)
    channels [2:4]  inner  (内界:   supervised by erode(GT, r(t)))
    channels [4:6]  outer  (外界:   supervised by dilate(GT, r(t)))

Initialisation ties the three heads (inner gets a negative bias offset on its
foreground logit, outer a positive one), so the interval starts WIDE OPEN -
p_in < p_main < p_out everywhere - and the curriculum + gap term have to close
it back down onto the true boundary.  Starting from "one head" (the stock
model) would make the interval a point from the very first step and hide the
mechanism.

Inference fusion ("squeeze projection")
---------------------------------------
The network is called in four selectable modes:

    main          p_m                       (the plain single-head decision:
                                             this is the baseline-equivalent)
    median        median(p_in, p_m, p_out)  (the squeeze-consistent decision;
                                             = L2 projection of the three
                                             probabilities onto the monotone
                                             cone p_in <= p <= p_out)
    inner         p_in                      (the guaranteed-inside bound)
    outer         p_out                     (the guaranteed-outside complement)

The wrapper returns a 2-channel tensor [1-p, p] in those modes, i.e. exactly
what nnU-Net's stock predictor/export/resample machinery expects, so the whole
downstream pipeline (TTA, Gaussian sliding window, crop-back, NIfTI export) is
stock and the fused boundary is geometrically identical to a normal nnU-Net
prediction.
"""
import torch
import torch.nn as nn


def _is_multi_head(module: nn.Module) -> bool:
    return isinstance(module, SqueezeHeadUNet)


def tie_seg_heads(net: nn.Module, head_bias: float = 1.0):
    """Copy head-0 weights into heads 1 and 2 of every final segmentation conv.

    `head_bias` shifts the *foreground* logit of the inner head down and of the
    outer head up, which opens the interval at initialisation.
    """
    tied = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == 6 and m.kernel_size == (1, 1):
            with torch.no_grad():
                w = m.weight
                b = m.bias
                w[2:4].copy_(w[0:2])
                w[4:6].copy_(w[0:2])
                if b is not None:
                    b[2:4].copy_(b[0:2])
                    b[4:6].copy_(b[0:2])
                    b[3] = b[1] - float(head_bias)      # inner fg logit  -> smaller
                    b[5] = b[1] + float(head_bias)      # outer fg logit  -> bigger
                tied += 1
    if tied == 0:
        raise RuntimeError("tie_seg_heads: no 6-channel 1x1 segmentation conv found - "
                           "the architecture changed, the head layout assumption is broken")
    return tied


class SqueezeHeadUNet(nn.Module):
    """Stock nnU-Net network with a 3-head segmentation output + fusion."""

    MODES = ("train", "main", "median", "inner", "outer")

    def __init__(self, net: nn.Module, fusion: str = "train"):
        super().__init__()
        self.net = net
        self.fusion = fusion

    # nnU-Net's own helpers reach into `network.decoder` (deep supervision flag,
    # architecture probing).  Delegating keeps every stock code path working.
    @property
    def decoder(self):
        return self.net.decoder

    def set_fusion(self, mode: str):
        assert mode in self.MODES, "unknown fusion mode %r (choose from %s)" % (mode, self.MODES)
        self.fusion = mode
        return self

    # ------------------------------------------------------------------ core
    def forward(self, x):
        out = self.net(x)
        if self.fusion == "train":
            return out
        logits = out[0] if isinstance(out, (list, tuple)) else out
        # NOTE: the mode MUST be passed explicitly - fuse() has a default, and
        # forgetting it silently evaluates every rule as "median" (which equals
        # "main" whenever the ordering holds, so the four rules come out
        # byte-identical and the bug hides).
        return self.fuse(logits, self.fusion)

    @staticmethod
    def fuse(logits: torch.Tensor, mode: str = "median") -> torch.Tensor:
        """3-head logits -> 2-channel [1-p, p] logits used by the stock export path.

        The projection step is the discrete squeeze: the three probabilities are
        sorted, which is the L2-optimal projection of (p_in, p_m, p_out) onto the
        monotone cone p_in <= p <= p_out.  The middle value of the sorted triple
        is therefore "the boundary estimate after it has been pinched by both
        bounds": where the main head disagrees with both bounds, the bounds win.
        """
        p_m = torch.softmax(logits[:, 0:2], dim=1)[:, 1:2]
        p_in = torch.softmax(logits[:, 2:4], dim=1)[:, 1:2]
        p_out = torch.softmax(logits[:, 4:6], dim=1)[:, 1:2]

        if mode == "main":
            p = p_m
        elif mode == "inner":
            p = p_in
        elif mode == "outer":
            p = p_out
        else:                                     # median (squeeze projection)
            stacked = torch.cat([p_in, p_m, p_out], dim=1)
            p = torch.sort(stacked, dim=1).values[:, 1:2]

        eps = 1e-6
        p = p.clamp(eps, 1.0 - eps)
        fused_logit = torch.log(p) - torch.log1p(-p)
        return torch.cat([torch.zeros_like(fused_logit), fused_logit], dim=1)
