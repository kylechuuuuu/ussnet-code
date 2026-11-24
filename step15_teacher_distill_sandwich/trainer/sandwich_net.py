#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/trainer/sandwich_net.py

The three-version network of the sandwich squeeze model.

The architecture is UNTOUCHED stock nnU-Net (whatever the plans built for this
dataset); ONLY the final segmentation convolution emits 6 channels instead of
2, i.e. three stock binary heads next to each other:

    [0:2]  base      (基础版本)   target = GT
    [2:4]  cautious  (谨慎版本)   target = erode(GT, 3 px)
    [4:6]  amplified (放大版本)   target = dilate(GT, 4 px)

At initialisation the three heads are tied with a logit offset, so the three
versions start nested (cautious inside, amplified outside).

`forward_mode` decides what a forward pass returns:

    "all"        the raw 6-channel deep-supervision output     (training)
    "base"       logits[:, 0:2]                                (prediction)
    "cautious"   logits[:, 2:4]
    "amplified"  logits[:, 4:6]

The last three return a normal 2-channel [background, foreground] map, exactly
what nnU-Net's stock predictor / TTA / export machinery expects, so the three
versions are produced by the SAME stock sliding-window inference and only the
head changes (tools/62_predict_sandwich.py runs it once per head).

The nesting used by the boundary arbitration is enforced by construction in
tools/62 + tools/63 (intersection/union with the base version); the network
itself carries no ordering constraint and needs none.
"""
import torch
import torch.nn as nn


def tie_seg_heads(net: nn.Module, head_bias: float = 1.0):
    """Copy head-0 weights into heads 1 and 2 of every final segmentation conv.

    head_bias shifts the cautious foreground logit DOWN and the amplified one
    UP, so the three versions start strictly nested.
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
                    b[3] = b[1] - float(head_bias)      # cautious  fg logit -> smaller
                    b[5] = b[1] + float(head_bias)      # amplified fg logit -> bigger
                tied += 1
    if tied == 0:
        raise RuntimeError("tie_seg_heads: no 6-channel 1x1 segmentation conv found - "
                           "the architecture changed, the head layout assumption is broken")
    return tied


class SandwichHeadUNet(nn.Module):
    """Stock nnU-Net network with a 3-version segmentation output."""

    MODES = ("all", "base", "cautious", "amplified")

    def __init__(self, net: nn.Module, forward_mode: str = "base"):
        super().__init__()
        self.net = net
        self.forward_mode = forward_mode

    # nnU-Net's helpers reach into `network.decoder` (deep-supervision flag,
    # architecture probing); delegating keeps every stock code path working.
    @property
    def decoder(self):
        return self.net.decoder

    def set_forward_mode(self, mode: str):
        assert mode in self.MODES, "unknown forward mode %r (choose from %s)" % (mode, self.MODES)
        self.forward_mode = mode
        return self

    def forward(self, x):
        out = self.net(x)
        if self.forward_mode == "all":
            return out
        logits = out[0] if isinstance(out, (list, tuple)) else out
        if self.forward_mode == "cautious":
            return logits[:, 2:4]
        if self.forward_mode == "amplified":
            return logits[:, 4:6]
        return logits[:, 0:2]
