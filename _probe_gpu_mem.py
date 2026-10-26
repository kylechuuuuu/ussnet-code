#!/usr/bin/env python3
"""Probe the peak GPU memory of the step6 sandwich networks at the plans' patch/batch.

Read-only w.r.t. the training pipeline: builds the network, runs one forward+backward
at patch 640x1024 batch 4, prints peak allocated/reserved.  Used to decide whether a
training run fits next to whatever else occupies the card.
"""
import json
import os
import sys

import torch

PRE = "/media/72TB_disk/personal/zk/USSnet/step1_nnunet_baseline/data/nnUNet_preprocessed/Dataset001_UltrasoundProstate/nnUNetPlans.json"
plans = json.load(open(PRE))
c = plans["configurations"]["2d"]
arch = c["architecture"]
patch = tuple(c["patch_size"])
batch = c["batch_size"]
n_stages = arch["arch_kwargs"]["n_stages"]

sys.path.insert(0, "/media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze/trainer")
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans  # noqa: E402
from sandwich_net import SandwichHeadUNet, tie_seg_heads                     # noqa: E402


def build(kind, out_ch):
    kw = json.loads(json.dumps(arch["arch_kwargs"]))
    cls = arch["network_class_name"]
    req = ["conv_op", "norm_op", "dropout_op", "nonlin"]
    if kind == "resenc":
        cls = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
        n_blocks = [1, 2, 2] + [3] * max(0, n_stages - 3)
        kw.pop("n_conv_per_stage", None)
        kw["n_blocks_per_stage"] = tuple(n_blocks[:n_stages])
    net = get_network_from_plans(cls, kw, req, 1, out_ch, allow_init=True,
                                 deep_supervision=True)
    tie_seg_heads(net, head_bias=1.0)
    return SandwichHeadUNet(net, forward_mode="all")


def probe(kind, out_ch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = build(kind, out_ch).cuda()
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    x = torch.randn(batch, 1, *patch, device="cuda")
    y = (torch.rand(batch, 1, *patch, device="cuda") > 0.5).float()
    out = net(x)
    outs = out if isinstance(out, (list, tuple)) else [out]
    loss = sum((o[:, 0:2] ** 2).mean() for o in outs)
    loss = loss + sum((o.sum() * 0.0) for o in outs)
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    res = torch.cuda.max_memory_reserved() / 2 ** 20
    print("%-8s out=%d  batch=%d patch=%s  peak_alloc %.0f MiB  peak_reserved %.0f MiB"
          % (kind, out_ch, batch, patch, peak, res))
    del net, opt, x, y, out, loss
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("free/total:", [int(v / 2 ** 20) for v in torch.cuda.mem_get_info()])
    probe("plain", 6)
    probe("resenc", 6)
    print("free/total after:", [int(v / 2 ** 20) for v in torch.cuda.mem_get_info()])
