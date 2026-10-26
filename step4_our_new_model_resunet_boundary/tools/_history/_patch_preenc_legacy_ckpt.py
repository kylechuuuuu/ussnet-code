#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tolerate legacy-format torchvision checkpoints in preenc_unet.py.
Idempotent."""
import io
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(os.path.dirname(TOOLS), "trainer", "preenc_unet.py")
src = io.open(TARGET, encoding="utf-8").read()
orig = src

OLD = '''            sd = torch.load(os.path.join(TORCH_CACHE, ckpt), map_location="cpu")'''
NEW = '''            try:
                sd = torch.load(os.path.join(TORCH_CACHE, ckpt), map_location="cpu")
            except RuntimeError:
                # resnet34-333f7ec4.pth is still in the LEGACY .tar format, which
                # torch >= 2.6 refuses to load with the default weights_only=True
                # ("Cannot use ``weights_only=True`` with files saved in the legacy
                # .tar format").  It is a trusted torchvision IMAGENET1K_V1 file
                # from the local cache, so falling back is safe.
                sd = torch.load(os.path.join(TORCH_CACHE, ckpt), map_location="cpu",
                                weights_only=False)'''

assert src.count(OLD) == 1, "torch.load anchor not found exactly once"
src = src.replace(OLD, NEW)

if src != orig:
    io.open(TARGET, "w", encoding="utf-8").write(src)
    print("[patch] updated %s" % TARGET)
else:
    print("[patch] already up to date")
