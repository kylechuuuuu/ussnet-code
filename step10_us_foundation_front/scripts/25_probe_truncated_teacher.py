#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/25_probe_truncated_teacher.py

"How much of the teacher is actually the core content?"

Run the L0 probe (5-NN over frozen features, val 31, in_domain + compound_B)
with the SAM3 trunk TRUNCATED to the first K transformer blocks, for K in
{8, 16, 24, 32}.  This gives a capacity curve for the teacher's features and
tells us whether the compression target should be "a smaller network distilled
from it" or "a pruned version of the trunk itself" (+ a small decoder).

It reuses 10_l0_probe.py wholesale (same bank, same metrics, same JSON format)
by substituting the encoder, so the numbers are directly comparable with
results/l0_probe_ultrasam3.json.

usage: ~/anaconda3/envs/zk/bin/python scripts/25_probe_truncated_teacher.py [--device cuda:0]
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "weights", "UltraSAM3.pt")
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location("l0", os.path.join(HERE, "10_l0_probe.py"))
l0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l0)


class TruncatedSAM3(l0.SAM3ViTEncoder):
    """SAM3 ViT-Det with only the first k blocks (RoPE/tiled pos-embed intact)."""

    def __init__(self, device, ckpt, canvas, k):
        super().__init__(device, ckpt, canvas, tag="sam3-trunc%d" % k)
        self.k = k
        n_blk = sum(p.numel() for b in self.model.blocks[:k] for p in b.parameters())
        print("[trunc] k=%d/%d blocks, block params %.1f M" %
              (k, len(self.model.blocks), n_blk / 1e6), flush=True)

    @torch.no_grad()
    def encode(self, img):
        # Reuse the parent's forward verbatim (it knows how SAM3's tiled
        # pos-embed / RoPE / window bookkeeping works); the only change is that
        # the model sees a shortened block list, so we never hand-roll the
        # pos-embed arithmetic ourselves.
        # SAM3's ViT.forward only EMITS a feature map at full_attn_ids[-1]
        # (block 31) or at the registered global-attention indices; with a
        # shortened block list that index never occurs and the output list is
        # empty.  So the last block of the truncated stack must become the
        # emitting block.  Our k values (8/16/24/32) all land on a global
        # attention block (7/15/23/31), which is exactly what SAM3 emits.
        m = self.model
        saved_blocks, saved_ids = m.blocks, m.full_attn_ids
        m.blocks = saved_blocks[:self.k]
        m.full_attn_ids = [self.k - 1]
        try:
            return super().encode(img)
        finally:
            m.blocks, m.full_attn_ids = saved_blocks, saved_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blocks", default="8,16,24,32")
    args = ap.parse_args()
    ks = [int(v) for v in args.blocks.split(",") if v.strip()]

    summary = {}
    for k in ks:
        print("\n" + "=" * 70)
        print("[trunc] probing with %d blocks" % k, flush=True)
        l0.build_encoder = (lambda kk: (lambda name, device, canvas:
                                        TruncatedSAM3(device, CKPT, canvas, kk)))(k)
        tag = "sam3_trunc%d" % k
        sys.argv = ["25", "--model", "ultrasam", "--device", args.device,
                    "--canvas", "1008x1008", "--conditions", "in_domain,compound_B",
                    "--stride", "8", "--k", "5", "--tag", tag]
        try:
            l0.main()
        except SystemExit:
            pass
        p = os.path.join(RES, "l0_probe_%s.json" % tag)
        if os.path.exists(p):
            summary[k] = json.load(open(p))["summary"]

    print("\n[trunc] capacity curve (5-NN probe, val 31):")
    print("%6s %10s %10s %10s %10s" % ("blocks", "in_domain", "bdice@1", "compound_B", "bdice@1"))
    for k, s in summary.items():
        print("%6d %10.2f %10.2f %10.2f %10.2f"
              % (k, s["in_domain"]["dice"], s["in_domain"]["bdice1"],
                 s["compound_B"]["dice"], s["compound_B"]["bdice1"]))
    with open(os.path.join(RES, "teacher_truncation.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[trunc] wrote %s" % os.path.join(RES, "teacher_truncation.json"))


if __name__ == "__main__":
    sys.exit(main())
