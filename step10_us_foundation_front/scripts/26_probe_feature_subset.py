#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/26_probe_feature_subset.py

Compression FRONTIER, second axis: how much of the feature vector is needed?

Depth was measured first (results/teacher_truncation.json): 24 of 32 blocks keep
the full robustness (compound_B 91.02 vs 91.31) while 8 blocks collapse
(34.18).  This script measures WIDTH redundancy on top of the cheapest robust
setting (24 blocks): the probe is run with only the first C of the 1024 feature
channels.  If 256-512 channels suffice, the width can be cut by 2-4x on top of
the depth cut -- for free, i.e. without any retraining or distillation.

Reuses 10_l0_probe.py wholesale (same bank / metrics / JSON), substituting an
encoder whose feature map is sliced along the channel dimension.

usage: ~/anaconda3/envs/zk/bin/python scripts/26_probe_feature_subset.py \
          [--blocks 24] [--channels 128,256,512,768,1024] [--device cuda:0]
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

spec2 = importlib.util.spec_from_file_location("tr25", os.path.join(HERE, "25_probe_truncated_teacher.py"))
t25 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(t25)


class SubsetSAM3(t25.TruncatedSAM3):
    """Truncated trunk whose feature map is sliced to the first C channels."""

    def __init__(self, device, ckpt, canvas, k, channels):
        super().__init__(device, ckpt, canvas, k)
        self.channels = channels
        self.tag = "sam3-trunc%d-sub%d" % (k, channels)

    @torch.no_grad()
    def encode(self, img):
        f = super().encode(img)
        return f[:self.channels].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--channels", default="128,256,512,768,1024")
    args = ap.parse_args()
    cs = [int(v) for v in args.channels.split(",") if v.strip()]

    summary = {}
    for c in cs:
        print("\n" + "=" * 70)
        print("[subset] %d blocks, first %d feature channels" % (args.blocks, c), flush=True)
        l0.build_encoder = (lambda kk, cc: (lambda name, device, canvas:
                                           SubsetSAM3(device, CKPT, canvas, kk, cc)))(args.blocks, c)
        tag = "sam3_t%d_c%d" % (args.blocks, c)
        sys.argv = ["26", "--model", "ultrasam", "--device", args.device,
                    "--canvas", "1008x1008", "--conditions", "in_domain,compound_B",
                    "--stride", "8", "--k", "5", "--tag", tag]
        try:
            l0.main()
        except SystemExit:
            pass
        p = os.path.join(RES, "l0_probe_%s.json" % tag)
        if os.path.exists(p):
            summary[c] = json.load(open(p))["summary"]

    print("\n[subset] width frontier at %d blocks (5-NN probe, val 31):" % args.blocks)
    print("%9s %10s %10s %10s %10s" % ("channels", "in_domain", "bdice@1", "compound_B", "bdice@1"))
    for c, s in summary.items():
        print("%9d %10.2f %10.2f %10.2f %10.2f"
              % (c, s["in_domain"]["dice"], s["in_domain"]["bdice1"],
                 s["compound_B"]["dice"], s["compound_B"]["bdice1"]))
    out = os.path.join(RES, "feature_subset_%d.json" % args.blocks)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[subset] wrote %s" % out)


if __name__ == "__main__":
    sys.exit(main())
