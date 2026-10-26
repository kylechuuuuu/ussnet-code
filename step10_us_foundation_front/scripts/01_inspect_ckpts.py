#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/01_inspect_ckpts.py

CPU-only.  Before any feature can be extracted, we need to know what each
downloaded checkpoint actually IS: state_dict vs wrapper, top-level keys, tensor
shapes, parameter count, dtype, and which local package could load it.

Run this on any subset of weights/ that has finished downloading:
    ~/anaconda3/envs/zk/bin/python scripts/01_inspect_ckpts.py
writes results/ckpt_inventory.json + prints a table.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
W = os.path.join(ROOT, "weights")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def describe(obj, depth=0, max_depth=2):
    """Compact shape of a nested checkpoint object."""
    if isinstance(obj, torch.Tensor):
        return "Tensor%s %s" % (tuple(obj.shape), obj.dtype)
    if isinstance(obj, dict):
        if depth >= max_depth:
            return "dict(%d keys)" % len(obj)
        keys = list(obj.keys())[:8]
        inner = ", ".join("%s: %s" % (k, describe(obj[k], depth + 1, max_depth)) for k in keys)
        more = "" if len(obj) == len(keys) else ", ... +%d" % (len(obj) - len(keys))
        return "{%s%s}" % (inner, more)
    if isinstance(obj, (list, tuple)):
        return "%s[%d] of %s" % (type(obj).__name__, len(obj),
                                 describe(obj[0], depth + 1, max_depth) if obj else "?")
    return "%s(%r)" % (type(obj).__name__, str(obj)[:60])


def state_dict_of(ckpt):
    """Find the biggest flat tensor dict inside the checkpoint.

    Walks only the *model* subtrees: optimizer / scaler states are large, not
    model parameters, and their nested int keys are what made an earlier version
    of this walker crash.
    """
    SKIP = ("optimizer", "optim", "scaler", "scheduler", "amp")
    best = None
    stack = [("", ckpt)]
    while stack:
        pre, node = stack.pop()
        if isinstance(node, dict):
            tens = {k: v for k, v in node.items() if isinstance(v, torch.Tensor)}
            if len(tens) >= 10 and (best is None or len(tens) > len(best[1])):
                best = (pre, tens)
            for k, v in node.items():
                if isinstance(k, str) and k.lower() in SKIP:
                    continue
                if isinstance(v, (dict, list, tuple)):
                    stack.append((pre + str(k) + ".", v))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                if isinstance(v, (dict, list, tuple)):
                    stack.append((pre + str(i) + ".", v))
    return best


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    rows = []
    for name in sorted(os.listdir(W)):
        if not name.endswith((".pth", ".pt")):
            continue
        if only and not any(o in name for o in only):
            continue
        path = os.path.join(W, name)
        size = os.path.getsize(path)
        print("=" * 70, flush=True)
        print("[inspect] %s  (%.1f MB)" % (name, size / 1e6), flush=True)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print("  LOAD FAILED: %s: %s" % (type(e).__name__, e), flush=True)
            rows.append({"file": name, "size": size, "error": "%s: %s" % (type(e).__name__, e)})
            continue

        print("  top level: %s" % describe(ckpt, 0, 1), flush=True)
        sd = state_dict_of(ckpt)
        row = {"file": name, "size": size, "top_level": describe(ckpt, 0, 1)}
        if sd:
            pre, tens = sd
            n_param = sum(v.numel() for v in tens.values())
            dtypes = sorted({str(v.dtype) for v in tens.values()})
            keys = list(tens.keys())
            row.update({
                "state_dict_prefix": pre,
                "n_tensors": len(tens),
                "n_params": n_param,
                "dtypes": dtypes,
                "first_keys": keys[:15],
                "last_keys": keys[-8:],
            })
            print("  state_dict @ %r : %d tensors, %.2f M params, dtypes %s"
                  % (pre or "<root>", len(tens), n_param / 1e6, dtypes), flush=True)
            print("  first keys:", flush=True)
            for k in keys[:12]:
                print("     %-70s %s" % (k[:70], tuple(tens[k].shape)), flush=True)
        rows.append(row)
        del ckpt
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    with open(os.path.join(OUT, "ckpt_inventory.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print("\n[inspect] wrote %s" % os.path.join(OUT, "ckpt_inventory.json"))


if __name__ == "__main__":
    sys.exit(main())
