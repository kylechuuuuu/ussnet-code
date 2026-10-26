#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add arm C members/combos to tools/69_ensemble_eval.py, and let the output
path be overridden with ENSEMBLE_OUT.  Idempotent."""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "tools", "69_ensemble_eval.py")
src = io.open(TARGET, encoding="utf-8").read()
orig = src

# ---- 1. output path override ---------------------------------------------
OLD_OUT = 'OUT = os.path.join(ROOT, "results", "ensemble_eval.json")'
NEW_OUT = ('OUT = os.environ.get("ENSEMBLE_OUT",\n'
           '                    os.path.join(ROOT, "results", "ensemble_eval.json"))')
if OLD_OUT in src:
    assert src.count(OLD_OUT) == 1
    src = src.replace(OLD_OUT, NEW_OUT)

# ---- 2. arm C members ----------------------------------------------------
OLD_M = '''    "Bf": os.path.join(P, "fpB_resenc_fploss_dir_checkpoint_final_nifti"),
}'''
NEW_M = '''    "Bf": os.path.join(P, "fpB_resenc_fploss_dir_checkpoint_final_nifti"),
    "Cb": os.path.join(P, "fp_armC_preenc_checkpoint_best_nifti"),
    "Cf": os.path.join(P, "fp_armC_preenc_checkpoint_final_nifti"),
}'''
if OLD_M in src:
    assert src.count(OLD_M) == 1
    src = src.replace(OLD_M, NEW_M)

# ---- 3. arm C combos -----------------------------------------------------
NEW_COMBOS = '''    ("Cb", ["Cb"]),
    ("Cf", ["Cf"]),
    ("Ab+Bb+Cb", ["Ab", "Bb", "Cb"]),
    ("Ab+Af+Bb+Bf+Cb+Cf", ["Ab", "Af", "Bb", "Bf", "Cb", "Cf"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+nnB", ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "nnB"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "nnB", "nnF"]),
]'''
OLD_COMBOS = '''    ("Ab+Af+Bb+Bf+nnB+nnF", ["Ab", "Af", "Bb", "Bf", "nnB", "nnF"]),
]'''
assert src.count(OLD_COMBOS) == 1, "combo tail not found exactly once"
src = src.replace(OLD_COMBOS, NEW_COMBOS)

if src != orig:
    io.open(TARGET, "w", encoding="utf-8").write(src)
    print("[patch] updated %s" % TARGET)
else:
    print("[patch] already up to date")
