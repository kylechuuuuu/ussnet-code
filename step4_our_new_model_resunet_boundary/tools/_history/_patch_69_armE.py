#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add arm E (3-class zone model) to the ensemble evaluator.

Arm E's npz has THREE probability channels (bg / 127 / 255) instead of two, so the
hardcoded `probabilities[1]` would have silently averaged P(zone 127) -- a small
inner region -- instead of the union.  The loader is therefore made
class-count aware:  C == 3  ->  union = 1 - P(background);  otherwise channel 1.
Both branches give the same foreground probability for the 2-class arms.
"""
import io
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "69_ensemble_eval.py")
s = io.open(P, encoding="utf-8").read()

# ---- 1. class-count-aware foreground probability -------------------------
OLD_LOAD = '''        for k, d in MEMBERS.items():
            p = os.path.join(d, cid + ".npz")
            if os.path.exists(p):
                z = np.load(p)
                pa = np.squeeze(z["probabilities"][1]).astype(np.float32)
                assert pa.ndim == 2, (k, nm, pa.shape)
                prob[k][nm] = pa
            else:
                prob[k][nm] = None
'''
NEW_LOAD = '''        for k, d in MEMBERS.items():
            p = os.path.join(d, cid + ".npz")
            if os.path.exists(p):
                z = np.load(p)
                pr = z["probabilities"]
                if pr.ndim == 4 and pr.shape[0] >= 3:
                    # 3-class zone model: foreground = union = 1 - P(background)
                    pa = 1.0 - np.asarray(pr[0], dtype=np.float32)
                else:
                    pa = np.asarray(pr[1], dtype=np.float32)
                pa = np.squeeze(pa).astype(np.float32)
                assert pa.ndim == 2, (k, nm, pa.shape)
                prob[k][nm] = pa
            else:
                prob[k][nm] = None
'''
assert s.count(OLD_LOAD) == 1, "loader block not found"
s = s.replace(OLD_LOAD, NEW_LOAD)

# ---- 2. arm E members ----------------------------------------------------
OLD_M = '''    "Df": os.path.join(P, "fp_armD_preenc_r34_checkpoint_final_nifti"),
}'''
NEW_M = '''    "Df": os.path.join(P, "fp_armD_preenc_r34_checkpoint_final_nifti"),
    "Eb": os.path.join(P, "fp_armE_zone3_checkpoint_best_nifti"),
    "Ef": os.path.join(P, "fp_armE_zone3_checkpoint_final_nifti"),
}'''
assert s.count(OLD_M) == 1, "member block not found"
s = s.replace(OLD_M, NEW_M)

# ---- 3. arm E combos -----------------------------------------------------
OLD_C = '''    ("Ab+Af+Bb+Bf+Cb+Cf+Db+Df+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "nnB", "nnF"]),
]'''
NEW_C = '''    ("Ab+Af+Bb+Bf+Cb+Cf+Db+Df+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "nnB", "nnF"]),
    ("Eb", ["Eb"]),
    ("Ef", ["Ef"]),
    ("Eb+nnB", ["Eb", "nnB"]),
    ("Cb+Cf+Db+Df+Eb+Ef", ["Cb", "Cf", "Db", "Df", "Eb", "Ef"]),
    ("AB+CDE", ["Ab", "Bb", "Cb", "Db", "Eb"]),
    ("AB+CDE+nnB", ["Ab", "Bb", "Cb", "Db", "Eb", "nnB"]),
    ("all-arms-best+nnB",
     ["Ab", "Bb", "Cb", "Db", "Eb", "nnB"]),
    ("all-arms-both+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "Eb", "Ef",
      "nnB", "nnF"]),
]'''
assert s.count(OLD_C) == 1, "combo block not found"
s = s.replace(OLD_C, NEW_C)

io.open(P, "w", encoding="utf-8").write(s)
print("[armE-ens] patched tools/69_ensemble_eval.py -> %d lines" % (s.count("\n") + 1))
print("[armE-ens] members: %s" % ", ".join(
    ["nnB", "nnF", "Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "Eb", "Ef"]))
