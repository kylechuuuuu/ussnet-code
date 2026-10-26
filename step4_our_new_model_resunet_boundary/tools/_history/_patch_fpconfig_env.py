#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make fp_config.py treat an EMPTY env var as unset.  Idempotent.

Why: scripts/env_step4.sh exports the ablation knobs as empty strings
(`export STEP4_W_TOL="${STEP4_W_TOL:-}"`), so `os.environ.get(name, default)`
returns `` rather than the default and `float('')` raises
`ValueError: could not convert string to float: ''`.
"""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "trainer", "fp_config.py")
src = io.open(TARGET, encoding="utf-8").read()
orig = src

HELPER = '''DEFAULT_JSON = os.path.join(STEP4_ROOT, "data", "boundary_fingerprint.json")


def _env(name, default):
    """Env override that treats UNSET and EMPTY identically.

    scripts/env_step4.sh deliberately exports the ablation knobs as empty
    strings (`export STEP4_W_TOL="${STEP4_W_TOL:-}"`), so the two-argument
    `os.environ.get(name, default)` form would return `` (not `default`) and
    `float("")` raises ValueError.  Restore the intended "empty means unset".
    """
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val
'''

OLD_CONST = '''DEFAULT_JSON = os.path.join(STEP4_ROOT, "data", "boundary_fingerprint.json")
'''
assert src.count(OLD_CONST) == 1, "DEFAULT_JSON anchor not found exactly once"
src = src.replace(OLD_CONST, HELPER)

REPL = [
    ('''        d["weight_tolerant_bdice"] = float(os.environ.get(
            "STEP4_W_TOL", d.get("weight_tolerant_bdice", 0.5)))
        d["weight_boundary_bce"] = float(os.environ.get(
            "STEP4_W_BCE", d.get("weight_boundary_bce", 0.2)))
        d["weight_perimeter"] = float(os.environ.get(
            "STEP4_W_PER", d.get("weight_perimeter", 0.05)))
        d["tolerant_band"] = int(os.environ.get(
            "STEP4_TOL_BAND", d.get("tolerant_band", 7)))
        d["warmup_epochs"] = int(os.environ.get(
            "STEP4_WARMUP", d.get("warmup_epochs", 100)))
        self.use_direction = int(os.environ.get("STEP4_USE_DIRECTION", 0)) == 1
        self.disabled = int(os.environ.get("STEP4_DISABLE_EXTRA", 0)) == 1''',
     '''        d["weight_tolerant_bdice"] = float(_env(
            "STEP4_W_TOL", d.get("weight_tolerant_bdice", 0.5)))
        d["weight_boundary_bce"] = float(_env(
            "STEP4_W_BCE", d.get("weight_boundary_bce", 0.2)))
        d["weight_perimeter"] = float(_env(
            "STEP4_W_PER", d.get("weight_perimeter", 0.05)))
        d["tolerant_band"] = int(_env(
            "STEP4_TOL_BAND", d.get("tolerant_band", 7)))
        d["warmup_epochs"] = int(_env(
            "STEP4_WARMUP", d.get("warmup_epochs", 100)))
        self.use_direction = int(_env("STEP4_USE_DIRECTION", 0)) == 1
        self.disabled = int(_env("STEP4_DISABLE_EXTRA", 0)) == 1'''),
]
for old, new in REPL:
    assert src.count(old) == 1, "override block not found exactly once"
    src = src.replace(old, new)

if src != orig:
    io.open(TARGET, "w", encoding="utf-8").write(src)
    print("[patch] updated %s" % TARGET)
else:
    print("[patch] already up to date")
