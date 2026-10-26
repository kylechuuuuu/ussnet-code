#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add arm C to scripts/54_predict_and_evaluate.sh.  Idempotent."""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "scripts", "54_predict_and_evaluate.sh")
src = io.open(TARGET, encoding="utf-8").read()
orig = src

OLD_T = '''        B) echo "nnUNetTrainerProstateResUNetFPBoundary" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;'''
NEW_T = '''        B) echo "nnUNetTrainerProstateResUNetFPBoundary" ;;
        C) echo "nnUNetTrainerProstateFPPreEnc" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;'''
assert src.count(OLD_T) == 1, "trainer case not found exactly once"
src = src.replace(OLD_T, NEW_T)

OLD_S = '''        B) echo "fpB_resenc_fploss_dir" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;'''
NEW_S = '''        B) echo "fpB_resenc_fploss_dir" ;;
        C) echo "fp_armC_preenc" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;'''
assert src.count(OLD_S) == 1, "short-name case not found exactly once"
src = src.replace(OLD_S, NEW_S)

OLD_U = '''#            arms   default "A B"          (A = stock arch + fp loss,
#                                           B = ResEnc + fp loss + direction)'''
NEW_U = '''#            arms   default "A B"          (A = stock arch + fp loss,
#                                           B = ResEnc + fp loss + direction,
#                                           C = ImageNet-pretrained encoder)'''
assert src.count(OLD_U) == 1, "usage block not found exactly once"
src = src.replace(OLD_U, NEW_U)

if src != orig:
    io.open(TARGET, "w", encoding="utf-8").write(src)
    print("[patch] updated %s" % TARGET)
else:
    print("[patch] already up to date")
