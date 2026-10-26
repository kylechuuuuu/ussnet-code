#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add arm C (pretrained-encoder UNet) to scripts/51_train_fp.sh.  Idempotent."""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "scripts", "51_train_fp.sh")
src = io.open(TARGET, encoding="utf-8").read()
orig = src

OLD = """  B) TRAINER="nnUNetTrainerProstateResUNetFPBoundary"; TAG="fp_armB_resenc_dir";  USE_DIR=1 ;;
  *) echo "[51] arm must be A or B (got '$ARM')" >&2; exit 2 ;;"""
NEW = """  B) TRAINER="nnUNetTrainerProstateResUNetFPBoundary"; TAG="fp_armB_resenc_dir";  USE_DIR=1 ;;
  C) TRAINER="nnUNetTrainerProstateFPPreEnc";          TAG="fp_armC_preenc";      USE_DIR=1 ;;
  *) echo "[51] arm must be A, B or C (got '$ARM')" >&2; exit 2 ;;"""
assert src.count(OLD) == 1, "case block not found exactly once"
src = src.replace(OLD, NEW)

OLD2 = """#     B  ResidualEncoderUNet  + fingerprint boundary loss + angular weighting"""
NEW2 = """#     B  ResidualEncoderUNet  + fingerprint boundary loss + angular weighting
#     C  ImageNet-pretrained ResNet18 encoder + fingerprint loss + angular w.
#        (encoder trunk at strides 1/2/4/8 projected onto the fingerprint's
#         feature budget; cs. trainer/preenc_unet.py, knobs STEP4_PREENC_*)"""
assert src.count(OLD2) == 1, "usage block not found exactly once"
src = src.replace(OLD2, NEW2)

OLD3 = """#     setsid nohup bash scripts/51_train_fp.sh B 1 500 > logs/51_wrap_B.log 2>&1 < /dev/null &"""
NEW3 = """#     setsid nohup bash scripts/51_train_fp.sh B 1 500 > logs/51_wrap_B.log 2>&1 < /dev/null &
#     setsid nohup bash scripts/51_train_fp.sh C 1 500 > logs/51_wrap_C.log 2>&1 < /dev/null &"""
assert src.count(OLD3) == 1, "launch example not found exactly once"
src = src.replace(OLD3, NEW3)

if src != orig:
    io.open(TARGET, "w", encoding="utf-8").write(src)
    print("[patch] updated %s" % TARGET)
else:
    print("[patch] already up to date")
