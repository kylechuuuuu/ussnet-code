#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Register arm D (nnUNetTrainerProstateFPPreEncR34) everywhere arm C is
registered.  Idempotent."""
import io
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(TOOLS)


def patch(path, reps):
    src = io.open(path, encoding="utf-8").read()
    orig = src
    for old, new in reps:
        if new in src and old not in src:
            continue
        assert src.count(old) == 1, "%s: anchor not unique: %r" % (
            os.path.basename(path), old[:70])
        src = src.replace(old, new)
    if src != orig:
        io.open(path, "w", encoding="utf-8").write(src)
        print("[patch] updated %s" % os.path.basename(path))
    else:
        print("[patch] %s already up to date" % os.path.basename(path))


# ---- 1. the reproducible shim installer ----------------------------------
patch(os.path.join(STEP4, "scripts", "53_install_trainer_shim.sh"), [
    ('''from nnUNetTrainerProstateFPPreEnc import (              # noqa: E402,F401
    nnUNetTrainerProstateFPPreEnc)''',
     '''from nnUNetTrainerProstateFPPreEnc import (              # noqa: E402,F401
    nnUNetTrainerProstateFPPreEnc)
from nnUNetTrainerProstateFPPreEncR34 import (           # noqa: E402,F401
    nnUNetTrainerProstateFPPreEncR34)'''),
    ('''           "nnUNetTrainerProstateFPPreEnc"]''',
     '''           "nnUNetTrainerProstateFPPreEnc",
           "nnUNetTrainerProstateFPPreEncR34"]'''),
    ('''        "nnUNetTrainerProstateFPPreEnc"]''',
     '''        "nnUNetTrainerProstateFPPreEnc",
        "nnUNetTrainerProstateFPPreEncR34"]'''),
])

# ---- 2. arm D must use the NEW class name (own results folder) -----------
patch(os.path.join(STEP4, "scripts", "51_train_fp.sh"), [
    ('''  D) TRAINER="nnUNetTrainerProstateFPPreEnc";          TAG="fp_armD_preenc_r34";  USE_DIR=1; BB="resnet34" ;;''',
     '''  D) TRAINER="nnUNetTrainerProstateFPPreEncR34";       TAG="fp_armD_preenc_r34";  USE_DIR=1; BB="resnet34" ;;'''),
])
patch(os.path.join(STEP4, "scripts", "54_predict_and_evaluate.sh"), [
    ('''        D) echo "nnUNetTrainerProstateFPPreEnc" ;;''',
     '''        D) echo "nnUNetTrainerProstateFPPreEncR34" ;;'''),
])
