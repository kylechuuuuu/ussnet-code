#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Register arm C (nnUNetTrainerProstateFPPreEnc) in the step4 trainer shim.

Edits scripts/53_install_trainer_shim.sh so the shim stays reproducible, then
that script is re-run.  Idempotent.
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "scripts", "53_install_trainer_shim.sh")

src = io.open(TARGET, encoding="utf-8").read()
orig = src

# 1) the re-export block inside the generated shim
OLD_IMPORTS = """from nnUNetTrainerProstateResUNetFPBoundary import (     # noqa: E402,F401
    nnUNetTrainerProstateResUNetFPBoundary)

__all__ = ["nnUNetTrainerProstateFPBoundary",
           "nnUNetTrainerProstateResUNetFPBoundary"]"""
NEW_IMPORTS = """from nnUNetTrainerProstateResUNetFPBoundary import (     # noqa: E402,F401
    nnUNetTrainerProstateResUNetFPBoundary)
from nnUNetTrainerProstateFPPreEnc import (              # noqa: E402,F401
    nnUNetTrainerProstateFPPreEnc)

__all__ = ["nnUNetTrainerProstateFPBoundary",
           "nnUNetTrainerProstateResUNetFPBoundary",
           "nnUNetTrainerProstateFPPreEnc"]"""
assert src.count(OLD_IMPORTS) == 1, "import block not found exactly once"
src = src.replace(OLD_IMPORTS, NEW_IMPORTS)

# 2) the verification list
OLD_WANT = """want = ["nnUNetTrainerProstateFPBoundary",
        "nnUNetTrainerProstateResUNetFPBoundary"]"""
NEW_WANT = """want = ["nnUNetTrainerProstateFPBoundary",
        "nnUNetTrainerProstateResUNetFPBoundary",
        "nnUNetTrainerProstateFPPreEnc"]"""
assert src.count(OLD_WANT) == 1, "verify list not found exactly once"
src = src.replace(OLD_WANT, NEW_WANT)

# 3) the closing message
OLD_MSG = 'print("[shim] OK - both trainers resolve for training and for prediction")'
NEW_MSG = 'print("[shim] OK - all trainers resolve for training and for prediction")'
assert src.count(OLD_MSG) == 1, "closing message not found exactly once"
src = src.replace(OLD_MSG, NEW_MSG)

if src != orig:
    io.open(TARGET, "w", encoding="utf-8").write(src)
    print("[patch] updated %s" % TARGET)
else:
    print("[patch] already up to date")
