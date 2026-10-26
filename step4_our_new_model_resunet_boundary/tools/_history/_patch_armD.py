#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add arm D (resnet34 pretrained encoder) and make the arm-C trainer's
labels follow the backbone.  Idempotent."""
import io
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(TOOLS)


def patch(path, reps, created_lines=None):
    src = io.open(path, encoding="utf-8").read()
    orig = src
    changed = []
    for old, new in reps:
        if new in src and old not in src:
            changed.append("already")
            continue
        assert src.count(old) == 1, "%s: anchor not unique: %r" % (path, old[:70])
        src = src.replace(old, new)
        changed.append("ok")
    if src != orig:
        io.open(path, "w", encoding="utf-8").write(src)
        print("[patch] %s -> %s" % (os.path.basename(path), changed))
    else:
        print("[patch] %s already up to date" % os.path.basename(path))


# ---------------------------------------------------------------- trainer labels
TR = os.path.join(STEP4, "trainer", "nnUNetTrainerProstateFPPreEnc.py")
reps = [
    ('''        self.trunk_lr_scale = float(os.environ.get("STEP4_PREENC_LR_SCALE", "0.1"))''',
     '''        self.trunk_lr_scale = float(os.environ.get("STEP4_PREENC_LR_SCALE", "0.1"))
        # the label/tag follow the backbone, so arm D (resnet34) logs honestly
        self.arm_label = (os.environ.get("STEP4_PREENC_LABEL")
                          or ("arm C" if self.backbone == "resnet18" else "arm D"))
        self.arm_tag = "preenc_%s_fp_loss" % self.backbone
        self.arm_desc = ("%s (ImageNet-pretrained %s encoder + fingerprint loss)"
                         % (self.arm_label, self.backbone))'''),
    ('''            % (self.ARM_DESC, self.num_epochs, type(self.network).__name__, enc.load_report()))''',
     '''            % (self.arm_desc, self.num_epochs, type(self.network).__name__, enc.load_report()))'''),
    ('''            "[fp] arm C config: backbone %s | freeze_upto %d | trunk lr scale %.3f | "
            "params %.2f M | trainable %.2f M"
            % (self.backbone, self.freeze_upto, self.trunk_lr_scale,''',
     '''            "[fp] %s config: backbone %s | freeze_upto %d | trunk lr scale %.3f | "
            "params %.2f M | trainable %.2f M"
            % (self.arm_label, self.backbone, self.freeze_upto, self.trunk_lr_scale,'''),
    ('''            "[fp] arm C param groups: " + " | ".join(''',
     '''            ("[fp] %s param groups: " % self.arm_label) + " | ".join('''),
    ('''                json.dump({"arm": self.ARM_TAG,''',
     '''                json.dump({"arm": self.arm_tag,
                           "arm_label": self.arm_label,'''),
]
patch(TR, reps)

# ---------------------------------------------------------------- launcher
L51 = os.path.join(STEP4, "scripts", "51_train_fp.sh")
reps = [
    ('''  C) TRAINER="nnUNetTrainerProstateFPPreEnc";          TAG="fp_armC_preenc";      USE_DIR=1 ;;
  *) echo "[51] arm must be A, B or C (got '$ARM')" >&2; exit 2 ;;''',
     '''  C) TRAINER="nnUNetTrainerProstateFPPreEnc";          TAG="fp_armC_preenc";      USE_DIR=1; BB="resnet18" ;;
  D) TRAINER="nnUNetTrainerProstateFPPreEnc";          TAG="fp_armD_preenc_r34";  USE_DIR=1; BB="resnet34" ;;
  *) echo "[51] arm must be A, B, C or D (got '$ARM')" >&2; exit 2 ;;'''),
    ('''export STEP4_USE_DIRECTION="$USE_DIR"''',
     '''export STEP4_USE_DIRECTION="$USE_DIR"
# arm C/D share one trainer; the backbone (and hence the architecture) is chosen here
export STEP4_PREENC_BACKBONE="${BB:-resnet18}"'''),
]
patch(L51, reps)

# ---------------------------------------------------------------- predict launcher
L54 = os.path.join(STEP4, "scripts", "54_predict_and_evaluate.sh")
reps = [
    ('''        C) echo "nnUNetTrainerProstateFPPreEnc" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;''',
     '''        C) echo "nnUNetTrainerProstateFPPreEnc" ;;
        D) echo "nnUNetTrainerProstateFPPreEnc" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;'''),
    ('''        C) echo "fp_armC_preenc" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;''',
     '''        C) echo "fp_armC_preenc" ;;
        D) echo "fp_armD_preenc_r34" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;'''),
]
patch(L54, reps)

# ---------------------------------------------------------------- ensemble members
L69 = os.path.join(STEP4, "tools", "69_ensemble_eval.py")
reps = [
    ('''    "Cf": os.path.join(P, "fp_armC_preenc_checkpoint_final_nifti"),
}''',
     '''    "Cf": os.path.join(P, "fp_armC_preenc_checkpoint_final_nifti"),
    "Db": os.path.join(P, "fp_armD_preenc_r34_checkpoint_best_nifti"),
    "Df": os.path.join(P, "fp_armD_preenc_r34_checkpoint_final_nifti"),
}'''),
    ('''    ("Ab+Af+Bb+Bf+Cb+Cf+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "nnB", "nnF"]),
]''',
     '''    ("Ab+Af+Bb+Bf+Cb+Cf+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "nnB", "nnF"]),
    ("Cb+Cf+Db+Df", ["Cb", "Cf", "Db", "Df"]),
    ("Cb+Cf+Db+Df+nnB", ["Cb", "Cf", "Db", "Df", "nnB"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+Db+Df", ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+Db+Df+nnB",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "nnB"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+Db+Df+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "nnB", "nnF"]),
]'''),
]
patch(L69, reps)
