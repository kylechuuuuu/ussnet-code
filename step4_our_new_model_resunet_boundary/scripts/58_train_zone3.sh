#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/58_train_zone3.sh
#
#   Arm E: train the fingerprint loss on the 3-class zone parameterisation
#   (Dataset002_UltrasoundProstate3) instead of the binary one.
#
#   Dataset002 lives in a STEP4-LOCAL nnU-Net raw root so the shared, read-only
#   step1 tree is never written to -- and so that overriding nnUNet_raw here
#   cannot disturb a running job that captured the old value at startup.
#   nnUNet_results deliberately stays the shared step4 directory so that all
#   arms' outputs sit together.
#
#   usage: setsid nohup bash scripts/58_train_zone3.sh 1 500 \
#              > logs/58_wrap_E.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP4_ROOT="$(pwd)"
source scripts/env_step4.sh

GPU="${1:-1}"
EPOCHS="${2:-500}"

# --- step4-local data roots for the 3-class dataset -----------------------
export nnUNet_raw="$STEP4_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$STEP4_ROOT/data/nnUNet_preprocessed"
DSET=Dataset002_UltrasoundProstate3
[ -f "$nnUNet_preprocessed/$DSET/nnUNetPlans.json" ] || {
    echo "[58] ABORT: $DSET is not preprocessed (run tools/72_build_zone3_dataset.py"
    echo "     then nnUNetv2_plan_and_preprocess -d 2 -c 2d)" >&2; exit 1; }

# --- arm E config ---------------------------------------------------------
export STEP4_EPOCHS="$EPOCHS"
export STEP4_USE_DIRECTION=1                 # same recipe as arm B
export STEP4_TRAINER="nnUNetTrainerProstateFPZone3"
export CUDA_VISIBLE_DEVICES="$GPU"

TR="$STEP4_TRAINER"
LOG="$STEP4_ROOT/logs/58_train_armE_zone3_fold0_ep${EPOCHS}_gpu${GPU}.log"

echo "[58] arm E | dataset $DSET | GPU $GPU | epochs $EPOCHS | direction on"
echo "[58] raw          = $nnUNet_raw"
echo "[58] preprocessed = $nnUNet_preprocessed"
echo "[58] results      = $nnUNet_results"
echo "[58] log          = $LOG"

"$STEP4_VENV/bin/nnUNetv2_train" 002 2d 0 -tr "$TR" -device cuda 2>&1 | tee "$LOG"
