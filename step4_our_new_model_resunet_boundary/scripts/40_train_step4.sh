#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/40_train_step4.sh
#
#   Train the redesigned model (residual encoder + boundary-aware loss) on GPU 0,
#   fold 0 -> the exact same 71/31 split the U-Net baseline and the nnU-Net
#   baseline use, so the three numbers are directly comparable.
#
#   usage: bash scripts/40_train_step4.sh [fold]
#
#   results -> data/nnUNet_results/Dataset001_UltrasoundProstate/
#              nnUNetTrainerProstateResUNetBoundary__nnUNetPlans__2d/fold_<fold>/
#   log     -> logs/40_train_fold<fold>_<trainer>_ep<epochs>.log
#
#   Long run: launch with
#     cd step4_our_new_model_resunet_boundary
#     setsid nohup bash scripts/40_train_step4.sh 0 > logs/40_train_wrapper.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step4.sh

FOLD="${1:-0}"
LOG="$STEP4_ROOT/logs/40_train_fold${FOLD}_${STEP4_TRAINER}_ep${STEP4_EPOCHS}.log"
mkdir -p "$STEP4_ROOT/logs"

echo "[step4] training $STEP4_TRAINER on fold $FOLD, $STEP4_EPOCHS epochs, GPU $CUDA_VISIBLE_DEVICES"
echo "[step4] log -> $LOG"

"$STEP4_VENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$FOLD" \
    -tr "$STEP4_TRAINER" -device cuda 2>&1 | tee "$LOG"
