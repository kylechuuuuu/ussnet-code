#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step6_sandwich_squeeze/scripts/71_train_sandwich.sh
#
#   Train SQ2 (sandwich squeeze: base + cautious + amplified) on fold 0.
#
#   usage: bash scripts/71_train_sandwich.sh [gpu] [epochs] [--c]
#
#     gpu     default 7
#     epochs  default 500  (PolyLR completes at this length)
#     --c     resume from checkpoint_latest (continue an interrupted run)
#
#   results -> data/nnUNet_results/Dataset001_UltrasoundProstate/
#              nnUNetTrainerSandwich__nnUNetPlans__2d/fold_0/
#   log     -> logs/71_train_sq2_fold0_ep<epochs>_gpu<gpu>.log
#
#   Launch detached (this is how the long run is started):
#     setsid nohup bash scripts/71_train_sandwich.sh 7 500 \
#         > logs/71_wrap_sq2.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

GPU="${1:-7}"
EPOCHS="${2:-500}"
EXTRA="${3:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export STEP6_EPOCHS="$EPOCHS"

TRAINER="${STEP6_TRAINER:-nnUNetTrainerSandwich}"
LOG="logs/71_train_${TRAINER}_fold0_ep${EPOCHS}_gpu${GPU}.log"
echo "=============================================================="
echo "[71] $(date -u +%Y-%m-%dT%H:%M:%SZ) trainer = $TRAINER"
echo "[71] GPU     = $CUDA_VISIBLE_DEVICES"
echo "[71] epochs  = $STEP6_EPOCHS   (PolyLR completes at this length)"
echo "[71] radii   = r_c=${STEP6_RC:-3} r_a=${STEP6_RA:-4} angular=${STEP6_ANGULAR:-1}"
echo "[71] extra   = ${EXTRA:-<none>}   (pass --c to resume from checkpoint_latest)"
echo "[71] log     = $LOG"
echo "=============================================================="

"$STEP6_VENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$FOLD" \
    -tr "$TRAINER" -device cuda $EXTRA 2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"
echo "[71] $(date -u +%Y-%m-%dT%H:%M:%SZ) training rc=$RC"
exit "$RC"
