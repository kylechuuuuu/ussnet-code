#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step15_teacher_distill_sandwich/scripts/71_train_distill.sh
#
#   Train the sandwich + frozen-teacher distillation arm.
#
#   usage: bash scripts/71_train_distill.sh [gpu] [epochs] [fold] [--c]
#
#     gpu     default 0
#     epochs  default 200   (PolyLR completes at this length in our runs)
#     fold    default $FOLD (0)
#     --c     resume from checkpoint_latest
#
#   results -> $nnUNet_results/<dataset>/nnUNetTrainerSandwichDistill__nnUNetPlans__2d/fold_<f>/
#   log     -> logs/71_train_distill_fold<f>_ep<epochs>_gpu<gpu>.log
#
#   Launch detached:
#     setsid nohup bash scripts/71_train_distill.sh 0 200 \
#         > logs/71_wrap_distill.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step15.sh > /dev/null 2>&1 || true

GPU="${1:-0}"
EPOCHS="${2:-200}"
FOLD="${3:-$FOLD}"
EXTRA="${4:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export STEP6_EPOCHS="$EPOCHS"
mkdir -p logs

LOG="logs/71_train_distill_fold${FOLD}_ep${EPOCHS}_gpu${GPU}.log"
echo "=============================================================="
echo "[71] $(date -u +%Y-%m-%dT%H:%M:%SZ) trainer = nnUNetTrainerSandwichDistill"
echo "[71] GPU     = $CUDA_VISIBLE_DEVICES"
echo "[71] epochs  = $STEP6_EPOCHS"
echo "[71] distil  = US_DISTILL_W=$US_DISTILL_W  stats=$US_DISTILL_STATS"
echo "[71] teacher = ${US_FM_CHECKPOINT:-<unset>}"
echo "[71] log     = $LOG"
echo "=============================================================="

"$STEP15_PYTHON" -m nnunetv2.run.run_training "$DATASET_ID" "$CONFIG" "$FOLD" \
    -tr nnUNetTrainerSandwichDistill -device cuda $EXTRA 2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"
echo "[71] $(date -u +%Y-%m-%dT%H:%M:%SZ) training rc=$RC"
exit "$RC"
