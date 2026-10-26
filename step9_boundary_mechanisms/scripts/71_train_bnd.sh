#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step9_boundary_mechanisms/scripts/71_train_bnd.sh
#
#   Fine-tune SQ2res-w25 with the step-9 BOUNDARY-METRIC objective
#   (mechanism 2: symmetric distance-weighted tolerance-band loss).
#
#   usage: bash scripts/71_train_bnd.sh [gpu] [epochs] [skip_band]
#
#     gpu        default 6            (CUDA_VISIBLE_DEVICES, coexist with others)
#     epochs     default 150          (PolyLR completes at this length)
#     skip_band  default 0            1 = CONTROL ARM: identical fine-tune but
#                                     with the stock Dice+CE objective only
#
#   env overrides: STEP9_LR STEP9_BAND_D STEP9_BAND_W STEP9_PRETRAINED
#                  STEP9_RESULTS (redirect nnUNet_results, used for probes)
#
#   results -> <nnUNet_results>/Dataset001_UltrasoundProstate/
#              nnUNetTrainerSandwichBnd__nnUNetPlans__2d/fold_0/
#   log     -> logs/71_train_bnd_skip<skip>_ep<epochs>_gpu<gpu>.log
#
#   Launch detached:
#     setsid nohup bash scripts/71_train_bnd.sh 6 150 0 \
#         > logs/71_wrap_bnd.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ../step6_sandwich_squeeze/scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

GPU="${1:-6}"
EPOCHS="${2:-150}"
SKIP="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export STEP9_EPOCHS="$EPOCHS"
export STEP9_SKIP_BAND="$SKIP"
if [ -n "${STEP9_RESULTS:-}" ]; then
    export nnUNet_results="$STEP9_RESULTS"
fi

TRAINER="nnUNetTrainerSandwichBnd"
mkdir -p logs
LOG="logs/71_train_bnd_skip${SKIP}_ep${EPOCHS}_gpu${GPU}.log"

{
  echo "=============================================================="
  echo "[71b] $(date -u +%Y-%m-%dT%H:%M:%SZ) trainer = $TRAINER"
  echo "[71b] GPU      = $CUDA_VISIBLE_DEVICES"
  echo "[71b] epochs   = $STEP9_EPOCHS   lr = ${STEP9_LR:-2e-4}"
  echo "[71b] band     = D=${STEP9_BAND_D:-12} w=${STEP9_BAND_W:-1.0} skip=${STEP9_SKIP_BAND}"
  echo "[71b] results  = $nnUNet_results"
  echo "[71b] log      = $LOG"
  echo "=============================================================="
} | tee "$LOG"

"$STEP6_VENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$FOLD" \
    -tr "$TRAINER" -device cuda 2>&1 | tee -a "$LOG"
RC="${PIPESTATUS[0]}"
echo "[71b] $(date -u +%Y-%m-%dT%H:%M:%SZ) training rc=$RC" | tee -a "$LOG"
exit "$RC"
