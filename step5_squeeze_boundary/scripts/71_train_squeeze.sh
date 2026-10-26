#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/71_train_squeeze.sh
#
#   Train one arm of the squeeze-theorem (夹逼) model on fold 0.
#
#   usage: bash scripts/71_train_squeeze.sh <A|I> [gpu] [epochs]
#
#     A  SQ-A: angular band   r(theta) = r(t) * s(theta)   [fingerprint-driven]
#     I  SQ-I: isotropic band r(theta) = r(t)             [ablation]
#
#   results -> data/nnUNet_results/Dataset001_UltrasoundProstate/
#              <trainer>__nnUNetPlans__2d/fold_0/
#   log     -> logs/71_train_<tag>_fold0_ep<epochs>_gpu<gpu>.log
#
#   Launch detached (this is how the long run was started):
#     setsid nohup bash scripts/71_train_squeeze.sh A 7 500 \
#         > logs/71_wrap_A.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step5.sh

ARM="${1:-A}"
GPU="${2:-7}"
EPOCHS="${3:-500}"

case "$ARM" in
  A) TRAINER="nnUNetTrainerProstateSqueeze";    TAG="SQ-A_angular" ;;
  I) TRAINER="nnUNetTrainerProstateSqueezeIso"; TAG="SQ-I_isotropic" ;;
  *) echo "[71] arm must be A or I (got '$ARM')" >&2; exit 2 ;;
esac

# env_step5.sh inherits CUDA_VISIBLE_DEVICES=0 from step 1's env.sh; override the
# requested card AFTER sourcing.
export CUDA_VISIBLE_DEVICES="$GPU"
export STEP5_EPOCHS="$EPOCHS"
export STEP5_ANGULAR="$([ "$ARM" = "A" ] && echo 1 || echo 0)"

LOG="logs/71_train_${TAG}_fold0_ep${EPOCHS}_gpu${GPU}.log"
echo "=============================================================="
echo "[71] arm     = $ARM ($TRAINER)"
echo "[71] GPU     = $CUDA_VISIBLE_DEVICES"
echo "[71] epochs  = $STEP5_EPOCHS   (PolyLR completes at this length)"
echo "[71] angular = $STEP5_ANGULAR"
echo "[71] extra   = ${4:-<none>}   (pass --c to resume from checkpoint_latest)"
echo "[71] log     = $LOG"
echo "=============================================================="

"$STEP5_VENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$FOLD" \
    -tr "$TRAINER" -device cuda ${4:-} 2>&1 | tee "$LOG"
