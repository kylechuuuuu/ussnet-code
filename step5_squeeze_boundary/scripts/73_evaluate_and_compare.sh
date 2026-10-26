#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/73_evaluate_and_compare.sh
#
#   Full post-training evaluation of one arm:
#     1. predict the 102 cases four times, once per decision rule
#        (main / median / inner / outer)  -> PNG -> shared evaluator
#     2. squeeze diagnostics on the validation split (certificates, interval
#        width, error bracketing, angular breakdown)
#     3. rebuild REPORT.md tables from the metrics that were just written
#
#   usage: bash scripts/73_evaluate_and_compare.sh <A|I> [checkpoint] [gpu]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step5.sh

ARM="${1:-A}"
CKPT="${2:-checkpoint_best.pth}"
GPU="${3:-7}"

case "$ARM" in
  A) ARM_TAG="sqA" ;;
  I) ARM_TAG="sqI" ;;
  *) echo "[73] arm must be A or I" >&2; exit 2 ;;
esac
CK="${CKPT%.pth}"

# env_step5.sh inherits CUDA_VISIBLE_DEVICES=0 from step 1; the evaluation must
# run on the card that was asked for (never on someone else's GPU 0).
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[73] ================ arm $ARM / $CKPT ================"
for FUSION in median main inner outer; do
    echo "[73] --- predict with fusion=$FUSION ---"
    bash scripts/72_predict_squeeze.sh "$ARM" "$CKPT" "$FUSION" "$GPU"
done
# the final checkpoint, median rule only (the two checkpoints are reported side by side)
if [ -f "$nnUNet_results/Dataset001_UltrasoundProstate/$([ "$ARM" = A ] && echo nnUNetTrainerProstateSqueeze || echo nnUNetTrainerProstateSqueezeIso)__nnUNetPlans__2d/fold_0/checkpoint_final.pth" ] \
   && [ "$CK" != "checkpoint_final" ]; then
    echo "[73] --- predict checkpoint_final / median ---"
    bash scripts/72_predict_squeeze.sh "$ARM" checkpoint_final.pth median "$GPU"
fi

echo "[73] --- squeeze diagnostics (validation split, mirroring TTA) ---"
"$STEP5_VENV/bin/python" tools/61_squeeze_diagnostics.py \
    --model-folder "$nnUNet_results/Dataset001_UltrasoundProstate/$([ "$ARM" = A ] && echo nnUNetTrainerProstateSqueeze || echo nnUNetTrainerProstateSqueezeIso)__nnUNetPlans__2d" \
    --checkpoint "$CKPT" --tag "${ARM_TAG}_${CK}" --examples 4 \
    2>&1 | tee "$STEP5_ROOT/logs/73_diagnostics_${ARM_TAG}_${CK}.log"

echo "[73] --- rebuild REPORT.md ---"
"$STEP5_VENV/bin/python" tools/64_make_report.py 2>&1 | tee "$STEP5_ROOT/logs/73_report.log"
echo "[73] done: REPORT.md + figures/ + metrics/"
