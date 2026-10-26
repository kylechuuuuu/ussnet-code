#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/49_ensemble_and_evaluate.sh
#
#   Build the probability-space ensemble "our model + nnU-Net baseline" (both
#   trained on fold 0 only -> fair, see 43_ensemble_with_nnunet.py) and score it
#   with the same shared evaluator.
#
#   usage: bash scripts/49_ensemble_and_evaluate.sh [checkpoint]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step4.sh

CKPT="${1:-checkpoint_best}"
OURS="$STEP4_ROOT/predictions/${STEP4_TAG}_${CONFIG}_fold0_${CKPT}_nifti"
NNUNET="$USS_ROOT/step1_nnunet_baseline/predictions/nnunet_${CONFIG}_fold0_${CKPT}_nifti"
OUT="$STEP4_ROOT/predictions/ensemble_ours_and_nnunet_png"
TAG="ensemble_ours_and_nnunet"

for d in "$OURS" "$NNUNET"; do
    [ -d "$d" ] || { echo "[step4] MISSING: $d"; exit 1; }
done

echo "[step4] ensembling $OURS  +  $NNUNET"
"$STEP4_VENV/bin/python" "$STEP4_ROOT/scripts/43_ensemble_with_nnunet.py" \
    --ours "$OURS" --nnunet "$NNUNET" --out "$OUT" --split val \
    2>&1 | tee "$STEP4_ROOT/logs/49_ensemble.log"

echo "[step4] evaluate tag=$TAG"
"$STEP4_VENV/bin/python" "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
    --pred "$OUT" --tag "$TAG" --split val \
    2>&1 | tee "$STEP4_ROOT/logs/49_ensemble_evaluate.log"
