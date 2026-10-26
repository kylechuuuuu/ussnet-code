#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/42_export_png_and_evaluate.sh
#
#   Turn the .nii.gz predictions of the redesigned model into single-channel PNG
#   masks named after the ORIGINAL Chinese files (so they are byte-comparable
#   with the U-Net's `unet/results/...` folder), then score them with the ONE
#   shared evaluator used for all three models (step 2).
#
#   usage: bash scripts/42_export_png_and_evaluate.sh [nifti_dir] [tag] [split]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step4.sh

NIFTI_DIR="${1:-$STEP4_ROOT/predictions/${STEP4_TAG}_${CONFIG}_fold0_checkpoint_best_nifti}"
TAG="${2:-$STEP4_TAG}"
SPLIT="${3:-val}"

PNG_DIR="$STEP4_ROOT/predictions/${TAG}_png"
mkdir -p "$PNG_DIR" "$STEP4_ROOT/logs"

echo "[step4] export  $NIFTI_DIR -> $PNG_DIR"
"$STEP4_VENV/bin/python" "$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py" \
    --pred "$NIFTI_DIR" --out "$PNG_DIR" --threshold 0.5 \
    2>&1 | tee "$STEP4_ROOT/logs/42_export_png_${TAG}.log"

echo "[step4] evaluate tag=$TAG split=$SPLIT (shared evaluator of step 2)"
"$STEP4_VENV/bin/python" "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
    --pred "$PNG_DIR" --tag "$TAG" --split "$SPLIT" \
    2>&1 | tee "$STEP4_ROOT/logs/42_evaluate_${TAG}.log"

echo "[step4] metrics -> $USS_ROOT/step2_compare_unet_vs_nnunet/metrics/${TAG}_*.csv"
