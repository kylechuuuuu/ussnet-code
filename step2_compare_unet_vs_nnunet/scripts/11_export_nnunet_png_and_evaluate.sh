#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step2_compare_unet_vs_nnunet/scripts/11_export_nnunet_png_and_evaluate.sh
#
#   The nnU-Net baseline writes .nii.gz (label maps) while the U-Net baseline
#   writes PNG masks.  To compare them with ONE evaluator, convert the nnU-Net
#   output to single-channel PNG masks named after the original files, then score
#   them with the shared evaluator.
#
#   usage: bash scripts/11_export_nnunet_png_and_evaluate.sh [nifti_dir] [tag] [split]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP2_ROOT="$(pwd)"
USS_ROOT="$(cd "$STEP2_ROOT/.." && pwd)"

NIFTI_DIR="${1:-$USS_ROOT/step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_best_nifti}"
TAG="${2:-nnunet_2d_fold0}"
SPLIT="${3:-val}"
PNG_DIR="$USS_ROOT/step1_nnunet_baseline/predictions/${TAG}_png"

mkdir -p "$PNG_DIR" "$STEP2_ROOT/logs"

echo "[step2] export $NIFTI_DIR -> $PNG_DIR"
"$HOME/anaconda3/envs/zk/bin/python" "$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py" \
    --pred "$NIFTI_DIR" --out "$PNG_DIR" --threshold 0.5 \
    2>&1 | tee "$STEP2_ROOT/logs/11_export_png_${TAG}.log"

echo "[step2] evaluate tag=$TAG split=$SPLIT"
"$HOME/anaconda3/envs/zk/bin/python" "$STEP2_ROOT/scripts/10_evaluate_predictions.py" \
    --pred "$PNG_DIR" --tag "$TAG" --split "$SPLIT" \
    2>&1 | tee "$STEP2_ROOT/logs/11_evaluate_${TAG}.log"

echo "[step2] metrics -> $STEP2_ROOT/metrics/${TAG}_*.csv"
