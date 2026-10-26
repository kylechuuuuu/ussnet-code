#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step9_boundary_mechanisms/scripts/73_eval_bnd.sh
#
#   Score one fine-tuned arm exactly the way every previous arm was scored, so
#   the numbers are directly comparable to SQ2res-w25 (val bank: Dice 96.38 /
#   bdice@1px 35.33 raw, 36.89 after arbitration):
#
#     1. stock nnU-Net predictor, BASE version only, mirroring TTA, plus the
#        per-case probability npz (needed by the step-9 boundary tools)
#     2. nifti -> png at threshold 0.5
#     3. step2's official evaluator (Dice / boundary-Dice@1px / HD95 / ASSD)
#
#   usage: bash scripts/73_eval_bnd.sh <model_folder> <checkpoint> <tag> [gpu]
#
#     model_folder  <...>/nnUNetTrainerSandwichBnd__nnUNetPlans__2d
#     checkpoint    checkpoint_best.pth | checkpoint_final.pth
#     tag           e.g. sq2resbnd_best   (also names the log + output dirs)
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ../step6_sandwich_squeeze/scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

MODEL="$1"
CKPT="${2:-checkpoint_best.pth}"
TAG="$3"
GPU="${4:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

OUT="$STEP6_ROOT/predictions/${TAG}_base_nifti"
PNG="$STEP6_ROOT/predictions/${TAG}_base_png"
IMG="$nnUNet_raw/Dataset001_UltrasoundProstate/imagesTr"

echo "[73] $(date -u +%Y-%m-%dT%H:%M:%SZ) model=$MODEL"
echo "[73] ckpt=$CKPT tag=$TAG gpu=$GPU"

echo "[73] $(date -u +%H:%M:%SZ) predict base version -> $OUT"
"$STEP6_VENV/bin/python" "$STEP6_ROOT/tools/62_predict_sandwich.py" \
    --model-folder "$MODEL" --checkpoint "$CKPT" --head base \
    --input "$IMG" --output "$OUT" --device cuda --save-probabilities \
    2>&1 | tee "$STEP6_ROOT/logs/73_predict_${TAG}.log" | tail -3

echo "[73] $(date -u +%H:%M:%SZ) nifti -> png"
"$STEP6_VENV/bin/python" "$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py" \
    --pred "$OUT" --out "$PNG" --threshold 0.5 \
    2>&1 | tee "$STEP6_ROOT/logs/73_export_${TAG}.log" | tail -2

echo "[73] $(date -u +%H:%M:%SZ) official evaluation"
"$STEP6_VENV/bin/python" "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
    --pred "$PNG" --tag "${TAG}_base" --split val \
    2>&1 | tee "$STEP6_ROOT/logs/73_eval_${TAG}.log" | tail -5

echo "[73] $(date -u +%Y-%m-%dT%H:%M:%SZ) done"
