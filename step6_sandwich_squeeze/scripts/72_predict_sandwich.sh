#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step6_sandwich_squeeze/scripts/72_predict_sandwich.sh
#
#   Predict all three versions of the sandwich model (base / cautious /
#   amplified) with the stock nnU-Net predictor + TTA, save the per-case
#   probability npz files, export the base version to PNG and score it with
#   the shared evaluator.
#
#   usage: bash scripts/72_predict_sandwich.sh [checkpoint] [gpu]
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

CKPT="${1:-checkpoint_best.pth}"
GPU="${2:-7}"
TAG="${3:-}"
TRAINER="${4:-nnUNetTrainerSandwich}"
export CUDA_VISIBLE_DEVICES="$GPU"

CTAG="${TAG:-sq2_${CKPT%.pth}}"                    # e.g. sq2w25s_checkpoint_best
MODEL="$nnUNet_results/Dataset001_UltrasoundProstate/${TRAINER}__nnUNetPlans__2d"

echo "[72] $(date -u +%Y-%m-%dT%H:%M:%SZ) predict 3 versions with $CKPT on gpu $GPU"
for HEAD in base cautious amplified; do
    OUT="predictions/${CTAG}_${HEAD}_nifti"
    echo "[72] $(date -u +%H:%M:%SZ) head=$HEAD -> $OUT"
    "$STEP6_VENV/bin/python" tools/62_predict_sandwich.py \
        --model-folder "$MODEL" --checkpoint "$CKPT" --head "$HEAD" \
        --input "$nnUNet_raw/Dataset001_UltrasoundProstate/imagesTr" \
        --output "$OUT" --device cuda --save-probabilities \
        2>&1 | tee "logs/72_predict_${CTAG}_${HEAD}.log" | tail -2
done

echo "[72] $(date -u +%H:%M:%SZ) export base version to PNG + official evaluation"
"$STEP6_VENV/bin/python" "$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py" \
    --pred "$STEP6_ROOT/predictions/${CTAG}_base_nifti" --out "$STEP6_ROOT/predictions/${CTAG}_base_png" --threshold 0.5 \
    2>&1 | tee "logs/72_export_${CTAG}_base.log" | tail -2
"$STEP6_VENV/bin/python" "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
    --pred "$STEP6_ROOT/predictions/${CTAG}_base_png" --tag "${CTAG}_base" --split val \
    2>&1 | tee "logs/72_eval_${CTAG}_base.log" | tail -3
echo "[72] $(date -u +%Y-%m-%dT%H:%M:%SZ) done"
