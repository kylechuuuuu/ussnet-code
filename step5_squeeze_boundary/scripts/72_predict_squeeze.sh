#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/72_predict_squeeze.sh
#
#   Predict ALL 102 cases with a trained squeeze model under one decision rule,
#   export to the project's PNG convention and score it with the shared
#   evaluator (the same one used for U-Net / nnU-Net / the step-4 arms).
#
#   usage: bash scripts/72_predict_squeeze.sh <A|I> <checkpoint> <fusion> [gpu] [extra 62-args...]
#     A|I        arm: A = angular (nnUNetTrainerProstateSqueeze),
#                     I = isotropic (nnUNetTrainerProstateSqueezeIso)
#     checkpoint checkpoint_best.pth | checkpoint_final.pth
#     fusion     main | median | inner | outer
#   example:
#     bash scripts/72_predict_squeeze.sh A checkpoint_best.pth median 7
#
#   outputs:
#     predictions/sq<arm>_<fusion>_<ckpt>_nifti/   raw NIfTI predictions
#     predictions/sq<arm>_<fusion>_<ckpt>_png/     PNG masks (original names)
#     metrics/sq<arm>_<fusion>_<ckpt>_{per_image.csv,summary.json,summary.txt}
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step5.sh

ARM="${1:-A}"
CKPT="${2:-checkpoint_best.pth}"
FUSION="${3:-median}"
GPU="${4:-7}"
shift $(( $# < 4 ? $# : 4 )) || true

case "$ARM" in
  A) TRAINER="nnUNetTrainerProstateSqueeze";    ARM_TAG="sqA" ;;
  I) TRAINER="nnUNetTrainerProstateSqueezeIso"; ARM_TAG="sqI" ;;
  *) echo "[72] arm must be A or I (got '$ARM')" >&2; exit 2 ;;
esac

CK="${CKPT%.pth}"
TAG="${ARM_TAG}_${FUSION}_${CK}"
MODEL_FOLDER="$nnUNet_results/Dataset001_UltrasoundProstate/${TRAINER}__nnUNetPlans__2d"
NIFTI_DIR="$STEP5_ROOT/predictions/${TAG}_nifti"
PNG_DIR="$STEP5_ROOT/predictions/${TAG}_png"
INPUT="$nnUNet_raw/Dataset001_UltrasoundProstate/imagesTr"

if [ ! -f "$MODEL_FOLDER/fold_0/$CKPT" ]; then
    echo "[72] missing checkpoint $MODEL_FOLDER/fold_0/$CKPT" >&2
    exit 3
fi

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$NIFTI_DIR" "$PNG_DIR" "$STEP5_ROOT/metrics" "$STEP5_ROOT/logs"

echo "[72] arm=$ARM checkpoint=$CKPT fusion=$FUSION -> tag=$TAG (gpu $GPU)"
"$STEP5_VENV/bin/python" tools/62_predict_squeeze.py \
    --model-folder "$MODEL_FOLDER" --checkpoint "$CKPT" --fusion "$FUSION" \
    --input "$INPUT" --output "$NIFTI_DIR" "$@" \
    2>&1 | tee "$STEP5_ROOT/logs/72_predict_${TAG}.log"

echo "[72] export NIfTI -> PNG (original file names)"
"$STEP5_VENV/bin/python" "$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py" \
    --pred "$NIFTI_DIR" --out "$PNG_DIR" --threshold 0.5 \
    2>&1 | tee "$STEP5_ROOT/logs/72_export_${TAG}.log"

echo "[72] score with the shared evaluator (split=val)"
"$STEP5_VENV/bin/python" "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
    --pred "$PNG_DIR" --tag "$TAG" --split val \
    2>&1 | tee "$STEP5_ROOT/logs/72_eval_${TAG}.log"

echo "[72] metrics -> $STEP5_ROOT/../step2_compare_unet_vs_nnunet/metrics/${TAG}_* (evaluator writes there)"
