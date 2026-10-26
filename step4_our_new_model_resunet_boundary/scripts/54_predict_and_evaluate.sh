#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/54_predict_and_evaluate.sh
#
#   Inference + evaluation for the fingerprint-driven arms.
#
#   For every (arm, checkpoint) pair:
#     1. predict ALL 102 cases (train + val, same as step 1 does) with the
#        checkpoint, writing NIfTI label maps AND probabilities
#     2. convert the NIfTI label maps to the PNG convention the U-Net baseline
#        writes (0/255, original resolution, original Chinese file name)
#     3. score the 31 fold-0 validation images with the ONE shared evaluator
#        (step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py)
#
#   Both checkpoints are reported, matching step 1's own reporting:
#     checkpoint_final.pth -> no selection on the 31 validation images at all
#     checkpoint_best.pth  -> selected on the same 31 images as the baseline
#
#   Everything lands under step 4, so the experiment folder stays
#   self-contained; only the raw data, the fingerprint and the evaluator are
#   shared with the other steps.
#
#   usage: bash scripts/54_predict_and_evaluate.sh [gpu] [arms] [ckpts]
#            gpu    default 0
#            arms   default "A B"          (A = stock arch + fp loss,
#                                           B = ResEnc + fp loss + direction,
#                                           C = ImageNet-pretrained encoder)
#            ckpts  default "checkpoint_best.pth checkpoint_final.pth"
#
#   example: bash scripts/54_predict_and_evaluate.sh 0 "A B" checkpoint_best.pth
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP4_ROOT="$(pwd)"
USS_ROOT="$(cd "$STEP4_ROOT/.." && pwd)"
source scripts/env_step4.sh

GPU="${1:-0}"
ARMS="${2:-A B}"
CKPTS="${3:-checkpoint_best.pth checkpoint_final.pth}"

# env_step4.sh sources step1's env.sh, which forces card 0 -- override it.
export CUDA_VISIBLE_DEVICES="$GPU"

EXPORT_PY="$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py"
EVAL_PY="$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py"
PY="$STEP4_VENV/bin/python"

mkdir -p "$STEP4_ROOT/predictions" "$STEP4_ROOT/metrics" "$STEP4_ROOT/logs"

trainer_for_arm () {
    case "$1" in
        A) echo "nnUNetTrainerProstateFPBoundary" ;;
        B) echo "nnUNetTrainerProstateResUNetFPBoundary" ;;
        C) echo "nnUNetTrainerProstateFPPreEnc" ;;
        D) echo "nnUNetTrainerProstateFPPreEncR34" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;
    esac
}
short_for_arm () {
    case "$1" in
        A) echo "fpA_stock_fploss" ;;
        B) echo "fpB_resenc_fploss_dir" ;;
        C) echo "fp_armC_preenc" ;;
        D) echo "fp_armD_preenc_r34" ;;
        *) echo "unknown arm '$1'" >&2; exit 1 ;;
    esac
}

echo "=================================================================="
echo "[54] gpu     = $CUDA_VISIBLE_DEVICES"
echo "[54] arms    = $ARMS"
echo "[54] ckpts   = $CKPTS"
echo "[54] started : $(date -u)"
echo "=================================================================="

for arm in $ARMS; do
    TR="$(trainer_for_arm "$arm")"
    SHORT="$(short_for_arm "$arm")"
    RUN_DIR="$nnUNet_results/$DATASET_NAME/${TR}__nnUNetPlans__2d/fold_0"

    if [ ! -d "$RUN_DIR" ]; then
        echo "[54] SKIP arm $arm: no run dir $RUN_DIR"
        continue
    fi

    for ckpt in $CKPTS; do
        [ -f "$RUN_DIR/$ckpt" ] || { echo "[54] SKIP $arm/$ckpt: not found"; continue; }
        stem="${ckpt%.pth}"
        tag="${SHORT}_${stem}"
        NIFTI_OUT="$STEP4_ROOT/predictions/${tag}_nifti"
        PNG_OUT="$STEP4_ROOT/predictions/${tag}_png"

        echo "------------------------------------------------------------------"
        echo "[54] arm $arm  trainer $TR"
        echo "[54]   ckpt    $RUN_DIR/$ckpt"
        echo "[54]   nifti   $NIFTI_OUT"
        echo "[54]   png     $PNG_OUT"
        echo "[54]   tag     $tag"
        echo "------------------------------------------------------------------"

        # ---- 1. predict all 102 cases ---------------------------------
        rm -rf "$NIFTI_OUT"; mkdir -p "$NIFTI_OUT"
        "$STEP4_VENV/bin/nnUNetv2_predict" \
            -i "$nnUNet_raw/$DATASET_NAME/imagesTr" \
            -o "$NIFTI_OUT" \
            -d "$DATASET_ID" \
            -c "$CONFIG" \
            -f "$FOLD" \
            -tr "$TR" \
            -chk "$ckpt" \
            -device cuda \
            --save_probabilities \
            2>&1 | tee "$STEP4_ROOT/logs/54_predict_${tag}.log"

        # ---- 2. NIfTI -> PNG (same convention as the U-Net baseline) ---
        "$PY" "$EXPORT_PY" --pred "$NIFTI_OUT" --out "$PNG_OUT" --threshold 0.5 \
            2>&1 | tee "$STEP4_ROOT/logs/54_export_${tag}.log"

        # ---- 3. score the 31 validation images with the shared evaluator
        "$PY" "$EVAL_PY" --pred "$PNG_OUT" --tag "$tag" --split val \
            2>&1 | tee "$STEP4_ROOT/logs/54_evaluate_${tag}.log"

        echo "[54] metrics -> $STEP4_ROOT/metrics/${tag}_*.csv"
    done
done

echo "=================================================================="
echo "[54] DONE : $(date -u)"
echo "=================================================================="
