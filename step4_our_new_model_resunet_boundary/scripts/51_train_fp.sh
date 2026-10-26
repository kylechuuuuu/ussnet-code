#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/51_train_fp.sh
#
#   Train one of the two fingerprint-driven redesigned models on fold 0.
#
#   usage: bash scripts/51_train_fp.sh <A|B> [gpu] [epochs]
#
#     A  stock PlainConvUNet  + fingerprint boundary loss        (isolates loss)
#     B  ResidualEncoderUNet  + fingerprint boundary loss + angular weighting
#     C  ImageNet-pretrained ResNet18 encoder + fingerprint loss + angular w.
#        (encoder trunk at strides 1/2/4/8 projected onto the fingerprint's
#         feature budget; cs. trainer/preenc_unet.py, knobs STEP4_PREENC_*)
#
#   results -> data/nnUNet_results/Dataset001_UltrasoundProstate/
#              <trainer>__nnUNetPlans__2d/fold_0/
#   log     -> logs/51_train_<tag>_fold<fold>_ep<epochs>_gpu<gpu>.log
#
#   Launch long runs detached:
#     setsid nohup bash scripts/51_train_fp.sh A 0 500 > logs/51_wrap_A.log 2>&1 < /dev/null &
#     setsid nohup bash scripts/51_train_fp.sh B 1 500 > logs/51_wrap_B.log 2>&1 < /dev/null &
#     setsid nohup bash scripts/51_train_fp.sh C 1 500 > logs/51_wrap_C.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step4.sh

ARM="${1:-A}"
GPU="${2:-0}"
EPOCHS="${3:-500}"

case "$ARM" in
  A) TRAINER="nnUNetTrainerProstateFPBoundary";       TAG="fp_armA_stock_arch";   USE_DIR=0 ;;
  B) TRAINER="nnUNetTrainerProstateResUNetFPBoundary"; TAG="fp_armB_resenc_dir";  USE_DIR=1 ;;
  C) TRAINER="nnUNetTrainerProstateFPPreEnc";          TAG="fp_armC_preenc";      USE_DIR=1; BB="resnet18" ;;
  D) TRAINER="nnUNetTrainerProstateFPPreEncR34";       TAG="fp_armD_preenc_r34";  USE_DIR=1; BB="resnet34" ;;
  *) echo "[51] arm must be A, B, C or D (got '$ARM')" >&2; exit 2 ;;
esac

# env_step4.sh (via step1 env.sh) forces CUDA_VISIBLE_DEVICES=0, so override
# here, AFTER sourcing, to honour the requested card.
export CUDA_VISIBLE_DEVICES="$GPU"
export STEP4_TRAINER="$TRAINER"
export STEP4_TAG="$TAG"
export STEP4_EPOCHS="$EPOCHS"
export STEP4_USE_DIRECTION="$USE_DIR"
# arm C/D share one trainer; the backbone (and hence the architecture) is chosen here
export STEP4_PREENC_BACKBONE="${BB:-resnet18}"

LOG="logs/51_train_${TAG}_fold${FOLD}_ep${EPOCHS}_gpu${GPU}.log"

echo "=============================================================="
echo "[51] arm            = $ARM   ($TRAINER)"
echo "[51] GPU            = $CUDA_VISIBLE_DEVICES"
echo "[51] epochs         = $STEP4_EPOCHS"
echo "[51] direction w.   = $STEP4_USE_DIRECTION"
echo "[51] fingerprint    = ${STEP4_FP_JSON:-<step4>/data/boundary_fingerprint.json}"
echo "[51] log            = $LOG"
echo "=============================================================="

"$STEP4_VENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$FOLD" \
    -tr "$STEP4_TRAINER" -device cuda 2>&1 | tee "$LOG"
