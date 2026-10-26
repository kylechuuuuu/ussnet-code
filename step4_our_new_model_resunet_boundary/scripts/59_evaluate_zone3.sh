#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/59_evaluate_zone3.sh
#
#   Predict + export + score arm E (the 3-class zone model).
#
#   The union is obtained for free: nnU-Net writes the ARGMAX label map
#   (0 = bg, 1 = zone127, 2 = zone255) and
#   `05_export_nifti_to_png.py` binarises with `np.where(arr > 0, 255, 0)`, so
#   classes 1 and 2 both become foreground -- exactly the union that
#   `tools/72_build_zone3_dataset.py` verified is byte-identical to
#   Dataset001's foreground.  The result is therefore directly comparable with
#   every other arm through the ONE shared evaluator (`eval_lib.py`).
#
#   usage: bash scripts/59_evaluate_zone3.sh [gpu]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP4_ROOT="$(pwd)"
source scripts/env_step4.sh

GPU="${1:-1}"
DSET="Dataset002_UltrasoundProstate3"
TR="nnUNetTrainerProstateFPZone3"
SHORT="fp_armE_zone3"

export nnUNet_raw="$STEP4_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$STEP4_ROOT/data/nnUNet_preprocessed"
export CUDA_VISIBLE_DEVICES="$GPU"

RUN="$nnUNet_results/$DSET/${TR}__nnUNetPlans__2d/fold_0"
for ck in checkpoint_best.pth checkpoint_final.pth; do
    [ -f "$RUN/$ck" ] || { echo "[59] ABORT: missing $RUN/$ck" >&2; exit 1; }
done

for ck in checkpoint_best.pth checkpoint_final.pth; do
    stem="${ck%.pth}"
    nifti="$STEP4_ROOT/predictions/${SHORT}_${stem}_nifti"
    png="$STEP4_ROOT/predictions/${SHORT}_${stem}_png"
    echo "=== [59] $ck ==="
    rm -rf "$nifti" "$png"
    mkdir -p "$nifti" "$png"
    "$STEP4_VENV/bin/nnUNetv2_predict" \
        -i "$nnUNet_raw/$DSET/imagesTr" -o "$nifti" \
        -d 002 -c "$CONFIG" -f "$FOLD" -tr "$TR" -chk "$ck" \
        -device cuda --save_probabilities
    "$STEP4_VENV/bin/python" \
        "$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py" \
        --pred "$nifti" --out "$png" --threshold 0.5
    "$STEP4_VENV/bin/python" \
        "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
        --pred "$png" --tag "${SHORT}_${stem}" --split val
done

echo "[59] arm E: predictions in $STEP4_ROOT/predictions/${SHORT}_checkpoint_{best,final}_{nifti,png}"
echo "[59] arm E: metrics    in $USS_ROOT/step2_compare_unet_vs_nnunet/metrics/${SHORT}_*"
