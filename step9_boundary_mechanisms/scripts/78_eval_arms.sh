#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step9_boundary_mechanisms/scripts/78_eval_arms.sh [gpu]
#
#   Once the two step9 fine-tune arms have finished, evaluate them against each
#   other and against the step-6 baseline:
#
#     band  = nnUNetTrainerSandwichBnd        (BoundaryBandLoss, D=12, w=1)
#     ctl   = nnUNetTrainerSandwichBnd with STEP9_SKIP_BAND=1  (same schedule)
#     res   = nnUNetTrainerSandwichResencW25 checkpoint_best  (the start point)
#
#   For each arm: base-head prediction -> PNG -> official step-2 evaluator ->
#   bdice@1px table (tools/99j).
#
#   usage: bash scripts/78_eval_arms.sh 5
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
S9="$PWD"
source ../step6_sandwich_squeeze/scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

GPU="${1:-5}"
export CUDA_VISIBLE_DEVICES="$GPU"
PY="$STEP6_VENV/bin/python"
INPUT="$nnUNet_raw/Dataset001_UltrasoundProstate/imagesTr"
DS="Dataset001_UltrasoundProstate"
TRAINER="nnUNetTrainerSandwichBnd__nnUNetPlans__2d"
EVAL="../step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py"
EXPORT="../step1_nnunet_baseline/scripts/05_export_nifti_to_png.py"
STEP1P="../step1_nnunet_baseline/predictions"
LOG="logs/78_eval_arms.log"
mkdir -p logs
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
n_png() { ls "$1"/*.png 2>/dev/null | wc -l; }

for pair in "band:$nnUNet_results/$DS/$TRAINER" \
            "ctl:$S9/data/nnUNet_results_ctl/$DS/$TRAINER"; do
    arm="${pair%%:*}"
    model="${pair#*:}"
    ckpt="$model/fold_0/checkpoint_final.pth"
    tag="sq2res_w25_${arm}_final"
    nifti="$STEP6_ROOT/predictions/${tag}_base_nifti"
    png="$STEP6_ROOT/predictions/${tag}_png"

    if [ ! -f "$ckpt" ]; then
        say "SKIP $arm: $ckpt not there yet"
        continue
    fi
    if [ ! -f "$nifti/USP001.npz" ]; then
        say "predict $arm (gpu=$GPU)"
        "$PY" "$STEP6_ROOT/tools/62_predict_sandwich.py" \
            --model-folder "$model" --checkpoint checkpoint_final.pth \
            --head base --input "$INPUT" --output "$nifti" --device cuda \
            --save-probabilities \
            2>&1 | tee "logs/78_predict_${arm}.log" | tail -2
    else
        say "skip predict $arm (already there)"
    fi
    if [ "$(n_png "$png")" -lt 100 ]; then
        say "export PNG $arm"
        "$PY" "$EXPORT" --pred "$nifti" --out "$png" --threshold 0.5 \
            2>&1 | tail -2
    fi
    say "evaluate $tag"
    "$PY" "$EVAL" --pred "$(cd "$(dirname "$png")" && pwd)/$(basename "$png")" \
        --tag "$tag" --split val 2>&1 | tail -14
done

say "bdice@1px table"
"$PY" tools/99j_bdice_table.py \
    --dirs "$STEP1P/nnunet_2d_fold0_checkpoint_best_png,$STEP6_ROOT/predictions/sq2resencw25_checkpoint_best_base_png,$STEP6_ROOT/predictions/sq2res_w25_band_final_png,$STEP6_ROOT/predictions/sq2res_w25_ctl_final_png,$STEP6_ROOT/predictions/ens_K_res6_med3open2fill_png" \
    --labels "nnunet,res_best,band,ctl,ens_K" --ref 1 2>&1 | tail -12
say "done - pairwise: python tools/99l_paired_metrics.py --cand <tag> --refs <tag> ..."
