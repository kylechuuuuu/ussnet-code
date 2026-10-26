#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step9_boundary_mechanisms/scripts/74_ensemble_eval.sh
#
#   Build the missing probability-level ensembles (the step-6 ens_A/B/C set
#   never included the strongest arm nnUNetTrainerSandwichResencW25) and score
#   each of them with the shared step-2 evaluator on the fold-0 val split.
#
#   Pure CPU: reads the per-case softmax .npz that the stock predictor already
#   wrote for every arm, so no GPU is needed.
#
#   usage: bash scripts/74_ensemble_eval.sh [gpu-unused]
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ../step6_sandwich_squeeze/scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

PY="$STEP6_VENV/bin/python"
RES="$STEP6_ROOT/predictions"
EVAL="$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py"
LOG="logs/74_ensemble_eval.log"
mkdir -p logs

RES_D="sq2resencw25_checkpoint_best_base_nifti"
W25_D="sq2w25_checkpoint_best_base_nifti"
W25S_D="sq2w25s_checkpoint_best_base_nifti"
SQ2_D="sq2_checkpoint_best_base_nifti"

say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

one() {                       # tag dirs weights
    local tag="$1" dirs="$2" wts="$3"
    say "=== build $tag"
    if [ -n "$wts" ]; then
        "$PY" tools/99i_ensemble.py --dirs "$dirs" --weights "$wts" --tag "$tag" 2>&1 | tee -a "$LOG"
    else
        "$PY" tools/99i_ensemble.py --dirs "$dirs" --tag "$tag" 2>&1 | tee -a "$LOG"
    fi
    say "=== eval $tag"
    "$PY" "$EVAL" --pred "$RES/${tag}_png" --tag "$tag" --split val 2>&1 | tee -a "$LOG" | tail -4
    say "=== $tag done"
}

say "start (cpu only)"
one ens_D_res_w25            "$RES_D,$W25_D"                          ""
one ens_E_res_w25s           "$RES_D,$W25S_D"                         ""
one ens_F_res_sq2            "$RES_D,$SQ2_D"                          ""
one ens_G_res_w25_w25s_sq2   "$RES_D,$W25_D,$W25S_D,$SQ2_D"           ""
one ens_H_res2_w25_w25s_sq2  "$RES_D,$W25_D,$W25S_D,$SQ2_D"           "2,1,1,1"
one ens_I_res3_w25_w25s_sq2  "$RES_D,$W25_D,$W25S_D,$SQ2_D"           "3,1,1,1"

say "all ensembles done"
