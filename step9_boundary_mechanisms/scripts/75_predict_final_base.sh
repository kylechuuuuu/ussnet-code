#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step9_boundary_mechanisms/scripts/75_predict_final_base.sh [gpu] [ckpt]
#
#   Predict the BASE head of every step-6 arm with a second checkpoint
#   (default checkpoint_final.pth) so the probability-level ensemble gets more
#   independent members: the 4 arms x {best, final} = 8 fields.
#
#   Inference only, ~80 s per arm for the 102 cases, so this runs happily on a
#   single GPU next to the training job.
#
#   usage: bash scripts/75_predict_final_base.sh 5 checkpoint_final.pth
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ../step6_sandwich_squeeze/scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

GPU="${1:-5}"
CKPT="${2:-checkpoint_final.pth}"
export CUDA_VISIBLE_DEVICES="$GPU"
PY="$STEP6_VENV/bin/python"
INPUT="$nnUNet_raw/Dataset001_UltrasoundProstate/imagesTr"
LOG="logs/75_predict_final_base.log"
mkdir -p logs

say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "start gpu=$GPU ckpt=$CKPT"
for pair in "sq2resencw25:nnUNetTrainerSandwichResencW25" \
            "sq2w25:nnUNetTrainerSandwichW25" \
            "sq2w25s:nnUNetTrainerSandwichW25s" \
            "sq2:nnUNetTrainerSandwich"; do
    tag="${pair%%:*}"
    trainer="${pair##*:}"
    MODEL="$nnUNet_results/Dataset001_UltrasoundProstate/${trainer}__nnUNetPlans__2d"
    OUT="$STEP6_ROOT/predictions/${tag}_${CKPT%.pth}_base_nifti"
    if [ -f "$OUT/USP001.npz" ]; then
        say "skip $tag (already predicted)"
        continue
    fi
    say "predict $tag <- $trainer / $CKPT"
    "$PY" "$STEP6_ROOT/tools/62_predict_sandwich.py" \
        --model-folder "$MODEL" --checkpoint "$CKPT" --head base \
        --input "$INPUT" --output "$OUT" --device cuda --save-probabilities \
        2>&1 | tee "logs/75_predict_${tag}_${CKPT%.pth}.log" | tail -2
done
say "all final-base predictions done"
