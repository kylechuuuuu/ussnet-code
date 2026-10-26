#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step9_boundary_mechanisms/scripts/76_predict_extra_members.sh [gpu]
#
#   Cheap ensemble diversity on an otherwise idle GPU: re-predict the strongest
#   arm (nnUNetTrainerSandwichResencW25 / checkpoint_best, base head) with a
#   different sliding-window budget and without mirroring TTA.  Each pass is
#   ~80 s for the 102 cases; the resulting probability fields are extra,
#   decorrelated members for the boundary-oriented ensemble.
#
#   usage: bash scripts/76_predict_extra_members.sh 5
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ../step6_sandwich_squeeze/scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

GPU="${1:-5}"
export CUDA_VISIBLE_DEVICES="$GPU"
PY="$STEP6_VENV/bin/python"
INPUT="$nnUNet_raw/Dataset001_UltrasoundProstate/imagesTr"
MODEL="$nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerSandwichResencW25__nnUNetPlans__2d"
LOG="logs/76_predict_extra_members.log"
mkdir -p logs
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

run() {                       # tag extra-args...
    local tag="$1"; shift
    local out="$STEP6_ROOT/predictions/${tag}_base_nifti"
    if [ -f "$out/USP001.npz" ]; then say "skip $tag"; return; fi
    say "predict $tag ($*)"
    "$PY" "$STEP6_ROOT/tools/62_predict_sandwich.py" \
        --model-folder "$MODEL" --checkpoint checkpoint_best.pth --head base \
        --input "$INPUT" --output "$out" --device cuda --save-probabilities \
        "$@" 2>&1 | tee "logs/76_predict_${tag}.log" | tail -2
}

say "start gpu=$GPU"
run sq2resencw25_best_step075 --step-size 0.75
run sq2resencw25_best_nostta --no-tta
say "extra members done"
