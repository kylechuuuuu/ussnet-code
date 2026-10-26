#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step6_sandwich_squeeze/scripts/73_arbitrate.sh
#
#   BOUNDARY DETERMINATION (然后确定分割的边界): run tools/63_arbitrate.py on
#   the three versions predicted by 72_predict_sandwich.sh, then score the
#   arbitrated masks with the shared evaluator.
#
#   The arbitration parameters (frac 0.7 / maxstep 4.0 / sigma 1.5) follow
#   step 4's train-split probe and are not re-tuned here; the TRAIN split run
#   is diagnostics only, the VAL split run is the headline.
#
#   usage: bash scripts/73_arbitrate.sh [ctag] [split: both|val|train]
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

CTAG="${1:-sq2_checkpoint_best}"
SPLIT="${2:-both}"

run_split () {
    local SPLIT="$1"
    local OUT="predictions/${CTAG}_arb_${SPLIT}"
    echo "[73] $(date -u +%H:%M:%SZ) arbitration split=$SPLIT -> $OUT"
    "$STEP6_VENV/bin/python" tools/63_arbitrate.py \
        --npz-base-dir "predictions/${CTAG}_base_nifti" \
        --npz-cautious-dir "predictions/${CTAG}_cautious_nifti" \
        --npz-amplified-dir "predictions/${CTAG}_amplified_nifti" \
        --split "$SPLIT" --out-dir "$OUT" --tag "${CTAG}_${SPLIT}" \
        2>&1 | tee "logs/73_arbitrate_${CTAG}_${SPLIT}.log" | tail -25
    echo "[73] $(date -u +%H:%M:%SZ) split=$SPLIT rc=${PIPESTATUS[0]}"
}

if [ "$SPLIT" = "both" ] || [ "$SPLIT" = "train" ]; then
    run_split train
fi
if [ "$SPLIT" = "both" ] || [ "$SPLIT" = "val" ]; then
    run_split val
    for V in free band_lrn band_thr veto_lrn veto_thr; do
        if [ -d "$STEP6_ROOT/predictions/${CTAG}_arb_val/$V" ]; then
            echo "[73] $(date -u +%H:%M:%SZ) official evaluation of variant $V"
            "$STEP6_VENV/bin/python" "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
                --pred "$STEP6_ROOT/predictions/${CTAG}_arb_val/$V" --tag "${CTAG}_arb_$V" --split val \
                2>&1 | tee "logs/73_eval_${CTAG}_arb_$V.log" | tail -3
        fi
    done
fi
echo "[73] $(date -u +%Y-%m-%dT%H:%M:%SZ) done"
