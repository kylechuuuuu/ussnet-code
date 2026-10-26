#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step6_sandwich_squeeze/scripts/74_calibrate_existing.sh
#
#   Calibration of the boundary arbitration on EXISTING predictions -- no GPU,
#   no training.  It establishes the reference deltas before the SQ2 model is
#   used, on the shared fold-0 split:
#
#     ctrl stock-200 (checkpoint_best) : free snap only (no interval yet)
#     step-5 SQ-A main/inner/outer     : free snap + learned-band clamp
#     step-1 nnU-Net baseline          : free snap only
#
#   Each set runs on the TRAIN split (fraction ordering) and on VAL.
#
#   usage: bash scripts/74_calibrate_existing.sh [train|val|both]
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

S5="$USS_ROOT/step5_squeeze_boundary"
S1="$USS_ROOT/step1_nnunet_baseline"
SPLITS="${1:-both}"

do_set () {   # tag pred-args...
    local TAG="$1"; shift
    for SPLIT in train val; do
        if [ "$SPLITS" != "both" ] && [ "$SPLITS" != "$SPLIT" ]; then
            continue
        fi
        echo "[74] $(date -u +%H:%M:%SZ) $TAG split=$SPLIT"
        "$STEP6_VENV/bin/python" tools/63_arbitrate.py --split "$SPLIT" \
            --out-dir "predictions/_calib_${TAG}_${SPLIT}" --tag "calib_${TAG}_${SPLIT}" \
            "$@" 2>&1 | tee "logs/74_calib_${TAG}_${SPLIT}.log" | tail -14
    done
}

do_set ctrl --base-png-dir "$S5/predictions/ctrl_stock200_png"
do_set sqA --base-png-dir "$S5/predictions/sqA_main_checkpoint_best_png" \
           --inner-png-dir "$S5/predictions/sqA_inner_checkpoint_best_png" \
           --outer-png-dir "$S5/predictions/sqA_outer_checkpoint_best_png"
do_set nnunet --base-png-dir "$S1/predictions/nnunet_2d_fold0_checkpoint_best_png"

echo "[74] $(date -u +%Y-%m-%dT%H:%M:%SZ) done"
