#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/76_squeeze_pipeline.sh
#
#   Expedited end-to-end pipeline (the user asked for results sooner):
#
#       arm SQ-A  (already running, resumed with --c to a 200-epoch schedule)
#         -> wait -> 4 decision rules predicted + scored + squeeze diagnostics
#         -> arm SQ-I (isotropic ablation, 200 epochs, same card)
#         -> wait -> same evaluation
#         -> REPORT.md rebuilt
#
#   The 200-epoch schedule is complete (PolyLR anneals to 0), it just trades
#   schedule length for wall clock: the main head already sat at 96.01 Dice /
#   35.16-style boundary after ~100 epochs, so the extra 400 epochs of the
#   original plan bought little and cost 5 h.  Everything is logged, and the
#   schedule length is written into squeeze_config_used.json for the report.
#
#   usage: setsid nohup bash scripts/76_squeeze_pipeline.sh 7 \
#              > logs/76_pipeline.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step5.sh > /dev/null 2>&1 || true
set +e

GPU="${1:-7}"
TA="nnUNetTrainerProstateSqueeze"
TI="nnUNetTrainerProstateSqueezeIso"
RUN_A="$nnUNet_results/Dataset001_UltrasoundProstate/${TA}__nnUNetPlans__2d/fold_0"
RUN_I="$nnUNet_results/Dataset001_UltrasoundProstate/${TI}__nnUNetPlans__2d/fold_0"

wait_train_done() {   # $1 trainer name, $2 run folder, $3 label
    # two stages: wait until it is running (or already finished), then until it is gone
    n=0
    until pgrep -f "nnUNetv2_train.*$1" > /dev/null || [ -f "$2/checkpoint_final.pth" ]; do
        sleep 60; n=$((n + 1)); [ $((n % 30)) -eq 0 ] && echo "[76]   still waiting for $3 ($((n / 60)) h)"
    done
    while pgrep -f "nnUNetv2_train.*$1" > /dev/null; do sleep 60; done
    for i in $(seq 1 30); do [ -f "$2/checkpoint_final.pth" ] && break; sleep 20; done
}

echo "[76] $(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline start (gpu $GPU)"

wait_train_done "$TA" "$RUN_A" "arm SQ-A"
echo "[76] $(date -u +%Y-%m-%dT%H:%M:%SZ) arm SQ-A training finished -> evaluating"
bash scripts/73_evaluate_and_compare.sh A checkpoint_best.pth "$GPU" > logs/76_eval_A.log 2>&1
echo "[76] $(date -u +%Y-%m-%dT%H:%M:%SZ) arm SQ-A evaluation rc=$? (logs/76_eval_A.log)"

echo "[76] $(date -u +%Y-%m-%dT%H:%M:%SZ) training arm SQ-I (isotropic ablation, 200 epochs)"
bash scripts/71_train_squeeze.sh I "$GPU" 200 > logs/76_train_I_wrap.log 2>&1
echo "[76] $(date -u +%Y-%m-%dT%H:%M:%SZ) arm SQ-I training rc=$?"

echo "[76] $(date -u +%Y-%m-%dT%H:%M:%SZ) evaluating arm SQ-I"
bash scripts/73_evaluate_and_compare.sh I checkpoint_best.pth "$GPU" > logs/76_eval_I.log 2>&1
echo "[76] $(date -u +%Y-%m-%dT%H:%M:%SZ) arm SQ-I evaluation rc=$? (logs/76_eval_I.log)"

echo "[76] $(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline done - see REPORT.md"
