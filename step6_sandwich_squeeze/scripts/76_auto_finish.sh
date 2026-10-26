#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step6_sandwich_squeeze/scripts/76_auto_finish.sh
#
#   Waits for BOTH training runs to finish and runs their post-chains:
#
#     ctrl stock-200 resume  -> 79_finish_ctrl200.sh (predict/export/eval)
#                               + tools/63_arbitrate.py (cpp calibration row)
#     SQ2 sandwich 500 ep    -> 72_predict_sandwich.sh (3 versions)
#                               + 73_arbitrate.sh (train + val)
#                               + tools/64_make_report6.py (REPORT.md)
#
#   Status log: logs/76_auto_finish.log
#
#   usage: setsid nohup bash scripts/76_auto_finish.sh \
#              > logs/76_auto_wrap.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

LOG="logs/76_auto_finish.log"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

SQ2_FOLD="$nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerSandwich__nnUNetPlans__2d/fold_0"
CTRL_FOLD="$USS_ROOT/step5_squeeze_boundary/data/nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerProstateStock200__nnUNetPlans__2d/fold_0"

sq2_done=0
ctrl_done=0
say "auto-finish monitor started (ctrl_done=$ctrl_done sq2_done=$sq2_done)"

while [ $sq2_done -eq 0 ] || [ $ctrl_done -eq 0 ]; do
    sleep 300

    if [ $ctrl_done -eq 0 ] && [ -f "$CTRL_FOLD/checkpoint_final.pth" ]; then
        say "ctrl resumed run finished; predict/export/eval"
        bash "$USS_ROOT/step5_squeeze_boundary/scripts/79_finish_ctrl200.sh" 6 >> "$LOG" 2>&1
        say "ctrl predict/export/eval done; arbitration"
        "$STEP6_VENV/bin/python" tools/63_arbitrate.py --split val \
            --base-png-dir "$USS_ROOT/step5_squeeze_boundary/predictions/ctrl_stock200_resumed_png" \
            --out-dir predictions/_calib_ctrl200_val --tag calib_ctrl200_val >> "$LOG" 2>&1
        say "ctrl chain done"
        ctrl_done=1
    fi

    if [ $sq2_done -eq 0 ] && [ -f "$SQ2_FOLD/checkpoint_final.pth" ]; then
        say "SQ2 training finished; predict 3 versions"
        bash scripts/72_predict_sandwich.sh checkpoint_best.pth 7 >> "$LOG" 2>&1
        say "SQ2 predict done; arbitration (train + val)"
        bash scripts/73_arbitrate.sh sq2_checkpoint_best both >> "$LOG" 2>&1
        say "SQ2 arbitration done; building report"
        "$STEP6_VENV/bin/python" tools/64_make_report6.py >> "$LOG" 2>&1
        say "SQ2 chain done"
        sq2_done=1
    fi

    if [ $(( (RANDOM) % 3 )) -eq 0 ]; then
        say "still waiting: ctrl_done=$ctrl_done sq2_done=$sq2_done"
    fi
done
say "auto-finish complete"
