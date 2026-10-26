#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step6_sandwich_squeeze/scripts/77_auto_finish_w25.sh
#
#   Waits for the two W25 retraining runs and chains, per run:
#       predict 3 versions -> arbitrate (train + val) -> official evaluation
#   and finally rebuilds REPORT.md.
#
#     GPU6: nnUNetTrainerSandwichW25s  500 ep   tag sq2w25s
#     GPU7: nnUNetTrainerSandwichW25  1000 ep   tag sq2w25
#
#   Status log: logs/77_auto_w25.log
#
#   usage: setsid nohup bash scripts/77_auto_finish_w25.sh \
#              > logs/77_auto_w25_wrap.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

LOG="logs/77_auto_w25.log"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

ROOT="$nnUNet_results/Dataset001_UltrasoundProstate"
RUNS=("nnUNetTrainerSandwichW25s sq2w25s 6" "nnUNetTrainerSandwichW25 sq2w25 7")
declare -A DONE=()

say "W25 auto-finish monitor started"
while true; do
    all=1
    for spec in "${RUNS[@]}"; do
        read -r TRAINER TAG GPU <<< "$spec"
        [ -n "${DONE[$TAG]:-}" ] && continue
        all=0
        FOLD="$ROOT/${TRAINER}__nnUNetPlans__2d/fold_0"
        if [ -f "$FOLD/checkpoint_final.pth" ]; then
            CT="${TAG}_checkpoint_best"
            say "$TAG ($TRAINER) finished - predicting 3 versions"
            bash scripts/72_predict_sandwich.sh checkpoint_best.pth "$GPU" "$CT" "$TRAINER" >> "$LOG" 2>&1
            say "$TAG predict done - arbitration train+val"
            bash scripts/73_arbitrate.sh "$CT" both >> "$LOG" 2>&1
            say "$TAG arbitration done"
            "$STEP6_VENV/bin/python" tools/64_make_report6.py >> "$LOG" 2>&1
            say "$TAG chain done"
            DONE[$TAG]=1
        fi
    done
    if [ $all -eq 1 ]; then
        say "all W25 chains complete"
        break
    fi
    sleep 300
done
