#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/60_chain_armE_after_C.sh
#
#   GPU 1 is the only card we are allowed to use that is not already taken by
#   another user, and arm C is on it right now.  Arm E must therefore WAIT for
#   arm C's whole chain (training + prediction + evaluation, which is what
#   `logs/57_watch_C.log` reports as "arm C ALL DONE") before it starts --
#   running both at once would halve arm C's throughput and delay the primary
#   deliverable for no reason.
#
#   usage: setsid nohup bash scripts/60_chain_armE_after_C.sh \
#              > logs/60_chain_armE.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP4_ROOT="$(pwd)"
LOG="$STEP4_ROOT/logs/60_chain_armE.log"
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[60] chained arm E runner started : $(date -u)"

DONE=0
for _ in $(seq 1 600); do                       # up to ~10 h
    if grep -q "arm C ALL DONE" logs/57_watch_C.log 2>/dev/null; then
        DONE=1; break
    fi
    sleep 60
done
if [ "$DONE" != "1" ]; then
    echo "[60] ABORT: arm C never reported ALL DONE -- not starting arm E"
    exit 1
fi
echo "[60] arm C chain finished : $(date -u) -- GPU1 is free"
grep -E "Mean Validation Dice|Dice" logs/57_watch_C.log | tail -3 || true

# guard: never clobber an existing arm E run
RUN="data/nnUNet_results/Dataset002_UltrasoundProstate3/nnUNetTrainerProstateFPZone3__nnUNetPlans__2d/fold_0"
if [ -d "$RUN" ]; then
    echo "[60] ABORT: $RUN already exists -- refusing to overwrite"
    exit 1
fi

bash scripts/58_train_zone3.sh 1 500
echo "[60] arm E training returned : $(date -u)"
bash scripts/59_evaluate_zone3.sh 1
echo "[60] arm E ALL DONE : $(date -u)"
echo "=================================================================="
