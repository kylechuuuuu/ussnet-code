#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step6_sandwich_squeeze/scripts/79_auto_resenc.sh
#
#   Overnight run #3: the strongest-base configuration.
#
#   1. waits for the sq2w25s chain (GPU6 window) to finish,
#   2. trains nnUNetTrainerSandwichResencW25 (residual encoder profile of
#      step-4 arm B + the sandwich heads, w=0.25) for 500 epochs on GPU6,
#   3. predict (3 versions) -> arbitrate (train+val) -> report on completion.
#
#   Status log: logs/79_auto_resenc.log
#
#   usage: setsid nohup bash scripts/79_auto_resenc.sh \
#              > logs/79_auto_resenc_wrap.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step6.sh > /dev/null 2>&1 || true
set +e

LOG="logs/79_auto_resenc.log"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "waiting for the sq2w25s chain to free GPU6"
for _ in $(seq 1 360); do
    grep -q "sq2w25s chain done" logs/77_auto_w25.log 2>/dev/null && break
    sleep 120
done
say "GPU6 window open: launching nnUNetTrainerSandwichResencW25 500 ep"

(STEP6_TRAINER=nnUNetTrainerSandwichResencW25 \
    setsid nohup bash scripts/71_train_sandwich.sh 6 500 \
    > logs/79_wrap_resenc.log 2>&1 < /dev/null &)

FOLD="$nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerSandwichResencW25__nnUNetPlans__2d/fold_0"
while [ ! -f "$FOLD/checkpoint_final.pth" ]; do
    sleep 300
done
say "resenc training finished; predict 3 versions"
bash scripts/72_predict_sandwich.sh checkpoint_best.pth 6 \
    "sq2resencw25_checkpoint_best" nnUNetTrainerSandwichResencW25 >> "$LOG" 2>&1
say "resenc predict done; arbitration train+val"
bash scripts/73_arbitrate.sh sq2resencw25_checkpoint_best both >> "$LOG" 2>&1
"$STEP6_VENV/bin/python" tools/64_make_report6.py >> "$LOG" 2>&1
say "resenc chain done"
