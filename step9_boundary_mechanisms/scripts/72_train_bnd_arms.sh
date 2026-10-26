#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step9_boundary_mechanisms/scripts/72_train_bnd_arms.sh
#
#   Runs the two mechanism-2 arms SEQUENTIALLY on one GPU (they are 8 GB each,
#   the card is shared, so they cannot run at the same time):
#
#     ARM 1  band    : fine-tune SQ2res-w25 with Dice+CE+band  (skip_band=0)
#                      -> <step6 results>/nnUNetTrainerSandwichBnd__nnUNetPlans__2d
#     ARM 2  control : identical fine-tune, stock Dice+CE only (skip_band=1)
#                      -> step9/data/nnUNet_results_ctl/...
#
#   The control isolates "the new objective" from "150 more epochs at a small
#   LR", which is exactly the confound that makes a bare before/after useless.
#
#   usage: setsid nohup bash scripts/72_train_bnd_arms.sh [gpu] [epochs] \
#              > logs/72_wrap_arms.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
GPU="${1:-6}"
EPOCHS="${2:-150}"

SRC="/media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze/data/nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerSandwichResencW25__nnUNetPlans__2d"
CTL_ROOT="$ROOT/data/nnUNet_results_ctl"
CTL_DS="$CTL_ROOT/Dataset001_UltrasoundProstate"
mkdir -p "$CTL_DS"
for f in dataset.json plans.json dataset_fingerprint.json; do
    [ -f "$CTL_DS/$f" ] || cp "$SRC/$f" "$CTL_DS/$f"
done

echo "[arms] $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$GPU epochs=$EPOCHS"
echo "[arms] ARM 1 band    -> <step6 results>  (D=${STEP9_BAND_D:-12} w=${STEP9_BAND_W:-1.0})"
STEP9_LR="${STEP9_LR:-2e-4}" \
    bash scripts/71_train_bnd.sh "$GPU" "$EPOCHS" 0
RC1=$?
echo "[arms] $(date -u +%Y-%m-%dT%H:%M:%SZ) ARM 1 rc=$RC1"

echo "[arms] ARM 2 control -> $CTL_ROOT  (stock Dice+CE, same LR schedule)"
STEP9_RESULTS="$CTL_ROOT" STEP9_LR="${STEP9_LR:-2e-4}" \
    bash scripts/71_train_bnd.sh "$GPU" "$EPOCHS" 1
RC2=$?
echo "[arms] $(date -u +%Y-%m-%dT%H:%M:%SZ) ARM 2 rc=$RC2"
echo "[arms] $(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE rc1=$RC1 rc2=$RC2"
