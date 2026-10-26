#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/75_watch_then_evaluate.sh
#
#   Waits until an arm's training is finished, then produces everything the
#   report needs - unattended:
#
#       train done  ->  4 decision rules predicted + scored
#                        squeeze diagnostics on the validation split
#                        REPORT.md rebuilt
#
#   usage: bash scripts/75_watch_then_evaluate.sh <A|I> [gpu] [more arms...]
#          setsid nohup bash scripts/75_watch_then_evaluate.sh A I 7 \
#              > logs/75_watch.log 2>&1 < /dev/null &
#
#   Idempotent: re-running it after the fact re-does the predictions and the
#   report.  Safe to start before the training, while it runs, or after.
#
#   NOTE the `set +e` below: env_step5.sh enables `set -e`, and a controller
#   script that must survive a hiccup in ONE arm (e.g. a failed prediction) has
#   to opt out of it - otherwise a single non-zero exit kills the whole watcher.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step5.sh > /dev/null 2>&1 || true
set +e

GPU_ARG="${@: -1}"
if ! [[ "$GPU_ARG" =~ ^[0-9]+$ ]]; then
    GPU_ARG=7
else
    set -- "${@:1:$#-1}"
fi

for ARM in "$@"; do
  case "$ARM" in
    A) TRAINER="nnUNetTrainerProstateSqueeze";    ARM_TAG="SQ-A_angular" ;;
    I) TRAINER="nnUNetTrainerProstateSqueezeIso"; ARM_TAG="SQ-I_isotropic" ;;
    *) echo "[75] skipping unknown arm '$ARM'" >&2; continue ;;
  esac
  RUN="$nnUNet_results/Dataset001_UltrasoundProstate/${TRAINER}__nnUNetPlans__2d/fold_0"

  echo "[75] $(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for arm $ARM (trainer $TRAINER)"
  echo "[75]   run folder: $RUN"

  # 1) wait until the training process shows up (in case we start first) ...
  n=0
  until pgrep -f "nnUNetv2_train.*$TRAINER" > /dev/null || [ -f "$RUN/checkpoint_final.pth" ]; do
      sleep 60; n=$((n+1)); [ $((n % 30)) -eq 0 ] && echo "[75]   still waiting ($((n/60)) min)"
  done
  # ... and until it is gone again
  while pgrep -f "nnUNetv2_train.*$TRAINER" > /dev/null; do sleep 120; done
  echo "[75] $(date -u +%Y-%m-%dT%H:%M:%SZ) arm $ARM training finished"

  # 2) checkpoints present?
  for CK in checkpoint_best.pth checkpoint_final.pth; do
      for i in $(seq 1 30); do
          [ -f "$RUN/$CK" ] && break
          sleep 20
      done
      [ -f "$RUN/$CK" ] && echo "[75]   found $CK" || echo "[75]   WARNING: $CK missing"
  done

  # 3) evaluate (best checkpoint: 4 decision rules + diagnostics + report)
  echo "[75] $(date -u +%Y-%m-%dT%H:%M:%SZ) evaluating arm $ARM on GPU $GPU_ARG"
  bash scripts/73_evaluate_and_compare.sh "$ARM" checkpoint_best.pth "$GPU_ARG" \
      > logs/75_eval_${ARM}.log 2>&1
  rc=$?
  echo "[75] $(date -u +%Y-%m-%dT%H:%M:%SZ) arm $ARM evaluation rc=$rc (logs/75_eval_${ARM}.log)"
done

echo "[75] $(date -u +%Y-%m-%dT%H:%M:%SZ) all requested arms evaluated"
