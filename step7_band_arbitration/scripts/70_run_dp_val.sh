#!/usr/bin/env bash
# step7_band_arbitration/scripts/70_run_dp_val.sh
# Full validation-split sweep of the DP band arbitration (mechanism 1 deepening).
# Baseline tag = the current best step6 model (SQ2res-w25, 500 ep, fold 0).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S7="$(dirname "$HERE")"
USS="$(dirname "$S7")"
S6="$USS/step6_sandwich_squeeze"
PY=/home/ta/anaconda3/envs/zk/bin/python
ST="$S6/predictions/sq2resencw25_checkpoint_best"

cd "$S7"
"$PY" tools/70_dp_arbitrate.py \
  --split val --tag dpval_sweep --dp-win 6.0 \
  --lam-list 0.05,0.15,0.4,1.0 \
  --npz-base-dir     "$ST"_base_nifti \
  --npz-cautious-dir "$ST"_cautious_nifti \
  --npz-amplified-dir "$ST"_amplified_nifti
echo "[70_run] done"
