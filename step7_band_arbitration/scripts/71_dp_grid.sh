#!/usr/bin/env bash
# step7_band_arbitration/scripts/71_dp_grid.sh
#
# Parameter grid for the reward (active-contour / snake) arbitration.
# Everything here is CPU-only, so it cannot disturb the GPU jobs.
#
#   energy per contour point:   -( R * [R >= frac] - gamma * (t / sigma_p)^2 )
#   transition penalty:          lam * |t_i - t_{i+1}|
#
#   R      ridge evidence (|dI/dt| normalised), frac = 0.70 (step-4 probe value)
#   gamma  elasticity toward the model's own contour (t = 0)
#   sigma_p  scale of "how far is far" for the elastic term
#   lam    smoothness of the displacement field
#
# usage: bash scripts/71_dp_grid.sh [max_parallel]
set -u
STEP7="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STEP7"
source ../step6_sandwich_squeeze/scripts/env_step6.sh >/dev/null 2>&1 || true

PY=/home/ta/anaconda3/envs/zk/bin/python
NPZ=../step6_sandwich_squeeze/predictions/sq2resencw25_checkpoint_best
LAMS="0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.80,1.00"
PAR="${1:-5}"

mkdir -p logs results
run_one () {
  local g="$1" s="$2"
  local tag="dpgrid_g${g}_s${s}"
  echo "[grid] start $tag"
  "$PY" tools/70_dp_arbitrate.py \
      --split val --tag "$tag" \
      --npz-base-dir      "${NPZ}_base_nifti" \
      --npz-cautious-dir  "${NPZ}_cautious_nifti" \
      --npz-amplified-dir "${NPZ}_amplified_nifti" \
      --dp-win 6.0 --lam-list "$LAMS" \
      --gamma "$g" --sigma-p "$s" \
      > "logs/71_grid_${tag}.log" 2>&1
  echo "[grid] done  $tag  rc=$?"
}
export -f run_one
export PY NPZ LAMS

for g in 0.10 0.30 0.60; do
  for s in 2.0 4.0 8.0; do
    echo "$g $s"
  done
done | xargs -P "$PAR" -n 2 bash -c 'run_one "$0" "$1"'
echo "[grid] all done"
