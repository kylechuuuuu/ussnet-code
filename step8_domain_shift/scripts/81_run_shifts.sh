#!/usr/bin/env bash
# step8_domain_shift/scripts/81_run_shifts.sh
#
# All inference for the domain-shift study, run SEQUENTIALLY in one process
# tree so the GPU is only ever touched by one pass at a time (the card is
# shared with another user's training job -- we keep the footprint to a single
# ~2-3 GB inference context).
#
#   A) sq2 (step-6 sandwich, resenc-w25) base head, every condition
#   B) stock nnU-Net 2d baseline,          every condition
#   C) sq2 all three heads on the conditions that actually bite
#   D) A + B again on the TTN-standardised images (appearance correction)
#
# usage: bash scripts/81_run_shifts.sh [gpu]
set -u
STEP8="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STEP8"
PY=/home/ta/anaconda3/envs/zk/bin/python
GPU="${1:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

USS=/media/72TB_disk/personal/zk/USSnet
SQ2="$USS/step6_sandwich_squeeze/data/nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerSandwichResencW25__nnUNetPlans__2d"
STOCK="$USS/step1_nnunet_baseline/data/nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainer__nnUNetPlans__2d"
HARD="in_domain,gamma_060,gamma_125,tgc_p60,shadow,speckle_35,blur_35,compound_A,compound_B"
ALL="all"

echo "=== [A] sq2 base, all conditions, shifted ==="
"$PY" tools/81_predict_shifts.py --model-folder "$SQ2" --heads base \
    --tag sq2shift --image-root data/shift --conditions "$ALL" --split val --device cuda:0

echo "=== [B] stock nnU-Net, all conditions, shifted ==="
"$PY" tools/81_predict_shifts.py --model-folder "$STOCK" \
    --tag stockshift --image-root data/shift --conditions "$ALL" --split val --device cuda:0

echo "=== [C] sq2 three heads, hard conditions, shifted ==="
"$PY" tools/81_predict_shifts.py --model-folder "$SQ2" --heads base,cautious,amplified \
    --tag sq23h --image-root data/shift --conditions "$HARD" --split val --device cuda:0

echo "=== [D] TTN-standardised images, both models ==="
"$PY" tools/81_predict_shifts.py --model-folder "$SQ2" --heads base \
    --tag sq2ttn --image-root data/shift_ttn --conditions "$ALL" --split val --device cuda:0
"$PY" tools/81_predict_shifts.py --model-folder "$STOCK" \
    --tag stockttn --image-root data/shift_ttn --conditions "$ALL" --split val --device cuda:0

echo "=== [81_run_shifts] ALL DONE ==="
