#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step1_nnunet_baseline/scripts/env.sh
#
# Environment for the nnU-Net v2 baseline.
#   source scripts/env.sh     <- from inside step1_nnunet_baseline
# before running any of the numbered scripts in this folder.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STEP1_ROOT="$(dirname "${SCRIPT_DIR}")"
export USS_ROOT="$(dirname "${STEP1_ROOT}")"

# --- nnU-Net v2 needs exactly these three variables -------------------------
export nnUNet_raw="${STEP1_ROOT}/data/nnUNet_raw"
export nnUNet_preprocessed="${STEP1_ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${STEP1_ROOT}/data/nnUNet_results"

# --- dedicated venv (system python 3.10 + system torch 2.1.1 + nnunetv2) ----
export NNUNET_VENV="${HOME}/anaconda3/envs/zk"
export PATH="${NNUNET_VENV}/bin:${PATH}"

# --- GPU 0 only (the user asked for card 0) ---------------------------------
export CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=8

# --- dataset / run identifiers ---------------------------------------------
export DATASET_ID=001
export DATASET_NAME="Dataset001_UltrasoundProstate"
export CONFIG=2d
export FOLD=0

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}" "${STEP1_ROOT}/logs"

echo "[env] nnUNet_raw           = ${nnUNet_raw}"
echo "[env] nnUNet_preprocessed  = ${nnUNet_preprocessed}"
echo "[env] nnUNet_results       = ${nnUNet_results}"
echo "[env] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "[env] python               = $(command -v python || echo MISSING)"
