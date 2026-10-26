#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step1_nnunet_baseline/scripts/06_evaluate_nnunet.sh
#   nnU-Net's own evaluator (Dice / IoU / HD95 / boundary metrics, plus the
#   automatic post-processing search).  The apples-to-apples comparison with
#   the U-Net baseline is done separately in step2 (one shared metric file).
#
#   usage: bash scripts/06_evaluate_nnunet.sh [fold]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

FOLD="${1:-0}"
PRED="predictions/nnunet_${CONFIG}_fold${FOLD}_checkpoint_best_nifti"
GT="${nnUNet_raw}/${DATASET_NAME}/labelsTr"
OUT="predictions/nnunet_${CONFIG}_fold${FOLD}_eval"
mkdir -p logs "${OUT}"

# --- restrict GT to the fold's validation split ------------------------------
SPLIT_DIR="predictions/_fold${FOLD}_gt"
rm -rf "${SPLIT_DIR}"; mkdir -p "${SPLIT_DIR}"
python - "$FOLD" "$SPLIT_DIR" <<'PY'
import json, os, shutil, sys
fold = int(sys.argv[1]); out = sys.argv[2]
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pre = os.path.join(base, "data", "nnUNet_preprocessed", "Dataset001_UltrasoundProstate")
raw = os.path.join(base, "data", "nnUNet_raw", "Dataset001_UltrasoundProstate", "labelsTr")
splits = json.load(open(os.path.join(pre, "splits_final.json")))
for k in splits[fold]["val"]:
    shutil.copy(os.path.join(raw, k + ".nii.gz"), os.path.join(out, k + ".nii.gz"))
print("copied", len(splits[fold]["val"]), "val labels")
PY

nnUNetv2_evaluate_folder \
    -ref "${SPLIT_DIR}" \
    -pred "${PRED}" \
    -l 1 \
    -o "${OUT}/summary.json" \
    -np 8 \
    -f 0 1 2 \
    2>&1 | tee "logs/06_evaluate_fold${FOLD}.log"

echo "summary -> ${OUT}/summary.json"
