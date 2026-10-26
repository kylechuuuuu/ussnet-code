#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step2_compare_unet_vs_nnunet/scripts/15_make_final_comparison.sh
#
#   ONE command that turns the per-image metrics of every finished model into the
#   final deliverables.  It discovers for itself which models are ready (a model
#   counts as ready when its PNG prediction folder AND its metrics summary exist),
#   so it is safe to run at any point of the project - it simply reports fewer
#   methods until the missing ones have been predicted and scored.
#
#   It produces
#     metrics/comparison_table.md / .csv          <- headline table (core methods)
#     metrics/comparison_vs_unet_baseline.md/.csv <- same, deltas vs the U-Net
#     metrics/comparison_all_checkpoints.md/.csv  <- every checkpoint, deltas vs nnU-Net
#     figures/*.png                               <- box plots, overlays, error maps
#
#   usage: bash scripts/15_make_final_comparison.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP2_ROOT="$(pwd)"
USS_ROOT="$(cd "$STEP2_ROOT/.." && pwd)"
STEP1="$USS_ROOT/step1_nnunet_baseline"
STEP4="$USS_ROOT/step4_our_new_model_resunet_boundary"
PY="$HOME/anaconda3/envs/zk/bin/python"
MET="$STEP2_ROOT/metrics"
mkdir -p "$MET" "$STEP2_ROOT/figures" "$STEP2_ROOT/logs"

# tag|label|prediction folder (PNG masks named after the original files)
CANDS=(
  "unet_73|U-Net (baseline)|$USS_ROOT/unet/results/ultrasound_prostate_73"
  "nnunet_2d_fold0_checkpoint_final|nnU-Net 2d (baseline, final ckpt)|$STEP1/predictions/nnunet_2d_fold0_checkpoint_final_png"
  "nnunet_2d_fold0_checkpoint_best|nnU-Net 2d (baseline, best ckpt)|$STEP1/predictions/nnunet_2d_fold0_checkpoint_best_png"
  "our_resunet_boundary|OURS: ResUNet + boundary loss (final ckpt)|$STEP4/predictions/our_resunet_boundary_png"
  "our_resunet_boundary_best|OURS: ResUNet + boundary loss (best ckpt)|$STEP4/predictions/our_resunet_boundary_best_png"
  "ensemble_ours_and_nnunet|OURS + nnU-Net (probability ensemble)|$STEP4/predictions/ensemble_ours_and_nnunet_png"
)
# the four methods that go into the headline table and the figures, in order
CORE=(unet_73 nnunet_2d_fold0_checkpoint_final our_resunet_boundary ensemble_ours_and_nnunet)

names=(); labels=(); dirs=()
for c in "${CANDS[@]}"; do
    IFS='|' read -r t l d <<< "$c"
    if [ -d "$d" ] && [ -f "$MET/${t}_summary.json" ]; then
        names+=("$t"); labels+=("$l"); dirs+=("$d")
    else
        echo "[step2] not ready yet, skipped: $t"
    fi
done
[ ${#names[@]} -ge 1 ] || { echo "[step2] nothing to compare yet"; exit 1; }

cnames=(); clabels=(); cdirs=()
for k in "${CORE[@]}"; do
    for i in "${!names[@]}"; do
        if [ "${names[$i]}" = "$k" ]; then
            cnames+=("$k"); clabels+=("${labels[$i]}"); cdirs+=("${dirs[$i]}"); break
        fi
    done
done
echo "[step2] ready methods (core table): ${cnames[*]:-none}"

# ---- 1. headline table, deltas and paired test vs the original U-Net -------
"$PY" scripts/12_make_comparison_tables.py \
    --tags "${cnames[@]}" --labels "${clabels[@]}" --baseline unet_73 \
    --title "Step 2 - headline comparison (U-Net vs nnU-Net vs OURS, one shared evaluator)" \
    2>&1 | tee "$STEP2_ROOT/logs/15_tables_core.log"
cp -f "$MET/comparison_table.md" "$MET/comparison_vs_unet_baseline.md"
cp -f "$MET/comparison_table.csv" "$MET/comparison_vs_unet_baseline.csv"

# ---- 2. every checkpoint we have, deltas and paired test vs nnU-Net --------
if [ -f "$MET/nnunet_2d_fold0_checkpoint_final_summary.json" ]; then
    "$PY" scripts/12_make_comparison_tables.py \
        --tags "${names[@]}" --labels "${labels[@]}" \
        --baseline nnunet_2d_fold0_checkpoint_final \
        --title "Step 2 - all checkpoints (deltas vs the nnU-Net baseline)" \
        --out-prefix comparison_all_checkpoints \
        2>&1 | tee "$STEP2_ROOT/logs/15_tables_all.log"
fi

# ---- 3. figures -----------------------------------------------------------
"$PY" scripts/13_make_figures.py \
    --tags "${cnames[@]}" --labels "${clabels[@]}" --dirs "${cdirs[@]}" \
    2>&1 | tee "$STEP2_ROOT/logs/15_figures.log"

echo "[step2] tables  -> $MET/comparison_table.md (+ comparison_vs_unet_baseline.md, comparison_all_checkpoints.md)"
echo "[step2] figures -> $STEP2_ROOT/figures/"
