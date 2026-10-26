#!/bin/bash
# Full fold-3-val (31 cases) boundary arbitration for the causal battery.
# The queue's inline arbitration uses step6's default fold-0 val list (intersection
# with fold3 val = 7 cases); this script reruns 63_arbitrate with the full fold3
# list (--names-file) so the delta(c) numbers have proper statistics.
# Run AFTER the training queues have exported masks (predictions/l2vit_<tag>/).
set -u
PY=/home/ta/anaconda3/envs/zk/bin/python
S=/media/72TB_disk/personal/zk/USSnet/step10_us_foundation_front
S6=/media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze
NAMES=$S/data/fold3_val_names.txt

for TAG in "$@"; do
  P=$S/predictions/l2vit_$TAG
  if [ ! -d "$P/base" ]; then echo "[skip] no masks for $TAG"; continue; fi
  echo "===== full31 arbitration: $TAG ====="
  cd "$S6"
  $PY tools/63_arbitrate.py --split val --names-file "$NAMES" \
    --out-dir "$P/arb_val31" --tag "${TAG}_full31" \
    --base-png-dir "$P/base" --inner-png-dir "$P/cautious" --outer-png-dir "$P/amplified" 2>&1 \
    | grep -E 'raw|free|band_lrn|veto_lrn|contain_lrn|band_lrn_px|====|written'
done
echo "[full31] done $(date -u +%H:%M:%SZ)"
