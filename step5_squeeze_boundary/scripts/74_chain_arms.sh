#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/74_chain_arms.sh
#
#   Run the two arms of the squeeze experiment back to back on ONE card, so the
#   second arm starts by itself when the first one finishes:
#
#       arm A  SQ-A  angular band  (fingerprint driven)   <- headline arm
#       arm I  SQ-I  isotropic band (ablation)
#
#   usage: bash scripts/74_chain_arms.sh [gpu] [epochs]
#          setsid nohup bash scripts/74_chain_arms.sh 7 500 \
#              > logs/74_chain.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${1:-7}"
EPOCHS="${2:-500}"

echo "[74] chain start $(date -u +%Y-%m-%dT%H:%M:%SZ) | gpu=$GPU epochs=$EPOCHS"
echo "[74] NOTE the two arms share the card; arm I only starts after A is done."

bash scripts/71_train_squeeze.sh A "$GPU" "$EPOCHS"
echo "[74] arm A finished rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

bash scripts/71_train_squeeze.sh I "$GPU" "$EPOCHS"
echo "[74] arm I finished rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[74] chain done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
