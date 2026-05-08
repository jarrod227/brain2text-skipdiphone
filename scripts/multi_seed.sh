#!/usr/bin/env bash
# Train Variant E (full model, lambda=5e-3) with 3 seeds for variance estimation.
# Run names land in experiments/variant_E_alpha0.6_beta0.1_lam0.005_seed{N}/.
#
# Usage:
#   conda activate b2t
#   bash scripts/multi_seed.sh                  # runs all 3 sequentially
#   GPU_ID=1 bash scripts/multi_seed.sh         # pin a single GPU
#   SEEDS="42 1 2 3 4" bash scripts/multi_seed.sh
#
# To parallelize across GPUs, run the loop body manually with
# CUDA_VISIBLE_DEVICES=N for each seed.

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/default.yaml}"
LOG_DIR="${LOG_DIR:-experiments}"
SEEDS="${SEEDS:-42 1 2}"
VARIANT="${VARIANT:-E}"
LAMBDA="${LAMBDA:-5e-3}"

mkdir -p "$LOG_DIR"

for seed in $SEEDS; do
    log="${LOG_DIR}/variant_${VARIANT}_lam${LAMBDA}_seed${seed}.log"
    echo "=== Variant ${VARIANT}, seed=${seed} -> ${log} ==="
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    nohup python -u src/train.py \
        --variant "$VARIANT" \
        --lambda_smooth "$LAMBDA" \
        --seed "$seed" \
        --deterministic \
        --config "$CONFIG" \
        > "$log" 2>&1 &
    wait
done

echo "Done. Logs in $LOG_DIR/"
