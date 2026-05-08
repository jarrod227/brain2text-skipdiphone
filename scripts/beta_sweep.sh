#!/usr/bin/env bash
# Beta ablation: train Variant D (diphone + skip-diphone) with multiple beta values
# to test how much of E's PER gain comes from the skip-diphone head vs. its
# regularization effect.
#
# Run from the repo root after the conda env is activated:
#   conda activate b2t
#   bash scripts/beta_sweep.sh
#
# Each run uses CUDA_VISIBLE_DEVICES=0 by default. Set GPU_ID=N to override.

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/default.yaml}"
LOG_DIR="${LOG_DIR:-experiments}"

mkdir -p "$LOG_DIR"

for beta in 0.05 0.1 0.2 0.3; do
    log="${LOG_DIR}/variant_D_beta${beta}.log"
    echo "=== Variant D, beta=${beta} -> ${log} ==="
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    nohup python -u src/train.py \
        --variant D \
        --config "$CONFIG" \
        --beta "$beta" \
        > "$log" 2>&1 &
    wait
done

echo "Done. Logs in $LOG_DIR/variant_D_beta*.log"
