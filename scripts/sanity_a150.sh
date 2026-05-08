#!/usr/bin/env bash
# A@150 sanity check: run the mono-only baseline for 150 epochs (instead of
# the default 80) to confirm Variant A is already saturated by epoch 80,
# i.e. that the gain of E (150 epochs) over A is from objective rather than
# training-time differences.
#
# Usage:
#   conda activate b2t
#   bash scripts/sanity_a150.sh

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/default.yaml}"
LOG_DIR="${LOG_DIR:-experiments}"

mkdir -p "$LOG_DIR"

log="${LOG_DIR}/variant_A_epochs150.log"
echo "=== Variant A, num_epochs=150 -> ${log} ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" \
nohup python -u src/train.py \
    --variant A \
    --num_epochs 150 \
    --config "$CONFIG" \
    > "$log" 2>&1 &
wait

echo "Done. Log at $log"
