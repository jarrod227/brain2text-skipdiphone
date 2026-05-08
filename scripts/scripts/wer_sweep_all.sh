#!/usr/bin/env bash
# Run 3-gram wer_sweep sequentially over all seed42 experiments to rank
# variants and produce per-run WER numbers. After identifying the best
# variant, run a separate 5-gram sweep on that variant only (the 5-gram
# TLG.fst is ~42GB; sweeping all variants on 5-gram is wasteful and the
# optimal acoustic_scale differs from 3-gram anyway).
#
# Usage (from repo root, in lm_decode env):
#   nohup bash scripts/wer_sweep_all.sh > experiments/wer_sweep_all.log 2>&1 &
#
# Each run's detailed output is also saved to <run_dir>/wer_sweep_3gram.log.

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=(
    experiments/variant_A_alpha0.6_beta0.1_lam0.001_seed42
    experiments/variant_B_alpha0.6_beta0.1_lam0.001_seed42
    experiments/variant_C_alpha0.6_beta0.1_lam0.001_seed42
    experiments/variant_C_alpha0.6_beta0.1_lam0.005_seed42
    experiments/variant_C_alpha0.6_beta0.1_lam0.01_seed42
    experiments/variant_D_alpha0.6_beta0.05_lam0.001_seed42
    experiments/variant_D_alpha0.6_beta0.1_lam0.001_seed42
    experiments/variant_D_alpha0.6_beta0.2_lam0.001_seed42
    experiments/variant_D_alpha0.6_beta0.3_lam0.001_seed42
    experiments/variant_E_alpha0.6_beta0.1_lam0.005_seed42
)

TOTAL=${#RUNS[@]}
IDX=0

for DIR in "${RUNS[@]}"; do
    IDX=$((IDX + 1))
    NAME=$(basename "$DIR")
    VARIANT=$(echo "$NAME" | cut -d_ -f2)

    echo "========================================================"
    echo "[${IDX}/${TOTAL}] $(date '+%Y-%m-%d %H:%M:%S')  $NAME  (variant $VARIANT)"
    echo "========================================================"

    python scripts/wer_sweep.py \
        --checkpoint "$DIR/best_dev.pt" \
        --variant    "$VARIANT" \
        --lm         3gram \
        --lm_dir     data/languageModel \
        --out        "$DIR/wer_sweep_3gram.csv" \
        2>&1 | tee "$DIR/wer_sweep_3gram.log"

    echo "[${IDX}/${TOTAL}] DONE  $NAME"
    echo
done

echo "========================================================"
echo "All sweeps complete.  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Summary of best WER per run:"
echo "--------------------------------------------------------"
# CSV columns: 1=variant, 2=acoustic_scale, 3=blank_penalty, 4=beam, 5=PER, 6=WER
for DIR in "${RUNS[@]}"; do
    NAME=$(basename "$DIR")
    CSV="$DIR/wer_sweep_3gram.csv"
    if [[ -f "$CSV" ]]; then
        BEST=$(tail -n +2 "$CSV" | sort -t, -k6 -n | head -1)
        AC=$(echo  "$BEST" | cut -d, -f2)
        BP=$(echo  "$BEST" | cut -d, -f3)
        WER=$(echo "$BEST" | cut -d, -f6)
        printf "%-55s  WER=%.4f  ac=%-5s  bp=%s\n" "$NAME" "$WER" "$AC" "$BP"
    else
        printf "%-55s  (no CSV)\n" "$NAME"
    fi
done
