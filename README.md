# Brain-to-Text: Skip-Diphone Auxiliary Supervision and Temporal Smoothness Regularization

Course project extending [DCoND](https://arxiv.org/abs/2411.10657) with:
(i) a **skip-diphone auxiliary head** (`z_{t-2} → z_t`), and
(ii) a **temporal smoothness loss** on marginalized phoneme probabilities.

See [docs/proposal.pdf](docs/proposal.pdf) for full motivation and evaluation plan.

---

## Variants

| Variant | Description |
|---------|-------------|
| A | GRU + monophone CTC (NPTL baseline) |
| B | GRU + diphone CTC + marginalization (DCoND-style baseline) |
| C | B + temporal smoothness loss |
| D | B + skip-diphone auxiliary head |
| E | B + skip-diphone + temporal smoothness loss (full model) |

---

## Setup

- **Environments**: see [docs/INSTALL.md](docs/INSTALL.md). Two conda envs are
  used: `b2t` for training/PER and `lm_decode` for WFST WER decoding.
- **Data**: see [data/README.md](data/README.md) for downloading
  `competitionData.tar.gz`, converting to `competitionData.pkl`, and
  extracting `languageModel.tar.gz` (3-gram) or `languageModel_5gram.tar.gz`
  (optional, ≥32 GB RAM if `G_no_prune.fst` is kept).

---

## Methodology

The pkl `train` bucket is partitioned into **train** (~90%) and **dev**
(~10%, every `dev_stride`-th trial per day, deterministic). `best_dev.pt`
is selected on dev PER; pkl `test` is held out and only decoded once on
the final checkpoint. Set `dev_stride: 0` to fall back to the legacy
protocol of using `test` for both validation and reporting.

> **Note on absolute PER.** Numbers from the legacy "lowest test PER
> across epochs" protocol cherry-pick the best test epoch and are not
> directly comparable to dev-split numbers. Under the dev-split protocol
> here, the A→E gap is 0.73pp (honest) vs 1.95pp under legacy.

---

## Training

Variant A runs 80 epochs. Variants B/C/D/E run 120–150 epochs because the diphone and skip-diphone variants have larger output spaces and additional objectives.

```bash
# Variant A: monophone baseline
nohup python src/train.py \
  --variant A \
  --config configs/default.yaml \
  > experiments/variant_A.log 2>&1 &

# Variant B: diphone baseline
nohup python src/train.py \
  --variant B \
  --config configs/default.yaml \
  > experiments/variant_B.log 2>&1 &

# Variant C: diphone + smoothness
for lam in 1e-3 5e-3 1e-2; do
  nohup python src/train.py \
    --variant C \
    --lambda_smooth $lam \
    --config configs/default.yaml \
    > experiments/variant_C_lam${lam}.log 2>&1 &
done

# Variant D: diphone + skip-diphone
nohup python src/train.py \
  --variant D \
  --config configs/default.yaml \
  > experiments/variant_D.log 2>&1 &

# Variant E: full model
nohup python src/train.py \
  --variant E \
  --lambda_smooth 5e-3 \
  --config configs/default.yaml \
  > experiments/variant_E_lam5e-3.log 2>&1 &
```

`train.py` also accepts:
- `--alpha` (std-diphone CTC weight) — variants B/C/D/E
- `--beta` (skip-diphone CTC weight) — variants D/E
- `--lambda_smooth` (smoothness weight) — variants C/E
- `--seed` (RNG seed)
- `--num_epochs` (override default by variant)
- `--deterministic` (force deterministic cuDNN; ~20% slower)

All of α / β / λ / seed are reflected in the run directory name, so sweep
runs land in distinct directories under `experiments/`.

### β ablation (Variant D)

```bash
bash scripts/beta_sweep.sh   # trains D with beta in {0.05, 0.1, 0.2, 0.3}
```

### A@150 sanity check

```bash
bash scripts/sanity_a150.sh   # Variant A trained for 150 epochs (vs. default 80)
```

Confirms whether the gain of E (150 epochs) over A (80 epochs) comes from
the objective rather than just training time.

Monitor training:

```bash
tail -f experiments/<log_file>
```

For multi-GPU systems, prefix commands with:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id>
```

---

## Evaluation

### Step 1 — PER (greedy CTC)

For Variant A the monophone head is decoded directly; for B–E, raw diphone
logits are marginalized to phoneme probabilities (log-sum-exp) before
CTC collapse. Decode every run on the test split:

```bash
for run in experiments/variant_*/; do
  v=$(basename "$run" | cut -d_ -f2)
  python src/decode.py \
    --checkpoint "$run/best_dev.pt" \
    --variant "$v" \
    --split test \
    --save_summary "$run/test_summary.json" \
    --config configs/default.yaml
done
```

`--save_summary` writes `{per, wer, split, ...}` so the notebook can pull
final numbers without re-running decode.

### Step 2 — WER (3-gram sweep, all variants)

WFST decoding uses the official speechBCI decoder (run in `lm_decode`).
Defaults: `acoustic_scale=0.8`, `beam=17`, `blank_penalty=log(2)≈0.693`,
`log_priors=zeros` (matching speechBCI / cffan; pass `--log_priors` to
opt into training-prior subtraction).

`acoustic_scale` and `blank_penalty` are LM-specific, so we sweep them
rather than rely on the defaults. The 3-gram TLG.fst is small enough to
sweep on every variant:

```bash
conda activate lm_decode
nohup bash scripts/wer_sweep_all.sh > experiments/wer_sweep_all.log 2>&1 &
```

Iterates all 10 seed42 runs sequentially, writes per-run CSVs to
`experiments/<run>/wer_sweep_3gram.csv`, and prints a summary of the
best `(acoustic_scale, blank_penalty, WER)` cell per run.

### Step 3 — WER (5-gram sweep, best variant only)

Optimal `(acoustic_scale, blank_penalty)` shifts with the LM (compare
speechBCI's 3-gram `ac=0.8, bp=log(2)` vs cffan's 5-gram `ac=0.5, bp=log(7)`).
The 5-gram TLG.fst is ~42 GB, so we only sweep it on the best variant
identified in Step 2:

```bash
nohup python scripts/wer_sweep.py \
    --checkpoint experiments/variant_E_alpha0.6_beta0.1_lam0.005_seed42/best_dev.pt \
    --variant E \
    --lm 5gram \
    --lm_dir data/speech_5gram/lang_test \
    --acoustic_scales 0.1 0.2 0.3 0.5 0.8 \
    --blank_penalties 0.0 0.69 1.0 2.0 \
    --out experiments/variant_E_alpha0.6_beta0.1_lam0.005_seed42/wer_sweep_5gram.csv \
    > experiments/variant_E_alpha0.6_beta0.1_lam0.005_seed42/wer_sweep_5gram.log 2>&1 &
```

The ac range extends below cffan's 0.5 because the 5-gram LM is more
informative than 3-gram. `wer_sweep.py` builds the decoder once per
`acoustic_scale` and reuses it across all `blank_penalty` values, so the
~42 GB FST is loaded 5 times instead of 20 on this 5×4 grid.

### Step 4 (optional) — GPT-2 n-best rescoring

Not claimed as a project contribution, but supported. Generate 100-best
with `--nbest 100 --save_nbest <path>`, then run `src/rescore.py
--nbest <path> --model_name gpt2 --alpha 0.5 --acoustic_scale <Step-3 ac>`.
The combination follows speechBCI/DCoND:
`total = α·GPT + (1−α)·LM + acoustic_scale·acoustic`.

---

## Results

Acoustic decoding results on the test split, under the dev-split protocol
(`best_dev.pt` selected on the held-out dev set, no test leakage):

| Rank | Variant | Core setting | Test PER (greedy) | WER | Notes |
|------|---------|--------------|------------------:|----:|-------|
| 1 | E | Skip-diphone + smoothness, λ=0.005 | 19.75% | TBD | Best acoustic model |
| 2 | D | Skip-diphone, β=0.2 | 20.12% | TBD | Best β for D |
| 3 | D | Skip-diphone, β=0.1 (default) | 20.20% | TBD |  |
| 4 | C | Diphone + smoothness, λ=0.01 | 20.36% | TBD |  |
| 4 | C | Diphone + smoothness, λ=0.001 | 20.36% | TBD |  |
| 4 | D | Skip-diphone, β=0.05 | 20.36% | TBD |  |
| 7 | B | Diphone baseline | 20.41% | TBD |  |
| 8 | C | Diphone + smoothness, λ=0.005 | 20.45% | TBD |  |
| 9 | D | Skip-diphone, β=0.3 | 20.47% | TBD |  |
| 10 | A | Monophone CTC baseline (150 ep) | 20.48% | TBD | Acoustic baseline |

Variant E improves test PER from 20.48% (A) to 19.75% (E) — a **0.73 absolute-point reduction** (3.6% relative). The gap is smaller than under the legacy "lowest test PER across epochs" protocol (1.95pp) because that protocol cherry-picked the best test epoch per run, inflating the apparent contribution.

**Sanity check (A@150):** Even when A is trained for 150 epochs (matching E's training budget), it reaches only 20.48% PER, still 0.73pp behind E. This isolates the contribution to the **objective** rather than to additional training time.

**β / λ observations:**
- D's β has a clear shallow optimum at β=0.2 (20.12%); β=0.05 and β=0.3 both lose ~0.3pp.
- C's λ ∈ {1e-3, 5e-3, 1e-2} produces nearly identical PER (20.36 / 20.45 / 20.36); smoothness alone is not strongly tunable on this acoustic head.
- E (smoothness + skip-diphone) outperforms either component alone, suggesting they are complementary.

WER columns will be filled by `scripts/wer_sweep_all.sh` (3-gram, all
variants) for ranking, and a separate 5-gram sweep on the best variant
for the headline WER. The 5-gram-optimal `(acoustic_scale, blank_penalty)`
differs from 3-gram (cffan reports `ac=0.5, bp=log(7)` for 5-gram vs
speechBCI's `ac=0.8, bp=log(2)` for 3-gram), so re-tuning is required
rather than reusing the 3-gram-best parameters.

---

## Notes on Fair Comparison

The project contribution is the **acoustic-model objective** (skip-diphone
auxiliary supervision + temporal smoothness regularization), measured by
**PER**. WER is reported as a downstream check using the standard
speechBCI/DCoND-style WFST + optional n-best rescoring pipeline.

For fair A/B/C/D/E comparison, all variants must be decoded with the
**same** `(acoustic_scale, blank_penalty, log_priors, lm)` settings. The
recommended workflow is to run `scripts/wer_sweep.py` once on the best
variant to find the optimal `(acoustic_scale, blank_penalty)`, then decode
every variant with that fixed pair. See `Methodology` for the
checkpoint-selection protocol.

---

## References

[1] F. R. Willett et al., A high-performance speech neuroprosthesis, *Nature* 620:1031–1036, 2023.  
[2] F. R. Willett et al., Data: A high-performance speech neuroprosthesis, *Dryad*, 2023. https://doi.org/10.5061/dryad.x69p8czpq  
[3] J. Li, T. Le, C. Fan, M. Chen, E. Shlizerman, Brain-to-Text Decoding with Context-Aware Neural Representations and LLMs, *arXiv:2411.10657*, 2024.  
[4] Brain-to-Text Benchmark '24, Eval.AI Challenge #2099. https://eval.ai/web/challenges/challenge-page/2099/overview  
[5] C. Fan et al., Neural Sequence Decoder, GitHub. https://github.com/cffan/neural_seq_decoder  
[6] F. Willett et al., speechBCI, GitHub. https://github.com/fwillett/speechBCI
