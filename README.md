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

### PER: greedy CTC

PER is computed by greedy CTC decoding on the requested split.

For Variant A, decoding uses the monophone head directly.  
For Variants B–E, diphone outputs are marginalized to phoneme probabilities before CTC collapse.

```bash
python src/decode.py \
  --checkpoint experiments/<run>/best_dev.pt \
  --variant <A|B|C|D|E> \
  --split test \
  --save_summary experiments/<run>/test_summary.json \
  --config configs/default.yaml
```

`--split` accepts `train` / `dev` / `test`. `--save_summary` writes a JSON
with `{per, wer, split, ...}` so the notebook can read final test numbers
without re-running decode.

To populate the Results table, decode every run directory in one pass:

```bash
# Decode all best_dev.pt checkpoints on the test split
for run in experiments/variant_*/; do
  # run name encodes variant: variant_E_alpha0.6_beta0.1_lam0.005_seed42
  v=$(basename "$run" | cut -d_ -f2)
  python src/decode.py \
    --checkpoint "$run/best_dev.pt" \
    --variant "$v" \
    --split test \
    --save_summary "$run/test_summary.json" \
    --config configs/default.yaml
done
```

---

### WER: 3-gram WFST decoding

Run this in the `lm_decode` environment.

WER decoding uses the official `speechBCI` WFST decoder. Defaults are aligned
with speechBCI's actual baseline (`rnn_step3_baselineRNNInference.ipynb`):

```text
acoustic_scale = 0.8
beam           = 17
blank_penalty  = log(2) ~= 0.693
log_priors     = zeros (matches speechBCI baseline + cffan eval_competition)
```

Earlier versions of this repo passed `acoustic_scale=1.5` and
`blank_penalty=0.0` — those values are not what speechBCI's baseline
notebook actually uses and inflate WER. The defaults above are now correct.

`log_priors` is left at zeros to match both reference implementations
(`speechBCI/.../lmDecoderUtils.py:185-186` and
`neural_seq_decoder/scripts/eval_competition.py:110-114` both pass `None`
which becomes `np.zeros`). `decode.py` can also estimate per-class
log-priors from the training-set phoneme distribution and subtract them at
decode time; this is opt-in via `--log_priors` and should be treated as
experimental, since `TLG.fst` is built assuming zero priors.

Single-run WER decode:

```bash
conda activate lm_decode
cd ~/brain2text-skipdiphone

python src/decode.py \
  --checkpoint experiments/<run>/best_dev.pt \
  --variant <A|B|C|D|E> \
  --config configs/default.yaml \
  --lm 3gram \
  --lm_dir data/languageModel
```

Implementation note: WER decoding uses raw acoustic logits. For diphone-based variants, raw diphone logits are marginalized to phoneme-level logits using log-sum-exp before Kaldi/WFST decoding.

### Recommended workflow: 3-gram sweep all → 5-gram sweep best variant

`acoustic_scale` and `blank_penalty` are LM-specific. The optimal values
shift between 3-gram and 5-gram (compare speechBCI 3-gram defaults
`ac=0.8, bp=log(2)` vs cffan 5-gram defaults `ac=0.5, bp=log(7)`). So the
recommended pipeline is:

**Step 1 — 3-gram sweep on every variant** (fast; ranks the ablation table
and fills the per-variant WER column):

```bash
conda activate lm_decode
nohup bash scripts/wer_sweep_all.sh > experiments/wer_sweep_all.log 2>&1 &
```

This iterates all 10 seed42 runs sequentially, writes per-run CSVs to
`experiments/<run>/wer_sweep_3gram.csv`, and prints a final summary of
the best (acoustic_scale, blank_penalty, WER) cell per run.

**Step 2 — 5-gram sweep on the best variant only** (re-tune around cffan's
5-gram defaults; the 5-gram TLG.fst is ~42GB so we don't sweep it on every
variant):

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

The acoustic_scale range extends below cffan's 5-gram default (0.5) because
the 5-gram LM is more informative than 3-gram, so the optimum can sit even
lower (more weight on the LM, less on the acoustic logits).
`wer_sweep.py` builds the WFST decoder once per `acoustic_scale` and reuses
it across all `blank_penalty` values, so the 5-gram (~42GB) FST is loaded
5 times instead of 20 on this 5×4 grid.

### Manual single-cell sweep

If you just want to explore one combination instead of the full pipeline:

```bash
python scripts/wer_sweep.py \
  --checkpoint experiments/<run>/best_dev.pt \
  --variant <A|B|C|D|E> \
  --lm 3gram --lm_dir data/languageModel \
  --acoustic_scales 0.3 0.5 0.8 1.0 1.2 \
  --blank_penalties 0.0 0.69 1.0 2.0 \
  --out experiments/<run>/wer_sweep.csv
```

Both ranges are centered on speechBCI's 3-gram baseline (`acoustic_scale=0.8`,
`blank_penalty=log(2)≈0.69`). `log_priors` defaults to zeros to match
speechBCI; pass `--log_priors` to opt into training-set prior subtraction.

---

### Optional: GPT-2 combined rescoring

First generate 100-best hypotheses with the 3-gram LM:

```bash
conda activate lm_decode
cd ~/brain2text-skipdiphone

python src/decode.py \
  --checkpoint experiments/<run>/best.pt \
  --variant <A|B|C|D|E> \
  --config configs/default.yaml \
  --lm 3gram \
  --lm_dir data/languageModel \
  --nbest 100 \
  --save_nbest experiments/<run>/nbest.pkl
```

Then rescore in the `b2t` environment:

```bash
conda activate b2t
cd ~/brain2text-skipdiphone

python src/rescore.py \
  --nbest experiments/<run>/nbest.pkl \
  --model_name gpt2 \
  --alpha 0.5 \
  --acoustic_scale 0.8
```

The rescoring score follows the speechBCI/DCoND-style combination:

```text
total_score = alpha * GPT_score
            + (1 - alpha) * old_LM_score
            + acoustic_scale * acoustic_score
```

GPT-2 rescoring is optional and is not claimed as a project contribution. The project contribution is the acoustic-model objective.

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
