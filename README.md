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

## Requirements

- Python 3.11
- CUDA 11.8+
- ≥16 GB VRAM recommended for training
- Large RAM is recommended for 5-gram WFST decoding, especially if using the unpruned rescoring graph

---

## Installation

```bash
conda create -n b2t python=3.11 -y
conda activate b2t

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### LM decode environment for WER

WER decoding uses the official `speechBCI` WFST language-model decoder.

```bash
conda create -n lm_decode python=3.9 -y
conda activate lm_decode

pip install torch==1.13.1
conda install -c conda-forge cmake gcc gxx make -y

git clone https://github.com/fwillett/speechBCI.git
cd ~/speechBCI/LanguageModelDecoder/runtime/server/x86
mkdir -p build
cd build
cmake ..
make -j8
cd ..
python setup.py install

pip install editdistance omegaconf "numpy<2"
python -c "import lm_decoder; print('OK')"
```

For GPT-2 rescoring, use the `b2t` environment or another PyTorch 2.x environment:

```bash
conda activate b2t
pip install transformers editdistance
```

---

## Data

1. Download `competitionData.tar.gz` from: https://doi.org/10.5061/dryad.x69p8czpq
2. Convert it with cffan's `formatCompetitionData.ipynb`.
3. Place the converted file at:

```bash
data/competitionData.pkl
```

For 3-gram WER decoding, download `languageModel.tar.gz` and extract it to:

```bash
data/languageModel/
```

Expected 3-gram structure:

```text
data/
├── competitionData.pkl
└── languageModel/
    ├── TLG.fst
    ├── G.fst
    ├── G_no_prune.fst
    ├── words.txt
    └── ...
```

For optional 5-gram WER decoding, download and extract the 5-gram language model. In my setup, the 5-gram files are located at:

```text
data/speech_5gram/lang_test/
├── TLG.fst
├── G.fst
├── G_no_prune.fst
└── words.txt
```

> ⚠️ **RAM requirement.** The unpruned 5-gram graph (`G_no_prune.fst`) is
> typically 5–10 GB and the WFST decoder must hold it in memory. Without
> ≥32 GB RAM the decoder will OOM or thrash. Skip this section if your
> hardware is limited; the 3-gram path is sufficient for the main results.

If `G_no_prune.fst` is too large for available RAM, it can be temporarily renamed so that decoding uses the pruned 5-gram graph only:

```bash
mv data/speech_5gram/lang_test/G_no_prune.fst \
   data/speech_5gram/lang_test/G_no_prune.fst.bak
```

Restore it with:

```bash
mv data/speech_5gram/lang_test/G_no_prune.fst.bak \
   data/speech_5gram/lang_test/G_no_prune.fst
```

---

## Methodology

To avoid using the same split for both checkpoint selection and final
reporting, the pkl `train` bucket is partitioned into:

- **train** (~90%) — used for gradient updates.
- **dev** (~10%) — every `dev_stride`-th trial within each recording day,
  deterministic across runs. Used to pick `best_dev.pt`.

The pkl `test` split is evaluated only every `eval_test_every` epochs as a
tracking signal during training. **Final test PER is reported by running
`decode.py` on `best_dev.pt`**, which is what this README's results table
should use. Each training run also writes `final.pt` at the last epoch and
a `summary.json` with `{best_dev_epoch, best_dev_per, test_per_at_best_dev,
final_dev_per, final_test_per}`.

For seed-noise control, runs are repeated with multiple seeds. Run
directories include the seed (`..._seed42`, `..._seed1`, ...) so the
notebook can aggregate as mean ± std. Set `cudnn_deterministic: true` (or
pass `--deterministic`) to force cuDNN GRU into deterministic mode (~20%
slower) for tighter seed comparisons.

Setting `dev_stride: 0` falls back to the legacy protocol of using `test`
for both validation and reporting; the trainer warns when this is in
effect.

> **Migration note.** Checkpoints saved before the dev-split methodology
> change are `best.pt`, selected on test PER, and **should not be used
> for final reporting** — they constitute a test-set leak. The Results
> table below is decoded from `best_dev.pt` checkpoints under the current
> protocol.

> **Effect on absolute PER.** Under the dev-split protocol, the
> A→E gap shrinks from 1.95pp (legacy cherry-picked) to 0.73pp (honest).
> Most of the shrinkage comes from removing the test-epoch cherry-pick
> on E (~0.8pp inflation). A also improves by ~0.5pp under the new
> protocol, because A@150 (sanity check) replaces the under-trained A@80.
> Concurrent works on this dataset that report numbers under the legacy
> convention are therefore not directly comparable in absolute terms.

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
