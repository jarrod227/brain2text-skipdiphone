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

### Multi-seed runs (variance estimation)

```bash
bash scripts/multi_seed.sh   # trains Variant E with seeds {42, 1, 2}
# Override:
SEEDS="42 1 2 3 4" VARIANT=E LAMBDA=5e-3 bash scripts/multi_seed.sh
```

The notebook (`notebooks/results.ipynb`) groups runs by
`(variant, alpha, beta, lambda)` and reports mean ± std across seeds.

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

---

### WER: 3-gram WFST decoding

Run this in the `lm_decode` environment.

WER decoding uses the official `speechBCI` WFST decoder. Defaults are aligned
with speechBCI's actual baseline (`rnn_step3_baselineRNNInference.ipynb`):

```text
acoustic_scale = 0.8
beam           = 17
blank_penalty  = log(2) ~= 0.693
log_priors     = log of training-set phoneme frequencies (Kaldi class order)
```

Earlier versions of this repo passed `acoustic_scale=1.5`, `blank_penalty=0.0`,
and zero log-priors to `lm_decoder.DecodeNumpy`, which is *not* what speechBCI
actually uses and inflates WER significantly. `decode.py` now estimates
`log_priors` from the training-split phoneme distribution (with the blank
prior pinned to 0, in Kaldi class order `[blank, SIL, phone_0, ...]`) and
passes it through; use `--no_log_priors` to revert to the legacy zero-prior
behavior.

```bash
conda activate lm_decode
cd ~/brain2text-skipdiphone

python src/decode.py \
  --checkpoint experiments/<run>/best.pt \
  --variant <A|B|C|D|E> \
  --config configs/default.yaml \
  --lm 3gram \
  --lm_dir data/languageModel
```

Implementation note: WER decoding uses raw acoustic logits. For diphone-based variants, raw diphone logits are marginalized to phoneme-level logits using log-sum-exp before Kaldi/WFST decoding.

### WER hyperparameter sweep

PER gains from the acoustic objective only show up in WER if `acoustic_scale`
(and to a lesser extent `blank_penalty`) is recalibrated for the new model.
`scripts/wer_sweep.py` loads a checkpoint once and grid-scans the WFST
hyperparameters, writing a CSV that the notebook reads back as a heatmap.

```bash
conda activate lm_decode
cd ~/brain2text-skipdiphone

python scripts/wer_sweep.py \
  --checkpoint experiments/<run>/best.pt \
  --variant <A|B|C|D|E> \
  --lm 3gram --lm_dir data/languageModel \
  --acoustic_scales 0.3 0.5 0.8 1.0 1.2 \
  --blank_penalties 0.0 0.69 1.0 2.0 \
  --out experiments/<run>/wer_sweep.csv
```

Both ranges are centered on speechBCI's baseline (`acoustic_scale=0.8`,
`blank_penalty=log(2)≈0.69`). The sweep also subtracts training-set
log-priors by default; pass `--no_log_priors` for the legacy zero-prior
behavior.

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

Current acoustic decoding results on the test split:

| Rank | Variant | Core setting | Best PER (greedy) | WER | Notes |
|------|---------|--------------|------------------:|----:|-------|
| 1 | E | Skip-diphone + smoothness, λ=0.005 | 18.99% | TBD | Best acoustic model |
| 2 | D | Skip-diphone, λ=0.001 | 19.50% | TBD | Skip-diphone auxiliary supervision |
| 3 | C | Diphone + smoothness, λ=0.01 | 19.58% | TBD | High smoothness weight |
| 4 | C | Diphone + smoothness, λ=0.005 | 19.63% | TBD |  |
| 5 | B | Diphone baseline | 19.64% | TBD |  |
| 6 | C | Diphone + smoothness, λ=0.001 | 19.67% | TBD |  |
| 7 | A | Monophone CTC baseline | 20.94% | TBD | Acoustic baseline |

Variant E improves PER from 20.94% to 18.99%, corresponding to a 1.95 absolute-point reduction and a 9.3% relative reduction over the monophone baseline.

Earlier decoding experiments showed that WER improves only modestly under 3-gram/5-gram WFST decoding and GPT-2 rescoring. This suggests that phoneme-level acoustic gains do not directly translate into word-level gains without stronger acoustic-LM calibration, a stronger baseline decoder, or the full unpruned/LLM rescoring pipeline.

---

## Notes on Fair Comparison

This project reports two types of metrics:

1. **PER**, which evaluates the acoustic neural-to-phoneme model directly.
2. **WER**, which evaluates the full decoding pipeline with a language model.

The main project contribution is the acoustic model objective: skip-diphone auxiliary supervision and temporal smoothness regularization. For word-level evaluation, this project follows the standard speechBCI/DCoND-style WFST and optional n-best rescoring pipeline.

For fair A/B/C/D/E comparison, all variants should use the same WER decoding settings.

---

## References

[1] F. R. Willett et al., A high-performance speech neuroprosthesis, *Nature* 620:1031–1036, 2023.  
[2] F. R. Willett et al., Data: A high-performance speech neuroprosthesis, *Dryad*, 2023. https://doi.org/10.5061/dryad.x69p8czpq  
[3] J. Li, T. Le, C. Fan, M. Chen, E. Shlizerman, Brain-to-Text Decoding with Context-Aware Neural Representations and LLMs, *arXiv:2411.10657*, 2024.  
[4] Brain-to-Text Benchmark '24, Eval.AI Challenge #2099. https://eval.ai/web/challenges/challenge-page/2099/overview  
[5] C. Fan et al., Neural Sequence Decoder, GitHub. https://github.com/cffan/neural_seq_decoder  
[6] F. Willett et al., speechBCI, GitHub. https://github.com/fwillett/speechBCI
