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
- ≥16 GB VRAM recommended

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

pip install editdistance omegaconf "numpy<2" transformers
python -c "import lm_decoder; print('OK')"
```

---

## Data

1. Download `competitionData.tar.gz` from: https://doi.org/10.5061/dryad.x69p8czpq
2. Convert it with cffan's `formatCompetitionData.ipynb`.
3. Place the converted file at:

```bash
data/competitionData.pkl
```

For WER decoding, also download `languageModel.tar.gz` and extract it to:

```bash
data/languageModel/
```

Expected structure:

```text
data/
├── competitionData.pkl
└── languageModel/
    ├── TLG.fst
    ├── words.txt
    └── ...
```

---

## Training

Variant A runs 80 epochs. Variants B/C/D/E run 120–150 epochs.

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

### Quick acoustic evaluation: PER only

PER is computed on the test split using greedy CTC decoding.

For Variant A, decoding uses the monophone head directly.  
For Variants B–E, diphone outputs are marginalized to phoneme probabilities before CTC collapse.

```bash
python src/decode.py \
  --checkpoint experiments/<run>/best.pt \
  --variant <A|B|C|D|E> \
  --config configs/default.yaml
```

---

### Full decoding evaluation: PER + 3-gram WER

Run this in the `lm_decode` environment.

For fair comparison, all variants are decoded using the same LM settings:

```text
acoustic_scale = 0.8
beam = 18
blank_penalty = log(7)
```

No per-variant decoding-parameter tuning is used.

```bash
python src/decode.py \
  --checkpoint experiments/<run>/best.pt \
  --variant <A|B|C|D|E> \
  --config configs/default.yaml \
  --lm 3gram \
  --lm_dir data/languageModel \
  --acoustic_scale 0.8 \
  --beam 18 \
  --blank_penalty 1.9459
```

Implementation note: WER decoding uses raw acoustic logits, matching the official `speechBCI` inference style. For diphone-based variants, raw diphone logits are marginalized to phoneme-level logits using log-sum-exp before Kaldi/WFST decoding.

---

### WER with GPT-2 rescoring

Step 1: generate 100-best hypotheses with the 3-gram LM.

```bash
python src/decode.py \
  --checkpoint experiments/<run>/best.pt \
  --variant <A|B|C|D|E> \
  --config configs/default.yaml \
  --lm 3gram \
  --lm_dir data/languageModel \
  --acoustic_scale 0.8 \
  --beam 18 \
  --blank_penalty 1.9459 \
  --nbest 100 \
  --save_nbest experiments/<run>/nbest.pkl
```

Step 2: rescore the 100-best hypotheses with GPT-2.

```bash
python src/rescore.py \
  --nbest experiments/<run>/nbest.pkl
```

GPT-2 rescoring is included as an optional post-processing experiment. The main ablation comparison should use the same fixed 3-gram LM decoding settings for all variants.

---

## Results

Current acoustic decoding results on the test split:

| Rank | Variant | Core setting | Best PER (greedy) | 3-gram WER | Notes |
|------|---------|--------------|------------------:|-----------:|-------|
| 1 | E | Skip-diphone + smoothness, λ=0.005 | 18.99% | TBD | Best acoustic model |
| 2 | D | Skip-diphone, λ=0.001 | 19.50% | TBD | Skip-diphone auxiliary supervision |
| 3 | C | Diphone + smoothness, λ=0.01 | 19.58% | TBD | High smoothness weight, epoch 114 |
| 4 | C | Diphone + smoothness, λ=0.005 | 19.63% | TBD |  |
| 5 | B | Diphone baseline | 19.64% | TBD |  |
| 6 | C | Diphone + smoothness, λ=0.001 | 19.67% | TBD |  |
| 7 | A | Monophone CTC baseline | 20.94% | TBD | Acoustic baseline |

Variant E improves PER from 20.94% to 18.99%, corresponding to a 1.95 absolute-point reduction and a 9.3% relative reduction over the monophone baseline.

WER results are being re-evaluated after updating the LM decoding path to use raw acoustic logits instead of normalized log probabilities.

---

## Notes on Fair Comparison

This project reports two types of metrics:

1. **PER**, which evaluates the acoustic neural-to-phoneme model directly.
2. **WER**, which evaluates the full decoding pipeline with a language model.

For fair A/B/C/D/E comparison, all variants should use the same WER decoding parameters. This avoids giving one variant an unfair advantage through per-model decoder tuning.

---

## References

[1] F. R. Willett et al., A high-performance speech neuroprosthesis, *Nature* 620:1031–1036, 2023.  
[2] F. R. Willett et al., Data: A high-performance speech neuroprosthesis, *Dryad*, 2023. https://doi.org/10.5061/dryad.x69p8czpq  
[3] J. Li, T. Le, C. Fan, M. Chen, E. Shlizerman, Brain-to-Text Decoding with Context-Aware Neural Representations and LLMs, *arXiv:2411.10657*, 2024.  
[4] Brain-to-Text Benchmark '24, Eval.AI Challenge #2099. https://eval.ai/web/challenges/challenge-page/2099/overview  
[5] C. Fan et al., Neural Sequence Decoder, GitHub. https://github.com/cffan/neural_seq_decoder
