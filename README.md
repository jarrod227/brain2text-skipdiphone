# Brain-to-Text: Skip-Diphone Auxiliary Supervision and Temporal Smoothness Regularization

Course project extending [DCoND](https://arxiv.org/abs/2411.10657) with:
(i) a **skip-diphone auxiliary head** (z_{t-2}→z_t), and
(ii) a **temporal smoothness loss** on marginalized phoneme probabilities.

See [docs/proposal.pdf](docs/proposal.pdf) for full motivation and evaluation plan.

---

## Variants

| Variant | Description |
|---------|-------------|
| A | GRU + monophone CTC (NPTL baseline) |
| B | GRU + diphone CTC + marginalization (DCoND) |
| C | B + temporal smoothness loss |
| D | B + skip-diphone auxiliary head |
| E | B + skip-diphone + smoothness (full model) |

---

## Requirements

- Python 3.11, CUDA 11.8+, ≥16 GB VRAM

---

## Installation

```bash
conda create -n b2t python=3.11 && conda activate b2t
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### LM decode environment (for WER)

```bash
conda create -n lm_decode python=3.9 -y && conda activate lm_decode
pip install torch==1.13.1
conda install -c conda-forge cmake gcc gxx make -y

git clone https://github.com/fwillett/speechBCI.git
cd ~/speechBCI/LanguageModelDecoder/runtime/server/x86
mkdir -p build && cd build && cmake .. && make -j8 && cd ..
python setup.py install

pip install editdistance omegaconf "numpy<2" transformers
python -c "import lm_decoder; print('OK')"
```

---

## Data

1. Download `competitionData.tar.gz` from https://doi.org/10.5061/dryad.x69p8czpq
2. Convert with cffan's `formatCompetitionData.ipynb`
3. Place at `data/competitionData.pkl`

For WER: also download `languageModel.tar.gz` and extract to `data/languageModel/`.

---

## Training

A runs 80 epochs (~40 min on TITAN X); B/C/D/E run 120–150 epochs (~1 h each).

```bash
# Variant A (baseline)
nohup python src/train.py --variant A --config configs/default.yaml > experiments/variant_A.log 2>&1 &

# Variant B
nohup python src/train.py --variant B --config configs/default.yaml > experiments/variant_B.log 2>&1 &

# Variant C: sweep lambda_smooth
for lam in 1e-3 5e-3 1e-2; do
  nohup python src/train.py --variant C --lambda_smooth $lam --config configs/default.yaml \
    > experiments/variant_C_lam${lam}.log 2>&1 &
done

# Variants D and E (use best lambda_smooth from C sweep for E)
nohup python src/train.py --variant D --config configs/default.yaml > experiments/variant_D.log 2>&1 &
nohup python src/train.py --variant E --lambda_smooth <best> --config configs/default.yaml > experiments/variant_E.log 2>&1 &
```

Monitor: `tail -f experiments/<log_file>`. Multi-GPU: prefix with `CUDA_VISIBLE_DEVICES=N`.

---

## Evaluation

### PER (greedy CTC)

```bash
python src/decode.py --checkpoint experiments/<run>/best.pt --variant <A|B|C|D|E> \
    --config configs/default.yaml
```

### WER with n-gram LM

Run in `lm_decode` env:

```bash
python src/decode.py --checkpoint experiments/<run>/best.pt --variant <A|B|C|D|E> \
    --config configs/default.yaml --lm 3gram --lm_dir data/languageModel
```

### WER with GPT-2 rescoring

```bash
# Step 1: generate 100-best (lm_decode env)
python src/decode.py --checkpoint experiments/<run>/best.pt --variant <A|B|C|D|E> \
    --config configs/default.yaml --lm 3gram --lm_dir data/languageModel \
    --nbest 100 --save_nbest experiments/<run>/nbest.pkl

# Step 2: rescore and sweep alpha
python src/rescore.py --nbest experiments/<run>/nbest.pkl

# Step 3: fix best alpha
python src/rescore.py --nbest experiments/<run>/nbest.pkl --alpha <best>
```

Note: n-gram-only WER is ~50%; GPT-2 rescoring brings it to ~25–30%.
cffan's published ~23% WER uses OPT-6B rescoring on the competition partition.

---

## Results

*To be filled after experiments.*

| Variant | PER (greedy) | WER (n-gram) | WER (GPT-2) |
|---------|-------------|-------------|-------------|
| A | — | — | — |
| B | — | — | — |
| C | — | — | — |
| D | — | — | — |
| E | — | — | — |

---

## References

[1] F. R. Willett et al., A high-performance speech neuroprosthesis, *Nature* 620:1031–1036, 2023.  
[2] F. R. Willett et al., Data: A high-performance speech neuroprosthesis, *Dryad*, 2023. https://doi.org/10.5061/dryad.x69p8czpq  
[3] J. Li, T. Le, C. Fan, M. Chen, E. Shlizerman, Brain-to-Text Decoding with Context-Aware Neural Representations and LLMs, *arXiv:2411.10657*, 2024.  
[4] Brain-to-Text Benchmark '24, Eval.AI Challenge #2099. https://eval.ai/web/challenges/challenge-page/2099/overview  
[5] C. Fan et al., Neural Sequence Decoder, GitHub. https://github.com/cffan/neural_seq_decoder
