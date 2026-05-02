# Brain-to-Text: Skip-Diphone Auxiliary Supervision and Temporal Smoothness Regularization

This repository contains the implementation for a course project that proposes to extend the [DCoND](https://arxiv.org/abs/2411.10657) framework for neural speech decoding with two additions:
(i) a **skip-diphone auxiliary head** predicting z_{t-2}→z_t context pairs, and
(ii) a **temporal smoothness loss** on marginalized phoneme probabilities.

For full motivation, method, and evaluation plan, see [docs/proposal.pdf](docs/proposal.pdf).

---

## Background

The standard Brain-to-Text pipeline (Willett et al., Nature 2023) maps intracortical neural features to
monophone probabilities via a bidirectional GRU under CTC loss. DCoND improves this by predicting
diphones (z_{t-1}→z_t, C²=1600 classes) and marginalizing back to phonemes. This project tests whether
broader articulatory context (skip-diphones) and output regularization (smoothness) further reduce PER/WER.

Five ablation variants are studied:

| Variant | Description |
|---------|-------------|
| A | GRU + monophone CTC (NPTL baseline) |
| B | GRU + standard diphone + marginalization (DCoND) |
| C | B + temporal smoothness loss |
| D | B + skip-diphone auxiliary head |
| E | B + skip-diphone + smoothness (full model) |

---

## Requirements

- Python 3.9
- CUDA 11.8+ / cuDNN 8+ recommended
- GPU: ≥16 GB VRAM (A100, V100, or RTX 3090 all sufficient)

---

## Installation

Create and activate a dedicated conda environment first (tested with Python 3.11):

```bash
conda create -n b2t python=3.11
conda activate b2t
```

Then install dependencies in this order:

```bash
# Run from anywhere:
# 1. Install PyTorch with CUDA support (adjust cu118 to match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
# See https://pytorch.org/get-started/locally/ for the right command

# Run from inside the brain2text-skipdiphone/ directory:
# 2. Install this project's dependencies
pip install -r requirements.txt

# Run from outside the brain2text-skipdiphone/ directory:
# 3. (Optional) Install language model decoder for n-gram rescoring
git clone https://github.com/fwillett/speechBCI.git
# Follow speechBCI/LanguageModelDecoder/README.md to compile and install KenLM
```

### Optional: separate LM decode environment (recommended for WER)

`--lm` decoding depends on `speechBCI/LanguageModelDecoder` Python bindings (`lm_decoder`),
which are often easiest to compile/run in a separate environment from training.
This project was trained in `b2t` (PyTorch >=2.1), while LM decoding was validated
in a separate `lm_decode` env with PyTorch 1.13.1.

```bash
# 1) Create dedicated env for LM decode
conda create -n lm_decode python=3.9 -y
conda activate lm_decode
pip install torch==1.13.1

# 2) Build dependencies (Ubuntu/Debian)
sudo apt install -y cmake gcc g++ make

# 3) Build speechBCI LanguageModelDecoder runtime
cd ~/speechBCI/LanguageModelDecoder/runtime/server/x86
mkdir -p build && cd build
cmake ..
make -j8
cd ..

# 4) Install Python bindings
python setup.py install
pip install editdistance omegaconf "numpy<2"

# 5) Verify
python -c "import lm_decoder; print('lm_decoder import OK')"
```

If `lm_decoder` import fails or `--lm` crashes, re-check that the runtime was built
inside the currently activated environment and that `data/languageModel/` contains
`TLG.fst` and `words.txt`.

---

## Data

See [data/README.md](data/README.md) for full download and conversion instructions.

Summary:
1. Download `competitionData.tar.gz` (required, 3.67 GB) and optionally `languageModel.tar.gz` (14.11 GB, for WER) from https://doi.org/10.5061/dryad.x69p8czpq
2. Convert `.mat` files to `.pkl` using cffan's `formatCompetitionData.ipynb`
3. Set `data_path: "data/competitionData.pkl"` in `configs/default.yaml`

---

## Training

Run all variants sequentially (one GPU). Each variant takes ~1 hour on a TITAN X.

```bash
nohup python src/train.py --variant A --config configs/default.yaml > experiments/variant_A.log 2>&1 &
nohup python src/train.py --variant B --config configs/default.yaml > experiments/variant_B.log 2>&1 &

# Variant C: sweep lambda_smooth to find the best value before running D/E
nohup python src/train.py --variant C --lambda_smooth 1e-4 --config configs/default.yaml > experiments/variant_C_lam1e-4.log 2>&1 &
nohup python src/train.py --variant C --lambda_smooth 1e-3 --config configs/default.yaml > experiments/variant_C_lam1e-3.log 2>&1 &
nohup python src/train.py --variant C --lambda_smooth 5e-3 --config configs/default.yaml > experiments/variant_C_lam5e-3.log 2>&1 &
nohup python src/train.py --variant C --lambda_smooth 1e-2 --config configs/default.yaml > experiments/variant_C_lam1e-2.log 2>&1 &

nohup python src/train.py --variant D --config configs/default.yaml > experiments/variant_D.log 2>&1 &
nohup python src/train.py --variant E --config configs/default.yaml > experiments/variant_E.log 2>&1 &
```

Run one at a time. Monitor progress with `tail -f experiments/<log_file>`.
Wait for `Epoch 050` before starting the next variant.
Checkpoints are written to `experiments/<run_name>/`.

**Tip — two GPUs:** PyTorch defaults to GPU 0. If a second GPU is available and idle,
run a second variant in parallel on GPU 1:
```bash
CUDA_VISIBLE_DEVICES=1 nohup python src/train.py --variant B --config configs/default.yaml > experiments/variant_B.log 2>&1 &
```
Check GPU status with `watch -n 1 nvidia-smi`.

---

## Evaluation

```bash
# PER only (no LM required)
python src/decode.py --checkpoint experiments/<run_name>/best.pt --variant <A|B|C|D|E> --config configs/default.yaml

# PER + WER with 3-gram LM (run in lm_decode env after compiling speechBCI LM decoder)
python src/decode.py --checkpoint experiments/<run_name>/best.pt --variant <A|B|C|D|E> --config configs/default.yaml --lm 3gram --lm_dir data/languageModel
```

---

## Results

*To be filled after experiments.*

| Variant | PER (3-gram) | WER (3-gram) |
|---------|-------------|-------------|
| A | — | — |
| B | — | — |
| C | — | — |
| D | — | — |
| E | — | — |

---

## References

[1] F. R. Willett et al., A high-performance speech neuroprosthesis, *Nature* 620:1031–1036, 2023.  
[2] F. R. Willett et al., Data for: A high-performance speech neuroprosthesis, *Dryad*, 2023. https://doi.org/10.5061/dryad.x69p8czpq  
[3] J. Li, T. Le, C. Fan, M. Chen, E. Shlizerman, Brain-to-Text Decoding with Context-Aware Neural Representations and LLMs, *arXiv:2411.10657*, 2024.  
[4] Brain-to-Text Benchmark '24, Eval.AI Challenge #2099. https://eval.ai/web/challenges/challenge-page/2099/overview  
[5] C. Fan et al., Neural Sequence Decoder (reference implementation of DCoND), GitHub. https://github.com/cffan/neural_seq_decoder
