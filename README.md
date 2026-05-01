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

Install in this order:

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

---

## Data

See [data/README.md](data/README.md) for full download and conversion instructions.

Summary:
1. Download `competitionData.tar.gz` (required, 3.67 GB) and optionally `languageModel.tar.gz` (14.11 GB, for WER) from https://doi.org/10.5061/dryad.x69p8czpq
2. Convert `.mat` files to `.pkl` using cffan's `formatCompetitionData.ipynb`
3. Set `data_path: "data/competitionData.pkl"` in `configs/default.yaml`

---

## Training

```bash
# Train a specific ablation variant (A–E)
python src/train.py --variant E --config configs/default.yaml

# Sweep smoothness weight lambda
python src/train.py --variant C --lambda_smooth 1e-3 --config configs/default.yaml
python src/train.py --variant C --lambda_smooth 5e-3 --config configs/default.yaml
python src/train.py --variant C --lambda_smooth 1e-2 --config configs/default.yaml
```

Checkpoints and logs are written to `experiments/<run_name>/`.

---

## Evaluation

```bash
# PER only (no LM required)
python src/decode.py --checkpoint experiments/<run_name>/best.pt

# PER + WER with 3-gram LM (requires speechBCI LanguageModelDecoder)
python src/decode.py --checkpoint experiments/<run_name>/best.pt --lm 3gram
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
