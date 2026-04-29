# Brain-to-Text: Skip-Diphone Auxiliary Supervision and Temporal Smoothness Regularization

This project extends the [DCoND](https://arxiv.org/abs/2411.10657) framework for neural speech decoding with two additions:
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
# 1. Clone and install upstream speechBCI (provides GRU encoder, data utilities)
git clone https://github.com/fwillett/speechBCI.git
pip install -e speechBCI/NeuralDecoder

# 2. Install this project's dependencies
pip install -r requirements.txt
```

For the language model decoder (KenLM-based), follow the instructions in
`speechBCI/LanguageModelDecoder/README.md` after step 1.

---

## Data

Download the Brain-to-Text '24 dataset from Dryad (DOI: 10.5061/dryad.x69p8czpq, Version 4)
and place it under `data/`. See [data/README.md](data/README.md) for the expected layout.

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
# Decode with 3-gram LM and report PER / WER
python src/decode.py --checkpoint experiments/<run_name>/best.pt --lm 3gram

# Decode with 5-gram LM
python src/decode.py --checkpoint experiments/<run_name>/best.pt --lm 5gram
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
