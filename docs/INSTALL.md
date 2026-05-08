# Installation

## Requirements

- Python 3.11
- CUDA 11.8+
- ≥16 GB VRAM recommended for training
- ≥32 GB RAM recommended for 5-gram WFST decoding (especially with the
  unpruned rescoring graph `G_no_prune.fst`)

---

## Training environment (`b2t`)

Used for training and PER evaluation.

```bash
conda create -n b2t python=3.11 -y
conda activate b2t

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## WFST decode environment (`lm_decode`)

Used for WER decoding. Builds the official `speechBCI` WFST language-model
decoder from source.

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

---

## GPT-2 rescoring (optional)

Use the `b2t` environment or any PyTorch 2.x environment:

```bash
conda activate b2t
pip install transformers editdistance
```
