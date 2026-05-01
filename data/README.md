# Data

## Step 1: Download raw data

Download from Dryad (DOI: 10.5061/dryad.x69p8czpq, Version 4).
In your browser, click each file to get its signed download URL, then on your
remote server run from inside `brain2text-skipdiphone/`:

```bash
# Required: competition splits as .mat files (3.67 GB)
wget -O data/competitionData.tar.gz "<competitionData.tar.gz URL>"

# Optional: 3-gram language model for WER evaluation (14.11 GB)
wget -O data/languageModel.tar.gz "<languageModel.tar.gz URL>"

# Extract
tar -xzf data/competitionData.tar.gz -C data/
tar -xzf data/languageModel.tar.gz  -C data/   # if downloaded

# Remove archives to save disk space
rm data/*.tar.gz
```

Files not needed: `derived`, `diagnosticBlocks`, `sentences`, `tuningTasks`,
`languageModel_5gram`.

## Step 2: Convert .mat files to .pkl

The downloaded `competitionData/` contains `.mat` files (one per recording
session/day), loaded with `scipy.io.loadmat`. Use cffan's notebook to convert
them to a single `.pkl` file and apply blockwise z-score normalization:

```bash
# Run from outside brain2text-skipdiphone/:
git clone https://github.com/cffan/neural_seq_decoder.git
# Open and run: neural_seq_decoder/notebooks/formatCompetitionData.ipynb
# Point the notebook to your data/competitionData/ directory
# Output: a single competitionData.pkl file
```

Move the output to `data/`:
```bash
mv competitionData.pkl data/
```

## Step 3: Update config

Set the path in `configs/default.yaml`:
```yaml
data_path: "data/competitionData.pkl"
```

## Expected layout

```
data/
├── competitionData/          # extracted .mat files (one per day, 24 total)
│   ├── t12.2022.04.28.mat
│   ├── t12.2022.05.05.mat
│   └── ...
├── competitionData.pkl       # converted by formatCompetitionData.ipynb
└── languageModel/            # optional, for WER
    ├── TLG.fst
    ├── tokens.txt
    └── words.txt
```

## Neural features

Each trial provides:
- `tx1[:, :128]`: threshold crossing counts, area 6v (128 channels)
- `spikePow[:, :128]`: spike band power, area 6v (128 channels)
- Concatenated → 256 features per 20 ms time bin

Phoneme set: 39 phonemes + SIL = 40 classes (blank at index 40).

This directory is excluded from version control (see `.gitignore`).
