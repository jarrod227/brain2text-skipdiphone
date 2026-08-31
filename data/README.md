# Data

## Step 1: Download raw data

Download from Dryad (DOI: 10.5061/dryad.x69p8czpq, Version 4).
Dryad blocks automated downloaders (wget/curl), so download the files
manually in your browser and transfer them to the remote server with `scp`.

**In your browser:** go to the Dryad page and click **Download** next to
`competitionData.tar.gz` (3.67 GB, required) and optionally
`languageModel.tar.gz` (14.11 GB, only needed for WER evaluation).

**Transfer from your local machine to the remote server** (run in a local
terminal, e.g. PowerShell on Windows):

```bash
scp /path/to/competitionData.tar.gz user@server:~/brain2text-skipdiphone/data/
scp /path/to/languageModel.tar.gz  user@server:~/brain2text-skipdiphone/data/  # optional
```

**On the remote server**, from inside `brain2text-skipdiphone/`:

```bash
# Extract
tar -xzf data/competitionData.tar.gz -C data/
tar -xzf data/languageModel.tar.gz   -C data/   # if downloaded

# Remove archives to save disk space
rm data/*.tar.gz
```

Files not needed: `derived`, `diagnosticBlocks`, `sentences`, `tuningTasks`,
`languageModel_5gram`.

## Step 2: Convert .mat files to .pkl

The downloaded `competitionData/` contains `.mat` files (one per recording
session/day), loaded with `scipy.io.loadmat`. Use the notebook in this repo
to convert them to a single `.pkl` file and apply blockwise z-score
normalization:

```bash
# Open and run from the repo root:
jupyter notebook notebooks/formatCompetitionData.ipynb
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
├── competitionData/          # extracted .mat files
│   ├── train/                # 24 .mat files (t12.2022.04.28.mat, ...)
│   ├── test/                 # 24 .mat files
│   └── competitionHoldOut/   # 15 .mat files (no labels)
├── competitionData.pkl       # converted by notebooks/formatCompetitionData.ipynb
└── languageModel/            # optional, for WER evaluation
    ├── TLG.fst
    ├── G.fst
    ├── LG.fst
    ├── L.fst
    ├── T.fst
    ├── tokens.txt
    ├── units.txt
    ├── words.txt
    └── lexicon_numbers.txt
```

## Converted pkl structure

`competitionData.pkl` is a single dict of splits, each holding one entry per
recording day:

```
{'train':       [day_0, ..., day_23],   # 24 days
 'test':        [day_0, ..., day_23],   # 24 days
 'competition': [day_0, ..., day_14]}   # 15 days, no labels
```

Each day is a dict of five parallel fields, all indexed by the same trial
index `i` — that is, `i` picks one sentence, and the five fields hold that
sentence's neural data, phonemes, lengths, and text.

Throughout: **`N`** = number of sentences recorded that day, **`T`** = number
of 20 ms frames in one sentence (varies from sentence to sentence).

| Field | Type / shape | Contents |
|-------|--------------|----------|
| `sentenceDat` | **list** of `N` arrays, each `(T, 256)` float | neural features |
| `phonemes` | 2-D array `(N, 500)` int | phoneme ids, **1-indexed**, zero-padded |
| `phoneLens` | 1-D array `(N,)` int | true phoneme count per sentence |
| `timeSeriesLens` | 1-D array `(N,)` int | true frame count per sentence |
| `transcriptions` | **list** of `N` `str` | reference sentence |

Note that `sentenceDat` is a Python list, not a 2-D array: `T` differs per
sentence, so the per-sentence arrays cannot be stacked into one rectangle.
`phonemes` *can* be, because it is zero-padded to a fixed width of 500. So
`len(day["sentenceDat"])` gives `N` (the sentence count), while
`day["sentenceDat"][i]` is the `(T, 256)` array for one sentence.

Three things to watch, all handled in `src/dataset.py`:

- `phonemes` is **1-indexed** (the converter adds 1 so 0 can serve as padding);
  subtract 1 to get the 0–39 targets the model expects.
- The `500` width of `phonemes` is padding — slice with `phoneLens`.
- `sentenceDat[i]` may carry trailing padding rows — slice with `timeSeriesLens`.

## Neural features

Each trial provides:
- `tx1[:, :128]`: threshold crossing counts, area 6v (128 channels)
- `spikePow[:, :128]`: spike band power, area 6v (128 channels)
- Concatenated → 256 features per 20 ms time bin

Phoneme set: 39 phonemes + SIL = 40 classes (blank at index 40).

## Note on transcript normalization

The pkl stores `transcriptions` as raw strings (original casing and
punctuation from the .mat files). The WFST decoder emits lowercase
strings without punctuation. `decode.py` applies `_normalize_transcript()`
before WER comparison to align the two; without this step WER is
artificially inflated.

This directory is excluded from version control (see `.gitignore`).
