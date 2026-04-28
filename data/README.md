# Data

Download the Brain-to-Text '24 dataset from Dryad:

> DOI: [10.5061/dryad.x69p8czpq](https://doi.org/10.5061/dryad.x69p8czpq) (Version 4)

After downloading and extracting, the directory should look like:

```
data/
├── competitionData/
│   ├── train/          # TFRecord files for training sessions
│   ├── test/           # TFRecord files for the test partition (40 sentences)
│   └── competitionHoldOut/   # held-out partition for leaderboard submission
│
├── languageModel/
│   ├── 3gram.arpa      # 3-gram phoneme-to-text language model
│   └── 5gram.arpa      # 5-gram phoneme-to-text language model (optional)
│
└── derived/
    ├── *.tfrecord       # pre-processed TFRecords (baseline validation)
    └── rnn_checkpoint/  # baseline RNN checkpoint from Willett et al.
```

This directory is excluded from version control (see `.gitignore`).
