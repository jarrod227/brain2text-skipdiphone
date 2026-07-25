"""Shared pytest fixtures and path setup.

The `src/` modules import each other by bare name (e.g. `from dataset import ...`),
so `src/` must be on `sys.path` for the tests to import them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _make_trial(phones_0indexed, n_frames, feat_dim=256, pad_width=500):
    """Build one trial in the on-disk pkl layout.

    The pkl stores phonemes 1-indexed and zero-padded (formatCompetitionData
    adds +1 so 0 is padding); the dataset subtracts 1 on load.
    """
    phones_1indexed = [p + 1 for p in phones_0indexed]
    padded = np.zeros(pad_width, dtype=np.int64)
    padded[: len(phones_1indexed)] = phones_1indexed
    return padded, len(phones_1indexed), n_frames


@pytest.fixture
def tiny_pkl(tmp_path):
    """Write a minimal two-trial pkl and return (path, expected metadata)."""
    import pickle

    trials = [
        ([3, 5, 7, 9], 50),   # 4 phones, 50 frames
        ([2, 4], 30),         # 2 phones, 30 frames (too short for skip-diphone)
    ]
    phonemes, phone_lens, ts_lens, sent_dat, transcripts = [], [], [], [], []
    for i, (phones, nf) in enumerate(trials):
        padded, plen, frames = _make_trial(phones, nf)
        phonemes.append(padded)
        phone_lens.append(plen)
        ts_lens.append(frames)
        sent_dat.append(np.random.randn(frames, 256).astype(np.float32))
        transcripts.append(f"trial {i}")

    day0 = {
        "sentenceDat": sent_dat,
        "phonemes": np.stack(phonemes),
        "phoneLens": np.array(phone_lens, dtype=np.int64),
        "timeSeriesLens": np.array(ts_lens, dtype=np.int64),
        "transcriptions": transcripts,
    }
    data = {"train": [day0]}
    path = tmp_path / "tiny.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)

    return path, trials
