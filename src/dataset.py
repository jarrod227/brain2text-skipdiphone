"""
Loads Brain-to-Text '24 data from the .pkl file produced by
cffan/neural_seq_decoder's formatCompetitionData.ipynb.

The pkl contains a single dict:
  {
    'train':       [dataset_day0, ..., dataset_day23],   # 24 days
    'test':        [dataset_day0, ..., dataset_day23],   # 24 days
    'competition': [dataset_day0, ..., dataset_day14],   # 15 days
  }

Each per-day dataset has:
  sentenceDat:    list of np.ndarray (T, 256)  -- tx1[:128] + spikePow[:128], z-scored
  phonemes:       np.ndarray (N, 500)          -- zero-padded phoneme sequences
  phoneLens:      np.ndarray (N,)              -- actual phoneme counts
  timeSeriesLens: np.ndarray (N,)              -- actual frame counts
  transcriptions: list of str
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class BrainToTextDataset(Dataset):
    def __init__(self, pkl_path, split="train", num_phonemes=40, debug_subset=False):
        self.num_phonemes = num_phonemes
        self.samples = self._load(pkl_path, split, debug_subset)

    def _load(self, pkl_path, split, debug_subset):
        with open(pkl_path, "rb") as f:
            all_data = pickle.load(f)

        day_datasets = all_data[split]   # list of per-day dicts
        samples = []

        for day_id, day_data in enumerate(day_datasets):
            n_trials = len(day_data["sentenceDat"])
            for i in range(n_trials):
                neural    = day_data["sentenceDat"][i].astype(np.float32)  # (T, 256)
                phone_len = int(day_data["phoneLens"][i])
                phones    = day_data["phonemes"][i][:phone_len].tolist()
                frame_len = int(day_data["timeSeriesLens"][i])
                neural    = neural[:frame_len]

                diphones = self._phones_to_diphones(phones)
                skip_dip = self._phones_to_skip_diphones(phones)

                samples.append({
                    "neural":       torch.tensor(neural,   dtype=torch.float32),
                    "day_id":       day_id,
                    "mono":         torch.tensor(phones,   dtype=torch.long),
                    "diphone":      torch.tensor(diphones, dtype=torch.long),
                    "skip_diphone": torch.tensor(skip_dip, dtype=torch.long),
                })

                if debug_subset and len(samples) >= 200:
                    return samples

        return samples

    def _phones_to_diphones(self, phones):
        C = self.num_phonemes
        return [p_prev * C + p_curr
                for p_prev, p_curr in zip(phones[:-1], phones[1:])]

    def _phones_to_skip_diphones(self, phones):
        C = self.num_phonemes
        return [phones[i] * C + phones[i + 2]
                for i in range(len(phones) - 2)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    neural   = [s["neural"]       for s in batch]
    mono     = [s["mono"]         for s in batch]
    diphone  = [s["diphone"]      for s in batch]
    skip_dip = [s["skip_diphone"] for s in batch]

    neural_padded = pad_sequence(neural, batch_first=True)
    lengths  = torch.tensor([n.shape[0] for n in neural], dtype=torch.long)
    day_ids  = torch.tensor([s["day_id"] for s in batch], dtype=torch.long)

    mono_lens    = torch.tensor([len(t) for t in mono],     dtype=torch.long)
    diphone_lens = torch.tensor([len(t) for t in diphone],  dtype=torch.long)
    skip_lens    = torch.tensor([len(t) for t in skip_dip], dtype=torch.long)

    return {
        "neural":   neural_padded,
        "lengths":  lengths,
        "day_ids":  day_ids,
        "targets": {
            "mono":         torch.cat(mono),
            "diphone":      torch.cat(diphone),
            "skip_diphone": torch.cat(skip_dip),
        },
        "target_lengths": {
            "mono":         mono_lens,
            "diphone":      diphone_lens,
            "skip_diphone": skip_lens,
        },
    }


def make_dataloader(pkl_path, split, batch_size, num_phonemes=40,
                    debug_subset=False, shuffle=True, num_workers=4):
    ds = BrainToTextDataset(pkl_path, split, num_phonemes, debug_subset)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=num_workers,
                      pin_memory=True)
