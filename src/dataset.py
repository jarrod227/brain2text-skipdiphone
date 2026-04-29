"""
Loads Brain-to-Text '24 data from TFRecords (via TensorFlow) and wraps
them in a PyTorch Dataset / DataLoader.

Each TFRecord file corresponds to one recording session (day).
The day_id is derived from the file's index in the sorted file list.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class BrainToTextDataset(Dataset):
    def __init__(self, tfrecord_paths, num_phonemes=40, debug_subset=False):
        self.num_phonemes = num_phonemes
        self.samples = self._load_tfrecords(tfrecord_paths, debug_subset)

    def _load_tfrecords(self, paths, debug_subset):
        import tensorflow as tf

        feature_spec = {
            "neuralFeatures":    tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "phoneSeq":          tf.io.FixedLenSequenceFeature([], tf.int64,   allow_missing=True),
            "neuralFeatureSize": tf.io.FixedLenFeature([], tf.int64),
        }

        samples = []
        # Each file = one recording day; file index becomes the day_id
        for day_id, path in enumerate(paths):
            raw_ds = tf.data.TFRecordDataset(path)
            for raw in raw_ds:
                ex       = tf.io.parse_single_example(raw, feature_spec)
                feat_dim = int(ex["neuralFeatureSize"].numpy())
                neural   = ex["neuralFeatures"].numpy().reshape(-1, feat_dim)
                phones   = ex["phoneSeq"].numpy().tolist()
                diphones  = self._phones_to_diphones(phones)
                skip_dip  = self._phones_to_skip_diphones(phones)
                samples.append({
                    "neural":       torch.tensor(neural,    dtype=torch.float32),
                    "day_id":       day_id,
                    "mono":         torch.tensor(phones,    dtype=torch.long),
                    "diphone":      torch.tensor(diphones,  dtype=torch.long),
                    "skip_diphone": torch.tensor(skip_dip,  dtype=torch.long),
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

    mono_cat    = torch.cat(mono)
    diphone_cat = torch.cat(diphone)
    skip_cat    = torch.cat(skip_dip)

    mono_lens    = torch.tensor([len(t) for t in mono],     dtype=torch.long)
    diphone_lens = torch.tensor([len(t) for t in diphone],  dtype=torch.long)
    skip_lens    = torch.tensor([len(t) for t in skip_dip], dtype=torch.long)

    return {
        "neural":   neural_padded,
        "lengths":  lengths,
        "day_ids":  day_ids,
        "targets": {
            "mono":         mono_cat,
            "diphone":      diphone_cat,
            "skip_diphone": skip_cat,
        },
        "target_lengths": {
            "mono":         mono_lens,
            "diphone":      diphone_lens,
            "skip_diphone": skip_lens,
        },
    }


def make_dataloader(tfrecord_paths, batch_size, num_phonemes=40,
                    debug_subset=False, shuffle=True, num_workers=4):
    ds = BrainToTextDataset(tfrecord_paths, num_phonemes, debug_subset)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=num_workers,
                      pin_memory=True)
