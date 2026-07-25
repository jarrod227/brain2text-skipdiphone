"""Dataset target construction, shapes, and collate masking."""

import torch

from dataset import BrainToTextDataset, collate_fn


def test_phones_to_diphones_and_skip():
    ds = object.__new__(BrainToTextDataset)   # bypass __init__ / pkl load
    ds.num_phonemes = 40
    phones = [3, 5, 7, 9]

    # diphone id = prev * C + curr  over ADJACENT pairs -> len N-1
    assert ds._phones_to_diphones(phones) == [3 * 40 + 5, 5 * 40 + 7, 7 * 40 + 9]

    # skip-diphone id = phones[i] * C + phones[i+2]  -> len N-2
    assert ds._phones_to_skip_diphones(phones) == [3 * 40 + 7, 5 * 40 + 9]


def test_skip_diphone_empty_for_short_sequences():
    ds = object.__new__(BrainToTextDataset)
    ds.num_phonemes = 40
    assert ds._phones_to_skip_diphones([2, 4]) == []      # N=2 -> no skip pairs
    assert ds._phones_to_diphones([2, 4]) == [2 * 40 + 4]  # N=2 -> one diphone


def test_load_shapes_and_1indexing(tiny_pkl):
    path, trials = tiny_pkl
    ds = BrainToTextDataset(str(path), split="train", num_phonemes=40, dev_stride=0)
    assert len(ds) == len(trials)

    s0 = ds[0]
    phones0, n_frames0 = trials[0]
    # phonemes are stored 1-indexed on disk and decremented on load
    assert s0["mono"].tolist() == phones0
    assert s0["neural"].shape == (n_frames0, 256)
    assert s0["diphone"].numel() == len(phones0) - 1
    assert s0["skip_diphone"].numel() == len(phones0) - 2


def test_collate_padding_and_masking(tiny_pkl):
    path, trials = tiny_pkl
    ds = BrainToTextDataset(str(path), split="train", num_phonemes=40, dev_stride=0)
    batch = collate_fn([ds[0], ds[1]])

    max_frames = max(nf for _, nf in trials)
    assert batch["neural"].shape == (2, max_frames, 256)
    # true lengths recorded separately so the model can ignore padding
    assert batch["lengths"].tolist() == [trials[0][1], trials[1][1]]

    # padded region of the shorter trial must be exactly zero
    short_len = trials[1][1]
    assert torch.count_nonzero(batch["neural"][1, short_len:]) == 0

    # concatenated targets and per-sample lengths stay consistent
    total_mono = sum(len(p) for p, _ in trials)
    assert batch["targets"]["mono"].numel() == total_mono
    assert batch["target_lengths"]["mono"].sum().item() == total_mono
