"""Smoothness-loss correctness: zero on constant posteriors, exact on a
hand-computed case, and invariant to padded frames beyond each length."""

import torch

from loss import smoothness_loss


def test_constant_posterior_has_zero_smoothness():
    # (B=1, T=4, C+1=3); identical every frame -> no frame-to-frame change
    probs = torch.tensor([[[0.2, 0.3, 0.5]] * 4])
    lengths = torch.tensor([4])
    assert torch.allclose(smoothness_loss(probs, lengths), torch.tensor(0.0))


def test_known_value():
    # Two frames, phone dims flip [1,0] -> [0,1]; blank (last dim) excluded.
    # ||diff||^2 over phone dims = (-1)^2 + 1^2 = 2; one valid pair -> loss 2.0
    probs = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]]])
    lengths = torch.tensor([2])
    assert torch.allclose(smoothness_loss(probs, lengths), torch.tensor(2.0))


def test_padded_frames_are_ignored():
    # length=2, so frame index 2 (garbage) must not contribute to the loss.
    probs = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [9.9, 9.9, 9.9]]])
    lengths = torch.tensor([2])
    assert torch.allclose(smoothness_loss(probs, lengths), torch.tensor(2.0))


def test_batch_is_mean_of_per_sample():
    probs = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],   # per-sample loss 2.0
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],   # per-sample loss 0.0
    ])
    lengths = torch.tensor([2, 2])
    assert torch.allclose(smoothness_loss(probs, lengths), torch.tensor(1.0))
