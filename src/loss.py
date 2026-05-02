"""
Training objective (Eq. 1 from proposal):

  L_total = L_CTC_phoneme
          + alpha  * L_CTC_std_diphone
          + beta   * L_CTC_skip_diphone
          + lambda * L_smooth

  L_smooth = (1 / T-1) * sum_{t=2}^{T} || p_t - p_{t-1} ||_2^2
"""

import torch
import torch.nn.functional as F


def smoothness_loss(phoneme_probs, lengths):
    """
    Args:
        phoneme_probs: (B, T, num_phonemes+1)  -- last dim is CTC blank
        lengths:       (B,) actual sequence lengths
    Returns:
        scalar mean smoothness loss

    Smoothness is computed over the phoneme dims only (excluding blank).
    Blank is intentionally excluded: CTC relies on blank spiking at phoneme
    boundaries, so penalizing its frame-to-frame change would conflict with
    the CTC objective.
    """
    B, T, _ = phoneme_probs.shape
    phone_only = phoneme_probs[..., :-1]                          # (B, T, C)
    diff = phone_only[:, 1:, :] - phone_only[:, :-1, :]           # (B, T-1, C)
    sq   = (diff ** 2).sum(dim=-1)                                # (B, T-1)

    mask = (torch.arange(T - 1, device=sq.device).unsqueeze(0)
            < (lengths - 1).clamp(min=0).unsqueeze(1)).float()
    denom = mask.sum().clamp(min=1.0)
    return (sq * mask).sum() / denom


def compute_loss(mono_log_probs, diphone_log_probs, skip_log_probs,
                 phoneme_probs, targets, input_lengths, target_lengths,
                 alpha, beta, lambda_smooth, variant, num_phonemes, num_diphones):
    """
    Args:
        mono_log_probs:    (T, B, num_phonemes+1)
        diphone_log_probs: (T, B, num_diphones+1)
        skip_log_probs:    (T, B, num_diphones+1)
        phoneme_probs:     (B, T, num_phonemes+1)  -- last dim is blank
        targets:           dict with keys 'mono', 'diphone', 'skip_diphone'
        input_lengths:     (B,)
        target_lengths:    dict with keys matching targets
        alpha, beta, lambda_smooth: loss weights
        variant:           one of 'A', 'B', 'C', 'D', 'E'

    Returns:
        total loss (scalar), dict of component losses for logging
    """
    # blank sits at the last index: num_phonemes for mono, num_diphones for diphone/skip
    def ctc_mono(lp, tgt, ilen, tlen):
        return F.ctc_loss(lp, tgt, ilen, tlen, blank=num_phonemes, zero_infinity=True)

    def ctc_dip(lp, tgt, ilen, tlen):
        return F.ctc_loss(lp, tgt, ilen, tlen, blank=num_diphones, zero_infinity=True)

    components = {}

    if variant == "A":
        l_mono = ctc_mono(mono_log_probs, targets["mono"],
                          input_lengths, target_lengths["mono"])
        components["ctc_mono"] = l_mono
        return l_mono, components

    # Variants B / C / D / E all use the diphone CTC as main objective
    l_mono    = ctc_mono(mono_log_probs, targets["mono"],
                         input_lengths, target_lengths["mono"])
    l_diphone = ctc_dip(diphone_log_probs, targets["diphone"],
                        input_lengths, target_lengths["diphone"])

    total = l_mono + alpha * l_diphone
    components["ctc_mono"]    = l_mono
    components["ctc_diphone"] = l_diphone

    if variant in ("D", "E"):
        l_skip = ctc_dip(skip_log_probs, targets["skip_diphone"],
                         input_lengths, target_lengths["skip_diphone"])
        total += beta * l_skip
        components["ctc_skip"] = l_skip

    if variant in ("C", "E"):
        l_smooth = smoothness_loss(phoneme_probs, input_lengths)
        total += lambda_smooth * l_smooth
        components["smooth"] = l_smooth

    return total, components
