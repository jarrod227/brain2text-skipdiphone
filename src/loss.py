"""
Training objective (Eq. 1 from proposal):

  L_total = L_CTC_phoneme
          + alpha  * L_CTC_std_diphone
          + beta   * L_CTC_skip_diphone
          + lambda * L_smooth

  L_smooth = mean over batch of  (1 / (T_i - 1)) * sum_{t=2}^{T_i} || p_t - p_{t-1} ||_2^2

The smoothness term is computed per sample over its valid length T_i, then
averaged across the batch (matching the proposal formulation).
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

    Per-sample mean is computed first (sum of squared diffs over valid
    frame-pairs, divided by that sample's valid pair count), then averaged
    across the batch. This matches the proposal's formulation
    L_smooth = (1 / (T-1)) * sum_t ||p_t - p_{t-1}||^2 .
    """
    B, T, _ = phoneme_probs.shape
    phone_only = phoneme_probs[..., :-1]                          # (B, T, C)
    diff = phone_only[:, 1:, :] - phone_only[:, :-1, :]           # (B, T-1, C)
    sq = (diff ** 2).sum(dim=-1)                                  # (B, T-1)

    mask = (torch.arange(T - 1, device=sq.device).unsqueeze(0)
            < (lengths - 1).clamp(min=0).unsqueeze(1)).float()
    pair_counts = mask.sum(dim=1).clamp(min=1.0)                  # (B,)
    per_sample = (sq * mask).sum(dim=1) / pair_counts             # (B,)
    return per_sample.mean()


def compute_loss(outputs, targets, target_lengths,
                 alpha, beta, lambda_smooth, variant,
                 num_phonemes, num_diphones):
    """
    Args:
        outputs:        dict from SkipDiphoneDecoder.forward(...).
                        Required keys depend on variant; see model.py.
        targets:        dict with keys 'mono', 'diphone', 'skip_diphone'
                        (only the keys needed for `variant` are read)
        target_lengths: dict with the same keys as `targets`
        alpha, beta, lambda_smooth: loss weights
        variant:        one of 'A', 'B', 'C', 'D', 'E'

    Returns:
        total loss (scalar), dict of component losses for logging
    """
    input_lengths = outputs["enc_lengths"]

    # blank sits at the last index: num_phonemes for mono, num_diphones for diphone/skip
    def ctc_mono(lp, tgt, ilen, tlen):
        return F.ctc_loss(lp, tgt, ilen, tlen, blank=num_phonemes, zero_infinity=True)

    def ctc_dip(lp, tgt, ilen, tlen):
        return F.ctc_loss(lp, tgt, ilen, tlen, blank=num_diphones, zero_infinity=True)

    components = {}

    if variant == "A":
        l_mono = ctc_mono(outputs["mono_log_probs"], targets["mono"],
                          input_lengths, target_lengths["mono"])
        components["ctc_mono"] = l_mono
        return l_mono, components

    # Variants B / C / D / E all use the diphone CTC as main objective.
    # l_mono is computed from marginalized phoneme probs so that both loss
    # terms train the diphone head (no competing gradient from a separate
    # mono head — that head is not built for these variants).
    phoneme_probs = outputs["phoneme_probs"]
    phone_log_probs = torch.log(phoneme_probs.clamp(min=1e-10)).permute(1, 0, 2)

    l_mono = ctc_mono(phone_log_probs, targets["mono"],
                      input_lengths, target_lengths["mono"])
    l_diphone = ctc_dip(outputs["diphone_log_probs"], targets["diphone"],
                        input_lengths, target_lengths["diphone"])

    total = l_mono + alpha * l_diphone
    components["ctc_mono"] = l_mono
    components["ctc_diphone"] = l_diphone

    if variant in ("D", "E"):
        # Guard against the rare case where every sample in the batch has
        # phoneme length < 3, producing no valid skip-diphone targets.
        skip_lens = target_lengths["skip_diphone"]
        if skip_lens.sum() > 0:
            l_skip = ctc_dip(outputs["skip_log_probs"], targets["skip_diphone"],
                             input_lengths, skip_lens)
            total = total + beta * l_skip
            components["ctc_skip"] = l_skip

    if variant in ("C", "E"):
        l_smooth = smoothness_loss(phoneme_probs, input_lengths)
        total = total + lambda_smooth * l_smooth
        components["smooth"] = l_smooth

    return total, components
