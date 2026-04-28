"""
GRU encoder with three output heads:
  - monophone CTC head       (variant A)
  - standard diphone head    (variants B/C/D/E)
  - skip-diphone aux head    (variants D/E)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipDiphoneDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_phonemes,
                 num_diphones, dropout=0.4, bidirectional=True):
        super().__init__()
        self.num_phonemes = num_phonemes
        self.num_diphones = num_diphones

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        enc_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Head A: monophone CTC (baseline)
        self.mono_head = nn.Linear(enc_out_dim, num_phonemes + 1)   # +1 for CTC blank

        # Head B/C/D/E: standard diphone (z_{t-1} -> z_t)
        self.diphone_head = nn.Linear(enc_out_dim, num_diphones + 1)

        # Head D/E: skip-diphone auxiliary (z_{t-2} -> z_t)
        self.skip_head = nn.Linear(enc_out_dim, num_diphones + 1)

    def forward(self, x, lengths):
        """
        Args:
            x:       (B, T, input_dim)
            lengths: (B,) actual sequence lengths before padding
        Returns:
            mono_log_probs:   (T, B, num_phonemes+1)
            diphone_log_probs:(T, B, num_diphones+1)
            skip_log_probs:   (T, B, num_diphones+1)
            phoneme_probs:    (B, T, num_phonemes)  -- marginalized from diphone
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_out, _ = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)

        mono_log_probs   = F.log_softmax(self.mono_head(enc_out),   dim=-1).permute(1, 0, 2)
        diphone_log_probs = F.log_softmax(self.diphone_head(enc_out), dim=-1).permute(1, 0, 2)
        skip_log_probs   = F.log_softmax(self.skip_head(enc_out),   dim=-1).permute(1, 0, 2)

        phoneme_probs = self._marginalize(diphone_log_probs.permute(1, 0, 2))

        return mono_log_probs, diphone_log_probs, skip_log_probs, phoneme_probs

    def _marginalize(self, diphone_log_probs):
        """
        Sum over the previous-phoneme dimension to recover per-frame phoneme probs.
        diphone class index = prev_phoneme * num_phonemes + curr_phoneme
        Shape: (B, T, num_diphones+1) -> (B, T, num_phonemes)
        """
        B, T, _ = diphone_log_probs.shape
        C = self.num_phonemes

        # Exclude the blank class (last index) before marginalizing
        diphone_probs = diphone_log_probs[..., :-1].exp()                 # (B, T, C*C)
        diphone_probs = diphone_probs.view(B, T, C, C)                    # (B, T, prev, curr)
        phoneme_probs = diphone_probs.sum(dim=2)                          # (B, T, curr)
        phoneme_probs = phoneme_probs / (phoneme_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return phoneme_probs
