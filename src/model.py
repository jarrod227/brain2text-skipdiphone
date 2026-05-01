"""
GRU encoder with three output heads:
  - monophone CTC head       (variant A)
  - standard diphone head    (variants B/C/D/E)
  - skip-diphone aux head    (variants D/E)

Input preprocessing pipeline (matching cffan/neural_seq_decoder):
  raw neural features
    -> GaussianSmoothing1D
    -> day-specific linear transform + Softsign
    -> sliding window (nn.Unfold, kernel=32, stride=4)
    -> hand-written bidirectional GRU

Two encoder implementations are provided:
  - GRUCell / BidirectionalGRU: hand-written (active, used by default)
  - nn.GRU-based encoder:       commented out below for reference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Input preprocessing
# ---------------------------------------------------------------------------

class GaussianSmoothing1D(nn.Module):
    """Fixed depthwise conv1d with a Gaussian kernel along the time axis."""

    def __init__(self, num_channels, kernel_size=21, sigma=2.0):
        super().__init__()
        # Build 1-D Gaussian kernel
        half = (kernel_size - 1) / 2.0
        x = torch.arange(-half, half + 1)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        # Shape: (num_channels, 1, kernel_size) for depthwise conv1d
        weight = kernel.view(1, 1, -1).repeat(num_channels, 1, 1)
        self.register_buffer("weight", weight)
        self.num_channels = num_channels
        self.padding = kernel_size // 2

    def forward(self, x):
        # x: (B, D, T)  — channel-first
        return F.conv1d(x, self.weight, padding=self.padding,
                        groups=self.num_channels)


class InputPreprocessor(nn.Module):
    """
    Gaussian smooth -> day-specific affine transform + Softsign
    -> sliding-window unfold.

    Matches the preprocessing in cffan/neural_seq_decoder GRUDecoder.
    Day weights are initialized as identity so the model starts at the
    same point regardless of how many recording days are in the dataset.
    """

    def __init__(self, input_dim, num_days, kernel_len=32, stride_len=4,
                 gaussian_smooth_width=2.0):
        super().__init__()
        self.input_dim   = input_dim
        self.kernel_len  = kernel_len
        self.stride_len  = stride_len

        self.smooth = GaussianSmoothing1D(input_dim, kernel_size=21,
                                          sigma=gaussian_smooth_width)

        # Day-specific linear transform: one (D, D) matrix + (1, D) bias per day
        day_w = torch.zeros(num_days, input_dim, input_dim)
        for i in range(num_days):
            day_w[i] = torch.eye(input_dim)          # start as identity
        self.day_weights = nn.Parameter(day_w)
        self.day_bias    = nn.Parameter(torch.zeros(num_days, 1, input_dim))

        self.unfolder = nn.Unfold((kernel_len, 1), stride=stride_len)

    @property
    def output_dim(self):
        return self.input_dim * self.kernel_len

    def forward(self, x, day_ids, lengths):
        """
        x:       (B, T, input_dim)
        day_ids: (B,)  integer day index per sample
        lengths: (B,)  true sequence lengths
        returns:
            out:     (B, T', input_dim * kernel_len)
            lengths: (B,)  updated lengths after striding
        """
        B, T, D = x.shape

        # Gaussian smoothing — operate channel-first
        x = x.permute(0, 2, 1)           # (B, D, T)
        x = self.smooth(x)

        # Day-specific affine transform
        # day_weights[day_ids]: (B, D, D)
        x = x.permute(0, 2, 1)           # (B, T, D)
        w = self.day_weights[day_ids]     # (B, D, D)
        b = self.day_bias[day_ids]        # (B, 1, D)
        x = torch.bmm(x, w) + b          # (B, T, D)
        x = torch.nn.functional.softsign(x)

        # Sliding-window unfold: (B, T, D) -> (B, D*kernel_len, T')
        x = x.permute(0, 2, 1).unsqueeze(-1)          # (B, D, T, 1)
        x = self.unfolder(x)                           # (B, D*kernel_len, T')
        x = x.permute(0, 2, 1).contiguous()            # (B, T', D*kernel_len)

        # Update lengths to match strided output
        new_lengths = ((lengths - self.kernel_len) // self.stride_len + 1).clamp(min=1)

        return x, new_lengths


# ---------------------------------------------------------------------------
# Hand-written GRU cell
# Equations:
#   z_t = sigmoid(W_z x_t + U_z h_{t-1})          update gate
#   r_t = sigmoid(W_r x_t + U_r h_{t-1})          reset gate
#   n_t = tanh(W_n x_t + U_n (r_t * h_{t-1}))    candidate hidden state
#   h_t = (1 - z_t) * h_{t-1} + z_t * n_t        new hidden state
# ---------------------------------------------------------------------------

class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Fused input projection for all three gates
        self.W = nn.Linear(input_dim,  3 * hidden_dim, bias=True)
        # Separate hidden projections so reset gate can be applied before U_n
        self.U_zr = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.U_n  = nn.Linear(hidden_dim,     hidden_dim, bias=False)

    def forward(self, x, h):
        """
        x: (B, input_dim)
        h: (B, hidden_dim)
        returns h_new: (B, hidden_dim)
        """
        W_z, W_r, W_n = self.W(x).chunk(3, dim=-1)
        U_z, U_r      = self.U_zr(h).chunk(2, dim=-1)

        z = torch.sigmoid(W_z + U_z)
        r = torch.sigmoid(W_r + U_r)
        n = torch.tanh(W_n + self.U_n(r * h))

        return (1 - z) * h + z * n


class BidirectionalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Stack of forward and backward GRU cells
        # Input to layer 0 is input_dim; subsequent layers receive 2*hidden_dim
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else 2 * hidden_dim
            self.fwd_cells.append(GRUCell(layer_input_dim, hidden_dim))
            self.bwd_cells.append(GRUCell(layer_input_dim, hidden_dim))

        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """
        x:       (B, T, input_dim)
        lengths: (B,)
        returns: (B, T, 2 * hidden_dim)
        """
        B, T, _ = x.shape

        out = x
        for layer in range(self.num_layers):
            h_fwd = torch.zeros(B, self.hidden_dim, device=x.device)
            h_bwd = torch.zeros(B, self.hidden_dim, device=x.device)

            fwd_outs = []
            for t in range(T):
                h_fwd = self.fwd_cells[layer](out[:, t, :], h_fwd)
                fwd_outs.append(h_fwd)

            # Process backward direction starting from each sequence's last real frame.
            # Build a per-step active mask so padded frames don't corrupt h_bwd.
            bwd_outs = [None] * T
            for t in range(T - 1, -1, -1):
                active = (t < lengths).float().unsqueeze(1)   # (B, 1)
                h_new  = self.bwd_cells[layer](out[:, t, :], h_bwd)
                h_bwd  = active * h_new + (1 - active) * h_bwd
                bwd_outs[t] = h_bwd

            fwd_tensor = torch.stack(fwd_outs, dim=1)   # (B, T, H)
            bwd_tensor = torch.stack(bwd_outs, dim=1)   # (B, T, H)
            out = torch.cat([fwd_tensor, bwd_tensor], dim=-1)  # (B, T, 2H)

            if layer < self.num_layers - 1:
                out = self.drop(out)

        # Zero out positions beyond each sequence's true length
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)

        return out


# ---------------------------------------------------------------------------
# PyTorch built-in alternative (uncomment to swap in, comment out the
# BidirectionalGRU block in SkipDiphoneDecoder.__init__ and forward)
# ---------------------------------------------------------------------------
class TorchGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out   # (B, T, 2 * hidden_dim)
# ---------------------------------------------------------------------------


class SkipDiphoneDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_phonemes,
                 num_diphones, num_days, kernel_len=32, stride_len=4,
                 gaussian_smooth_width=2.0, dropout=0.4):
        super().__init__()
        self.num_phonemes = num_phonemes
        self.num_diphones = num_diphones

        self.preprocessor = InputPreprocessor(
            input_dim, num_days, kernel_len, stride_len, gaussian_smooth_width
        )

        # nn.GRU encoder (cuDNN-accelerated, used for actual training)
        self.encoder = TorchGRUEncoder(
            self.preprocessor.output_dim, hidden_dim, num_layers, dropout
        )

        # Hand-written bidirectional GRU (pedagogical reference, swap in to use):
        # self.encoder = BidirectionalGRU(
        #     self.preprocessor.output_dim, hidden_dim, num_layers, dropout
        # )

        enc_out_dim = 2 * hidden_dim  # bidirectional

        # Head A: monophone CTC (baseline)
        self.mono_head    = nn.Linear(enc_out_dim, num_phonemes + 1)  # +1 for CTC blank

        # Head B/C/D/E: standard diphone CTC (z_{t-1} -> z_t)
        self.diphone_head = nn.Linear(enc_out_dim, num_diphones + 1)

        # Head D/E: skip-diphone auxiliary CTC (z_{t-2} -> z_t)
        self.skip_head    = nn.Linear(enc_out_dim, num_diphones + 1)

    def forward(self, x, lengths, day_ids):
        """
        Args:
            x:       (B, T, input_dim)
            lengths: (B,) actual sequence lengths before padding
            day_ids: (B,) integer recording-day index per sample
        Returns:
            mono_log_probs:    (T', B, num_phonemes+1)
            diphone_log_probs: (T', B, num_diphones+1)
            skip_log_probs:    (T', B, num_diphones+1)
            phoneme_probs:     (B, T', num_phonemes)  -- marginalized from diphone
            enc_lengths:       (B,)  sequence lengths after preprocessing stride
        """
        x, enc_lengths = self.preprocessor(x, day_ids, lengths)   # (B, T', D*K)
        enc_out = self.encoder(x, enc_lengths)                     # (B, T', 2H)

        mono_log_probs    = F.log_softmax(self.mono_head(enc_out),    dim=-1).permute(1, 0, 2)
        diphone_log_probs = F.log_softmax(self.diphone_head(enc_out), dim=-1).permute(1, 0, 2)
        skip_log_probs    = F.log_softmax(self.skip_head(enc_out),    dim=-1).permute(1, 0, 2)

        phoneme_probs = self._marginalize(diphone_log_probs.permute(1, 0, 2))

        return mono_log_probs, diphone_log_probs, skip_log_probs, phoneme_probs, enc_lengths

    def _marginalize(self, diphone_log_probs):
        """
        Recover per-frame phoneme probs by summing over the previous-phoneme dim.
        diphone class index = prev_phoneme * num_phonemes + curr_phoneme
        (B, T', num_diphones+1) -> (B, T', num_phonemes)
        """
        B, T, _ = diphone_log_probs.shape
        C = self.num_phonemes

        diphone_probs = diphone_log_probs[..., :-1].exp()      # drop blank, (B, T, C*C)
        diphone_probs = diphone_probs.view(B, T, C, C)         # (B, T, prev, curr)
        phoneme_probs = diphone_probs.sum(dim=2)               # (B, T, curr)
        phoneme_probs = phoneme_probs / (phoneme_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return phoneme_probs
