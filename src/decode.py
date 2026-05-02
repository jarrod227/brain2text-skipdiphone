"""
Decode a trained checkpoint and report PER (greedy CTC) and WER (with LM).

Usage:
    # PER only (no LM)  — runs in the b2t training env
    python src/decode.py --checkpoint experiments/<run>/best.pt --variant A

    # PER + WER with 3-gram LM  — must run in the lm_decode env (PyTorch 1.13.1
    # + speechBCI lm_decoder package)
    python src/decode.py --checkpoint experiments/<run>/best.pt --variant E \
        --lm 3gram --lm_dir data/languageModel
"""

import argparse
from pathlib import Path

import editdistance
import numpy as np
import torch
from omegaconf import OmegaConf

from dataset import make_dataloader
from model import SkipDiphoneDecoder


SIL_ID = 39  # silence phoneme; excluded from PER following cffan convention


def phoneme_error_rate(hyp, ref):
    hyp = [p for p in hyp if p != SIL_ID]
    ref = [p for p in ref if p != SIL_ID]
    return editdistance.eval(hyp, ref) / max(len(ref), 1)


def word_error_rate(hyp_text, ref_text):
    hyp = hyp_text.split()
    ref = ref_text.split()
    return editdistance.eval(hyp, ref) / max(len(ref), 1)


def _ctc_collapse(frame_ids, blank):
    """Collapse consecutive duplicate tokens then remove blank."""
    collapsed = []
    prev = None
    for t in frame_ids:
        if t != prev:
            collapsed.append(t)
            prev = t
    return [t for t in collapsed if t != blank]


def _rearrange_for_kaldi(log_probs):
    """
    Reorder the last dim from our model layout to the Kaldi/LM layout.
        Ours:  [phone_0, phone_1, ..., phone_38=ZH, phone_39=SIL, blank]
        Kaldi: [blank, SIL, phone_0, phone_1, ..., phone_38=ZH]
    Equivalent to speechBCI's rearrange_speech_logits(has_sil=True).
    """
    blank  = log_probs[..., -1:]
    sil    = log_probs[..., -2:-1]
    phones = log_probs[..., :-2]
    return torch.cat([blank, sil, phones], dim=-1)


def _build_lm_decoder(lm_dir):
    """Build a 3-gram WFST decoder using the speechBCI lm_decoder package."""
    import lm_decoder  # noqa: only available in lm_decode conda env
    opts = lm_decoder.DecodeOptions(
        7000,   # max_active
        200,    # min_active
        17.0,   # beam
        8.0,    # lattice_beam
        1.5,    # acoustic_scale
        1.0,    # ctc_blank_skip_threshold
        0.0,    # length_penalty
        1,      # nbest
    )
    lm_path = Path(lm_dir)
    resource = lm_decoder.DecodeResource(
        str(lm_path / "TLG.fst"),
        "",                              # G.fst (optional, for lattice rescoring)
        "",                              # G_no_prune.fst (optional)
        str(lm_path / "words.txt"),
        "",
    )
    return lm_decoder.BrainSpeechDecoder(resource, opts)


@torch.no_grad()
def decode(model, loader, device, variant="E", lm=None, lm_dir=None):
    model.eval()
    per_scores, wer_scores = [], []

    decoder_lm = None
    lm_module = None
    if lm is not None:
        import lm_decoder as lm_module
        decoder_lm = _build_lm_decoder(lm_dir)

    blank = model.num_phonemes

    for batch in loader:
        neural   = batch["neural"].to(device)
        lengths  = batch["lengths"].to(device)
        day_ids  = batch["day_ids"].to(device)

        mono_lp, _, _, phoneme_probs, enc_lengths = model(neural, lengths, day_ids)

        # Get per-frame log-probs (B, T', C+1) for both PER and LM decode.
        # A: read from the mono CTC head (already log-softmax).
        # B-E: convert marginalized phoneme probs back to log-domain.
        if variant == "A":
            log_probs = mono_lp.permute(1, 0, 2)
        else:
            log_probs = torch.log(phoneme_probs.clamp(min=1e-10))

        # === PER: greedy CTC ===
        frame_ids = log_probs.argmax(dim=-1).cpu().tolist()
        hyp_phones = [_ctc_collapse(seq[:L], blank)
                      for seq, L in zip(frame_ids, enc_lengths.cpu().tolist())]

        ref_phones = batch["targets"]["mono"].cpu().tolist()
        ref_lens   = batch["target_lengths"]["mono"].cpu().tolist()
        offset = 0
        for i in range(len(hyp_phones)):
            ref_seq = ref_phones[offset: offset + ref_lens[i]]
            per_scores.append(phoneme_error_rate(hyp_phones[i], ref_seq))
            offset += ref_lens[i]

        # === WER: WFST + n-gram LM ===
        if lm is not None:
            lp_kaldi = _rearrange_for_kaldi(log_probs).cpu().numpy().astype(np.float32)
            log_priors = np.zeros([1, lp_kaldi.shape[-1]], dtype=np.float32)
            blank_penalty = float(np.log(2.0))

            for i, L in enumerate(enc_lengths.cpu().tolist()):
                decoder_lm.Reset()
                lm_module.DecodeNumpy(decoder_lm, lp_kaldi[i, :L], log_priors, blank_penalty)
                decoder_lm.FinishDecoding()
                results = decoder_lm.result()
                hyp_text = results[0].sentence.strip() if results else ""
                ref_text = batch["transcripts"][i].strip()
                wer_scores.append(word_error_rate(hyp_text, ref_text))

    mean_per = sum(per_scores) / len(per_scores) if per_scores else float("nan")
    mean_wer = sum(wer_scores) / len(wer_scores) if wer_scores else float("nan")
    return mean_per, mean_wer


def _load_checkpoint(path, device):
    """Load a state dict; weights_only is only available on PyTorch >= 2.0."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default=None, choices=["A","B","C","D","E"],
                        help="ablation variant (default: read from config)")
    parser.add_argument("--lm",     default=None, choices=["3gram", "5gram"],
                        help="n-gram LM for WER (requires speechBCI lm_decoder)")
    parser.add_argument("--lm_dir", default=None,
                        help="path to TLG.fst / words.txt (defaults to cfg.lm_dir)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg     = OmegaConf.load(args.config)
    variant = args.variant or cfg.variant
    lm_dir  = args.lm_dir or cfg.lm_dir
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkipDiphoneDecoder(
        input_dim=cfg.encoder.input_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        num_layers=cfg.encoder.num_layers,
        num_phonemes=cfg.num_phonemes,
        num_diphones=cfg.num_diphones,
        num_days=cfg.num_days,
        kernel_len=cfg.encoder.kernel_len,
        stride_len=cfg.encoder.stride_len,
        gaussian_smooth_width=cfg.encoder.gaussian_smooth_width,
        dropout=cfg.encoder.dropout,
    ).to(device)
    model.load_state_dict(_load_checkpoint(args.checkpoint, device))

    loader = make_dataloader(cfg.data_path, "test", cfg.batch_size,
                             num_phonemes=cfg.num_phonemes, shuffle=False)

    per, wer = decode(model, loader, device, variant=variant, lm=args.lm, lm_dir=lm_dir)
    print(f"PER: {per * 100:.2f}%")
    if args.lm is not None:
        print(f"WER: {wer * 100:.2f}%")


if __name__ == "__main__":
    main()
