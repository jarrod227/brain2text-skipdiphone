"""
Decode a trained checkpoint and report PER / WER using the n-gram LM.

Usage:
    python src/decode.py --checkpoint experiments/<run>/best.pt --lm 3gram
    python src/decode.py --checkpoint experiments/<run>/best.pt --lm 5gram
"""

import argparse
import glob
import os

import editdistance
import torch
from omegaconf import OmegaConf

from dataset import make_dataloader
from model import SkipDiphoneDecoder


def load_lm(lm_dir, lm_type):
    """Returns a callable that takes a list of phoneme sequences and returns word sequences."""
    # Placeholder: integrate speechBCI's LanguageModelDecoder here.
    # The decoder expects phoneme-level beam search output and returns word strings.
    raise NotImplementedError(
        f"Wire up speechBCI LanguageModelDecoder for '{lm_type}' LM in {lm_dir}"
    )


def phoneme_error_rate(hyp, ref):
    return editdistance.eval(hyp, ref) / max(len(ref), 1)


def word_error_rate(hyp_words, ref_words):
    hyp = hyp_words.split()
    ref = ref_words.split()
    return editdistance.eval(hyp, ref) / max(len(ref), 1)


@torch.no_grad()
def decode(model, loader, lm, device):
    model.eval()
    per_scores, wer_scores = [], []

    for batch in loader:
        neural  = batch["neural"].to(device)
        lengths = batch["lengths"].to(device)

        _, _, _, phoneme_probs = model(neural, lengths)

        # phoneme_probs: (B, T, num_phonemes) -> greedy decode per frame
        hyp_phones = phoneme_probs.argmax(dim=-1).cpu().tolist()

        ref_phones = batch["targets"]["mono"].cpu().tolist()
        ref_lens   = batch["target_lengths"]["mono"].cpu().tolist()

        offset = 0
        for i, L in enumerate(lengths.cpu().tolist()):
            ref_seq = ref_phones[offset: offset + ref_lens[i]]
            hyp_seq = hyp_phones[i][:L]
            per_scores.append(phoneme_error_rate(hyp_seq, ref_seq))
            offset += ref_lens[i]

        # LM decoding for WER (requires LanguageModelDecoder integration)
        # hyp_words = lm(phoneme_probs)
        # for hyp, ref in zip(hyp_words, batch["transcripts"]):
        #     wer_scores.append(word_error_rate(hyp, ref))

    mean_per = sum(per_scores) / len(per_scores) if per_scores else float("nan")
    mean_wer = sum(wer_scores) / len(wer_scores) if wer_scores else float("nan")
    return mean_per, mean_wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lm",         default="3gram", choices=["3gram", "5gram"])
    parser.add_argument("--config",     default="configs/default.yaml")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkipDiphoneDecoder(
        input_dim=cfg.encoder.input_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        num_layers=cfg.encoder.num_layers,
        num_phonemes=cfg.num_phonemes,
        num_diphones=cfg.num_diphones,
        dropout=cfg.encoder.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_paths = sorted(glob.glob(os.path.join(cfg.data_dir, "test", "*.tfrecord")))
    loader = make_dataloader(test_paths, cfg.batch_size, shuffle=False)

    lm = load_lm(cfg.lm_dir, args.lm)
    per, wer = decode(model, loader, lm, device)
    print(f"PER: {per * 100:.2f}%   WER: {wer * 100:.2f}%")


if __name__ == "__main__":
    main()
