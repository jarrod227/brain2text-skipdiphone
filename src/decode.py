"""
Decode a trained checkpoint and report PER (greedy CTC) and WER (with LM).

PER follows cffan/neural_seq_decoder convention: micro edit-distance over all
phonemes including SIL, only CTC blank removed from the hypothesis.

Usage:
    # PER only (no LM)  — runs in the b2t training env
    python src/decode.py --checkpoint experiments/<run>/best.pt --variant A

    # PER + WER with n-gram LM  — must run in the lm_decode env
    python src/decode.py --checkpoint experiments/<run>/best.pt --variant E \
        --lm 3gram --lm_dir data/languageModel

    # Save 100-best for GPT-2 rescoring
    python src/decode.py --checkpoint experiments/<run>/best.pt --variant E \
        --lm 3gram --lm_dir data/languageModel --nbest 100 \
        --save_nbest experiments/<run>/nbest.pkl
"""

import argparse
import pickle
from pathlib import Path

import editdistance
import numpy as np
import torch
from omegaconf import OmegaConf

from dataset import make_dataloader
from model import SkipDiphoneDecoder


def word_error_rate(hyp_text, ref_text):
    hyp = hyp_text.split()
    ref = ref_text.split()
    return editdistance.eval(hyp, ref) / max(len(ref), 1)


def _ctc_collapse(frame_ids, blank):
    """Collapse consecutive duplicates then remove blank."""
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
        Ours:  [phone_0, ..., phone_38=ZH, phone_39=SIL, blank]
        Kaldi: [blank, SIL, phone_0, ..., phone_38=ZH]
    Equivalent to speechBCI's rearrange_speech_logits(has_sil=True).
    """
    blank  = log_probs[..., -1:]
    sil    = log_probs[..., -2:-1]
    phones = log_probs[..., :-2]
    return torch.cat([blank, sil, phones], dim=-1)


def _marginalize_diphone_logits(diphone_logits, num_phonemes):
    """
    Convert raw diphone logits to raw-ish phoneme logits using logsumexp.

    diphone class index = prev_phoneme * C + curr_phoneme

    Args:
        diphone_logits: (B, T, C*C + 1), last dim is CTC blank
        num_phonemes: C

    Returns:
        phoneme_logits: (B, T, C + 1), last dim is CTC blank
    """
    B, T, _ = diphone_logits.shape
    C = num_phonemes

    dip = diphone_logits[..., :-1].view(B, T, C, C)  # (B, T, prev, curr)
    phone_logits = torch.logsumexp(dip, dim=2)       # sum over prev phoneme
    blank_logits = diphone_logits[..., -1:]          # keep blank logit

    return torch.cat([phone_logits, blank_logits], dim=-1)


def _build_lm_decoder(lm_dir, nbest=1):
    """Build a WFST decoder using the speechBCI lm_decoder package."""
    import lm_decoder  # noqa: only available in lm_decode conda env
    opts = lm_decoder.DecodeOptions(
        7000,   # max_active
        200,    # min_active
        17.0,   # beam
        8.0,    # lattice_beam
        0.5,    # acoustic_scale  (matches cffan eval_competition.py)
        1.0,    # ctc_blank_skip_threshold
        0.0,    # length_penalty
        nbest,
    )
    lm_path = Path(lm_dir)
    resource = lm_decoder.DecodeResource(
        str(lm_path / "TLG.fst"),
        "",
        "",
        str(lm_path / "words.txt"),
        "",
    )
    return lm_decoder.BrainSpeechDecoder(resource, opts)


@torch.no_grad()
def decode(model, loader, device, variant="E", lm=None, lm_dir=None,
           dump_examples=0, nbest=1, save_nbest=None):
    model.eval()
    total_edits = 0
    total_ref_len = 0
    wer_scores = []
    nbest_outputs = []   # list of lists of Result objects
    all_transcripts = []
    dumped = 0

    decoder_lm = None
    lm_module = None
    if lm is not None:
        import lm_decoder as lm_module
        decoder_lm = _build_lm_decoder(lm_dir, nbest=nbest)

    blank = model.num_phonemes

    for batch in loader:
        neural   = batch["neural"].to(device)
        lengths  = batch["lengths"].to(device)
        day_ids  = batch["day_ids"].to(device)

        mono_lp, _, _, phoneme_probs, enc_lengths = model(neural, lengths, day_ids)

        if variant == "A":
            log_probs = mono_lp.permute(1, 0, 2)
        else:
            log_probs = torch.log(phoneme_probs.clamp(min=1e-10))

        # === PER: greedy CTC, micro accumulation, no SIL filter (cffan convention) ===
        frame_ids  = log_probs.argmax(dim=-1).cpu().tolist()
        hyp_phones = [_ctc_collapse(seq[:L], blank)
                      for seq, L in zip(frame_ids, enc_lengths.cpu().tolist())]

        ref_phones = batch["targets"]["mono"].cpu().tolist()
        ref_lens   = batch["target_lengths"]["mono"].cpu().tolist()
        offset = 0
        for i, hyp in enumerate(hyp_phones):
            ref = ref_phones[offset: offset + ref_lens[i]]
            total_edits   += editdistance.eval(hyp, ref)
            total_ref_len += len(ref)
            if dumped < dump_examples:
                print(f"[sample {dumped}] ref={ref[:20]} hyp={hyp[:20]}")
                dumped += 1
            offset += ref_lens[i]

        # === WER: WFST + n-gram LM ===
        if lm is not None:
            lp_kaldi = _rearrange_for_kaldi(log_probs).cpu().numpy().astype(np.float32)
            log_priors = np.zeros([1, lp_kaldi.shape[-1]], dtype=np.float32)
            blank_penalty = float(np.log(7))  # matches cffan eval_competition.py
            for i, L in enumerate(enc_lengths.cpu().tolist()):
                decoder_lm.Reset()
                lm_module.DecodeNumpy(decoder_lm, lp_kaldi[i, :L], log_priors, blank_penalty)
                decoder_lm.FinishDecoding()
                results = decoder_lm.result()
                hyp_text = results[0].sentence.strip() if results else ""
                wer_scores.append(word_error_rate(hyp_text, batch["transcripts"][i].strip()))
                if save_nbest:
                    nbest_outputs.append([r.sentence.strip() for r in results])
                    all_transcripts.append(batch["transcripts"][i].strip())

    if save_nbest and nbest_outputs:
        Path(save_nbest).parent.mkdir(parents=True, exist_ok=True)
        with open(save_nbest, "wb") as f:
            pickle.dump({"nbest": nbest_outputs, "transcripts": all_transcripts}, f)
        print(f"Saved {len(nbest_outputs)} nbest lists to {save_nbest}")

    per = total_edits / max(total_ref_len, 1)
    wer = sum(wer_scores) / len(wer_scores) if wer_scores else float("nan")
    return per, wer


def _load_checkpoint(path, device):
    """Load a state dict; weights_only is only available on PyTorch >= 2.0."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=["A","B","C","D","E"])
    parser.add_argument("--lm",     default=None, choices=["3gram", "5gram"],
                        help="n-gram LM for WER (requires speechBCI lm_decoder)")
    parser.add_argument("--lm_dir", default=None,
                        help="path to TLG.fst / words.txt (defaults to cfg.lm_dir)")
    parser.add_argument("--nbest",  default=1, type=int,
                        help="number of hypotheses to return from n-gram decoder")
    parser.add_argument("--save_nbest", default=None,
                        help="save nbest lists to this .pkl path for GPT-2 rescoring")
    parser.add_argument("--dump_examples", default=0, type=int,
                        help="print first N decoded examples for debugging")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    lm_dir = args.lm_dir or cfg.lm_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    per, wer = decode(model, loader, device, variant=args.variant,
                      lm=args.lm, lm_dir=lm_dir, dump_examples=args.dump_examples,
                      nbest=args.nbest, save_nbest=args.save_nbest)
    print(f"PER: {per * 100:.2f}%")
    if args.lm is not None:
        print(f"WER (n-gram): {wer * 100:.2f}%")


if __name__ == "__main__":
    main()
