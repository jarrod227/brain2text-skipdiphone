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


def _build_lm_decoder(lm_dir, nbest=1, acoustic_scale=0.8, beam=18.0):
    """
    Build a WFST decoder using the speechBCI lm_decoder package.

    Default acoustic_scale=0.8 and beam=18 follow the official
    speechBCI baseline inference notebook more closely than 0.5 / 17.
    """
    import lm_decoder  # noqa: only available in lm_decode conda env

    opts = lm_decoder.DecodeOptions(
        7000,             # max_active
        200,              # min_active
        float(beam),      # beam
        8.0,              # lattice_beam
        float(acoustic_scale),
        1.0,              # ctc_blank_skip_threshold
        0.0,              # length_penalty
        int(nbest),
    )

    lm_path = Path(lm_dir)
    resource = lm_decoder.DecodeResource(
        str(lm_path / "TLG.fst"),
        str(lm_path / "G.fst") if (lm_path / "G.fst").exists() else "",
        str(lm_path / "G_no_prune.fst") if (lm_path / "G_no_prune.fst").exists() else "",
        str(lm_path / "words.txt"),
        "",
    )

    return lm_decoder.BrainSpeechDecoder(resource, opts)


@torch.no_grad()
def decode(
    model,
    loader,
    device,
    variant="E",
    lm=None,
    lm_dir=None,
    dump_examples=0,
    nbest=1,
    save_nbest=None,
    acoustic_scale=0.8,
    beam=18.0,
    blank_penalty=None,
):
    model.eval()

    total_phone_edits = 0
    total_ref_phones = 0

    total_word_edits = 0
    total_ref_words = 0

    nbest_outputs = []
    all_transcripts = []
    dumped = 0

    decoder_lm = None
    lm_module = None
    if lm is not None:
        import lm_decoder as lm_module
        decoder_lm = _build_lm_decoder(
            lm_dir,
            nbest=nbest,
            acoustic_scale=acoustic_scale,
            beam=beam,
        )

    if blank_penalty is None:
        blank_penalty = float(np.log(7))

    blank = model.num_phonemes

    for batch in loader:
        neural = batch["neural"].to(device)
        lengths = batch["lengths"].to(device)
        day_ids = batch["day_ids"].to(device)

        (
            mono_lp,
            _,
            _,
            phoneme_probs,
            enc_lengths,
            mono_logits,
            diphone_logits,
            _,
        ) = model(neural, lengths, day_ids, return_logits=True)

        # ------------------------------------------------------------
        # PER path: use log_probs / probabilities for greedy CTC.
        # This preserves your old PER behavior.
        # ------------------------------------------------------------
        if variant == "A":
            per_log_probs = mono_lp.permute(1, 0, 2)
        else:
            per_log_probs = torch.log(phoneme_probs.clamp(min=1e-10))

        frame_ids = per_log_probs.argmax(dim=-1).cpu().tolist()
        enc_lens_list = enc_lengths.cpu().tolist()

        ref_phones = batch["targets"]["mono"].cpu().tolist()
        ref_lens = batch["target_lengths"]["mono"].cpu().tolist()

        offset = 0
        for i, hyp_frame_ids in enumerate(frame_ids):
            hyp = _ctc_collapse(hyp_frame_ids[:enc_lens_list[i]], blank)
            ref = ref_phones[offset: offset + ref_lens[i]]

            total_phone_edits += editdistance.eval(hyp, ref)
            total_ref_phones += len(ref)

            if dumped < dump_examples:
                print(f"[sample {dumped}]")
                print(f"  ref phones: {ref[:30]}")
                print(f"  hyp phones: {hyp[:30]}")
                print(f"  transcript: {batch['transcripts'][i].strip()}")
                dumped += 1

            offset += ref_lens[i]

        # ------------------------------------------------------------
        # WER path: use RAW LOGITS, not log_probs.
        # This is the important fix.
        # ------------------------------------------------------------
        if lm is not None:
            if variant == "A":
                acoustic_logits = mono_logits
            else:
                acoustic_logits = _marginalize_diphone_logits(
                    diphone_logits,
                    model.num_phonemes,
                )

            logits_kaldi = _rearrange_for_kaldi(acoustic_logits)
            logits_kaldi = logits_kaldi.cpu().numpy().astype(np.float32)

            log_priors = np.zeros([1, logits_kaldi.shape[-1]], dtype=np.float32)

            for i, L in enumerate(enc_lens_list):
                decoder_lm.Reset()
                lm_module.DecodeNumpy(
                    decoder_lm,
                    logits_kaldi[i, :L],
                    log_priors,
                    float(blank_penalty),
                )
                decoder_lm.FinishDecoding()

                results = decoder_lm.result()
                hyp_text = results[0].sentence.strip() if results else ""
                ref_text = batch["transcripts"][i].strip()

                hyp_words = hyp_text.split()
                ref_words = ref_text.split()

                # Official-style micro WER:
                # total word errors / total reference words.
                total_word_edits += editdistance.eval(hyp_words, ref_words)
                total_ref_words += max(len(ref_words), 1)

                if save_nbest:
                    nbest_outputs.append([r.sentence.strip() for r in results])
                    all_transcripts.append(ref_text)

    if save_nbest and nbest_outputs:
        Path(save_nbest).parent.mkdir(parents=True, exist_ok=True)
        with open(save_nbest, "wb") as f:
            pickle.dump(
                {
                    "nbest": nbest_outputs,
                    "transcripts": all_transcripts,
                },
                f,
            )
        print(f"Saved {len(nbest_outputs)} nbest lists to {save_nbest}")

    per = total_phone_edits / max(total_ref_phones, 1)
    wer = total_word_edits / max(total_ref_words, 1) if lm is not None else float("nan")

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
    parser.add_argument("--acoustic_scale", default=0.8, type=float,
                    help="WFST acoustic scale; official notebook uses about 0.8")
    parser.add_argument("--beam", default=18.0, type=float,
                    help="WFST beam; official notebook uses about 18")
    parser.add_argument("--blank_penalty", default=None, type=float,
                    help="blank penalty for lm_decoder; default is log(7)")
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

    per, wer = decode(
        model,
        loader,
        device,
        variant=args.variant,
        lm=args.lm,
        lm_dir=lm_dir,
        dump_examples=args.dump_examples,
        nbest=args.nbest,
        save_nbest=args.save_nbest,
        acoustic_scale=args.acoustic_scale,
        beam=args.beam,
        blank_penalty=args.blank_penalty,
    )
    print(f"PER: {per * 100:.2f}%")
    if args.lm is not None:
        print(f"WER (n-gram): {wer * 100:.2f}%")


if __name__ == "__main__":
    main()
