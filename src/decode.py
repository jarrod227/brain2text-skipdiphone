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
import re
import torch

from dataset import make_dataloader
from model import SkipDiphoneDecoder


_NORMALIZE_PUNCT_RE = re.compile(r"[^a-zA-Z\- ']")


def _normalize_transcript(text):
    """
    Normalize a transcript to match the WFST decoder's output style.

    The competition pkl (formatCompetitionData.ipynb) stores raw transcripts
    with capitalization and punctuation, but the WFST decoder emits
    lowercase strings without punctuation. Without this step, micro-WER is
    inflated by every capitalized/punctuated reference word.

    Mirrors cffan eval_competition.py and speechBCI makeTFRecordsFromSession.py.
    """
    text = _NORMALIZE_PUNCT_RE.sub("", text)
    text = text.replace("--", "").lower().strip()
    return text


def word_error_rate(hyp_text, ref_text):
    hyp = hyp_text.split()
    ref = ref_text.split()
    return editdistance.eval(hyp, ref) / max(len(ref), 1)


def _ctc_collapse(frame_ids, blank):
    """Collapse consecutive duplicates, then remove the CTC blank."""
    collapsed = []
    prev = None
    for t in frame_ids:
        if t != prev:
            collapsed.append(t)
            prev = t
    return [t for t in collapsed if t != blank]


def _rearrange_for_kaldi(logits):
    """
    Reorder the last dimension from the model layout to the speechBCI/Kaldi layout.

        Model layout:  [phone_0, ..., phone_38=ZH, phone_39=SIL, blank]
        Kaldi layout:  [blank, SIL, phone_0, ..., phone_38=ZH]

    This matches speechBCI's rearrange_speech_logits(has_sil=True).
    """
    blank  = logits[..., -1:]
    sil    = logits[..., -2:-1]
    phones = logits[..., :-2]
    return torch.cat([blank, sil, phones], dim=-1)


def _marginalize_diphone_logits(diphone_logits, num_phonemes):
    """
    Convert raw diphone logits to phoneme-level logits using log-sum-exp.

    Diphone class index:
        diphone_id = previous_phoneme * C + current_phoneme

    Args:
        diphone_logits: (B, T, C*C + 1), where the last class is CTC blank
        num_phonemes: C

    Returns:
        phoneme_logits: (B, T, C + 1), where the last class is CTC blank
    """
    B, T, _ = diphone_logits.shape
    C = num_phonemes

    dip = diphone_logits[..., :-1].view(B, T, C, C)  # (B, T, previous, current)
    phone_logits = torch.logsumexp(dip, dim=2)       # marginalize previous phoneme
    blank_logits = diphone_logits[..., -1:]          # keep CTC blank logit

    return torch.cat([phone_logits, blank_logits], dim=-1)


def _build_lm_decoder(lm_dir, nbest=1, acoustic_scale=0.8, beam=17.0):
    """
    Build a WFST decoder using the official speechBCI lm_decoder package.

    Defaults follow speechBCI's actual baseline settings (see
    speechBCI/AnalysisExamples/rnn_step3_baselineRNNInference.ipynb):
        acoustic_scale = 0.8
        beam           = 17.0
        blank_penalty  = log(2) ~= 0.693  (passed separately to DecodeNumpy)

    Other DecodeOptions are kept aligned with speechBCI defaults:
        max_active=7000, min_active=200, lattice_beam=8.0,
        ctc_blank_skip_threshold=1.0, length_penalty=0.0.
    """
    import lm_decoder  # noqa: only available in the lm_decode conda env

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


def compute_log_priors(pkl_path, num_phonemes, dev_stride=10):
    """
    Estimate per-class log-priors from training-split phoneme frequencies.

    speechBCI's WFST decoder expects log-priors in Kaldi class order, i.e.
    [blank, SIL, phone_0, ..., phone_{C-2}]. The blank prior is set to 0
    (no prior subtraction for blank), matching the speechBCI baseline.

    Returns:
        np.ndarray of shape (1, C+1) with log-priors in Kaldi order.
    """
    from dataset import BrainToTextDataset

    ds = BrainToTextDataset(pkl_path, split="train", num_phonemes=num_phonemes,
                            dev_stride=dev_stride)
    counts = np.zeros(num_phonemes, dtype=np.float64)
    for s in ds.samples:
        ph = s["mono"].numpy()
        if ph.size == 0:
            continue
        bc = np.bincount(ph, minlength=num_phonemes)
        counts += bc[:num_phonemes]

    total = counts.sum()
    if total <= 0:
        return np.zeros([1, num_phonemes + 1], dtype=np.float32)

    priors = counts / total
    log_priors_phones = np.log(np.clip(priors, 1e-10, None))

    # Kaldi order: [blank, SIL, phone_0, ..., phone_{C-2}].
    # Model phone order: [phone_0, ..., phone_{C-2}=ZH, phone_{C-1}=SIL].
    blank_lp = np.zeros(1, dtype=np.float64)
    sil_lp   = log_priors_phones[-1:]   # SIL is last in model order
    rest_lp  = log_priors_phones[:-1]   # phones 0..C-2

    log_priors = np.concatenate([blank_lp, sil_lp, rest_lp]).astype(np.float32)
    return log_priors[None, :]   # shape (1, C+1)


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
    beam=17.0,
    blank_penalty=None,
    log_priors=None,
    decoder=None,
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
        if decoder is not None:
            decoder_lm = decoder
        else:
            decoder_lm = _build_lm_decoder(
                lm_dir,
                nbest=nbest,
                acoustic_scale=acoustic_scale,
                beam=beam,
            )

    # speechBCI's actual baseline uses log(2). Older code defaulted to 0.0
    # which leaves a strong blank-vs-text imbalance during WFST decoding.
    if blank_penalty is None:
        blank_penalty = float(np.log(2.0))

    blank = model.num_phonemes

    for batch in loader:
        neural = batch["neural"].to(device)
        lengths = batch["lengths"].to(device)
        day_ids = batch["day_ids"].to(device)

        outputs = model(neural, lengths, day_ids, return_logits=True)
        enc_lengths = outputs["enc_lengths"]

        # ------------------------------------------------------------
        # PER path: greedy CTC over phoneme probabilities/log-probs.
        # This preserves the acoustic PER behavior used during training.
        # ------------------------------------------------------------
        if variant == "A":
            per_log_probs = outputs["mono_log_probs"].permute(1, 0, 2)
        else:
            per_log_probs = torch.log(outputs["phoneme_probs"].clamp(min=1e-10))

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
        # WER path: use raw acoustic logits, matching speechBCI style.
        # For diphone variants, raw diphone logits are marginalized to
        # phoneme-level logits before Kaldi/WFST decoding.
        # ------------------------------------------------------------
        if lm is not None:
            if variant == "A":
                acoustic_logits = outputs["mono_logits"]
            else:
                acoustic_logits = _marginalize_diphone_logits(
                    outputs["diphone_logits"],
                    model.num_phonemes,
                )

            logits_kaldi = _rearrange_for_kaldi(acoustic_logits)
            logits_kaldi = logits_kaldi.cpu().numpy().astype(np.float32)

            if log_priors is None:
                log_priors_arr = np.zeros([1, logits_kaldi.shape[-1]],
                                          dtype=np.float32)
            else:
                log_priors_arr = np.asarray(log_priors, dtype=np.float32)
                if log_priors_arr.ndim == 1:
                    log_priors_arr = log_priors_arr[None, :]

            for i, L in enumerate(enc_lens_list):
                decoder_lm.Reset()
                lm_module.DecodeNumpy(
                    decoder_lm,
                    logits_kaldi[i, :L],
                    log_priors_arr,
                    float(blank_penalty),
                )
                decoder_lm.FinishDecoding()

                results = decoder_lm.result()
                hyp_text = _normalize_transcript(results[0].sentence) if results else ""
                ref_text = _normalize_transcript(batch["transcripts"][i])

                hyp_words = hyp_text.split()
                ref_words = ref_text.split()

                # Official-style micro WER:
                # total word errors / total reference words.
                total_word_edits += editdistance.eval(hyp_words, ref_words)
                total_ref_words += max(len(ref_words), 1)

                if save_nbest:
                    nbest_outputs.append([
                        {
                            "sentence": r.sentence.strip(),
                            "ac_score": float(r.ac_score),
                            "lm_score": float(r.lm_score),
                        }
                        for r in results
                    ])
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
                    help="WFST acoustic scale; speechBCI baseline is 0.8")
    parser.add_argument("--beam", default=17.0, type=float,
                    help="WFST beam; speechBCI default is 17")
    parser.add_argument("--blank_penalty", default=float(np.log(2.0)),
                    type=float,
                    help="blank penalty for lm_decoder; "
                         "speechBCI baseline is log(2) ~= 0.693")
    parser.add_argument("--log_priors", action="store_true",
                        help="estimate per-class log-priors from the training "
                             "split and pass them to lm_decoder. Off by "
                             "default (zeros, matching speechBCI's "
                             "rnn_step3_baselineRNNInference and "
                             "cffan/eval_competition). Treat as an "
                             "experimental opt-in.")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"],
                        help="which split to evaluate on (default: test)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--save_summary", default=None,
                        help="if set, write {per, wer} to this JSON path")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg    = OmegaConf.load(args.config)
    lm_dir = args.lm_dir or cfg.lm_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_stride = int(cfg.get("dev_stride", 10))

    model = SkipDiphoneDecoder(
        input_dim=cfg.encoder.input_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        num_layers=cfg.encoder.num_layers,
        num_phonemes=cfg.num_phonemes,
        num_diphones=cfg.num_diphones,
        num_days=cfg.num_days,
        variant=args.variant,
        kernel_len=cfg.encoder.kernel_len,
        stride_len=cfg.encoder.stride_len,
        gaussian_smooth_width=cfg.encoder.gaussian_smooth_width,
        dropout=cfg.encoder.dropout,
    ).to(device)
    # Old checkpoints may contain extra heads not used by the current
    # variant (e.g. mono_head from a previous build of variant E). Load
    # non-strictly so unused weights are silently ignored.
    state = _load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state, strict=False)

    loader = make_dataloader(cfg.data_path, args.split, cfg.batch_size,
                             num_phonemes=cfg.num_phonemes, shuffle=False,
                             dev_stride=dev_stride)
    print(f"[decode] split={args.split}  size={len(loader.dataset)}  "
          f"checkpoint={args.checkpoint}")

    log_priors = None
    if args.lm is not None and args.log_priors:
        log_priors = compute_log_priors(cfg.data_path, cfg.num_phonemes,
                                        dev_stride=dev_stride)
        print(f"[decode] log_priors computed from train split, "
              f"shape={log_priors.shape} (opt-in)")

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
        log_priors=log_priors,
    )
    print(f"PER: {per * 100:.2f}%")
    if args.lm is not None:
        print(f"WER (n-gram): {wer * 100:.2f}%")

    if args.save_summary:
        from pathlib import Path as _Path
        import json as _json
        out = {"split": args.split, "checkpoint": args.checkpoint,
               "variant": args.variant, "per": per, "wer": wer,
               "acoustic_scale": args.acoustic_scale,
               "blank_penalty": args.blank_penalty, "beam": args.beam,
               "log_priors": bool(args.log_priors and args.lm is not None)}
        _Path(args.save_summary).parent.mkdir(parents=True, exist_ok=True)
        _Path(args.save_summary).write_text(_json.dumps(out, indent=2))
        print(f"Wrote summary to {args.save_summary}")


if __name__ == "__main__":
    main()
