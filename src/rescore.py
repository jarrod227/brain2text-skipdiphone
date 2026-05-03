"""
GPT-2 rescoring of n-gram nbest hypotheses.

Usage:
    # 1. First generate nbest with decode.py:
    python src/decode.py --checkpoint experiments/<run>/best.pt --variant E \
        --lm 3gram --lm_dir data/languageModel \
        --nbest 100 --save_nbest experiments/<run>/nbest.pkl

    # 2. Rescore with GPT-2 (sweep alpha):
    python src/rescore.py --nbest experiments/<run>/nbest.pkl

    # 3. Or fix alpha:
    python src/rescore.py --nbest experiments/<run>/nbest.pkl --alpha 0.5
"""

import argparse
import pickle

import editdistance
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_gpt2(device):
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def gpt2_log_prob(model, tokenizer, text, device):
    """Return total log-probability of text under GPT-2."""
    text = text.strip()
    if not text:
        return -1e9
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    loss = model(ids, labels=ids).loss   # mean NLL per token
    return -loss.item() * ids.shape[1]  # total log-prob


def word_error_rate(hyp, ref):
    return editdistance.eval(hyp.split(), ref.split()) / max(len(ref.split()), 1)


def rescore_with_alpha(nbest_outputs, transcripts, gpt2_model, gpt2_tok, alpha, device):
    total_edits = 0
    total_words = 0
    for results, ref in zip(nbest_outputs, transcripts):
        best_score = -1e18
        best_hyp = ""
        for r in results:
            hyp = r["sentence"]
            gpt2_s = gpt2_log_prob(gpt2_model, gpt2_tok, hyp, device)
            combined = r["score"] + alpha * gpt2_s
            if combined > best_score:
                best_score = combined
                best_hyp = hyp
        ref_words = ref.split()
        total_edits += editdistance.eval(best_hyp.split(), ref_words)
        total_words += len(ref_words)
    return total_edits / max(total_words, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest", required=True,
                        help="path to nbest.pkl saved by decode.py")
    parser.add_argument("--alpha", default=None, type=float,
                        help="GPT-2 weight; if omitted, sweeps [0.1, 0.3, 0.5, 0.7, 1.0]")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: cuda if available)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.nbest, "rb") as f:
        data = pickle.load(f)
    nbest_outputs = data["nbest"]
    transcripts   = data["transcripts"]
    print(f"Loaded {len(nbest_outputs)} samples, "
          f"{len(nbest_outputs[0])} hypotheses each")

    gpt2_model, gpt2_tok = load_gpt2(device)

    if args.alpha is not None:
        wer = rescore_with_alpha(nbest_outputs, transcripts,
                                 gpt2_model, gpt2_tok, args.alpha, device)
        print(f"WER (alpha={args.alpha:.2f}): {wer * 100:.2f}%")
    else:
        print("Sweeping alpha...")
        for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            wer = rescore_with_alpha(nbest_outputs, transcripts,
                                     gpt2_model, gpt2_tok, alpha, device)
            print(f"  alpha={alpha:.1f}  WER={wer * 100:.2f}%")


if __name__ == "__main__":
    main()
