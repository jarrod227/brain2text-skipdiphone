"""
GPT-2 rescoring of n-gram nbest hypotheses.

n-gram decoder produces 100-best candidates; GPT-2 re-ranks them by
language model score. No n-gram score is available from lm_decoder,
so ranking is purely by GPT-2 log-probability.

Usage:
    # 1. Generate 100-best (lm_decode env):
    python src/decode.py --checkpoint experiments/<run>/best.pt --variant <X> \
        --lm 3gram --lm_dir data/languageModel \
        --nbest 100 --save_nbest experiments/<run>/nbest.pkl

    # 2. Rescore:
    python src/rescore.py --nbest experiments/<run>/nbest.pkl
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
    """Total log-probability of text under GPT-2."""
    text = text.strip()
    if not text:
        return -1e9
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    loss = model(ids, labels=ids).loss
    return -loss.item() * ids.shape[1]


def rescore(nbest_outputs, transcripts, gpt2_model, gpt2_tok, device):
    total_edits = 0
    total_words = 0
    for hyps, ref in zip(nbest_outputs, transcripts):
        best_hyp = max(hyps, key=lambda h: gpt2_log_prob(gpt2_model, gpt2_tok, h, device))
        total_edits += editdistance.eval(best_hyp.split(), ref.split())
        total_words += max(len(ref.split()), 1)
    return total_edits / max(total_words, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest",  required=True, help="nbest.pkl from decode.py")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.nbest, "rb") as f:
        data = pickle.load(f)
    nbest_outputs = data["nbest"]
    transcripts   = data["transcripts"]
    print(f"Loaded {len(nbest_outputs)} samples, {len(nbest_outputs[0])} hypotheses each")

    gpt2_model, gpt2_tok = load_gpt2(device)
    wer = rescore(nbest_outputs, transcripts, gpt2_model, gpt2_tok, device)
    print(f"WER (GPT-2 rescore): {wer * 100:.2f}%")


if __name__ == "__main__":
    main()
