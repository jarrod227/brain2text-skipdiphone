"""
GPT-2 / GPT-2-XL rescoring for n-best hypotheses.

This follows the speechBCI/DCoND-style idea:
    total_score =
        alpha * GPT_score
      + (1 - alpha) * old_LM_score
      + acoustic_scale * acoustic_score

This is stronger than pure GPT reranking because it keeps acoustic evidence.
"""

import argparse
import pickle

import editdistance
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_lm(model_name, device):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def lm_log_prob(model, tokenizer, text, device, length_penalty=0.0):
    """
    Return total LM log-probability of a sentence.
    Higher is better.
    """
    text = text.strip()
    if not text:
        return -1e9

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    outputs = model(**inputs)
    logits = outputs.logits.float()

    # predict token j using logits at j-1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    score = token_log_probs.sum().item()
    n_tokens = target_ids.numel()

    return score - length_penalty * n_tokens


def normalize_hyp_item(item):
    """
    Supports both old format:
        "sentence"
    and new format:
        {"sentence": ..., "ac_score": ..., "lm_score": ...}
    """
    if isinstance(item, str):
        return {
            "sentence": item.strip(),
            "ac_score": 0.0,
            "lm_score": 0.0,
        }

    return {
        "sentence": item.get("sentence", "").strip(),
        "ac_score": float(item.get("ac_score", 0.0)),
        "lm_score": float(item.get("lm_score", 0.0)),
    }


def rescore(
    nbest_outputs,
    transcripts,
    model,
    tokenizer,
    device,
    alpha=0.5,
    acoustic_scale=0.8,
    length_penalty=0.0,
    dump_examples=0,
):
    total_edits = 0
    total_words = 0

    for idx, (hyps_raw, ref) in enumerate(zip(nbest_outputs, transcripts)):
        hyps = [normalize_hyp_item(h) for h in hyps_raw]
        hyps = [h for h in hyps if h["sentence"]]

        if not hyps:
            best_sentence = ""
        else:
            scored = []
            for h in hyps:
                gpt_score = lm_log_prob(
                    model,
                    tokenizer,
                    h["sentence"],
                    device,
                    length_penalty=length_penalty,
                )

                total_score = (
                    alpha * gpt_score
                    + (1.0 - alpha) * h["lm_score"]
                    + acoustic_scale * h["ac_score"]
                )

                scored.append((total_score, gpt_score, h))

            scored.sort(key=lambda x: x[0], reverse=True)
            best_sentence = scored[0][2]["sentence"]

        ref_words = ref.strip().split()
        hyp_words = best_sentence.strip().split()

        total_edits += editdistance.eval(hyp_words, ref_words)
        total_words += max(len(ref_words), 1)

        if idx < dump_examples:
            print(f"\n[sample {idx}]")
            print(f"REF: {ref}")
            print(f"HYP: {best_sentence}")

    return total_edits / max(total_words, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest", required=True, help="nbest.pkl from decode.py")
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--model_name",
        default="gpt2",
        help="Use gpt2 for faster runs, or gpt2-xl for stronger rescoring.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight on GPT score. old LM score weight is 1-alpha.",
    )
    parser.add_argument(
        "--acoustic_scale",
        type=float,
        default=0.8,
        help="Weight on acoustic score from WFST decoder.",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.0,
        help="Penalty per GPT token.",
    )
    parser.add_argument(
        "--dump_examples",
        type=int,
        default=0,
        help="Print first N reference/hypothesis examples.",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.nbest, "rb") as f:
        data = pickle.load(f)

    nbest_outputs = data["nbest"]
    transcripts = data["transcripts"]

    print(f"Loaded {len(nbest_outputs)} samples.")
    print(f"First sample has {len(nbest_outputs[0])} hypotheses.")

    model, tokenizer = load_lm(args.model_name, device)

    wer = rescore(
        nbest_outputs,
        transcripts,
        model,
        tokenizer,
        device,
        alpha=args.alpha,
        acoustic_scale=args.acoustic_scale,
        length_penalty=args.length_penalty,
        dump_examples=args.dump_examples,
    )

    print(f"WER ({args.model_name} combined rescore): {wer * 100:.2f}%")
    print(f"alpha={args.alpha}, acoustic_scale={args.acoustic_scale}, length_penalty={args.length_penalty}")


if __name__ == "__main__":
    main()
