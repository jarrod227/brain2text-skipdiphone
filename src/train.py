"""
Training entry point.

Usage:
    python src/train.py --variant E --config configs/default.yaml
    python src/train.py --variant C --lambda_smooth 5e-3 --config configs/default.yaml
    python src/train.py --variant E --seed 1 --lambda_smooth 5e-3 \
        --config configs/default.yaml

Methodology
-----------
We split the pkl `train` bucket into a ~90% training subset and a ~10%
deterministic dev subset (every `dev_stride`-th trial per recording day).
`best_dev.pt` is selected by dev PER. The pkl `test` split is evaluated
periodically only as a tracking signal — final reported PER comes from
running `decode.py` on `best_dev.pt`.

Run names include the seed (`..._seed{N}`) so multi-seed runs land in
distinct directories.
"""

import argparse
import json
import random
from pathlib import Path

import editdistance
import numpy as np
import torch
from tqdm import tqdm

from dataset import make_dataloader
from loss import compute_loss
from model import SkipDiphoneDecoder
from decode import _ctc_collapse


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Forces cuDNN GRU to use a deterministic algorithm. Slightly
        # slower (~20% on GRU) but eliminates seed-to-seed jitter from
        # the backward pass.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_run_name(cfg, variant, lambda_smooth, seed):
    return (f"variant_{variant}_alpha{cfg.alpha}_beta{cfg.beta}"
            f"_lam{lambda_smooth}_seed{seed}")


def train_epoch(model, loader, optimizer, cfg, variant, lambda_smooth, device):
    model.train()
    total_loss = 0.0
    total_edits = 0
    total_ref_len = 0
    blank = model.num_phonemes
    white_noise_sd     = cfg.white_noise_sd
    constant_offset_sd = cfg.constant_offset_sd
    for batch in tqdm(loader, leave=False, desc="train"):
        neural   = batch["neural"].to(device)
        lengths  = batch["lengths"].to(device)
        day_ids  = batch["day_ids"].to(device)

        if white_noise_sd > 0:
            neural = neural + torch.randn_like(neural) * white_noise_sd
        if constant_offset_sd > 0:
            neural = neural + torch.randn(neural.shape[0], 1, neural.shape[2],
                                          device=device) * constant_offset_sd

        outputs = model(neural, lengths, day_ids)

        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        tlens   = {k: v.to(device) for k, v in batch["target_lengths"].items()}

        loss, _ = compute_loss(
            outputs, targets, tlens,
            cfg.alpha, cfg.beta, lambda_smooth, variant,
            cfg.num_phonemes, cfg.num_diphones,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += loss.item()

        # Cheap running train PER, useful for diagnosing overfitting
        # (loss continues to drop while PER stalls).
        with torch.no_grad():
            if variant == "A":
                lp = outputs["mono_log_probs"].permute(1, 0, 2)
            else:
                lp = torch.log(outputs["phoneme_probs"].clamp(min=1e-10))
            frame_ids = lp.argmax(dim=-1).cpu().tolist()
            enc_lens_list = outputs["enc_lengths"].cpu().tolist()
            ref_phones = batch["targets"]["mono"].cpu().tolist()
            ref_lens = batch["target_lengths"]["mono"].cpu().tolist()
            offset = 0
            for i in range(len(frame_ids)):
                hyp = _ctc_collapse(frame_ids[i][:enc_lens_list[i]], blank)
                ref = ref_phones[offset: offset + ref_lens[i]]
                total_edits += editdistance.eval(hyp, ref)
                total_ref_len += len(ref)
                offset += ref_lens[i]

    avg_loss = total_loss / len(loader)
    avg_per  = total_edits / max(total_ref_len, 1)
    return avg_loss, avg_per


@torch.no_grad()
def eval_epoch(model, loader, cfg, variant, lambda_smooth, device):
    model.eval()
    total_loss    = 0.0
    total_edits   = 0
    total_ref_len = 0
    blank = model.num_phonemes
    for batch in loader:
        neural   = batch["neural"].to(device)
        lengths  = batch["lengths"].to(device)
        day_ids  = batch["day_ids"].to(device)
        outputs = model(neural, lengths, day_ids)
        enc_lengths = outputs["enc_lengths"]
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        tlens   = {k: v.to(device) for k, v in batch["target_lengths"].items()}
        loss, _ = compute_loss(
            outputs, targets, tlens,
            cfg.alpha, cfg.beta, lambda_smooth, variant,
            cfg.num_phonemes, cfg.num_diphones,
        )
        total_loss += loss.item()

        if variant == "A":
            log_probs = outputs["mono_log_probs"].permute(1, 0, 2)
        else:
            log_probs = torch.log(outputs["phoneme_probs"].clamp(min=1e-10))
        frame_ids = log_probs.argmax(dim=-1).cpu().tolist()
        enc_lens_list = enc_lengths.cpu().tolist()
        ref_phones = batch["targets"]["mono"].cpu().tolist()
        ref_lens   = batch["target_lengths"]["mono"].cpu().tolist()
        offset = 0
        for i in range(len(frame_ids)):
            hyp = _ctc_collapse(frame_ids[i][:enc_lens_list[i]], blank)
            ref = ref_phones[offset: offset + ref_lens[i]]
            total_edits   += editdistance.eval(hyp, ref)
            total_ref_len += len(ref)
            offset += ref_lens[i]

    avg_loss = total_loss / len(loader)
    avg_per  = total_edits / max(total_ref_len, 1)
    return avg_loss, avg_per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/default.yaml")
    parser.add_argument("--variant",       required=True, choices=["A","B","C","D","E"])
    parser.add_argument("--lambda_smooth", default=None, type=float,
                        help="override smoothness weight (variants C/E)")
    parser.add_argument("--alpha",         default=None, type=float,
                        help="override std-diphone CTC weight (variants B/C/D/E)")
    parser.add_argument("--beta",          default=None, type=float,
                        help="override skip-diphone CTC weight (variants D/E)")
    parser.add_argument("--num_epochs",    default=None, type=int,
                        help="override number of training epochs")
    parser.add_argument("--seed",          default=None, type=int,
                        help="override RNG seed; run name will include it")
    parser.add_argument("--deterministic", action="store_true",
                        help="force cuDNN deterministic mode (slower)")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    variant       = args.variant
    lambda_smooth = args.lambda_smooth if args.lambda_smooth is not None else cfg.lambda_smooth
    if args.alpha is not None:
        cfg.alpha = float(args.alpha)
    if args.beta is not None:
        cfg.beta = float(args.beta)
    if args.num_epochs is not None:
        num_epochs = args.num_epochs
    else:
        num_epochs = int(cfg.num_epochs_by_variant[variant])

    seed = args.seed if args.seed is not None else int(cfg.seed)
    deterministic = bool(args.deterministic) or bool(cfg.get("cudnn_deterministic", False))
    set_seed(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev_stride = int(cfg.get("dev_stride", 10))
    eval_test_every = int(cfg.get("eval_test_every", 5))

    train_loader = make_dataloader(cfg.data_path, "train", cfg.batch_size,
                                   num_phonemes=cfg.num_phonemes,
                                   debug_subset=cfg.debug_subset, shuffle=True,
                                   dev_stride=dev_stride)

    if dev_stride > 0:
        dev_loader = make_dataloader(cfg.data_path, "dev", cfg.batch_size,
                                     num_phonemes=cfg.num_phonemes,
                                     debug_subset=cfg.debug_subset, shuffle=False,
                                     dev_stride=dev_stride)
    else:
        dev_loader = None

    test_loader = make_dataloader(cfg.data_path, "test", cfg.batch_size,
                                  num_phonemes=cfg.num_phonemes,
                                  debug_subset=cfg.debug_subset, shuffle=False,
                                  dev_stride=dev_stride)

    # If dev was disabled, fall back to old protocol (test = dev). Loud
    # warning so the user notices.
    if dev_loader is None:
        print("[warn] dev_stride=0; using test split for best.pt selection (legacy mode)")
        dev_loader = test_loader

    model = SkipDiphoneDecoder(
        input_dim=cfg.encoder.input_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        num_layers=cfg.encoder.num_layers,
        num_phonemes=cfg.num_phonemes,
        num_diphones=cfg.num_diphones,
        num_days=cfg.num_days,
        variant=variant,
        kernel_len=cfg.encoder.kernel_len,
        stride_len=cfg.encoder.stride_len,
        gaussian_smooth_width=cfg.encoder.gaussian_smooth_width,
        dropout=cfg.encoder.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    # Constant LR, matching cffan/neural_seq_decoder (lrStart == lrEnd == 0.02).

    run_name = build_run_name(cfg, variant, lambda_smooth, seed)
    save_dir = Path(cfg.log_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run]  {run_name}")
    print(f"[data] train={len(train_loader.dataset)} dev={len(dev_loader.dataset)} "
          f"test={len(test_loader.dataset)}")
    print(f"[meta] seed={seed} deterministic={deterministic} epochs={num_epochs}")

    best_dev_per = float("inf")
    best_dev_epoch = -1
    best_dev_test_per = float("nan")
    log = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_per = train_epoch(
            model, train_loader, optimizer, cfg, variant, lambda_smooth, device)
        dev_loss, dev_per = eval_epoch(
            model, dev_loader, cfg, variant, lambda_smooth, device)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss, "train_per": train_per,
            "dev_loss": dev_loss, "dev_per": dev_per,
        }
        msg = (f"Epoch {epoch:03d} | "
               f"train {train_loss:.4f} PER {train_per*100:.2f}% | "
               f"dev {dev_loss:.4f} PER {dev_per*100:.2f}%")

        if eval_test_every > 0 and (epoch % eval_test_every == 0 or epoch == num_epochs):
            test_loss, test_per = eval_epoch(
                model, test_loader, cfg, variant, lambda_smooth, device)
            entry["test_loss"] = test_loss
            entry["test_per"] = test_per
            msg += f" | test {test_loss:.4f} PER {test_per*100:.2f}%"

        print(msg)
        log.append(entry)
        (save_dir / "loss.json").write_text(json.dumps(log, indent=2))

        if dev_per < best_dev_per:
            best_dev_per = dev_per
            best_dev_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_dev.pt")
            best_dev_test_per = entry.get("test_per", float("nan"))

        if epoch % cfg.save_every == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch:03d}.pt")

    # Always save the final-epoch checkpoint as a methodological alternative
    # to best_dev.pt (some venues report final-epoch PER instead).
    torch.save(model.state_dict(), save_dir / "final.pt")

    summary = {
        "run": run_name,
        "variant": variant,
        "seed": seed,
        "num_epochs": num_epochs,
        "best_dev_epoch": best_dev_epoch,
        "best_dev_per": best_dev_per,
        "test_per_at_best_dev": best_dev_test_per,
        "final_dev_per": log[-1]["dev_per"],
        "final_test_per": log[-1].get("test_per", None),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] best_dev epoch={best_dev_epoch} dev PER {best_dev_per*100:.2f}% "
          f"(test PER at that epoch: {best_dev_test_per*100:.2f}%)")


if __name__ == "__main__":
    main()
