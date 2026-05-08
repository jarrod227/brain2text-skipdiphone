"""
Training entry point.

Usage:
    python src/train.py --variant E --config configs/default.yaml
    python src/train.py --variant C --lambda_smooth 5e-3 --config configs/default.yaml
"""

import argparse
import json
import random
from pathlib import Path

import editdistance
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import make_dataloader
from loss import compute_loss
from model import SkipDiphoneDecoder
from decode import _ctc_collapse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_run_name(cfg, variant, lambda_smooth):
    return f"variant_{variant}_alpha{cfg.alpha}_beta{cfg.beta}_lam{lambda_smooth}"


def train_epoch(model, loader, optimizer, cfg, variant, lambda_smooth, device):
    model.train()
    total_loss = 0.0
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

    return total_loss / len(loader)


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

        # Greedy CTC PER: micro, no SIL filter — matches cffan/neural_seq_decoder
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
    args = parser.parse_args()

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

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(cfg.data_path, "train", cfg.batch_size,
                                   num_phonemes=cfg.num_phonemes,
                                   debug_subset=cfg.debug_subset, shuffle=True)
    val_loader   = make_dataloader(cfg.data_path, "test",  cfg.batch_size,
                                   num_phonemes=cfg.num_phonemes,
                                   debug_subset=cfg.debug_subset, shuffle=False)

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
    # No scheduler is needed — kept here only as a comment for clarity.

    run_name = build_run_name(cfg, variant, lambda_smooth)
    save_dir = Path(cfg.log_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_per = float("inf")
    log = []
    print(f"Training variant {variant} for {num_epochs} epochs")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, cfg,
                                 variant, lambda_smooth, device)
        val_loss, val_per = eval_epoch(model, val_loader, cfg,
                                       variant, lambda_smooth, device)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | PER {val_per*100:.2f}%")

        log.append({"epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "val_per": val_per})
        (save_dir / "loss.json").write_text(json.dumps(log, indent=2))

        if val_per < best_per:
            best_per = val_per
            torch.save(model.state_dict(), save_dir / "best.pt")

        if epoch % cfg.save_every == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
