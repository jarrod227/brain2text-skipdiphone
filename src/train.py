"""
Training entry point.

Usage:
    python src/train.py --variant E --config configs/default.yaml
    python src/train.py --variant C --lambda_smooth 5e-3 --config configs/default.yaml
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import make_dataloader
from loss import compute_loss
from model import SkipDiphoneDecoder


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
    for batch in tqdm(loader, leave=False, desc="train"):
        neural  = batch["neural"].to(device)
        lengths = batch["lengths"].to(device)

        mono_lp, dip_lp, skip_lp, phone_p = model(neural, lengths)

        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        tlens   = {k: v.to(device) for k, v in batch["target_lengths"].items()}

        loss, _ = compute_loss(
            mono_lp, dip_lp, skip_lp, phone_p,
            targets, lengths, tlens,
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
    total_loss = 0.0
    for batch in loader:
        neural  = batch["neural"].to(device)
        lengths = batch["lengths"].to(device)
        mono_lp, dip_lp, skip_lp, phone_p = model(neural, lengths)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        tlens   = {k: v.to(device) for k, v in batch["target_lengths"].items()}
        loss, _ = compute_loss(
            mono_lp, dip_lp, skip_lp, phone_p,
            targets, lengths, tlens,
            cfg.alpha, cfg.beta, lambda_smooth, variant,
            cfg.num_phonemes, cfg.num_diphones,
        )
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/default.yaml")
    parser.add_argument("--variant",       default=None, choices=["A","B","C","D","E"])
    parser.add_argument("--lambda_smooth", default=None, type=float)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    variant      = args.variant      or cfg.variant
    lambda_smooth = args.lambda_smooth if args.lambda_smooth is not None else cfg.lambda_smooth

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = sorted(glob.glob(os.path.join(cfg.data_dir, "train", "*.tfrecord")))
    val_paths   = sorted(glob.glob(os.path.join(cfg.data_dir, "test",  "*.tfrecord")))

    train_loader = make_dataloader(train_paths, cfg.batch_size,
                                   debug_subset=cfg.debug_subset, shuffle=True)
    val_loader   = make_dataloader(val_paths,   cfg.batch_size,
                                   debug_subset=cfg.debug_subset, shuffle=False)

    model = SkipDiphoneDecoder(
        input_dim=cfg.encoder.input_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        num_layers=cfg.encoder.num_layers,
        num_phonemes=cfg.num_phonemes,
        num_diphones=cfg.num_diphones,
        dropout=cfg.encoder.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )

    run_name = build_run_name(cfg, variant, lambda_smooth)
    save_dir = Path(cfg.log_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    log = []
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, cfg,
                                 variant, lambda_smooth, device)
        val_loss   = eval_epoch(model, val_loader, cfg, variant, lambda_smooth, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        (save_dir / "loss.json").write_text(json.dumps(log, indent=2))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")

        if epoch % cfg.save_every == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
