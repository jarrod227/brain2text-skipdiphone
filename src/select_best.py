"""
Sweep all checkpoints in a run directory, compute PER on the test split,
and report the checkpoint with the lowest PER.

Useful when training was completed with val_loss-based best selection but
you want PER-based selection retroactively.

Usage:
    python src/select_best.py --run_dir experiments/variant_C_alpha0.6_beta0.1_lam0.001 --variant C
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dataset import make_dataloader
from decode import decode, _load_checkpoint
from model import SkipDiphoneDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                        help="path to experiments/<run_name>/ containing epoch_*.pt and best.pt")
    parser.add_argument("--variant", required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--promote", action="store_true",
                        help="overwrite best.pt with the lowest-PER checkpoint")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = make_dataloader(cfg.data_path, "test", cfg.batch_size,
                             num_phonemes=cfg.num_phonemes, shuffle=False)

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

    run_dir = Path(args.run_dir)
    ckpts = sorted(run_dir.glob("epoch_*.pt"))
    best_pt = run_dir / "best.pt"
    if best_pt.exists():
        ckpts.append(best_pt)

    if not ckpts:
        print(f"No checkpoints found in {run_dir}")
        return

    results = []
    for ckpt in ckpts:
        model.load_state_dict(_load_checkpoint(ckpt, device))
        per, _ = decode(model, loader, device, variant=args.variant)
        results.append((ckpt, per))
        print(f"{ckpt.name:30s} PER {per * 100:.2f}%")

    best_ckpt, best_per = min(results, key=lambda x: x[1])
    print(f"\nLowest PER: {best_ckpt.name} ({best_per * 100:.2f}%)")

    if args.promote and best_ckpt.name != "best.pt":
        sd = _load_checkpoint(best_ckpt, device)
        torch.save(sd, best_pt)
        print(f"Promoted {best_ckpt.name} -> best.pt")


if __name__ == "__main__":
    main()
