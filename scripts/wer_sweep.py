"""
Grid sweep over WFST decoder hyperparameters (acoustic_scale, blank_penalty)
for a single trained checkpoint.

Acoustic gains at the PER level often do not translate to WER unless the
WFST acoustic_scale (and to a lesser extent blank_penalty) is recalibrated
for the new acoustic model. This script loads the model and the test loader
once, then iterates over a grid and reports WER for each cell.

Run inside the `lm_decode` conda environment.

Default grid (5x5):
    acoustic_scales = 0.3, 0.5, 0.8, 1.0, 1.2
    blank_penalties = 1.0, 2.0, 3.0, 4.0, 5.0
The bp range starts at 1.0 because earlier sweeps showed bp<1.0 is
strictly dominated by bp>=1.0 on this acoustic model + LM combination
(mono A and diphone B both peak at the upper end of [0, 2]). Values
above 2.0 are needed because B's bp curve was still monotonically
improving at bp=2.0 (cffan's log(7)~=1.95 baseline is for 5-gram and
under-penalizes blank with our 3-gram setup).

Usage:
    python scripts/wer_sweep.py \
        --checkpoint experiments/variant_E_alpha0.6_beta0.1_lam0.005/best.pt \
        --variant E \
        --lm 3gram \
        --lm_dir data/languageModel \
        --beam 17.0 \
        --out experiments/variant_E_alpha0.6_beta0.1_lam0.005/wer_sweep.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Make `import dataset / model / decode` work whether the script is run
# from the repo root or from inside scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import make_dataloader  # noqa: E402
from decode import (  # noqa: E402
    decode,
    _build_lm_decoder,
    _load_checkpoint,
    compute_log_priors,
)
from model import SkipDiphoneDecoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lm", default="3gram", choices=["3gram", "5gram"])
    parser.add_argument("--lm_dir", default=None)
    parser.add_argument("--beam", default=17.0, type=float)
    parser.add_argument(
        "--acoustic_scales",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.8, 1.0, 1.2],
    )
    parser.add_argument(
        "--blank_penalties",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"],
                        help="which split to sweep on (default: test)")
    parser.add_argument("--log_priors", action="store_true",
                        help="opt-in: subtract training-set log-priors. Off "
                             "by default to match speechBCI/cffan (zeros).")
    parser.add_argument("--rescore", action="store_true",
                        help="opt-in: run lattice rescoring with G_no_prune.fst "
                             "(loads ~75GB into RAM). Off by default; the "
                             "sweep's job is to find optimal (ac, bp), not "
                             "to do final eval with rescore.")
    parser.add_argument("--out", default=None,
                        help="optional CSV file to dump the full grid")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
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
    model.load_state_dict(_load_checkpoint(args.checkpoint, device), strict=False)
    model.eval()

    loader = make_dataloader(cfg.data_path, args.split, cfg.batch_size,
                             num_phonemes=cfg.num_phonemes, shuffle=False,
                             dev_stride=dev_stride)

    log_priors = None
    if args.log_priors:
        log_priors = compute_log_priors(cfg.data_path, cfg.num_phonemes,
                                        dev_stride=dev_stride)
        print(f"[wer_sweep] log_priors computed from train split, "
              f"shape={log_priors.shape} (opt-in)")

    rows = []
    print(f"{'ac_scale':>10} {'blank_pen':>10} {'PER %':>8} {'WER %':>8}")
    print("-" * 40)
    for ac in args.acoustic_scales:
        # Build the WFST decoder once per acoustic_scale and reuse it
        # across all blank_penalty values. The 5-gram TLG.fst is ~42GB,
        # so this avoids 20 redundant 42GB loads on the default 5x5 grid.
        print(f"[wer_sweep] loading decoder for acoustic_scale={ac:.2f} ...",
              flush=True)
        decoder_lm = _build_lm_decoder(
            lm_dir, nbest=1, acoustic_scale=ac, beam=args.beam,
            rescore=args.rescore,
        )
        for bp in args.blank_penalties:
            per, wer = decode(
                model,
                loader,
                device,
                variant=args.variant,
                lm=args.lm,
                lm_dir=lm_dir,
                acoustic_scale=ac,
                beam=args.beam,
                blank_penalty=bp,
                log_priors=log_priors,
                decoder=decoder_lm,
                rescore=args.rescore,
            )
            print(f"{ac:>10.2f} {bp:>10.2f} {per * 100:>7.2f}% {wer * 100:>7.2f}%")
            rows.append({
                "variant": args.variant,
                "acoustic_scale": ac,
                "blank_penalty": bp,
                "beam": args.beam,
                "PER": per,
                "WER": wer,
            })
        # Drop the decoder before building the next one to free FST memory.
        del decoder_lm

    # Best WER cell
    best = min(rows, key=lambda r: r["WER"])
    print("-" * 40)
    print(f"Best WER: {best['WER'] * 100:.2f}% at "
          f"acoustic_scale={best['acoustic_scale']}, "
          f"blank_penalty={best['blank_penalty']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote sweep to {out_path}")


if __name__ == "__main__":
    main()
