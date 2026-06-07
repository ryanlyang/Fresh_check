#!/usr/bin/env python3
"""Run Stage A and/or Stage2 training commands for the seven reco variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.reconstructor import RECONSTRUCTOR_VARIANT_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Step 2 split manifest path")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-root", default="checkpoints/jetclass_fresh_reco7")
    parser.add_argument("--hlt-baseline-report", default=None)
    parser.add_argument("--variants", nargs="+", choices=RECONSTRUCTOR_VARIANT_NAMES, default=list(RECONSTRUCTOR_VARIANT_NAMES))
    parser.add_argument("--stage", choices=["stage-a", "stage2", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--model-size", choices=["base", "tiny"], default="base", help="Stage2 dual-view Particle Transformer size")
    return parser.parse_args()


def append_common_training_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd.extend(
        [
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--early-stop-patience",
            str(args.early_stop_patience),
        ]
    )
    if args.no_amp:
        cmd.append("--no-amp")
    if args.max_train_batches is not None:
        cmd.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_val_batches is not None:
        cmd.extend(["--max-val-batches", str(args.max_val_batches)])
    if args.max_train_jets is not None:
        cmd.extend(["--max-train-jets", str(args.max_train_jets)])
    if args.max_val_jets is not None:
        cmd.extend(["--max-val-jets", str(args.max_val_jets)])
    return cmd


def stage_a_command(args: argparse.Namespace, variant: str) -> list[str]:
    output_dir = Path(args.output_root) / variant / "stage_a"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_reconstructor_stage_a.py"),
        "--manifest",
        args.manifest,
        "--hlt-cache-dir",
        args.hlt_cache_dir,
        "--output-dir",
        str(output_dir),
        "--variant",
        variant,
    ]
    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])
    return append_common_training_args(cmd, args)


def stage2_command(args: argparse.Namespace, variant: str) -> list[str]:
    output_dir = Path(args.output_root) / variant / "stage2_dual_view"
    checkpoint = Path(args.output_root) / variant / "stage_a" / "best_model_val.pt"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_dual_view_tagger.py"),
        "--hlt-cache-dir",
        args.hlt_cache_dir,
        "--reconstructor-checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--variant",
        variant,
        "--model-size",
        args.model_size,
    ]
    if args.hlt_baseline_report:
        cmd.extend(["--hlt-baseline-report", args.hlt_baseline_report])
    return append_common_training_args(cmd, args)


def main() -> int:
    args = parse_args()
    commands: list[list[str]] = []
    for variant in args.variants:
        if args.stage in ("stage-a", "both"):
            commands.append(stage_a_command(args, variant))
        if args.stage in ("stage2", "both"):
            commands.append(stage2_command(args, variant))

    print(json.dumps({"commands": commands}, indent=2))
    if args.dry_run:
        return 0
    for cmd in commands:
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
