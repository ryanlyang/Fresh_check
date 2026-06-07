#!/usr/bin/env python3
"""Train and/or fuse the Step 11 five-seed HLT-only control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID  # noqa: E402
from jetclass_fresh.hlt_control import DEFAULT_HLT5_CHECKPOINT_ROOT, HLT5_SEEDS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--baseline-root", default=DEFAULT_HLT5_CHECKPOINT_ROOT)
    parser.add_argument("--fusion-output-dir", default="checkpoints/jetclass_fresh_fusion/hlt5_seed_control")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(HLT5_SEEDS))
    parser.add_argument("--stage", choices=["train", "fusion", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--model-size", choices=["base", "tiny"], default="base")
    parser.add_argument("--compile-model", action="store_true")

    parser.add_argument("--splits", nargs="+", choices=["stack_train", "stack_val", "final_test"], default=["stack_train", "stack_val", "final_test"])
    parser.add_argument("--confirm-final-test", action="store_true", help="Required when fusion --splits includes final_test")
    parser.add_argument("--max-jets-per-split", type=int, default=None)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--C-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--feature-mode", choices=["logits", "probs", "logits_probs"], default="logits_probs")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--compare-reco7-report", default=None)
    parser.add_argument("--comparison-output", default=None)
    return parser.parse_args(argv)


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
            "--grad-clip-norm",
            str(args.grad_clip_norm),
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--model-size",
            args.model_size,
        ]
    )
    if args.no_amp:
        cmd.append("--no-amp")
    if args.compile_model:
        cmd.append("--compile-model")
    if args.max_train_batches is not None:
        cmd.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_val_batches is not None:
        cmd.extend(["--max-val-batches", str(args.max_val_batches)])
    if args.max_train_jets is not None:
        cmd.extend(["--max-train-jets", str(args.max_train_jets)])
    if args.max_val_jets is not None:
        cmd.extend(["--max-val-jets", str(args.max_val_jets)])
    return cmd


def hlt_training_command(args: argparse.Namespace, seed: int) -> list[str]:
    output_dir = Path(args.baseline_root) / f"seed{int(seed)}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_hlt_baseline.py"),
        "--cache-dir",
        args.hlt_cache_dir,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(int(seed)),
    ]
    return append_common_training_args(cmd, args)


def hlt5_fusion_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_hlt5_fusion.py"),
        "--hlt-cache-dir",
        args.hlt_cache_dir,
        "--hlt-checkpoint-root",
        args.baseline_root,
        "--output-dir",
        args.fusion_output_dir,
        "--seeds",
        *[str(int(seed)) for seed in args.seeds],
        "--splits",
        *list(args.splits),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--C-grid",
        *[str(value) for value in args.C_grid],
        "--feature-mode",
        args.feature_mode,
        "--max-iter",
        str(args.max_iter),
    ]
    if args.confirm_final_test:
        cmd.append("--confirm-final-test")
    if args.max_jets_per_split is not None:
        cmd.extend(["--max-jets-per-split", str(args.max_jets_per_split)])
    if args.overwrite_predictions:
        cmd.append("--overwrite-predictions")
    if args.no_skip_existing_predictions:
        cmd.append("--no-skip-existing-predictions")
    if args.compare_reco7_report:
        cmd.extend(["--compare-reco7-report", args.compare_reco7_report])
    if args.comparison_output:
        cmd.extend(["--comparison-output", args.comparison_output])
    return cmd


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    if args.stage in ("train", "both"):
        commands.extend(hlt_training_command(args, seed) for seed in args.seeds)
    if args.stage in ("fusion", "both"):
        commands.append(hlt5_fusion_command(args))
    return commands


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_commands(args)
    print(json.dumps({"commands": commands}, indent=2))
    if args.dry_run:
        return 0
    for cmd in commands:
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
