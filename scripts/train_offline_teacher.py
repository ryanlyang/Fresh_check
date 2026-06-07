#!/usr/bin/env python3
"""Train the Step 6 offline-only Particle Transformer teacher reference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.offline_teacher import OfflineTeacherTrainConfig, train_offline_teacher  # noqa: E402
from jetclass_fresh.hlt_baseline import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Step 2 split manifest path (.json or .json.gz)")
    parser.add_argument("--data-dir", default=None, help="Override JetClass data directory")
    parser.add_argument("--output-dir", default="checkpoints/jetclass_fresh_offline_teacher/offline_teacher_seed707")
    parser.add_argument("--seed", type=int, default=707)
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
    parser.add_argument("--max-train-jets", type=int, default=None, help="Debug-only limit after loading model_train")
    parser.add_argument("--max-val-jets", type=int, default=None, help="Debug-only limit after loading model_val")
    parser.add_argument("--model-size", choices=["base", "tiny"], default="base")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OfflineTeacherTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        model_size=args.model_size,
        compile_model=args.compile_model,
        verify_label_branches=args.verify_label_branches,
        read_chunk_size=args.read_chunk_size,
    )
    report = train_offline_teacher(
        config,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
    )
    save_json(Path(args.output_dir) / "run_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
