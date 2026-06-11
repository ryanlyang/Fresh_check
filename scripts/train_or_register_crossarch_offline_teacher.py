#!/usr/bin/env python3
"""Train or register one cross-architecture offline teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.crossarch_experiment import TEACHER_ARCHITECTURES, CrossArchExperimentLayout  # noqa: E402
from teacher_logit_reco.crossarch_offline_teachers import (  # noqa: E402
    CrossArchOfflineTeacherTrainConfig,
    normalize_crossarch_teacher_architecture,
    register_crossarch_offline_teacher_checkpoint,
    train_crossarch_offline_teacher,
)


def parse_args() -> argparse.Namespace:
    layout = CrossArchExperimentLayout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=TEACHER_ARCHITECTURES, required=True)
    parser.add_argument("--manifest", default=str(layout.split_manifest_path))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--model-size", choices=["base", "tiny"], default="base")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--register-checkpoint",
        default=None,
        help="Copy an existing trusted offline teacher checkpoint instead of training.",
    )
    parser.add_argument(
        "--register-source-report",
        default=None,
        help="Optional source run/report JSON to copy beside a registered checkpoint.",
    )
    parser.add_argument("--overwrite-registration", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arch = normalize_crossarch_teacher_architecture(args.architecture)
    if args.register_checkpoint:
        report = register_crossarch_offline_teacher_checkpoint(
            architecture=arch,
            source_checkpoint=args.register_checkpoint,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            source_report=args.register_source_report,
            overwrite=args.overwrite_registration,
        )
    else:
        config = CrossArchOfflineTeacherTrainConfig(
            architecture=arch,
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
            max_train_jets=args.max_train_jets,
            max_val_jets=args.max_val_jets,
            model_size=args.model_size,
            compile_model=args.compile_model,
            verify_label_branches=args.verify_label_branches,
            read_chunk_size=args.read_chunk_size,
        )
        report = train_crossarch_offline_teacher(config)
    print("crossarch_offline_teacher_complete:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
