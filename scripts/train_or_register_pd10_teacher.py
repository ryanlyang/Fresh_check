#!/usr/bin/env python3
"""Train or register one PD10 ParT teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_10class import (  # noqa: E402
    PD10_PART_TEACHER_TARGETS,
    PD10_SPLIT_SIZES,
    PD10PartTeacherTrainConfig,
    default_pd10_experiment_layout,
    normalize_pd10_part_teacher_target,
    register_pd10_part_teacher_checkpoint,
    train_pd10_part_teacher,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=PD10_PART_TEACHER_TARGETS, required=True)
    parser.add_argument("--manifest", default=str(layout.split_manifest_path))
    parser.add_argument("--hlt-cache-dir", default=str(layout.hlt_cache_dir))
    parser.add_argument(
        "--offline-cache-dir",
        default=None,
        help="Optional cached offline inputs for offline teacher training/evaluation.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Teacher output directory; defaults to the canonical PD10 teacher directory.",
    )
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=PD10_SPLIT_SIZES["model_train"])
    parser.add_argument("--max-val-jets", type=int, default=PD10_SPLIT_SIZES["model_val"])
    parser.add_argument("--max-final-test-jets", type=int, default=PD10_SPLIT_SIZES["final_test"])
    parser.add_argument("--model-size", choices=["base", "tiny", "large"], default="base")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required before evaluating the selected checkpoint on final_test.",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Debug/registration escape hatch; canonical PD10 runs should not use this.",
    )
    parser.add_argument(
        "--register-checkpoint",
        default=None,
        help="Copy an existing trusted teacher checkpoint instead of training.",
    )
    parser.add_argument(
        "--register-source-report",
        default=None,
        help="Optional source model-val report JSON to copy beside a registered checkpoint.",
    )
    parser.add_argument(
        "--register-source-final-test-report",
        default=None,
        help="Optional source final-test report JSON to copy beside a registered checkpoint.",
    )
    parser.add_argument("--overwrite-registration", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    teacher = normalize_pd10_part_teacher_target(args.teacher)
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    output_dir = args.output_dir or str(layout.teacher_dir(teacher))
    config = PD10PartTeacherTrainConfig(
        teacher_target=teacher,
        output_dir=output_dir,
        manifest_path=args.manifest,
        cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
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
        max_final_test_batches=args.max_final_test_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        model_size=args.model_size,
        compile_model=args.compile_model,
        verify_label_branches=args.verify_label_branches,
        read_chunk_size=args.read_chunk_size,
        confirm_final_test=args.confirm_final_test,
        evaluate_final_test=not args.skip_final_test,
    )
    if args.register_checkpoint:
        report = register_pd10_part_teacher_checkpoint(
            config,
            source_checkpoint=args.register_checkpoint,
            source_model_val_report=args.register_source_report,
            source_final_test_report=args.register_source_final_test_report,
            overwrite=args.overwrite_registration,
        )
    else:
        report = train_pd10_part_teacher(config)
    print("pd10_teacher_complete:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
