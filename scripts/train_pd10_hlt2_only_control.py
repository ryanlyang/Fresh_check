#!/usr/bin/env python3
"""Train the deployable PD10 HLT2-only ParT control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_self_dualview import (  # noqa: E402
    HLT2OnlyTrainConfig,
    default_hlt2_cache_dir,
    default_hlt_sdv_experiment_layout,
    hlt_sdv_strength_tag,
    train_hlt2_only_control,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pd10-root",
        default=os.environ.get("PD10_ROOT"),
        help="Existing PD10 root; defaults to $PD10_ROOT or checkpoints/privileged_distill_10class_5m.",
    )
    parser.add_argument("--strength", type=float, default=0.20)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hlt2-cache-dir", default=None)
    parser.add_argument("--hlt-teacher-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=8811)
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "large"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=5_000_000)
    parser.add_argument("--max-val-jets", type=int, default=1_000_000)
    parser.add_argument("--max-final-test-jets", type=int, default=1_000_000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--skip-model-val-predictions", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _pd10_root(args: argparse.Namespace) -> Path:
    if args.pd10_root:
        return Path(args.pd10_root)
    return Path("checkpoints") / "privileged_distill_10class_5m"


def _variant(args: argparse.Namespace) -> str:
    if args.variant:
        return str(args.variant)
    if abs(float(args.strength) - 0.20) <= 1.0e-12:
        return "hlt2_only_part_s0p20"
    return f"hlt2_only_part_{hlt_sdv_strength_tag(args.strength)}"


def build_config(args: argparse.Namespace) -> HLT2OnlyTrainConfig:
    pd10_root = _pd10_root(args)
    layout = default_hlt_sdv_experiment_layout(
        output_root=pd10_root.parent,
        pd10_experiment_name=pd10_root.name,
    )
    variant = _variant(args)
    return HLT2OnlyTrainConfig(
        output_dir=str(Path(args.output_dir) if args.output_dir else layout.variant_dir(variant)),
        hlt2_cache_dir=str(
            Path(args.hlt2_cache_dir) if args.hlt2_cache_dir else default_hlt2_cache_dir(pd10_root, args.strength)
        ),
        hlt_teacher_checkpoint=str(
            Path(args.hlt_teacher_checkpoint) if args.hlt_teacher_checkpoint else layout.hlt_teacher_checkpoint
        ),
        variant_name=variant,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.no_amp),
        compile_model=bool(args.compile_model),
        initialize_from_hlt_checkpoint=not bool(args.no_warm_start),
        evaluate_model_val_predictions=not bool(args.skip_model_val_predictions),
        evaluate_final_test=not bool(args.skip_final_test),
        confirm_final_test=bool(args.confirm_final_test),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_final_test_batches=args.max_final_test_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        overwrite=bool(args.overwrite),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = train_hlt2_only_control(build_config(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
