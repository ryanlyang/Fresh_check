#!/usr/bin/env python3
"""Train one deployable PD10 HLT self-dualview model."""

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
    HLTSDVTrainConfig,
    HLT_SDV_VARIANT_SAME_VIEW,
    default_hlt2_cache_dir,
    default_hlt_sdv_experiment_layout,
    hlt_sdv_branch2_mode_from_variant,
    hlt_sdv_dual_hlt2_variant_name,
    hlt_sdv_strength_from_variant,
    normalize_hlt_sdv_variant,
    train_hlt_sdv_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pd10-root",
        default=os.environ.get("PD10_ROOT"),
        help="Existing PD10 root; defaults to $PD10_ROOT or checkpoints/privileged_distill_10class_5m.",
    )
    parser.add_argument("--variant", default=hlt_sdv_dual_hlt2_variant_name(0.20))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hlt-cache-dir", default=None)
    parser.add_argument("--hlt2-cache-dir", default=None)
    parser.add_argument("--hlt-teacher-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--head-warmup-lr", type=float, default=1.0e-3)
    parser.add_argument("--branch-lr", type=float, default=3.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--representation-dim", type=int, default=256)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=8801)
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "large"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=5_000_000)
    parser.add_argument("--max-val-jets", type=int, default=1_000_000)
    parser.add_argument("--max-final-test-jets", type=int, default=1_000_000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-branch-init", action="store_true")
    parser.add_argument("--skip-model-val-predictions", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _pd10_root(args: argparse.Namespace) -> Path:
    if args.pd10_root:
        return Path(args.pd10_root)
    return Path("checkpoints") / "privileged_distill_10class_5m"


def build_config(args: argparse.Namespace) -> HLTSDVTrainConfig:
    pd10_root = _pd10_root(args)
    variant = normalize_hlt_sdv_variant(args.variant)
    layout = default_hlt_sdv_experiment_layout(
        output_root=pd10_root.parent,
        pd10_experiment_name=pd10_root.name,
    )
    branch2_mode = hlt_sdv_branch2_mode_from_variant(variant)
    strength = hlt_sdv_strength_from_variant(variant)
    if args.hlt2_cache_dir:
        hlt2_cache_dir = args.hlt2_cache_dir
    elif variant == HLT_SDV_VARIANT_SAME_VIEW:
        hlt2_cache_dir = None
    else:
        hlt2_cache_dir = str(default_hlt2_cache_dir(pd10_root, float(strength)))
    return HLTSDVTrainConfig(
        output_dir=str(Path(args.output_dir) if args.output_dir else layout.variant_dir(variant)),
        hlt_cache_dir=str(Path(args.hlt_cache_dir) if args.hlt_cache_dir else pd10_root / "hlt_cache"),
        hlt2_cache_dir=hlt2_cache_dir,
        hlt_teacher_checkpoint=str(
            Path(args.hlt_teacher_checkpoint) if args.hlt_teacher_checkpoint else layout.hlt_teacher_checkpoint
        ),
        variant_name=variant,
        branch2_mode=branch2_mode,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        head_warmup_epochs=args.head_warmup_epochs,
        head_warmup_lr=args.head_warmup_lr,
        branch_lr=args.branch_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        fusion_hidden_dim=args.fusion_hidden_dim,
        representation_dim=args.representation_dim,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.no_amp),
        compile_model=bool(args.compile_model),
        initialize_branches=not bool(args.no_branch_init),
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
    report = train_hlt_sdv_model(build_config(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
