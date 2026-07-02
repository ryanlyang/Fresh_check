#!/usr/bin/env python3
"""Train the PD10 dual-view logit-fusion teacher and cache its logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_10class import (  # noqa: E402
    PD10_DUAL_VIEW_DEFAULT_BATCH_SIZE,
    PD10_DUAL_VIEW_DEFAULT_DROPOUT,
    PD10_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE,
    PD10_DUAL_VIEW_DEFAULT_EPOCHS,
    PD10_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE,
    PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM,
    PD10_DUAL_VIEW_DEFAULT_LR,
    PD10_DUAL_VIEW_DEFAULT_SEED,
    PD10_DUAL_VIEW_DEFAULT_WEIGHT_DECAY,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_LOGIT_SPLITS,
    PD10DualViewLogitTeacherConfig,
    default_pd10_experiment_layout,
    train_pd10_dual_view_logit_teacher,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-logit-dir",
        default=str(layout.teacher_logits_dir),
        help="Root containing cached HLT/offline teacher logits.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layout.teacher_dir(PD10_TEACHER_DUAL_VIEW)),
        help="Directory for the selected dual-view teacher checkpoint and reports.",
    )
    parser.add_argument(
        "--prediction-output-dir",
        default=str(layout.teacher_logits_dir),
        help="Root where dual-view teacher prediction blocks are written.",
    )
    parser.add_argument("--splits", nargs="+", choices=PD10_TEACHER_LOGIT_SPLITS, default=list(PD10_TEACHER_LOGIT_SPLITS))
    parser.add_argument("--seed", type=int, default=PD10_DUAL_VIEW_DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=PD10_DUAL_VIEW_DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=PD10_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=PD10_DUAL_VIEW_DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=PD10_DUAL_VIEW_DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=PD10_DUAL_VIEW_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--hidden-dim", type=int, default=PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=PD10_DUAL_VIEW_DEFAULT_DROPOUT)
    parser.add_argument("--early-stop-patience", type=int, default=PD10_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=PD10_SPLIT_SIZES["model_train"])
    parser.add_argument("--max-val-jets", type=int, default=PD10_SPLIT_SIZES["model_val"])
    parser.add_argument("--max-final-test-jets", type=int, default=PD10_SPLIT_SIZES["final_test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PD10DualViewLogitTeacherConfig(
        output_dir=args.output_dir,
        teacher_logit_dir=args.teacher_logit_dir,
        prediction_output_dir=args.prediction_output_dir,
        prediction_splits=tuple(args.splits),
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        early_stop_patience=args.early_stop_patience,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        overwrite=bool(args.overwrite),
        skip_existing_predictions=not bool(args.no_skip_existing_predictions),
        confirm_final_test=bool(args.confirm_final_test),
    )
    report = train_pd10_dual_view_logit_teacher(config)
    print("pd10_dual_view_logit_teacher_complete:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
