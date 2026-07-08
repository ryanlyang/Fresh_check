#!/usr/bin/env python3
"""Train one scratch particle dual-view model for HLT multiview fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_multiview_source_fusion import (  # noqa: E402
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BATCH_SIZE,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BRANCH_LR,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_DROPOUT,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EPOCHS,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_GRAD_CLIP_NORM,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_LR,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_EPOCHS,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_LR,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_TRAIN_JETS,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_VAL_JETS,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MODEL_SIZE,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_REPRESENTATION_DIM,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_SEED,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_WEIGHT_DECAY,
    build_hlt_mv_scratch_dualview_config,
    train_hlt_mv_scratch_dualview,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--pdv3-experiment-name", default=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hlt-cache-dir", default=None)
    parser.add_argument("--hlt2-cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EPOCHS)
    parser.add_argument(
        "--head-warmup-epochs",
        type=int,
        default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_EPOCHS,
        help="Must remain 0 for scratch dual-view runs.",
    )
    parser.add_argument("--batch-size", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--head-warmup-lr", type=float, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_LR)
    parser.add_argument("--branch-lr", type=float, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BRANCH_LR)
    parser.add_argument("--head-lr", type=float, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_LR)
    parser.add_argument("--weight-decay", type=float, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_DROPOUT)
    parser.add_argument("--fusion-hidden-dim", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM)
    parser.add_argument("--representation-dim", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_REPRESENTATION_DIM)
    parser.add_argument("--early-stop-patience", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--grad-clip-norm", type=float, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_GRAD_CLIP_NORM)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_SEED)
    parser.add_argument(
        "--model-size",
        default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MODEL_SIZE,
        choices=["tiny", "base", "large"],
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_TRAIN_JETS)
    parser.add_argument("--max-val-jets", type=int, default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_VAL_JETS)
    parser.add_argument(
        "--max-final-test-jets",
        type=int,
        default=HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS,
    )
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP. Default is full precision.")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-model-val-predictions", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_hlt_mv_scratch_dualview_config(
        model_name=args.variant,
        output_root=args.output_root,
        pdv3_experiment_name=args.pdv3_experiment_name,
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        hlt_cache_dir=None if args.hlt_cache_dir is None else Path(args.hlt_cache_dir),
        hlt2_cache_dir=None if args.hlt2_cache_dir is None else Path(args.hlt2_cache_dir),
        seed=args.seed,
        epochs=args.epochs,
        head_warmup_epochs=args.head_warmup_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        head_warmup_lr=args.head_warmup_lr,
        branch_lr=args.branch_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        fusion_hidden_dim=args.fusion_hidden_dim,
        representation_dim=args.representation_dim,
        early_stop_patience=args.early_stop_patience,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        device=args.device,
        amp=bool(args.amp),
        compile_model=bool(args.compile_model),
        evaluate_model_val_predictions=not bool(args.skip_model_val_predictions),
        evaluate_final_test=not bool(args.skip_final_test),
        confirm_final_test=bool(args.confirm_final_test),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_final_test_batches=args.max_final_test_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        model_size=args.model_size,
        overwrite=bool(args.overwrite),
    )
    report = train_hlt_mv_scratch_dualview(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
