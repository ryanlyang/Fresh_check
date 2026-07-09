#!/usr/bin/env python3
"""Train an HLT-MV N-view particle fusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_self_dualview.multiview import (  # noqa: E402
    HLTMultiViewSpec,
    HLTMultiViewTrainConfig,
    train_hlt_multiview_model,
)


def _parse_view_spec(text: str) -> HLTMultiViewSpec:
    parts = str(text).split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--view must be name,source_view,cache_dir,checkpoint; "
            f"got {text!r}"
        )
    name, source_view, cache_dir, checkpoint = parts
    return HLTMultiViewSpec(
        name=name,
        source_view=source_view,
        cache_dir=cache_dir,
        checkpoint=checkpoint,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--view",
        action="append",
        type=_parse_view_spec,
        required=True,
        help="Repeat as name,source_view,cache_dir,checkpoint. First view must be fixed_hlt.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--head-warmup-lr", type=float, default=3.0e-4)
    parser.add_argument("--branch-lr", type=float, default=2.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--representation-dim", type=int, default=256)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "large"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=5_000_000)
    parser.add_argument("--max-val-jets", type=int, default=1_000_000)
    parser.add_argument("--max-final-test-jets", type=int, default=1_000_000)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP. Default is full precision.")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-model-val-predictions", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = HLTMultiViewTrainConfig(
        output_dir=str(Path(args.output_dir)),
        view_specs=tuple(args.view),
        model_name=args.model_name,
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
    report = train_hlt_multiview_model(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
