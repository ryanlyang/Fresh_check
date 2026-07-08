#!/usr/bin/env python3
"""Train one single-view source branch for the deployable HLT tri-view test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_self_dualview import (  # noqa: E402
    HLT_TRIVIEW_DEFAULT_FINAL_TEST_JETS,
    HLT_TRIVIEW_DEFAULT_TRAIN_JETS,
    HLT_TRIVIEW_DEFAULT_VAL_JETS,
    HLTTriViewSourceConfig,
    train_hlt_triview_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-view", choices=["fixed_hlt", "hlt2"], required=True)
    parser.add_argument("--warm-start-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "large"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=HLT_TRIVIEW_DEFAULT_TRAIN_JETS)
    parser.add_argument("--max-val-jets", type=int, default=HLT_TRIVIEW_DEFAULT_VAL_JETS)
    parser.add_argument("--max-final-test-jets", type=int, default=HLT_TRIVIEW_DEFAULT_FINAL_TEST_JETS)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-model-val-predictions", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> HLTTriViewSourceConfig:
    return HLTTriViewSourceConfig(
        output_dir=str(Path(args.output_dir)),
        cache_dir=str(Path(args.cache_dir)),
        source_name=str(args.source_name),
        source_view=str(args.source_view),
        warm_start_checkpoint=None if args.warm_start_checkpoint is None else str(Path(args.warm_start_checkpoint)),
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = train_hlt_triview_source(build_config(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
