#!/usr/bin/env python3
"""Cache model-val/final-test predictions for one trained HLT-MV source model."""

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
    HLT_MV_SOURCE_DEFAULT_EVAL_BATCH_SIZE,
    HLT_MV_SOURCE_DEFAULT_MAX_FINAL_TEST_JETS,
    HLT_MV_SOURCE_DEFAULT_MAX_VAL_JETS,
    HLT_MV_SOURCE_PREDICTION_SPLITS,
    build_hlt_mv_source_config,
    cache_hlt_mv_source_predictions,
    normalize_hlt_mv_source_prediction_splits,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--pdv3-experiment-name", default=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--source-view", choices=["fixed_hlt", "hlt2"], default=None)
    parser.add_argument("--splits", nargs="+", default=list(HLT_MV_SOURCE_PREDICTION_SPLITS), choices=list(HLT_MV_SOURCE_PREDICTION_SPLITS))
    parser.add_argument("--eval-batch-size", type=int, default=HLT_MV_SOURCE_DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=HLT_MV_SOURCE_DEFAULT_MAX_VAL_JETS)
    parser.add_argument("--max-final-test-jets", type=int, default=HLT_MV_SOURCE_DEFAULT_MAX_FINAL_TEST_JETS)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    splits = normalize_hlt_mv_source_prediction_splits(args.splits)
    config = build_hlt_mv_source_config(
        source_name=args.source_name,
        output_root=args.output_root,
        pdv3_experiment_name=args.pdv3_experiment_name,
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        cache_dir=None if args.cache_dir is None else Path(args.cache_dir),
        source_view=args.source_view,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=False,
        evaluate_model_val_predictions="model_val" in splits,
        evaluate_final_test="final_test" in splits,
        confirm_final_test=bool(args.confirm_final_test),
        max_val_batches=args.max_val_batches,
        max_final_test_batches=args.max_final_test_batches,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        overwrite=bool(args.overwrite),
    )
    report = cache_hlt_mv_source_predictions(config, splits=splits)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
