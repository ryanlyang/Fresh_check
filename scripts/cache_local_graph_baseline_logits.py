#!/usr/bin/env python3
"""Cache frozen HLT ParT baseline logits for residual-expert training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_graph_part.residual_cache import (  # noqa: E402
    LOCAL_GRAPH_BASELINE_CHECKPOINT_VARIANT,
    LOCAL_GRAPH_BASELINE_LOGIT_CACHE_DEFAULT_METRIC_SPLITS,
    LOCAL_GRAPH_RESIDUAL_DEFAULT_SPLITS,
    LocalGraphBaselineLogitCacheConfig,
    cache_local_graph_baseline_logits,
)


def _split_words(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return LOCAL_GRAPH_RESIDUAL_DEFAULT_SPLITS
    output: list[str] = []
    for value in values:
        output.extend(str(value).split())
    return tuple(output)


def _metric_split_words(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return LOCAL_GRAPH_BASELINE_LOGIT_CACHE_DEFAULT_METRIC_SPLITS
    return _split_words(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Frozen HLT ParT baseline best_model_val.pt")
    parser.add_argument(
        "--expected-checkpoint-variant",
        default=LOCAL_GRAPH_BASELINE_CHECKPOINT_VARIANT,
        help="Expected local-graph checkpoint variant for the frozen baseline logits.",
    )
    parser.add_argument("--splits", nargs="+", default=list(LOCAL_GRAPH_RESIDUAL_DEFAULT_SPLITS))
    parser.add_argument(
        "--metric-splits",
        nargs="+",
        default=list(LOCAL_GRAPH_BASELINE_LOGIT_CACHE_DEFAULT_METRIC_SPLITS),
        help="Splits for which cache-time metrics/operating points are computed. Defaults exclude final_test.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7717)
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=0.6)
    parser.add_argument("--max-model-train-jets", type=int)
    parser.add_argument("--max-model-val-jets", type=int)
    parser.add_argument("--max-stack-train-jets", type=int)
    parser.add_argument("--max-stack-val-jets", type=int)
    parser.add_argument("--max-final-test-jets", type=int)
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--skip-hlt-params-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_jets_by_split = {
        "model_train": args.max_model_train_jets,
        "model_val": args.max_model_val_jets,
        "stack_train": args.max_stack_train_jets,
        "stack_val": args.max_stack_val_jets,
        "final_test": args.max_final_test_jets,
    }
    config = LocalGraphBaselineLogitCacheConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        checkpoint_path=args.checkpoint,
        splits=_split_words(args.splits),
        metric_splits=_metric_split_words(args.metric_splits),
        max_jets_by_split=max_jets_by_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_hlt_params=not bool(args.skip_hlt_params_check),
        expected_hlt_degradation_strength=args.expected_hlt_degradation_strength,
        expected_checkpoint_variant=args.expected_checkpoint_variant,
        overwrite=bool(args.overwrite),
    )
    report = cache_local_graph_baseline_logits(config)
    print("local_graph_baseline_logit_cache_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  checkpoint: {args.checkpoint}")
    for row in report["manifest_rows"]:
        print(
            "  split: "
            f"{row['split']} n={row['n_jets']} "
            f"fpr50={row.get('fpr_at_signal_eff_0p50')} "
            f"tau50_margin={row.get('threshold_margin_0p50')}"
        )
    print(f"  run_report: {report['outputs']['run_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
