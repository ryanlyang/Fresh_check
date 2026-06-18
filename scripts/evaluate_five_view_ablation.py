#!/usr/bin/env python3
"""Evaluate Step 11 five-view baseline and ablation checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.set_matching.five_view_ablation import (  # noqa: E402
    FiveViewAblationEvalConfig,
    evaluate_five_view_ablation_suite,
    parse_ablation_checkpoint_spec,
)
from teacher_logit_reco.set_matching.five_view_data import FIVE_VIEW_SELECTION_MODES  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402


def label_names_to_indices(values: list[str]) -> tuple[int, ...]:
    if not values:
        return ()
    by_name = {name: index for index, name in enumerate(LABEL_NAMES)}
    output: list[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            output.append(int(text))
            continue
        if text not in by_name:
            raise ValueError(f"Unknown JetClass label {text!r}; expected one of {list(LABEL_NAMES)}")
        output.append(by_name[text])
    return tuple(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--tagger-root", default=None)
    parser.add_argument("--reconstructed-view-dir", default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Extra checkpoint spec as name=path. If no canonical checkpoints exist, explicit specs are enough.",
    )
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--no-canonical", action="store_true")
    parser.add_argument("--require-all-canonical", action="store_true")

    parser.add_argument("--val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--label-filter-names", nargs="*", default=())

    parser.add_argument("--max-tokens-per-view", type=int, default=128)
    parser.add_argument("--min-tokens-per-view", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--selection-mode", choices=tuple(FIVE_VIEW_SELECTION_MODES), default="topk_or_threshold")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--seed", type=int, default=1205)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = FiveViewAblationEvalConfig(
        output_dir=args.output_dir,
        experiment_dir=args.experiment_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        tagger_root=args.tagger_root,
        reconstructed_view_dir=args.reconstructed_view_dir,
        checkpoint_specs=tuple(parse_ablation_checkpoint_spec(value) for value in args.checkpoint),
        only=tuple(args.only),
        require_all_canonical=args.require_all_canonical,
        include_canonical=not bool(args.no_canonical),
        val_split=args.val_split,
        final_test_split=args.final_test_split,
        confirm_final_test=args.confirm_final_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        max_val_batches=args.max_val_batches,
        max_final_test_batches=args.max_final_test_batches,
        label_filter=label_names_to_indices(list(args.label_filter_names)),
        max_tokens_per_view=args.max_tokens_per_view,
        min_tokens_per_view=args.min_tokens_per_view,
        confidence_threshold=args.confidence_threshold,
        selection_mode=args.selection_mode,
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        seed=args.seed,
    )
    report = evaluate_five_view_ablation_suite(config)
    print("set_matching_five_view_ablation_eval_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  evaluated_ablations: {len(report['evaluated_ablations'])}")
    print(f"  skipped: {len(report['skipped'])}")
    print(f"  final_test_evaluated: {report['final_test_evaluated']}")
    print(f"  summary_csv: {report['summary_csv']}")
    print(f"  per_class_metrics_csv: {report['per_class_metrics_csv']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
