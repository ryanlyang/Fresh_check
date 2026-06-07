#!/usr/bin/env python3
"""Run Step 11 HLT5 frozen prediction collection and fusion."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID  # noqa: E402
from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from jetclass_fresh.hlt_control import (  # noqa: E402
    DEFAULT_HLT5_CHECKPOINT_ROOT,
    HLT5_SEEDS,
    HLT5FusionRunConfig,
    compare_hlt5_to_reco7,
    run_hlt5_fusion,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--hlt-checkpoint-root", default=DEFAULT_HLT5_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", default="checkpoints/jetclass_fresh_fusion/hlt5_seed_control")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(HLT5_SEEDS))
    parser.add_argument("--splits", nargs="+", choices=["stack_train", "stack_val", "final_test"], default=["stack_train", "stack_val", "final_test"])
    parser.add_argument("--confirm-final-test", action="store_true", help="Required when --splits includes final_test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-jets-per-split", type=int, default=None)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--C-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--feature-mode", choices=["logits", "probs", "logits_probs"], default="logits_probs")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument(
        "--compare-reco7-report",
        default=None,
        help="Optional locked reco7+HLT fusion_report.json to compare against after HLT5 fusion",
    )
    parser.add_argument(
        "--comparison-output",
        default=None,
        help="Where to save the optional HLT5-vs-reco7 comparison JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = HLT5FusionRunConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        hlt_checkpoint_root=args.hlt_checkpoint_root,
        seeds=list(args.seeds),
        splits=list(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_jets_per_split=args.max_jets_per_split,
        overwrite_predictions=args.overwrite_predictions,
        skip_existing_predictions=not args.no_skip_existing_predictions,
        confirm_final_test=args.confirm_final_test,
        C_grid=list(args.C_grid),
        feature_mode=args.feature_mode,
        max_iter=args.max_iter,
    )
    report = run_hlt5_fusion(config)
    save_json(Path(args.output_dir) / "run_report.json", report)

    if args.compare_reco7_report:
        comparison_output = args.comparison_output or str(Path(args.output_dir) / "hlt5_vs_reco7_comparison.json")
        compare_hlt5_to_reco7(
            Path(args.output_dir) / "fusion_report.json",
            args.compare_reco7_report,
            output_path=comparison_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
