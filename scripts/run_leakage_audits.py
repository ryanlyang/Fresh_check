#!/usr/bin/env python3
"""Run Step 12 leakage and stacker sanity audits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.audits import AuditRunConfig, run_audit_suite  # noqa: E402
from jetclass_fresh.fusion import DEFAULT_C_GRID  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Step 2 split manifest path")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--fusion-dir", default="checkpoints/jetclass_fresh_fusion/reco7_plus_hlt")
    parser.add_argument("--prediction-dir", default=None, help="Defaults to --fusion-dir/predictions")
    parser.add_argument("--fusion-report", default=None, help="Defaults to --fusion-dir/fusion_report.json")
    parser.add_argument("--output-dir", default="checkpoints/jetclass_fresh_audits/reco7_plus_hlt")
    parser.add_argument("--model-names", nargs="+", default=None, help="Defaults to model_names from the fusion report")
    parser.add_argument("--splits", nargs="+", choices=["stack_train", "stack_val", "final_test"], default=["stack_train", "stack_val", "final_test"])
    parser.add_argument("--allow-file-overlap", action="store_true", help="Use only the jet-identity audit as the split hard check")
    parser.add_argument("--verify-hlt-cache-arrays", action="store_true", help="Also load full HLT cache arrays and verify hashes")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--C-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--feature-mode", choices=["logits", "probs", "logits_probs"], default="logits_probs")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--permutation-accuracy-slack", type=float, default=0.05)
    parser.add_argument("--holdout-max-accuracy-gap", type=float, default=0.10)
    parser.add_argument("--block-shuffle-model", default=None)
    parser.add_argument("--fail-on-audit-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fusion_dir = Path(args.fusion_dir)
    prediction_dir = args.prediction_dir or str(fusion_dir / "predictions")
    fusion_report = args.fusion_report or str(fusion_dir / "fusion_report.json")
    config = AuditRunConfig(
        manifest_path=args.manifest,
        prediction_dir=prediction_dir,
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        fusion_report_path=fusion_report,
        model_names=[] if args.model_names is None else list(args.model_names),
        splits=list(args.splits),
        require_file_disjoint=not args.allow_file_overlap,
        verify_hlt_cache_arrays=args.verify_hlt_cache_arrays,
        seed=args.seed,
        C_grid=list(args.C_grid),
        feature_mode=args.feature_mode,
        max_iter=args.max_iter,
        permutation_accuracy_slack=args.permutation_accuracy_slack,
        holdout_max_accuracy_gap=args.holdout_max_accuracy_gap,
        block_shuffle_model=args.block_shuffle_model,
    )
    report = run_audit_suite(config)
    print(f"wrote {Path(args.output_dir) / 'audit_report.json'}")
    print(f"ok={report['ok']}")
    if args.fail_on_audit_failure and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
