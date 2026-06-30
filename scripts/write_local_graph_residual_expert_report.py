#!/usr/bin/env python3
"""Write Step 9 local-graph residual expert comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_graph_part.residual_report import (  # noqa: E402
    LocalGraphResidualExpertReportConfig,
    build_local_graph_residual_expert_report,
)
from teacher_logit_reco.local_graph_part.train import LOCAL_GRAPH_SELECTION_METRICS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--baseline-logit-cache-dir", required=True)
    parser.add_argument("--residual-expert-root", required=True)
    parser.add_argument("--residual-variants", nargs="*", default=())
    parser.add_argument("--standalone-tagger-root", default=None)
    parser.add_argument("--standalone-variants", nargs="*", default=())
    parser.add_argument("--score-fusion-report-path", default=None)
    parser.add_argument("--primary-metric", choices=LOCAL_GRAPH_SELECTION_METRICS, default="fpr_at_signal_eff_0p50")
    parser.add_argument("--comparison-split", choices=("model_val", "stack_val", "final_test"), default="final_test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=9713)
    parser.add_argument("--max-model-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--skip-checkpoint-evaluation", action="store_true")
    parser.add_argument(
        "--allow-precomputed-evaluations",
        action="store_true",
        help=(
            "Permit --skip-checkpoint-evaluation only for trusted run_report payloads carrying the "
            "local_graph_residual_precomputed_eval_v2 contract."
        ),
    )
    parser.add_argument("--allow-missing-residual-variants", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--skip-hlt-params-check", action="store_true")
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=0.6)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> LocalGraphResidualExpertReportConfig:
    return LocalGraphResidualExpertReportConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        baseline_logit_cache_dir=args.baseline_logit_cache_dir,
        residual_expert_root=args.residual_expert_root,
        residual_variants=tuple(args.residual_variants),
        standalone_tagger_root=args.standalone_tagger_root,
        standalone_variants=tuple(args.standalone_variants),
        score_fusion_report_path=args.score_fusion_report_path,
        primary_metric=args.primary_metric,
        comparison_split=args.comparison_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=bool(args.amp),
        seed=args.seed,
        max_model_val_jets=args.max_model_val_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        confirm_final_test=bool(args.confirm_final_test),
        evaluate_checkpoints=not bool(args.skip_checkpoint_evaluation),
        allow_precomputed_evaluations=bool(args.allow_precomputed_evaluations),
        require_all_residual_variants=not bool(args.allow_missing_residual_variants),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_hlt_params=not bool(args.skip_hlt_params_check),
        expected_hlt_degradation_strength=args.expected_hlt_degradation_strength,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_local_graph_residual_expert_report(build_config(args))
    summary = report["comparison_summary"]
    print("local_graph_residual_expert_report_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  comparison_split: {summary['comparison_split']}")
    print(f"  primary_metric: {summary['primary_metric']}")
    print(f"  primary_metric_direction: {summary['primary_metric_direction']}")
    print(f"  baseline_metric_value: {summary['baseline_metric_value']}")
    print(f"  best_source_type: {summary['best_source_type']}")
    print(f"  best_variant: {summary['best_variant']}")
    print(f"  best_metric_value: {summary['best_metric_value']}")
    print(f"  report_json: {report['outputs']['report_json']}")
    print(f"  metric_table_csv: {report['outputs']['metric_table_csv']}")
    if report["problems"]:
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
