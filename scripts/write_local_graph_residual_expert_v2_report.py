#!/usr/bin/env python3
"""Write the Step 12 local-graph residual expert V2 final report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_graph_part.residual_v2_report import (  # noqa: E402
    LocalGraphResidualExpertV2ReportConfig,
    build_local_graph_residual_expert_v2_report,
)


def _optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--baseline-embedding-cache-dir", required=True)
    parser.add_argument("--residual-expert-root", required=True)
    parser.add_argument("--residual-variants", nargs="*", default=())
    parser.add_argument("--v1-residual-report-path")
    parser.add_argument("--score-fusion-report-path")
    parser.add_argument("--standalone-report-path")
    parser.add_argument("--primary-metric", default="fpr_at_signal_eff_0p50")
    parser.add_argument("--comparison-split", choices=("model_val", "stack_val", "final_test"), default="final_test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=9821)
    parser.add_argument("--max-model-val-jets", type=_optional_int)
    parser.add_argument("--max-stack-train-jets", type=_optional_int)
    parser.add_argument("--max-stack-val-jets", type=_optional_int)
    parser.add_argument("--max-final-test-jets", type=_optional_int)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--disable-calibration-control", action="store_true")
    parser.add_argument("--allow-missing-residual-variants", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--skip-hlt-param-check", action="store_true")
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=0.6)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> LocalGraphResidualExpertV2ReportConfig:
    return LocalGraphResidualExpertV2ReportConfig(
        output_dir=str(args.output_dir),
        hlt_cache_dir=str(args.hlt_cache_dir),
        baseline_embedding_cache_dir=str(args.baseline_embedding_cache_dir),
        residual_expert_root=str(args.residual_expert_root),
        residual_variants=tuple(str(item) for item in args.residual_variants),
        v1_residual_report_path=args.v1_residual_report_path,
        score_fusion_report_path=args.score_fusion_report_path,
        standalone_report_path=args.standalone_report_path,
        primary_metric=str(args.primary_metric),
        comparison_split=str(args.comparison_split),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        amp=bool(args.amp),
        seed=int(args.seed),
        max_model_val_jets=args.max_model_val_jets,
        max_stack_train_jets=args.max_stack_train_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        confirm_final_test=bool(args.confirm_final_test),
        include_calibration_control=not bool(args.disable_calibration_control),
        require_all_residual_variants=not bool(args.allow_missing_residual_variants),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_hlt_params=not bool(args.skip_hlt_param_check),
        expected_hlt_degradation_strength=float(args.expected_hlt_degradation_strength),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    report = build_local_graph_residual_expert_v2_report(config)
    summary = report.get("comparison_summary", {})
    print("local_graph_residual_expert_v2_report_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {config.output_dir}")
    print(f"  comparison_split: {summary.get('comparison_split')}")
    print(f"  primary_metric: {summary.get('primary_metric')}")
    print(f"  baseline: {summary.get('baseline_metric_value')}")
    print(f"  best_source_type: {summary.get('best_source_type')}")
    print(f"  best_variant: {summary.get('best_variant')}")
    print(f"  best_metric_value: {summary.get('best_metric_value')}")
    print(f"  report_json: {report['outputs']['report_json']}")
    print(f"  metric_table_csv: {report['outputs']['metric_table_csv']}")
    if report["problems"]:
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
