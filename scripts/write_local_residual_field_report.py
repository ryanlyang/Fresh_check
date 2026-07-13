#!/usr/bin/env python3
"""Write the final local particle residual-field campaign report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    DEFAULT_REQUIRED_FUSION_GROUPS,
    DEFAULT_REQUIRED_RECONSTRUCTOR_RUN_IDS,
    DEFAULT_REQUIRED_TAGGER_RUN_IDS,
    LocalResidualFieldReportConfig,
    build_local_residual_field_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tagger-root", required=True)
    parser.add_argument("--reconstructor-root", required=True)
    parser.add_argument("--fusion-dir", default="")
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--target-cache-dir", default="")
    parser.add_argument(
        "--required-tagger-run-ids",
        nargs="+",
        default=list(DEFAULT_REQUIRED_TAGGER_RUN_IDS),
        help="Tagger run IDs that must exist and have ok=true.",
    )
    parser.add_argument(
        "--required-reconstructor-run-ids",
        nargs="+",
        default=list(DEFAULT_REQUIRED_RECONSTRUCTOR_RUN_IDS),
        help="Reconstructor run IDs that must exist and have ok=true.",
    )
    parser.add_argument(
        "--required-fusion-groups",
        nargs="+",
        default=list(DEFAULT_REQUIRED_FUSION_GROUPS),
        help="Fusion groups that must appear in fusion_metrics.csv when --require-fusion is set.",
    )
    parser.add_argument("--require-fusion", action="store_true")
    parser.add_argument("--allow-missing-runs", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--require-final-test-provenance", action="store_true")
    parser.add_argument("--summary-title", default="Local Particle Residual Field Report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_local_residual_field_report(
        LocalResidualFieldReportConfig(
            output_dir=args.output_dir,
            tagger_root=args.tagger_root,
            reconstructor_root=args.reconstructor_root,
            fusion_dir=args.fusion_dir or None,
            prediction_dir=args.prediction_dir or None,
            target_cache_dir=args.target_cache_dir or None,
            required_tagger_run_ids=tuple(args.required_tagger_run_ids),
            required_reconstructor_run_ids=tuple(args.required_reconstructor_run_ids),
            required_fusion_groups=tuple(args.required_fusion_groups),
            require_fusion=bool(args.require_fusion),
            allow_missing_runs=bool(args.allow_missing_runs),
            confirm_final_test=bool(args.confirm_final_test),
            require_final_test_provenance=bool(args.require_final_test_provenance),
            summary_title=str(args.summary_title),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
