#!/usr/bin/env python3
"""Publish the strictly separated particle-view campaign report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.particle_view.reporting import (  # noqa: E402
    build_separated_campaign_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--stack-report-sha256", required=True)
    parser.add_argument("--fairness-ledger-sha256", required=True)
    parser.add_argument("--label-exposure-ledger-sha256", required=True)
    parser.add_argument("--storage-reservation-sha256", required=True)
    parser.add_argument("--lineage-graph-sha256", required=True)
    parser.add_argument("--deployment-export-sha256", action="append", required=True)
    parser.add_argument("--warning-summary-sha256", required=True)
    parser.add_argument("--final-test-permit-sha256")
    parser.add_argument("--final-test-result-sha256", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sections = json.loads(args.sections.read_text(encoding="utf-8"))
    report = build_separated_campaign_report(
        sections=sections,
        selection_sha256=args.selection_sha256,
        stack_report_sha256=args.stack_report_sha256,
        fairness_ledger_sha256=args.fairness_ledger_sha256,
        label_exposure_ledger_sha256=args.label_exposure_ledger_sha256,
        storage_reservation_sha256=args.storage_reservation_sha256,
        lineage_graph_sha256=args.lineage_graph_sha256,
        deployment_export_sha256=args.deployment_export_sha256,
        aggregate_warning_summary_sha256=args.warning_summary_sha256,
        final_test_permit_sha256=args.final_test_permit_sha256,
        final_test_result_sha256=args.final_test_result_sha256,
    )
    write_immutable_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
