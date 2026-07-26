#!/usr/bin/env python3
"""Write the locked 3M validation report or explicitly confirmed final-test report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.high_data_seed_study import (  # noqa: E402
    build_high_data_final_test_report,
    build_high_data_validation_report,
    load_json_object,
    save_json,
    write_rows_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-report", default="")
    parser.add_argument("--final-test-predictions-root", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_json_object(args.study_manifest)
    if bool(args.validation_report) != bool(args.final_test_predictions_root):
        raise ValueError(
            "--validation-report and --final-test-predictions-root must be supplied together"
        )
    if args.validation_report:
        report, rows = build_high_data_final_test_report(
            manifest,
            validation_report=load_json_object(args.validation_report),
            predictions_root=args.final_test_predictions_root,
        )
        csv_name = "final_test_seed_metrics.csv"
    else:
        report, rows = build_high_data_validation_report(manifest)
        csv_name = "validation_seed_metrics.csv"
    output_dir = Path(args.output_dir)
    save_json(output_dir / "run_report.json", report)
    write_rows_csv(output_dir / csv_name, rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
