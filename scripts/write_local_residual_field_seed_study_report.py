#!/usr/bin/env python3
"""Aggregate five matched A0/P7b seeds without opening final_test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.seed_study import (  # noqa: E402
    build_seed_study_report,
    load_json_object,
    save_json,
    write_rows_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, rows = build_seed_study_report(load_json_object(args.study_manifest))
    output_dir = Path(args.output_dir)
    save_json(output_dir / "run_report.json", report)
    write_rows_csv(output_dir / "seed_metrics.csv", rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
