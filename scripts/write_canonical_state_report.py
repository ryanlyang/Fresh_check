#!/usr/bin/env python3
"""Write the Canonical Multi-Scale Jet State Step 9 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.canonical_state import (  # noqa: E402
    CANONICAL_STATE_EXPECTED_RUN_IDS,
    CanonicalStateReportConfig,
    build_canonical_state_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-ids", nargs="+", default=list(CANONICAL_STATE_EXPECTED_RUN_IDS))
    parser.add_argument("--baseline-run-id", default="A0")
    parser.add_argument("--allow-missing-runs", action="store_true")
    parser.add_argument("--no-require-all-runs", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=args.output_dir,
            run_root=args.run_root,
            run_ids=tuple(args.run_ids),
            baseline_run_id=args.baseline_run_id,
            require_all_runs=not bool(args.no_require_all_runs),
            allow_missing_runs=bool(args.allow_missing_runs),
            confirm_final_test=bool(args.confirm_final_test),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
