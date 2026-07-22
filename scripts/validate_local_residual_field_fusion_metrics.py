#!/usr/bin/env python3
"""Reproduce the exact known A0/P7b exploratory-final metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    audit_local_residual_field_pilot_metric_reproduction,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-exploratory-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = audit_local_residual_field_pilot_metric_reproduction(
        args.source_artifact_audit,
        output_path=args.output,
        confirm_exploratory_final_test=bool(args.confirm_exploratory_final_test),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
