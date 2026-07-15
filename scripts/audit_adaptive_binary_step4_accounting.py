#!/usr/bin/env python3
"""Run the hash-verified Step 4 root/binary/renderer feasibility preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    ABPH_HIERARCHY_GROUPINGS,
    ABPH_TARGET_CACHE_SPLITS,
    audit_target_cache_feasibility,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=list(ABPH_TARGET_CACHE_SPLITS))
    parser.add_argument("--groupings", nargs="+", default=list(ABPH_HIERARCHY_GROUPINGS))
    parser.add_argument("--max-jets-per-class", type=int, default=64)
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--report")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = audit_target_cache_feasibility(
        args.target_cache_dir,
        splits=tuple(args.splits),
        groupings=tuple(args.groupings),
        max_jets_per_class=args.max_jets_per_class,
        verify_hash=not args.no_verify_hash,
    )
    report_path = (
        Path(args.report)
        if args.report
        else Path(args.target_cache_dir) / "step4_accounting_preflight_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
