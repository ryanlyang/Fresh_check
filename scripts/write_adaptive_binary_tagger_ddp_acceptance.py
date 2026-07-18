#!/usr/bin/env python3
"""Compile the immutable E7/F0 tagger DDP parity and speed gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.tagger_distributed import (  # noqa: E402
    build_tagger_ddp_acceptance,
)


_VARIANTS = (
    "E7_dual_hierarchy_dualcross",
    "F0_ce_reco_primary",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-root", required=True)
    parser.add_argument("--ddp4-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    single_root = Path(args.single_root)
    ddp4_root = Path(args.ddp4_root)
    report = build_tagger_ddp_acceptance(
        single_reports={
            name: single_root / "runs" / name / "run_report.json"
            for name in _VARIANTS
        },
        ddp4_reports={
            name: ddp4_root / "runs" / name / "run_report.json"
            for name in _VARIANTS
        },
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
