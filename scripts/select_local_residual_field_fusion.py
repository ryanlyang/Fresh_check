#!/usr/bin/env python3
"""Freeze matched accuracy/rejection champions before final-test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import FusionSelectionConfig, select_fusion_champions  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--candidates-root", required=True)
    parser.add_argument("--prediction-sources", required=True)
    parser.add_argument("--source-artifact-audit", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--pristine-confirmation-manifest")
    parser.add_argument("--overwrite", action="store_true", help="Debug only; production selection is immutable.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = select_fusion_champions(FusionSelectionConfig(**vars(parse_args(argv))))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

