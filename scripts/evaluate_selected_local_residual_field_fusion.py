#!/usr/bin/env python3
"""Evaluate final_test using only an immutable selected_fusion.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import evaluate_selected_fusion_final  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-fusion", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = evaluate_selected_fusion_final(parse_args(argv).selected_fusion)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

