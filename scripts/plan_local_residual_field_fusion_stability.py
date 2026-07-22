#!/usr/bin/env python3
"""Freeze the symmetric representation stability union after screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.fusion_stability import write_representation_stability_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--candidates-root", required=True)
    parser.add_argument("--prediction-sources", required=True)
    parser.add_argument("--source-artifact-audit", required=True)
    parser.add_argument("--output-path", required=True)
    report = write_representation_stability_plan(**vars(parser.parse_args(argv)))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
