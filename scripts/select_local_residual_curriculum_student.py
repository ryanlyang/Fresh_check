#!/usr/bin/env python3
"""Select the best deployable first-stage P model for G0 fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.curriculum_campaign import (  # noqa: E402
    select_best_curriculum_student,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--close-accuracy-tolerance", type=float, default=0.001)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = select_best_curriculum_student(
        args.report,
        output_path=args.output,
        close_accuracy_tolerance=args.close_accuracy_tolerance,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
