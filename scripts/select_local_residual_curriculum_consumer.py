#!/usr/bin/env python3
"""Select the Stage 1b oracle consumer from both Stage 1a alpha diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.curriculum_campaign import (  # noqa: E402
    select_curriculum_consumer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofull-report", required=True)
    parser.add_argument("--orobust-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.002)
    parser.add_argument("--close-accuracy-tolerance", type=float, default=0.001)
    parser.add_argument("--drop-tolerance", type=float, default=0.002)
    parser.add_argument("--stack-brittleness-tolerance", type=float, default=0.003)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = select_curriculum_consumer(
        (args.ofull_report, args.orobust_report),
        output_path=args.output,
        minimum_gain=args.minimum_gain,
        close_accuracy_tolerance=args.close_accuracy_tolerance,
        drop_tolerance=args.drop_tolerance,
        stack_brittleness_tolerance=args.stack_brittleness_tolerance,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
