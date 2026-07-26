#!/usr/bin/env python3
"""Audit a completed 3M P7b replicate and write its completion marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.high_data_seed_study import (  # noqa: E402
    load_json_object,
    save_json,
    validate_high_data_p7b_run,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    completion = validate_high_data_p7b_run(
        load_json_object(args.study_manifest),
        seed=args.seed,
        output_dir=args.output_dir,
    )
    save_json(Path(args.output_dir) / "high_data_completion.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
