#!/usr/bin/env python3
"""Bind a frozen Step-2 teacher registration to one contextual token tap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    ParticleTokenTapSpec,
    build_token_tap_registration,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-registration", required=True)
    parser.add_argument(
        "--particle-source", choices=("fixed_hlt", "offline"), required=True
    )
    parser.add_argument("--architecture", choices=("base", "large"), required=True)
    parser.add_argument(
        "--tap-choice",
        choices=("raw_embed", "middle", "penultimate", "final", "mix_last3"),
        required=True,
    )
    parser.add_argument("--input-normalization-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    teacher = load_hashed_json(args.teacher_registration)
    artifact = build_token_tap_registration(
        teacher_registration=teacher,
        tap_spec=ParticleTokenTapSpec(
            particle_source=args.particle_source,
            architecture=args.architecture,
            tap_choice=args.tap_choice,
        ),
        input_normalization_sha256=args.input_normalization_sha256,
    )
    receipt = write_immutable_json(args.output, artifact)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
