#!/usr/bin/env python3
"""Publish actual R0/selected-T10 resource provenance after B4 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    publish_confirmed_runtime_resource_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representative-reference", required=True)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--r0-checkpoint", required=True)
    parser.add_argument("--r0-registration", required=True)
    parser.add_argument("--selected-consumer", required=True)
    parser.add_argument("--physical45-recipe", required=True)
    parser.add_argument("--physical45-scaler", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = publish_confirmed_runtime_resource_reference(
        representative_reference_path=args.representative_reference,
        execution_spec_path=args.execution_spec,
        r0_checkpoint_path=args.r0_checkpoint,
        r0_registration_path=args.r0_registration,
        selected_consumer_path=args.selected_consumer,
        physical45_recipe_path=args.physical45_recipe,
        physical45_scaler_path=args.physical45_scaler,
        output_path=args.output,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(Path(args.output).resolve()),
                "content_hash": artifact["content_hash"],
                "r0_checkpoint_sha256": artifact["r0_checkpoint_sha256"],
                "t10_checkpoint_sha256": artifact["t10_checkpoint_sha256"],
                "physical45_scaler_sha256": artifact["physical45_scaler_sha256"],
                "execution_spec_sha256": artifact["execution_spec_sha256"],
                "selected_consumer_sha256": artifact["selected_consumer_sha256"],
                "physical45_recipe_sha256": artifact["physical45_recipe_sha256"],
                "resource_values_identical_to_representative": artifact[
                    "resource_values_identical_to_representative"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
