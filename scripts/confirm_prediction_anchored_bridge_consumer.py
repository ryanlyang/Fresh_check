#!/usr/bin/env python3
"""Run the sealed stack_val_consumer confirmation numerically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_consumer_execution import (  # noqa: E402
    confirm_selected_consumer_from_execution_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--preconfirmation", required=True)
    parser.add_argument("--r0-checkpoint", required=True)
    parser.add_argument("--r0-registration", required=True)
    parser.add_argument("--physical45-recipe", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ram-root", required=True)
    parser.add_argument(
        "--allocation-id", default=os.environ.get("SLURM_JOB_ID", "local_consumer_confirm")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--test-capacity-bytes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--allow-unverified-test-root", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = confirm_selected_consumer_from_execution_spec(
        args.execution_spec,
        preconfirmation_path=args.preconfirmation,
        r0_checkpoint_path=args.r0_checkpoint,
        r0_registration_path=args.r0_registration,
        physical45_recipe_path=args.physical45_recipe,
        output_dir=args.output_dir,
        ram_root=args.ram_root,
        allocation_id=args.allocation_id,
        device=args.device,
        batch_size=int(args.batch_size),
        shard_size=int(args.shard_size),
        capacity_bytes=(int(args.test_capacity_bytes) or None),
        allow_unverified_test_root=bool(args.allow_unverified_test_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
