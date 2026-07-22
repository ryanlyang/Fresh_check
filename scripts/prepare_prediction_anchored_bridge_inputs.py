#!/usr/bin/env python3
"""Fit stack_train_distill scalers and virtual recipes from an execution spec."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_execution import (  # noqa: E402
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_numerical import (  # noqa: E402
    prepare_bridge_inputs_from_execution_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--r0-checkpoint", required=True)
    parser.add_argument("--r0-registration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ram-root", required=True)
    parser.add_argument(
        "--allocation-id", default=os.environ.get("SLURM_JOB_ID", "local_bridge_inputs")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-capacity-bytes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--allow-unverified-test-root", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        spec = load_hashed_json(
            args.execution_spec,
            expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
        )
        result = {
            "ok": True,
            "dry_run": True,
            "execution_spec": validate_prediction_anchored_execution_spec(
                spec, verify_file_hashes=False
            ),
            "fit_child": "stack_train_distill",
            "persistent_artifacts": [
                "bridge_absolute_scaler_physical45.json",
                "bridge_quantile_reference_physical45.json",
                "bridge_recipe_all50.json",
                "bridge_recipe_physical45.json",
                "bridge_scalers_all50.json",
                "bridge_scalers_physical45.json",
                "step2_audit_metrics.json",
            ],
            "persistent_dense_fields_written": False,
        }
    else:
        result = prepare_bridge_inputs_from_execution_spec(
            args.execution_spec,
            r0_checkpoint_path=args.r0_checkpoint,
            r0_registration_path=args.r0_registration,
            output_dir=args.output_dir,
            ram_root=args.ram_root,
            allocation_id=args.allocation_id,
            batch_size=int(args.batch_size),
            shard_size=int(args.shard_size),
            device=args.device,
            capacity_bytes=(int(args.test_capacity_bytes) or None),
            allow_unverified_test_root=bool(args.allow_unverified_test_root),
        )
        result["dry_run"] = False
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
