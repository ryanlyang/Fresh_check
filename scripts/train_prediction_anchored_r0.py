#!/usr/bin/env python3
"""Train the production ordinary R0 from a bound execution specification."""

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
    run_streamed_r0_from_execution_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ram-root", required=True)
    parser.add_argument(
        "--allocation-id", default=os.environ.get("SLURM_JOB_ID", "local_r0")
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--device", default="")
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
        audit = validate_prediction_anchored_execution_spec(
            spec, verify_file_hashes=False
        )
        result = {
            "ok": True,
            "dry_run": True,
            "execution_spec": audit,
            "output_dir": str(args.output_dir),
            "source_splits": ["model_train", "model_val"],
            "train_child": "model_train",
            "checkpoint_child": "model_val_stop",
            "persistent_artifacts": [
                "r0_weights.pt",
                "r0_registration.json",
                "r0_metrics.json",
            ],
            "persistent_dense_fields_written": False,
        }
    else:
        result = run_streamed_r0_from_execution_spec(
            args.execution_spec,
            output_dir=args.output_dir,
            ram_root=args.ram_root,
            allocation_id=args.allocation_id,
            batch_size=int(args.batch_size),
            shard_size=int(args.shard_size),
            device=args.device or None,
            capacity_bytes=(int(args.test_capacity_bytes) or None),
            allow_unverified_test_root=bool(args.allow_unverified_test_root),
        )
        result["dry_run"] = False
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
