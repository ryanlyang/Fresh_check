#!/usr/bin/env python3
"""Run all eight paired3 bridge-consumer configurations numerically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_consumer import (  # noqa: E402
    PAIRED_SEED_IDS,
    STEP3_RUN_IDS,
)
from teacher_logit_reco.local_particle_residual_field.bridge_consumer_execution import (  # noqa: E402
    run_consumer_campaign_from_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_execution import (  # noqa: E402
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--r0-checkpoint", required=True)
    parser.add_argument("--r0-registration", required=True)
    parser.add_argument("--physical45-recipe", required=True)
    parser.add_argument("--all50-recipe", required=True)
    parser.add_argument("--replica-output", required=True)
    parser.add_argument("--evaluation-output", required=True)
    parser.add_argument("--execution-report", default="")
    parser.add_argument("--ram-root", required=True)
    parser.add_argument(
        "--allocation-id", default=os.environ.get("SLURM_JOB_ID", "local_consumers")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--generation-batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
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
            "run_ids": list(STEP3_RUN_IDS),
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "replica_count": len(STEP3_RUN_IDS) * len(PAIRED_SEED_IDS),
            "full_model_val_select_evidence": True,
            "persistent_dense_fields_written": False,
        }
    else:
        result = run_consumer_campaign_from_execution_spec(
            args.execution_spec,
            r0_checkpoint_path=args.r0_checkpoint,
            r0_registration_path=args.r0_registration,
            physical45_recipe_path=args.physical45_recipe,
            all50_recipe_path=args.all50_recipe,
            replica_output_dir=args.replica_output,
            evaluation_output_dir=args.evaluation_output,
            ram_root=args.ram_root,
            allocation_id=args.allocation_id,
            device=args.device,
            shard_size=int(args.shard_size),
            generation_batch_size=int(args.generation_batch_size),
            evaluation_batch_size=int(args.evaluation_batch_size),
            bootstrap_resamples=int(args.bootstrap_resamples),
            capacity_bytes=(int(args.test_capacity_bytes) or None),
            allow_unverified_test_root=bool(args.allow_unverified_test_root),
        )
        result["dry_run"] = False
        if args.execution_report:
            report = with_content_hash(result)
            result["execution_report_publication"] = write_immutable_json(
                args.execution_report, report
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
