#!/usr/bin/env python3
"""Profile, select, and configure the two Stage-A direct HLT controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_direct_control_factory_config,
    build_stage_a_direct_resource_plan,
    build_stage_a_direct_task_specs,
    load_hashed_json,
    validate_stage_a_direct_resource_plan,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-data-config", required=True)
    parser.add_argument("--resource-plan")
    parser.add_argument("--resource-plan-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-specs-output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    resource_plan = (
        load_hashed_json(args.resource_plan)
        if args.resource_plan
        else build_stage_a_direct_resource_plan()
    )
    audit = validate_stage_a_direct_resource_plan(resource_plan)
    config = build_direct_control_factory_config(
        runtime_data_config=load_hashed_json(args.runtime_data_config),
        resource_plan=resource_plan,
        device=args.device,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    if not args.dry_run:
        write_immutable_json(args.resource_plan_output, resource_plan)
        write_immutable_json(args.output, config)
        if args.task_specs_output:
            specs = build_stage_a_direct_task_specs(
                factory_config_path=args.output
            )
            destination = Path(args.task_specs_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, sort_keys=True, indent=2) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different direct task specs"
                    )
            else:
                destination.write_text(encoded, encoding="utf-8", newline="\n")
    parameter = resource_plan["selections"]["parameters"]
    flops = resource_plan["selections"]["flops"]
    print(
        f"candidates={audit['candidate_count']} "
        f"parameter_match={audit['parameter_config_id']} "
        f"parameter_warning={parameter['quality_warning']} "
        f"flop_match={audit['flop_config_id']} "
        f"flop_warning={flops['quality_warning']} "
        f"content_hash={config['content_hash']} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
