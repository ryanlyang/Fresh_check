#!/usr/bin/env python3
"""Build cache-backed Stage-A teacher factory configuration and task specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_baseline_factory_config,
    build_stage_a_teacher_task_specs,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-data-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-specs-output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--existing-checkpoint")
    parser.add_argument("--existing-observed-train-identity-sha256")
    parser.add_argument("--existing-serialized-recipe")
    parser.add_argument("--existing-recipe-reproduced-exactly", action="store_true")
    parser.add_argument("--existing-provenance-metadata-sha256")
    parser.add_argument(
        "--existing-description",
        default="pre-existing offline particle teacher",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_baseline_factory_config(
        runtime_data_config=load_hashed_json(args.runtime_data_config),
        device=args.device,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        existing_checkpoint_path=args.existing_checkpoint,
        existing_observed_train_identity_sha256=(
            args.existing_observed_train_identity_sha256
        ),
        existing_serialized_recipe_path=args.existing_serialized_recipe,
        existing_recipe_reproduced_exactly=(
            args.existing_recipe_reproduced_exactly
        ),
        existing_provenance_metadata_sha256=(
            args.existing_provenance_metadata_sha256
        ),
        existing_description=args.existing_description,
    )
    if not args.dry_run:
        write_immutable_json(args.output, config)
        if args.task_specs_output:
            specs = build_stage_a_teacher_task_specs(
                factory_config_path=args.output
            )
            destination = Path(args.task_specs_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, sort_keys=True, indent=2) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different teacher task specs"
                    )
            else:
                destination.write_text(encoded, encoding="utf-8", newline="\n")
    print(
        f"trained_teacher_roles=3 existing_configured="
        f"{config['existing_teacher'] is not None} "
        f"content_hash={config['content_hash']} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
