#!/usr/bin/env python3
"""Build the complete cache-backed Stage-B target-discovery factory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_target_discovery_task_specs,
    build_target_discovery_factory_config,
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
    parser.add_argument("--existing-teacher-compatible", action="store_true")
    parser.add_argument("--teacher-mix-compatible", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_target_discovery_factory_config(
        runtime_data_config=load_hashed_json(args.runtime_data_config),
        device=args.device,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        existing_teacher_compatible=args.existing_teacher_compatible,
        teacher_mix_compatible=args.teacher_mix_compatible,
    )
    if not args.dry_run:
        write_immutable_json(args.output, config)
        if args.task_specs_output:
            specs = build_target_discovery_task_specs(
                factory_config_path=args.output
            )
            destination = Path(args.task_specs_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, sort_keys=True, indent=2) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different target task specs"
                    )
            else:
                destination.write_text(
                    encoded, encoding="utf-8", newline="\n"
                )
    print(
        "supported_target_runs=36 compiled_target_screens=36 "
        f"content_hash={config['content_hash']} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
