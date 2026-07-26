#!/usr/bin/env python3
"""Bind selected-view, clean-consumer, and robust-chain production tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_post_target_factory_config,
    build_post_target_task_specs,
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_post_target_factory_config(
        runtime_data_config=load_hashed_json(args.runtime_data_config),
        device=args.device,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    output = Path(args.output).resolve()
    if not args.dry_run:
        write_immutable_json(output, config)
        if args.task_specs_output:
            specs = build_post_target_task_specs(
                factory_config_path=output
            )
            destination = Path(args.task_specs_output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, indent=2, sort_keys=True) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different post-target specs"
                    )
            else:
                destination.write_text(
                    encoded, encoding="utf-8", newline="\n"
                )
    print(
        json.dumps(
            {
                "content_hash": config["content_hash"],
                "run_count": 6,
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
