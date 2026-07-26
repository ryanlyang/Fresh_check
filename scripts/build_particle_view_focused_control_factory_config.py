#!/usr/bin/env python3
"""Bind focused interactions and pre-Stage-G trained controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_focused_control_factory_config,
    build_focused_control_task_specs,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distillation-factory-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-specs-output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_focused_control_factory_config(
        distillation_factory_config=load_hashed_json(
            args.distillation_factory_config
        )
    )
    output = Path(args.output).resolve()
    if not args.dry_run:
        write_immutable_json(output, config)
        if args.task_specs_output:
            specs = build_focused_control_task_specs(
                factory_config_path=output
            )
            destination = Path(args.task_specs_output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, indent=2, sort_keys=True) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different focused/control specs"
                    )
            else:
                destination.write_text(
                    encoded, encoding="utf-8", newline="\n"
                )
    print(
        json.dumps(
            {
                "content_hash": config["content_hash"],
                "focused_interaction_count": config[
                    "focused_interaction_count"
                ],
                "trained_control_count": config[
                    "trained_control_count"
                ],
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
