#!/usr/bin/env python3
"""Bind every low-data run to its concrete cache-backed scientific factory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_scientific_handler_commands,
    build_scientific_task_catalog,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--task-specs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--handler-commands-output")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    registry = load_hashed_json(args.registry)
    task_specs = json.loads(
        Path(args.task_specs).read_text(encoding="utf-8")
    )
    if not isinstance(task_specs, dict):
        raise ValueError("task specs must be a JSON object keyed by run_id")
    catalog = build_scientific_task_catalog(
        registry=registry,
        task_specs=task_specs,
    )
    if not args.dry_run:
        write_immutable_json(args.output, catalog)
        if args.handler_commands_output:
            commands = build_scientific_handler_commands(
                catalog=catalog,
                catalog_path=args.output,
                python_executable=args.python_executable,
            )
            destination = Path(args.handler_commands_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(commands, sort_keys=True, indent=2) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite scientific handler commands"
                    )
            else:
                destination.write_text(
                    encoded,
                    encoding="utf-8",
                    newline="\n",
                )
    print(
        f"runs={catalog['run_count']} "
        f"content_hash={catalog['content_hash']} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
