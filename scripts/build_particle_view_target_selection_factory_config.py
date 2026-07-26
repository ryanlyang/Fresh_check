#!/usr/bin/env python3
"""Build Stage-B target-selection factory configuration and task specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_target_selection_factory_config,
    build_target_selection_task_specs,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-specs-output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_target_selection_factory_config(
        source_commit=args.source_commit
    )
    if not args.dry_run:
        write_immutable_json(args.output, config)
        if args.task_specs_output:
            specs = build_target_selection_task_specs(
                factory_config_path=args.output
            )
            destination = Path(args.task_specs_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, sort_keys=True, indent=2) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different selection task specs"
                    )
            else:
                destination.write_text(
                    encoded, encoding="utf-8", newline="\n"
                )
    print(
        "candidate_target_runs=36 forward_count=2 "
        f"content_hash={config['content_hash']} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
