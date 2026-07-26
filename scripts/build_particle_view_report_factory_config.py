#!/usr/bin/env python3
"""Build the authenticated PV09 report/export/reload factory config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_report_factory_config,
    build_report_task_specs,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-factory-config", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-specs-output")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--reload-fixture-batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_report_factory_config(
        stack_factory_config=load_hashed_json(args.stack_factory_config),
        source_commit=args.source_commit,
        python_executable=args.python_executable,
        reload_fixture_batch_size=args.reload_fixture_batch_size,
    )
    output = Path(args.output).resolve()
    if not args.dry_run:
        write_immutable_json(output, config)
        if args.task_specs_output:
            specs = build_report_task_specs(factory_config_path=output)
            destination = Path(args.task_specs_output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, indent=2, sort_keys=True) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different PV09 task specs"
                    )
            else:
                destination.write_text(
                    encoded, encoding="utf-8", newline="\n"
                )
    print(
        json.dumps(
            {
                "content_hash": config["content_hash"],
                "report_export_run_count": 7,
                "fresh_process_reload_required": True,
                "final_test_loaded": False,
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
