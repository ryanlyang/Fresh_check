#!/usr/bin/env python3
"""Build the authenticated PV10 HLT-only final-test factory config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_final_factory_config,
    build_final_task_specs,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-factory-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-specs-output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_final_factory_config(
        report_factory_config=load_hashed_json(
            args.report_factory_config
        ),
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    output = Path(args.output).resolve()
    if not args.dry_run:
        write_immutable_json(output, config)
        if args.task_specs_output:
            specs = build_final_task_specs(factory_config_path=output)
            destination = Path(args.task_specs_output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, indent=2, sort_keys=True) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different PV10 task specs"
                    )
            else:
                destination.write_text(
                    encoded, encoding="utf-8", newline="\n"
                )
    print(
        json.dumps(
            {
                "content_hash": config["content_hash"],
                "final_test_run_count": 2,
                "hlt_only": True,
                "offline_cache_forbidden": True,
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
