#!/usr/bin/env python3
"""Build the authenticated PV08 sealed-stack/fusion factory config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_stack_factory_config,
    build_stack_task_specs,
    load_hashed_json,
    write_immutable_json,
)


def _optional_p7b(path: str | None):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("optional P7b resource must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairness-factory-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-specs-output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-stack-batches", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--linear-fusion-steps", type=int, default=300)
    parser.add_argument("--optional-p7b-resource")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_stack_factory_config(
        fairness_factory_config=load_hashed_json(
            args.fairness_factory_config
        ),
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_stack_batches=args.max_stack_batches,
        bootstrap_replicates=args.bootstrap_replicates,
        linear_fusion_steps=args.linear_fusion_steps,
        optional_p7b_resource=_optional_p7b(args.optional_p7b_resource),
    )
    output = Path(args.output).resolve()
    if not args.dry_run:
        write_immutable_json(output, config)
        if args.task_specs_output:
            specs = build_stack_task_specs(factory_config_path=output)
            destination = Path(args.task_specs_output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(specs, indent=2, sort_keys=True) + "\n"
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise FileExistsError(
                        "refusing to overwrite different stack task specs"
                    )
            else:
                destination.write_text(
                    encoded, encoding="utf-8", newline="\n"
                )
    print(
        json.dumps(
            {
                "content_hash": config["content_hash"],
                "stack_run_count": 19,
                "optional_p7b_configured": (
                    config["optional_p7b_resource"] is not None
                ),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
