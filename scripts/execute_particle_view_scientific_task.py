#!/usr/bin/env python3
"""Execute one locked scientific operation selected by runtime run identity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    execute_scientific_task,
    load_hashed_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--run-id",
        default=os.environ.get("PARTICLE_VIEW_RUN_ID"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=os.environ.get("PARTICLE_VIEW_SEED"),
    )
    parser.add_argument(
        "--task-id",
        default=os.environ.get("PARTICLE_VIEW_TASK_ID"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("PARTICLE_VIEW_TASK_OUTPUT_DIR"),
    )
    args = parser.parse_args(argv)
    for name in ("run_id", "seed", "task_id", "output_dir"):
        if getattr(args, name) in {None, ""}:
            raise ValueError(f"--{name.replace('_', '-')} is required")
    result = execute_scientific_task(
        catalog=load_hashed_json(args.catalog),
        registry=load_hashed_json(args.registry),
        run_id=args.run_id,
        seed=args.seed,
        task_id=args.task_id,
        output_dir=Path(args.output_dir),
    )
    print(
        f"task_id={args.task_id} run_id={args.run_id} "
        f"content_hash={result['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
