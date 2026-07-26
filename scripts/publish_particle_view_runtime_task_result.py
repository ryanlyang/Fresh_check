#!/usr/bin/env python3
"""Publish the mandatory hash-authenticated result of one runtime task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_runtime_task_result,
    sha256_file,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        default=os.environ.get("PARTICLE_VIEW_TASK_ID"),
    )
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--warning-sha256", action="append", default=[])
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.task_id:
        raise ValueError("--task-id or PARTICLE_VIEW_TASK_ID is required")
    output = args.output
    if output is None:
        task_root = os.environ.get("PARTICLE_VIEW_TASK_OUTPUT_DIR")
        if not task_root:
            raise ValueError(
                "--output or PARTICLE_VIEW_TASK_OUTPUT_DIR is required"
            )
        output = str(Path(task_root) / "task_result.json")
    artifacts = []
    for raw in args.artifact:
        path = Path(raw).resolve()
        artifacts.append({"path": str(path), "sha256": sha256_file(path)})
    result = build_runtime_task_result(
        task_id=args.task_id,
        artifacts=artifacts,
        warning_sha256=args.warning_sha256,
    )
    write_immutable_json(output, result)
    print(
        f"task_id={args.task_id} artifacts={len(artifacts)} "
        f"content_hash={result['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
