#!/usr/bin/env python3
"""Execute or resume every authenticated task assigned to one graph node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    execute_runtime_node,
    load_hashed_json,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_hashed_json(args.execution_manifest)
    if (
        args.expected_manifest_sha256 is not None
        and manifest["content_hash"] != args.expected_manifest_sha256
    ):
        raise ValueError("runtime execution manifest changed after graph creation")
    report = execute_runtime_node(
        manifest=manifest,
        node_id=args.node_id,
        task_ids=None if args.task_id is None else [args.task_id],
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        output = (
            Path(manifest["artifact_root"])
            / "runtime_node_reports"
            / args.node_id
            / f"{report['content_hash']}.json"
        )
        write_immutable_json(output, report)
    print(json.dumps(report, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
