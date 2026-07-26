#!/usr/bin/env python3
"""Build seed-expanded runtime tasks and automatic graph-node commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_runtime_command_catalog,
    build_runtime_execution_manifest,
    build_runtime_handler_catalog,
    load_hashed_json,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--handler-commands", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--handler-catalog-output", required=True)
    parser.add_argument("--execution-manifest-output", required=True)
    parser.add_argument("--command-catalog-output", required=True)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = load_hashed_json(args.registry)
    commands = json.loads(
        Path(args.handler_commands).read_text(encoding="utf-8")
    )
    if not isinstance(commands, dict):
        raise ValueError("handler commands must be a JSON object")
    handler_catalog = build_runtime_handler_catalog(commands)
    manifest = build_runtime_execution_manifest(
        registry=registry,
        registry_path=args.registry,
        handler_catalog=handler_catalog,
        handler_catalog_path=args.handler_catalog_output,
        artifact_root=args.artifact_root,
    )
    command_catalog = build_runtime_command_catalog(
        execution_manifest_path=args.execution_manifest_output,
        python_executable=args.python_executable,
    )
    if not args.dry_run:
        write_immutable_json(args.handler_catalog_output, handler_catalog)
        write_immutable_json(args.execution_manifest_output, manifest)
        destination = Path(args.command_catalog_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            command_catalog,
            sort_keys=True,
            indent=2,
        ) + "\n"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != encoded:
                raise FileExistsError("refusing to overwrite command catalog")
        else:
            destination.write_text(encoded, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "handler_catalog_sha256": handler_catalog["content_hash"],
                "execution_manifest_sha256": manifest["content_hash"],
                "task_count": manifest["seed_expanded_task_count"],
                "node_command_count": len(command_catalog),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
