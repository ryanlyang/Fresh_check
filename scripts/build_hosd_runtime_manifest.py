#!/usr/bin/env python3
"""Bind infrastructure-only paths and arguments for automatic HOSD factories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_runtime_manifest,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    write_immutable_json,
)


def _pairs(values: list[str], *, name: str) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, payload = value.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"{name} must be unique NAME=VALUE")
        result[key] = payload
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--directory", action="append", default=[])
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help=(
            "single JSON document containing files, directories, and "
            "infrastructure_arguments_by_node"
        ),
    )
    parser.add_argument(
        "--node-arguments-json",
        type=Path,
        help="JSON node-to-argv mapping; scientific row/seed options are rejected",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(
        args.campaign_root, repo_root=REPO_ROOT
    )
    if args.runtime_config is not None and (
        args.file or args.directory or args.node_arguments_json is not None
    ):
        raise ValueError(
            "--runtime-config is mutually exclusive with component options"
        )
    if args.runtime_config is not None:
        config = json.loads(args.runtime_config.read_text(encoding="utf-8"))
        if set(config) != {
            "files",
            "directories",
            "infrastructure_arguments_by_node",
        }:
            raise ValueError("runtime config has unexpected or missing keys")
        files = config["files"]
        directories = config["directories"]
        arguments = config["infrastructure_arguments_by_node"]
        if not all(isinstance(value, dict) for value in (files, directories, arguments)):
            raise ValueError("runtime config sections must be JSON objects")
    else:
        files = _pairs(args.file, name="file")
        directories = _pairs(args.directory, name="directory")
        arguments = (
            {}
            if args.node_arguments_json is None
            else json.loads(
                args.node_arguments_json.read_text(encoding="utf-8")
            )
        )
    artifact = build_runtime_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=arguments,
        source=campaign["source"],
    )
    output = args.output or (
        args.campaign_root / "registry" / "runtime_manifest.json"
    )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
