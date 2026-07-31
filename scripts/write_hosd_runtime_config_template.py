#!/usr/bin/env python3
"""Write the complete editable infrastructure binding template for HOSD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    DIRECTORY_INFRASTRUCTURE_OPTIONS,
    REQUIRED_INFRASTRUCTURE_OPTION_KEYS,
    REQUIRED_INFRASTRUCTURE_OPTION_MIN_COUNTS,
    REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE,
)


SCALAR_DEFAULTS = {
    "--available-storage-bytes": "__REQUIRED_INTEGER_AVAILABLE_STORAGE_BYTES__",
    "--clock-power-mode": "__REQUIRED_MEASURED_CLOCK_POWER_MODE__",
    "--production-batch-size": "__REQUIRED_INTEGER_PRODUCTION_BATCH_SIZE__",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_template() -> dict[str, object]:
    files: dict[str, str] = {}
    directories: dict[str, str] = {}
    arguments: dict[str, list[str]] = {}
    for node_id, options in sorted(
        REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE.items()
    ):
        argv: list[str] = []
        minimums = REQUIRED_INFRASTRUCTURE_OPTION_MIN_COUNTS.get(node_id, {})
        keyed = REQUIRED_INFRASTRUCTURE_OPTION_KEYS.get(node_id, {})
        for option in options:
            required_keys = sorted(keyed.get(option, ()))
            count = max(int(minimums.get(option, 1)), len(required_keys))
            for index in range(count):
                argv.append(option)
                if option in SCALAR_DEFAULTS:
                    argv.append(SCALAR_DEFAULTS[option])
                    continue
                binding_key = _slug(
                    f"{node_id}_{option}_{required_keys[index] if index < len(required_keys) else index}"
                )
                collection = (
                    directories
                    if option in DIRECTORY_INFRASTRUCTURE_OPTIONS
                    else files
                )
                collection[binding_key] = (
                    f"__REQUIRED_{'DIRECTORY' if collection is directories else 'FILE'}_"
                    f"{binding_key.upper()}__"
                )
                reference = (
                    f"{{{'directory' if collection is directories else 'file'}_"
                    f"{binding_key}}}"
                )
                argv.append(
                    f"{required_keys[index]}={reference}"
                    if index < len(required_keys)
                    else reference
                )
        arguments[node_id] = argv
    return {
        "files": files,
        "directories": directories,
        "infrastructure_arguments_by_node": arguments,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"runtime template already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_template(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
