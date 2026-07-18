#!/usr/bin/env python3
"""Create, inspect, or safely clean one rank-local ABPH RAM workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.ram_workspace import (  # noqa: E402
    RankLocalWorkspace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--path-only", action="store_true")
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--workspace", required=True)
    cleanup.add_argument("--job-id", required=True)
    cleanup.add_argument("--rank", type=int, required=True)
    cleanup.add_argument("--require-empty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        workspace = RankLocalWorkspace.from_environment()
        if args.path_only:
            print(workspace.root)
        else:
            print(json.dumps(workspace._require_owner(), indent=2, sort_keys=True))
        return 0
    workspace = RankLocalWorkspace(
        args.workspace,
        job_id=args.job_id,
        rank=args.rank,
        create=False,
    )
    workspace.cleanup(require_empty=bool(args.require_empty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
