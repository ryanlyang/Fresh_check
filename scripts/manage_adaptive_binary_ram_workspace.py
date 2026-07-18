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
    reserve = commands.add_parser("reserve")
    reserve.add_argument("--workspace", required=True)
    reserve.add_argument("--job-id", required=True)
    reserve.add_argument("--rank", type=int, required=True)
    reserve.add_argument("--owner", required=True)
    reserve.add_argument("--role", required=True)
    reserve.add_argument("--expected-bytes", type=int, required=True)
    reserve.add_argument("--id-only", action="store_true")
    commit = commands.add_parser("commit")
    commit.add_argument("--workspace", required=True)
    commit.add_argument("--job-id", required=True)
    commit.add_argument("--rank", type=int, required=True)
    commit.add_argument("--reservation-id", required=True)
    commit.add_argument("--measured-path", required=True)
    release = commands.add_parser("release")
    release.add_argument("--workspace", required=True)
    release.add_argument("--job-id", required=True)
    release.add_argument("--rank", type=int, required=True)
    release.add_argument("--reservation-id", required=True)
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
    if args.command == "reserve":
        reservation_id = workspace.reserve(
            owner=args.owner,
            role=args.role,
            expected_bytes=args.expected_bytes,
        )
        if args.id_only:
            print(reservation_id)
        else:
            print(json.dumps({"reservation_id": reservation_id}, sort_keys=True))
        return 0
    if args.command == "commit":
        measured_bytes = workspace.commit_tree(
            args.reservation_id, args.measured_path
        )
        print(
            json.dumps(
                {
                    "reservation_id": args.reservation_id,
                    "measured_bytes": measured_bytes,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "release":
        workspace.release(args.reservation_id)
        return 0
    workspace.cleanup(require_empty=bool(args.require_empty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
