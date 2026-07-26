#!/usr/bin/env python3
"""Bind selected winner replicas to Stage-G ledger/resource inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_fairness_input_index,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--bindings",
        required=True,
        help="JSON keyed by configuration ID and seed with ledger/resource bindings",
    )
    parser.add_argument("--flop-fixture-sha256", required=True)
    parser.add_argument("--flop-counter-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    raw = json.loads(Path(args.bindings).read_text(encoding="utf-8"))
    configurations = {
        configuration_id: {
            int(seed): binding for seed, binding in replicas.items()
        }
        for configuration_id, replicas in raw.items()
    }
    artifact = build_fairness_input_index(
        selection=load_hashed_json(args.selection),
        configurations=configurations,
        flop_fixture_sha256=args.flop_fixture_sha256,
        flop_counter_sha256=args.flop_counter_sha256,
    )
    if not args.dry_run:
        write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "configuration_count": artifact["configuration_count"],
                "replica_count": artifact["replica_count"],
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
