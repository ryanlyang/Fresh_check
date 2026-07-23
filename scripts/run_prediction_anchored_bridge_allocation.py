#!/usr/bin/env python3
"""Validate one Step 10 packed allocation and publish its launch manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    build_allocation_launch_manifest,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--reservations", required=True)
    parser.add_argument("--representative-reference", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--ram-root", required=True)
    parser.add_argument("--selected-consumer", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    graph = load_hashed_json(args.graph)
    execution_spec = load_hashed_json(args.execution_spec)
    reservations = load_hashed_json(args.reservations)
    representative_reference = load_hashed_json(args.representative_reference)
    selected = (
        None
        if not args.selected_consumer
        else load_hashed_json(args.selected_consumer)
    )
    artifact = build_allocation_launch_manifest(
        graph,
        node_id=args.node_id,
        environment=os.environ,
        ram_root=args.ram_root,
        selected_consumer=selected,
        execution_spec=execution_spec,
        reservations=reservations,
        representative_reference=representative_reference,
        dry_run=bool(args.dry_run),
    )
    result = {"dry_run": bool(args.dry_run), "launch_manifest": artifact}
    if not args.dry_run:
        if not args.output:
            raise ValueError("allocation launch requires --output unless --dry-run")
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
