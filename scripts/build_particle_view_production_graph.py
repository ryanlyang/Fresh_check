#!/usr/bin/env python3
"""Build and reconcile the immutable Step-10 production Slurm graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_particle_view_production_graph,
    load_hashed_json,
    reconcile_particle_view_production_graph,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--command-catalog", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--graph-id", default="particle_view_full_pilot_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reconciliation-output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = load_hashed_json(args.registry)
    catalog = json.loads(Path(args.command_catalog).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("command catalog must be a JSON object")
    graph = build_particle_view_production_graph(
        registry=registry,
        artifact_root=str(Path(args.artifact_root).resolve()),
        source_commit=args.source_commit,
        command_catalog=catalog,
        graph_id=args.graph_id,
    )
    reconciliation = reconcile_particle_view_production_graph(
        graph=graph, registry=registry
    )
    if not args.dry_run:
        write_immutable_json(args.output, graph)
        write_immutable_json(args.reconciliation_output, reconciliation)
    print(
        json.dumps(
            {
                "graph_sha256": graph["content_hash"],
                "reconciled": reconciliation["reconciled"],
                "counts": reconciliation["counts"],
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
