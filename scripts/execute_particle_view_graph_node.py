#!/usr/bin/env python3
"""Execute one authenticated logical graph node inside a Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_node_completion,
    load_hashed_json,
    load_quality_warning_jsonl,
    validate_particle_view_production_graph,
    write_immutable_json,
    write_quality_warning_jsonl,
    write_quality_warning_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    graph = load_hashed_json(args.graph)
    audit = validate_particle_view_production_graph(graph)
    if args.node_id not in audit["nodes"]:
        raise ValueError("logical node is absent from the production graph")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("production nodes require PYTHONNOUSERSITE=1")
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    if account is not None and account != "reu-aisocial":
        raise RuntimeError("Tigris job is running under the wrong account")
    node = audit["nodes"][args.node_id]
    print("NODE_COMMAND:")
    print("  " + " ".join(node["command"]))
    if args.dry_run:
        return 0
    completed = subprocess.run(node["command"], check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    warning_path = (
        Path(graph["artifact_root"])
        / "quality_warnings"
        / args.node_id
        / "quality_warnings.jsonl"
    )
    if not warning_path.exists():
        write_quality_warning_jsonl(warning_path, [])
    warnings = load_quality_warning_jsonl(warning_path)
    warning_hashes = [warning["content_hash"] for warning in warnings]
    if args.node_id == "pv10_hlt_only_final_test":
        campaign_warnings = []
        warning_root = Path(graph["artifact_root"]) / "quality_warnings"
        for path in sorted(warning_root.glob("*/quality_warnings.jsonl")):
            campaign_warnings.extend(load_quality_warning_jsonl(path))
        write_quality_warning_summary(
            output_dir=graph["artifact_root"],
            warnings=campaign_warnings,
        )
    completion = build_node_completion(
        graph=graph,
        node_id=args.node_id,
        output_artifacts=[],
        warning_sha256=warning_hashes,
    )
    output = (
        Path(graph["artifact_root"])
        / "node_completions"
        / f"{args.node_id}.json"
    )
    write_immutable_json(output, completion)
    print(
        json.dumps(
            {
                "node_id": args.node_id,
                "warning_count": len(warning_hashes),
                "completion_sha256": completion["content_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
