#!/usr/bin/env python3
"""Resolve and validate the complete Step-8 Tigris submission graph."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    build_job_ledger,
    build_production_graph,
    source_snapshot,
    write_immutable_json,
)


def _campaign_id(source: dict[str, object]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"rpt_attention_bias_{stamp}_"
        f"{str(source['source_commit'])[:10]}_"
        f"{str(source['source_status_sha256'])[:10]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ryreu/atlas/Fresh_check/checkpoints"),
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--screening-array-concurrency", type=int, default=4)
    parser.add_argument("--tree-array-concurrency", type=int, default=16)
    parser.add_argument("--miniature", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-simulate", action="store_true")
    parser.add_argument("--write-artifacts", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.write_artifacts:
        raise ValueError("--dry-run may not mutate the campaign root")
    source = source_snapshot(REPO_ROOT)
    campaign_id = args.campaign_id or _campaign_id(source)
    campaign_root = (
        args.campaign_root
        if args.campaign_root is not None
        else (
            args.output_root
            / "relational_particle_transformer"
            / campaign_id
        )
    )
    graph = build_production_graph(
        campaign_root=campaign_root,
        campaign_id=campaign_id,
        source_commit=str(source["source_commit"]),
        source_status_sha256=str(source["source_status_sha256"]),
        miniature=bool(args.miniature),
        screening_array_concurrency=args.screening_array_concurrency,
        tree_array_concurrency=args.tree_array_concurrency,
    )
    jobs = {
        node["node_id"]: (
            str(index + 10_000) if args.smoke_simulate else None
        )
        for index, node in enumerate(graph["nodes"])
    }
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=jobs,
        submission_mode=(
            "smoke_simulation" if args.smoke_simulate else "dry_run"
        ),
    )
    result = {
        "dry_run": bool(args.dry_run),
        "smoke_simulate": bool(args.smoke_simulate),
        "campaign_root": str(campaign_root),
        "source": source,
        "production_graph": graph,
        "job_ledger": ledger,
        "production_submission_performed": False,
        "next_command": (
            "bash sbatch/submit_relational_part_tigris_full.sh"
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.write_artifacts:
        if campaign_root.exists() and not campaign_root.is_dir():
            raise ValueError("campaign root exists and is not a directory")
        campaign_root.mkdir(parents=True, exist_ok=True)
        (campaign_root / "job_ledgers").mkdir(exist_ok=True)
        write_immutable_json(
            campaign_root / "job_ledgers" / "production_graph.json", graph
        )
        write_immutable_json(
            campaign_root
            / "job_ledgers"
            / "graph_resolution_ledger.json",
            ledger,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
