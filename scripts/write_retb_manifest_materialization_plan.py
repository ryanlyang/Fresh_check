#!/usr/bin/env python3
"""Publish an immutable downstream RETB manifest row plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.manifest_orchestration import (  # noqa: E402
    build_manifest_materialization_plan,
    publish_manifest_materialization_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--producer-node-id", required=True)
    parser.add_argument("--target-node-id", required=True)
    parser.add_argument("--rows-json", required=True, type=Path)
    parser.add_argument("--trigger-artifact", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    rows = json.loads(args.rows_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("manifest materialization rows JSON must be a list")
    trigger = (
        None
        if args.trigger_artifact is None
        else load_hashed_json(args.trigger_artifact)
    )
    plan = build_manifest_materialization_plan(
        campaign=campaign,
        production_graph=graph,
        producer_node_id=args.producer_node_id,
        target_node_id=args.target_node_id,
        rows=rows,
        trigger_artifact_path=args.trigger_artifact,
        trigger_artifact_sha256=(
            None if trigger is None else trigger["content_hash"]
        ),
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "plan_sha256": plan["content_hash"],
        "target_node_id": args.target_node_id,
        "row_count": plan["row_count"],
    }
    if not args.dry_run:
        result["publication"] = publish_manifest_materialization_plan(
            campaign_root=args.campaign_root,
            plan=plan,
            campaign=campaign,
            production_graph=graph,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
