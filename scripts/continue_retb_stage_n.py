#!/usr/bin/env python3
"""Publish one completeness-gated sealed Stage-N continuation."""

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
    build_final_continuation,
    load_hashed_json,
    publish_final_continuation,
    validate_final_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--trigger-artifact", required=True, type=Path)
    parser.add_argument("--rows-json", required=True, type=Path)
    parser.add_argument("--production-graph", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.production_graph
        or args.campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    trigger = load_hashed_json(args.trigger_artifact)
    rows = json.loads(args.rows_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Stage-N rows JSON must be a list")
    payload = build_final_continuation(
        campaign=campaign,
        production_graph=graph,
        campaign_root=args.campaign_root,
        node_id=args.node_id,
        trigger_artifact=trigger,
        rows=rows,
    )
    validate_final_continuation(
        payload,
        campaign=campaign,
        production_graph=graph,
        campaign_root=args.campaign_root,
        trigger_artifact=trigger,
        rows=rows,
    )
    result = {
        "dry_run": args.dry_run,
        "node_id": args.node_id,
        "task_count": payload["final_continuation_bundle"]["task_count"],
        "bundle_sha256": payload["final_continuation_bundle"][
            "content_hash"
        ],
    }
    if not args.dry_run:
        result["publication"] = publish_final_continuation(
            campaign_root=args.campaign_root, payload=payload
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
