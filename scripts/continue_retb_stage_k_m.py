#!/usr/bin/env python3
"""Publish one completeness-gated RETB Stage K--M task manifest."""

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
    LATE_CONTINUATION_MANIFEST_NODES,
    PRODUCTION_GRAPH_CONTRACT,
    build_late_continuation,
    load_hashed_json,
    publish_late_continuation,
    validate_late_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.dynamic_continuation import (  # noqa: E402
    load_continuation_rows,
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
    if args.node_id not in LATE_CONTINUATION_MANIFEST_NODES:
        raise ValueError("node is outside the Stage K--M continuation")
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.production_graph
        or args.campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    trigger = load_hashed_json(args.trigger_artifact)
    rows = load_continuation_rows(args.rows_json)
    payload = build_late_continuation(
        campaign=campaign,
        production_graph=graph,
        campaign_root=args.campaign_root,
        node_id=args.node_id,
        trigger_artifact=trigger,
        rows=rows,
    )
    digest = validate_late_continuation(
        payload,
        campaign=campaign,
        production_graph=graph,
        campaign_root=args.campaign_root,
        trigger_artifact=trigger,
        rows=rows,
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "node_id": args.node_id,
        "task_count": payload["dynamic_continuation"][
            "task_manifest"
        ]["task_count"],
        "gate_sha256": payload["gate"]["content_hash"],
        "task_manifest_sha256": payload["dynamic_continuation"][
            "task_manifest"
        ]["content_hash"],
        "continuation_binding_sha256": payload[
            "dynamic_continuation"
        ]["continuation_binding"]["content_hash"],
        "late_continuation_bundle_sha256": digest,
    }
    if not args.dry_run:
        result["publication"] = publish_late_continuation(
            campaign_root=args.campaign_root,
            payload=payload,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
