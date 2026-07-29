#!/usr/bin/env python3
"""Aggregate available matched-seed RETB 500k confirmation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    SEED_CONFIRMATION_CONTRACT,
    STAGE_L_GRAPH_REGISTRY_CONTRACT,
    aggregate_500k_confirmation,
    validate_stage_l_graph_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-registry", required=True, type=Path)
    parser.add_argument(
        "--seed-confirmation", action="append", default=[], type=Path
    )
    parser.add_argument(
        "--val-design-label-manifest-sha256", required=True
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    registry = load_hashed_json(
        args.graph_registry,
        expected_contract=STAGE_L_GRAPH_REGISTRY_CONTRACT,
    )
    validate_stage_l_graph_registry(registry)
    rows = [
        load_hashed_json(
            path, expected_contract=SEED_CONFIRMATION_CONTRACT
        )
        for path in args.seed_confirmation
    ]
    if (
        registry.get("source") != campaign.get("source")
        or any(row.get("source") != campaign.get("source") for row in rows)
    ):
        raise ValueError("500k confirmation source differs")
    artifact = bind_source(
        aggregate_500k_confirmation(
            graph_registry=registry,
            seed_confirmations=rows,
            val_design_label_manifest_sha256=(
                args.val_design_label_manifest_sha256
            ),
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": args.dry_run,
        "confirmation_summary_sha256": artifact["content_hash"],
        "complete_graph_count": artifact["complete_graph_count"],
        "incomplete_graph_count": len(
            artifact["ineligible_incomplete_graphs"]
        ),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
