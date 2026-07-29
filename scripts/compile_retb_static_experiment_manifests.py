#!/usr/bin/env python3
"""Compile and optionally publish all selector-independent RETB run manifests."""

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
    build_static_experiment_bundle,
    load_hashed_json,
    publish_static_experiment_bundle,
    validate_static_experiment_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    bundle = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=graph,
        campaign_root=args.campaign_root,
    )
    validate_static_experiment_bundle(
        bundle,
        campaign=campaign,
        production_graph=graph,
        campaign_root=args.campaign_root,
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "static_experiment_plan_sha256": bundle[
            "static_experiment_plan"
        ]["content_hash"],
        "static_experiment_bundle_sha256": bundle[
            "static_experiment_bundle"
        ]["content_hash"],
        "full_matrix_counts": bundle["static_experiment_plan"][
            "full_matrix_counts"
        ],
        "execution_counts": bundle["static_experiment_plan"][
            "execution_counts"
        ],
        "task_manifest_hashes": bundle["static_experiment_bundle"][
            "task_manifest_hashes"
        ],
    }
    if not args.dry_run:
        result["publication"] = publish_static_experiment_bundle(
            campaign_root=args.campaign_root,
            bundle=bundle,
            campaign=campaign,
            production_graph=graph,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
