#!/usr/bin/env python3
"""Register complete immutable graph definitions for Stage-L confirmation."""

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
    build_stage_l_graph_registry,
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
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    STEP12_BUNDLE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--candidate-shape-id", action="append", required=True
    )
    parser.add_argument(
        "--robustness-controls-completion-sha256", required=True
    )
    parser.add_argument(
        "--semantic-controls-completion-sha256", required=True
    )
    parser.add_argument("--definitions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step12 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step12_final_consumers_bundle.json",
        expected_contract=STEP12_BUNDLE_CONTRACT,
    )
    definitions = json.loads(args.definitions.read_text("utf-8"))
    if (
        step12.get("source") != campaign.get("source")
        or not isinstance(definitions, list)
    ):
        raise ValueError("Stage-L graph registry parent lineage differs")
    artifact = bind_source(
        build_stage_l_graph_registry(
            definitions=definitions,
            step12_bundle_sha256=step12["content_hash"],
            candidate_shape_ids=args.candidate_shape_id,
            robustness_controls_completion_sha256=(
                args.robustness_controls_completion_sha256
            ),
            semantic_controls_completion_sha256=(
                args.semantic_controls_completion_sha256
            ),
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_stage_l_graph_registry(artifact)
    result = {
        "dry_run": args.dry_run,
        "graph_registry_sha256": artifact["content_hash"],
        "definition_count": artifact["definition_count"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
