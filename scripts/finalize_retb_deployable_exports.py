#!/usr/bin/env python3
"""Authenticate complete Step-12 deployable-export coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    COMPLETE_GRAPH_CAPACITY_CONTRACT,
    DEPLOYABLE_EXPORT_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    STAGE_J_CONSUMER_REGISTRY_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


CONTRACT = "retb_deployable_export_index_v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    registry = load_hashed_json(
        root / "registry" / "retb_final_consumer_registry.json",
        expected_contract=STAGE_J_CONSUMER_REGISTRY_CONTRACT,
    )
    exports = {}
    for row in registry["rows"]:
        output = root / "exports" / row["run_id"]
        manifest = load_hashed_json(
            output / "deployable_retb_graph.json",
            expected_contract=DEPLOYABLE_EXPORT_CONTRACT,
        )
        parity = load_hashed_json(output / "research_graph_parity.json")
        capacity = load_hashed_json(
            output / "complete_graph_capacity.json",
            expected_contract=COMPLETE_GRAPH_CAPACITY_CONTRACT,
        )
        if (
            manifest.get("source") != campaign.get("source")
            or parity.get("source") != campaign.get("source")
            or capacity.get("source") != campaign.get("source")
            or parity["deployment_export_sha256"]
            != manifest["content_hash"]
            or capacity["deployment_export_sha256"]
            != manifest["content_hash"]
            or parity.get("passed") is not True
        ):
            raise ValueError("deployable export/parity lineage differs")
        exports[row["run_id"]] = {
            "pipeline_seed": row["pipeline_seed"],
            "carried_shape_role": row["carried_shape_role"],
            "consumer_kind": row["consumer_kind"],
            "model_variant": row["model_variant"],
            "native_dropout_mode": row["native_dropout_mode"],
            "token_input": row["token_input"],
            "export_path": str(
                (output / "deployable_retb_graph.json").resolve()
            ),
            "export_sha256": manifest["content_hash"],
            "graph_sha256": manifest["graph_sha256"],
            "research_graph_parity_sha256": parity["content_hash"],
            "complete_graph_capacity_path": str(
                (output / "complete_graph_capacity.json").resolve()
            ),
            "complete_graph_capacity_sha256": capacity["content_hash"],
            "capacity": capacity["totals"],
        }
    if len(exports) != int(registry["membership_count"]):
        raise ValueError("deployable export coverage differs")
    artifact = bind_source(
        with_content_hash(
            {
                "contract": CONTRACT,
                "schema_version": 1,
                "final_consumer_registry_sha256": registry["content_hash"],
                "expected_export_count": registry["membership_count"],
                "export_count": len(exports),
                "exports": dict(sorted(exports.items())),
                "complete_coverage": True,
                "all_research_graph_parity_passed": True,
                "scientific_performance_used_to_omit_exports": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
