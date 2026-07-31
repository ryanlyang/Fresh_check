#!/usr/bin/env python3
"""Write the locked duplicate-free 500k accuracy/rejection shortlist."""

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
    BRIDGE_SHAPE_SELECTION_CONTRACT,
    CONFIRMATION_SUMMARY_CONTRACT,
    SHORTLIST_PARENT_KEYS,
    STAGE_L_GRAPH_REGISTRY_CONTRACT,
    select_scale_shortlist,
    validate_scale_shortlist,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step13 import (  # noqa: E402
    STEP13_BUNDLE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-registry", required=True, type=Path)
    parser.add_argument("--confirmation-summary", required=True, type=Path)
    parser.add_argument("--bridge-shape-selection", required=True, type=Path)
    parser.add_argument("--parent-hashes", required=True, type=Path)
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
    confirmation = load_hashed_json(
        args.confirmation_summary,
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    shape = load_hashed_json(
        args.bridge_shape_selection,
        expected_contract=BRIDGE_SHAPE_SELECTION_CONTRACT,
    )
    parent_artifact = load_hashed_json(
        args.parent_hashes,
        expected_contract="retb_scale_shortlist_parent_input_v1",
    )
    if parent_artifact.get("source") != campaign.get("source"):
        raise ValueError("scale-shortlist parent input source differs")
    parents = parent_artifact["parent_hashes"]
    step13 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step13_confirmation_shortlist_bundle.json",
        expected_contract=STEP13_BUNDLE_CONTRACT,
    )
    if (
        set(parents) != set(SHORTLIST_PARENT_KEYS)
        or any(
            row.get("source") != campaign.get("source")
            for row in (registry, confirmation, shape, step13)
        )
        or parents["campaign_spec"] != campaign["content_hash"]
        or parents["step13_bundle"] != step13["content_hash"]
        or parents["step12_bundle"]
        != step13["parents"]["step12_bundle"]
        or parents["step12_bundle"]
        != registry["step12_bundle_sha256"]
        or parents["validation_partition_manifest"]
        != campaign["parent_artifact_hashes"][
            "validation_partition_manifest"
        ]
        or parents["hlt_replica_manifest"]
        != campaign["parent_artifact_hashes"]["hlt_replica_manifest"]
    ):
        raise ValueError("scale-shortlist source/parent lineage differs")
    artifact = bind_source(
        select_scale_shortlist(
            confirmation_summary=confirmation,
            graph_registry=registry,
            bridge_shape_selection=shape,
            parent_hashes=parents,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_scale_shortlist(
        artifact,
        confirmation_summary=confirmation,
        graph_registry=registry,
        bridge_shape_selection=shape,
    )
    result = {
        "dry_run": args.dry_run,
        "locked_scale_shortlist_sha256": artifact["content_hash"],
        "SCALE_SHORTLIST": artifact["SCALE_SHORTLIST"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
