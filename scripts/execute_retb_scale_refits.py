#!/usr/bin/env python3
"""Execute and attest every locked 3M normalizer/calibrator refit."""

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
    SCALE_SHORTLIST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    SCALE_REFIT_EXECUTION_PLAN_CONTRACT,
    execute_plan_steps,
    validate_scale_refit_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    build_scale_refit_bundle,
    validate_scale_refit_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    plan = load_hashed_json(
        args.execution_plan,
        expected_contract=SCALE_REFIT_EXECUTION_PLAN_CONTRACT,
    )
    validate_scale_refit_execution_plan(
        plan,
        campaign_source=campaign["source"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    if (
        shortlist.get("source") != campaign["source"]
        or plan["locked_scale_shortlist_sha256"]
        != shortlist["content_hash"]
    ):
        raise ValueError("scale-refit execution lineage differs")
    authorize_dataset_access(
        worker_role="scale_training_worker",
        requested_resource="scale_train",
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    refits = {}
    receipts = {}
    for name, operation in plan["operations"].items():
        receipts[name] = execute_plan_steps(
            operation["steps"],
            campaign_root=args.campaign_root,
            repo_root=REPO_ROOT,
        )
        artifact = load_hashed_json(Path(operation["output_artifact"]))
        if artifact.get("source") != campaign["source"]:
            raise ValueError("scale-refit output source differs")
        refits[name] = {
            "artifact_sha256": artifact["content_hash"],
            "population": operation["population"],
            "identity_manifest_sha256": operation[
                "identity_manifest_sha256"
            ],
            "recipe_sha256": operation["recipe_sha256"],
            "fitted_values_sha256": artifact["content_hash"],
            "labels_consumed": False,
            "replica_ids": operation["replica_ids"],
        }
    bundle = bind_source(
        build_scale_refit_bundle(
            graph_id=plan["graph_id"],
            pipeline_seed=int(plan["pipeline_seed"]),
            locked_scale_shortlist_sha256=shortlist["content_hash"],
            scale_train_manifest_sha256=plan[
                "scale_train_manifest_sha256"
            ],
            val_design_identity_manifest_sha256=plan[
                "val_design_identity_manifest_sha256"
            ],
            refits=refits,
            five_hundred_k_artifact_hashes=plan[
                "five_hundred_k_artifact_hashes"
            ],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_scale_refit_bundle(bundle)
    publication = write_immutable_json(args.output, bundle)
    print(
        json.dumps(
            {
                "graph_id": bundle["graph_id"],
                "pipeline_seed": bundle["pipeline_seed"],
                "scale_refit_bundle_sha256": bundle["content_hash"],
                "operation_receipts": receipts,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
