#!/usr/bin/env python3
"""Generate and attest one post-finalist-lock oracle target cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    build_postlock_oracle_target,
    validate_postlock_oracle_target,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    execute_plan_steps,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    POSTLOCK_TARGET_EXECUTION_PLAN_CONTRACT,
    validate_postlock_target_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    LOCKED_SCALE_FINALISTS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-finalists", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    finalists = load_hashed_json(
        args.locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    plan = load_hashed_json(
        args.execution_plan,
        expected_contract=POSTLOCK_TARGET_EXECUTION_PLAN_CONTRACT,
    )
    validate_postlock_target_execution_plan(
        plan,
        campaign_source=campaign["source"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    role, resource = (
        ("postlock_stack_diagnostic", "stack_val_oracle_targets")
        if plan["split"] == "stack_val"
        else ("final_test_worker", "final_test_targets")
    )
    authorize_dataset_access(worker_role=role, requested_resource=resource)
    if (
        finalists.get("source") != campaign["source"]
        or plan["locked_scale_finalists_sha256"]
        != finalists["content_hash"]
    ):
        raise ValueError("postlock target execution lineage differs")
    receipts = execute_plan_steps(
        plan["steps"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
        forbidden_terms=(),
        forbidden_entrypoints=frozenset(
            {
                "build_retb_postlock_oracle_targets.py",
                "execute_retb_postlock_oracle_target.py",
            }
        ),
    )
    evidence = load_hashed_json(plan["target_evidence"])
    required = {
        "graph_id",
        "pipeline_seed",
        "split",
        "parent_hashes",
        "target_cache_manifest_sha256",
        "target_identity_order_sha256",
        "target_dtype",
        "float16_audit_sha256",
        "float16_audit_passed",
        "actual_target_generation_executed",
    }
    if (
        set(evidence)
        - {"contract", "schema_version", "source", "content_hash"}
        != required
        or evidence.get("source") != campaign["source"]
        or evidence["graph_id"] != plan["graph_id"]
        or int(evidence["pipeline_seed"]) != int(plan["pipeline_seed"])
        or evidence["split"] != plan["split"]
        or evidence["actual_target_generation_executed"] is not True
    ):
        raise ValueError("postlock target generation evidence differs")
    artifact = bind_source(
        build_postlock_oracle_target(
            locked_scale_finalists=finalists,
            graph_id=plan["graph_id"],
            pipeline_seed=int(plan["pipeline_seed"]),
            split=plan["split"],
            parent_hashes=evidence["parent_hashes"],
            target_cache_manifest_sha256=evidence[
                "target_cache_manifest_sha256"
            ],
            target_identity_order_sha256=evidence[
                "target_identity_order_sha256"
            ],
            target_dtype=evidence["target_dtype"],
            float16_audit_sha256=evidence["float16_audit_sha256"],
            float16_audit_passed=evidence["float16_audit_passed"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_postlock_oracle_target(
        artifact, locked_scale_finalists=finalists
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "postlock_target_sha256": artifact["content_hash"],
                "execution_receipts": receipts,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
