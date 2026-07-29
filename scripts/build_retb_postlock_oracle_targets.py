#!/usr/bin/env python3
"""Attest one post-finalist-lock stack/final oracle target cache."""

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
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    finalists = load_hashed_json(
        args.locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
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
    }
    if (
        set(configuration) != required
        or finalists.get("source") != campaign.get("source")
    ):
        raise ValueError("postlock target source/configuration differs")
    if configuration["split"] == "stack_val":
        authorize_dataset_access(
            worker_role="postlock_stack_diagnostic",
            requested_resource="stack_val_oracle_targets",
        )
    elif configuration["split"] == "final_test":
        authorize_dataset_access(
            worker_role="final_test_worker",
            requested_resource="final_test_targets",
        )
    artifact = bind_source(
        build_postlock_oracle_target(
            locked_scale_finalists=finalists, **configuration
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_postlock_oracle_target(
        artifact, locked_scale_finalists=finalists
    )
    result = {
        "dry_run": args.dry_run,
        "postlock_target_sha256": artifact["content_hash"],
        "split": artifact["split"],
        "selection_eligible": False,
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
