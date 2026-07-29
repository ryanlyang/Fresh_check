#!/usr/bin/env python3
"""Materialize one immutable RETB Stage-J run from the locked registry."""

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
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step11 import (  # noqa: E402
    STAGE_J_REGISTRY_CONTRACT,
    materialize_stage_j_run,
    validate_materialized_stage_j_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step11_bundle = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step11_joint_bridge_bundle.json"
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_j_registry.json",
        expected_contract=STAGE_J_REGISTRY_CONTRACT,
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    required = {
        "run_id",
        "variant",
        "pipeline_seed",
        "final_particle_blocks",
        "predictor_bundle_lock_sha256",
        "step11_bundle_sha256",
        "parent_hashes",
        "semantic_label",
    }
    if set(configuration) != required:
        raise ValueError("Stage-J materialization fields differ")
    registered = [
        row
        for row in registry["rows"]
        if row["run_id"] == configuration["run_id"]
    ]
    if (
        step11_bundle.get("source") != campaign.get("source")
        or registry.get("source") != campaign.get("source")
        or configuration["step11_bundle_sha256"]
        != step11_bundle["content_hash"]
        or configuration["predictor_bundle_lock_sha256"]
        != step11_bundle["parents"]["predictor_bundle_lock"]
        or len(registered) != 1
        or any(
            registered[0][name] != configuration[name]
            for name in (
                "variant",
                "pipeline_seed",
                "final_particle_blocks",
                "semantic_label",
            )
        )
    ):
        raise ValueError("Stage-J registry materialization lineage differs")
    artifact = bind_source(
        materialize_stage_j_run(**configuration),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_materialized_stage_j_run(artifact)
    result = {
        "dry_run": args.dry_run,
        "run_id": artifact["run_id"],
        "run_sha256": artifact["content_hash"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
