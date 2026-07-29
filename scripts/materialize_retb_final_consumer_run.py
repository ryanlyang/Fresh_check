#!/usr/bin/env python3
"""Materialize one immutable RETB Step-12 final-consumer run."""

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
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    STAGE_J_CONSUMER_REGISTRY_CONTRACT,
    materialize_final_consumer_run,
    validate_materialized_final_consumer_run,
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
    step12 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step12_final_consumers_bundle.json"
    )
    registry = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_final_consumer_registry.json",
        expected_contract=STAGE_J_CONSUMER_REGISTRY_CONTRACT,
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    if set(configuration) != {"run_id", "parent_hashes"}:
        raise ValueError("final-consumer materialization fields differ")
    rows = [
        row
        for row in registry["rows"]
        if row["run_id"] == configuration["run_id"]
    ]
    if (
        len(rows) != 1
        or step12.get("source") != campaign.get("source")
        or registry.get("source") != campaign.get("source")
        or registry["policy_sha256"]
        != step12["artifact_hashes"]["final_consumer_policy"]
    ):
        raise ValueError("final-consumer registry lineage differs")
    artifact = bind_source(
        materialize_final_consumer_run(
            registry_row=rows[0],
            step12_bundle_sha256=step12["content_hash"],
            parent_hashes=configuration["parent_hashes"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_materialized_final_consumer_run(artifact)
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
