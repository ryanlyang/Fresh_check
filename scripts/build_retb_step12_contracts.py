#!/usr/bin/env python3
"""Build and publish RETB Step-12 final-consumer contracts."""

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
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    PREDICTOR_BUNDLE_LOCK_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    build_step12_bundle,
    publish_step12_bundle,
    validate_step12_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--predictor-bundle-lock", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step11 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step11_joint_bridge_bundle.json"
    )
    lock = load_hashed_json(
        args.predictor_bundle_lock,
        expected_contract=PREDICTOR_BUNDLE_LOCK_CONTRACT,
    )
    joint_lock = load_hashed_json(
        args.campaign_root
        / "selection"
        / "joint"
        / "joint_campaign_lock.json"
    )
    carried_locks = {}
    for role, row in joint_lock.get("carried_by_shape_role", {}).items():
        carried = load_hashed_json(
            args.campaign_root
            / "selection"
            / "predictor_bundle"
            / "carried"
            / f"{role}.json",
            expected_contract=PREDICTOR_BUNDLE_LOCK_CONTRACT,
        )
        if (
            carried["content_hash"]
            != row["predictor_bundle_lock_sha256"]
            or carried.get("source") != campaign.get("source")
        ):
            raise ValueError("Step-12 carried bundle lineage differs")
        carried_locks[str(role)] = carried["content_hash"]
    if (
        step11.get("source") != campaign.get("source")
        or lock.get("source") != campaign.get("source")
        or step11["parents"]["predictor_bundle_lock"]
        != lock["content_hash"]
        or joint_lock.get("source") != campaign.get("source")
    ):
        raise ValueError("Step-12 parent lineage differs")
    bundle = build_step12_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step11_bundle_sha256=step11["content_hash"],
        predictor_bundle_lock_sha256=lock["content_hash"],
        joint_campaign_lock_sha256=joint_lock["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=source_snapshot(REPO_ROOT),
        carried_predictor_bundle_locks=carried_locks,
    )
    digest = validate_step12_bundle(bundle)
    result = {"dry_run": args.dry_run, "step12_bundle_sha256": digest}
    if not args.dry_run:
        result["publication"] = publish_step12_bundle(
            campaign_root=args.campaign_root, bundle=bundle
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
