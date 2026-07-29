#!/usr/bin/env python3
"""Build and publish RETB Step-11 joint-bridge contracts."""

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
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    PREDICTOR_BUNDLE_LOCK_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.step11 import (  # noqa: E402
    build_step11_bundle,
    publish_step11_bundle,
    validate_step11_bundle,
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
    step10 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step10_joint_predictor_bundle.json"
    )
    lock = load_hashed_json(
        args.predictor_bundle_lock,
        expected_contract=PREDICTOR_BUNDLE_LOCK_CONTRACT,
    )
    if (
        step10.get("source") != campaign.get("source")
        or lock.get("source") != campaign.get("source")
    ):
        raise ValueError("Step-11 parent source lineage differs")
    bundle = build_step11_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step10_bundle_sha256=step10["content_hash"],
        predictor_bundle_lock=lock,
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    digest = validate_step11_bundle(
        bundle, predictor_bundle_lock=lock
    )
    result = {"dry_run": args.dry_run, "step11_bundle_sha256": digest}
    if not args.dry_run:
        result["publication"] = publish_step11_bundle(
            campaign_root=args.campaign_root,
            bundle=bundle,
            predictor_bundle_lock=lock,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
