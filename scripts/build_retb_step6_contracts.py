#!/usr/bin/env python3
"""Build and publish RETB Step-6 native-HLT contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.step6 import (
    build_step6_bundle,
    publish_step6_bundle,
    validate_step6_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--shared-hlt-normalizer", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step5 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step5_offline_fusion_bundle.json"
    )
    normalizer = load_hashed_json(args.shared_hlt_normalizer)
    if (
        step5.get("source") != campaign.get("source")
        or normalizer.get("source") != campaign.get("source")
    ):
        raise ValueError("Step-6 parent source lineage differs")
    bundle = build_step6_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step5_bundle_sha256=step5["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        hlt_replica_manifest_sha256=campaign["parent_artifact_hashes"][
            "hlt_replica_manifest"
        ],
        shared_hlt_normalizer_sha256=normalizer["content_hash"],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    digest = validate_step6_bundle(bundle)
    result = {
        "dry_run": bool(args.dry_run),
        "step6_bundle_sha256": digest,
        "row_counts": bundle["stage_d_run_registry"]["row_counts"],
    }
    if not args.dry_run:
        result["publication"] = publish_step6_bundle(
            campaign_root=args.campaign_root, bundle=bundle
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
