#!/usr/bin/env python3
"""Build and publish RETB Step-7 bridge-target contracts."""

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
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    build_step7_bundle,
    publish_step7_bundle,
    validate_step7_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step6 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step6_native_hlt_bundle.json"
    )
    if step6.get("source") != campaign.get("source"):
        raise ValueError("Step-7 parent source lineage differs")
    bundle = build_step7_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step6_bundle_sha256=step6["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    digest = validate_step7_bundle(bundle)
    result = {
        "dry_run": bool(args.dry_run),
        "step7_bundle_sha256": digest,
        "pilot_membership_count": bundle["stage_e_templates"][
            "pilot_membership_count"
        ],
        "candidate_membership_count": bundle["stage_e_templates"][
            "candidate_membership_count"
        ],
    }
    if not args.dry_run:
        result["publication"] = publish_step7_bundle(
            campaign_root=args.campaign_root, bundle=bundle
        )
        from scripts.materialize_retb_stage_e_parents import (
            main as materialize_stage_e_parents,
        )

        materialize_stage_e_parents(
            ["--campaign-root", str(args.campaign_root)]
        )
        result["stage_e_parent_bundle_count"] = 105
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
