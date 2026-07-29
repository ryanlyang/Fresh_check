#!/usr/bin/env python3
"""Build and publish RETB Step-13 confirmation/shortlist contracts."""

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
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    STEP12_BUNDLE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.step13 import (  # noqa: E402
    build_step13_bundle,
    publish_step13_bundle,
    validate_step13_bundle,
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
    step12 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step12_final_consumers_bundle.json",
        expected_contract=STEP12_BUNDLE_CONTRACT,
    )
    if step12.get("source") != campaign.get("source"):
        raise ValueError("Step-13 parent source differs")
    bundle = build_step13_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step12_bundle_sha256=step12["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    digest = validate_step13_bundle(bundle)
    result = {"dry_run": args.dry_run, "step13_bundle_sha256": digest}
    if not args.dry_run:
        result["publication"] = publish_step13_bundle(
            campaign_root=args.campaign_root, bundle=bundle
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
