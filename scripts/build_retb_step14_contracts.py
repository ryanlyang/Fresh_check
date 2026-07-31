#!/usr/bin/env python3
"""Build source-bound RETB Step-14 Stage-M/N contracts."""

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
    SHORTLISTED_CONTROLS_CONTRACT,
    validate_shortlisted_500k_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.determinism import (  # noqa: E402
    GLOBAL_DETERMINISM_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step13 import (  # noqa: E402
    STEP13_BUNDLE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.step14 import (  # noqa: E402
    build_step14_bundle,
    publish_step14_bundle,
    validate_step14_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step13 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step13_confirmation_shortlist_bundle.json",
        expected_contract=STEP13_BUNDLE_CONTRACT,
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    controls = load_hashed_json(
        args.campaign_root
        / "selection"
        / "stage_l"
        / "shortlisted_500k_controls.json",
        expected_contract=SHORTLISTED_CONTROLS_CONTRACT,
    )
    validate_shortlisted_500k_controls(
        controls, locked_scale_shortlist=shortlist
    )
    determinism = load_hashed_json(
        args.campaign_root / "registry" / "global_determinism.json",
        expected_contract=GLOBAL_DETERMINISM_CONTRACT,
    )
    if any(
        row.get("source") != campaign.get("source")
        for row in (step13, shortlist, controls, determinism)
    ):
        raise ValueError("Step-14 contract parent source differs")
    bundle = build_step14_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step13_bundle_sha256=step13["content_hash"],
        locked_scale_shortlist_sha256=shortlist["content_hash"],
        shortlisted_500k_controls_sha256=controls["content_hash"],
        global_determinism_sha256=determinism["content_hash"],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_step14_bundle(bundle)
    result = {
        "dry_run": args.dry_run,
        "step14_bundle_sha256": bundle["step14_bundle"]["content_hash"],
    }
    if not args.dry_run:
        result["publication"] = publish_step14_bundle(
            campaign_root=args.campaign_root, bundle=bundle
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
