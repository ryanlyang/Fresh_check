#!/usr/bin/env python3
"""Write complete source-bound Stage-L JSON and Markdown reports."""

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
    CONFIRMATION_SUMMARY_CONTRACT,
    SCALE_SHORTLIST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_l_reporting import (  # noqa: E402
    build_stage_l_report,
    publish_stage_l_report,
)
from teacher_logit_reco.relation_expert_token_bridge.step13 import (  # noqa: E402
    STEP13_BUNDLE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--confirmation-summary", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--shortlisted-controls", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    confirmation = load_hashed_json(
        args.confirmation_summary,
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    controls = load_hashed_json(
        args.shortlisted_controls,
        expected_contract=(
            "retb_shortlisted_500k_controls_v3"
        ),
    )
    if any(
        row.get("source") != campaign.get("source")
        for row in (step13, confirmation, shortlist, controls)
    ):
        raise ValueError("Stage-L report source differs")
    artifact, markdown = build_stage_l_report(
        confirmation=confirmation,
        shortlist=shortlist,
        shortlisted_controls=controls,
        step13_bundle_sha256=step13["content_hash"],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": args.dry_run,
        "stage_l_report_sha256": artifact["content_hash"],
        "markdown_sha256": artifact["markdown_sha256"],
    }
    if not args.dry_run:
        result["publication"] = publish_stage_l_report(
            output_dir=args.output_dir,
            artifact=artifact,
            markdown=markdown,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
