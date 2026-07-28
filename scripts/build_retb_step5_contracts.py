#!/usr/bin/env python3
"""Build or publish immutable RETB Step-5 offline-fusion contracts."""

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
from teacher_logit_reco.relation_expert_token_bridge.step4 import validate_step4_bundle
from teacher_logit_reco.relation_expert_token_bridge.step5 import (
    build_step5_bundle,
    publish_step5_bundle,
    validate_step5_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    load_and_validate_campaign_source,
)


def _load_step4(root: Path) -> dict[str, dict[str, object]]:
    paths = {
        "expert_loss_registry": root / "registry" / "retb_expert_losses.json",
        "optimization_registry": root / "registry" / "retb_expert_optimization.json",
        "stage_b_run_registry": root / "registry" / "retb_stage_b_runs.json",
        "primary_training_protocol": root / "registry" / "retb_offline_expert_training.json",
        "step4_bundle": root / "registry" / "retb_step4_offline_expert_bundle.json",
        "step4_report": root / "reports" / "retb_step4_report.json",
    }
    return {name: load_hashed_json(path) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step4_sha = validate_step4_bundle(_load_step4(args.campaign_root))
    bundle = build_step5_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step4_bundle_sha256=step4_sha,
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    digest = validate_step5_bundle(bundle)
    result = {
        "dry_run": bool(args.dry_run),
        "campaign_root": str(args.campaign_root.resolve()),
        "step4_bundle_sha256": step4_sha,
        "step5_bundle_sha256": digest,
        "expert_confirmation_rows": 147,
        "canonical_fusion_rows": 21,
    }
    if not args.dry_run:
        result["publication"] = publish_step5_bundle(
            campaign_root=args.campaign_root, bundle=bundle
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
