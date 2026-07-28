#!/usr/bin/env python3
"""Freeze three-seed Stage-E offline noninferiority and eligibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.bridge_certification import (  # noqa: E402
    build_bridge_candidate_eligibility,
    certify_offline_noninferiority,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--t0-metrics", required=True, type=Path)
    parser.add_argument("--candidate-registration", action="append", required=True)
    parser.add_argument("--content-certification", action="append", required=True)
    parser.add_argument("--output-noninferiority", required=True, type=Path)
    parser.add_argument("--output-eligibility", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.candidate_registration) != 3 or len(
        args.content_certification
    ) != 3:
        raise ValueError("bridge eligibility requires exact three-seed inputs")
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    candidate_metrics = load_hashed_json(args.candidate_metrics)
    t0_metrics = load_hashed_json(args.t0_metrics)
    registrations = [
        load_hashed_json(path) for path in args.candidate_registration
    ]
    content = [
        load_hashed_json(path) for path in args.content_certification
    ]
    if any(
        parent.get("source") != campaign.get("source")
        for parent in [candidate_metrics, t0_metrics, *content]
    ) or any(
        row.get("source") is not None
        and row.get("source") != campaign.get("source")
        for row in registrations
    ):
        raise ValueError("bridge noninferiority source lineage differs")
    modes = {registration.get("target_mode") for registration in registrations}
    experts = {registration.get("expert_id") for registration in registrations}
    shapes = {registration.get("shape_id") for registration in registrations}
    seeds = {int(registration.get("pipeline_seed", -1)) for registration in registrations}
    if len(modes) != 1 or len(experts) != 1 or len(shapes) != 1 or seeds != {
        101,
        202,
        303,
    }:
        raise ValueError("bridge registration three-seed identity differs")
    mode, expert, shape = modes.pop(), experts.pop(), shapes.pop()
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    noninferiority = bind_source(
        certify_offline_noninferiority(
            target_mode=mode,
            candidate_rows=candidate_metrics["rows"],
            t0_rows=t0_metrics["rows"],
            candidate_bundle_sha256=candidate_metrics["content_hash"],
            t0_bundle_sha256=t0_metrics["content_hash"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    eligibility = bind_source(
        build_bridge_candidate_eligibility(
            target_mode=mode,
            expert_id=expert,
            shape_id=shape,
            checkpoint_hashes_by_seed={
                int(row["pipeline_seed"]): row["checkpoint_sha256"]
                for row in registrations
            },
            noninferiority=noninferiority,
            content_certifications=content,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": bool(args.dry_run),
        "noninferiority": noninferiority,
        "eligibility": eligibility,
    }
    if not args.dry_run:
        result["publications"] = {
            "noninferiority": write_immutable_json(
                args.output_noninferiority, noninferiority
            ),
            "eligibility": write_immutable_json(
                args.output_eligibility, eligibility
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
