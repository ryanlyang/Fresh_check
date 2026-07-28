#!/usr/bin/env python3
"""Freeze the three-seed T0_PURE eligibility parent for Stage-E selection."""

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
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--t0-registration", action="append", required=True)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--expert-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.t0_registration) != 3:
        raise ValueError("T0 eligibility requires exactly three registrations")
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registrations = [
        load_hashed_json(path) for path in args.t0_registration
    ]
    if any(
        (
            row.get("source") is not None
            and row.get("source") != campaign.get("source")
        )
        or row.get("expert_id") != args.expert_id
        for row in registrations
    ):
        raise ValueError("T0 eligibility registration lineage differs")
    by_seed = {
        int(row["pipeline_seed"]): row["checkpoint_sha256"]
        for row in registrations
    }
    eligibility = bind_source(
        build_bridge_candidate_eligibility(
            target_mode="T0_PURE",
            expert_id=args.expert_id,
            shape_id=args.shape_id,
            checkpoint_hashes_by_seed=by_seed,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": bool(args.dry_run),
        "eligibility": eligibility,
        "output": str(args.output.resolve()),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, eligibility)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
