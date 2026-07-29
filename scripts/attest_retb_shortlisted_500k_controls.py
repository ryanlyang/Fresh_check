#!/usr/bin/env python3
"""Attest post-shortlist capacity and label-exposure controls."""

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
    build_shortlisted_500k_controls,
    validate_shortlisted_500k_controls,
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
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--control-rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    rows = json.loads(args.control_rows.read_text("utf-8"))
    if (
        shortlist.get("source") != campaign.get("source")
        or not isinstance(rows, list)
    ):
        raise ValueError("shortlisted controls source/input differs")
    artifact = bind_source(
        build_shortlisted_500k_controls(
            locked_scale_shortlist=shortlist, rows=rows
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_shortlisted_500k_controls(
        artifact, locked_scale_shortlist=shortlist
    )
    result = {
        "dry_run": args.dry_run,
        "shortlisted_controls_sha256": artifact["content_hash"],
        "graph_count": len(artifact["rows"]),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
