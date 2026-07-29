#!/usr/bin/env python3
"""Aggregate every and only locked Stage-M graph/seed run."""

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
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    SCALE_GRAPH_RUN_CONTRACT,
    aggregate_scale_completion,
    validate_scale_completion,
)
from teacher_logit_reco.relation_expert_token_bridge.step14 import (  # noqa: E402
    STEP14_BUNDLE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--scale-run", action="append", required=True, type=Path)
    parser.add_argument("--scale-train-manifest-sha256", required=True)
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
    step14 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step14_scale_final_seal_bundle.json",
        expected_contract=STEP14_BUNDLE_CONTRACT,
    )
    runs = [
        load_hashed_json(path, expected_contract=SCALE_GRAPH_RUN_CONTRACT)
        for path in args.scale_run
    ]
    if any(
        row.get("source") != campaign.get("source")
        for row in (shortlist, step14, *runs)
    ):
        raise ValueError("scale completion source differs")
    artifact = bind_source(
        aggregate_scale_completion(
            locked_scale_shortlist=shortlist,
            scale_graph_runs=runs,
            step14_bundle_sha256=step14["content_hash"],
            scale_train_manifest_sha256=args.scale_train_manifest_sha256,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_scale_completion(
        artifact,
        locked_scale_shortlist=shortlist,
        scale_graph_runs=runs,
    )
    result = {
        "dry_run": args.dry_run,
        "scale_completion_sha256": artifact["content_hash"],
        "run_count": artifact["expected_run_count"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
