#!/usr/bin/env python3
"""Select O/H monolithic controls against an exact RETB exported graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.capacity import (  # noqa: E402
    select_monolithic_capacity_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    COMPLETE_GRAPH_CAPACITY_CONTRACT,
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
    parser.add_argument("--capacity-artifact", required=True, type=Path)
    parser.add_argument("--candidate-grid", required=True, type=Path)
    parser.add_argument("--domain", choices=("offline", "hlt"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    capacity = load_hashed_json(
        args.capacity_artifact,
        expected_contract=COMPLETE_GRAPH_CAPACITY_CONTRACT,
    )
    candidates = json.loads(args.candidate_grid.read_text("utf-8"))
    if (
        capacity.get("source") != campaign.get("source")
        or not isinstance(candidates, list)
    ):
        raise ValueError("complete-graph capacity selector lineage differs")
    totals = capacity["totals"]
    selection = bind_source(
        select_monolithic_capacity_controls(
            target_parameters=totals["parameter_count"],
            target_flops_batch1=totals[
                "analytical_inference_flops_batch1"
            ],
            target_flops_batch128=totals[
                "analytical_inference_flops_batch128"
            ],
            candidates=candidates,
            domain=args.domain,
            target_complete_graph_sha256=capacity["content_hash"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(args.output, selection)
    print(
        json.dumps(
            {
                "selection_sha256": selection["content_hash"],
                "domain": args.domain,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
