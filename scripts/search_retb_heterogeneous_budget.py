#!/usr/bin/env python3
"""Select RETB greedy and beam heterogeneous 56-slot allocations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.dynamic_continuation import (
    add_dynamic_continuation_arguments,
    resolve_selector_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.selection import select_heterogeneous_allocations
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--score-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    add_dynamic_continuation_arguments(parser)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(args.campaign_root, repo_root=REPO_ROOT)
    authorize_dataset_access(worker_role="design_worker", requested_resource="val_design")
    table = load_hashed_json(args.score_table)
    if table.get("contract") != "retb_heterogeneous_score_table_v1":
        raise ValueError("heterogeneous score-table contract differs")
    if table.get("source") != campaign.get("source"):
        raise ValueError("heterogeneous score table belongs to another source")
    scores = {
        (
            row["readout"],
            int(row["seed"]),
            tuple(row["allocation"][name] for name in EXPERT_ORDER),
        ): row
        for row in table["rows"]
    }
    def score(kind, allocation, seed):
        key = (
            kind,
            int(seed),
            tuple(allocation[name] for name in EXPERT_ORDER),
        )
        if key not in scores:
            raise ValueError(f"heterogeneous score table lacks {key}")
        return scores[key]
    selection = select_heterogeneous_allocations(
        greedy_scorer=lambda values, seed: score("GREEDY_POOLED", values, seed),
        beam_pooled_scorer=lambda values, seed: score("BEAM_POOLED", values, seed),
        beam_transformer_scorer=lambda values, seed: score("F_TOKEN_TRANSFORMER", values, seed),
    )
    selection = bind_source(selection, source_snapshot=source_snapshot(REPO_ROOT))
    result = {
        "dry_run": bool(args.dry_run),
        "HET_SELECTED": selection["HET_SELECTED"]["allocation"],
        "HET_BEAM": selection["HET_BEAM"]["allocation"],
        "selection": selection,
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, selection)
    continuation = resolve_selector_continuation(
        args=args,
        campaign=campaign,
        campaign_root=args.campaign_root,
        selector_output=selection,
        selector_output_path=args.output,
        load_hashed_json=load_hashed_json,
        dry_run=bool(args.dry_run),
    )
    if continuation is not None:
        result["continuation"] = continuation
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
