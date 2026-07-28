#!/usr/bin/env python3
"""Run the deterministic joint seven-expert offline-loss beam selector."""

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
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.selection import select_joint_expert_losses
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
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(args.campaign_root, repo_root=REPO_ROOT)
    authorize_dataset_access(worker_role="design_worker", requested_resource="val_design")
    table = load_hashed_json(args.score_table)
    if table.get("contract") != "retb_joint_loss_score_table_v1":
        raise ValueError("joint loss score-table contract differs")
    if table.get("source") != campaign.get("source"):
        raise ValueError("joint loss score table belongs to another source")
    scores = {
        (row["readout"], tuple(row["loss_tuple"])): row
        for row in table["rows"]
    }
    def score(kind, loss_tuple, seed):
        del seed
        key = (kind, tuple(loss_tuple))
        if key not in scores:
            raise ValueError(f"joint loss score table lacks {key}")
        return scores[key]
    selection = select_joint_expert_losses(
        eligible_variants=table["eligible_variants"],
        individual_metrics=table["individual_metrics"],
        pooled_scorer=lambda values, seed: score("F_POOLED_MLP", values, seed),
        transformer_scorer=lambda values, seed: score("F_TOKEN_TRANSFORMER", values, seed),
        shape_id=table["shape_id"],
    )
    selection = bind_source(selection, source_snapshot=source_snapshot(REPO_ROOT))
    result = {
        "dry_run": bool(args.dry_run),
        "selected_tuple": selection["selected_tuple"],
        "selection": selection,
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, selection)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
