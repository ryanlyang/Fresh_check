#!/usr/bin/env python3
"""Run the deterministic joint Stage-E target-coordinate beam selector."""

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
    validate_bridge_candidate_eligibility,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_selection import (  # noqa: E402
    select_joint_bridge_coordinates,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _eligibility_paths(rows: Sequence[str]) -> dict[tuple[str, str], Path]:
    output = {}
    for row in rows:
        if "=" not in row or ":" not in row.split("=", 1)[0]:
            raise ValueError("--eligibility requires EXPERT:MODE=JSON")
        identity, path = row.split("=", 1)
        expert, mode = identity.split(":", 1)
        key = (expert, mode)
        if expert not in EXPERT_ORDER or key in output:
            raise ValueError("bridge eligibility key differs")
        output[key] = Path(path)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--score-table", required=True, type=Path)
    parser.add_argument("--eligibility", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    table = load_hashed_json(args.score_table)
    if (
        table.get("contract") != "retb_joint_bridge_coordinate_score_table_v1"
        or table.get("source") != campaign.get("source")
    ):
        raise ValueError("bridge coordinate score table lineage differs")
    paths = _eligibility_paths(args.eligibility)
    expected = {
        (expert, mode)
        for expert in EXPERT_ORDER
        for mode in table["eligible_modes"][expert]
    }
    if set(paths) != expected:
        raise ValueError("bridge selector eligibility coverage differs")
    eligibility_artifacts = {
        key: load_hashed_json(path) for key, path in paths.items()
    }
    eligibility_hashes = {expert: {} for expert in EXPERT_ORDER}
    for (expert, mode), artifact in eligibility_artifacts.items():
        digest = validate_bridge_candidate_eligibility(artifact)
        if (
            artifact.get("source") != campaign.get("source")
            or artifact["expert_id"] != expert
            or artifact["target_mode"] != mode
            or artifact["shape_id"] != table["shape_id"]
            or not artifact["maximum_performance_eligible"]
        ):
            raise ValueError("bridge selector eligibility artifact differs")
        eligibility_hashes[expert][mode] = digest
    scores = {
        (row["readout"], tuple(row["target_tuple"])): row
        for row in table["rows"]
    }
    if len(scores) != len(table["rows"]):
        raise ValueError("bridge coordinate score table duplicates a tuple")

    def score(readout: str, target_tuple: tuple[str, ...], seed: int):
        if seed != 41703:
            raise ValueError("bridge readout seed differs")
        key = (readout, tuple(target_tuple))
        if key not in scores:
            raise ValueError(f"bridge coordinate score table lacks {key}")
        return scores[key]

    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    selection = bind_source(
        select_joint_bridge_coordinates(
            eligible_modes=table["eligible_modes"],
            default_metrics=table["individual_metrics"],
            pooled_scorer=lambda values, seed: score(
                "F_POOLED_MLP", values, seed
            ),
            transformer_scorer=lambda values, seed: score(
                "F_TOKEN_TRANSFORMER", values, seed
            ),
            shape_id=table["shape_id"],
            eligibility_hashes=eligibility_hashes,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": bool(args.dry_run),
        "selection": selection,
        "output": str(args.output.resolve()),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, selection)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
