#!/usr/bin/env python3
"""Replay and publish the complete Stage-E target-coordinate selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    root = args.campaign_root / "selection" / "stage_e"
    index = load_hashed_json(root / "bridge_certification_index.json")
    table = load_hashed_json(root / "bridge_coordinate_score_table.json")
    if (
        index.get("contract") != "retb_bridge_certification_index_v3"
        or table.get("contract")
        != "retb_joint_bridge_coordinate_score_table_v3"
        or index["bridge_coordinate_score_table_sha256"]
        != table["content_hash"]
        or index.get("source") != campaign.get("source")
        or table.get("source") != campaign.get("source")
    ):
        raise ValueError("target-coordinate selector lineage differs")
    eligibility_hashes = {expert: {} for expert in EXPERT_ORDER}
    for key, path in index["eligibility_paths"].items():
        expert, mode = key.split(":", 1)
        artifact = load_hashed_json(path)
        digest = validate_bridge_candidate_eligibility(artifact)
        if (
            artifact["expert_id"] != expert
            or artifact["target_mode"] != mode
            or artifact["shape_id"] != table["shape_id"]
            or not artifact["maximum_performance_eligible"]
            or artifact.get("source") != campaign.get("source")
        ):
            raise ValueError("target-coordinate eligibility differs")
        eligibility_hashes[expert][mode] = digest
    if any(
        set(eligibility_hashes[expert])
        != set(table["eligible_modes"][expert])
        or "T0_PURE" not in eligibility_hashes[expert]
        for expert in EXPERT_ORDER
    ):
        raise ValueError("target-coordinate eligibility coverage differs")
    scores: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in table["rows"]:
        key = (str(row["readout"]), tuple(row["target_tuple"]))
        if key in scores:
            raise ValueError("target-coordinate score is duplicated")
        scores[key] = dict(row)

    def scorer(
        readout: str, values: tuple[str, ...], seed: int
    ) -> dict[str, Any]:
        if seed != 41703:
            raise ValueError("target-coordinate readout seed differs")
        try:
            return scores[(readout, tuple(values))]
        except KeyError as exc:
            raise ValueError(
                f"target-coordinate score table lacks {readout}:{values}"
            ) from exc

    selection = bind_source(
        select_joint_bridge_coordinates(
            eligible_modes=table["eligible_modes"],
            default_metrics=table["individual_metrics"],
            pooled_scorer=lambda values, seed: scorer(
                "F_POOLED_MLP", values, seed
            ),
            transformer_scorer=lambda values, seed: scorer(
                "F_TOKEN_TRANSFORMER", values, seed
            ),
            shape_id=table["shape_id"],
            eligibility_hashes=eligibility_hashes,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    # Selection is emitted for every complete table, even if every bridge
    # failed scientifically; T0 is the immutable all-expert fallback.
    if not selection["selected_target_tuple"]:
        raise RuntimeError("target-coordinate selector emitted no fallback")
    write_immutable_json(args.output, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
