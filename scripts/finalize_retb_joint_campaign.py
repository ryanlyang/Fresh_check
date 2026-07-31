#!/usr/bin/env python3
"""Authenticate complete J1--J5 coverage and lock Stage-J handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    JOINT_REGISTRATION_CONTRACT,
    STAGE_J_METRICS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    STAGE_E_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


JOINT_CAMPAIGN_LOCK_CONTRACT = "retb_joint_campaign_lock_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    primary_bundle = load_hashed_json(
        args.campaign_root
        / "selection"
        / "predictor_bundle"
        / "predictor_bundle_lock.json"
    )
    primary_role = str(primary_bundle["coordinate_id"]).split(":", 1)[1]
    if primary_role not in STAGE_E_SHAPES:
        raise ValueError("primary joint carried-shape role differs")
    records: list[dict[str, Any]] = []
    carried = {}
    for role in STAGE_E_SHAPES:
        bundle = load_hashed_json(
            args.campaign_root
            / "selection"
            / "predictor_bundle"
            / "carried"
            / f"{role}.json"
        )
        selection = load_hashed_json(
            args.campaign_root
            / "selection"
            / "joint"
            / role
            / "j4_blocks.json"
        )
        selected_blocks = int(selection["selected_final_particle_blocks"])
        role_records = []
        for variant in (
            "J1_SHARED_CONTEXT",
            "J2_COUPLED_DECODER",
            "J3_INDEPENDENT_PLUS_ADAPTER",
            "J4_BRIDGE_FINETUNE",
            "J5_END_TO_END",
        ):
            block_values = (
                (2, 4) if variant == "J4_BRIDGE_FINETUNE" else (None,)
            )
            for blocks in block_values:
                for seed in (101, 202, 303):
                    suffix = "" if blocks is None else f"_N{blocks}"
                    root = (
                        args.campaign_root
                        / "runs"
                        / "joint"
                        / role
                        / f"RETB_{variant}_S{seed}{suffix}"
                    )
                    registration = load_hashed_json(
                        root / "registration.json",
                        expected_contract=JOINT_REGISTRATION_CONTRACT,
                    )
                    metrics = load_hashed_json(
                        root / "val_design" / "metrics.json",
                        expected_contract=STAGE_J_METRICS_CONTRACT,
                    )
                    checkpoint = root / "best_model_val.pt"
                    if (
                        registration.get("source") != campaign.get("source")
                        or metrics.get("source") != campaign.get("source")
                        or registration["checkpoint_sha256"]
                        != _sha256(checkpoint)
                        or metrics["registration_sha256"]
                        != registration["content_hash"]
                        or metrics["variant"] != variant
                        or int(metrics["pipeline_seed"]) != seed
                        or metrics["final_particle_blocks"] != blocks
                        or registration["predictor_bundle_lock_sha256"]
                        != bundle["content_hash"]
                    ):
                        raise ValueError(
                            "joint campaign result lineage differs"
                        )
                    record = {
                        "carried_shape_role": role,
                        "variant": variant,
                        "pipeline_seed": seed,
                        "final_particle_blocks": blocks,
                        "registration_sha256": registration["content_hash"],
                        "checkpoint_sha256": registration[
                            "checkpoint_sha256"
                        ],
                        "prediction_manifest_sha256": metrics[
                            "prediction_manifest_sha256"
                        ],
                        "metrics_sha256": metrics["content_hash"],
                        "output_root": str(root),
                    }
                    role_records.append(record)
                    records.append(record)
        selected = {
            str(seed): next(
                row
                for row in role_records
                if row["variant"] == "J5_END_TO_END"
                and row["pipeline_seed"] == seed
            )
            for seed in (101, 202, 303)
        }
        selected_j4 = {
            str(seed): next(
                row
                for row in role_records
                if row["variant"] == "J4_BRIDGE_FINETUNE"
                and row["pipeline_seed"] == seed
                and row["final_particle_blocks"] == selected_blocks
            )
            for seed in (101, 202, 303)
        }
        carried[role] = {
            "predictor_bundle_lock_sha256": bundle["content_hash"],
            "j4_block_selection_sha256": selection["content_hash"],
            "selected_final_particle_blocks": selected_blocks,
            "selected_j4_by_seed": selected_j4,
            "selected_j5_by_seed": selected,
        }
    expected_count = len(STAGE_E_SHAPES) * 3 * (3 + 2 + 1)
    if len(records) != expected_count:
        raise RuntimeError("joint campaign coverage differs")
    primary = carried[primary_role]
    lock = bind_source(
        with_content_hash(
            {
                "contract": JOINT_CAMPAIGN_LOCK_CONTRACT,
                "schema_version": 2,
                "primary_carried_shape_role": primary_role,
                "j4_block_selection_sha256": primary[
                    "j4_block_selection_sha256"
                ],
                "selected_final_particle_blocks": primary[
                    "selected_final_particle_blocks"
                ],
                "complete_candidate_records": records,
                "selected_j4_by_seed": primary["selected_j4_by_seed"],
                "selected_j5_by_seed": primary["selected_j5_by_seed"],
                "carried_by_shape_role": carried,
                "all_five_carried_shapes_complete": True,
                "all_predeclared_candidates_complete": True,
                "scientific_underperformance_blocks_continuation": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
