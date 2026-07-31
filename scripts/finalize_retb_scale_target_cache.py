#!/usr/bin/env python3
"""Seal one Stage-M target cache after its scale fusion is immutable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_retb_scale_target_coordinates import (  # noqa: E402
    CONFIGURATION_CONTRACT,
    INDEX_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (  # noqa: E402
    load_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    build_target_cache_specification,
    identity_order_sha256,
    load_frozen_token_head_reproducer,
    publish_offline_target_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _descriptor(
    *,
    expert: str,
    mode: str,
    target: Mapping[str, Any],
    locked: Mapping[str, Any],
) -> dict[str, Any]:
    source = dict(locked["descriptor"])
    row = {
        "checkpoint_sha256": target["sha256"],
        "registration_sha256": target["registration_sha256"],
        "slot_query_sha256": target["slot_query_sha256"],
        "eligibility_sha256": source["eligibility_sha256"],
    }
    if mode != "T0_PURE":
        row.update(
            {
                "pilot_checkpoint_sha256": source[
                    "pilot_checkpoint_sha256"
                ],
                "content_certification_sha256": source[
                    "content_certification_sha256"
                ],
                "noninferiority_sha256": source[
                    "noninferiority_sha256"
                ],
            }
        )
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--coordinate-index", required=True, type=Path)
    parser.add_argument("--coordinate-cache", required=True, type=Path)
    parser.add_argument("--fusion-registration", required=True, type=Path)
    parser.add_argument("--fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=2048)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    config = load_hashed_json(
        args.configuration, expected_contract=CONFIGURATION_CONTRACT
    )
    coordinate_index = load_hashed_json(
        args.coordinate_index, expected_contract=INDEX_CONTRACT
    )
    coordinate_manifest, arrays = load_frozen_token_cache(
        args.coordinate_cache
    )
    fusion = load_hashed_json(args.fusion_registration)
    split = str(coordinate_index["split"])
    if (
        any(
            row.get("source") != campaign.get("source")
            for row in (
                config,
                coordinate_index,
                coordinate_manifest,
                fusion,
            )
        )
        or coordinate_index["configuration_sha256"]
        != config["content_hash"]
        or coordinate_index["coordinate_cache_sha256"]
        != coordinate_manifest["content_hash"]
        or fusion.get("checkpoint_sha256")
        is None
        or fusion["checkpoint_sha256"]
        != __import__("hashlib").sha256(
            args.fusion_checkpoint.read_bytes()
        ).hexdigest()
    ):
        raise ValueError("scale target-cache sealing lineage differs")
    authorize_dataset_access(
        worker_role=(
            "scale_training_worker"
            if split == "scale_train"
            else "training_worker"
            if split == "val_stop"
            else "design_worker"
            if split == "val_design"
            else "postlock_stack_diagnostic"
            if split == "stack_val"
            else "final_test_worker"
        ),
        requested_resource=(
            "stack_val_oracle_targets"
            if split == "stack_val"
            else "final_test_targets"
            if split == "final_test"
            else split
        ),
    )
    identities = [str(value) for value in arrays["identities"].tolist()]
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    snapshot = source_snapshot(REPO_ROOT)
    identity = bind_source(
        with_content_hash(
            {
                "contract": "retb_scale_target_identity_manifest_v1",
                "schema_version": 1,
                "split": split,
                "pipeline_seed": int(config["pipeline_seed"]),
                "shape_role": config["shape_role"],
                "event_count": len(identities),
                "identity_order_sha256": identity_order_sha256(
                    identities, labels
                ),
                "parent_identity_manifest_sha256": (
                    coordinate_manifest["identity_manifest_sha256"]
                ),
            }
        ),
        source_snapshot=snapshot,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_immutable_json(
        args.output_dir / "identity_manifest.json", identity
    )
    target_descriptors = {
        expert: _descriptor(
            expert=expert,
            mode=config["target_modes"][expert],
            target=coordinate_index["target_checkpoints"][expert],
            locked=config["locked_targets"][expert],
        )
        for expert in EXPERT_ORDER
    }
    lineage = bind_source(
        with_content_hash(
            {
                "contract": "retb_scale_selected_target_lineage_v1",
                "schema_version": 1,
                "pipeline_seed": int(config["pipeline_seed"]),
                "shape_role": config["shape_role"],
                "target_modes": dict(config["target_modes"]),
                "scale_target_coordinate_configuration_sha256": config[
                    "content_hash"
                ],
                "coordinate_index_sha256": coordinate_index[
                    "content_hash"
                ],
                "target_descriptors": target_descriptors,
                "coordinate_only_weights_reselected": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(args.output_dir / "target_lineage.json", lineage)
    specification = bind_source(
        build_target_cache_specification(
            split=split,
            pipeline_seed=int(config["pipeline_seed"]),
            shape_id=str(config["shape_role"]),
            allocation=coordinate_manifest["allocation"],
            target_tuple=[
                config["target_modes"][expert] for expert in EXPERT_ORDER
            ],
            target_descriptors=target_descriptors,
            selected_target_lineage_sha256=lineage["content_hash"],
            target_cache_namespace=str(config["target_cache_namespace"]),
            locked_coordinate_contract_sha256=config[
                "coordinate_contract_sha256"
            ],
            locked_coordinate_selection_sha256=config[
                "coordinate_selection_sha256"
            ],
            offline_fusion_checkpoint_sha256=fusion[
                "checkpoint_sha256"
            ],
            offline_fusion_registration_sha256=fusion["content_hash"],
            normalizer_set_sha256=config[
                "locked_coordinate_normalizer_set_sha256"
            ],
            identity_manifest_sha256=identity["content_hash"],
            identity_order_sha256=identity["identity_order_sha256"],
            event_count=len(identities),
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(
        args.output_dir / "target_cache_specification.json",
        specification,
    )
    reproducers = {
        expert: load_frozen_token_head_reproducer(
            checkpoint_path=coordinate_index["target_checkpoints"][expert][
                "path"
            ],
            expected_checkpoint_sha256=coordinate_index[
                "target_checkpoints"
            ][expert]["sha256"],
            target_mode=config["target_modes"][expert],
            token_dimension=int(
                coordinate_manifest["allocation"][expert][1]
            ),
        )
        for expert in EXPERT_ORDER
    }

    def generate(start: int, stop: int) -> dict[str, Any]:
        return {
            "tokens": {
                expert: arrays["token_banks"][expert][start:stop]
                for expert in EXPERT_ORDER
            },
            "expert_logits": {
                expert: arrays["expert_logits"][expert][start:stop]
                for expert in EXPERT_ORDER
            },
        }

    manifest = publish_offline_target_cache(
        output_dir=args.output_dir,
        specification=specification,
        identities=identities,
        labels=labels,
        generator=generate,
        logit_reproducers=reproducers,
        source_snapshot=snapshot,
        shard_size=args.shard_size,
    )
    print(
        json.dumps(
            {
                "target_cache_manifest_sha256": manifest["content_hash"],
                "split": split,
                "event_count": len(identities),
                "performance_based_termination": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
