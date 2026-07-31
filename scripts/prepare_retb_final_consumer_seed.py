#!/usr/bin/env python3
"""Build the shared authenticated Step-12 dataset for one pipeline seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    canonical_sha256,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    FinalConsumerDataset,
    publish_final_consumer_dataset,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    load_joint_dataset_cache,
    load_joint_graph_template,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_losses import (  # noqa: E402
    apply_uncertainty_calibration,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.replicas import (  # noqa: E402
    replica_for,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    SCALE_COMPONENT_INDEX_CONTRACT,
    SCALE_JOINT_COMPLETION_CONTRACT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calibrations(
    root: Path,
    lock: Mapping[str, Any],
    seed: int,
    scale_index: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    if scale_index is not None:
        rows = {
            expert: load_hashed_json(
                scale_index["uncertainty_calibrations"][expert]
            )
            for expert in EXPERT_ORDER
        }
        return rows, canonical_sha256(
            {expert: row["content_hash"] for expert, row in rows.items()}
        )
    configuration = json.loads(
        (
            root
            / "selection"
            / "predictor_bundle"
            / "inputs"
            / "selector_configuration.json"
        ).read_text("utf-8")
    )
    rows = {}
    for expert in EXPERT_ORDER:
        candidate = lock["selected_candidate_descriptors"][expert][
            "candidate_id"
        ]
        values = configuration["calibration_artifact_paths"][candidate]
        path = Path(values.get(str(seed), values.get(seed)))
        row = load_hashed_json(path)
        if (
            row["content_hash"]
            != lock["seed_specific_artifacts"][str(seed)][expert][
                "uncertainty_calibration"
            ]
        ):
            raise ValueError("final-consumer uncertainty lineage differs")
        rows[expert] = row
    return rows, canonical_sha256(
        {expert: row["content_hash"] for expert, row in rows.items()}
    )


def _evaluate_replica(
    graph: Any,
    dataset: Any,
    *,
    replica: int,
    device: torch.device,
    calibrations: Mapping[str, Any],
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    predicted = {expert: [] for expert in EXPERT_ORDER}
    uncertainty = {expert: [] for expert in EXPERT_ORDER}
    native = {expert: [] for expert in EXPERT_ORDER}
    native_logits = {expert: [] for expert in EXPERT_ORDER}
    predicted_logits = {expert: [] for expert in EXPERT_ORDER}
    graph.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            raw = dataset.shared_raw_view
            shared = {
                "identities": list(dataset.identities[start:stop]),
                "replica_ids": torch.full(
                    (stop - start,),
                    replica,
                    dtype=torch.int64,
                    device=device,
                ),
                "degraded_view_hashes": list(
                    dataset.degraded_view_hashes_by_replica[replica][
                        start:stop
                    ]
                ),
                "features": torch.from_numpy(
                    raw["features"][replica][start:stop]
                ).to(device),
                "vectors": torch.from_numpy(
                    raw["vectors"][replica][start:stop]
                ).to(device),
                "mask": torch.from_numpy(
                    raw["mask"][replica][start:stop]
                ).to(device),
                "raw_tokens": torch.from_numpy(
                    raw["raw_tokens"][replica][start:stop]
                ).to(device),
                "region_trees_by_expert": {
                    expert: list(
                        raw["region_trees_by_expert"][expert][replica][
                            start:stop
                        ]
                    )
                    for expert in EXPERT_ORDER
                },
            }
            evidence = graph._live_evidence(shared)
            output = graph(evidence=evidence)
            for expert in EXPERT_ORDER:
                predicted[expert].append(
                    output["predicted_tokens"][expert]
                    .float()
                    .cpu()
                    .numpy()
                )
                calibrated = apply_uncertainty_calibration(
                    output["log_variance"][expert],
                    calibrations[expert],
                )
                uncertainty[expert].append(
                    calibrated.float().cpu().numpy()
                )
                native[expert].append(
                    evidence["hlt_token_banks"][expert]
                    .float()
                    .cpu()
                    .numpy()
                )
                native_logits[expert].append(
                    evidence["native_hlt_logits"][expert]
                    .float()
                    .cpu()
                    .numpy()
                )
                predicted_logits[expert].append(
                    output["predicted_expert_logits"][expert]
                    .float()
                    .cpu()
                    .numpy()
                )
    return {
        "predicted": {
            expert: np.concatenate(predicted[expert])
            for expert in EXPERT_ORDER
        },
        "uncertainty": {
            expert: np.concatenate(uncertainty[expert])
            for expert in EXPERT_ORDER
        },
        "native": {
            expert: np.concatenate(native[expert])
            for expert in EXPERT_ORDER
        },
        "native_logits": {
            expert: np.concatenate(native_logits[expert])
            for expert in EXPERT_ORDER
        },
        "predicted_logits": {
            expert: np.concatenate(predicted_logits[expert])
            for expert in EXPERT_ORDER
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--carried-shape-role", required=True)
    parser.add_argument("--scale-joint-root", type=Path)
    parser.add_argument("--scale-component-index", type=Path)
    parser.add_argument("--scale-joint-completion", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    lock = load_hashed_json(
        args.campaign_root
        / "selection"
        / "predictor_bundle"
        / "carried"
        / f"{args.carried_shape_role}.json"
    )
    joint_lock = load_hashed_json(
        args.campaign_root
        / "selection"
        / "joint"
        / "joint_campaign_lock.json"
    )
    selected = joint_lock["carried_by_shape_role"][
        args.carried_shape_role
    ]["selected_j5_by_seed"][str(args.pipeline_seed)]
    scale_index = (
        None
        if args.scale_component_index is None
        else load_hashed_json(args.scale_component_index)
    )
    if scale_index is not None and (
        scale_index.get("contract") != SCALE_COMPONENT_INDEX_CONTRACT
        or scale_index.get("source") != campaign.get("source")
        or int(scale_index.get("pipeline_seed", -1))
        != args.pipeline_seed
    ):
        raise ValueError("scale final-consumer component lineage differs")
    joint_completion = (
        None
        if args.scale_joint_completion is None
        else load_hashed_json(args.scale_joint_completion)
    )
    if any(
        value is None
        for value in (
            args.scale_joint_root,
            scale_index,
            joint_completion,
        )
    ) != all(
        value is None
        for value in (
            args.scale_joint_root,
            scale_index,
            joint_completion,
        )
    ):
        raise ValueError("scale final-consumer inputs must be provided together")
    if joint_completion is not None and (
        joint_completion.get("contract") != SCALE_JOINT_COMPLETION_CONTRACT
        or joint_completion.get("source") != campaign.get("source")
        or int(joint_completion.get("pipeline_seed", -1))
        != args.pipeline_seed
        or joint_completion.get("scale_component_index_sha256")
        != scale_index["content_hash"]
    ):
        raise ValueError("scale joint completion lineage differs")
    joint_root = (
        Path(selected["output_root"])
        if args.scale_joint_root is None
        else args.scale_joint_root
    )
    run = load_hashed_json(joint_root / "assets" / "run.json")
    _, graph, _, _ = load_joint_graph_template(
        joint_root / "assets" / "graph" / "joint_graph_template.json",
        expected_variant="J5_END_TO_END",
        expected_run_record_sha256=run["content_hash"],
        expected_predictor_bundle_lock_sha256=lock["content_hash"],
        expected_source=campaign["source"],
    )
    checkpoint_path = joint_root / "best_model_val.pt"
    expected_checkpoint = (
        selected["checkpoint_sha256"]
        if scale_index is None
        else joint_completion["joint_checkpoint_sha256"]
    )
    if _sha256(checkpoint_path) != expected_checkpoint:
        raise ValueError("final-consumer selected J5 checkpoint differs")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    graph.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    graph.to(device).eval()
    calibrations, calibration_sha = _calibrations(
        args.campaign_root,
        lock,
        args.pipeline_seed,
        scale_index=scale_index,
    )
    training_role = (
        "scale_train" if scale_index is not None else "model_train"
    )
    for split in (training_role, "val_stop", "val_design"):
        joint_manifest, joint_dataset = load_joint_dataset_cache(
            joint_root
            / "assets"
            / "datasets"
            / split
            / "joint_dataset.json",
            expected_split=split,
            expected_source=campaign["source"],
        )
        replicas = (
            (0, 1, 2, 3)
            if split in {"model_train", "scale_train"}
            else (0,)
        )
        outputs = {
            replica: _evaluate_replica(
                graph,
                joint_dataset,
                replica=replica,
                device=device,
                calibrations=calibrations,
                batch_size=args.batch_size,
            )
            for replica in replicas
        }
        declared = np.asarray(
            [
                replica_for(
                    policy="R_MULTI",
                    logical_role=split,
                    epoch=0,
                    canonical_identity=identity,
                )
                for identity in joint_dataset.identities
            ],
            dtype=np.int64,
        )
        parent_hashes = {
            "identity_manifest": run["parent_hashes"][
                {
                    "model_train": "model_train_identity_manifest",
                    "scale_train": "scale_train_identity_manifest",
                    "val_stop": "val_stop_identity_manifest",
                    "val_design": "val_design_identity_manifest",
                }[split]
            ],
            "HLT_view_cache": run["parent_hashes"][
                {
                    "model_train": "model_train_R_MULTI_view_cache",
                    "scale_train": "scale_train_R_MULTI_view_cache",
                    "val_stop": "val_stop_R_MULTI_view_cache",
                    "val_design": "val_design_fixed_view_cache",
                }[split]
            ],
            "joint_prediction_cache": (
                selected["registration_sha256"]
                if scale_index is None
                else joint_completion["joint_registration_sha256"]
            ),
            "native_HLT_cache": run["parent_hashes"][
                "selected_HLT_expert_seed_artifacts"
            ],
            "offline_target_cache": run["parent_hashes"][
                "offline_target_cache"
            ],
            "target_normalizer_set": run["parent_hashes"][
                "target_normalizer_set"
            ],
            "uncertainty_calibration": calibration_sha,
        }
        dataset = FinalConsumerDataset(
            identities=joint_dataset.identities,
            labels=joint_dataset.labels,
            replica_ids=declared,
            degraded_view_hashes={
                replica: joint_dataset.degraded_view_hashes_by_replica[
                    replica
                ]
                for replica in replicas
            },
            split=split,
            predicted_banks={
                expert: {
                    replica: outputs[replica]["predicted"][expert]
                    for replica in replicas
                }
                for expert in EXPERT_ORDER
            },
            calibrated_log_variance={
                expert: {
                    replica: outputs[replica]["uncertainty"][expert]
                    for replica in replicas
                }
                for expert in EXPERT_ORDER
            },
            native_banks={
                expert: {
                    replica: outputs[replica]["native"][expert]
                    for replica in replicas
                }
                for expert in EXPERT_ORDER
            },
            native_expert_logits={
                expert: {
                    replica: outputs[replica]["native_logits"][expert]
                    for replica in replicas
                }
                for expert in EXPERT_ORDER
            },
            predicted_expert_logits={
                expert: {
                    replica: outputs[replica]["predicted_logits"][expert]
                    for replica in replicas
                }
                for expert in EXPERT_ORDER
            },
            oracle_banks=joint_dataset.oracle_banks,
            target_normalized_banks=(
                joint_dataset.target_normalized_banks
            ),
            target_expert_logits=joint_dataset.target_expert_logits,
            oracle_fusion_logits=joint_dataset.oracle_fusion_logits,
            lineage_hashes=parent_hashes,
        )
        publish_final_consumer_dataset(
            output_dir=args.output_dir / split,
            dataset=dataset,
            parent_hashes=parent_hashes,
            source_snapshot=source_snapshot(REPO_ROOT),
        )
    print(
        json.dumps(
            {
                "pipeline_seed": args.pipeline_seed,
                "joint_registration_sha256": (
                    selected["registration_sha256"]
                    if scale_index is None
                    else joint_completion["joint_registration_sha256"]
                ),
                "uncertainty_calibration_set_sha256": calibration_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
