#!/usr/bin/env python3
"""Assemble, authenticate, and execute one concrete RETB Step-12 row."""

from __future__ import annotations

import argparse
import copy
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

from scripts.evaluate_retb_final_consumer_reference import (  # noqa: E402
    main as evaluate_reference_main,
)
from scripts.train_retb_final_consumer import main as train_main  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    FINAL_CONSUMER_CHECKPOINT_CONTRACT,
    publish_final_consumer_template,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumers import (  # noqa: E402
    FrozenPredictedOfflineFusion,
    HLTResidualAdapter,
    NativeConditionedTokenRefiner,
    UnrestrictedHLTFusion,
    clone_robust_offline_fusion,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    load_joint_graph_template,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    STAGE_J_CONSUMER_REGISTRY_CONTRACT,
    materialize_final_consumer_run,
    validate_materialized_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    SCALE_COMPONENT_INDEX_CONTRACT,
    SCALE_JOINT_COMPLETION_CONTRACT,
    build_scale_final_consumer_run,
    validate_scale_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_joint(
    root: Path,
    *,
    seed: int,
    carried_shape_role: str,
    joint_lock: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Any]:
    selected = joint_lock["carried_by_shape_role"][carried_shape_role][
        "selected_j5_by_seed"
    ][str(seed)]
    output = Path(selected["output_root"])
    run = load_hashed_json(output / "assets" / "run.json")
    _, graph, _, _ = load_joint_graph_template(
        output / "assets" / "graph" / "joint_graph_template.json",
        expected_variant="J5_END_TO_END",
        expected_run_record_sha256=run["content_hash"],
        expected_predictor_bundle_lock_sha256=run[
            "predictor_bundle_lock_sha256"
        ],
        expected_source=source,
    )
    checkpoint_path = output / "best_model_val.pt"
    if _sha256(checkpoint_path) != selected["checkpoint_sha256"]:
        raise ValueError("selected J5 checkpoint bytes differ")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    graph.load_state_dict(checkpoint["model_state_dict"], strict=True)
    graph.eval()
    return output, run, graph


def _normalizer_parents(root: Path) -> dict[str, str]:
    bundle = load_hashed_json(
        root / "inputs" / "normalization" / "stage_a_normalizer_bundle.json"
    )
    profile = load_hashed_json(root / "inputs" / "hlt_v3_profile.json")
    return {
        "HLT_input_normalizer": bundle["content_hash"],
        "HLT_relation_normalizer": bundle["artifact_hashes"][
            "shared_hlt_500k_relation"
        ],
        "HLT_region_normalizer": bundle["artifact_hashes"][
            "shared_hlt_500k_region"
        ],
        "degradation_profile": profile["content_hash"],
    }


def _uncertainty_widths(dataset_manifest: Path) -> dict[str, int]:
    payload = torch.load(
        dataset_manifest.parent / "final_consumer_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )
    dataset = payload["dataset"]
    return {
        expert: int(
            dataset.calibrated_log_variance[expert][
                dataset.replica_set[0]
            ].shape[-1]
        )
        for expert in EXPERT_ORDER
    }


def _load_selected_refiner(
    root: Path,
    *,
    seed: int,
    bank_dimensions: Mapping[str, int],
    token_counts: Mapping[str, int],
    uncertainty_widths: Mapping[str, int],
    carried_shape_role: str,
    scale_index: Mapping[str, Any] | None = None,
    scale_selected_refiner: Mapping[str, Any] | None = None,
) -> tuple[NativeConditionedTokenRefiner, str]:
    if scale_index is None:
        lock = load_hashed_json(
            root
            / "selection"
            / "final_consumers"
            / carried_shape_role
            / "token_refiner_lock.json"
        )
        selected = lock["selected_by_seed"][str(seed)]
        checkpoint_path = Path(selected["checkpoint_path"])
        variant = lock["selected_variant"]
        checkpoint_sha = selected["checkpoint_sha256"]
    else:
        selected = (
            scale_selected_refiner
            if scale_selected_refiner is not None
            else scale_index["selected_token_refiner"]
        )
        if selected is None:
            raise ValueError("scale selected token refiner is absent")
        checkpoint_path = Path(selected["checkpoint_path"])
        variant = str(selected["variant"])
        checkpoint_sha = selected["checkpoint_sha256"]
    if _sha256(checkpoint_path) != checkpoint_sha:
        raise ValueError("selected token-refiner checkpoint bytes differ")
    model = NativeConditionedTokenRefiner(
        variant=variant,
        bank_dimensions=bank_dimensions,
        token_counts=token_counts,
        uncertainty_widths=uncertainty_widths,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("contract") != FINAL_CONSUMER_CHECKPOINT_CONTRACT:
        raise ValueError("selected token-refiner checkpoint contract differs")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint_sha


def _model(
    row: Mapping[str, Any],
    *,
    frozen_fusion: Any,
    bank_dimensions: Mapping[str, int],
    token_counts: Mapping[str, int],
    uncertainty_widths: Mapping[str, int],
) -> Any:
    kind = row["consumer_kind"]
    if kind == "PF_FROZEN":
        return FrozenPredictedOfflineFusion(copy.deepcopy(frozen_fusion))
    if kind == "OF_ROBUST":
        return clone_robust_offline_fusion(frozen_fusion)
    if kind == "TR_REFINE":
        return NativeConditionedTokenRefiner(
            variant=row["model_variant"],
            bank_dimensions=bank_dimensions,
            token_counts=token_counts,
            uncertainty_widths=uncertainty_widths,
        )
    if kind == "HF_ADAPTER":
        return HLTResidualAdapter(
            variant=row["model_variant"],
            native_dropout_mode=row["native_dropout_mode"],
            bank_dimensions=bank_dimensions,
            token_counts=token_counts,
            uncertainty_widths=uncertainty_widths,
        )
    if kind == "HF_UNRESTRICTED":
        return UnrestrictedHLTFusion(
            evidence_variant=row["model_variant"],
            native_dropout_mode=row["native_dropout_mode"],
            bank_dimensions=bank_dimensions,
            token_counts=token_counts,
            uncertainty_widths=uncertainty_widths,
        )
    raise ValueError("final-consumer kind differs")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--graph-id")
    parser.add_argument("--scale-component-index", type=Path)
    parser.add_argument("--scale-joint-completion", type=Path)
    parser.add_argument("--scale-joint-root", type=Path)
    parser.add_argument("--scale-base-run-id")
    parser.add_argument("--scale-run-id")
    parser.add_argument("--scale-selected-refiner", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    scale_values = (
        args.graph_id,
        args.scale_component_index,
        args.scale_joint_completion,
        args.scale_joint_root,
    )
    if any(value is None for value in scale_values) != all(
        value is None for value in scale_values
    ):
        raise ValueError("scale final-consumer arguments are incomplete")
    if all(value is None for value in scale_values) and any(
        value is not None
        for value in (
            args.scale_base_run_id,
            args.scale_run_id,
            args.scale_selected_refiner,
        )
    ):
        raise ValueError("scale-only final-consumer option used at 500k")
    scale_index = (
        None
        if args.scale_component_index is None
        else load_hashed_json(
            args.scale_component_index,
            expected_contract=SCALE_COMPONENT_INDEX_CONTRACT,
        )
    )
    joint_completion = (
        None
        if args.scale_joint_completion is None
        else load_hashed_json(
            args.scale_joint_completion,
            expected_contract=SCALE_JOINT_COMPLETION_CONTRACT,
        )
    )
    scale_selected_refiner = (
        None
        if args.scale_selected_refiner is None
        else load_hashed_json(args.scale_selected_refiner)
    )
    if scale_selected_refiner is not None and (
        scale_index is None
        or scale_selected_refiner.get("contract")
        != "retb_scale_selected_token_refiner_v1"
        or scale_selected_refiner.get("source") != campaign.get("source")
        or int(scale_selected_refiner.get("pipeline_seed", -1))
        != int(scale_index["pipeline_seed"])
        or scale_selected_refiner.get("graph_id") != args.graph_id
    ):
        raise ValueError("scale selected-token-refiner lineage differs")
    if scale_index is not None and (
        scale_index.get("source") != campaign.get("source")
        or joint_completion.get("source") != campaign.get("source")
        or scale_index["graph_id"] != args.graph_id
        or joint_completion["graph_id"] != args.graph_id
        or int(scale_index["pipeline_seed"])
        != int(joint_completion["pipeline_seed"])
        or joint_completion["scale_component_index_sha256"]
        != scale_index["content_hash"]
    ):
        raise ValueError("scale final-consumer lineage differs")
    step12 = load_hashed_json(
        root / "registry" / "retb_step12_final_consumers_bundle.json"
    )
    registry = load_hashed_json(
        root / "registry" / "retb_final_consumer_registry.json",
        expected_contract=STAGE_J_CONSUMER_REGISTRY_CONTRACT,
    )
    requested_run_id = args.run_id
    if scale_index is not None:
        shortlist = load_hashed_json(
            root / "selection" / "locked_scale_shortlist.json"
        )
        definition = shortlist["locked_graph_definitions"][args.graph_id]
        requested_run_id = (
            args.scale_base_run_id
            or definition["configuration"]["run_ids_by_seed"][
                str(scale_index["pipeline_seed"])
            ]
        )
    rows = [
        row for row in registry["rows"] if row["run_id"] == requested_run_id
    ]
    if len(rows) != 1:
        raise ValueError("final-consumer run ID is absent or duplicated")
    row = rows[0]
    seed = int(row["pipeline_seed"])
    joint_lock = load_hashed_json(
        root / "selection" / "joint" / "joint_campaign_lock.json"
    )
    if scale_index is None:
        _, joint_run, joint_graph = _selected_joint(
            root,
            seed=seed,
            carried_shape_role=row["carried_shape_role"],
            joint_lock=joint_lock,
            source=campaign["source"],
        )
    else:
        joint_root = args.scale_joint_root
        joint_run = load_hashed_json(joint_root / "assets" / "run.json")
        _, joint_graph, _, _ = load_joint_graph_template(
            joint_root
            / "assets"
            / "graph"
            / "joint_graph_template.json",
            expected_variant="J5_END_TO_END",
            expected_run_record_sha256=joint_run["content_hash"],
            expected_predictor_bundle_lock_sha256=joint_run[
                "predictor_bundle_lock_sha256"
            ],
            expected_source=campaign["source"],
        )
        checkpoint_path = joint_root / "best_model_val.pt"
        if (
            _sha256(checkpoint_path)
            != joint_completion["joint_checkpoint_sha256"]
        ):
            raise ValueError("scale joint checkpoint bytes differ")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        joint_graph.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        joint_graph.eval()
    training_role = (
        "scale_train" if scale_index is not None else "model_train"
    )
    dataset_manifests = {
        split: load_hashed_json(
            args.dataset_root
            / row["carried_shape_role"]
            / f"seed_{seed}"
            / split
            / "final_consumer_dataset.json"
        )
        for split in (training_role, "val_stop", "val_design")
    }
    dataset_parent = dataset_manifests["val_design"]["parent_hashes"]
    bank_dimensions = {
        expert: int(joint_graph.allocation[expert][1])
        for expert in EXPERT_ORDER
    }
    token_counts = {
        expert: int(joint_graph.allocation[expert][0])
        for expert in EXPERT_ORDER
    }
    uncertainty_widths = _uncertainty_widths(
        args.dataset_root
        / row["carried_shape_role"]
        / f"seed_{seed}"
        / "val_design"
        / "final_consumer_dataset.json"
    )
    frozen_fusion = copy.deepcopy(joint_graph.frozen_offline_fusion)
    frozen_heads = {
        expert: copy.deepcopy(joint_graph.frozen_expert_heads[expert])
        for expert in EXPERT_ORDER
    }
    for component in (frozen_fusion, *frozen_heads.values()):
        component.eval()
        for parameter in component.parameters():
            parameter.requires_grad_(False)
    selected_refiner = None
    selected_refiner_sha = None
    if row["token_input"] == "TOKEN_REFINED_SELECTED":
        selected_refiner, selected_refiner_sha = _load_selected_refiner(
            root,
            seed=seed,
            bank_dimensions=bank_dimensions,
            token_counts=token_counts,
            uncertainty_widths=uncertainty_widths,
            carried_shape_role=row["carried_shape_role"],
            scale_index=scale_index,
            scale_selected_refiner=scale_selected_refiner,
        )
    parents = {
        name: joint_run["parent_hashes"][name]
        for name in (
            f"{training_role}_identity_manifest",
            "val_stop_identity_manifest",
            "val_design_identity_manifest",
            "val_design_label_manifest",
            f"{training_role}_R_MULTI_view_cache",
            "val_stop_R_MULTI_view_cache",
            "val_design_fixed_view_cache",
            "offline_target_cache",
            "target_normalizer_set",
            "frozen_offline_fusion",
            "frozen_offline_expert_heads",
        )
    }
    parents.update(
        {
            "joint_prediction_checkpoint": (
                joint_lock["carried_by_shape_role"][
                    row["carried_shape_role"]
                ]["selected_j5_by_seed"][str(seed)]["registration_sha256"]
                if scale_index is None
                else joint_completion["joint_registration_sha256"]
            ),
            "native_HLT_checkpoint_bundle": joint_run["parent_hashes"][
                "selected_HLT_expert_seed_artifacts"
            ],
            "uncertainty_calibration": dataset_parent[
                "uncertainty_calibration"
            ],
            **(
                _normalizer_parents(root)
                if scale_index is None
                else {
                    "HLT_input_normalizer": scale_index[
                        "scale_normalizer_bundle_sha256"
                    ],
                    "HLT_relation_normalizer": scale_index[
                        "hlt_relation_normalizer_sha256"
                    ],
                    "HLT_region_normalizer": scale_index[
                        "hlt_region_normalizer_sha256"
                    ],
                    "degradation_profile": scale_index[
                        "degradation_profile_sha256"
                    ],
                }
            ),
        }
    )
    if selected_refiner_sha is not None:
        parents["selected_token_refiner"] = selected_refiner_sha
    if scale_index is None:
        run = bind_source(
            materialize_final_consumer_run(
                registry_row=row,
                step12_bundle_sha256=step12["content_hash"],
                parent_hashes=parents,
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        validate_materialized_final_consumer_run(run)
    else:
        base_run = load_hashed_json(
            root
            / "runs"
            / "final_consumers"
            / requested_run_id
            / "assets"
            / "run.json"
        )
        run = bind_source(
            build_scale_final_consumer_run(
                base_run=base_run,
                scale_parent_hashes=parents,
                graph_id=args.graph_id,
                scale_run_id=args.scale_run_id,
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        validate_scale_final_consumer_run(run)
    assets = args.output_dir / "assets"
    run_path = assets / "run.json"
    write_immutable_json(run_path, run)
    model = _model(
        row,
        frozen_fusion=frozen_fusion,
        bank_dimensions=bank_dimensions,
        token_counts=token_counts,
        uncertainty_widths=uncertainty_widths,
    )
    component_parents = {
        name: parents[name]
        for name in (
            "joint_prediction_checkpoint",
            "native_HLT_checkpoint_bundle",
            "uncertainty_calibration",
            "frozen_offline_fusion",
            "frozen_offline_expert_heads",
        )
    }
    if selected_refiner_sha is not None:
        component_parents["selected_token_refiner"] = selected_refiner_sha
    template_dir = assets / "template"
    publish_final_consumer_template(
        output_dir=template_dir,
        model=model,
        frozen_expert_heads=frozen_heads,
        frozen_offline_fusion=frozen_fusion,
        refiner=selected_refiner,
        run_record_sha256=run["content_hash"],
        component_parent_hashes=component_parents,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    common = [
        "--campaign-root",
        str(root),
        "--run",
        str(run_path),
        "--template",
        str(template_dir / "final_consumer_template.json"),
        "--val-design-cache",
        str(
            args.dataset_root
            / row["carried_shape_role"]
            / f"seed_{seed}"
            / "val_design"
            / "final_consumer_dataset.json"
        ),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
    ]
    if row["trainable"]:
        train_main(
            [
                *common,
                "--model-train-cache",
                str(
                    args.dataset_root
                    / row["carried_shape_role"]
                    / f"seed_{seed}"
                    / training_role
                    / "final_consumer_dataset.json"
                ),
                *(
                    ["--training-role", "scale_train"]
                    if scale_index is not None
                    else []
                ),
                "--val-stop-cache",
                str(
                    args.dataset_root
                    / row["carried_shape_role"]
                    / f"seed_{seed}"
                    / "val_stop"
                    / "final_consumer_dataset.json"
                ),
            ]
        )
    else:
        evaluate_reference_main(common)
    print(
        json.dumps(
            {
                "run_id": row["run_id"],
                "run_sha256": run["content_hash"],
                "dataset_set_sha256": canonical_sha256(
                    {
                        split: manifest["content_hash"]
                        for split, manifest in dataset_manifests.items()
                    }
                ),
                "performance_based_termination": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
