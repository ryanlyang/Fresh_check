#!/usr/bin/env python3
"""Assemble and export one exact selected HLT-only RETB inference graph."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_retb_final_consumer_seed import _calibrations  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    DeployableRetbGraph,
    JointBridgeDeployableFrontend,
    analytical_export_flops,
    build_capacity_parameter_components,
    build_complete_graph_capacity,
    export_deployable_retb_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    FINAL_CONSUMER_CHECKPOINT_CONTRACT,
    FINAL_CONSUMER_REGISTRATION_CONTRACT,
    load_final_consumer_template,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumers import (  # noqa: E402
    FrozenPredictedOfflineFusion,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    load_joint_dataset_cache,
    load_joint_graph_template,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    FINAL_CONSUMER_RUN_CONTRACT,
    validate_materialized_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    SCALE_COMPONENT_INDEX_CONTRACT,
    SCALE_FINAL_CONSUMER_RUN_CONTRACT,
    SCALE_JOINT_COMPLETION_CONTRACT,
    validate_scale_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


PARITY_CONTRACT = "retb_research_export_parity_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_consumer(
    run_root: Path,
    *,
    run: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], Any, Any, dict[str, Any]]:
    template, model, heads, fusion, refiner = load_final_consumer_template(
        run_root / "assets" / "template" / "final_consumer_template.json",
        expected_run_record_sha256=run["content_hash"],
        expected_source=source,
    )
    if run["trainable"]:
        registration = load_hashed_json(
            run_root / "registration.json",
            expected_contract=FINAL_CONSUMER_REGISTRATION_CONTRACT,
        )
        checkpoint_path = run_root / "best_model_val.pt"
        if _sha256(checkpoint_path) != registration["checkpoint_sha256"]:
            raise ValueError("final-consumer checkpoint bytes differ")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("contract") != FINAL_CONSUMER_CHECKPOINT_CONTRACT:
            raise ValueError("final-consumer checkpoint contract differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        registration = load_hashed_json(
            run_root / "reference_registration.json",
            expected_contract="retb_final_consumer_reference_registration_v1",
        )
    return model, heads, fusion, refiner, registration


def _joint_graph(
    root: Path,
    *,
    seed: int,
    carried_shape_role: str,
    source: Mapping[str, Any],
    scale_joint_root: Path | None = None,
    scale_joint_completion: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Mapping[str, Any]]:
    if scale_joint_root is None:
        joint_lock = load_hashed_json(
            root / "selection" / "joint" / "joint_campaign_lock.json"
        )
        selected = joint_lock["carried_by_shape_role"][carried_shape_role][
            "selected_j5_by_seed"
        ][str(seed)]
        output = Path(selected["output_root"])
    else:
        if scale_joint_completion is None:
            raise ValueError("scale joint completion is absent")
        output = scale_joint_root
        selected = {
            "checkpoint_sha256": scale_joint_completion[
                "joint_checkpoint_sha256"
            ],
            "registration_sha256": scale_joint_completion[
                "joint_registration_sha256"
            ],
        }
    joint_run = load_hashed_json(output / "assets" / "run.json")
    _, graph, _, _ = load_joint_graph_template(
        output / "assets" / "graph" / "joint_graph_template.json",
        expected_variant="J5_END_TO_END",
        expected_run_record_sha256=joint_run["content_hash"],
        expected_predictor_bundle_lock_sha256=joint_run[
            "predictor_bundle_lock_sha256"
        ],
        expected_source=source,
    )
    checkpoint_path = output / "best_model_val.pt"
    if _sha256(checkpoint_path) != selected["checkpoint_sha256"]:
        raise ValueError("deployable selected J5 checkpoint differs")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    graph.load_state_dict(checkpoint["model_state_dict"], strict=True)
    graph.eval()
    _, dataset = load_joint_dataset_cache(
        output
        / "assets"
        / "datasets"
        / "val_design"
        / "joint_dataset.json",
        expected_split="val_design",
        expected_source=source,
    )
    return graph, dataset, selected


def _smoke_inputs(dataset: Any, *, count: int) -> dict[str, Any]:
    stop = min(int(count), len(dataset))
    if stop <= 0 or dataset.shared_raw_view is None:
        raise ValueError("deployable smoke population is empty")
    raw = dataset.shared_raw_view
    return {
        "identities": list(dataset.identities[:stop]),
        "replica_ids": torch.zeros(stop, dtype=torch.int64),
        "degraded_view_hashes": list(
            dataset.degraded_view_hashes_by_replica[0][:stop]
        ),
        "features": torch.from_numpy(raw["features"][0][:stop]),
        "vectors": torch.from_numpy(raw["vectors"][0][:stop]),
        "mask": torch.from_numpy(raw["mask"][0][:stop]),
        "raw_tokens": torch.from_numpy(raw["raw_tokens"][0][:stop]),
        "region_trees_by_expert": {
            expert: list(
                raw["region_trees_by_expert"][expert][0][:stop]
            )
            for expert in EXPERT_ORDER
        },
    }


def _research_logits(run_root: Path, *, trainable: bool) -> np.ndarray:
    path = (
        run_root / "val_design" / "final_consumer_predictions.npz"
        if trainable
        else run_root / "final_consumer_predictions.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["logits"], dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke-events", type=int, default=8)
    parser.add_argument("--scale-component-index", type=Path)
    parser.add_argument("--scale-joint-root", type=Path)
    parser.add_argument("--scale-joint-completion", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    scale_args = (
        args.scale_component_index,
        args.scale_joint_root,
        args.scale_joint_completion,
    )
    if any(value is None for value in scale_args) != all(
        value is None for value in scale_args
    ):
        raise ValueError("scale export arguments are incomplete")
    scale_index = (
        None
        if args.scale_component_index is None
        else load_hashed_json(
            args.scale_component_index,
            expected_contract=SCALE_COMPONENT_INDEX_CONTRACT,
        )
    )
    scale_completion = (
        None
        if args.scale_joint_completion is None
        else load_hashed_json(
            args.scale_joint_completion,
            expected_contract=SCALE_JOINT_COMPLETION_CONTRACT,
        )
    )
    run = load_hashed_json(args.run_root / "assets" / "run.json")
    if scale_index is None:
        if run.get("contract") != FINAL_CONSUMER_RUN_CONTRACT:
            raise ValueError("deployable final-consumer contract differs")
        validate_materialized_final_consumer_run(run)
    else:
        if (
            run.get("contract") != SCALE_FINAL_CONSUMER_RUN_CONTRACT
            or scale_index.get("source") != campaign.get("source")
            or scale_completion.get("source") != campaign.get("source")
            or scale_completion["scale_component_index_sha256"]
            != scale_index["content_hash"]
        ):
            raise ValueError("scale export lineage differs")
        validate_scale_final_consumer_run(run)
    model, heads, fusion, refiner, registration = _load_consumer(
        args.run_root, run=run, source=campaign["source"]
    )
    joint, dataset, selected_joint = _joint_graph(
        root,
        seed=int(run["pipeline_seed"]),
        carried_shape_role=run["carried_shape_role"],
        source=campaign["source"],
        scale_joint_root=args.scale_joint_root,
        scale_joint_completion=scale_completion,
    )
    predictor_lock = load_hashed_json(
        root
        / "selection"
        / "predictor_bundle"
        / "carried"
        / f"{run['carried_shape_role']}.json"
    )
    calibrations, _ = _calibrations(
        root,
        predictor_lock,
        int(run["pipeline_seed"]),
        scale_index=scale_index,
    )
    frontend = JointBridgeDeployableFrontend(
        joint_graph=joint,
        calibration_offsets={
            expert: calibrations[expert]["additive_offset_by_group"]
            for expert in EXPERT_ORDER
        },
    )
    deployable_kind = run["consumer_kind"]
    deployable_model = model
    deployable_refiner = refiner
    if run["consumer_kind"] == "TR_REFINE":
        deployable_model = FrozenPredictedOfflineFusion(fusion)
        deployable_refiner = model
    graph = DeployableRetbGraph(
        frontend=frontend,
        final_consumer=deployable_model,
        consumer_kind=deployable_kind,
        frozen_offline_fusion=fusion,
        frozen_expert_heads=heads,
        token_refiner=deployable_refiner,
    )
    smoke = _smoke_inputs(dataset, count=args.smoke_events)
    parents = {
        "campaign_spec": campaign["content_hash"],
        "step12_bundle": run["step12_bundle_sha256"],
        "HLT_frontend_checkpoint": run["parent_hashes"][
            "native_HLT_checkpoint_bundle"
        ],
        "joint_predictor_or_J_checkpoint": run["parent_hashes"][
            "joint_prediction_checkpoint"
        ],
        "final_consumer_checkpoint": registration["content_hash"],
        "frozen_offline_fusion_checkpoint": run["parent_hashes"][
            "frozen_offline_fusion"
        ],
        "frozen_offline_expert_heads": run["parent_hashes"][
            "frozen_offline_expert_heads"
        ],
        "HLT_input_normalizer": run["parent_hashes"][
            "HLT_input_normalizer"
        ],
        "HLT_relation_normalizer": run["parent_hashes"][
            "HLT_relation_normalizer"
        ],
        "HLT_region_normalizer": run["parent_hashes"][
            "HLT_region_normalizer"
        ],
        "degradation_profile": run["parent_hashes"][
            "degradation_profile"
        ],
        "uncertainty_calibration": run["parent_hashes"][
            "uncertainty_calibration"
        ],
    }
    if deployable_refiner is not None:
        parents["token_refiner_checkpoint"] = (
            run["parent_hashes"]["selected_token_refiner"]
            if run["token_input"] == "TOKEN_REFINED_SELECTED"
            else registration["content_hash"]
        )
    manifest = export_deployable_retb_graph(
        output_dir=args.output_dir,
        graph=graph,
        hlt_smoke_inputs=smoke,
        parent_hashes=parents,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    component_modules = build_capacity_parameter_components(graph)
    analytical_flops, measured_diagnostics = analytical_export_flops(
        exported_graph=graph,
        hlt_smoke_inputs=smoke,
    )
    capacity = build_complete_graph_capacity(
        graph_id=run["run_id"],
        deployment_export_sha256=manifest["content_hash"],
        component_modules=component_modules,
        exported_graph=graph,
        analytical_component_flops=analytical_flops,
        measured_diagnostics=measured_diagnostics,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(
        args.output_dir / "complete_graph_capacity.json", capacity
    )
    graph.eval()
    with torch.no_grad():
        actual = graph(hlt_inputs=smoke)["logits"].float().cpu().numpy()
    expected = _research_logits(
        args.run_root, trainable=bool(run["trainable"])
    )[: len(actual)]
    maximum = float(np.max(np.abs(actual - expected)))
    passed = bool(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)
    )
    if not passed:
        raise RuntimeError(
            "deployable graph differs from corresponding research graph"
        )
    parity = bind_source(
        with_content_hash(
            {
                "contract": PARITY_CONTRACT,
                "schema_version": 1,
                "run_sha256": run["content_hash"],
                "deployment_export_sha256": manifest["content_hash"],
                "selected_joint_registration_sha256": selected_joint[
                    "registration_sha256"
                ],
                "research_registration_sha256": registration["content_hash"],
                "event_count": len(actual),
                "absolute_tolerance": 1.0e-5,
                "relative_tolerance": 1.0e-5,
                "maximum_absolute_error": maximum,
                "logits_shape": list(actual.shape),
                "passed": True,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(
        args.output_dir / "research_graph_parity.json", parity
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
