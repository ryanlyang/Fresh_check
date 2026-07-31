#!/usr/bin/env python3
"""Train one post-lock, scale-population RETB HLT capacity control."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.train_retb_native_hlt_expert import _labels  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.capacity import (  # noqa: E402
    select_monolithic_capacity_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_capacity_controls import (  # noqa: E402
    build_hlt_capacity_control_model,
    build_hlt_capacity_control_row,
    publish_hlt_capacity_control_export,
    validate_hlt_capacity_control_row,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    NativeHLTExpertDataset,
    make_native_hlt_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (  # noqa: E402
    MonolithicBase4ParticleTransformer,
    analytical_particle_transformer_flops,
    build_monolithic_grid,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_training import (  # noqa: E402
    OfflineCapacityTrainingConfig,
    build_capacity_profile,
    train_offline_capacity_model,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    VARIANT_LOSS_WEIGHTS,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_losses import (  # noqa: E402
    FIXED_WEIGHTS,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    LOCKED_SCALE_FINALISTS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part.capacity import (  # noqa: E402
    pair_encoder_flops,
)


CONTROL_KINDS = ("H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG")
BASE_CONFIGURATION = (128, 4, 8, 8, 2)
LABEL_LEDGER_CONTRACT = "retb_hlt_long_exposure_ledger_v3"


def _hlt_cache(root: Path, split: str, replica: int) -> Path:
    return (
        root
        / "inputs"
        / "hlt_v3"
        / split
        / f"replica_{replica}"
        / (
            "R_MULTI"
            if split in {"model_train", "scale_train"}
            else "R_FIXED"
        )
        / "D_NOMINAL"
    )


def _dataset(root: Path, split: str) -> NativeHLTExpertDataset:
    replicas = range(4) if split in {"model_train", "scale_train"} else (0,)
    arrays, metadata = {}, {}
    for replica in replicas:
        arrays[replica], metadata[replica] = load_hlt_v3_cache(
            _hlt_cache(root, split, replica)
        )
    labels, identities = _labels(
        root / "inputs" / "offline" / split / "offline_inputs.npz"
    )
    return NativeHLTExpertDataset(
        replica_arrays=arrays,
        replica_metadata=metadata,
        labels=labels,
        identities=identities,
        logical_role=split,
        realization_policy=(
            "R_MULTI"
            if split in {"model_train", "scale_train"}
            else "R_FIXED"
        ),
    )


def _shortlist_long_exposure_ledger(
    *,
    root: Path,
    graph_definition: Mapping[str, Any],
    seed: int,
    train_events: int,
) -> dict[str, Any]:
    """Resolve only the 500k HLT phases owned by one locked graph.

    Paths are selected from the same immutable locks used to assemble the
    graph.  No recursive discovery is permitted: unrelated graph roles and
    unselected predictor candidates must never increase H_BASE_LONG.
    """

    graph_id = str(graph_definition["graph_id"])
    role = str(graph_definition["configuration"]["source_carried_shape_role"])
    run_id = str(graph_definition["configuration"]["run_ids_by_seed"][str(seed)])
    confirmation = load_hashed_json(
        root / "selection" / "predictor_phases"
        / "stage_d_evidence_confirmations.json"
    )
    predictor_lock = load_hashed_json(
        root / "selection" / "predictor_bundle" / "carried" / f"{role}.json"
    )
    coordinate_id = str(predictor_lock["coordinate_id"])
    if ":" not in coordinate_id:
        raise ValueError("500k H_BASE_LONG carried coordinate is malformed")
    native_shape = coordinate_id.split(":", 1)[1]
    phases: list[tuple[str, Path, Path]] = []
    for expert in (
        "BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"
    ):
        matches = [
            row
            for row in confirmation["rows"]
            if row["component"] == "HLT_EXPERT"
            and int(row["seed"]) == int(seed)
            and row["configuration"]["shape_id"] == native_shape
            and row["configuration"]["expert_id"] == expert
            and row["configuration"]["mode"] == "HE_SCRATCH_CE"
            and row["configuration"]["realization_policy"] == "R_MULTI"
            and not row["configuration"]["measurement_embedding"]
        ]
        if len(matches) != 1:
            raise ValueError("500k H_BASE_LONG native-expert ownership differs")
        phase_root = (
            root / "runs" / "stage_d" / "hlt_experts"
            / matches[0]["run_id"] / f"seed_{seed}"
        )
        phases.append(
            (
                f"native_hlt_expert:{expert}",
                phase_root / "training_curves.json",
                phase_root / "checkpoint_registration.json",
            )
        )

    stage_d = load_hashed_json(root / "registry" / "retb_stage_d_runs.json")
    fusion_matches = [
        row
        for row in stage_d["native_fusion_rows"]
        if row["configuration"]["shape_id"] == role
        and row["configuration"]["fusion_variant"] == "HF_NATIVE"
    ]
    if len(fusion_matches) != 1:
        raise ValueError("500k H_BASE_LONG native-fusion ownership differs")
    native_fusion_root = (
        root
        / "runs"
        / "stage_d"
        / "native_fusions"
        / fusion_matches[0]["run_id"]
        / f"seed_{seed}"
    )
    phases.append(
        (
            "native_hlt_fusion",
            native_fusion_root / "training_curves.json",
            native_fusion_root / "fusion_registration.json",
        )
    )

    configuration = json.loads(
        (
            root / "selection" / "predictor_bundle" / "inputs"
            / "selector_configuration.json"
        ).read_text("utf-8")
    )
    for expert in (
        "BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"
    ):
        candidate = predictor_lock["selected_candidate_descriptors"][expert][
            "candidate_id"
        ]
        selected_path = Path(
            configuration["materialized_run_paths"][candidate][str(seed)]
        )
        selected_run = load_hashed_json(selected_path)
        phase_root = root / "runs" / "predictors" / selected_run["run_id"]
        phases.append(
            (
                f"predictor:{expert}",
                phase_root / "training" / "training_curves.json",
                phase_root / "training" / "registration.json",
            )
        )

    joint_lock = load_hashed_json(
        root / "selection" / "joint" / "joint_campaign_lock.json"
    )
    j4_selection = load_hashed_json(
        root / "selection" / "joint" / role / "j4_blocks.json"
    )
    j4_root = (
        root
        / "runs"
        / "joint"
        / role
        / (
            f"RETB_J4_BRIDGE_FINETUNE_S{seed}_"
            f"N{int(j4_selection['selected_final_particle_blocks'])}"
        )
    )
    phases.append(
        (
            "joint:J4_BRIDGE_FINETUNE",
            j4_root / "training_curves.json",
            j4_root / "registration.json",
        )
    )
    joint_root = Path(
        joint_lock["carried_by_shape_role"][role]["selected_j5_by_seed"][
            str(seed)
        ]["output_root"]
    )
    phases.append(
        (
            "joint:J5_END_TO_END",
            joint_root / "training_curves.json",
            joint_root / "registration.json",
        )
    )

    if graph_definition["configuration"]["token_input"] == "TOKEN_REFINED_SELECTED":
        refiner_lock = load_hashed_json(
            root
            / "selection"
            / "final_consumers"
            / role
            / "token_refiner_lock.json"
        )
        selected_refiner = refiner_lock["selected_by_seed"][str(seed)]
        refiner_root = Path(selected_refiner["checkpoint_path"]).resolve().parent
        phases.append(
            (
                "token_refiner",
                refiner_root / "training_curves.json",
                refiner_root / "registration.json",
            )
        )

    final_root = root / "runs" / "final_consumers" / run_id
    final_registration = final_root / "registration.json"
    excluded_rows: list[dict[str, Any]] = []
    if final_registration.is_file():
        phases.append(
            (
                "final_consumer",
                final_root / "training_curves.json",
                final_registration,
            )
        )
    else:
        reference = final_root / "reference_registration.json"
        reference_payload = load_hashed_json(reference)
        excluded_rows.append(
            {
                "phase_id": "final_consumer",
                "component_path": str(reference.resolve()),
                "component_sha256": reference_payload["content_hash"],
                "reason": "locked_reference_graph_has_no_optimizer_updates",
                "ground_truth_CE_evidence": {
                    "eligible": False,
                    "basis": "serialized_reference_registration",
                },
            }
        )

    rows: list[dict[str, Any]] = []
    for phase_id, curves_path, registration_path in phases:
        curves = load_hashed_json(curves_path)
        registration = load_hashed_json(registration_path)
        parsed = _updates_and_presentations(curves_path)
        if parsed is None:
            raise ValueError(f"500k H_BASE_LONG phase counts unreadable: {phase_id}")
        updates, presentations = parsed
        if presentations <= 0:
            effective_batch = int(
                curves.get("config", {}).get("effective_batch_size", 128)
            )
            updates_per_epoch = math.ceil(int(train_events) / effective_batch)
            epochs, partial = divmod(updates, updates_per_epoch)
            presentations = epochs * int(train_events) + min(
                partial * effective_batch, int(train_events)
            )
        # Reuse the corrected serialized CE semantics used by the scale ledger.
        if phase_id.startswith("native_hlt_expert:"):
            mode = str(registration.get("mode", ""))
            if mode not in {"HE_SCRATCH_CE", "HE_OFFLINE_INIT", "HE_DUAL_OBJECTIVE"}:
                raise ValueError("500k native-HLT CE mode is unregistered")
            evidence = {
                "eligible": True,
                "basis": "serialized_native_HLT_mode_CE_coefficient",
                "mode": mode,
                "ground_truth_CE_weight": 1.0,
            }
        elif phase_id == "native_hlt_fusion":
            variant = str(registration.get("variant", ""))
            if variant not in {
                "HF_NATIVE",
                "HF_TRAINED_LOGIT",
                "HF_7X_UNBIASED_TOKEN_FUSION",
            }:
                raise ValueError("500k native-fusion CE variant is unregistered")
            evidence = {
                "eligible": True,
                "basis": "serialized_native_fusion_variant_CE_objective",
                "variant": variant,
                "ground_truth_CE_weight": 1.0,
            }
        elif phase_id.startswith("predictor:"):
            objective = str(registration.get("objective_id", ""))
            weights = FIXED_WEIGHTS.get(
                "W_CANONICAL" if objective == "W_GRADNORM" else objective
            )
            if weights is None:
                raise ValueError("500k predictor objective is unregistered")
            evidence = {
                "eligible": float(weights[-1]) > 0.0,
                "basis": "serialized_predictor_objective_logit_CE_weight",
                "objective_id": objective,
                "ground_truth_CE_weight": float(weights[-1]),
            }
        elif phase_id.startswith("joint:"):
            variant = str(registration.get("variant", ""))
            weights = VARIANT_LOSS_WEIGHTS.get(variant)
            if weights is None:
                raise ValueError("500k joint variant is unregistered")
            weight = float(weights["fused_CE"]) + float(weights["native_HLT_CE"])
            evidence = {
                "eligible": weight > 0.0,
                "basis": "serialized_joint_variant_direct_CE_weights",
                "variant": variant,
                "ground_truth_CE_weight": weight,
            }
        else:
            kind = str(registration.get("consumer_kind", ""))
            weights = {
                "PF_FROZEN": 1.0,
                "OF_ROBUST": 1.0,
                "TR_REFINE": 0.25,
                "HF_ADAPTER": 1.0,
                "HF_UNRESTRICTED": 1.0,
            }
            if kind not in weights:
                raise ValueError("500k final-consumer CE kind is unregistered")
            evidence = {
                "eligible": weights[kind] > 0.0,
                "basis": "serialized_final_consumer_kind_CE_weight",
                "consumer_kind": kind,
                "ground_truth_CE_weight": weights[kind],
            }
        row = {
            "phase_id": phase_id,
            "component_path": str(curves_path.resolve()),
            "component_sha256": curves["content_hash"],
            "registration_path": str(registration_path.resolve()),
            "registration_sha256": registration["content_hash"],
            "optimizer_updates": updates,
            "labeled_example_presentations": presentations,
            "ground_truth_CE_evidence": evidence,
        }
        (rows if evidence["eligible"] else excluded_rows).append(row)
    total = sum(row["labeled_example_presentations"] for row in rows)
    if total <= 0:
        raise ValueError("500k H_BASE_LONG has no owned CE presentations")
    return with_content_hash(
        {
            "contract": LABEL_LEDGER_CONTRACT,
            "schema_version": 3,
            "owner_finalist_graph_id": graph_id,
            "owner_carried_shape_role": role,
            "pipeline_seed": int(seed),
            "owner_graph_definition_sha256": graph_definition[
                "complete_graph_definition_sha256"
            ],
            "component_rows": rows,
            "excluded_zero_CE_component_rows": excluded_rows,
            "total_labeled_example_presentations": total,
            "effective_batch_size": 128,
            "optimizer_update_budget": math.ceil(total / 128),
            "rounding": "ceil",
            "pure_offline_KD_and_calibration_phases_excluded": True,
            "every_included_phase_has_proven_nonzero_ground_truth_CE": True,
            "unreadable_candidate_artifacts_rejected": True,
            "unreferenced_shortlist_roles_excluded": True,
            "performance_used_to_set_budget": False,
        }
    )


def _capacity_selection(
    *, target: Mapping[str, Any], weaver: Any
) -> dict[str, Any]:
    candidates = []
    for configuration in build_monolithic_grid():
        model = MonolithicBase4ParticleTransformer(
            configuration, weaver_module=weaver
        )
        flops = (
            analytical_particle_transformer_flops(
                configuration=configuration
            )
            + pair_encoder_flops(4, (64, 64, 64))
        )
        candidates.append(
            {
                "configuration": list(configuration),
                "parameter_count": sum(
                    int(value.numel()) for value in model.parameters()
                ),
                "inference_flops_batch1": int(flops),
                "inference_flops_batch128": 128 * int(flops),
            }
        )
        del model
    return select_monolithic_capacity_controls(
        target_parameters=int(target["parameter_count"]),
        target_flops_batch1=float(target["analytical_flops_batch1"]),
        target_flops_batch128=float(target["analytical_flops_batch128"]),
        candidates=candidates,
        domain="hlt",
        target_complete_graph_sha256=target[
            "complete_graph_capacity_sha256"
        ],
    )


def _updates_and_presentations(path: Path) -> tuple[int, int] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_content_hash(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    counts = payload.get(
        "optimizer_update_counts",
        payload.get("planned_update_counts", {}),
    )
    rows = payload.get("rows", [])
    row_updates = [
        int(
            row.get(
                "optimizer_updates_completed",
                row.get("optimizer_update_ordinal", 0),
            )
        )
        for row in rows
        if isinstance(row, Mapping)
    ]
    updates = int(
        payload.get(
            "optimizer_updates_completed",
            counts.get(
                "total_optimizer_updates",
                max(row_updates, default=0),
            ),
        )
    )
    presentations = int(payload.get("labeled_example_presentations", 0))
    if presentations <= 0 and isinstance(rows, list):
        presentations = sum(
            int(row.get("training_examples_presented", 0))
            for row in rows
            if isinstance(row, Mapping)
        )
    if (
        updates <= 0
        or payload.get("fixed_budget_completed") is False
        or payload.get("fixed_epoch_budget_completed") is False
    ):
        return None
    return updates, presentations


def _long_exposure_ledger(
    *,
    root: Path,
    owner_graph_id: str,
    seed: int,
    scale_train_events: int,
    model_train_events: int | None = None,
) -> dict[str, Any]:
    finalist = root / "runs" / "scale" / "graphs" / owner_graph_id / f"seed_{seed}"
    seed_root = root / "runs" / "scale" / "refits" / f"seed_{seed}"
    shortlist = load_hashed_json(
        root / "selection" / "locked_scale_shortlist.json",
        expected_contract="retb_locked_scale_shortlist_v2",
    )
    definition = shortlist["locked_graph_definitions"].get(owner_graph_id)
    if definition is None:
        raise ValueError("H_BASE_LONG owner graph is not shortlist-locked")
    role = str(definition["configuration"]["source_carried_shape_role"])
    component_index_path = seed_root / "component_indexes" / f"{owner_graph_id}.json"
    component_index = load_hashed_json(
        component_index_path, expected_contract="retb_scale_component_index_v1"
    )
    if (
        component_index.get("source") != shortlist.get("source")
        or component_index.get("graph_id") != owner_graph_id
        or int(component_index.get("pipeline_seed", -1)) != int(seed)
    ):
        raise ValueError("H_BASE_LONG component-index ownership differs")

    phases: list[tuple[str, Path, Path]] = []
    for expert, record in sorted(component_index["native_hlt_experts"].items()):
        phase_root = Path(record["output_root"]).resolve()
        phases.append(
            (
                f"native_hlt_expert:{expert}",
                phase_root / "training_curves.json",
                phase_root / "checkpoint_registration.json",
            )
        )
    native_fusion_root = seed_root / "roles" / role / "native_fusion"
    phases.append(
        (
            "native_hlt_fusion",
            native_fusion_root / "training_curves.json",
            native_fusion_root / "fusion_registration.json",
        )
    )
    for expert, record in sorted(component_index["predictors"].items()):
        phase_root = Path(record["output_root"]).resolve() / "training"
        phases.append(
            (
                f"predictor:{expert}",
                phase_root / "training_curves.json",
                phase_root / "registration.json",
            )
        )
    j4_selection = load_hashed_json(
        root / "selection" / "joint" / role / "j4_blocks.json"
    )
    j4_root = (
        root
        / "runs"
        / "joint"
        / role
        / (
            f"RETB_J4_BRIDGE_FINETUNE_S{seed}_"
            f"N{int(j4_selection['selected_final_particle_blocks'])}"
        )
    )
    phases.append(
        (
            "joint:J4_BRIDGE_FINETUNE",
            j4_root / "training_curves.json",
            j4_root / "registration.json",
        )
    )
    phases.append(
        (
            "joint:J5_END_TO_END",
            finalist / "joint" / "training_curves.json",
            finalist / "joint" / "registration.json",
        )
    )
    refiner_root = finalist / "token_refiner"
    if definition["configuration"]["token_input"] == "TOKEN_REFINED_SELECTED":
        phases.append(
            (
                "token_refiner",
                refiner_root / "training_curves.json",
                refiner_root / "registration.json",
            )
        )
    elif refiner_root.exists():
        raise ValueError("H_BASE_LONG found an unowned scale token refiner")
    final_registration = finalist / "final_consumer" / "registration.json"
    final_curves = finalist / "final_consumer" / "training_curves.json"
    excluded_rows: list[dict[str, Any]] = []
    if final_registration.is_file() or final_curves.is_file():
        phases.append(("final_consumer", final_curves, final_registration))
    else:
        reference = finalist / "final_consumer" / "reference_registration.json"
        reference_payload = load_hashed_json(reference)
        excluded_rows.append(
            {
                "phase_id": "final_consumer",
                "component_path": str(reference.resolve()),
                "component_sha256": reference_payload["content_hash"],
                "reason": "locked_reference_graph_has_no_optimizer_updates",
                "ground_truth_CE_evidence": {
                    "eligible": False,
                    "basis": "serialized_reference_registration",
                },
            }
        )

    def ce_evidence(phase_id: str, registration: Mapping[str, Any]) -> dict[str, Any]:
        if phase_id.startswith("native_hlt_expert:"):
            mode = str(registration.get("mode", ""))
            if mode not in {"HE_SCRATCH_CE", "HE_OFFLINE_INIT", "HE_DUAL_OBJECTIVE"}:
                raise ValueError("native-HLT CE mode is unregistered")
            return {
                "eligible": True,
                "basis": "serialized_native_HLT_mode_CE_coefficient",
                "mode": mode,
                "ground_truth_CE_weight": 1.0,
            }
        if phase_id == "native_hlt_fusion":
            variant = str(registration.get("variant", ""))
            if variant not in {"HF_NATIVE", "HF_TRAINED_LOGIT", "HF_7X_UNBIASED_TOKEN_FUSION"}:
                raise ValueError("native-fusion CE variant is unregistered")
            return {
                "eligible": True,
                "basis": "serialized_native_fusion_variant_CE_objective",
                "variant": variant,
                "ground_truth_CE_weight": 1.0,
            }
        if phase_id.startswith("predictor:"):
            objective = str(registration.get("objective_id", ""))
            weights = (
                FIXED_WEIGHTS["W_CANONICAL"]
                if objective == "W_GRADNORM"
                else FIXED_WEIGHTS.get(objective)
            )
            if weights is None:
                raise ValueError("predictor objective is unregistered")
            return {
                "eligible": float(weights[-1]) > 0.0,
                "basis": "serialized_predictor_objective_CE_weight",
                "objective_id": objective,
                "ground_truth_CE_weight": float(weights[-1]),
            }
        if phase_id.startswith("joint:"):
            variant = str(registration.get("variant", ""))
            weights = VARIANT_LOSS_WEIGHTS.get(variant)
            if weights is None:
                raise ValueError("joint variant is unregistered")
            direct = float(weights["fused_CE"]) + float(weights["native_HLT_CE"])
            return {
                "eligible": direct > 0.0,
                "basis": "serialized_joint_variant_direct_CE_weights",
                "variant": variant,
                "ground_truth_CE_weight": direct,
            }
        if phase_id in {"token_refiner", "final_consumer"}:
            kind = str(registration.get("consumer_kind", ""))
            weights = {
                "PF_FROZEN": 1.0,
                "OF_ROBUST": 1.0,
                "TR_REFINE": 0.25,
                "HF_ADAPTER": 1.0,
                "HF_UNRESTRICTED": 1.0,
            }
            if kind not in weights:
                raise ValueError("final-consumer CE kind is unregistered")
            return {
                "eligible": weights[kind] > 0.0,
                "basis": "serialized_final_consumer_kind_CE_weight",
                "consumer_kind": kind,
                "ground_truth_CE_weight": weights[kind],
            }
        raise ValueError(f"HLT exposure phase is unclassified: {phase_id}")

    rows: list[dict[str, Any]] = []
    seen_roots: set[Path] = set()
    for phase_id, curves_path, registration_path in phases:
        curves_path = curves_path.resolve()
        registration_path = registration_path.resolve()
        if curves_path.parent in seen_roots:
            raise ValueError("H_BASE_LONG phase ownership is duplicated")
        seen_roots.add(curves_path.parent)
        curves = load_hashed_json(curves_path)
        registration = load_hashed_json(registration_path)
        if (
            registration.get("training_curves_sha256") is not None
            and registration["training_curves_sha256"] != curves["content_hash"]
        ):
            raise ValueError("H_BASE_LONG registration/curve lineage differs")
        parsed = _updates_and_presentations(curves_path)
        if parsed is None:
            raise ValueError(f"H_BASE_LONG phase counts are unreadable: {phase_id}")
        updates, presentations = parsed
        # Some trainers record update counts but omit a presentation total.
        # Infer it from the fixed effective batch and authenticated population.
        if presentations <= 0:
            effective_batch = int(
                curves.get("config", {}).get(
                    "effective_batch_size", 128
                )
            )
            phase_events = (
                int(model_train_events)
                if phase_id == "joint:J4_BRIDGE_FINETUNE"
                and model_train_events is not None
                else int(scale_train_events)
            )
            updates_per_epoch = math.ceil(phase_events / effective_batch)
            complete_epochs, partial_updates = divmod(
                updates, updates_per_epoch
            )
            presentations = (
                complete_epochs * phase_events
                + min(
                    partial_updates * effective_batch,
                    phase_events,
                )
            )
        evidence = ce_evidence(phase_id, registration)
        row = {
            "phase_id": phase_id,
            "component_path": str(curves_path),
            "component_sha256": curves["content_hash"],
            "registration_path": str(registration_path),
            "registration_sha256": registration["content_hash"],
            "optimizer_updates": updates,
            "labeled_example_presentations": presentations,
            "ground_truth_CE_evidence": evidence,
        }
        (rows if evidence["eligible"] else excluded_rows).append(row)
    if not rows:
        raise ValueError("H_BASE_LONG found no HLT-side training phases")
    total = sum(row["labeled_example_presentations"] for row in rows)
    return with_content_hash(
        {
            "contract": LABEL_LEDGER_CONTRACT,
            "schema_version": 3,
            "owner_finalist_graph_id": owner_graph_id,
            "owner_carried_shape_role": role,
            "pipeline_seed": int(seed),
            "scale_component_index_sha256": component_index["content_hash"],
            "component_rows": rows,
            "excluded_zero_CE_component_rows": excluded_rows,
            "total_labeled_example_presentations": total,
            "effective_batch_size": 128,
            "optimizer_update_budget": math.ceil(total / 128),
            "rounding": "ceil",
            "pure_offline_KD_and_calibration_phases_excluded": True,
            "every_included_phase_has_proven_nonzero_ground_truth_CE": True,
            "unreadable_candidate_artifacts_rejected": True,
            "unreferenced_shortlist_roles_excluded": True,
            "selected_500k_J4_initialization_included": True,
            "performance_used_to_set_budget": False,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    lock_group = parser.add_mutually_exclusive_group(required=True)
    lock_group.add_argument("--locked-scale-finalists", type=Path)
    lock_group.add_argument("--locked-scale-shortlist", type=Path)
    parser.add_argument("--owner-finalist-graph-id", required=True)
    parser.add_argument("--control-kind", required=True, choices=CONTROL_KINDS)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    output = args.output_dir.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    is_500k = args.locked_scale_shortlist is not None
    if is_500k:
        owner_lock = load_hashed_json(
            args.locked_scale_shortlist,
            expected_contract="retb_locked_scale_shortlist_v2",
        )
        owner_ids = set(owner_lock["SCALE_SHORTLIST"])
        graph_definition = owner_lock["locked_graph_definitions"].get(
            args.owner_finalist_graph_id
        )
    else:
        owner_lock = load_hashed_json(
            args.locked_scale_finalists,
            expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
        )
        owner_ids = set(owner_lock["finalist_graph_ids"])
        shortlist = load_hashed_json(
            root / "selection" / "locked_scale_shortlist.json",
            expected_contract="retb_locked_scale_shortlist_v2",
        )
        graph_definition = shortlist["locked_graph_definitions"].get(
            args.owner_finalist_graph_id
        )
    if (
        owner_lock.get("source") != campaign.get("source")
        or args.owner_finalist_graph_id not in owner_ids
        or graph_definition is None
        or args.pipeline_seed not in {101, 202, 303}
    ):
        raise ValueError("capacity-control owner lineage differs")
    completed_path = output / "control_row.json"
    if completed_path.is_file():
        completed = load_hashed_json(completed_path)
        validate_hlt_capacity_control_row(completed)
        export = load_hashed_json(output / "deployable_control.json")
        if (
            completed.get("source") != campaign["source"]
            or export.get("source") != campaign["source"]
            or completed["owner_finalist_graph_id"]
            != args.owner_finalist_graph_id
            or completed["control_kind"] != args.control_kind
            or int(completed["pipeline_seed"]) != args.pipeline_seed
            or completed["deployable_export_sha256"]
            != export["content_hash"]
        ):
            raise ValueError("reusable scale finalist-control differs")
        print(json.dumps(completed, indent=2, sort_keys=True))
        return 0
    training_split = "model_train" if is_500k else "scale_train"
    for resource in (training_split, "val_stop", "val_design"):
        authorize_dataset_access(
            worker_role=(
                "design_worker"
                if resource == "val_design"
                else (
                    "training_worker" if is_500k else "scale_training_worker"
                )
            ),
            requested_resource=resource,
        )
    if is_500k:
        completion = None
        run_id = graph_definition["configuration"]["run_ids_by_seed"][
            str(args.pipeline_seed)
        ]
        capacity_path = root / "exports" / run_id / "complete_graph_capacity.json"
    else:
        completion = load_hashed_json(root / "selection" / "scale_completion.json")
        scale_row = next(
            row
            for row in completion["runs"]
            if row["graph_id"] == args.owner_finalist_graph_id
            and int(row["pipeline_seed"]) == args.pipeline_seed
        )
        capacity_path = (
            root
            / "runs"
            / "scale"
            / "graphs"
            / args.owner_finalist_graph_id
            / f"seed_{args.pipeline_seed}"
            / "export"
            / "complete_graph_capacity.json"
        )
    capacity = load_hashed_json(capacity_path)
    target = (
        {
            "parameter_count": int(capacity["totals"]["parameter_count"]),
            "analytical_flops_batch1": int(
                capacity["totals"]["analytical_flops_batch1"]
            ),
            "analytical_flops_batch128": int(
                capacity["totals"]["analytical_flops_batch128"]
            ),
            "complete_graph_capacity_sha256": capacity["content_hash"],
        }
        if is_500k
        else {
            **scale_row["capacity"],
            "complete_graph_capacity_sha256": capacity["content_hash"],
        }
    )
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    scale_train = _dataset(root, training_split)
    val_stop = _dataset(root, "val_stop")
    val_design = _dataset(root, "val_design")
    miniature = campaign["campaign_profile"] == "miniature_test"

    if args.control_kind == "H_BASE_LONG":
        ledger = bind_source(
            (
                _shortlist_long_exposure_ledger(
                    root=root,
                    graph_definition=graph_definition,
                    seed=args.pipeline_seed,
                    train_events=len(scale_train),
                )
                if is_500k
                else _long_exposure_ledger(
                    root=root,
                    owner_graph_id=args.owner_finalist_graph_id,
                    seed=args.pipeline_seed,
                    scale_train_events=len(scale_train),
                    model_train_events=int(
                        load_hashed_json(
                            root
                            / "inputs"
                            / "offline"
                            / "model_train"
                            / "offline_input_manifest.json"
                        )["event_count"]
                    ),
                )
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        selection = ledger
        configuration = None
        exact_updates = (
            min(4, int(ledger["optimizer_update_budget"]))
            if miniature
            else int(ledger["optimizer_update_budget"])
        )
        write_immutable_json(output / "label_exposure_ledger.json", ledger)
        flops = (
            analytical_particle_transformer_flops(
                configuration=BASE_CONFIGURATION
            )
            + pair_encoder_flops(4, (64, 64, 64))
        )
    else:
        selection = bind_source(
            _capacity_selection(target=target, weaver=weaver),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        configuration = selection[args.control_kind]["configuration"]
        exact_updates = None
        flops = int(selection[args.control_kind]["inference_flops_batch1"])
        write_immutable_json(
            output / "capacity_control_selection.json", selection
        )
    model = build_hlt_capacity_control_model(
        control_kind=args.control_kind,
        configuration=configuration,
        weaver_module=weaver,
    )
    profile = build_capacity_profile(
        control_id=args.control_kind,
        model=model,
        analytical_flops_batch1=int(flops),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(output / "complete_graph_profile.json", profile)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    registration = train_offline_capacity_model(
        model=model,
        train_loader=make_native_hlt_expert_loader(
            scale_train,
            seed=args.pipeline_seed,
            training=True,
            batch_size=args.microbatch_size,
        ),
        val_stop_loader=make_native_hlt_expert_loader(
            val_stop, seed=0, training=False, batch_size=args.microbatch_size
        ),
        val_design_loader=make_native_hlt_expert_loader(
            val_design,
            seed=0,
            training=False,
            batch_size=args.microbatch_size,
        ),
        output_dir=output / "training",
        config=OfflineCapacityTrainingConfig(
            control_id=args.control_kind,
            seed=args.pipeline_seed,
            maximum_epochs=2 if miniature else 40,
            microbatch_size=args.microbatch_size,
            gradient_accumulation_steps=2,
            campaign_profile=(
                "miniature_test" if miniature else "production"
            ),
            exact_optimizer_update_budget=exact_updates,
        ),
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        execution_registry_sha256=selection["content_hash"],
        lineage_hashes={
            "campaign_spec": campaign["content_hash"],
            (
                "locked_scale_shortlist"
                if is_500k
                else "locked_scale_finalists"
            ): owner_lock["content_hash"],
            **(
                {}
                if completion is None
                else {"scale_completion": completion["content_hash"]}
            ),
            "complete_graph_capacity": capacity["content_hash"],
            "control_selection_or_ledger": selection["content_hash"],
        },
        profile=profile,
        source_snapshot=source_snapshot(REPO_ROOT),
        device=device,
    )
    checkpoint = output / "training" / "best_model_val.pt"
    export = publish_hlt_capacity_control_export(
        output=output / "deployable_control.json",
        owner_finalist_graph_id=args.owner_finalist_graph_id,
        control_kind=args.control_kind,
        pipeline_seed=args.pipeline_seed,
        configuration=configuration,
        checkpoint_path=checkpoint,
        checkpoint_sha256=registration["checkpoint_sha256"],
        training_registration_sha256=registration["content_hash"],
        capacity_selection_sha256=selection["content_hash"],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    row = bind_source(
        build_hlt_capacity_control_row(
            owner_finalist_graph_id=args.owner_finalist_graph_id,
            control_kind=args.control_kind,
            pipeline_seed=args.pipeline_seed,
            checkpoint_sha256=registration["checkpoint_sha256"],
            deployable_export_sha256=export["content_hash"],
            training_registration_sha256=registration["content_hash"],
            optimizer_updates_completed=registration[
                "optimizer_updates_completed"
            ],
            labeled_example_presentations=registration[
                "labeled_example_presentations"
            ],
            capacity_selection_sha256=selection["content_hash"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(output / "control_row.json", row)
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
