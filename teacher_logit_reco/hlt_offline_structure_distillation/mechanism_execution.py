"""Concrete Stage-G execution against the locked design-confirm split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)

from .auxiliary import resolve_stage_d_phase_two
from .capacity import (
    combination_model_flop_ledger,
    exact_trainable_parameter_count,
)
from .combination_runtime import (
    COMBINATION_CHECKPOINT_CONTRACT,
    build_combination_loader_manifest,
    build_combination_model,
    load_combination_loaders,
    train_combination,
)
from .contracts import (
    AUXILIARY_CHECKPOINT_CONTRACT,
    FEEDBACK_CHECKPOINT_CONTRACT,
    MECHANISM_CONTROL_PLAN_CONTRACT,
    SINGLE_FAMILY_PHASE_LOCK_CONTRACT,
    load_hashed_json,
    validate_content_hash,
    write_immutable_json,
)
from .feedback import build_feedback_model
from .feedback_data import (
    FeedbackInterventionDataset,
    make_feedback_loader,
    materialize_feedback_intervention,
)
from .mechanism_runtime import (
    evaluate_auxiliary_head_removal,
    eventwise_error_gain_tracking,
    mechanism_result_from_measurement,
)
from .stage_d_data_factory import (
    build_default_stage_d_role_definitions,
    build_stage_d_loader_manifest,
    load_stage_d_loaders_from_manifest,
)
from .stage_d_training import evaluate_auxiliary
from .baselines import HOSDTrainingProtocol, component_seed
from .baselines import build_baseline_model
from .auxiliary import build_auxiliary_model

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_stage_d_rows(
    stage_d_plan: Mapping[str, Any],
    phase_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    phase_two = resolve_stage_d_phase_two(
        stage_d_plan=stage_d_plan, phase_lock=phase_lock
    )
    return [
        *[
            dict(row)
            for row in stage_d_plan["all_rows"]
            if row["phase"]
            not in {"LOCKED_RELATION_HET", "MATCHED_HLT_SELF"}
        ],
        *phase_two,
    ]


def _row_loader(
    *,
    row: Mapping[str, Any],
    root: Path,
    base_roles: Mapping[str, Mapping[str, Any]],
    campaign: Mapping[str, Any],
    target_registry: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    definitions = build_default_stage_d_role_definitions(
        row=row,
        campaign_root=root,
        base_role_definitions=base_roles,
        evaluation_role="design_confirm",
    )
    manifest = build_stage_d_loader_manifest(
        row=row,
        role_definitions=definitions,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
        evaluation_role="design_confirm",
    )
    write_immutable_json(output, manifest)
    loaded = load_stage_d_loaders_from_manifest(
        manifest_path=output,
        campaign_root=root,
        row=dict(row),
        campaign=dict(campaign),
        target_registry=dict(target_registry),
    )
    return loaded, manifest


def _checkpoint(
    *,
    path: Path,
    expected_contract: str,
    row_id: str | None,
    graph_id: str | None,
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("contract") != expected_contract
        or payload.get("source") != dict(source)
        or (row_id is not None and payload.get("row_id") != row_id)
        or (graph_id is not None and payload.get("graph_id") != graph_id)
    ):
        raise ValueError("Stage-G checkpoint lineage differs")
    return payload


def _evaluate_registered_row(
    *,
    row: Mapping[str, Any],
    kind: str,
    root: Path,
    base_roles: Mapping[str, Mapping[str, Any]],
    campaign: Mapping[str, Any],
    target_registry: Mapping[str, Any],
    module: Any,
    device: str,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    loaded, manifest = _row_loader(
        row=row,
        root=root,
        base_roles=base_roles,
        campaign=campaign,
        target_registry=target_registry,
        output=output_root / "loaders" / f"{row['row_id']}.json",
    )
    if kind == "auxiliary":
        model, groups = build_auxiliary_model(row, weaver_module=module)
        checkpoint_path = (
            root
            / "auxiliary"
            / row["row_id"]
            / "seed_101"
            / "best_model_val.pt"
        )
        contract = AUXILIARY_CHECKPOINT_CONTRACT
    else:
        model = build_feedback_model(row, weaver_module=module)
        groups = loaded["component_group_ids"]
        checkpoint_path = (
            root
            / "feedback"
            / row["row_id"]
            / "seed_101"
            / "best_model_val.pt"
        )
        contract = FEEDBACK_CHECKPOINT_CONTRACT
    state = _checkpoint(
        path=checkpoint_path,
        expected_contract=contract,
        row_id=row["row_id"],
        graph_id=None,
        source=campaign["source"],
    )
    model.load_state_dict(state["model_state_dict"], strict=True)
    measurement = evaluate_auxiliary(
        model,
        loaded["design_confirm_loader"],
        row=row,
        component_group_ids=groups,
        split="design_confirm",
        device=torch.device(device),
    )
    return measurement, {
        "checkpoint": _sha(checkpoint_path),
        "loader_manifest": manifest["content_hash"],
    }


def _native_paths(root: Path) -> dict[str, Path]:
    return {
        **{
            f"model_train:{replica}": root
            / "targets"
            / "native_relations"
            / "model_train"
            / f"replica_{replica}.npz"
            for replica in range(4)
        },
        "val_stop:0": root
        / "targets"
        / "native_relations"
        / "val_stop"
        / "replica_0.npz",
        "design_confirm:0": root
        / "targets"
        / "native_relations"
        / "val_design"
        / "replica_0.npz",
    }


def _combination_loaders(
    *,
    graph: Mapping[str, Any],
    root: Path,
    base_roles: Mapping[str, Mapping[str, Any]],
    campaign: Mapping[str, Any],
    target_registry: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    members = {}
    for member in graph["members"]:
        row = {
            **dict(member),
            "row_id": member["selected_row_id"],
            "row_kind": "SCIENTIFIC",
            "resolved": True,
            "pipeline_seed": 101,
        }
        _, manifest = _row_loader(
            row=row,
            root=root,
            base_roles=base_roles,
            campaign=campaign,
            target_registry=target_registry,
            output=output_root
            / "loaders"
            / graph["graph_id"]
            / f"{member['target_id']}.json",
        )
        members[member["target_id"]] = (
            output_root
            / "loaders"
            / graph["graph_id"]
            / f"{member['target_id']}.json"
        )
    artifact = build_combination_loader_manifest(
        graph=graph,
        member_loader_manifests=members,
        native_relation_target_files=(
            _native_paths(root)
            if graph.get("native_relation_auxiliary") is not None
            else None
        ),
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
        evaluation_role="design_confirm",
    )
    manifest_path = output_root / "loaders" / f"{graph['graph_id']}.json"
    write_immutable_json(manifest_path, artifact)
    return (
        load_combination_loaders(
            manifest=artifact,
            graph=graph,
            campaign_root=root,
            campaign=campaign,
            target_registry=target_registry,
        ),
        artifact,
    )


def _load_intervention(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        identities = tuple(
            str(value) for value in payload["identities"].tolist()
        )
        values = {
            identity: {
                key: np.asarray(payload[key][index])
                for key in payload.files
                if key not in {"identities", "donor_identities"}
            }
            for index, identity in enumerate(identities)
        }
        donors = {
            identity: str(donor)
            for identity, donor in zip(
                identities, payload["donor_identities"].tolist()
            )
        }
    return values, donors


def _evaluate_wrong_event_row(
    *,
    control_row: Mapping[str, Any],
    scientific_row: Mapping[str, Any],
    root: Path,
    base_roles: Mapping[str, Mapping[str, Any]],
    campaign: Mapping[str, Any],
    target_registry: Mapping[str, Any],
    module: Any,
    device: str,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    loaded, manifest = _row_loader(
        row=control_row,
        root=root,
        base_roles=base_roles,
        campaign=campaign,
        target_registry=target_registry,
        output=output_root
        / "loaders"
        / f"{control_row['row_id']}.json",
    )
    model = build_feedback_model(scientific_row, weaver_module=module)
    checkpoint_path = (
        root
        / "feedback"
        / scientific_row["row_id"]
        / "seed_101"
        / "best_model_val.pt"
    )
    checkpoint = _checkpoint(
        path=checkpoint_path,
        expected_contract=FEEDBACK_CHECKPOINT_CONTRACT,
        row_id=scientific_row["row_id"],
        graph_id=None,
        source=campaign["source"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    split_plan = load_hashed_json(
        root
        / "targets"
        / "controls"
        / "plans"
        / "feedback"
        / "design_confirm"
        / "global"
        / f"{control_row['target_id']}.json",
        expected_contract="hosd_target_shuffle_plan_v1",
    )
    npz = output_root / "interventions" / f"{control_row['row_id']}.npz"
    materialize_feedback_intervention(
        row=control_row,
        loader=loaded["design_confirm_loader"],
        output_path=npz,
        role="design_confirm",
        prediction_model=model,
        shuffle_plan=split_plan,
        device=device,
    )
    values, donors = _load_intervention(npz)
    wrapped = FeedbackInterventionDataset(
        loaded["design_confirm_loader"].dataset,
        intervention="predicted_feedback_override",
        values_by_identity=values,
        donor_identity_by_identity=donors,
        parent_hashes={"intervention_npz": _sha(npz)},
    )
    loader = make_feedback_loader(
        wrapped, seed=101, training=False, batch_size=128
    )
    measurement = evaluate_auxiliary(
        model,
        loader,
        row=control_row,
        component_group_ids=loaded["component_group_ids"],
        split="design_confirm",
        device=torch.device(device),
    )
    return measurement, {
        "checkpoint": _sha(checkpoint_path),
        "loader_manifest": manifest["content_hash"],
        "intervention_npz": _sha(npz),
        "shuffle_plan": split_plan["content_hash"],
    }


def execute_mechanism_plan(
    *,
    plan: Mapping[str, Any],
    selected_graph: Mapping[str, Any],
    stage_f_plan_sha256: str,
    stage_d_plan: Mapping[str, Any],
    phase_lock: Mapping[str, Any],
    stage_e_plan: Mapping[str, Any],
    campaign: Mapping[str, Any],
    target_registry: Mapping[str, Any],
    base_roles: Mapping[str, Mapping[str, Any]],
    campaign_root: str | Path,
    weaver_module: Any,
    device: str,
) -> list[dict[str, Any]]:
    """Execute every locked intervention; unusual metrics never prune work."""

    validate_content_hash(plan, expected_contract=MECHANISM_CONTROL_PLAN_CONTRACT)
    root = Path(campaign_root)
    output_root = root / "mechanism_controls"
    resolved_rows = _resolved_stage_d_rows(stage_d_plan, phase_lock)
    stage_d_by_id = {row["row_id"]: row for row in resolved_rows}
    stage_e_by_id = {row["row_id"]: row for row in stage_e_plan["all_rows"]}
    scientific_e = list(stage_e_plan["scientific_rows"])
    results = []
    for intervention in plan["interventions"]:
        intervention_id = intervention["intervention_id"]
        kind = intervention["kind"]
        evidence: dict[str, str] = {
            "mechanism_plan": plan["content_hash"]
        }
        if intervention["execution"]["worker"] == "emit_registered_not_applicable":
            result = mechanism_result_from_measurement(
                plan=plan,
                intervention_id=intervention_id,
                measurement={"reason": intervention["execution"]["reason"]},
                evidence_hashes=evidence,
                source=campaign["source"],
                status="not_applicable",
            )
        elif kind == "leave_one_family_out":
            graph = dict(intervention["execution"]["graph"])
            loaded, manifest = _combination_loaders(
                graph=graph,
                root=root,
                base_roles=base_roles,
                campaign=campaign,
                target_registry=target_registry,
                output_root=output_root,
            )
            model = build_combination_model(
                graph, seed=101, weaver_module=weaver_module
            )
            ledger = combination_model_flop_ledger(model)
            protocol = HOSDTrainingProtocol(
                maximum_epochs=(
                    2
                    if campaign["campaign_profile"] == "miniature_test"
                    else 40
                ),
                campaign_profile=(
                    "miniature_test"
                    if campaign["campaign_profile"] == "miniature_test"
                    else "production"
                ),
            )
            completion = train_combination(
                model=model,
                train_loader=loaded["train_loader"],
                val_stop_loader=loaded["val_stop_loader"],
                design_select_loader=loaded["design_confirm_loader"],
                graph=graph,
                output_dir=output_root / "retrained" / intervention_id,
                stage_f_plan_sha256=stage_f_plan_sha256,
                campaign_spec_sha256=campaign["content_hash"],
                lineage_hashes=loaded["lineage_hashes"],
                protocol=protocol,
                source=campaign["source"],
                deployed_analytical_flops=float(
                    ledger["deployed_total_flops"]
                ),
                deployed_parameter_count=exact_trainable_parameter_count(
                    model.classifier
                ),
                device=device,
                evaluation_split="design_confirm",
            )
            result_path = (
                output_root
                / "retrained"
                / intervention_id
                / "design_select_result.json"
            )
            row_result = load_hashed_json(result_path)
            measurement = {
                "retrained_graph_id": graph["graph_id"],
                "design_confirm": row_result["design_confirm"],
                "common_initialization_seed": 101,
                "fixed_epoch_budget": protocol.maximum_epochs,
            }
            evidence.update(
                {
                    "loader_manifest": manifest["content_hash"],
                    "training_completion": completion["content_hash"],
                    "training_result": row_result["content_hash"],
                }
            )
            result = mechanism_result_from_measurement(
                plan=plan,
                intervention_id=intervention_id,
                measurement=measurement,
                evidence_hashes=evidence,
                source=campaign["source"],
            )
        elif kind == "inference_head_removal":
            loaded, manifest = _combination_loaders(
                graph=selected_graph,
                root=root,
                base_roles=base_roles,
                campaign=campaign,
                target_registry=target_registry,
                output_root=output_root,
            )
            model = build_combination_model(
                selected_graph, seed=101, weaver_module=weaver_module
            )
            checkpoint_path = (
                root
                / "combinations"
                / selected_graph["graph_id"]
                / "seed_101"
                / "best_model_val.pt"
            )
            checkpoint = _checkpoint(
                path=checkpoint_path,
                expected_contract=COMBINATION_CHECKPOINT_CONTRACT,
                row_id=None,
                graph_id=selected_graph["graph_id"],
                source=campaign["source"],
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            measurement = evaluate_auxiliary_head_removal(
                model, loaded["design_confirm_loader"], device=device
            )
            evidence.update(
                {
                    "checkpoint": _sha(checkpoint_path),
                    "loader_manifest": manifest["content_hash"],
                }
            )
            result = mechanism_result_from_measurement(
                plan=plan,
                intervention_id=intervention_id,
                measurement=measurement,
                evidence_hashes=evidence,
                source=campaign["source"],
            )
        elif kind in {
            "zero_feedback",
            "wrong_event_prediction",
            "matched_capacity",
        }:
            control_names = {
                "zero_feedback": {"ZERO", "ZERO_GATE"},
                "wrong_event_prediction": {
                    "SHUFFLED_PREDICTION",
                    "SHUFFLED",
                },
                "matched_capacity": {"UNRESTRICTED", "UNRESTRICTED_MLP"},
            }[kind]
            control_rows = [
                row
                for row in stage_e_by_id.values()
                if row.get("control") in control_names
            ]
            measurements, hashes = {}, {}
            for row in control_rows:
                if kind == "wrong_event_prediction":
                    sources = [
                        source_row
                        for source_row in scientific_e
                        if source_row["target_id"] == row["target_id"]
                        and source_row["interface"] == row["interface"]
                        and source_row["gradient_path"] == "END_TO_END"
                    ]
                    if len(sources) != 1:
                        raise ValueError(
                            "mechanism feedback source coverage differs"
                        )
                    measurement, row_hashes = _evaluate_wrong_event_row(
                        control_row=row,
                        scientific_row=sources[0],
                        root=root,
                        base_roles=base_roles,
                        campaign=campaign,
                        target_registry=target_registry,
                        module=weaver_module,
                        device=device,
                        output_root=output_root,
                    )
                else:
                    measurement, row_hashes = _evaluate_registered_row(
                        row=row,
                        kind="feedback",
                        root=root,
                        base_roles=base_roles,
                        campaign=campaign,
                        target_registry=target_registry,
                        module=weaver_module,
                        device=device,
                        output_root=output_root,
                    )
                measurements[row["row_id"]] = measurement
                hashes.update(
                    {
                        f"{row['row_id']}::{key}": value
                        for key, value in row_hashes.items()
                    }
                )
            evidence.update(hashes)
            result = mechanism_result_from_measurement(
                plan=plan,
                intervention_id=intervention_id,
                measurement={
                    "registered_control_rows": measurements,
                    "row_count": len(measurements),
                },
                evidence_hashes=evidence,
                source=campaign["source"],
            )
        elif kind in {"target_mean", "parameterization"}:
            selected_ids = {
                member["selected_row_id"] for member in selected_graph["members"]
            }
            selected_targets = {
                member["target_id"] for member in selected_graph["members"]
            }
            if kind == "target_mean":
                rows = [
                    row
                    for row in resolved_rows
                    if row["row_kind"] == "TARGET_MEAN"
                    and row["target_id"] in selected_targets
                ]
            else:
                rows = [
                    row
                    for row in resolved_rows
                    if row["row_kind"] == "SCIENTIFIC"
                    and row["target_id"] in selected_targets
                    and row["resolved"]
                ]
            measurements, hashes = {}, {}
            for row in rows:
                measurement, row_hashes = _evaluate_registered_row(
                    row=row,
                    kind="auxiliary",
                    root=root,
                    base_roles=base_roles,
                    campaign=campaign,
                    target_registry=target_registry,
                    module=weaver_module,
                    device=device,
                    output_root=output_root,
                )
                measurements[row["row_id"]] = measurement
                hashes.update(
                    {
                        f"{row['row_id']}::{key}": value
                        for key, value in row_hashes.items()
                    }
                )
            evidence.update(hashes)
            result = mechanism_result_from_measurement(
                plan=plan,
                intervention_id=intervention_id,
                measurement={
                    "registered_control_rows": measurements,
                    "row_count": len(measurements),
                    "selected_row_ids": sorted(selected_ids),
                },
                evidence_hashes=evidence,
                source=campaign["source"],
            )
        elif kind == "eventwise_correlation":
            loaded, manifest = _combination_loaders(
                graph=selected_graph,
                root=root,
                base_roles=base_roles,
                campaign=campaign,
                target_registry=target_registry,
                output_root=output_root,
            )
            candidate = build_combination_model(
                selected_graph, seed=101, weaver_module=weaver_module
            )
            candidate_checkpoint = (
                root
                / "combinations"
                / selected_graph["graph_id"]
                / "seed_101"
                / "best_model_val.pt"
            )
            candidate_state = _checkpoint(
                path=candidate_checkpoint,
                expected_contract=COMBINATION_CHECKPOINT_CONTRACT,
                row_id=None,
                graph_id=selected_graph["graph_id"],
                source=campaign["source"],
            )
            candidate.load_state_dict(
                candidate_state["model_state_dict"], strict=True
            )
            baseline = build_baseline_model(
                "H_BASE", weaver_module=weaver_module
            )
            baseline_checkpoint = (
                root
                / "baselines"
                / "H_BASE"
                / "seed_101"
                / "best_model_val.pt"
            )
            baseline_state = torch.load(
                baseline_checkpoint, map_location="cpu", weights_only=False
            )
            if (
                baseline_state.get("contract")
                != "hosd_baseline_checkpoint_v1"
                or baseline_state.get("baseline_id") != "H_BASE"
                or baseline_state.get("source") != campaign["source"]
            ):
                raise ValueError(
                    "mechanism H_BASE checkpoint lineage differs"
                )
            baseline.load_state_dict(
                baseline_state["model_state_dict"], strict=True
            )
            resolved_device = torch.device(device)
            candidate.to(resolved_device).eval()
            baseline.to(resolved_device).eval()
            ids: list[str] = []
            labels, base_logits, candidate_logits, errors = [], [], [], []
            target_id = selected_graph["members"][0]["target_id"]
            with torch.no_grad():
                for raw in loaded["design_confirm_loader"]:
                    batch = {
                        key: (
                            value.to(resolved_device)
                            if hasattr(value, "to")
                            else value
                        )
                        for key, value in raw.items()
                    }
                    vectors = batch.get(
                        "lorentz_vectors", batch.get("vectors")
                    )
                    points = batch.get(
                        "points", batch["features"][:, 15:17]
                    )
                    candidate_value, predictions = (
                        candidate.forward_with_auxiliaries(
                            points,
                            batch["features"],
                            vectors,
                            batch["mask"],
                        )
                    )
                    base_value = baseline(
                        points,
                        batch["features"],
                        vectors,
                        batch["mask"],
                    )
                    target = batch["combination_targets"][target_id]
                    difference = (
                        predictions[target_id]["value"]
                        - target["target"]
                    ).abs()
                    mask = target["target_mask"].bool()
                    flattened_mask = mask.reshape(mask.shape[0], -1)
                    flattened_difference = difference.reshape(
                        difference.shape[0], -1
                    )
                    count = flattened_mask.sum(dim=-1)
                    error = (
                        flattened_difference.masked_fill(
                            ~flattened_mask, 0
                        ).sum(dim=-1)
                        / count.clamp_min(1)
                    )
                    raw_ids = (
                        raw["identities"]
                        if "identities" in raw
                        else raw["event_identities"]
                    )
                    ids.extend(
                        str(value) for value in raw_ids
                    )
                    labels.append(batch["labels"].cpu().numpy())
                    base_logits.append(base_value.float().cpu().numpy())
                    candidate_logits.append(
                        candidate_value.float().cpu().numpy()
                    )
                    errors.append(error.float().cpu().numpy())
            measurement = eventwise_error_gain_tracking(
                identities=ids,
                labels=np.concatenate(labels),
                baseline_logits=np.concatenate(base_logits),
                candidate_logits=np.concatenate(candidate_logits),
                target_error=np.concatenate(errors),
            )
            measurement["tracked_target_id"] = target_id
            evidence.update(
                {
                    "loader_manifest": manifest["content_hash"],
                    "candidate_checkpoint": _sha(candidate_checkpoint),
                    "baseline_checkpoint": _sha(baseline_checkpoint),
                }
            )
            result = mechanism_result_from_measurement(
                plan=plan,
                intervention_id=intervention_id,
                measurement=measurement,
                evidence_hashes=evidence,
                source=campaign["source"],
            )
        else:
            raise ValueError(f"unknown mechanism intervention kind {kind!r}")
        write_immutable_json(
            output_root / "results" / f"{intervention_id}.json", result
        )
        results.append(result)
    return results


__all__ = ["execute_mechanism_plan"]
