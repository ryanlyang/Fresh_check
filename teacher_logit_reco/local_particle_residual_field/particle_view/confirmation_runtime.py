"""Production runtime for structural controls and three-seed confirmation.

This module closes the ``pv06_confirmation_selection`` node.  It resolves the
predeclared confirmation roles from completed broad-screen artifacts, replays
the resolved configuration at seeds 101/202/303, evaluates the structural
controls without retraining the consumer, and publishes the winner selection
plus the exact fairness inputs consumed by ``pv07``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .campaign import CONFIRMATION_ROLE_IDS
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .controls import STRUCTURAL_CONTROL_IDS, apply_particle_view_control
from .direct_control import (
    _direct_model,
    _flop_counter_sha256,
    particle_transformer_semantic_flops,
    particle_view_consumer_semantic_flops,
)
from .distillation import (
    _consumer_forward,
    train_frozen_consumer_distillation,
    train_joint_finetuning,
)
from .distillation_runtime import (
    _consumer_for_row,
    _pview_registration_and_model,
    _target_loaders,
    build_distillation_factory,
    validate_distillation_factory_config,
)
from .fairness_runtime import build_fairness_input_index
from .ledger import LabelExposureRecord, build_label_exposure_ledger
from .offline_teacher import build_predeclared_direct_control_grid
from .post_target_runtime import _artifact, _task_artifacts, _teacher_from_task
from .predictor import (
    PARTICLE_VIEW_FLOP_COUNTER,
    count_unique_parameters,
    flop_fixture_sha256,
    predictor_semantic_flops,
)
from .registry import validate_particle_view_registry
from .runtime_data import (
    load_aligned_logical_jet_view,
    make_logical_data_loader,
    validate_runtime_data_config,
)
from .selection import select_particle_view_winner_families
from .teacher_train import evaluate_particle_view_teacher


PARTICLE_VIEW_CONFIRMATION_FACTORY_CONFIG_CONTRACT = (
    "particle_view_confirmation_factory_config_v1"
)
PARTICLE_VIEW_CONFIRMATION_REPLICA_CONTRACT = (
    "particle_view_confirmation_replica_v1"
)
PARTICLE_VIEW_CONFIRMATION_RESOURCE_CONTRACT = (
    "particle_view_confirmation_bundle_resource_v1"
)
PARTICLE_VIEW_STRUCTURAL_CONTROL_RESULT_CONTRACT = (
    "particle_view_structural_control_result_v1"
)
PARTICLE_VIEW_CONFIRMATION_SUMMARY_CONTRACT = (
    "particle_view_confirmation_summary_v1"
)

_CONFIRM_PREFIX = "CONFIRM_"
_STRUCTURAL_PREFIX = "STRUCTURAL_CONTROL_"


def _role_policy() -> dict[str, dict[str, Any]]:
    """Return deterministic resolution rules for all declared role slots."""

    policies = {
        "CANONICAL_PREDECLARED": {
            "kind": "canonical",
            "privileged": True,
            "diagnostic": False,
        },
        "BEST_ARCHITECTURE": {
            "kind": "best_architecture",
            "privileged": True,
            "diagnostic": False,
        },
        "BEST_NO_CE": {
            "kind": "best_no_ce",
            "privileged": True,
            "diagnostic": False,
        },
        "BEST_SMALL_CE": {
            "kind": "best_small_ce",
            "privileged": True,
            "diagnostic": False,
        },
        "CE_ONLY_UPPER_BOUND": {
            "kind": "ce_only",
            "privileged": False,
            "diagnostic": False,
        },
        "REPRESENTATION_ONLY": {
            "kind": "representation_only",
            "privileged": True,
            "diagnostic": False,
        },
        "DIRECT_PARAMETER_CONTROL": {
            "kind": "direct",
            "source_run_id": "STAGE_A_PARAMETER_MATCH",
            "privileged": False,
            "diagnostic": True,
        },
        "DIRECT_FLOP_CONTROL": {
            "kind": "direct",
            "source_run_id": "STAGE_A_FLOP_MATCH",
            "privileged": False,
            "diagnostic": True,
        },
        "BEST_ALTERNATIVE_TARGET": {
            "kind": "alternative_target",
            "privileged": True,
            "diagnostic": False,
        },
        "RECOVERABILITY_CODESIGNED": {
            "kind": "recodesigned",
            "privileged": True,
            "diagnostic": False,
        },
        "HLT_MEMORY_CONTROL": {
            "kind": "hlt_memory",
            "privileged": False,
            "diagnostic": False,
        },
        "DVIEW_JOINT": {
            "kind": "source_run",
            "source_run_id": "TRAINED_CONTROL_DVIEW_JOINT",
            "privileged": True,
            "diagnostic": False,
        },
        "DVIEW_JOINT_CE_ONLY": {
            "kind": "source_run",
            "source_run_id": "TRAINED_CONTROL_DVIEW_JOINT_CE_ONLY",
            "privileged": False,
            "diagnostic": False,
        },
    }
    if set(policies) != set(CONFIRMATION_ROLE_IDS):
        raise RuntimeError("confirmation role policy inventory drifted")
    return policies


def build_confirmation_factory_config(
    *,
    distillation_factory_config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_distillation_factory_config(distillation_factory_config)
    data = distillation_factory_config["runtime_data_config"]
    validate_runtime_data_config(data, verify_cache_files=True)
    policies = _role_policy()
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONFIRMATION_FACTORY_CONFIG_CONTRACT,
            "distillation_factory_config": dict(
                distillation_factory_config
            ),
            "distillation_factory_config_sha256": (
                distillation_factory_config["content_hash"]
            ),
            "runtime_data_config_sha256": data["content_hash"],
            "role_policy": policies,
            "role_policy_sha256": canonical_sha256(policies),
            "confirmation_role_ids": list(CONFIRMATION_ROLE_IDS),
            "structural_control_ids": list(STRUCTURAL_CONTROL_IDS),
            "confirmation_seeds": [101, 202, 303],
            "selection_split": "model_val_select",
            "flop_fixture_sha256": flop_fixture_sha256(
                input_dim=17, particles=128
            ),
            "flop_counter_sha256": _flop_counter_sha256(),
            "single_training_pool": True,
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "performance_gates": False,
            "quality_warnings_stop_execution": False,
        }
    )
    validate_confirmation_factory_config(artifact)
    return artifact


def validate_confirmation_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload,
        expected_contract=PARTICLE_VIEW_CONFIRMATION_FACTORY_CONFIG_CONTRACT,
    )
    expected = {
        "contract",
        "distillation_factory_config",
        "distillation_factory_config_sha256",
        "runtime_data_config_sha256",
        "role_policy",
        "role_policy_sha256",
        "confirmation_role_ids",
        "structural_control_ids",
        "confirmation_seeds",
        "selection_split",
        "flop_fixture_sha256",
        "flop_counter_sha256",
        "single_training_pool",
        "stack_val_loaded",
        "final_test_loaded",
        "performance_gates",
        "quality_warnings_stop_execution",
        "content_hash",
    }
    distillation = payload["distillation_factory_config"]
    validate_distillation_factory_config(distillation)
    policies = _role_policy()
    if (
        set(payload) != expected
        or payload["distillation_factory_config_sha256"]
        != distillation["content_hash"]
        or payload["runtime_data_config_sha256"]
        != distillation["runtime_data_config_sha256"]
        or payload["role_policy"] != policies
        or payload["role_policy_sha256"] != canonical_sha256(policies)
        or payload["confirmation_role_ids"] != list(CONFIRMATION_ROLE_IDS)
        or payload["structural_control_ids"]
        != list(STRUCTURAL_CONTROL_IDS)
        or payload["confirmation_seeds"] != [101, 202, 303]
        or payload["selection_split"] != "model_val_select"
        or payload["flop_fixture_sha256"]
        != flop_fixture_sha256(input_dim=17, particles=128)
        or payload["flop_counter_sha256"] != _flop_counter_sha256()
        or payload["single_training_pool"] is not True
        or payload["stack_val_loaded"] is not False
        or payload["final_test_loaded"] is not False
        or payload["performance_gates"] is not False
        or payload["quality_warnings_stop_execution"] is not False
    ):
        raise ValueError("confirmation factory production policy changed")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "confirmation_role_count": len(CONFIRMATION_ROLE_IDS),
        "structural_control_count": len(STRUCTURAL_CONTROL_IDS),
    }


def _candidate_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["model_val_select"]
    recovery = metrics.get("recovery_fraction")
    finite = (
        metrics.get("recovery_status") == "finite"
        and recovery is not None
    )
    return (
        -float(metrics["deployable_accuracy"]),
        float(metrics["deployable_cross_entropy"]),
        0 if finite else 1,
        -float(recovery) if finite else 0.0,
        int(row["deployed_parameters"]),
        str(row["source_run_id"]),
        str(row["configuration_id"]),
    )


def resolve_confirmation_role(
    role_id: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve one role without applying a scientific quality threshold."""

    policies = _role_policy()
    if role_id not in policies:
        raise KeyError(f"unknown confirmation role {role_id!r}")
    policy = policies[role_id]
    if policy["kind"] == "direct":
        return {
            "role_id": role_id,
            "source_kind": "direct",
            "source_run_id": policy["source_run_id"],
            "privileged_claim_eligible": False,
            "pre_stage_g_deployable_eligible": False,
            "diagnostic": True,
            "resolution_reason": "predeclared_stage_a_direct_control",
        }
    rows = [dict(row) for row in candidates]
    if not rows:
        raise ValueError("confirmation role resolution has no candidates")

    def frozen(row: Mapping[str, Any]) -> bool:
        return row["campaign_row"]["mode"] == "frozen"

    def loss(row: Mapping[str, Any]) -> str:
        return str(row["campaign_row"]["loss_id"])

    kind = policy["kind"]
    if kind == "source_run":
        pool = [
            row
            for row in rows
            if row["source_run_id"] == policy["source_run_id"]
        ]
    elif kind == "canonical":
        pool = [
            row
            for row in rows
            if row["campaign_row"]["architecture_id"]
            == "P_HIER_DECODER_REFINE"
            and row["campaign_row"]["consumer_id"] == "C_ROBUST_MIX"
            and row["campaign_row"]["target_id"]
            == "TARGET_ALTERNATE_SELECTED"
            and row["campaign_row"]["mode"] == "frozen"
            and row["campaign_row"]["loss_id"] == "L_PRIMARY"
        ]
    elif kind == "best_architecture":
        pool = [
            row
            for row in rows
            if row["source_run_id"].startswith("ARCH_")
            and row["selectable"]
        ]
    elif kind == "best_no_ce":
        no_ce = {
            "L_VIEW",
            "L_VIEW_COS",
            "L_VIEW_REL",
            "L_VIEW_ALL",
            "L_KD",
            "L_KD_VIEW",
            "L_KD_VIEW_REL",
            "L_PRIMARY_NO_CE",
            "L_UNCERTAINTY",
        }
        pool = [
            row
            for row in rows
            if frozen(row)
            and loss(row) in no_ce
            and row["privileged_claim_eligible"]
            and row["selectable"]
        ]
    elif kind == "best_small_ce":
        pool = [
            row
            for row in rows
            if frozen(row)
            and loss(row)
            in {"L_PRIMARY", "L_PRIMARY_CE05", "L_PRIMARY_CE15"}
            and row["privileged_claim_eligible"]
            and row["selectable"]
        ]
    elif kind == "ce_only":
        preferred = [
            row
            for row in rows
            if row["source_run_id"]
            == "TRAINED_CONTROL_IDENTICAL_CE_ONLY"
        ]
        pool = preferred or [
            row for row in rows if frozen(row) and loss(row) == "L_CE"
        ]
    elif kind == "representation_only":
        pool = [
            row
            for row in rows
            if frozen(row) and loss(row) == "L_VIEW_ALL"
        ]
    elif kind == "alternative_target":
        pool = [
            row
            for row in rows
            if row["campaign_row"]["target_id"]
            == "TARGET_ALTERNATE_SELECTED"
            and row["privileged_claim_eligible"]
            and row["selectable"]
        ]
    elif kind == "recodesigned":
        pool = [
            row
            for row in rows
            if "RECODESIGN" in row["source_run_id"]
            or row["campaign_row"]["target_id"] == "VGEN_RECODESIGN"
        ]
    elif kind == "hlt_memory":
        preferred = [
            row
            for row in rows
            if row["source_run_id"]
            == "TRAINED_CONTROL_HLT_SELF_DISTILLATION"
        ]
        pool = preferred or [
            row
            for row in rows
            if row["campaign_row"]["target_id"] == "VGEN_MEMORY_HLT"
        ]
    else:  # pragma: no cover - guarded by _role_policy
        raise AssertionError(kind)
    if not pool:
        raise ValueError(f"confirmation role {role_id} has no eligible source")
    winner = min(pool, key=_candidate_order)
    return {
        **winner,
        "role_id": role_id,
        "source_kind": "distillation",
        "resolution_reason": kind,
        "diagnostic": bool(policy["diagnostic"]),
        "privileged_claim_eligible": bool(policy["privileged"]),
        "pre_stage_g_deployable_eligible": (
            not bool(policy["diagnostic"])
        ),
    }


def _candidate_records(
    root: Path,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for row in registry["runs"]:
        if row["stage"] != "predictor":
            continue
        run_id = row["run_id"]
        try:
            artifacts = _task_artifacts(root, registry, run_id, 101)
        except (FileNotFoundError, ValueError):
            continue
        registration_path = None
        for name in ("distillation_registration.json", "joint_registration.json"):
            try:
                registration_path = _artifact(artifacts, name)
                break
            except ValueError:
                continue
        if registration_path is None:
            continue
        try:
            binding_path = _artifact(
                artifacts, "distillation_runtime_binding.json"
            )
        except ValueError:
            continue
        registration = load_hashed_json(registration_path)
        binding = load_hashed_json(binding_path)
        metrics = registration["model_val_select"]
        if metrics.get("split") != "model_val_select":
            raise ValueError("candidate accessed the wrong selection split")
        checkpoint = _artifact(artifacts, registration["checkpoint_file"])
        records.append(
            {
                "configuration_id": registration["configuration_id"],
                "source_run_id": run_id,
                "campaign_row": dict(binding["campaign_row"]),
                "model_val_select": dict(metrics),
                "deployed_parameters": int(
                    registration.get(
                        "deployed_parameters",
                        1,
                    )
                ),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "registration_path": str(registration_path),
                "registration_sha256": registration["content_hash"],
                "resolved_target_run_id": binding[
                    "resolved_target_run_id"
                ],
                "resolved_target_registration_sha256": binding[
                    "resolved_target_registration_sha256"
                ],
                "consumer_registration_sha256": binding[
                    "consumer_registration_sha256"
                ],
                "predictor_initialization": binding[
                    "predictor_initialization"
                ],
                "selectable": bool(row["selectable"]),
                "diagnostic": bool(row["diagnostic"]),
                "privileged_claim_eligible": bool(
                    registration.get(
                        "privileged_claim_eligible",
                        row["selection_family"]
                        == "privileged_scientific",
                    )
                ),
                "pre_stage_g_deployable_eligible": (
                    not bool(row["diagnostic"])
                ),
            }
        )
    if not records:
        raise ValueError("no completed predictor candidates were found")
    return records


def _bundle_resource_payload(
    *,
    predictor: torch.nn.Module,
    consumer: torch.nn.Module,
    a0_architecture: Mapping[str, Any],
    configuration_id: str,
) -> dict[str, Any]:
    predictor_flops = predictor_semantic_flops(predictor, particles=128)
    a0_flops = particle_transformer_semantic_flops(a0_architecture)
    adapter_flops = particle_view_consumer_semantic_flops(
        consumer.config, particles=128
    )
    breakdown = {
        **{
            f"predictor/{key}": int(value)
            for key, value in predictor_flops["per_operator"].items()
        },
        **{f"a0/{key}": int(value) for key, value in a0_flops.items()},
        **{f"consumer/{key}": int(value) for key, value in adapter_flops.items()},
    }
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONFIRMATION_RESOURCE_CONTRACT,
            "configuration_id": configuration_id,
            "architecture_config_sha256": predictor.config.content_hash,
            "total_parameters": count_unique_parameters(
                (predictor, consumer)
            ),
            "forward_flops": {
                "counter": PARTICLE_VIEW_FLOP_COUNTER,
                "fixture_contract": "flop_fixture_v1",
                "fixture_sha256": flop_fixture_sha256(
                    input_dim=predictor.config.input_dim,
                    particles=128,
                ),
                "batch_size": 1,
                "particles": 128,
                "per_operator": dict(sorted(breakdown.items())),
                "exact_integer_total": sum(breakdown.values()),
            },
            "precision": "float32",
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )


def _publish_confirmation_outputs(
    *,
    output_dir: str,
    run_id: str,
    seed: int,
    role_id: str,
    source: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
    unified_manifest: Mapping[str, Any],
    campaign_root: str,
    registry: Mapping[str, Any],
    max_train_batches: int | None,
) -> None:
    output = Path(output_dir)
    registration_name = (
        "joint_registration.json"
        if source["campaign_row"]["mode"] != "frozen"
        else "distillation_registration.json"
    )
    registration = load_hashed_json(output / registration_name)
    checkpoint = output / registration["checkpoint_file"]
    metrics = registration["model_val_select"]
    parameters = int(resource_profile["total_parameters"])
    forward_flops = int(
        resource_profile["forward_flops"]["exact_integer_total"]
    )
    curves_name = (
        "joint_training_curves.json"
        if registration_name == "joint_registration.json"
        else "distillation_training_curves.json"
    )
    curves = load_hashed_json(output / curves_name)
    curve_rows = curves.get("rows", curves.get("epochs"))
    selected_curve = next(
        row
        for row in curve_rows
        if int(row["epoch"]) == int(registration["selected_epoch"])
    )
    updates = int(selected_curve["optimizer_updates"])
    label_steps = updates if int(
        registration.get("label_bearing_updates", 0)
    ) else 0
    train_identity = unified_manifest["logical_splits"]["train"][
        "ordered_identity_sha256"
    ]
    records: list[LabelExposureRecord] = []

    def append_record(
        *,
        source_run_id: str,
        component: str,
        stage: str,
        optimizer_steps: int,
        label_steps: int,
        labeled_examples: int,
        ce_steps: int,
        kd_steps: int,
        view_steps: int,
        retained: bool = True,
        component_seed: int | None = None,
    ) -> None:
        records.append(
            LabelExposureRecord(
                run_id=source_run_id,
                component=component,
                stage=stage,
                seed=int(seed if component_seed is None else component_seed),
                train_identity_sha256=train_identity,
                optimizer_steps=int(optimizer_steps),
                label_bearing_steps=int(label_steps),
                labeled_examples_processed=int(labeled_examples),
                ce_bearing_steps=int(ce_steps),
                teacher_kd_steps=int(kd_steps),
                view_supervision_steps=int(view_steps),
                training_flops=int(optimizer_steps) * forward_flops * 3,
                retained_in_deployable_path=bool(retained),
            )
        )

    root = Path(campaign_root)
    train_count = int(
        unified_manifest["logical_splits"]["train"]["count"]
    )

    def batches_per_epoch(batch_size: int) -> int:
        batches = (train_count + int(batch_size) - 1) // int(batch_size)
        if max_train_batches is not None:
            batches = min(batches, int(max_train_batches))
        return batches

    def selected_consumer_budget(
        registration_payload: Mapping[str, Any],
    ) -> tuple[int, int]:
        batch_size = int(
            registration_payload["training_config"]["batch_size"]
        )
        selected_epoch = int(registration_payload["selected_epoch"])
        batches = batches_per_epoch(batch_size)
        optimizer_steps = selected_epoch * batches
        examples = selected_epoch * min(
            train_count, batches * batch_size
        )
        return optimizer_steps, examples

    a0_artifacts = _task_artifacts(root, registry, "A0_VIEW", seed)
    a0 = load_hashed_json(
        _artifact(a0_artifacts, "teacher_registration.json")
    )
    append_record(
        source_run_id="A0_VIEW",
        component="hlt_particle_encoder_initialization",
        stage="baseline",
        optimizer_steps=int(a0["optimizer_updates"]),
        label_steps=int(a0["optimizer_updates"]),
        labeled_examples=(
            int(a0["selected_epoch"]) * train_count
        ),
        ce_steps=int(a0["optimizer_updates"]),
        kd_steps=0,
        view_steps=0,
        component_seed=int(a0["recipe"]["seed"]),
    )

    # The total-label-budget control includes the selected privileged target
    # construction even though those weights are not present in the deployed
    # HLT graph.  Bind the records to the exact target selected by the source
    # candidate rather than to the campaign's current rank-one alias.
    target_run_id = str(source["resolved_target_run_id"])
    target_artifacts = _task_artifacts(root, registry, target_run_id, seed)
    target_registration = load_hashed_json(
        _artifact(target_artifacts, "target_candidate_registration.json")
    )
    if (
        target_registration["content_hash"]
        != source["resolved_target_registration_sha256"]
    ):
        raise ValueError("confirmation target registration lineage changed")
    target_recipe = load_hashed_json(
        _artifact(target_artifacts, "target_discovery_recipe.json")
    )
    discovery = load_hashed_json(
        _artifact(target_artifacts, "consumer_registration.json")
    )
    discovery_updates, discovery_examples = selected_consumer_budget(
        discovery
    )
    append_record(
        source_run_id=target_run_id,
        component="oracle_view_discovery_consumer_and_generator",
        stage="target_discovery",
        optimizer_steps=discovery_updates,
        label_steps=discovery_updates,
        labeled_examples=discovery_examples,
        ce_steps=discovery_updates,
        kd_steps=discovery_updates,
        view_steps=discovery_updates,
        retained=False,
        component_seed=int(discovery["training_config"]["seed"]),
    )

    target_probe = load_hashed_json(
        _artifact(
            target_artifacts,
            "probe_consumer/consumer_registration.json",
        )
    )
    target_probe_updates, target_probe_examples = selected_consumer_budget(
        target_probe
    )
    if source["campaign_row"]["consumer_id"] != "C_TARGET_PROBE":
        append_record(
            source_run_id=target_run_id,
            component="target_ranking_probe_consumer",
            stage="target_discovery",
            optimizer_steps=target_probe_updates,
            label_steps=target_probe_updates,
            labeled_examples=target_probe_examples,
            ce_steps=target_probe_updates,
            kd_steps=0,
            view_steps=0,
            retained=False,
            component_seed=int(target_probe["training_config"]["seed"]),
        )

    if "codesign_ledger.json" in target_artifacts:
        codesign = load_hashed_json(
            _artifact(target_artifacts, "codesign_ledger.json")
        )
        projection_steps = sum(
            int(row["projection_consumer_optimizer_steps"])
            for row in codesign["cycles"]
        )
        probe_steps = sum(
            int(row["probe_optimizer_steps"])
            for row in codesign["cycles"]
        )
        append_record(
            source_run_id=target_run_id,
            component="recoverability_codesign",
            stage="target_discovery",
            optimizer_steps=projection_steps + probe_steps,
            label_steps=projection_steps,
            labeled_examples=projection_steps
            * int(codesign["config"]["batch_size"]),
            ce_steps=projection_steps,
            kd_steps=projection_steps,
            view_steps=projection_steps + probe_steps,
            retained=False,
            component_seed=int(codesign["config"]["seed"]),
        )

    # Authenticate and count the exact offline teacher(s) used to construct
    # the selected target.  HLT-memory controls point back to A0 and are
    # already represented by the retained A0 record above.
    required_teacher_hashes = {
        value
        for value in (
            target_recipe["memory_teacher_registration_sha256"],
            target_recipe.get(
                "secondary_memory_teacher_registration_sha256"
            ),
        )
        if value is not None and value != a0["content_hash"]
    }
    found_teacher_hashes: set[str] = set()
    target_row = next(
        row for row in registry["runs"] if row["run_id"] == target_run_id
    )
    for parent_run_id in target_row["parents"]:
        try:
            parent_artifacts = _task_artifacts(
                root, registry, parent_run_id, seed
            )
            teacher_path = _artifact(
                parent_artifacts, "teacher_registration.json"
            )
        except (FileNotFoundError, ValueError):
            continue
        teacher = load_hashed_json(teacher_path)
        if teacher["content_hash"] not in required_teacher_hashes:
            continue
        found_teacher_hashes.add(teacher["content_hash"])
        teacher_updates = int(teacher["optimizer_updates"])
        append_record(
            source_run_id=parent_run_id,
            component="offline_memory_teacher",
            stage="offline_teacher",
            optimizer_steps=teacher_updates,
            label_steps=teacher_updates,
            labeled_examples=int(teacher["selected_epoch"]) * train_count,
            ce_steps=teacher_updates,
            kd_steps=0,
            view_steps=0,
            retained=False,
            component_seed=int(teacher["recipe"]["seed"]),
        )
    if found_teacher_hashes != required_teacher_hashes:
        raise ValueError("selected target offline-teacher lineage is incomplete")

    consumer_id = str(source["campaign_row"]["consumer_id"])
    selected_target_consumer = (
        source["campaign_row"]["target_id"]
        == "TARGET_ALTERNATE_SELECTED"
    )
    if (
        selected_target_consumer
        and consumer_id in {"C_CLEAN", "C_ROBUST_MIX"}
    ):
        clean_artifacts = _task_artifacts(
            root, registry, "FINAL_CLEAN_CONSUMER", seed
        )
        clean = load_hashed_json(
            _artifact(clean_artifacts, "consumer_registration.json")
        )
        clean_updates, clean_examples = selected_consumer_budget(clean)
        append_record(
            source_run_id="FINAL_CLEAN_CONSUMER",
            component="clean_particle_view_consumer",
            stage="view_publication",
            optimizer_steps=clean_updates,
            label_steps=clean_updates,
            labeled_examples=clean_examples,
            ce_steps=clean_updates,
            kd_steps=0,
            view_steps=0,
            component_seed=int(clean["training_config"]["seed"]),
        )
        if consumer_id == "C_ROBUST_MIX":
            robust_artifacts = _task_artifacts(
                root, registry, "ROBUST_CONSUMER", seed
            )
            robust = load_hashed_json(
                _artifact(
                    robust_artifacts,
                    "robust_consumer_registration.json",
                )
            )
            robust_curves = load_hashed_json(
                _artifact(
                    robust_artifacts,
                    "robust_consumer_training_curves.json",
                )
            )
            robust_selected = next(
                row
                for row in robust_curves["epochs"]
                if int(row["epoch"]) == int(robust["selected_epoch"])
            )
            robust_updates = int(robust_selected["optimizer_updates"])
            append_record(
                source_run_id="ROBUST_CONSUMER",
                component="robust_particle_view_consumer",
                stage="representation",
                optimizer_steps=robust_updates,
                label_steps=robust_updates,
                labeled_examples=min(
                    robust_updates * 128,
                    int(robust["selected_epoch"]) * train_count,
                ),
                ce_steps=robust_updates,
                kd_steps=0,
                view_steps=0,
                component_seed=int(robust["train_config"]["seed"]),
            )
    elif consumer_id == "C_TARGET_PROBE":
        target_artifacts = _task_artifacts(
            root, registry, source["resolved_target_run_id"], seed
        )
        probe = load_hashed_json(
            _artifact(
                target_artifacts,
                "probe_consumer/consumer_registration.json",
            )
        )
        probe_updates, probe_examples = selected_consumer_budget(probe)
        append_record(
            source_run_id=source["resolved_target_run_id"],
            component="selected_target_probe_consumer",
            stage="target_discovery",
            optimizer_steps=probe_updates,
            label_steps=probe_updates,
            labeled_examples=probe_examples,
            ce_steps=probe_updates,
            kd_steps=0,
            view_steps=0,
            component_seed=int(probe["training_config"]["seed"]),
        )
    else:
        screen_run_id = f"SCREEN_{consumer_id}"
        screen_artifacts = _task_artifacts(
            root, registry, screen_run_id, seed
        )
        screen = load_hashed_json(
            _artifact(screen_artifacts, "consumer_registration.json")
        )
        screen_updates, screen_examples = selected_consumer_budget(screen)
        append_record(
            source_run_id=screen_run_id,
            component="selected_consumer_interface",
            stage="consumer_screen",
            optimizer_steps=screen_updates,
            label_steps=screen_updates,
            labeled_examples=screen_examples,
            ce_steps=screen_updates,
            kd_steps=0,
            view_steps=0,
            component_seed=int(screen["training_config"]["seed"]),
        )

    if source["predictor_initialization"] == "exact_pview0_checkpoint":
        pview_artifacts = _task_artifacts(root, registry, "PVIEW0", seed)
        pview = load_hashed_json(
            _artifact(pview_artifacts, "pview0_registration.json")
        )
        append_record(
            source_run_id="PVIEW0",
            component="predictor_representation_warmup",
            stage="representation",
            optimizer_steps=int(pview["optimizer_updates"]),
            label_steps=0,
            labeled_examples=0,
            ce_steps=0,
            kd_steps=0,
            view_steps=int(pview["optimizer_updates"]),
            component_seed=int(pview["warmup_config"]["seed"]),
        )

    # Joint confirmation starts from the already-selected frozen primary
    # predictor.  Its optimizer history remains encoded in the deployed
    # weights and therefore belongs in the retained-path budget.
    if registration_name == "joint_registration.json":
        parent_sha256 = registration["lineage"][
            "parent_distillation_registration_sha256"
        ]
        parent_matches: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for registry_row in registry["runs"]:
            if registry_row["stage"] != "predictor":
                continue
            parent_run_id = registry_row["run_id"]
            try:
                parent_artifacts = _task_artifacts(
                    root, registry, parent_run_id, seed
                )
                parent_registration = load_hashed_json(
                    _artifact(
                        parent_artifacts,
                        "distillation_registration.json",
                    )
                )
            except (FileNotFoundError, ValueError):
                continue
            if parent_registration["content_hash"] != parent_sha256:
                continue
            parent_curves = load_hashed_json(
                _artifact(
                    parent_artifacts,
                    "distillation_training_curves.json",
                )
            )
            parent_matches.append(
                (parent_run_id, parent_registration, parent_curves)
            )
        if len(parent_matches) != 1:
            raise ValueError(
                "joint confirmation frozen-primary lineage is ambiguous"
            )
        parent_run_id, parent_registration, parent_curves = parent_matches[0]
        parent_selected = next(
            row
            for row in parent_curves["rows"]
            if int(row["epoch"])
            == int(parent_registration["selected_epoch"])
        )
        parent_updates = int(parent_selected["optimizer_updates"])
        parent_label_steps = (
            parent_updates
            if int(parent_registration["label_bearing_updates"])
            else 0
        )
        append_record(
            source_run_id=parent_run_id,
            component="selected_frozen_primary_predictor",
            stage="distillation",
            optimizer_steps=parent_updates,
            label_steps=parent_label_steps,
            labeled_examples=(
                min(
                    parent_updates * 128,
                    int(parent_registration["selected_epoch"])
                    * train_count,
                )
                if parent_label_steps
                else 0
            ),
            ce_steps=(
                parent_updates
                if int(parent_registration["ce_bearing_updates"])
                else 0
            ),
            kd_steps=(
                parent_updates
                if int(parent_registration["teacher_kd_updates"])
                else 0
            ),
            view_steps=(
                parent_updates
                if int(parent_registration["view_supervision_updates"])
                else 0
            ),
            component_seed=int(parent_registration["seed"]),
        )

    append_record(
        source_run_id=run_id,
        component="confirmed_hlt_only_bundle",
        stage="confirmation",
        optimizer_steps=updates,
        label_steps=label_steps,
        labeled_examples=(
            min(
                updates * 128,
                int(registration["selected_epoch"]) * train_count,
            )
            if label_steps
            else 0
        ),
        ce_steps=(
            updates if int(registration.get("ce_bearing_updates", 0)) else 0
        ),
        kd_steps=(
            updates if int(registration.get("teacher_kd_updates", 0)) else 0
        ),
        view_steps=(
            updates
            if int(registration.get("view_supervision_updates", 0))
            else 0
        ),
        component_seed=int(registration["seed"]),
    )
    ledger = build_label_exposure_ledger(
        unified_split_manifest=unified_manifest,
        pipeline_id=f"{source['configuration_id']}__seed_{seed}",
        records=records,
    )
    write_immutable_json(output / "training_ledger.json", ledger)
    write_immutable_json(output / "resource_profile.json", resource_profile)
    replica = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONFIRMATION_REPLICA_CONTRACT,
            "configuration_id": source["configuration_id"],
            "run_id": run_id,
            "role_id": role_id,
            "seed": int(seed),
            "split": "model_val_select",
            "deployable_accuracy": float(metrics["deployable_accuracy"]),
            "deployable_cross_entropy": float(
                metrics["deployable_cross_entropy"]
            ),
            "recovery_status": metrics["recovery_status"],
            "recovery_fraction": metrics["recovery_fraction"],
            "oracle_gain": metrics["oracle_gain"],
            "predicted_gain": metrics.get("predicted_gain"),
            "zero_view_accuracy": metrics.get("zero_view_accuracy"),
            "deployed_parameters": parameters,
            "bundle_sha256": sha256_file(checkpoint),
            "bundle_path": str(checkpoint.resolve()),
            "source_run_id": source["source_run_id"],
            "source_registration_sha256": source[
                "registration_sha256"
            ],
            "source_resolution_reason": source["resolution_reason"],
            "training_ledger_sha256": ledger["content_hash"],
            "resource_profile_sha256": resource_profile["content_hash"],
            "privileged_claim_eligible": bool(
                source["privileged_claim_eligible"]
            ),
            "pre_stage_g_deployable_eligible": bool(
                source["pre_stage_g_deployable_eligible"]
            ),
            "diagnostic": bool(source["diagnostic"]),
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "performance_gate_used": False,
        }
    )
    write_immutable_json(output / "confirmation_replica.json", replica)


def run_confirmation_training(
    *,
    training_kind: str,
    trainer_kwargs: Mapping[str, Any],
    finalizer_kwargs: Mapping[str, Any] | None = None,
) -> None:
    if training_kind == "frozen":
        train_frozen_consumer_distillation(**dict(trainer_kwargs))
    elif training_kind == "joint":
        train_joint_finetuning(**dict(trainer_kwargs))
    elif training_kind == "direct":
        _run_direct_confirmation(**dict(trainer_kwargs))
        return
    else:
        raise ValueError("unknown confirmation training kind")
    if finalizer_kwargs is None:
        raise ValueError("trained confirmation omitted its finalizer")
    _publish_confirmation_outputs(**dict(finalizer_kwargs))


def _direct_candidate_config(config_id: str) -> Mapping[str, Any]:
    rows = [
        row
        for row in build_predeclared_direct_control_grid()["candidates"]
        if row["config_id"] == config_id
    ]
    if len(rows) != 1:
        raise ValueError("direct confirmation config resolution failed")
    return rows[0]


def _run_direct_confirmation(
    *,
    model: torch.nn.Module,
    loader: Any,
    checkpoint_source: str,
    registration: Mapping[str, Any],
    output_dir: str,
    source: Mapping[str, Any],
    unified_manifest: Mapping[str, Any],
    device: str,
    max_val_batches: int | None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto"
        else device
    )
    model = model.to(torch.device(resolved))
    metrics = evaluate_particle_view_teacher(
        model,
        loader,
        device=torch.device(resolved),
        max_batches=max_val_batches,
    )
    checkpoint = output / "selected_confirmation_bundle.pt"
    shutil.copyfile(checkpoint_source, checkpoint)
    selected = registration["recipe"]["model_config"]
    resource = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONFIRMATION_RESOURCE_CONTRACT,
            "configuration_id": source["configuration_id"],
            "architecture_config_sha256": selected["config_sha256"],
            "total_parameters": int(selected["deployed_parameters"]),
            "forward_flops": {
                "counter": PARTICLE_VIEW_FLOP_COUNTER,
                "fixture_contract": "flop_fixture_v1",
                "fixture_sha256": flop_fixture_sha256(
                    input_dim=17, particles=128
                ),
                "batch_size": 1,
                "particles": 128,
                "per_operator": {
                    "direct_hlt_part": int(selected["forward_flops"])
                },
                "exact_integer_total": int(selected["forward_flops"]),
            },
            "precision": "float32",
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    record = LabelExposureRecord(
        run_id=source["confirmation_run_id"],
        component="stage_a_direct_hlt_control",
        stage="confirmation",
        seed=int(source["seed"]),
        train_identity_sha256=unified_manifest["logical_splits"]["train"][
            "ordered_identity_sha256"
        ],
        optimizer_steps=int(registration["optimizer_updates"]),
        label_bearing_steps=int(registration["optimizer_updates"]),
        labeled_examples_processed=min(
            int(registration["optimizer_updates"]) * 128,
            int(registration["selected_epoch"])
            * int(unified_manifest["logical_splits"]["train"]["count"]),
        ),
        ce_bearing_steps=int(registration["optimizer_updates"]),
        teacher_kd_steps=0,
        view_supervision_steps=0,
        training_flops=(
            int(registration["optimizer_updates"])
            * int(selected["forward_flops"])
            * 3
        ),
        retained_in_deployable_path=True,
    )
    ledger = build_label_exposure_ledger(
        unified_split_manifest=unified_manifest,
        pipeline_id=f"{source['configuration_id']}__seed_{source['seed']}",
        records=[record],
    )
    write_immutable_json(output / "training_ledger.json", ledger)
    write_immutable_json(output / "resource_profile.json", resource)
    replica = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONFIRMATION_REPLICA_CONTRACT,
            "configuration_id": source["configuration_id"],
            "run_id": source["confirmation_run_id"],
            "role_id": source["role_id"],
            "seed": int(source["seed"]),
            "split": "model_val_select",
            "deployable_accuracy": float(metrics["accuracy"]),
            "deployable_cross_entropy": float(metrics["cross_entropy"]),
            "recovery_status": "undefined",
            "recovery_fraction": None,
            "oracle_gain": None,
            "predicted_gain": None,
            "zero_view_accuracy": None,
            "deployed_parameters": int(selected["deployed_parameters"]),
            "bundle_sha256": sha256_file(checkpoint),
            "bundle_path": str(checkpoint.resolve()),
            "source_run_id": source["source_run_id"],
            "source_registration_sha256": registration["content_hash"],
            "source_resolution_reason": source["resolution_reason"],
            "training_ledger_sha256": ledger["content_hash"],
            "resource_profile_sha256": resource["content_hash"],
            "privileged_claim_eligible": False,
            "pre_stage_g_deployable_eligible": False,
            "diagnostic": True,
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "performance_gate_used": False,
        }
    )
    write_immutable_json(output / "confirmation_replica.json", replica)


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("consumer output has no logits tensor")
    return logits


def run_structural_control_evaluation(
    *,
    predictor: torch.nn.Module,
    consumer: torch.nn.Module,
    loader: Any,
    control_id: str,
    seed: int,
    output_path: str,
    device: str,
    max_val_batches: int | None,
    source_sha256: str,
) -> None:
    resolved = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto"
        else device
    )
    predictor.to(resolved).eval()
    consumer.to(resolved).eval()
    totals = {
        "events": 0,
        "source_correct": 0,
        "controlled_correct": 0,
        "source_ce": 0.0,
        "controlled_ce": 0.0,
        "valid_particles": 0,
        "fixed_points": 0,
        "padding_preserved": True,
    }
    per_class: dict[int, dict[str, int]] = defaultdict(
        lambda: {"events": 0, "source_correct": 0, "controlled_correct": 0}
    )
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            if max_val_batches is not None and batch_index >= max_val_batches:
                break
            batch = {
                key: (
                    value.to(resolved, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in raw.items()
            }
            prediction = predictor(
                batch["features"],
                batch["lorentz_vectors"],
                batch["mask"],
            ).mean
            source_view = (
                prediction
                if control_id.startswith("PREDICTED_")
                else batch["true_view"]
            )
            controlled, audit = apply_particle_view_control(
                source_view,
                batch["mask"],
                control_id=control_id,
                seed=int(seed) + batch_index,
                particle_pt=(
                    batch["lorentz_vectors"][:, 0].square()
                    + batch["lorentz_vectors"][:, 1].square()
                ).sqrt(),
                labels=batch["labels"],
            )
            source_logits = _extract_logits(
                _consumer_forward(consumer, batch, source_view)
            )
            controlled_logits = _extract_logits(
                _consumer_forward(consumer, batch, controlled)
            )
            labels = batch["labels"]
            source_prediction = source_logits.argmax(dim=-1)
            controlled_prediction = controlled_logits.argmax(dim=-1)
            count = int(labels.numel())
            totals["events"] += count
            totals["source_correct"] += int(
                source_prediction.eq(labels).sum().item()
            )
            totals["controlled_correct"] += int(
                controlled_prediction.eq(labels).sum().item()
            )
            totals["source_ce"] += float(
                F.cross_entropy(
                    source_logits, labels, reduction="sum"
                ).item()
            )
            totals["controlled_ce"] += float(
                F.cross_entropy(
                    controlled_logits, labels, reduction="sum"
                ).item()
            )
            totals["valid_particles"] += int(audit["valid_particles"])
            totals["fixed_points"] += int(
                audit["event_permutation_fixed_points"] or 0
            )
            totals["padding_preserved"] = bool(
                totals["padding_preserved"] and audit["padding_preserved"]
            )
            for label in labels.unique().tolist():
                selected = labels == int(label)
                row = per_class[int(label)]
                row["events"] += int(selected.sum().item())
                row["source_correct"] += int(
                    source_prediction[selected].eq(labels[selected]).sum().item()
                )
                row["controlled_correct"] += int(
                    controlled_prediction[selected]
                    .eq(labels[selected])
                    .sum()
                    .item()
                )
    if totals["events"] == 0:
        raise ValueError("structural control loader is empty")
    events = totals["events"]
    source_accuracy = totals["source_correct"] / events
    controlled_accuracy = totals["controlled_correct"] / events
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STRUCTURAL_CONTROL_RESULT_CONTRACT,
            "control_id": control_id,
            "seed": int(seed),
            "split": "model_val_select",
            "source_bundle_sha256": source_sha256,
            "source_accuracy": source_accuracy,
            "controlled_accuracy": controlled_accuracy,
            "controlled_minus_source_accuracy": (
                controlled_accuracy - source_accuracy
            ),
            "source_cross_entropy": totals["source_ce"] / events,
            "controlled_cross_entropy": totals["controlled_ce"] / events,
            "events": events,
            "per_class": {
                str(label): {
                    **row,
                    "source_accuracy": row["source_correct"] / row["events"],
                    "controlled_accuracy": (
                        row["controlled_correct"] / row["events"]
                    ),
                }
                for label, row in sorted(per_class.items())
            },
            "valid_particles": totals["valid_particles"],
            "event_permutation_fixed_points": totals["fixed_points"],
            "padding_preserved": totals["padding_preserved"],
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "performance_gate_used": False,
        }
    )
    write_immutable_json(output_path, artifact)


def _prepare_structural(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    root: Path,
    output: Path,
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    control_id = run_id[len(_STRUCTURAL_PREFIX) :]
    candidates = _candidate_records(root, registry)
    source = resolve_confirmation_role(
        "CANONICAL_PREDECLARED", candidates
    )
    row = source["campaign_row"]
    runtime = dict(config["distillation_factory_config"]["runtime"])
    if runtime["device"] == "auto":
        runtime["device"] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    resources = _target_loaders(
        data=config["distillation_factory_config"]["runtime_data_config"],
        root=root,
        registry=registry,
        seed=seed,
        alias=row["target_id"],
        runtime=runtime,
    )
    view_dim = int(
        resources["target"].registration["generator_config"][
            "bottleneck_width"
        ]
    )
    consumer, _, _ = _consumer_for_row(
        root=root,
        registry=registry,
        seed=seed,
        alias=row["target_id"],
        consumer_id=row["consumer_id"],
        view_dim=view_dim,
    )
    predictor, _, _ = _pview_registration_and_model(
        root=root,
        registry=registry,
        seed=seed,
        architecture_id=row["architecture_id"],
        view_dim=view_dim,
        consumer=consumer,
    )
    checkpoint = torch.load(
        source["checkpoint_path"], map_location="cpu", weights_only=False
    )
    predictor.load_state_dict(
        checkpoint["predictor_state_dict"], strict=True
    )
    if "consumer_state_dict" in checkpoint:
        consumer.load_state_dict(checkpoint["consumer_state_dict"], strict=True)
    destination = output / "structural_control_result.json"
    return {
        "kwargs": {
            "predictor": predictor,
            "consumer": consumer,
            "loader": resources["loaders"]["model_val_select"],
            "control_id": control_id,
            "seed": int(seed),
            "output_path": str(destination),
            "device": runtime["device"],
            "max_val_batches": runtime["max_val_batches"],
            "source_sha256": source["checkpoint_sha256"],
        },
        "artifact_paths": [str(destination)],
        "action": None,
    }


def _prepare_confirmation(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    root: Path,
    output: Path,
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    role_id = run_id[len(_CONFIRM_PREFIX) :]
    source = resolve_confirmation_role(
        role_id, _candidate_records(root, registry)
    )
    data = config["distillation_factory_config"]["runtime_data_config"]
    unified = load_hashed_json(data["unified_manifest"]["path"])
    runtime = config["distillation_factory_config"]["runtime"]
    if source["source_kind"] == "direct":
        artifacts = _task_artifacts(
            root, registry, source["source_run_id"], seed
        )
        registration = load_hashed_json(
            _artifact(artifacts, "direct_control_registration.json")
        )
        checkpoint = _artifact(artifacts, "best_model_val_stop.pt")
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        config_id = registration["recipe"]["model_config"]["config_id"]
        model = _direct_model(_direct_candidate_config(config_id))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        aligned = load_aligned_logical_jet_view(data, "model_val_select")
        loader = make_logical_data_loader(
            aligned,
            mode="fixed_hlt",
            batch_size=128,
            shuffle=False,
            num_workers=runtime["num_workers"],
            seed=int(seed),
        )
        source = {
            **source,
            "role_id": role_id,
            "seed": int(seed),
            "confirmation_run_id": run_id,
            "configuration_id": f"direct::{config_id}",
        }
        return {
            "kwargs": {
                "training_kind": "direct",
                "trainer_kwargs": {
                    "model": model,
                    "loader": loader,
                    "checkpoint_source": str(checkpoint),
                    "registration": registration,
                    "output_dir": str(output),
                    "source": source,
                    "unified_manifest": unified,
                    "device": runtime["device"],
                    "max_val_batches": runtime["max_val_batches"],
                },
                "finalizer_kwargs": None,
            },
            "artifact_paths": [
                str(output / "selected_confirmation_bundle.pt"),
                str(output / "training_ledger.json"),
                str(output / "resource_profile.json"),
                str(output / "confirmation_replica.json"),
            ],
            "action": None,
        }
    row = dict(source["campaign_row"])
    prepared = build_distillation_factory(
        operation=(
            "frozen_distillation"
            if row["mode"] == "frozen"
            else "joint_finetuning"
        ),
        config=config["distillation_factory_config"],
        registry=registry,
        run_id=run_id,
        seed=int(seed),
        task_id=f"{run_id}__seed_{int(seed)}",
        output_dir=str(output),
        _row_override=row,
    )
    predictor = prepared["kwargs"]["predictor"]
    consumer = prepared["kwargs"].get(
        "frozen_consumer",
        prepared["kwargs"].get("selected_frozen_consumer"),
    )
    a0_registration, _, _ = _teacher_from_task(
        root, registry, "A0_VIEW", seed
    )
    resource = _bundle_resource_payload(
        predictor=predictor,
        consumer=consumer,
        a0_architecture=a0_registration["recipe"]["architecture"],
        configuration_id=source["configuration_id"],
    )
    finalizer = {
        "output_dir": str(output),
        "run_id": run_id,
        "seed": int(seed),
        "role_id": role_id,
        "source": source,
        "resource_profile": resource,
        "unified_manifest": unified,
        "campaign_root": str(root),
        "registry": dict(registry),
        "max_train_batches": runtime["max_train_batches"],
    }
    return {
        "kwargs": {
            "training_kind": (
                "frozen" if row["mode"] == "frozen" else "joint"
            ),
            "trainer_kwargs": prepared["kwargs"],
            "finalizer_kwargs": finalizer,
        },
        "artifact_paths": [
            *prepared["artifact_paths"],
            str(output / "training_ledger.json"),
            str(output / "resource_profile.json"),
            str(output / "confirmation_replica.json"),
        ],
        "action": None,
    }


def _publish_winner_selection(
    *,
    root: str,
    registry: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: str,
) -> None:
    campaign_root = Path(root)
    rows = []
    bindings: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    aliases: dict[str, set[str]] = defaultdict(set)
    for role_id in CONFIRMATION_ROLE_IDS:
        run_id = f"{_CONFIRM_PREFIX}{role_id}"
        for seed in (101, 202, 303):
            artifacts = _task_artifacts(
                campaign_root, registry, run_id, seed
            )
            replica_path = _artifact(
                artifacts, "confirmation_replica.json"
            )
            ledger_path = _artifact(artifacts, "training_ledger.json")
            resource_path = _artifact(artifacts, "resource_profile.json")
            replica = load_hashed_json(replica_path)
            validate_content_hash(
                replica,
                expected_contract=PARTICLE_VIEW_CONFIRMATION_REPLICA_CONTRACT,
            )
            configuration_id = replica["configuration_id"]
            aliases[configuration_id].add(role_id)
            current = bindings[configuration_id].get(seed)
            candidate = {
                "replica": replica,
                "role_id": role_id,
                "training_ledger": {
                    "path": str(ledger_path),
                    "sha256": sha256_file(ledger_path),
                },
                "resource_profile": {
                    "path": str(resource_path),
                    "sha256": sha256_file(resource_path),
                },
            }
            if current is None or role_id < current["role_id"]:
                bindings[configuration_id][seed] = candidate
    for configuration_id in sorted(bindings):
        if set(bindings[configuration_id]) != {101, 202, 303}:
            raise ValueError("confirmation configuration lacks three seeds")
        rows.extend(
            bindings[configuration_id][seed]["replica"]
            for seed in (101, 202, 303)
        )
    selection = select_particle_view_winner_families(rows)
    output = Path(output_dir)
    write_immutable_json(output / "winner_selection.json", selection)
    required = {
        selection["selected_privileged_scientific_model"][
            "configuration_id"
        ],
        selection["selected_pre_stage_g_hlt_deployable_model"][
            "configuration_id"
        ],
    }
    fairness_bindings = {
        configuration_id: {
            seed: {
                "training_ledger": bindings[configuration_id][seed][
                    "training_ledger"
                ],
                "resource_profile": bindings[configuration_id][seed][
                    "resource_profile"
                ],
            }
            for seed in (101, 202, 303)
        }
        for configuration_id in required
    }
    fairness_index = build_fairness_input_index(
        selection=selection,
        configurations=fairness_bindings,
        flop_fixture_sha256=config["flop_fixture_sha256"],
        flop_counter_sha256=config["flop_counter_sha256"],
    )
    write_immutable_json(
        output / "fairness_input_index.json", fairness_index
    )
    summary = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONFIRMATION_SUMMARY_CONTRACT,
            "winner_selection_sha256": selection["content_hash"],
            "fairness_input_index_sha256": fairness_index["content_hash"],
            "resolved_configuration_count": len(bindings),
            "replica_count": len(rows),
            "role_aliases": {
                key: sorted(value) for key, value in sorted(aliases.items())
            },
            "selected_privileged_configuration_id": selection[
                "selected_privileged_scientific_model"
            ]["configuration_id"],
            "selected_deployable_configuration_id": selection[
                "selected_pre_stage_g_hlt_deployable_model"
            ]["configuration_id"],
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "performance_gate_used": False,
        }
    )
    write_immutable_json(output / "confirmation_summary.json", summary)


def build_confirmation_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_confirmation_factory_config(config)
    validate_particle_view_registry(registry)
    if task_id != f"{run_id}__seed_{int(seed)}":
        raise ValueError("confirmation task identity changed")
    output = Path(output_dir).resolve()
    root = output.parent.parent
    if run_id.startswith(_STRUCTURAL_PREFIX):
        if operation != "structural_control_evaluation":
            raise ValueError("structural control operation changed")
        return _prepare_structural(
            config=config,
            registry=registry,
            root=root,
            output=output,
            run_id=run_id,
            seed=int(seed),
        )
    if run_id.startswith(_CONFIRM_PREFIX):
        if operation != "confirmation_training":
            raise ValueError("confirmation training operation changed")
        return _prepare_confirmation(
            config=config,
            registry=registry,
            root=root,
            output=output,
            run_id=run_id,
            seed=int(seed),
        )
    if run_id == "SELECT_WINNER_FAMILIES":
        if operation != "configuration_selection" or int(seed) != 101:
            raise ValueError("winner-selection operation changed")
        return {
            "kwargs": {
                "root": str(root),
                "registry": dict(registry),
                "config": dict(config),
                "output_dir": str(output),
            },
            "artifact_paths": [
                str(output / "winner_selection.json"),
                str(output / "fairness_input_index.json"),
                str(output / "confirmation_summary.json"),
            ],
            "action": _publish_winner_selection,
        }
    raise ValueError("unknown confirmation-stage run")


def build_confirmation_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_confirmation_factory_config(load_hashed_json(path))
    common = {
        "factory": (
            "teacher_logit_reco.local_particle_residual_field.particle_view."
            "confirmation_runtime:build_confirmation_factory"
        ),
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    specs = {
        f"{_STRUCTURAL_PREFIX}{control_id}": {
            **common,
            "operation": "structural_control_evaluation",
        }
        for control_id in STRUCTURAL_CONTROL_IDS
    }
    specs.update(
        {
            f"{_CONFIRM_PREFIX}{role_id}": {
                **common,
                "operation": "confirmation_training",
            }
            for role_id in CONFIRMATION_ROLE_IDS
        }
    )
    specs["SELECT_WINNER_FAMILIES"] = {
        **common,
        "operation": "configuration_selection",
    }
    return specs


__all__ = [
    "PARTICLE_VIEW_CONFIRMATION_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_CONFIRMATION_REPLICA_CONTRACT",
    "PARTICLE_VIEW_CONFIRMATION_RESOURCE_CONTRACT",
    "PARTICLE_VIEW_CONFIRMATION_SUMMARY_CONTRACT",
    "PARTICLE_VIEW_STRUCTURAL_CONTROL_RESULT_CONTRACT",
    "build_confirmation_factory",
    "build_confirmation_factory_config",
    "build_confirmation_task_specs",
    "resolve_confirmation_role",
    "run_confirmation_training",
    "run_structural_control_evaluation",
    "validate_confirmation_factory_config",
]
