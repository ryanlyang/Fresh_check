"""Stage-I confirmation, graph-specific controls, and bounded Stage-J scale."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (
    mean_log_selection_rejection,
)

from .contracts import (
    CONFIRMATION_PLAN_CONTRACT,
    CONFIRMATION_RESULT_CONTRACT,
    CONFIRMATION_SUMMARY_CONTRACT,
    DEPLOYABLE_EXPORT_AUDIT_CONTRACT,
    GRAPH_REGISTRY_CONTRACT,
    SCALE_EXECUTION_PLAN_CONTRACT,
    SCALE_SHORTLIST_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
    CAPACITY_CONTROL_RESULT_CONTRACT,
    SCALE_ROW_RESULT_CONTRACT,
    SCALE_COMPLETION_CONTRACT,
)


CONFIRMATION_SEEDS = (202, 303, 404)
CAPACITY_CONTROL_KINDS = ("H_MONO_PARAM", "H_MONO_FLOP", "H_PARTICLENET_PARAM")
ROLE_ORDER = (
    "H_BASE",
    "BEST_ACCURACY",
    "BEST_REJECTION",
    "BEST_PHYSICAL_AUX",
    "BEST_FEEDBACK",
    "BEST_COMBINATION",
    "H_PARTICLENET",
)


def build_confirmation_result(
    *,
    plan: Mapping[str, Any],
    row_id: str,
    classification_metrics: Mapping[str, Any],
    checkpoint_sha256: str,
    prediction_sha256: str,
    training_completion_sha256: str,
    deployable_export_sha256: str,
    deployable_export_file: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one executed confirmation row to its immutable graph definition."""

    validate_content_hash(plan, expected_contract=CONFIRMATION_PLAN_CONTRACT)
    rows = {str(row["row_id"]): row for row in plan["training_rows"]}
    if row_id not in rows:
        raise ValueError("confirmation result row is absent")
    row = rows[row_id]
    if row.get("graph_definition") is None:
        raise ValueError("confirmation row lacks an executable graph definition")
    if not str(deployable_export_file).strip():
        raise ValueError("confirmation result lacks its deployable export file")
    return with_content_hash(
        {
            "contract": CONFIRMATION_RESULT_CONTRACT,
            "schema_version": 3,
            "source": dict(source),
            "confirmation_plan_sha256": plan["content_hash"],
            **dict(row),
            "classification_metrics": dict(classification_metrics),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "prediction_sha256": require_sha256(
                prediction_sha256, name="prediction_sha256"
            ),
            "training_completion_sha256": require_sha256(
                training_completion_sha256,
                name="training_completion_sha256",
            ),
            "deployable_export_sha256": require_sha256(
                deployable_export_sha256,
                name="deployable_export_sha256",
            ),
            "deployable_export_file": str(deployable_export_file),
            "completed": True,
        }
    )


def build_confirmation_plan(
    *,
    h_base_graph_id: str,
    particle_net_graph_id: str,
    kd_graph_ids: Sequence[str],
    best_physical_aux_graph_id: str,
    best_feedback_graph_id: str,
    best_combination_graph_id: str,
    physical_kd_graph_id: str,
    retb_comparators: Mapping[str, str | None],
    parent_lock_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    graph_definitions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    additional_required_graphs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if len(kd_graph_ids) != 2 or len(set(kd_graph_ids)) != 2:
        raise ValueError("confirmation requires both distinct KD baselines")
    required = [
        ("H_BASE", h_base_graph_id),
        ("H_PARTICLENET", particle_net_graph_id),
        ("H_KD_O_BASE", kd_graph_ids[0]),
        ("H_KD_O_FULLREL", kd_graph_ids[1]),
        ("BEST_PHYSICAL_AUX", best_physical_aux_graph_id),
        ("BEST_FEEDBACK", best_feedback_graph_id),
        ("BEST_COMBINATION", best_combination_graph_id),
        ("C_PHYSICAL_KD", physical_kd_graph_id),
        *[
            (str(role), str(graph))
            for role, graph in sorted(
                (additional_required_graphs or {}).items()
            )
        ],
    ]
    graphs = [
        {
            "role": role,
            "graph_id": str(graph),
            "required": True,
            "comparator_status": "required",
        }
        for role, graph in required
    ]
    for role, graph in sorted(retb_comparators.items()):
        graphs.append(
            {
                "role": str(role),
                "graph_id": None if graph is None else str(graph),
                "required": False,
                "comparator_status": (
                    "not_applicable" if graph is None else "compatible"
                ),
            }
        )
    active = [row for row in graphs if row["graph_id"] is not None]
    if graph_definitions_by_id is not None:
        active_ids = {str(row["graph_id"]) for row in active}
        if set(graph_definitions_by_id) != active_ids:
            raise ValueError("confirmation executable graph definition coverage differs")
        for graph_id, definition in graph_definitions_by_id.items():
            if str(definition.get("graph_id")) != graph_id:
                raise ValueError("confirmation graph definition ID differs")
    rows = [
        {
            "row_id": f"CONF_{canonical_sha256([row['graph_id'], seed])[:16]}",
            "graph_id": row["graph_id"],
            "role": row["role"],
            "seed": seed,
            "training_scale": "500k",
            "fixed_budget": True,
            "performance_can_cancel": False,
            "graph_definition": (
                None
                if graph_definitions_by_id is None
                else dict(graph_definitions_by_id[str(row["graph_id"])])
            ),
        }
        for row in active
        for seed in CONFIRMATION_SEEDS
    ]
    hosd_graphs = [
        row
        for row in active
        if row["role"]
        in {"BEST_PHYSICAL_AUX", "BEST_FEEDBACK", "BEST_COMBINATION", "C_PHYSICAL_KD"}
    ]
    capacity_rows = [
        {
            "row_id": f"CAP_{canonical_sha256([row['graph_id'], kind, seed])[:16]}",
            "parent_graph_id": row["graph_id"],
            "control_kind": kind,
            "seed": seed,
            "selection_eligible": False,
            "graph_specific": True,
            "parent_graph_definition": (
                None
                if graph_definitions_by_id is None
                else dict(graph_definitions_by_id[str(row["graph_id"])])
            ),
        }
        for row in hosd_graphs
        for kind in CAPACITY_CONTROL_KINDS
        for seed in CONFIRMATION_SEEDS
    ]
    return with_content_hash(
        {
            "contract": CONFIRMATION_PLAN_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "parent_lock_hashes": {
                key: require_sha256(value, name=f"parent.{key}")
                for key, value in sorted(parent_lock_hashes.items())
            },
            "discovery_seed": 101,
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "graphs": graphs,
            "training_rows": rows,
            "capacity_control_rows": capacity_rows,
            "training_row_count": len(rows),
            "capacity_control_row_count": len(capacity_rows),
            "missing_optional_comparator_representation": "not_applicable",
            "incomplete_seed_coverage_eligible": False,
            "performance_based_termination": False,
            "all_graph_definitions_executable": graph_definitions_by_id is not None,
        }
    )


def build_confirmation_plan_from_registry(
    *,
    graph_registry: Mapping[str, Any],
    parent_lock_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(graph_registry, expected_contract=GRAPH_REGISTRY_CONTRACT)
    if graph_registry.get("source") != dict(source):
        raise ValueError("confirmation graph registry source differs")
    definitions = graph_registry["definitions_by_role"]
    retb = graph_registry["retb_comparators"]
    active_definitions = {
        str(definition["graph_id"]): dict(definition)
        for definition in definitions.values()
    }
    retb_ids = {}
    for role, definition in retb.items():
        if definition is None:
            retb_ids[role] = None
        else:
            retb_ids[role] = str(definition["graph_id"])
            active_definitions[str(definition["graph_id"])] = dict(definition)
    locks = dict(parent_lock_hashes)
    locks["graph_registry"] = graph_registry["content_hash"]
    return build_confirmation_plan(
        h_base_graph_id=definitions["H_BASE"]["graph_id"],
        particle_net_graph_id=definitions["H_PARTICLENET"]["graph_id"],
        kd_graph_ids=[
            definitions["H_KD_O_BASE"]["graph_id"],
            definitions["H_KD_O_FULLREL"]["graph_id"],
        ],
        best_physical_aux_graph_id=definitions["BEST_PHYSICAL_AUX"]["graph_id"],
        best_feedback_graph_id=definitions["BEST_FEEDBACK"]["graph_id"],
        best_combination_graph_id=definitions["BEST_COMBINATION"]["graph_id"],
        physical_kd_graph_id=definitions["C_PHYSICAL_KD"]["graph_id"],
        additional_required_graphs={
            "H_BASE_LONG": definitions["H_BASE_LONG"]["graph_id"],
        },
        retb_comparators=retb_ids,
        parent_lock_hashes=locks,
        source=source,
        graph_definitions_by_id=active_definitions,
    )


def aggregate_confirmation(
    *,
    plan: Mapping[str, Any],
    training_results: Sequence[Mapping[str, Any]],
    capacity_results: Sequence[Mapping[str, Any]],
    capacity_execution_plan: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(plan, expected_contract=CONFIRMATION_PLAN_CONTRACT)
    validate_content_hash(
        capacity_execution_plan,
        expected_contract=CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
    )
    if (
        capacity_execution_plan.get("confirmation_plan_sha256")
        != plan["content_hash"]
        or capacity_execution_plan.get("source") != dict(source)
    ):
        raise ValueError("confirmation capacity execution plan lineage differs")
    training = {row["row_id"]: row for row in training_results}
    capacity = {row["row_id"]: row for row in capacity_results}
    expected_training = {row["row_id"] for row in plan["training_rows"]}
    expected_capacity = {row["row_id"] for row in plan["capacity_control_rows"]}
    if set(training) != expected_training:
        raise ValueError("confirmation training seed coverage differs")
    if set(capacity) != expected_capacity:
        raise ValueError("confirmation capacity coverage differs")
    capacity_rows = {
        row["row_id"]: row for row in capacity_execution_plan["rows"]
    }
    if set(capacity_rows) != expected_capacity:
        raise ValueError("capacity execution row coverage differs")
    for row_id, result in capacity.items():
        validate_content_hash(
            result, expected_contract=CAPACITY_CONTROL_RESULT_CONTRACT
        )
        expected = capacity_rows[row_id]
        if (
            result.get("source") != dict(source)
            or result.get("capacity_execution_plan_sha256")
            != capacity_execution_plan["content_hash"]
            or result.get("control_graph_id") != expected["control_graph_id"]
            or result.get("control_definition") != expected["control_definition"]
            or int(result.get("seed", -1)) != int(expected["seed"])
        ):
            raise ValueError("confirmation capacity result semantics differ")
    expected_by_id = {row["row_id"]: row for row in plan["training_rows"]}
    by_graph: dict[str, list[Mapping[str, Any]]] = {}
    for row_id, result in training.items():
        validate_content_hash(
            result, expected_contract=CONFIRMATION_RESULT_CONTRACT
        )
        expected = expected_by_id[row_id]
        if (
            result.get("graph_id") != expected["graph_id"]
            or int(result.get("seed", -1)) != expected["seed"]
            or result.get("source") != dict(source)
            or result.get("confirmation_plan_sha256") != plan["content_hash"]
            or result.get("graph_definition") != expected["graph_definition"]
        ):
            raise ValueError("confirmation result semantics differ")
        by_graph.setdefault(expected["graph_id"], []).append(result)
    graph_summaries = []
    for graph_id, rows in sorted(by_graph.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(CONFIRMATION_SEEDS):
            raise ValueError("confirmation graph has incomplete seed coverage")
        accuracies = [
            float(row["classification_metrics"]["macro_per_class_accuracy"])
            for row in rows
        ]
        rejections = [
            mean_log_selection_rejection(row["classification_metrics"])
            for row in rows
        ]
        graph_summaries.append(
            {
                "graph_id": graph_id,
                "seeds": list(CONFIRMATION_SEEDS),
                "mean_balanced_accuracy": math.fsum(accuracies) / len(accuracies),
                "sample_std_balanced_accuracy": _sample_std(accuracies),
                "mean_log_rejection": math.fsum(rejections) / len(rejections),
                "sample_std_mean_log_rejection": _sample_std(rejections),
                "result_hashes": [row["content_hash"] for row in rows],
                "complete": True,
                "graph_definition": dict(
                    next(
                        item["graph_definition"]
                        for item in plan["training_rows"]
                        if item["graph_id"] == graph_id
                    )
                )
                if next(
                    item["graph_definition"]
                    for item in plan["training_rows"]
                    if item["graph_id"] == graph_id
                )
                is not None
                else None,
                "roles": sorted(
                    item["role"]
                    for item in plan["graphs"]
                    if item["graph_id"] == graph_id
                ),
            }
        )
    return with_content_hash(
        {
            "contract": CONFIRMATION_SUMMARY_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "confirmation_plan_sha256": plan["content_hash"],
            "graph_summaries": graph_summaries,
            "training_result_hashes": {
                key: training[key]["content_hash"] for key in sorted(training)
            },
            "capacity_result_hashes": {
                key: capacity[key]["content_hash"] for key in sorted(capacity)
            },
            "capacity_execution_plan_sha256": capacity_execution_plan[
                "content_hash"
            ],
            "all_required_seeds_complete": True,
            "all_capacity_controls_complete": True,
            "negative_results_reported": True,
        }
    )


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = math.fsum(values) / len(values)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def build_scale_shortlist(
    *,
    confirmation_summary: Mapping[str, Any],
    role_graph_ids: Mapping[str, str] | None,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        confirmation_summary, expected_contract=CONFIRMATION_SUMMARY_CONTRACT
    )
    summaries = {
        row["graph_id"]: row
        for row in confirmation_summary["graph_summaries"]
        if bool(row["complete"])
    }
    if role_graph_ids is None:
        derived = {}
        role_alias = {
            "BEST_PHYSICAL_AUX": "BEST_PHYSICAL_AUX",
            "BEST_FEEDBACK": "BEST_FEEDBACK",
            "BEST_COMBINATION": "BEST_COMBINATION",
            "H_BASE": "H_BASE",
            "H_PARTICLENET": "H_PARTICLENET",
        }
        for summary in summaries.values():
            for role in summary.get("roles", ()):
                if role in role_alias:
                    derived[role_alias[role]] = summary["graph_id"]
        role_graph_ids = derived
    required_roles = {
        "H_BASE",
        "BEST_PHYSICAL_AUX",
        "BEST_FEEDBACK",
        "BEST_COMBINATION",
        "H_PARTICLENET",
    }
    if not required_roles.issubset(role_graph_ids):
        raise ValueError("scale role registry is incomplete")
    accuracy_best = max(
        summaries.values(),
        key=lambda row: (
            row["mean_balanced_accuracy"],
            row["mean_log_rejection"],
            row["graph_id"],
        ),
    )["graph_id"]
    rejection_best = max(
        summaries.values(),
        key=lambda row: (
            row["mean_log_rejection"],
            row["mean_balanced_accuracy"],
            row["graph_id"],
        ),
    )["graph_id"]
    ordered_roles = [
        ("H_BASE", role_graph_ids["H_BASE"]),
        ("BEST_ACCURACY", accuracy_best),
        ("BEST_REJECTION", rejection_best),
        ("BEST_PHYSICAL_AUX", role_graph_ids["BEST_PHYSICAL_AUX"]),
        ("BEST_FEEDBACK", role_graph_ids["BEST_FEEDBACK"]),
        ("BEST_COMBINATION", role_graph_ids["BEST_COMBINATION"]),
        ("H_PARTICLENET", role_graph_ids["H_PARTICLENET"]),
    ]
    first_role, roles_by_graph = {}, {}
    for role, graph in ordered_roles:
        first_role.setdefault(graph, role)
        roles_by_graph.setdefault(graph, []).append(role)
    ranked = sorted(
        first_role,
        key=lambda graph: (
            -float(summaries[graph]["mean_balanced_accuracy"]),
            -float(summaries[graph]["mean_log_rejection"]),
            ROLE_ORDER.index(first_role[graph]),
            graph,
        ),
    )[:7]
    return with_content_hash(
        {
            "contract": SCALE_SHORTLIST_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "confirmation_summary_sha256": confirmation_summary["content_hash"],
            "graphs": [
                {
                    "graph_id": graph,
                    "roles": roles_by_graph[graph],
                    "confirmation": summaries[graph],
                    "graph_definition": summaries[graph].get("graph_definition"),
                }
                for graph in ranked
            ],
            "graph_count": len(ranked),
            "hard_maximum": 7,
            "duplicate_free": True,
            "all_negative_campaign_still_shortlisted": True,
            "performance_can_disable_scale": False,
            "locked_after_complete_confirmation": True,
        }
    )


def build_scale_execution_plan(
    *,
    shortlist: Mapping[str, Any],
    scale_train_manifest_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(shortlist, expected_contract=SCALE_SHORTLIST_CONTRACT)
    if any(row.get("graph_definition") is None for row in shortlist["graphs"]):
        raise ValueError("scale shortlist lacks executable graph definitions")
    teacher_rows = [
        {
            "teacher_id": teacher,
            "seed": 101,
            "population": "target_scale",
            "fit_split": "scale_train",
            "retrain_from_scratch": True,
            "worker": "train_hosd_offline_teacher",
        }
        for teacher in ("O_BASE", "O_FULLREL")
    ]
    graph_definitions = [
        {
            "graph_id": row["graph_id"],
            "graph_definition": row.get("graph_definition"),
            "epoch_selection_split": "val_stop",
            "architecture_redesign_allowed": False,
        }
        for row in shortlist["graphs"]
    ]
    graph_rows = [
        {
            **row,
            "seed": seed,
            "retrain_from_scratch": True,
        }
        for row in graph_definitions
        for seed in CONFIRMATION_SEEDS
    ]
    target_parameterizations: dict[str, set[str]] = {}
    def register_target(target_id: str, parameterization: str) -> None:
        target_parameterizations.setdefault(str(target_id), set()).add(
            str(parameterization)
        )
    for row in graph_definitions:
        definition = row["graph_definition"]
        kind = definition["graph_kind"]
        needs_native_relations = (
            kind == "BASELINE"
            and definition.get("baseline_id") == "H_NATIVE_REL_AUX"
        ) or (
            kind == "COMBINATION"
            and definition["graph"].get("native_relation_auxiliary")
            is not None
        )
        if needs_native_relations:
            from .native_relations import native_relation_target_ids

            for native_target_id in native_relation_target_ids():
                register_target(native_target_id, "ABS")
        if kind in {"AUXILIARY", "FEEDBACK"}:
            register_target(
                definition["row"]["target_id"],
                definition["row"].get("parameterization", "ABS"),
            )
        elif kind == "COMBINATION":
            for member in definition["graph"]["members"]:
                register_target(
                    member["target_id"], member.get("parameterization", "ABS")
                )
    target_refit_rows = [
        {
            "target_id": target_id,
            "required_parameterizations": sorted(
                target_parameterizations[target_id]
            ),
            "fit_split": "scale_train",
            "fitting_population": "target_scale",
            "build_canonical_cache": (
                target_id.startswith("T_OFFLINE_")
                and not target_id.startswith(
                    ("T_OFFLINE_LOGITS_", "T_OFFLINE_POOLED_")
                )
            ),
            "consume_scale_teacher_cache": target_id.startswith(
                ("T_OFFLINE_LOGITS_", "T_OFFLINE_POOLED_")
            ),
            "build_hlt_analogue": (
                target_id.startswith("T_HLT_")
                or (
                    target_id.startswith("T_OFFLINE_")
                    and not target_id.startswith(
                        ("T_OFFLINE_LOGITS_", "T_OFFLINE_POOLED_")
                    )
                )
            ),
            "build_residual_cache": (
                "RES" in target_parameterizations[target_id]
            ),
            "fit_target_normalizer": True,
            "fit_residual_normalizer": (
                "RES" in target_parameterizations[target_id]
            ),
            "fit_heteroscedastic_metadata": (
                "HET" in target_parameterizations[target_id]
            ),
            "fit_latent_whitening": target_id == "T_OFFLINE_POOLED_LATENT",
            "worker_chain": [
                "build_hosd_targets",
                "infer_hosd_teacher_targets"
                if target_id.startswith(("T_OFFLINE_LOGITS_", "T_OFFLINE_POOLED_"))
                else "build_hosd_target_derivatives",
                "fit_hosd_target_normalizers",
            ],
        }
        for target_id in sorted(target_parameterizations)
    ]
    return with_content_hash(
        {
            "contract": SCALE_EXECUTION_PLAN_CONTRACT,
            "schema_version": 3,
            "source": dict(source),
            "scale_shortlist_sha256": shortlist["content_hash"],
            "scale_train_manifest_sha256": require_sha256(
                scale_train_manifest_sha256, name="scale_train_manifest_sha256"
            ),
            "teacher_rows": teacher_rows,
            "target_refit_rows": target_refit_rows,
            "pre_student_artifacts": [
                "scale_inputs",
                "scale_trees",
                "scale_relation_and_region_normalizers",
                "scale_teacher_locks",
                "scale_teacher_adapter_configs",
                "scale_teacher_outputs",
                "scale_target_caches",
                "scale_target_normalizers",
                "scale_residual_statistics",
                "scale_latent_whitening",
                "scale_native_relation_target_wave",
            ],
            "graph_rows": graph_rows,
            "graph_definitions": graph_definitions,
            "graph_count": len(graph_definitions),
            "training_row_count": len(graph_rows),
            "student_seeds": list(CONFIRMATION_SEEDS),
            "target_refit_row_count": len(target_refit_rows),
            "all_statistics_refit_from_scale_train": True,
            "student_training_before_target_completion_allowed": False,
            "performance_can_disable_scale": False,
        }
    )


def build_scale_row_result(
    *,
    scale_plan: Mapping[str, Any],
    graph_id: str,
    seed: int,
    checkpoint_sha256: str,
    deployable_export_sha256: str,
    classification_metrics: Mapping[str, Any],
    analytical_forward_flops_by_role: Mapping[str, int],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate one 3M student fit and its HLT-only export."""

    validate_content_hash(
        scale_plan, expected_contract=SCALE_EXECUTION_PLAN_CONTRACT
    )
    matches = [
        row
        for row in scale_plan["graph_rows"]
        if row["graph_id"] == graph_id and int(row["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError("scale row is absent or duplicated")
    required_flop_roles = {
        "model_train",
        "val_stop",
        "design_confirm",
    }
    if (
        set(analytical_forward_flops_by_role) != required_flop_roles
        or any(
            int(value) <= 0
            for value in analytical_forward_flops_by_role.values()
        )
    ):
        raise ValueError("scale row analytical forward FLOP roles differ")
    return with_content_hash(
        {
            "contract": SCALE_ROW_RESULT_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "scale_execution_plan_sha256": scale_plan["content_hash"],
            **dict(matches[0]),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "deployable_export_sha256": require_sha256(
                deployable_export_sha256,
                name="deployable_export_sha256",
            ),
            "classification_metrics": dict(classification_metrics),
            "analytical_forward_flops_by_role": {
                role: int(analytical_forward_flops_by_role[role])
                for role in (
                    "model_train",
                    "val_stop",
                    "design_confirm",
                )
            },
            "completed": True,
        }
    )


def build_scale_completion(
    *,
    scale_plan: Mapping[str, Any],
    teacher_completion_hashes: Mapping[str, str],
    target_completion_hashes: Mapping[str, str],
    pre_student_artifact_hashes: Mapping[str, str],
    graph_results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exhaustive teacher, target, graph, seed, and export coverage."""

    validate_content_hash(
        scale_plan, expected_contract=SCALE_EXECUTION_PLAN_CONTRACT
    )
    expected_teachers = {
        str(row["teacher_id"]) for row in scale_plan["teacher_rows"]
    }
    expected_targets = {
        str(row["target_id"]) for row in scale_plan["target_refit_rows"]
    }
    if set(teacher_completion_hashes) != expected_teachers:
        raise ValueError("scale teacher completion coverage differs")
    if set(target_completion_hashes) != expected_targets:
        raise ValueError("scale target completion coverage differs")
    if set(pre_student_artifact_hashes) != {
        "scale_inputs",
        "scale_trees",
        "scale_normalizers",
        "teacher_lock",
        "teacher_adapters",
        "teacher_outputs",
        "target_wave",
        "scale_native_relations",
        "graph_wave",
    }:
        raise ValueError("scale pre-student artifact coverage differs")
    expected_rows = {
        (str(row["graph_id"]), int(row["seed"]))
        for row in scale_plan["graph_rows"]
    }
    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for result in graph_results:
        validate_content_hash(
            result, expected_contract=SCALE_ROW_RESULT_CONTRACT
        )
        key = (str(result["graph_id"]), int(result["seed"]))
        if (
            key in observed
            or result.get("source") != dict(source)
            or result.get("scale_execution_plan_sha256")
            != scale_plan["content_hash"]
        ):
            raise ValueError("scale graph result lineage differs")
        observed[key] = result
    if set(observed) != expected_rows:
        raise ValueError("scale graph/seed coverage differs")
    return with_content_hash(
        {
            "contract": SCALE_COMPLETION_CONTRACT,
            "schema_version": 3,
            "source": dict(source),
            "scale_execution_plan_sha256": scale_plan["content_hash"],
            "teacher_completion_hashes": {
                key: require_sha256(
                    value, name=f"teacher_completion.{key}"
                )
                for key, value in sorted(teacher_completion_hashes.items())
            },
            "target_completion_hashes": {
                key: require_sha256(
                    value, name=f"target_completion.{key}"
                )
                for key, value in sorted(target_completion_hashes.items())
            },
            "pre_student_artifact_hashes": {
                key: require_sha256(
                    value, name=f"pre_student_artifact.{key}"
                )
                for key, value in sorted(pre_student_artifact_hashes.items())
            },
            "graph_result_hashes": {
                f"{graph_id}__seed_{seed}": observed[
                    (graph_id, seed)
                ]["content_hash"]
                for graph_id, seed in sorted(observed)
            },
            "all_shortlisted_graphs_and_seeds_complete": True,
            "negative_performance_can_cancel": False,
        }
    )


def build_deployable_export_audit(
    *,
    scale_plan: Mapping[str, Any],
    export_rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(scale_plan, expected_contract=SCALE_EXECUTION_PLAN_CONTRACT)
    by_row = {}
    for row in export_rows:
        descriptor = row.get("descriptor", row)
        key = (str(descriptor["graph_id"]), int(descriptor["seed"]))
        if key in by_row:
            raise ValueError("scale export graph/seed is duplicated")
        by_row[key] = row
    expected = {
        (str(row["graph_id"]), int(row["seed"]))
        for row in scale_plan["graph_rows"]
    }
    if set(by_row) != expected:
        raise ValueError("scale export graph/seed coverage differs")
    for graph_seed, row in by_row.items():
        if (
            row.get("source") != dict(source)
            or not bool(row.get("hlt_only"))
            or row.get("forbidden_runtime_dependencies")
            or not bool(row.get("research_export_logits_parity"))
        ):
            raise ValueError(f"scale export is not deployable: {graph_seed}")
    return with_content_hash(
        {
            "contract": DEPLOYABLE_EXPORT_AUDIT_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "scale_execution_plan_sha256": scale_plan["content_hash"],
            "export_hashes": {
                f"{graph}__seed_{seed}": by_row[(graph, seed)]["content_hash"]
                for graph, seed in sorted(by_row)
            },
            "all_shortlisted_exports_valid": True,
            "all_shortlisted_seeds_exported": True,
            "hlt_only": True,
        }
    )


__all__ = [
    "CAPACITY_CONTROL_KINDS",
    "CONFIRMATION_SEEDS",
    "aggregate_confirmation",
    "build_confirmation_plan",
    "build_confirmation_plan_from_registry",
    "build_confirmation_result",
    "build_deployable_export_audit",
    "build_scale_completion",
    "build_scale_execution_plan",
    "build_scale_row_result",
    "build_scale_shortlist",
]
