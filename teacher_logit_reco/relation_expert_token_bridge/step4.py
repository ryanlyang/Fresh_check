"""Immutable RETB Step-4 offline-expert screen and optimization registries."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    bind_source,
    canonical_sha256,
    require_sha256,
    source_record,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .expert_model import expert_relation_family
from .expert_training import (
    EXPERT_LOSS_CANDIDATES,
    INITIALIZATION_MODES,
    REGISTERED_LEARNING_RATES,
    REGISTERED_PARTICLE_DROPOUTS,
    OfflineExpertTrainingConfig,
    build_expert_loss_registry,
    validate_expert_loss_registry,
)
from .registry import EXPERT_ORDER, resolve_run_id
from .token_shape_registry import resolve_uniform_shape


STEP4_OPTIMIZATION_REGISTRY_CONTRACT = "retb_expert_optimization_registry_v1"
STEP4_RUN_REGISTRY_CONTRACT = "retb_stage_b_run_registry_v1"
STEP4_BUNDLE_CONTRACT = "retb_step4_offline_expert_bundle_v1"
STEP4_REPORT_CONTRACT = "retb_step4_report_v1"
STEP4_SELECTION_CONTRACT = "retb_step4_optimization_selection_v1"
STEP4_OPTIMIZATION_METRICS_CONTRACT = (
    "retb_step4_optimization_candidate_metrics_v1"
)
STEP4_MINIATURE_COMPLETION_CONTRACT = "retb_step4_miniature_completion_v1"

SHAPE_SCREEN_ORDER = (
    "S1_128",
    "S2_128",
    "S4_128",
    "S8_128",
    "S16_128",
    "S8_64",
    "S16_64",
)
OPTIMIZATION_EXPERTS = ("BASE4", "PT", "TRACK", "REGION")
FULL_OPTIMIZATION_EXPERTS = ("PT", "TRACK")
ROW_ROLES = (
    "reference_baseline",
    "capacity_control",
    "architecture_control",
    "scientific_candidate",
    "semantic_control",
    "robustness_control",
)


def _configuration(
    *,
    expert_id: str,
    shape_id: str,
    topology: str = "B_CONCAT",
    tokenizer_mode: str = "TOK_CANONICAL",
    loss_id: str = "ELOSS_CE",
    initialization: str = "INIT_SCRATCH",
    learning_rate: float = 1.0e-3,
    particle_dropout: float = 0.0,
    screen_name: str,
    measurement_embedding: bool = False,
) -> dict[str, Any]:
    token_count, token_dimension = resolve_uniform_shape(shape_id)
    return {
        "screen_name": str(screen_name),
        "expert_id": str(expert_id),
        "relation_family": expert_relation_family(expert_id),
        "all_particle_fields": True,
        "base4_present": True,
        "shape_id": str(shape_id),
        "token_count": token_count,
        "token_dimension": token_dimension,
        "topology": str(topology),
        "tokenizer_mode": str(tokenizer_mode),
        "loss_id": str(loss_id),
        "initialization": str(initialization),
        "learning_rate": float(learning_rate),
        "particle_dropout": float(particle_dropout),
        "measurement_embedding": bool(measurement_embedding),
        "epochs": 40,
        "checkpoint_selection": "val_stop_only",
        "architecture_selection": "val_design_only_after_training",
        "performance_based_termination": False,
    }


def _row(
    *,
    component: str,
    role: str,
    configuration: Mapping[str, Any],
    seed: int = 101,
    selection_eligible: bool,
    reuse_primary_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if role not in ROW_ROLES:
        raise ValueError("Stage-B row role is not registered")
    run_id = (
        str(reuse_primary_row["run_id"])
        if reuse_primary_row is not None
        else resolve_run_id(
            stage="B",
            component=component,
            seed=seed,
            configuration=configuration,
        )
    )
    return {
        "run_id": run_id,
        "stage": "B",
        "component": component,
        "role": role,
        "seed": int(seed),
        "selection_eligible": bool(selection_eligible),
        "reuses_primary_run": reuse_primary_row is not None,
        "configuration": (
            dict(reuse_primary_row["configuration"])
            if reuse_primary_row is not None
            else dict(configuration)
        ),
        "screen_membership_configuration": dict(configuration),
    }


def build_primary_shape_screen_rows() -> list[dict[str, Any]]:
    rows = []
    for expert_id in EXPERT_ORDER:
        for shape_id in SHAPE_SCREEN_ORDER:
            configuration = _configuration(
                expert_id=expert_id,
                shape_id=shape_id,
                screen_name="PRIMARY_49_CE_B_CONCAT",
            )
            rows.append(
                _row(
                    component="TOKEN_SHAPE",
                    role="scientific_candidate",
                    configuration=configuration,
                    selection_eligible=True,
                )
            )
    if len(rows) != 49 or len({row["run_id"] for row in rows}) != 49:
        raise RuntimeError("primary Stage-B screen must contain 49 unique rows")
    return rows


def _primary_lookup(
    primary: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (
            str(row["configuration"]["expert_id"]),
            str(row["configuration"]["shape_id"]),
        ): row
        for row in primary
    }


def build_tokenizer_control_rows(
    primary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = _primary_lookup(primary)
    rows = []
    experts = ("BASE4", "PT", "TRACK")
    shapes = ("S1_128", "S8_128", "S16_128")
    for expert_id in experts:
        for shape_id in shapes:
            canonical = _configuration(
                expert_id=expert_id,
                shape_id=shape_id,
                screen_name="TOKENIZER_CONTROL",
            )
            rows.append(
                _row(
                    component="TOKEN_SHAPE",
                    role="scientific_candidate",
                    configuration=canonical,
                    selection_eligible=True,
                    reuse_primary_row=lookup[(expert_id, shape_id)],
                )
            )
            for mode in ("TOK_WEAVER_CLASS", "TOK_K_QUERY_NO_SELF"):
                configuration = _configuration(
                    expert_id=expert_id,
                    shape_id=shape_id,
                    tokenizer_mode=mode,
                    screen_name="TOKENIZER_CONTROL",
                )
                configuration["token_target_eligible"] = mode != "TOK_WEAVER_CLASS"
                configuration["ordinary_weaver_head"] = mode == "TOK_WEAVER_CLASS"
                rows.append(
                    _row(
                        component="TOKEN_SHAPE",
                        role="architecture_control",
                        configuration=configuration,
                        selection_eligible=False,
                    )
                )
        for mode in ("TOK_MASKED_MEAN", "TOK_ONE_QUERY_NO_SELF"):
            configuration = _configuration(
                expert_id=expert_id,
                shape_id="S1_128",
                tokenizer_mode=mode,
                screen_name="TOKENIZER_CONTROL",
            )
            rows.append(
                _row(
                    component="TOKEN_SHAPE",
                    role="architecture_control",
                    configuration=configuration,
                    selection_eligible=False,
                )
            )
    for expert_id in ("BASE4", "PT", "TRACK", "REGION"):
        configuration = _configuration(
            expert_id=expert_id,
            shape_id="S8_128",
            tokenizer_mode="TOK_MULTI_DEPTH",
            screen_name="MULTI_DEPTH_CONTROL",
        )
        rows.append(
            _row(
                component="TOKEN_SHAPE",
                role="architecture_control",
                configuration=configuration,
                selection_eligible=False,
            )
        )
    return rows


def build_dual_topology_rows() -> list[dict[str, Any]]:
    rows = []
    for expert_id in EXPERT_ORDER:
        for topology in ("B_DUAL_FIXED", "B_DUAL_GATED"):
            configuration = _configuration(
                expert_id=expert_id,
                shape_id="S8_128",
                topology=topology,
                screen_name="DUAL_TOPOLOGY_CONTROL",
            )
            if expert_id == "BASE4":
                configuration["dual_base4_capacity_control"] = True
            rows.append(
                _row(
                    component="OFFLINE_EXPERT",
                    role="architecture_control",
                    configuration=configuration,
                    selection_eligible=False,
                )
            )
    if len(rows) != 14:
        raise RuntimeError("dual topology screen must contain 14 rows")
    return rows


def build_expert_loss_rows(
    primary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = _primary_lookup(primary)
    rows = []
    for expert_id in ("BASE4", "PT", "TRACK", "REGION"):
        for loss_id in EXPERT_LOSS_CANDIDATES:
            configuration = _configuration(
                expert_id=expert_id,
                shape_id="S8_128",
                loss_id=loss_id,
                screen_name="REPRESENTATIVE_EXPERT_LOSS",
            )
            reuse = (
                lookup[(expert_id, "S8_128")]
                if loss_id == "ELOSS_CE"
                else None
            )
            rows.append(
                _row(
                    component="OFFLINE_EXPERT",
                    role="scientific_candidate",
                    configuration=configuration,
                    selection_eligible=True,
                    reuse_primary_row=(
                        lookup[(expert_id, "S8_128")]
                        if reuse is not None
                        else None
                    ),
                )
            )
    return rows


def build_full_optimization_rows(
    primary: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    primary_rows = (
        build_primary_shape_screen_rows() if primary is None else list(primary)
    )
    lookup = _primary_lookup(primary_rows)
    rows = []
    for expert_id in FULL_OPTIMIZATION_EXPERTS:
        for initialization in INITIALIZATION_MODES:
            for learning_rate in REGISTERED_LEARNING_RATES:
                for dropout in REGISTERED_PARTICLE_DROPOUTS:
                    configuration = _configuration(
                        expert_id=expert_id,
                        shape_id="S8_128",
                        initialization=initialization,
                        learning_rate=learning_rate,
                        particle_dropout=dropout,
                        screen_name="FULL_OPTIMIZATION_GRID",
                    )
                    rows.append(
                        _row(
                            component="OPTIMIZATION_CONTROL",
                            role="scientific_candidate",
                            configuration=configuration,
                            selection_eligible=True,
                            reuse_primary_row=(
                                lookup[(expert_id, "S8_128")]
                                if initialization == "INIT_SCRATCH"
                                and learning_rate == 1.0e-3
                                and dropout == 0.0
                                else None
                            ),
                        )
                    )
    if len(rows) != 36:
        raise RuntimeError("PT/TRACK optimization grid must contain 36 rows")
    return rows


def build_fixed_followup_reference_rows(
    primary: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    primary_rows = (
        build_primary_shape_screen_rows() if primary is None else list(primary)
    )
    lookup = _primary_lookup(primary_rows)
    rows = []
    for expert_id in ("BASE4", "REGION"):
        for initialization, role_name in (
            ("INIT_SCRATCH", "SCRATCH_REFERENCE"),
            ("INIT_ATTACH_AFTER_PRETRAIN", "ATTACH_REFERENCE"),
        ):
            configuration = _configuration(
                expert_id=expert_id,
                shape_id="S8_128",
                initialization=initialization,
                learning_rate=1.0e-3,
                particle_dropout=0.0,
                screen_name=role_name,
            )
            rows.append(
                _row(
                    component="OPTIMIZATION_CONTROL",
                    role="architecture_control",
                    configuration=configuration,
                    selection_eligible=False,
                    reuse_primary_row=(
                        lookup[(expert_id, "S8_128")]
                        if initialization == "INIT_SCRATCH"
                        else None
                    ),
                )
            )
    return rows


def materialize_optimization_winner_followups(
    selected_configuration: Mapping[str, Any],
    *,
    existing_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    required = {"initialization", "learning_rate", "particle_dropout"}
    if not required.issubset(selected_configuration):
        raise ValueError("optimization winner lacks its three screened controls")
    initialization = str(selected_configuration["initialization"])
    learning_rate = float(selected_configuration["learning_rate"])
    dropout = float(selected_configuration["particle_dropout"])
    if initialization not in INITIALIZATION_MODES:
        raise ValueError("optimization winner initialization is unregistered")
    if learning_rate not in REGISTERED_LEARNING_RATES:
        raise ValueError("optimization winner learning rate is unregistered")
    if dropout not in REGISTERED_PARTICLE_DROPOUTS:
        raise ValueError("optimization winner dropout is unregistered")
    rows = []
    for expert_id in ("BASE4", "REGION"):
        configuration = _configuration(
            expert_id=expert_id,
            shape_id="S8_128",
            initialization=initialization,
            learning_rate=learning_rate,
            particle_dropout=dropout,
            screen_name="OPTIMIZATION_WINNER_REPLAY",
        )
        rows.append(
            _row(
                component="OPTIMIZATION_CONTROL",
                role="scientific_candidate",
                configuration=configuration,
                selection_eligible=True,
                reuse_primary_row=next(
                    (
                        row
                        for row in existing_rows
                        if all(
                            row["configuration"].get(name)
                            == configuration.get(name)
                            for name in (
                                "expert_id",
                                "shape_id",
                                "topology",
                                "tokenizer_mode",
                                "loss_id",
                                "initialization",
                                "learning_rate",
                                "particle_dropout",
                                "measurement_embedding",
                            )
                        )
                    ),
                    None,
                ),
            )
        )
    return rows


def build_optimization_registry() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": STEP4_OPTIMIZATION_REGISTRY_CONTRACT,
            "schema_version": 1,
            "primary_reference": {
                "initialization": "INIT_SCRATCH",
                "learning_rate": 1.0e-3,
                "particle_dropout": 0.0,
            },
            "initializations": list(INITIALIZATION_MODES),
            "learning_rates": list(REGISTERED_LEARNING_RATES),
            "particle_dropouts": list(REGISTERED_PARTICLE_DROPOUTS),
            "dropout_scope": (
                "particle_block_residual_and_activation_only;"
                "particle_attention_dropout_zero;"
                "tokenizer_contract_unchanged"
            ),
            "full_cartesian_experts": list(FULL_OPTIMIZATION_EXPERTS),
            "optimization_candidate_unit": (
                "one_initialization_learning_rate_dropout_tuple_paired_across_PT_TRACK"
            ),
            "paired_aggregation": {
                "required_experts": list(FULL_OPTIMIZATION_EXPERTS),
                "accuracy": "arithmetic_mean",
                "cross_entropy": "arithmetic_mean",
                "measured_flops": "arithmetic_mean",
                "parameter_count": "rounded_arithmetic_mean",
                "complete_36_row_grid_required": True,
                "candidate_count": 18,
                "candidate_metrics_contract": (
                    STEP4_OPTIMIZATION_METRICS_CONTRACT
                ),
            },
            "followup_experts": ["BASE4", "REGION"],
            "followup_rows": [
                "selected_winner",
                "scratch_reference",
                "attach_after_pretrain_reference",
            ],
            "attachment": {
                "pretraining_parent_required": True,
                "epochs_1_through_5": "ordinary_particle_backbone_frozen",
                "epochs_6_through_10": "last_four_particle_blocks_trainable",
                "epochs_11_through_40": "complete_graph_trainable",
                "attachment_epochs": 40,
                "pretraining_cost_recorded": True,
                "long_baseline_matching_required": True,
            },
            "selector": {
                "split": "val_design",
                "accuracy_window": 0.0001,
                "order": [
                    "maximum_accuracy",
                    "within_global_0p0001_window",
                    "minimum_cross_entropy",
                    "minimum_measured_flops",
                    "minimum_parameter_count",
                    "lexicographically_smaller_run_id",
                ],
                "always_emit_best_available": True,
                "negative_result_stops_campaign": False,
            },
        }
    )


def validate_optimization_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STEP4_OPTIMIZATION_REGISTRY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_optimization_registry()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Step-4 optimization registry differs")
    return digest


def build_stage_b_run_registry() -> dict[str, Any]:
    primary = build_primary_shape_screen_rows()
    tokenizer = build_tokenizer_control_rows(primary)
    dual = build_dual_topology_rows()
    losses = build_expert_loss_rows(primary)
    optimization = build_full_optimization_rows(primary)
    followup = build_fixed_followup_reference_rows(primary)
    rows = primary + tokenizer + dual + losses + optimization + followup
    unique_physical_runs = {row["run_id"] for row in rows}
    return with_content_hash(
        {
            "contract": STEP4_RUN_REGISTRY_CONTRACT,
            "schema_version": 1,
            "stage": "B",
            "seed": 101,
            "expert_order": list(EXPERT_ORDER),
            "shape_order": list(SHAPE_SCREEN_ORDER),
            "worker_access": ["model_train", "val_stop"],
            "worker_forbidden_access": [
                "val_design",
                "stack_val",
                "final_test",
            ],
            "selector_access": ["val_design_predictions_and_labels"],
            "fixed_training_epochs": 40,
            "early_stopping": False,
            "performance_based_termination": False,
            "row_counts": {
                "primary_shape_screen": len(primary),
                "tokenizer_controls_including_reused_references": len(tokenizer),
                "dual_topology_controls": len(dual),
                "representative_expert_loss_rows_including_reused_CE": len(losses),
                "full_PT_TRACK_optimization_grid": len(optimization),
                "fixed_BASE4_REGION_followup_references": len(followup),
                "registered_rows_before_winner_followup": len(rows),
                "unique_physical_runs_before_winner_followup": len(
                    unique_physical_runs
                ),
            },
            "primary_shape_screen": primary,
            "tokenizer_controls": tokenizer,
            "dual_topology_controls": dual,
            "representative_expert_loss_rows": losses,
            "full_optimization_grid": optimization,
            "fixed_followup_references": followup,
            "winner_followup_materialization": {
                "function": "materialize_optimization_winner_followups",
                "experts": ["BASE4", "REGION"],
                "requires_locked_selection_contract": STEP4_SELECTION_CONTRACT,
            },
        }
    )


def validate_stage_b_run_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STEP4_RUN_REGISTRY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_b_run_registry()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-B run registry differs")
    return digest


def resolve_stage_b_run(
    registry: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    validate_stage_b_run_registry(registry)
    sections = (
        "primary_shape_screen",
        "tokenizer_controls",
        "dual_topology_controls",
        "representative_expert_loss_rows",
        "full_optimization_grid",
        "fixed_followup_references",
    )
    matches = [
        dict(row)
        for section in sections
        for row in registry[section]
        if row["run_id"] == str(run_id)
    ]
    if not matches:
        raise ValueError(f"Stage-B run ID is not registered: {run_id!r}")
    configurations = {repr(row["configuration"]) for row in matches}
    if len(configurations) != 1:
        raise RuntimeError("one physical Stage-B run has conflicting configurations")
    primary = next(
        (row for row in matches if not row["reuses_primary_run"]),
        matches[0],
    )
    return {
        **primary,
        "registry_memberships": [
            {
                "component": row["component"],
                "role": row["role"],
                "selection_eligible": row["selection_eligible"],
                "screen_membership_configuration": row[
                    "screen_membership_configuration"
                ],
            }
            for row in matches
        ],
    }


def build_optimization_candidate_metrics(
    *,
    run_registry: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    val_design_label_manifest_sha256: str,
) -> dict[str, Any]:
    """Authenticate the complete PT/TRACK grid used by the design selector."""

    registry_sha = validate_stage_b_run_registry(run_registry)
    label_sha = require_sha256(
        val_design_label_manifest_sha256,
        name="val_design_label_manifest_sha256",
    )
    expected = {
        str(row["run_id"]): row
        for row in run_registry["full_optimization_grid"]
    }
    if len(rows) != len(expected):
        raise ValueError("optimization metrics do not cover the complete grid")
    checked = []
    seen = set()
    for raw in rows:
        row = dict(raw)
        run_id = str(row.get("run_id"))
        if run_id in seen or run_id not in expected:
            raise ValueError("optimization metric run ID is duplicated or unknown")
        seen.add(run_id)
        if row.get("split") != "val_design":
            raise ValueError("optimization metrics must use val_design")
        if row.get("configuration") != expected[run_id]["configuration"]:
            raise ValueError("optimization metric configuration differs")
        if row.get("role", "scientific_candidate") != "scientific_candidate":
            raise ValueError("optimization selection accepts scientific rows only")
        if row.get("label_manifest_sha256") != label_sha:
            raise ValueError("optimization metric label-manifest lineage differs")
        for name in (
            "checkpoint_sha256",
            "prediction_shard_sha256",
            "metrics_artifact_sha256",
        ):
            require_sha256(row.get(name), name=f"{run_id}.{name}")
        numeric = {
            "accuracy": float(row["accuracy"]),
            "cross_entropy": float(row["cross_entropy"]),
            "measured_flops": float(row["measured_flops"]),
            "parameter_count": int(row["parameter_count"]),
        }
        if not all(math.isfinite(float(value)) for value in numeric.values()):
            raise FloatingPointError("optimization metric is nonfinite")
        if (
            not 0.0 <= numeric["accuracy"] <= 1.0
            or numeric["cross_entropy"] < 0.0
            or numeric["measured_flops"] < 0.0
            or numeric["parameter_count"] <= 0
        ):
            raise ValueError("optimization metric lies outside its valid domain")
        checked.append(
            {
                "run_id": run_id,
                "role": "scientific_candidate",
                "split": "val_design",
                "configuration": dict(row["configuration"]),
                **numeric,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "prediction_shard_sha256": row["prediction_shard_sha256"],
                "label_manifest_sha256": label_sha,
                "metrics_artifact_sha256": row["metrics_artifact_sha256"],
            }
        )
    if seen != set(expected):
        raise ValueError("optimization metric grid coverage differs")
    return with_content_hash(
        {
            "contract": STEP4_OPTIMIZATION_METRICS_CONTRACT,
            "schema_version": 1,
            "run_registry_sha256": registry_sha,
            "val_design_label_manifest_sha256": label_sha,
            "row_count": len(checked),
            "rows": sorted(checked, key=lambda value: value["run_id"]),
        }
    )


def validate_optimization_candidate_metrics(
    payload: Mapping[str, Any],
    *,
    run_registry: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=STEP4_OPTIMIZATION_METRICS_CONTRACT,
    )
    expected = build_optimization_candidate_metrics(
        run_registry=run_registry,
        rows=payload.get("rows", []),
        val_design_label_manifest_sha256=payload.get(
            "val_design_label_manifest_sha256"
        ),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("optimization candidate metrics differ")
    return digest


def aggregate_optimization_candidate_metrics(
    payload: Mapping[str, Any],
    *,
    run_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reduce the PT/TRACK screen to 18 paired hyperparameter candidates."""

    validate_optimization_candidate_metrics(payload, run_registry=run_registry)
    grouped: dict[tuple[str, float, float], list[Mapping[str, Any]]] = {}
    for row in payload["rows"]:
        configuration = row["configuration"]
        key = (
            str(configuration["initialization"]),
            float(configuration["learning_rate"]),
            float(configuration["particle_dropout"]),
        )
        grouped.setdefault(key, []).append(row)
    if len(grouped) != 18:
        raise ValueError("optimization grid must yield 18 paired candidates")
    candidates = []
    for (initialization, learning_rate, particle_dropout), rows in grouped.items():
        experts = {
            str(row["configuration"]["expert_id"]) for row in rows
        }
        if len(rows) != 2 or experts != set(FULL_OPTIMIZATION_EXPERTS):
            raise ValueError("optimization candidate lacks its PT/TRACK pair")
        configuration = {
            "initialization": initialization,
            "learning_rate": learning_rate,
            "particle_dropout": particle_dropout,
        }
        run_id = (
            "retb_b_optimization_paired_selection_s101_"
            + canonical_sha256(
                {
                    "contract": "retb_step4_paired_optimization_id_v1",
                    "configuration": configuration,
                }
            )[:12]
        )
        candidates.append(
            {
                "run_id": run_id,
                "configuration": configuration,
                "split": "val_design",
                "accuracy": sum(float(row["accuracy"]) for row in rows) / 2.0,
                "cross_entropy": (
                    sum(float(row["cross_entropy"]) for row in rows) / 2.0
                ),
                "measured_flops": (
                    sum(float(row["measured_flops"]) for row in rows) / 2.0
                ),
                "parameter_count": int(
                    round(
                        sum(int(row["parameter_count"]) for row in rows)
                        / 2.0
                    )
                ),
                "contributing_run_ids": sorted(
                    str(row["run_id"]) for row in rows
                ),
                "candidate_metrics_sha256": payload["content_hash"],
            }
        )
    return sorted(candidates, key=lambda value: value["run_id"])


def select_optimization_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_accuracy: float | None = None,
    capacity_control_reproduces_gain: bool | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("optimization selection requires candidates")
    checked = []
    for row in candidates:
        if row.get("split") != "val_design":
            raise ValueError("optimization selector may read only val_design metrics")
        values = {
            "accuracy": float(row["accuracy"]),
            "cross_entropy": float(row["cross_entropy"]),
            "measured_flops": float(row["measured_flops"]),
            "parameter_count": int(row["parameter_count"]),
        }
        if not all(
            math_value == math_value
            and math_value not in {float("inf"), float("-inf")}
            for math_value in values.values()
        ):
            raise FloatingPointError("optimization selector received nonfinite input")
        checked.append((row, values))
    maximum = max(values["accuracy"] for _, values in checked)
    eligible = [
        (row, values)
        for row, values in checked
        if maximum - values["accuracy"] <= 0.0001
    ]
    selected, selected_values = min(
        eligible,
        key=lambda item: (
            item[1]["cross_entropy"],
            item[1]["measured_flops"],
            item[1]["parameter_count"],
            str(item[0]["run_id"]),
        ),
    )
    baseline = (
        None if baseline_accuracy is None else float(baseline_accuracy)
    )
    return with_content_hash(
        {
            "contract": STEP4_SELECTION_CONTRACT,
            "schema_version": 1,
            "split": "val_design",
            "candidate_count": len(checked),
            "global_maximum_accuracy": maximum,
            "accuracy_window": 0.0001,
            "eligible_run_ids": sorted(str(row["run_id"]) for row, _ in eligible),
            "selected_run_id": str(selected["run_id"]),
            "selected_configuration": dict(selected["configuration"]),
            "selected_metrics": selected_values,
            "baseline_accuracy": baseline,
            "gain_positive": (
                None
                if baseline is None
                else selected_values["accuracy"] > baseline
            ),
            "all_candidates_worse_than_baseline": (
                None
                if baseline is None
                else all(
                    values["accuracy"] < baseline for _, values in checked
                )
            ),
            "capacity_control_reproduces_gain": (
                None
                if capacity_control_reproduces_gain is None
                else bool(capacity_control_reproduces_gain)
            ),
            "candidate_metrics_sha256": selected.get(
                "candidate_metrics_sha256"
            ),
            "selected_contributing_run_ids": list(
                selected.get("contributing_run_ids", [])
            ),
            "selection_emitted_despite_scientific_result": True,
        }
    )


def build_locked_optimization_selection(
    *,
    candidate_metrics: Mapping[str, Any],
    run_registry: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    baseline_accuracy: float | None = None,
    capacity_control_reproduces_gain: bool | None = None,
) -> dict[str, Any]:
    if candidate_metrics.get("source") != source_record(source_snapshot):
        raise ValueError(
            "optimization candidate metrics belong to another source snapshot"
        )
    candidates = aggregate_optimization_candidate_metrics(
        candidate_metrics,
        run_registry=run_registry,
    )
    selection = select_optimization_candidate(
        candidates,
        baseline_accuracy=baseline_accuracy,
        capacity_control_reproduces_gain=capacity_control_reproduces_gain,
    )
    return bind_source(
        with_content_hash(
            {
                **{
                    key: value
                    for key, value in selection.items()
                    if key != "content_hash"
                },
                "winner_followup_rows": (
                    materialize_optimization_winner_followups(
                        selection["selected_configuration"],
                        existing_rows=[
                            *run_registry["primary_shape_screen"],
                            *run_registry["fixed_followup_references"],
                        ],
                    )
                ),
            }
        ),
        source_snapshot=source_snapshot,
    )


def validate_locked_optimization_selection(
    payload: Mapping[str, Any],
    *,
    candidate_metrics: Mapping[str, Any],
    run_registry: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=STEP4_SELECTION_CONTRACT,
    )
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("locked optimization selection lacks source provenance")
    expected = build_locked_optimization_selection(
        candidate_metrics=candidate_metrics,
        run_registry=run_registry,
        source_snapshot={
            "source_commit": source.get("commit"),
            "source_status_sha256": source.get("status_sha256"),
            "source_dirty": source.get("dirty"),
        },
        baseline_accuracy=payload.get("baseline_accuracy"),
        capacity_control_reproduces_gain=payload.get(
            "capacity_control_reproduces_gain"
        ),
    )
    if dict(payload) != expected:
        raise ValueError("locked optimization selection differs")
    return digest


def execute_miniature_stage_b(
    registry: Mapping[str, Any],
    *,
    executor: Any,
    selected_optimization_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise every primary and declared optimization row through one callback."""

    validate_stage_b_run_registry(registry)
    winner = materialize_optimization_winner_followups(
        selected_optimization_configuration,
        existing_rows=[
            *registry["primary_shape_screen"],
            *registry["fixed_followup_references"],
        ],
    )
    sections = {
        "primary_shape_screen": list(registry["primary_shape_screen"]),
        "full_optimization_grid": list(registry["full_optimization_grid"]),
        "fixed_followup_references": list(registry["fixed_followup_references"]),
        "winner_followups": winner,
    }
    results = {}
    for section, rows in sections.items():
        section_results = []
        for row in rows:
            result = dict(executor(dict(row)))
            if result.get("status") != "completed":
                raise RuntimeError(
                    f"miniature Stage-B row did not complete: {row['run_id']}"
                )
            if int(result.get("epochs_completed", 0)) <= 0:
                raise RuntimeError("miniature Stage-B row completed no epochs")
            if bool(result.get("performance_based_termination", False)):
                raise RuntimeError(
                    "miniature Stage-B executor enabled performance termination"
                )
            section_results.append(
                {
                    "run_id": row["run_id"],
                    "status": "completed",
                    "epochs_completed": int(result["epochs_completed"]),
                    "performance_based_termination": False,
                }
            )
        results[section] = section_results
    return with_content_hash(
        {
            "contract": STEP4_MINIATURE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "run_registry_sha256": registry["content_hash"],
            "section_counts": {
                name: len(rows) for name, rows in sections.items()
            },
            "results": results,
            "primary_49_complete": len(results["primary_shape_screen"]) == 49,
            "declared_optimization_subset_complete": (
                len(results["full_optimization_grid"]) == 36
                and len(results["fixed_followup_references"]) == 4
                and len(results["winner_followups"]) == 2
            ),
            "scientific_performance_inspected": False,
        }
    )


def _bind(payload: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    return bind_source(payload, source_snapshot=source)


def build_step4_bundle(
    *,
    campaign_spec_sha256: str,
    step3_bundle_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_sha = require_sha256(
        campaign_spec_sha256, name="campaign_spec_sha256"
    )
    step3_sha = require_sha256(
        step3_bundle_sha256, name="step3_bundle_sha256"
    )
    determinism_sha = require_sha256(
        global_determinism_sha256, name="global_determinism_sha256"
    )
    losses = _bind(build_expert_loss_registry(), source_snapshot)
    optimization = _bind(build_optimization_registry(), source_snapshot)
    runs = _bind(build_stage_b_run_registry(), source_snapshot)
    primary_training = _bind(
        OfflineExpertTrainingConfig(seed=101).artifact(
            global_determinism_sha256=determinism_sha,
            expert_loss_registry_sha256=losses["content_hash"],
        ),
        source_snapshot,
    )
    artifacts = {
        "expert_loss_registry": losses,
        "optimization_registry": optimization,
        "stage_b_run_registry": runs,
        "primary_training_protocol": primary_training,
    }
    manifest = _bind(
        with_content_hash(
            {
                "contract": STEP4_BUNDLE_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign_sha,
                "step3_bundle_sha256": step3_sha,
                "global_determinism_sha256": determinism_sha,
                "artifact_hashes": {
                    name: artifact["content_hash"]
                    for name, artifact in sorted(artifacts.items())
                },
                "primary_shape_screen_rows": 49,
                "fixed_training_epochs": 40,
                "performance_based_termination": False,
            }
        ),
        source_snapshot,
    )
    report = _bind(
        with_content_hash(
            {
                "contract": STEP4_REPORT_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign_sha,
                "step4_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "seven_experts_registered": True,
                    "seven_uniform_shapes_registered": True,
                    "primary_49_rows_exact": True,
                    "expert_losses_frozen": True,
                    "teacher_logits_checkpoint_lineage_required": True,
                    "initialization_controls_frozen": True,
                    "learning_rate_dropout_grid_frozen": True,
                    "attachment_schedule_frozen": True,
                    "checkpoint_selection_val_stop_only": True,
                    "architecture_selection_val_design_only": True,
                    "optimization_selection_pairs_PT_and_TRACK": True,
                    "fixed_40_epoch_budget": True,
                    "performance_based_termination_disabled": True,
                    "diagnostic_sufficient_statistics_required": True,
                    "single_checkpoint_retention_required": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot,
    )
    return {**artifacts, "step4_bundle": manifest, "step4_report": report}


def validate_step4_bundle(bundle: Mapping[str, Any]) -> str:
    names = {
        "expert_loss_registry",
        "optimization_registry",
        "stage_b_run_registry",
        "primary_training_protocol",
        "step4_bundle",
        "step4_report",
    }
    if set(bundle) != names:
        raise ValueError("Step-4 bundle members differ")
    hashes = {
        "expert_loss_registry": validate_expert_loss_registry(
            bundle["expert_loss_registry"]
        ),
        "optimization_registry": validate_optimization_registry(
            bundle["optimization_registry"]
        ),
        "stage_b_run_registry": validate_stage_b_run_registry(
            bundle["stage_b_run_registry"]
        ),
        "primary_training_protocol": validate_content_hash(
            bundle["primary_training_protocol"],
            expected_contract="retb_offline_expert_training_v2",
        ),
    }
    manifest_sha = validate_content_hash(
        bundle["step4_bundle"], expected_contract=STEP4_BUNDLE_CONTRACT
    )
    validate_content_hash(
        bundle["step4_report"], expected_contract=STEP4_REPORT_CONTRACT
    )
    if bundle["step4_bundle"]["artifact_hashes"] != {
        name: value for name, value in sorted(hashes.items())
    }:
        raise ValueError("Step-4 artifact hashes differ")
    if bundle["step4_report"]["step4_bundle_sha256"] != manifest_sha:
        raise ValueError("Step-4 report belongs to another bundle")
    source = bundle["step4_bundle"].get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Step-4 bundle lacks source provenance")
    expected = build_step4_bundle(
        campaign_spec_sha256=bundle["step4_bundle"].get(
            "campaign_spec_sha256"
        ),
        step3_bundle_sha256=bundle["step4_bundle"].get(
            "step3_bundle_sha256"
        ),
        global_determinism_sha256=bundle["step4_bundle"].get(
            "global_determinism_sha256"
        ),
        source_snapshot={
            "source_commit": source.get("commit"),
            "source_status_sha256": source.get("status_sha256"),
            "source_dirty": source.get("dirty"),
        },
    )
    for name in sorted(names):
        if dict(bundle[name]) != expected[name]:
            raise ValueError(f"Step-4 artifact {name!r} differs")
    return manifest_sha


def publish_step4_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    digest = validate_step4_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "expert_loss_registry": root / "registry" / "retb_expert_losses.json",
        "optimization_registry": (
            root / "registry" / "retb_expert_optimization.json"
        ),
        "stage_b_run_registry": (
            root / "registry" / "retb_stage_b_runs.json"
        ),
        "primary_training_protocol": (
            root / "registry" / "retb_offline_expert_training.json"
        ),
        "step4_bundle": (
            root / "registry" / "retb_step4_offline_expert_bundle.json"
        ),
        "step4_report": root / "reports" / "retb_step4_report.json",
    }
    publications = {
        name: write_immutable_json(paths[name], bundle[name])
        for name in sorted(paths)
    }
    return {
        "campaign_root": str(root.resolve()),
        "step4_bundle_sha256": digest,
        "publications": publications,
    }


__all__ = [
    "STEP4_BUNDLE_CONTRACT",
    "STEP4_MINIATURE_COMPLETION_CONTRACT",
    "STEP4_OPTIMIZATION_METRICS_CONTRACT",
    "STEP4_OPTIMIZATION_REGISTRY_CONTRACT",
    "STEP4_REPORT_CONTRACT",
    "STEP4_RUN_REGISTRY_CONTRACT",
    "STEP4_SELECTION_CONTRACT",
    "aggregate_optimization_candidate_metrics",
    "build_dual_topology_rows",
    "build_expert_loss_rows",
    "build_full_optimization_rows",
    "build_locked_optimization_selection",
    "build_optimization_candidate_metrics",
    "build_optimization_registry",
    "build_primary_shape_screen_rows",
    "build_stage_b_run_registry",
    "build_step4_bundle",
    "build_tokenizer_control_rows",
    "execute_miniature_stage_b",
    "materialize_optimization_winner_followups",
    "publish_step4_bundle",
    "resolve_stage_b_run",
    "select_optimization_candidate",
    "validate_optimization_registry",
    "validate_optimization_candidate_metrics",
    "validate_locked_optimization_selection",
    "validate_stage_b_run_registry",
    "validate_step4_bundle",
]
