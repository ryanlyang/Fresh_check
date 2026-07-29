"""Stage-L matched-seed confirmation and bounded scale shortlisting."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from .contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .evaluation import (
    CLASSIFICATION_METRICS_CONTRACT,
    CLASS_NAMES,
)
from .predictor_bundle import PIPELINE_SEEDS
from .paired_statistics import (
    validate_paired_confirmation_statistics,
)


BRIDGE_SHAPE_SELECTION_CONTRACT = "retb_bridge_shape_selection_v1"
STAGE_L_GRAPH_REGISTRY_CONTRACT = "retb_stage_l_graph_registry_v1"
SEED_CONFIRMATION_CONTRACT = "retb_500k_seed_confirmation_v1"
CONFIRMATION_SUMMARY_CONTRACT = "retb_500k_confirmation_summary_v1"
SCALE_SHORTLIST_CONTRACT = "retb_locked_scale_shortlist_v1"
SHORTLISTED_CONTROLS_CONTRACT = "retb_shortlisted_500k_controls_v1"

GRAPH_ROLES = (
    "reference_baseline",
    "capacity_control",
    "architecture_control",
    "scientific_candidate",
    "semantic_control",
    "robustness_control",
)
REQUIRED_CONFIRMATION_CATEGORIES = frozenset(
    {
        "PRIMARY_BASELINE",
        "UNIFORM_FINALIST",
        "HETEROGENEOUS_FINALIST",
        "NATIVE_HLT_FUSION",
        "FROZEN_RECONSTRUCTION",
        "TOKEN_REFINER",
        "CONSTRAINED_ADAPTER",
        "UNRESTRICTED_FUSION",
    }
)
SEED_COMPONENT_KEYS = frozenset(
    {
        "offline_experts",
        "offline_fusion",
        "offline_target_cache",
        "native_hlt_experts",
        "native_hlt_fusion",
        "predictor_bundle",
        "refiner_or_identity",
        "final_consumer",
        "deployable_export",
        "complete_graph_capacity",
        "prediction_manifest",
        "metrics_artifact",
        "paired_statistics",
    }
)
SHORTLIST_PARENT_KEYS = frozenset(
    {
        "campaign_spec",
        "step12_bundle",
        "step13_bundle",
        "graph_registry",
        "confirmation_summary",
        "bridge_shape_selection",
        "validation_partition_manifest",
        "val_design_identity_manifest",
        "val_design_label_manifest",
        "hlt_replica_manifest",
        "degradation_profile",
        "offline_normalizer_bundle",
        "shared_hlt_normalizer_bundle",
        "predictor_bundle_lock",
    }
)


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is nonfinite")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric sequence")
    return float(math.fsum(float(value) for value in values) / len(values))


def _sample_standard_deviation(values: Sequence[float]) -> float:
    return 0.0 if len(values) == 1 else float(statistics.stdev(values))


def mean_log_selection_rejection(
    classification_metrics: Mapping[str, Any],
) -> float:
    """Return the finite 18-term Jeffreys mean-log-rejection score."""

    validate_content_hash(
        classification_metrics,
        expected_contract=CLASSIFICATION_METRICS_CONTRACT,
    )
    rejection = classification_metrics.get("qcd_signal_rejection", {})
    if set(rejection) != set(CLASS_NAMES[1:]):
        raise ValueError("classification rejection class coverage differs")
    terms = []
    for signal in CLASS_NAMES[1:]:
        if set(rejection[signal]) != {"0.3", "0.5"}:
            raise ValueError("classification rejection target coverage differs")
        for target in ("0.3", "0.5"):
            value = _finite(
                rejection[signal][target]["finite_selection_rejection"],
                name=f"{signal}.{target}.finite_selection_rejection",
            )
            if value <= 0.0:
                raise ValueError("selection rejection must be positive")
            terms.append(math.log(value))
    if len(terms) != 18:
        raise RuntimeError("mean-log-rejection term count differs")
    return _mean(terms)


def select_bridge_shape(
    *,
    rows: Sequence[Mapping[str, Any]],
    compact_shape_id: str,
    high_shape_id: str,
    step12_bundle_sha256: str,
    val_design_label_manifest_sha256: str,
) -> dict[str, Any]:
    """Select SHAPE_BRIDGE from compact/high using the predeclared ordering."""

    compact_id = str(compact_shape_id)
    high_id = str(high_shape_id)
    if not compact_id or not high_id:
        raise ValueError("bridge shape IDs must be nonempty")
    # SHAPE_COMPACT is explicitly allowed to equal SHAPE_HIGH.  In that
    # case the carried-shape contract removes the duplicate and confirmation
    # has one three-seed candidate rather than two interchangeable rows.
    shape_ids = tuple(dict.fromkeys((compact_id, high_id)))
    label_sha = require_sha256(
        val_design_label_manifest_sha256,
        name="val_design_label_manifest_sha256",
    )
    expected = {
        (shape_id, seed)
        for shape_id in shape_ids
        for seed in PIPELINE_SEEDS
    }
    if len(rows) != len(expected):
        raise ValueError(
            "bridge shape selection requires exactly three rows per "
            "distinct candidate shape"
        )
    checked, seen = [], set()
    dimensions: dict[str, tuple[int, int]] = {}
    for raw in rows:
        row = dict(raw)
        key = (str(row.get("shape_id")), int(row.get("pipeline_seed", -1)))
        if key not in expected or key in seen:
            raise ValueError("bridge-shape row is duplicated or unknown")
        seen.add(key)
        if (
            row.get("split") != "val_design"
            or row.get("pipeline_lineage_kind") != "PRIMARY_MATCHED_SEED"
            or row.get("label_manifest_sha256") != label_sha
            or row.get("stack_val_consumed") is not False
            or row.get("final_test_consumed") is not False
        ):
            raise ValueError("bridge-shape row lineage differs")
        K, D = int(row["K"]), int(row["D"])
        if min(K, D) <= 0:
            raise ValueError("bridge-shape dimensions must be positive")
        if key[0] in dimensions and dimensions[key[0]] != (K, D):
            raise ValueError("bridge-shape dimensions drift across seeds")
        dimensions[key[0]] = (K, D)
        values = {
            "all_predicted_accuracy": _finite(
                row["all_predicted_accuracy"],
                name="all_predicted_accuracy",
            ),
            "shape_matched_HF_NATIVE_accuracy": _finite(
                row["shape_matched_HF_NATIVE_accuracy"],
                name="shape_matched_HF_NATIVE_accuracy",
            ),
            "frozen_fusion_cross_entropy": _finite(
                row["frozen_fusion_cross_entropy"],
                name="frozen_fusion_cross_entropy",
            ),
            "normalized_token_error": _finite(
                row["normalized_token_error"],
                name="normalized_token_error",
            ),
        }
        if (
            not 0.0 <= values["all_predicted_accuracy"] <= 1.0
            or not 0.0
            <= values["shape_matched_HF_NATIVE_accuracy"]
            <= 1.0
            or min(
                values["frozen_fusion_cross_entropy"],
                values["normalized_token_error"],
            )
            < 0.0
        ):
            raise ValueError("bridge-shape metric lies outside its domain")
        checked.append(
            {
                "shape_id": key[0],
                "pipeline_seed": key[1],
                "K": K,
                "D": D,
                "split": "val_design",
                "pipeline_lineage_kind": "PRIMARY_MATCHED_SEED",
                **values,
                "paired_gain_over_HF_NATIVE": (
                    values["all_predicted_accuracy"]
                    - values["shape_matched_HF_NATIVE_accuracy"]
                ),
                "prediction_artifact_sha256": require_sha256(
                    row.get("prediction_artifact_sha256"),
                    name=f"{key}.prediction_artifact_sha256",
                ),
                "native_metrics_artifact_sha256": require_sha256(
                    row.get("native_metrics_artifact_sha256"),
                    name=f"{key}.native_metrics_artifact_sha256",
                ),
                "token_metrics_artifact_sha256": require_sha256(
                    row.get("token_metrics_artifact_sha256"),
                    name=f"{key}.token_metrics_artifact_sha256",
                ),
                "label_manifest_sha256": label_sha,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        )
    if seen != expected:
        raise ValueError("bridge-shape row coverage differs")
    aggregates = []
    for shape_id in shape_ids:
        selected = [row for row in checked if row["shape_id"] == shape_id]
        K, D = dimensions[shape_id]
        aggregates.append(
            {
                "shape_id": shape_id,
                "K": K,
                "D": D,
                "scalars": K * D,
                "mean_paired_gain_over_HF_NATIVE": _mean(
                    [row["paired_gain_over_HF_NATIVE"] for row in selected]
                ),
                "mean_frozen_fusion_cross_entropy": _mean(
                    [row["frozen_fusion_cross_entropy"] for row in selected]
                ),
                "mean_normalized_token_error": _mean(
                    [row["normalized_token_error"] for row in selected]
                ),
                "seed_rows": selected,
            }
        )
    ranking = sorted(
        aggregates,
        key=lambda row: (
            -row["mean_paired_gain_over_HF_NATIVE"],
            row["mean_frozen_fusion_cross_entropy"],
            row["mean_normalized_token_error"],
            row["scalars"],
            row["K"],
            row["D"],
            row["shape_id"],
        ),
    )
    selected = ranking[0]
    return with_content_hash(
        {
            "contract": BRIDGE_SHAPE_SELECTION_CONTRACT,
            "schema_version": 1,
            "step12_bundle_sha256": require_sha256(
                step12_bundle_sha256,
                name="step12_bundle_sha256",
            ),
            "val_design_label_manifest_sha256": label_sha,
            "candidate_shape_ids": list(shape_ids),
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "ranking_order": [
                "higher_mean_paired_gain_over_shape_matched_HF_NATIVE",
                "lower_mean_frozen_fusion_cross_entropy",
                "lower_mean_normalized_token_error",
                "fewer_total_scalars",
                "smaller_K",
                "smaller_D",
                "lexicographic_shape_id",
            ],
            "ranking": ranking,
            "SHAPE_BRIDGE": {
                key: selected[key]
                for key in ("shape_id", "K", "D", "scalars")
            },
            "split": "val_design",
            "stack_val_consumed": False,
            "final_test_consumed": False,
            "selection_emitted_despite_scientific_result": True,
        }
    )


def validate_bridge_shape_selection(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=BRIDGE_SHAPE_SELECTION_CONTRACT
    )
    candidate_shape_ids = payload.get("candidate_shape_ids", [])
    if (
        not isinstance(candidate_shape_ids, list)
        or len(candidate_shape_ids) not in {1, 2}
    ):
        raise ValueError("bridge-shape candidate coverage differs")
    expected = select_bridge_shape(
        rows=[
            row
            for aggregate in payload.get("ranking", [])
            for row in aggregate.get("seed_rows", [])
        ],
        compact_shape_id=candidate_shape_ids[0],
        high_shape_id=candidate_shape_ids[-1],
        step12_bundle_sha256=payload.get("step12_bundle_sha256"),
        val_design_label_manifest_sha256=payload.get(
            "val_design_label_manifest_sha256"
        ),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("bridge-shape selection semantics differ")
    return digest


def build_stage_l_graph_registry(
    *,
    definitions: Sequence[Mapping[str, Any]],
    step12_bundle_sha256: str,
    bridge_shape_selection: Mapping[str, Any],
) -> dict[str, Any]:
    bridge_shape_sha = validate_bridge_shape_selection(
        bridge_shape_selection
    )
    uniform_candidate_count = len(
        bridge_shape_selection["candidate_shape_ids"]
    )
    expected_uniform_roles = (
        {"SHAPE_COMPACT", "SHAPE_HIGH"}
        if uniform_candidate_count == 2
        else {"SHAPE_COMPACT_AND_HIGH"}
    )
    checked, seen = [], set()
    for raw in definitions:
        row = dict(raw)
        required = {
            "graph_id",
            "role",
            "semantic_category",
            "shortlist_eligible",
            "named_baseline_graph_id",
            "shape_id",
            "complete_graph_definition_sha256",
            "training_recipe_sha256",
            "inference_recipe_sha256",
            "deployable_without_offline_or_oracle",
            "predicts_tokens",
            "configuration",
        }
        if set(row) != required:
            raise ValueError("Stage-L graph-definition fields differ")
        graph_id = str(row["graph_id"])
        if (
            not graph_id
            or graph_id in seen
            or row["role"] not in GRAPH_ROLES
            or str(row["semantic_category"])
            not in REQUIRED_CONFIRMATION_CATEGORIES
            or bool(row["shortlist_eligible"])
            != (row["role"] == "scientific_candidate")
            or not isinstance(row["configuration"], Mapping)
        ):
            raise ValueError("Stage-L graph definition differs")
        seen.add(graph_id)
        checked.append(
            {
                "graph_id": graph_id,
                "role": row["role"],
                "semantic_category": str(row["semantic_category"]),
                "shortlist_eligible": bool(row["shortlist_eligible"]),
                "named_baseline_graph_id": str(
                    row["named_baseline_graph_id"]
                ),
                "shape_id": str(row["shape_id"]),
                "complete_graph_definition_sha256": require_sha256(
                    row["complete_graph_definition_sha256"],
                    name=f"{graph_id}.complete_graph_definition_sha256",
                ),
                "training_recipe_sha256": require_sha256(
                    row["training_recipe_sha256"],
                    name=f"{graph_id}.training_recipe_sha256",
                ),
                "inference_recipe_sha256": require_sha256(
                    row["inference_recipe_sha256"],
                    name=f"{graph_id}.inference_recipe_sha256",
                ),
                "deployable_without_offline_or_oracle": bool(
                    row["deployable_without_offline_or_oracle"]
                ),
                "predicts_tokens": bool(row["predicts_tokens"]),
                "configuration": dict(row["configuration"]),
            }
        )
    if not checked:
        raise ValueError("Stage-L graph registry is empty")
    graph_ids = {row["graph_id"] for row in checked}
    uniform_roles = {
        row["configuration"].get("carried_shape_role")
        for row in checked
        if row["semantic_category"] == "UNIFORM_FINALIST"
    }
    heterogeneous_roles = {
        row["configuration"].get("carried_shape_role")
        for row in checked
        if row["semantic_category"] == "HETEROGENEOUS_FINALIST"
    }
    if (
        {row["semantic_category"] for row in checked}
        != REQUIRED_CONFIRMATION_CATEGORIES
        or uniform_roles != expected_uniform_roles
        or sum(
            row["semantic_category"] == "UNIFORM_FINALIST"
            for row in checked
        )
        != len(expected_uniform_roles)
        or heterogeneous_roles
        != {"HET_PHYSICS", "HET_SELECTED", "HET_BEAM"}
        or sum(
            row["semantic_category"] == "HETEROGENEOUS_FINALIST"
            for row in checked
        )
        != 3
        or any(
            row["named_baseline_graph_id"] not in graph_ids
            for row in checked
        )
        or not any(row["shortlist_eligible"] for row in checked)
    ):
        raise ValueError("Stage-L graph registry coverage differs")
    return with_content_hash(
        {
            "contract": STAGE_L_GRAPH_REGISTRY_CONTRACT,
            "schema_version": 1,
            "step12_bundle_sha256": require_sha256(
                step12_bundle_sha256,
                name="step12_bundle_sha256",
            ),
            "bridge_shape_selection_sha256": bridge_shape_sha,
            "bridge_shape_selection": dict(bridge_shape_selection),
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "definition_count": len(checked),
            "definitions": sorted(
                checked, key=lambda row: row["graph_id"]
            ),
            "required_confirmation_categories": sorted(
                REQUIRED_CONFIRMATION_CATEGORIES
            ),
            "scientific_underperformance_removes_graph": False,
            "stack_val_permitted": False,
            "final_test_permitted": False,
        }
    )


def validate_stage_l_graph_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_L_GRAPH_REGISTRY_CONTRACT
    )
    expected = build_stage_l_graph_registry(
        definitions=payload.get("definitions", []),
        step12_bundle_sha256=payload.get("step12_bundle_sha256"),
        bridge_shape_selection=payload.get("bridge_shape_selection", {}),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-L graph registry semantics differ")
    return digest


def build_seed_confirmation(
    *,
    graph_definition: Mapping[str, Any],
    pipeline_seed: int,
    classification_metrics: Mapping[str, Any],
    normalized_token_error: float | None,
    analytical_flops_batch1: int,
    parameter_count: int,
    component_hashes: Mapping[str, str],
    paired_statistics: Mapping[str, Any],
    val_design_label_manifest_sha256: str,
) -> dict[str, Any]:
    if (
        int(pipeline_seed) not in PIPELINE_SEEDS
        or graph_definition.get("graph_id") is None
        or set(component_hashes) != set(SEED_COMPONENT_KEYS)
    ):
        raise ValueError("seed-confirmation identity/parent coverage differs")
    validate_content_hash(
        classification_metrics,
        expected_contract=CLASSIFICATION_METRICS_CONTRACT,
    )
    paired_sha = validate_paired_confirmation_statistics(
        paired_statistics
    )
    if classification_metrics.get("split") != "val_design":
        raise ValueError("500k confirmation may use only val_design")
    accuracy = _finite(
        classification_metrics["accuracy"], name="accuracy"
    )
    cross_entropy = _finite(
        classification_metrics["cross_entropy"],
        name="cross_entropy",
    )
    macro = _finite(
        classification_metrics["macro_per_class_accuracy"],
        name="macro_per_class_accuracy",
    )
    token_error = (
        None
        if normalized_token_error is None
        else _finite(
            normalized_token_error, name="normalized_token_error"
        )
    )
    if (
        not 0.0 <= accuracy <= 1.0
        or not 0.0 <= macro <= 1.0
        or cross_entropy < 0.0
        or (token_error is not None and token_error < 0.0)
        or int(analytical_flops_batch1) <= 0
        or int(parameter_count) <= 0
        or (
            bool(graph_definition.get("predicts_tokens"))
            != (token_error is not None)
        )
        or paired_statistics.get("candidate_graph_id")
        != str(graph_definition["graph_id"])
        or paired_statistics.get("baseline_graph_id")
        != str(graph_definition["named_baseline_graph_id"])
        or int(paired_statistics.get("pipeline_seed", -1))
        != int(pipeline_seed)
        or paired_statistics.get("parents", {}).get(
            "candidate_prediction"
        )
        != component_hashes.get("prediction_manifest")
        or paired_sha != component_hashes.get("paired_statistics")
        or not isinstance(classification_metrics.get("source"), Mapping)
        or paired_statistics.get("source")
        != classification_metrics.get("source")
    ):
        raise ValueError("seed-confirmation metric semantics differ")
    return with_content_hash(
        {
            "contract": SEED_CONFIRMATION_CONTRACT,
            "schema_version": 1,
            "graph_id": str(graph_definition["graph_id"]),
            "complete_graph_definition_sha256": require_sha256(
                graph_definition["complete_graph_definition_sha256"],
                name="complete_graph_definition_sha256",
            ),
            "pipeline_seed": int(pipeline_seed),
            "pipeline_lineage_kind": "PRIMARY_MATCHED_SEED",
            "split": "val_design",
            "classification_metrics_sha256": classification_metrics[
                "content_hash"
            ],
            "classification_metrics": dict(classification_metrics),
            "paired_statistics": dict(paired_statistics),
            "metrics": {
                "accuracy": accuracy,
                "macro_per_class_accuracy": macro,
                "cross_entropy": cross_entropy,
                "per_class_efficiency": dict(
                    classification_metrics["per_class_efficiency"]
                ),
                "mean_log_Jeffreys_selection_rejection": (
                    mean_log_selection_rejection(classification_metrics)
                ),
                "normalized_token_error": token_error,
                "analytical_flops_batch1": int(analytical_flops_batch1),
                "parameter_count": int(parameter_count),
            },
            "component_hashes": {
                name: require_sha256(
                    value, name=f"component_hashes.{name}"
                )
                for name, value in sorted(component_hashes.items())
            },
            "val_design_label_manifest_sha256": require_sha256(
                val_design_label_manifest_sha256,
                name="val_design_label_manifest_sha256",
            ),
            "deployable_without_offline_or_oracle": bool(
                graph_definition["deployable_without_offline_or_oracle"]
            ),
            "post_shortlist_controls_resolved": False,
            "fixed_40_epoch_schedule": True,
            "performance_based_termination": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_seed_confirmation(
    payload: Mapping[str, Any],
    *,
    graph_definition: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SEED_CONFIRMATION_CONTRACT
    )
    metrics = payload.get("metrics", {})
    expected = build_seed_confirmation(
        graph_definition=graph_definition,
        pipeline_seed=payload.get("pipeline_seed"),
        classification_metrics=payload.get("classification_metrics", {}),
        normalized_token_error=metrics.get("normalized_token_error"),
        analytical_flops_batch1=metrics.get(
            "analytical_flops_batch1", 0
        ),
        parameter_count=metrics.get("parameter_count", 0),
        component_hashes=payload.get("component_hashes", {}),
        paired_statistics=payload.get("paired_statistics", {}),
        val_design_label_manifest_sha256=payload.get(
            "val_design_label_manifest_sha256"
        ),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("seed-confirmation semantics differ")
    return digest


def aggregate_500k_confirmation(
    *,
    graph_registry: Mapping[str, Any],
    seed_confirmations: Sequence[Mapping[str, Any]],
    val_design_label_manifest_sha256: str,
) -> dict[str, Any]:
    registry_sha = validate_stage_l_graph_registry(graph_registry)
    label_sha = require_sha256(
        val_design_label_manifest_sha256,
        name="val_design_label_manifest_sha256",
    )
    definitions = {
        row["graph_id"]: row for row in graph_registry["definitions"]
    }
    by_graph: dict[str, dict[int, Mapping[str, Any]]] = {
        graph_id: {} for graph_id in definitions
    }
    for row in seed_confirmations:
        graph_id = str(row.get("graph_id"))
        if graph_id not in definitions:
            raise ValueError("confirmation row references an unknown graph")
        validate_seed_confirmation(
            row, graph_definition=definitions[graph_id]
        )
        seed = int(row["pipeline_seed"])
        if seed in by_graph[graph_id]:
            raise ValueError("confirmation graph/seed row is duplicated")
        if (
            row.get("val_design_label_manifest_sha256") != label_sha
            or row.get("source") != graph_registry.get("source")
        ):
            raise ValueError("confirmation source/label lineage differs")
        by_graph[graph_id][seed] = row
    aggregates, incomplete = [], []
    for graph_id, definition in definitions.items():
        seed_map = by_graph[graph_id]
        baseline_id = definition["named_baseline_graph_id"]
        baseline_seed_map = by_graph[baseline_id]
        missing = sorted(
            (set(PIPELINE_SEEDS) - set(seed_map))
            | (set(PIPELINE_SEEDS) - set(baseline_seed_map))
        )
        if missing:
            incomplete.append(
                {
                    "graph_id": graph_id,
                    "reason": "incomplete_matched_seed_coverage",
                    "missing_pipeline_seeds": missing,
                }
            )
            continue
        rows = [seed_map[seed] for seed in PIPELINE_SEEDS]
        for row in rows:
            seed = row["pipeline_seed"]
            baseline_row = baseline_seed_map[seed]
            paired = row["paired_statistics"]
            central = paired["central"]
            comparisons = (
                (
                    central["candidate_accuracy"],
                    row["metrics"]["accuracy"],
                ),
                (
                    central["baseline_accuracy"],
                    baseline_row["metrics"]["accuracy"],
                ),
                (
                    central["paired_accuracy_difference"],
                    row["metrics"]["accuracy"]
                    - baseline_row["metrics"]["accuracy"],
                ),
                (
                    central["candidate_mean_log_rejection"],
                    row["metrics"][
                        "mean_log_Jeffreys_selection_rejection"
                    ],
                ),
                (
                    central["baseline_mean_log_rejection"],
                    baseline_row["metrics"][
                        "mean_log_Jeffreys_selection_rejection"
                    ],
                ),
            )
            if (
                paired["parents"]["baseline_prediction"]
                != baseline_row["component_hashes"][
                    "prediction_manifest"
                ]
                or any(
                    abs(float(left) - float(right)) > 1e-12
                    for left, right in comparisons
                )
            ):
                raise ValueError(
                    "paired statistics differ from the candidate/named "
                    "baseline metrics at the matched seed"
                )
        metrics = [row["metrics"] for row in rows]
        flops = {row["metrics"]["analytical_flops_batch1"] for row in rows}
        parameters = {row["metrics"]["parameter_count"] for row in rows}
        if len(flops) != 1 or len(parameters) != 1:
            raise ValueError("complete graph capacity drifts across seeds")
        token_values = [
            row["metrics"]["normalized_token_error"] for row in rows
        ]
        if definition["predicts_tokens"] and any(
            value is None for value in token_values
        ):
            raise ValueError("token-predicting graph lacks token error")
        aggregates.append(
            {
                "graph_id": graph_id,
                "role": definition["role"],
                "semantic_category": definition["semantic_category"],
                "shortlist_eligible": definition["shortlist_eligible"],
                "named_baseline_graph_id": definition[
                    "named_baseline_graph_id"
                ],
                "shape_id": definition["shape_id"],
                "complete_graph_definition_sha256": definition[
                    "complete_graph_definition_sha256"
                ],
                "deployable_without_offline_or_oracle": definition[
                    "deployable_without_offline_or_oracle"
                ],
                "mean_accuracy": _mean(
                    [row["accuracy"] for row in metrics]
                ),
                "accuracy_sample_standard_deviation": (
                    _sample_standard_deviation(
                        [row["accuracy"] for row in metrics]
                    )
                ),
                "mean_macro_per_class_accuracy": _mean(
                    [row["macro_per_class_accuracy"] for row in metrics]
                ),
                "mean_cross_entropy": _mean(
                    [row["cross_entropy"] for row in metrics]
                ),
                "mean_log_Jeffreys_selection_rejection": _mean(
                    [
                        row["mean_log_Jeffreys_selection_rejection"]
                        for row in metrics
                    ]
                ),
                "mean_normalized_token_error": (
                    None
                    if not definition["predicts_tokens"]
                    else _mean([float(value) for value in token_values])
                ),
                "analytical_flops_batch1": next(iter(flops)),
                "parameter_count": next(iter(parameters)),
                "seed_rows": [
                    {
                        "pipeline_seed": row["pipeline_seed"],
                        "seed_confirmation_sha256": row["content_hash"],
                        "accuracy": row["metrics"]["accuracy"],
                        "cross_entropy": row["metrics"]["cross_entropy"],
                        "mean_log_Jeffreys_selection_rejection": row[
                            "metrics"
                        ]["mean_log_Jeffreys_selection_rejection"],
                        "prediction_manifest_sha256": row[
                            "component_hashes"
                        ]["prediction_manifest"],
                        "paired_statistics": row["paired_statistics"],
                    }
                    for row in rows
                ],
                "post_shortlist_controls_resolved": False,
            }
        )
    aggregate_map = {row["graph_id"]: row for row in aggregates}
    resolved = []
    for row in aggregates:
        baseline = aggregate_map.get(row["named_baseline_graph_id"])
        paired = None
        if baseline is not None:
            baseline_by_seed = {
                item["pipeline_seed"]: item
                for item in baseline["seed_rows"]
            }
            differences = [
                seed_row["accuracy"]
                - baseline_by_seed[seed_row["pipeline_seed"]]["accuracy"]
                for seed_row in row["seed_rows"]
            ]
            rejection_differences = [
                seed_row["mean_log_Jeffreys_selection_rejection"]
                - baseline_by_seed[seed_row["pipeline_seed"]][
                    "mean_log_Jeffreys_selection_rejection"
                ]
                for seed_row in row["seed_rows"]
            ]
            paired = {
                "baseline_graph_id": baseline["graph_id"],
                "per_seed_accuracy_difference": differences,
                "mean_accuracy_difference": _mean(differences),
                "accuracy_difference_sample_standard_deviation": (
                    _sample_standard_deviation(differences)
                ),
                "per_seed_mean_log_rejection_difference": (
                    rejection_differences
                ),
                "mean_log_rejection_difference": _mean(
                    rejection_differences
                ),
            }
        resolved.append(
            {
                **row,
                "paired_vs_named_baseline": paired,
                "gain_positive": (
                    None
                    if paired is None
                    else paired["mean_accuracy_difference"] > 0.0
                ),
            }
        )
    candidates = [row for row in resolved if row["shortlist_eligible"]]
    all_worse = bool(candidates) and all(
        row["paired_vs_named_baseline"] is not None
        and row["paired_vs_named_baseline"]["mean_accuracy_difference"] < 0.0
        for row in candidates
    )
    return with_content_hash(
        {
            "contract": CONFIRMATION_SUMMARY_CONTRACT,
            "schema_version": 1,
            "graph_registry_sha256": registry_sha,
            "val_design_label_manifest_sha256": label_sha,
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "split": "val_design",
            "complete_graph_count": len(resolved),
            "registered_graph_count": len(definitions),
            "rows": sorted(resolved, key=lambda row: row["graph_id"]),
            "ineligible_incomplete_graphs": incomplete,
            "all_candidates_worse_than_baseline": all_worse,
            "scientific_underperformance_blocked_execution": False,
            "performance_based_termination": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_500k_confirmation(
    payload: Mapping[str, Any],
    *,
    graph_registry: Mapping[str, Any],
    seed_confirmations: Sequence[Mapping[str, Any]],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=CONFIRMATION_SUMMARY_CONTRACT
    )
    expected = aggregate_500k_confirmation(
        graph_registry=graph_registry,
        seed_confirmations=seed_confirmations,
        val_design_label_manifest_sha256=payload.get(
            "val_design_label_manifest_sha256"
        ),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("500k confirmation summary semantics differ")
    return digest


def _window_ranking(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary: str,
    window: float,
    tie_key: Any,
) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in rows]
    ranked = []
    while remaining:
        maximum = max(float(row[primary]) for row in remaining)
        group = [
            row
            for row in remaining
            if maximum - float(row[primary]) <= float(window)
        ]
        group.sort(key=tie_key)
        ranked.extend(group)
        selected_ids = {row["graph_id"] for row in group}
        remaining = [
            row for row in remaining if row["graph_id"] not in selected_ids
        ]
    return ranked


def select_scale_shortlist(
    *,
    confirmation_summary: Mapping[str, Any],
    graph_registry: Mapping[str, Any],
    bridge_shape_selection: Mapping[str, Any],
    parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    validate_content_hash(
        confirmation_summary,
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    registry_sha = validate_stage_l_graph_registry(graph_registry)
    shape_sha = validate_bridge_shape_selection(bridge_shape_selection)
    if (
        set(parent_hashes) != set(SHORTLIST_PARENT_KEYS)
        or parent_hashes.get("graph_registry") != registry_sha
        or parent_hashes.get("confirmation_summary")
        != confirmation_summary.get("content_hash")
        or parent_hashes.get("bridge_shape_selection") != shape_sha
        or graph_registry.get("bridge_shape_selection_sha256")
        != shape_sha
        or parent_hashes.get("step12_bundle")
        != graph_registry.get("step12_bundle_sha256")
        or confirmation_summary.get("graph_registry_sha256") != registry_sha
        or confirmation_summary.get("val_design_label_manifest_sha256")
        != parent_hashes.get("val_design_label_manifest")
        or confirmation_summary.get("split") != "val_design"
        or confirmation_summary.get("source")
        != graph_registry.get("source")
        or bridge_shape_selection.get("source")
        != graph_registry.get("source")
        or confirmation_summary.get("stack_val_consumed")
        or confirmation_summary.get("final_test_consumed")
    ):
        raise ValueError("scale-shortlist parent/access lineage differs")
    eligible = [
        row
        for row in confirmation_summary["rows"]
        if row["shortlist_eligible"]
        and row["deployable_without_offline_or_oracle"]
    ]
    if not eligible:
        raise ValueError("scale shortlist has no complete deployable graph")

    def token_error(row: Mapping[str, Any]) -> float:
        value = row["mean_normalized_token_error"]
        return math.inf if value is None else float(value)

    accuracy_ranking = _window_ranking(
        eligible,
        primary="mean_accuracy",
        window=0.0001,
        tie_key=lambda row: (
            row["mean_cross_entropy"],
            token_error(row),
            row["analytical_flops_batch1"],
            row["parameter_count"],
            row["graph_id"],
        ),
    )
    rejection_ranking = _window_ranking(
        eligible,
        primary="mean_log_Jeffreys_selection_rejection",
        window=0.005,
        tie_key=lambda row: (
            -row["mean_accuracy"],
            row["mean_cross_entropy"],
            row["analytical_flops_batch1"],
            row["parameter_count"],
            row["graph_id"],
        ),
    )
    accuracy_top = [row["graph_id"] for row in accuracy_ranking[:3]]
    rejection_top = [row["graph_id"] for row in rejection_ranking[:3]]
    shortlist_ids = sorted(set(accuracy_top) | set(rejection_top))
    if not 1 <= len(shortlist_ids) <= 6:
        raise RuntimeError("scale shortlist size differs")
    definitions = {
        row["graph_id"]: row for row in graph_registry["definitions"]
    }
    return with_content_hash(
        {
            "contract": SCALE_SHORTLIST_CONTRACT,
            "schema_version": 1,
            "parent_hashes": {
                name: require_sha256(value, name=f"parent_hashes.{name}")
                for name, value in sorted(parent_hashes.items())
            },
            "population": "500k_val_design_three_matched_pipeline_seeds",
            "accuracy_window": 0.0001,
            "rejection_window": 0.005,
            "ACC_SCALE_TOP3": accuracy_top,
            "REJ_SCALE_TOP3": rejection_top,
            "SCALE_SHORTLIST": shortlist_ids,
            "shortlist_size": len(shortlist_ids),
            "maximum_shortlist_size": 6,
            "accuracy_ranking_trace": [
                {
                    "rank": index + 1,
                    "graph_id": row["graph_id"],
                    "mean_accuracy": row["mean_accuracy"],
                    "mean_cross_entropy": row["mean_cross_entropy"],
                    "mean_normalized_token_error": row[
                        "mean_normalized_token_error"
                    ],
                    "analytical_flops_batch1": row[
                        "analytical_flops_batch1"
                    ],
                    "parameter_count": row["parameter_count"],
                }
                for index, row in enumerate(accuracy_ranking)
            ],
            "rejection_ranking_trace": [
                {
                    "rank": index + 1,
                    "graph_id": row["graph_id"],
                    "mean_log_Jeffreys_selection_rejection": row[
                        "mean_log_Jeffreys_selection_rejection"
                    ],
                    "mean_accuracy": row["mean_accuracy"],
                    "mean_cross_entropy": row["mean_cross_entropy"],
                    "analytical_flops_batch1": row[
                        "analytical_flops_batch1"
                    ],
                    "parameter_count": row["parameter_count"],
                }
                for index, row in enumerate(rejection_ranking)
            ],
            "locked_graph_definitions": {
                graph_id: definitions[graph_id]
                for graph_id in shortlist_ids
            },
            "SHAPE_BRIDGE": bridge_shape_selection["SHAPE_BRIDGE"],
            "all_candidates_worse_than_baseline": confirmation_summary[
                "all_candidates_worse_than_baseline"
            ],
            "selection_emitted_despite_scientific_result": True,
            "contains_3M_checkpoint": False,
            "contains_finalist_identity": False,
            "stage_M_must_train_every_and_only_locked_definition": True,
            "shortlist_membership_affected_by_capacity_controls": False,
            "performance_based_termination": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_scale_shortlist(
    payload: Mapping[str, Any],
    *,
    confirmation_summary: Mapping[str, Any],
    graph_registry: Mapping[str, Any],
    bridge_shape_selection: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_SHORTLIST_CONTRACT
    )
    expected = select_scale_shortlist(
        confirmation_summary=confirmation_summary,
        graph_registry=graph_registry,
        bridge_shape_selection=bridge_shape_selection,
        parent_hashes=payload.get("parent_hashes", {}),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("locked scale shortlist semantics differ")
    return digest


def build_shortlisted_500k_controls(
    *,
    locked_scale_shortlist: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    shortlist_sha = validate_content_hash(
        locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    expected_ids = set(locked_scale_shortlist["SCALE_SHORTLIST"])
    if len(rows) != len(expected_ids):
        raise ValueError("shortlisted-control coverage differs")
    checked, seen = [], set()
    required = {
        "graph_id",
        "complete_graph_capacity_sha256",
        "monolithic_parameter_control_sha256",
        "monolithic_flop_control_sha256",
        "H_BASE_LONG_label_exposure_control_sha256",
        "control_metrics_artifact_sha256",
        "capacity_control_reproduces_gain",
    }
    for raw in rows:
        row = dict(raw)
        graph_id = str(row.get("graph_id"))
        if (
            set(row) != required
            or graph_id not in expected_ids
            or graph_id in seen
        ):
            raise ValueError("shortlisted-control row differs")
        seen.add(graph_id)
        checked.append(
            {
                "graph_id": graph_id,
                **{
                    name: require_sha256(
                        row[name], name=f"{graph_id}.{name}"
                    )
                    for name in sorted(required - {
                        "graph_id",
                        "capacity_control_reproduces_gain",
                    })
                },
                "capacity_control_reproduces_gain": bool(
                    row["capacity_control_reproduces_gain"]
                ),
            }
        )
    if seen != expected_ids:
        raise ValueError("shortlisted-control graph IDs differ")
    return with_content_hash(
        {
            "contract": SHORTLISTED_CONTROLS_CONTRACT,
            "schema_version": 1,
            "locked_scale_shortlist_sha256": shortlist_sha,
            "rows": sorted(checked, key=lambda row: row["graph_id"]),
            "complete_coverage": True,
            "created_after_shortlist_lock": True,
            "used_to_change_shortlist_membership": False,
            "performance_result_blocked_stage_M": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_shortlisted_500k_controls(
    payload: Mapping[str, Any],
    *,
    locked_scale_shortlist: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SHORTLISTED_CONTROLS_CONTRACT
    )
    expected = build_shortlisted_500k_controls(
        locked_scale_shortlist=locked_scale_shortlist,
        rows=payload.get("rows", []),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("shortlisted-control attestation semantics differ")
    return digest


__all__ = [
    "BRIDGE_SHAPE_SELECTION_CONTRACT",
    "CONFIRMATION_SUMMARY_CONTRACT",
    "GRAPH_ROLES",
    "REQUIRED_CONFIRMATION_CATEGORIES",
    "SCALE_SHORTLIST_CONTRACT",
    "SHORTLISTED_CONTROLS_CONTRACT",
    "SEED_COMPONENT_KEYS",
    "SEED_CONFIRMATION_CONTRACT",
    "SHORTLIST_PARENT_KEYS",
    "STAGE_L_GRAPH_REGISTRY_CONTRACT",
    "aggregate_500k_confirmation",
    "build_seed_confirmation",
    "build_shortlisted_500k_controls",
    "build_stage_l_graph_registry",
    "mean_log_selection_rejection",
    "select_bridge_shape",
    "select_scale_shortlist",
    "validate_500k_confirmation",
    "validate_bridge_shape_selection",
    "validate_scale_shortlist",
    "validate_seed_confirmation",
    "validate_shortlisted_500k_controls",
    "validate_stage_l_graph_registry",
]
