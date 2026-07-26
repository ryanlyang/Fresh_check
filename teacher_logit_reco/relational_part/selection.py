"""Deterministic screening, confirmation, and finalist-lock selection."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .registry import (
    CANONICAL_FAMILY_ORDER,
    CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    validate_screening_registry,
)
from .pair_builder import SUPPORTED_FAMILY_DIMENSIONS, canonical_supported_families


SCREENING_SUMMARY_CONTRACT = "relational_part_screening_summary_v2"
CONFIRMATION_REGISTRY_CONTRACT = "relational_part_confirmation_registry_v2"
CONFIRMATION_SUMMARY_CONTRACT = "relational_part_confirmation_summary_v2"
LOCKED_FINALISTS_CONTRACT = "relational_part_locked_finalists_v2"
SELECTED_UNION_MODEL_CONTRACT = "relational_part_selected_union_model_v1"
CONFIRMATION_SEEDS = (101, 202, 303)
SINGLE_RUN_IDS = (
    "RPT_PT",
    "RPT_TRACK",
    "RPT_PID",
    "RPT_CHARGE",
    "RPT_DENSITY",
    "RPT_REGION",
)
CONTROL_RUN_IDS = (
    "RPT_BASE_WIDE_MAX",
    "RPT_FULL_ZERO_REL",
    "RPT_BASE_LAYERWISE",
    "RPT_BASE_EDGEVALUE",
)


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError(f"{name} must be finite")
    return number


def _parameter_count(record: Mapping[str, Any]) -> int:
    if "parameter_count" in record:
        value = int(record["parameter_count"])
    else:
        profile = record.get("parameter_and_flop_profile")
        if not isinstance(profile, Mapping):
            raise ValueError("selection record lacks its parameter profile")
        value = int(profile["trainable_parameters"])
    if value <= 0:
        raise ValueError("parameter count must be positive")
    return value


def _metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("val_select", "val_select_metrics", "metrics"):
        value = record.get(key)
        if isinstance(value, Mapping):
            if value.get("split") not in (None, "val_select", "stack_val"):
                raise ValueError("selection may only consume val_select metrics")
            return value
    raise ValueError("selection record lacks val_select metrics")


def _validate_result_lineage(
    record: Mapping[str, Any],
    *,
    campaign_spec_sha256: str,
    split_manifest_sha256: str,
    hlt_cache_hashes: Mapping[str, str],
    expected_common_lineage_hashes: Mapping[str, str],
) -> None:
    if record.get("lineage_authenticated") is not True:
        raise ValueError("selection result lineage is not authenticated")
    for field in (
        "checkpoint_sha256",
        "checkpoint_registration_sha256",
        "val_select_metrics_sha256",
        "model_contract_sha256",
        "training_contract_sha256",
        "run_registry_sha256",
        "relation_registry_sha256",
    ):
        require_sha256(record.get(field), name=field)
    lineage = record.get("lineage_hashes")
    if not isinstance(lineage, Mapping):
        raise ValueError("selection result lacks lineage hashes")
    expected = {
        "campaign_spec": require_sha256(
            campaign_spec_sha256, name="campaign_spec_sha256"
        ),
        "split_manifest": require_sha256(
            split_manifest_sha256, name="split_manifest_sha256"
        ),
    }
    for split in ("model_train", "model_val", "stack_val"):
        expected[f"hlt_{split}"] = require_sha256(
            hlt_cache_hashes.get(split), name=f"hlt_cache_hashes.{split}"
        )
    for field, value in expected.items():
        if lineage.get(field) != value:
            raise ValueError(f"selection result lineage differs at {field}")
    for field, value in expected_common_lineage_hashes.items():
        expected_value = require_sha256(
            value, name=f"expected_common_lineage_hashes.{field}"
        )
        if lineage.get(field) != expected_value:
            raise ValueError(f"selection result lineage differs at {field}")
    for field, value in lineage.items():
        require_sha256(value, name=f"lineage_hashes.{field}")


def _rank_global_window(
    rows: Sequence[Mapping[str, Any]],
    *,
    accuracy_window: float = 0.0001,
) -> list[Mapping[str, Any]]:
    """Repeated global-window ranking; never use pairwise tolerance ordering."""

    remaining = list(rows)
    output = []
    while remaining:
        maximum = max(_finite(_metrics(row)["accuracy"], name="accuracy") for row in remaining)
        eligible = [
            row
            for row in remaining
            if maximum - _finite(_metrics(row)["accuracy"], name="accuracy")
            <= accuracy_window
        ]
        selected = min(
            eligible,
            key=lambda row: (
                _finite(_metrics(row)["cross_entropy"], name="cross_entropy"),
                _parameter_count(row),
                str(row["run_id"]),
            ),
        )
        output.append(selected)
        remaining.remove(selected)
    return output


def build_screening_summary(
    *,
    screening_registry: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    campaign_spec_sha256: str,
    split_manifest_sha256: str,
    hlt_cache_hashes: Mapping[str, str],
    results_envelope_sha256: str,
    expected_common_lineage_hashes: Mapping[str, str],
) -> dict[str, Any]:
    registry_sha = validate_screening_registry(screening_registry)
    expected_rows = {str(row["run_id"]): row for row in screening_registry["rows"]}
    actual = {str(row["run_id"]): row for row in results}
    if len(actual) != len(results):
        raise ValueError("screening results contain duplicate run IDs")
    if set(actual) != set(expected_rows):
        raise ValueError(
            "screening results differ from the fixed 21-row registry: "
            f"missing={sorted(set(expected_rows)-set(actual))}, "
            f"extra={sorted(set(actual)-set(expected_rows))}"
        )
    normalized = []
    for run_id, registry_row in expected_rows.items():
        result = actual[run_id]
        if int(result.get("seed", -1)) != 101:
            raise ValueError(f"{run_id} screening result is not seed 101")
        role = str(registry_row["configuration_role"])
        if result.get("configuration_role", role) != role:
            raise ValueError(f"{run_id} configuration role drifted")
        if result.get("checkpoint_sha256") is None:
            raise ValueError(f"{run_id} lacks a checkpoint hash")
        _validate_result_lineage(
            result,
            campaign_spec_sha256=campaign_spec_sha256,
            split_manifest_sha256=split_manifest_sha256,
            hlt_cache_hashes=hlt_cache_hashes,
            expected_common_lineage_hashes=expected_common_lineage_hashes,
        )
        normalized.append(
            {
                **dict(result),
                "run_id": run_id,
                "configuration_role": role,
                "relational_selection_eligible": bool(
                    registry_row["relational_selection_eligible"]
                ),
                "new_relation_families": list(
                    registry_row["new_relation_families"]
                ),
                "parameter_count": _parameter_count(result),
            }
        )
    complete = _rank_global_window(normalized)
    eligible = [
        row
        for row in normalized
        if row["configuration_role"] == "scientific_finalist"
    ]
    scientific = _rank_global_window(eligible)
    singles = _rank_global_window(
        [row for row in eligible if row["run_id"] in SINGLE_RUN_IDS]
    )
    if len(singles) != 6:
        raise ValueError("screening summary requires all six singles")
    best = scientific[0]
    baseline = actual["RPT_BASE"]
    baseline_accuracy = _finite(
        _metrics(baseline)["accuracy"], name="baseline_accuracy"
    )
    selected_union = tuple(
        family
        for family in CANONICAL_FAMILY_ORDER[1:]
        if family
        in {
            *singles[0]["new_relation_families"],
            *singles[1]["new_relation_families"],
        }
    )
    return with_content_hash(
        {
            "contract": SCREENING_SUMMARY_CONTRACT,
            "schema_version": 2,
            "results_envelope_sha256": require_sha256(
                results_envelope_sha256, name="results_envelope_sha256"
            ),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "hlt_cache_hashes": {
                str(name): require_sha256(value, name=f"hlt_cache_hashes.{name}")
                for name, value in sorted(hlt_cache_hashes.items())
            },
            "screening_registry_sha256": registry_sha,
            "selection_split": "val_select",
            "accuracy_window": 0.0001,
            "complete_role_labelled_ranking": [
                {
                    "rank": index,
                    "run_id": row["run_id"],
                    "configuration_role": row["configuration_role"],
                    "relational_selection_eligible": row[
                        "relational_selection_eligible"
                    ],
                    "accuracy": _finite(
                        _metrics(row)["accuracy"], name="accuracy"
                    ),
                    "cross_entropy": _finite(
                        _metrics(row)["cross_entropy"], name="cross_entropy"
                    ),
                    "parameter_count": _parameter_count(row),
                }
                for index, row in enumerate(complete, start=1)
            ],
            "scientific_finalist_ranking": [
                str(row["run_id"]) for row in scientific
            ],
            "single_family_ranking": [
                str(row["run_id"]) for row in singles
            ],
            "best_available_run_id": str(best["run_id"]),
            "selected_relation_set": list(best["new_relation_families"]),
            "selected_relation_set_sha256": canonical_sha256(
                list(best["new_relation_families"])
            ),
            "selected_union_families": list(selected_union),
            "selected_union_sha256": canonical_sha256(list(selected_union)),
            "screening_gain_positive": (
                _finite(_metrics(best)["accuracy"], name="best_accuracy")
                - baseline_accuracy
                > 0.0
            ),
            "baseline_accuracy": baseline_accuracy,
            "result_checkpoint_hashes": {
                run_id: require_sha256(
                    actual[run_id]["checkpoint_sha256"],
                    name=f"{run_id}.checkpoint_sha256",
                )
                for run_id in sorted(actual)
            },
            "negative_result_blocks_confirmation": False,
        }
    )


def validate_screening_summary(
    summary: Mapping[str, Any],
    *,
    screening_registry_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        summary, expected_contract=SCREENING_SUMMARY_CONTRACT
    )
    if screening_registry_sha256 is not None and summary.get(
        "screening_registry_sha256"
    ) != require_sha256(
        screening_registry_sha256, name="screening_registry_sha256"
    ):
        raise ValueError("screening summary belongs to another registry")
    return digest


def build_confirmation_registry(
    *,
    screening_registry: Mapping[str, Any],
    architecture_registry: Mapping[str, Any],
    screening_summary: Mapping[str, Any],
) -> dict[str, Any]:
    screening_sha = validate_screening_registry(screening_registry)
    architecture_sha = validate_content_hash(
        architecture_registry,
        expected_contract=CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
    )
    if architecture_registry.get("screening_registry_sha256") != screening_sha:
        raise ValueError("architecture registry belongs to another screen")
    summary_sha = validate_screening_summary(
        screening_summary, screening_registry_sha256=screening_sha
    )
    screening_rows = {
        str(row["run_id"]): row for row in screening_registry["rows"]
    }
    ranking = list(screening_summary["scientific_finalist_ranking"])
    combinations = [
        run_id
        for run_id in ranking
        if len(screening_rows[run_id]["new_relation_families"]) >= 2
        and run_id != "RPT_FULL_ALL"
    ]
    best_two_combinations = combinations[:2]
    if len(best_two_combinations) != 2:
        raise ValueError("screening lacks two non-full combination rows")
    fixed = list(architecture_registry["mandatory_rows"]["fixed"])
    selected_families = tuple(screening_summary["selected_relation_set"])
    union_families = tuple(screening_summary["selected_union_families"])
    existing_union = next(
        (
            run_id
            for run_id, row in screening_rows.items()
            if tuple(row["new_relation_families"]) == union_families
        ),
        None,
    )
    run_ids = list(dict.fromkeys([*fixed, *best_two_combinations]))
    union_synthesized = existing_union is None
    if union_synthesized:
        run_ids.append("RPT_SELECTED_UNION")
    elif existing_union not in run_ids:
        run_ids.append(existing_union)
    rows = []
    architecture_ids = {
        "RPT_BASE_LAYERWISE",
        "RPT_BASE_EDGEVALUE",
        "RPT_SELECTED_LAYERWISE",
        "RPT_SELECTED_EDGEVALUE",
    }
    for run_id in run_ids:
        if run_id == "RPT_SELECTED_UNION":
            role = "scientific_finalist"
            families = union_families
        elif run_id in architecture_ids:
            role = (
                "architecture_control"
                if run_id.startswith("RPT_BASE_")
                else "scientific_finalist"
            )
            families = () if role == "architecture_control" else selected_families
        else:
            row = screening_rows[run_id]
            role = str(row["configuration_role"])
            families = tuple(row["new_relation_families"])
        seed101_hash = screening_summary["result_checkpoint_hashes"].get(run_id)
        rows.append(
            {
                "run_id": run_id,
                "configuration_role": role,
                "relational_selection_eligible": role == "scientific_finalist",
                "new_relation_families": list(families),
                "seeds": list(CONFIRMATION_SEEDS),
                "seed_101": {
                    "mode": (
                        "reuse_hash_exact"
                        if seed101_hash is not None
                        else "train_from_scratch"
                    ),
                    "expected_checkpoint_sha256": seed101_hash,
                },
                "seeds_202_303": "train_from_scratch",
            }
        )
    return with_content_hash(
        {
            "contract": CONFIRMATION_REGISTRY_CONTRACT,
            "schema_version": 2,
            "screening_registry_sha256": screening_sha,
            "architecture_registry_sha256": architecture_sha,
            "screening_summary_sha256": summary_sha,
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "selected_relation_set": list(selected_families),
            "selected_relation_set_sha256": canonical_sha256(
                list(selected_families)
            ),
            "selected_union": {
                "families": list(union_families),
                "synthesized": union_synthesized,
                "reused_screening_run_id": existing_union,
            },
            "best_two_non_full_combinations": best_two_combinations,
            "rows": rows,
            "row_count": len(rows),
            "all_six_singles_mandatory": all(
                run_id in run_ids for run_id in SINGLE_RUN_IDS
            ),
            "performance_gate": False,
        }
    )


def validate_confirmation_registry(registry: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        registry, expected_contract=CONFIRMATION_REGISTRY_CONTRACT
    )
    rows = list(registry.get("rows", ()))
    if int(registry.get("row_count", -1)) != len(rows):
        raise ValueError("confirmation row count drifted")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("confirmation registry contains duplicate run IDs")
    if any(tuple(row["seeds"]) != CONFIRMATION_SEEDS for row in rows):
        raise ValueError("every confirmation row requires seeds 101/202/303")
    return digest


def build_selected_union_model_contract(
    *,
    confirmation_registry: Mapping[str, Any],
    relation_normalization_sha256: str,
    relation_registry_sha256: str,
    pair_base_sha256: str,
    family_contract_sha256: Mapping[str, str],
    weaver_runtime_sha256: str,
    global_determinism_sha256: str,
    region_normalization_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind the synthesized shared-bias union before any seed is trained."""

    registry_sha = validate_confirmation_registry(confirmation_registry)
    selected = confirmation_registry["selected_union"]
    if selected.get("synthesized") is not True:
        raise ValueError("selected union already has a screening model contract")
    families = canonical_supported_families(selected["families"])
    row = next(
        (
            value
            for value in confirmation_registry["rows"]
            if value["run_id"] == "RPT_SELECTED_UNION"
        ),
        None,
    )
    if row is None or tuple(row["new_relation_families"]) != families:
        raise ValueError("synthesized union row differs from its family set")
    family_hashes = {
        family: require_sha256(
            family_contract_sha256.get(family),
            name=f"family_contract_sha256.{family}",
        )
        for family in families
    }
    region_sha = None
    if "REGION" in families:
        region_sha = require_sha256(
            region_normalization_sha256,
            name="region_normalization_sha256",
        )
    combined = 4 + sum(
        SUPPORTED_FAMILY_DIMENSIONS[family] for family in families
    )
    return with_content_hash(
        {
            "contract": SELECTED_UNION_MODEL_CONTRACT,
            "schema_version": 1,
            "run_id": "RPT_SELECTED_UNION",
            "configuration_role": "scientific_finalist",
            "relational_selection_eligible": True,
            "confirmation_registry_sha256": registry_sha,
            "relation_normalization_sha256": require_sha256(
                relation_normalization_sha256,
                name="relation_normalization_sha256",
            ),
            "region_normalization_sha256": region_sha,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "pair_base_sha256": require_sha256(
                pair_base_sha256, name="pair_base_sha256"
            ),
            "family_contract_sha256": family_hashes,
            "weaver_runtime_sha256": require_sha256(
                weaver_runtime_sha256, name="weaver_runtime_sha256"
            ),
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "enabled_relations": ["base4", *families],
            "new_relation_families": list(families),
            "canonical_concatenation_order": ["base4", *families],
            "combined_dimension": combined,
            "relation_input_mode": "explicit_uu",
            "attention_architecture": "shared_directional_pair_bias",
            "initialization": "from_scratch",
            "persistent_N_by_N_cache": False,
            "hlt_only_inference": True,
            "offline_or_teacher_required": False,
        }
    )


def _aggregate_one(
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
    baseline_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    families: Sequence[str] = (),
) -> dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in rows}
    if len(rows) != len(CONFIRMATION_SEEDS) or set(by_seed) != set(
        CONFIRMATION_SEEDS
    ):
        raise ValueError(f"{run_id} lacks exact three-seed confirmation")
    roles = {str(row["configuration_role"]) for row in rows}
    eligibility = {
        bool(row["relational_selection_eligible"]) for row in rows
    }
    parameter_counts = {_parameter_count(row) for row in rows}
    if len(roles) != 1 or len(eligibility) != 1:
        raise ValueError(f"{run_id} role/selectability drifted across seeds")
    if len(parameter_counts) != 1:
        raise ValueError(f"{run_id} parameter count drifted across seeds")
    if any(row.get("lineage_authenticated") is not True for row in rows):
        raise ValueError(f"{run_id} contains unauthenticated result lineage")
    lineages = [dict(row["lineage_hashes"]) for row in rows]
    if any(value != lineages[0] for value in lineages[1:]):
        raise ValueError(f"{run_id} lineage drifted across seeds")
    model_contracts = {
        require_sha256(
            row["model_contract_sha256"], name="model_contract_sha256"
        )
        for row in rows
    }
    if len(model_contracts) != 1:
        raise ValueError(f"{run_id} model contract drifted across seeds")
    accuracies = [
        _finite(_metrics(by_seed[seed])["accuracy"], name="accuracy")
        for seed in CONFIRMATION_SEEDS
    ]
    cross_entropies = [
        _finite(
            _metrics(by_seed[seed])["cross_entropy"], name="cross_entropy"
        )
        for seed in CONFIRMATION_SEEDS
    ]
    deltas = [
        accuracies[index]
        - _finite(
            _metrics(baseline_by_seed[seed])["accuracy"],
            name="baseline_accuracy",
        )
        for index, seed in enumerate(CONFIRMATION_SEEDS)
    ]
    first = by_seed[101]
    return {
        "run_id": run_id,
        "configuration_role": str(first["configuration_role"]),
        "relational_selection_eligible": bool(
            first["relational_selection_eligible"]
        ),
        "new_relation_families": list(families),
        "mean_accuracy": float(statistics.fmean(accuracies)),
        "median_accuracy": float(statistics.median(accuracies)),
        "accuracy_sample_standard_deviation": float(statistics.stdev(accuracies)),
        "mean_cross_entropy": float(statistics.fmean(cross_entropies)),
        "per_seed_accuracy": {
            str(seed): accuracies[index]
            for index, seed in enumerate(CONFIRMATION_SEEDS)
        },
        "per_seed_matched_baseline_difference": {
            str(seed): deltas[index]
            for index, seed in enumerate(CONFIRMATION_SEEDS)
        },
        "mean_matched_seed_accuracy_difference": float(
            statistics.fmean(deltas)
        ),
        "seeds_beating_matched_baseline": sum(delta > 0 for delta in deltas),
        "parameter_count": _parameter_count(first),
        "checkpoint_hashes": {
            str(seed): require_sha256(
                by_seed[seed]["checkpoint_sha256"],
                name=f"{run_id}.seed_{seed}.checkpoint_sha256",
            )
            for seed in CONFIRMATION_SEEDS
        },
        "checkpoint_registration_hashes": {
            str(seed): require_sha256(
                by_seed[seed]["checkpoint_registration_sha256"],
                name=f"{run_id}.seed_{seed}.checkpoint_registration_sha256",
            )
            for seed in CONFIRMATION_SEEDS
        },
        "val_select_metrics_hashes": {
            str(seed): require_sha256(
                by_seed[seed]["val_select_metrics_sha256"],
                name=f"{run_id}.seed_{seed}.val_select_metrics_sha256",
            )
            for seed in CONFIRMATION_SEEDS
        },
        "model_contract_sha256": next(iter(model_contracts)),
        "lineage_hashes": lineages[0],
        "lineage_authenticated": True,
    }


def aggregate_confirmation(
    *,
    confirmation_registry: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    campaign_spec_sha256: str,
    split_manifest_sha256: str,
    hlt_cache_hashes: Mapping[str, str],
    results_envelope_sha256: str,
    expected_common_lineage_hashes: Mapping[str, str],
    semantic_unary_results: Sequence[Mapping[str, Any]] = (),
    unary_results_envelope_sha256: str | None = None,
    semantic_perturbation_sha256: str | None = None,
    unary_control_registry_sha256: str | None = None,
    seal_finalists: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    registry_sha = validate_confirmation_registry(confirmation_registry)
    expected = {str(row["run_id"]): row for row in confirmation_registry["rows"]}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        _validate_result_lineage(
            result,
            campaign_spec_sha256=campaign_spec_sha256,
            split_manifest_sha256=split_manifest_sha256,
            hlt_cache_hashes=hlt_cache_hashes,
            expected_common_lineage_hashes=expected_common_lineage_hashes,
        )
        grouped.setdefault(str(result["run_id"]), []).append(result)
    if set(grouped) != set(expected):
        raise ValueError("confirmation results differ from its immutable registry")
    baseline_by_seed = {
        int(row["seed"]): row for row in grouped["RPT_BASE"]
    }
    aggregates = []
    for run_id, registry_row in expected.items():
        for result in grouped[run_id]:
            if result.get("configuration_role") != registry_row[
                "configuration_role"
            ]:
                raise ValueError(f"{run_id} role differs from confirmation registry")
        if registry_row["seed_101"]["mode"] == "reuse_hash_exact":
            observed = next(
                row for row in grouped[run_id] if int(row["seed"]) == 101
            )
            if observed["checkpoint_sha256"] != registry_row["seed_101"][
                "expected_checkpoint_sha256"
            ]:
                raise ValueError(f"{run_id} seed-101 reuse hash mismatch")
        aggregates.append(
            _aggregate_one(
                run_id,
                grouped[run_id],
                baseline_by_seed,
                families=registry_row["new_relation_families"],
            )
        )
    scientific = [
        row
        for row in aggregates
        if row["configuration_role"] == "scientific_finalist"
    ]
    scientific.sort(
        key=lambda row: (
            -row["mean_matched_seed_accuracy_difference"],
            row["mean_cross_entropy"],
            row["accuracy_sample_standard_deviation"],
            row["parameter_count"],
            row["run_id"],
        )
    )
    winner = scientific[0]
    capacity_delta = max(
        next(
            row["mean_matched_seed_accuracy_difference"]
            for row in aggregates
            if row["run_id"] == control
        )
        for control in ("RPT_BASE_WIDE_MAX", "RPT_FULL_ZERO_REL")
    )
    gain_positive = winner["mean_matched_seed_accuracy_difference"] > 0.0
    capacity_reproduces = (
        capacity_delta >= winner["mean_matched_seed_accuracy_difference"]
        if gain_positive
        else False
    )
    summary = with_content_hash(
        {
            "contract": CONFIRMATION_SUMMARY_CONTRACT,
            "schema_version": 2,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "hlt_cache_hashes": {
                str(name): require_sha256(
                    value, name=f"hlt_cache_hashes.{name}"
                )
                for name, value in sorted(hlt_cache_hashes.items())
            },
            "results_envelope_sha256": require_sha256(
                results_envelope_sha256, name="results_envelope_sha256"
            ),
            "all_result_lineage_authenticated": True,
            "confirmation_registry_sha256": registry_sha,
            "rows": aggregates,
            "scientific_finalist_ordering": [
                row["run_id"] for row in scientific
            ],
            "nominal_relational_winner_id": winner["run_id"],
            "confirmation_gain_positive": gain_positive,
            "capacity_control_reproduces_gain": capacity_reproduces,
            "capacity_control_max_mean_delta": capacity_delta,
            "selection_reason": (
                "role_filtered_highest_mean_matched_seed_accuracy_difference_"
                "then_mean_CE_then_accuracy_sample_std_then_parameters_then_run_id"
            ),
            "negative_campaign_valid": True,
        }
    )
    if not seal_finalists:
        return summary, None
    if not semantic_unary_results:
        raise ValueError(
            "finalist lock requires the trained three-seed unary control"
        )
    if semantic_perturbation_sha256 is None:
        raise ValueError("finalist lock requires semantic perturbation results")
    if unary_control_registry_sha256 is None:
        raise ValueError("finalist lock requires the unary control registry")
    unary = _aggregate_one(
        "RPT_SELECTED_UNARY",
        semantic_unary_results,
        baseline_by_seed,
        families=confirmation_registry["selected_relation_set"],
    )
    if unary["configuration_role"] != "semantic_control":
        raise ValueError("unary result must retain semantic_control role")
    for result in semantic_unary_results:
        _validate_result_lineage(
            result,
            campaign_spec_sha256=campaign_spec_sha256,
            split_manifest_sha256=split_manifest_sha256,
            hlt_cache_hashes=hlt_cache_hashes,
            expected_common_lineage_hashes=expected_common_lineage_hashes,
        )
    unary_envelope_sha = require_sha256(
        unary_results_envelope_sha256,
        name="unary_results_envelope_sha256",
    )
    evaluation_rows = [*aggregates, unary]
    lock = with_content_hash(
        {
            "contract": LOCKED_FINALISTS_CONTRACT,
            "schema_version": 2,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "hlt_cache_hashes": {
                str(name): require_sha256(value, name=f"hlt_cache_hashes.{name}")
                for name, value in sorted(hlt_cache_hashes.items())
            },
            "confirmation_registry_sha256": registry_sha,
            "confirmation_summary_sha256": summary["content_hash"],
            "confirmation_results_envelope_sha256": require_sha256(
                results_envelope_sha256, name="results_envelope_sha256"
            ),
            "unary_results_envelope_sha256": unary_envelope_sha,
            "all_selection_lineage_authenticated": True,
            "semantic_perturbation_sha256": require_sha256(
                semantic_perturbation_sha256,
                name="semantic_perturbation_sha256",
            ),
            "unary_control_registry_sha256": require_sha256(
                unary_control_registry_sha256,
                name="unary_control_registry_sha256",
            ),
            "baseline_id": "RPT_BASE",
            "evaluation_rows": [
                {
                    "run_id": row["run_id"],
                    "configuration_role": row["configuration_role"],
                    "relational_selection_eligible": row[
                        "relational_selection_eligible"
                    ],
                    "new_relation_families": row[
                        "new_relation_families"
                    ],
                    "mean_matched_seed_accuracy_difference": row[
                        "mean_matched_seed_accuracy_difference"
                    ],
                    "checkpoint_hashes": row["checkpoint_hashes"],
                    "checkpoint_registration_hashes": row[
                        "checkpoint_registration_hashes"
                    ],
                    "val_select_metrics_hashes": row[
                        "val_select_metrics_hashes"
                    ],
                    "model_contract_sha256": row[
                        "model_contract_sha256"
                    ],
                    "lineage_hashes": row["lineage_hashes"],
                    "lineage_authenticated": True,
                }
                for row in evaluation_rows
            ],
            "nominal_relational_winner_id": winner["run_id"],
            "confirmation_gain_positive": gain_positive,
            "capacity_control_reproduces_gain": capacity_reproduces,
            "selection_metrics": {
                row["run_id"]: {
                    "mean_accuracy": row["mean_accuracy"],
                    "mean_cross_entropy": row["mean_cross_entropy"],
                    "accuracy_sample_standard_deviation": row[
                        "accuracy_sample_standard_deviation"
                    ],
                    "mean_matched_seed_accuracy_difference": row[
                        "mean_matched_seed_accuracy_difference"
                    ],
                }
                for row in evaluation_rows
            },
            "selection_reason": summary["selection_reason"],
            "final_test_used_for_selection": False,
            "final_test_reporting_only": True,
        }
    )
    return summary, lock


def validate_locked_finalists(
    lock: Mapping[str, Any],
    *,
    campaign_spec_sha256: str,
    split_manifest_sha256: str,
    hlt_cache_hashes: Mapping[str, str],
) -> str:
    digest = validate_content_hash(
        lock, expected_contract=LOCKED_FINALISTS_CONTRACT
    )
    expected = {
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="campaign_spec_sha256"
        ),
        "split_manifest_sha256": require_sha256(
            split_manifest_sha256, name="split_manifest_sha256"
        ),
    }
    for name, value in expected.items():
        if lock.get(name) != value:
            raise ValueError(f"finalist lock {name} mismatch")
    locked_hlt = lock.get("hlt_cache_hashes")
    supplied_hlt = {
        str(name): require_sha256(value, name=f"hlt_cache_hashes.{name}")
        for name, value in sorted(hlt_cache_hashes.items())
    }
    if locked_hlt != supplied_hlt:
        raise ValueError("finalist lock HLT-cache inventory mismatch")
    if lock.get("final_test_used_for_selection") is not False:
        raise ValueError("finalist lock is contaminated by final-test selection")
    require_sha256(
        lock.get("semantic_perturbation_sha256"),
        name="semantic_perturbation_sha256",
    )
    require_sha256(
        lock.get("unary_control_registry_sha256"),
        name="unary_control_registry_sha256",
    )
    rows = list(lock.get("evaluation_rows", ()))
    run_ids = [str(row.get("run_id")) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("finalist lock has duplicate evaluation rows")
    by_id = {str(row["run_id"]): row for row in rows}
    baseline_id = str(lock.get("baseline_id"))
    if (
        baseline_id not in by_id
        or by_id[baseline_id].get("configuration_role")
        != "reference_baseline"
        or by_id[baseline_id].get("relational_selection_eligible") is not False
    ):
        raise ValueError("finalist lock baseline row is invalid")
    winner_id = str(lock.get("nominal_relational_winner_id"))
    if (
        winner_id not in by_id
        or by_id[winner_id].get("configuration_role")
        != "scientific_finalist"
        or by_id[winner_id].get("relational_selection_eligible") is not True
    ):
        raise ValueError("finalist lock nominal winner is invalid")
    for row in rows:
        if row.get("lineage_authenticated") is not True:
            raise ValueError("finalist row lineage is not authenticated")
        require_sha256(
            row.get("model_contract_sha256"),
            name=f"{row.get('run_id')}.model_contract_sha256",
        )
        if set(row.get("checkpoint_hashes", {})) != {
            str(seed) for seed in CONFIRMATION_SEEDS
        }:
            raise ValueError("finalist row lacks exact three-seed checkpoints")
        for seed, value in row["checkpoint_hashes"].items():
            require_sha256(value, name=f"{row['run_id']}.seed_{seed}")
        for field in (
            "checkpoint_registration_hashes",
            "val_select_metrics_hashes",
        ):
            values = row.get(field, {})
            if set(values) != {str(seed) for seed in CONFIRMATION_SEEDS}:
                raise ValueError(f"finalist row lacks exact {field}")
            for seed, value in values.items():
                require_sha256(
                    value, name=f"{row['run_id']}.seed_{seed}.{field}"
                )
        lineage = row.get("lineage_hashes")
        if not isinstance(lineage, Mapping):
            raise ValueError("finalist row lacks lineage hashes")
        for name, value in lineage.items():
            require_sha256(value, name=f"lineage_hashes.{name}")
    if "final_test" not in supplied_hlt:
        raise ValueError("finalist lock lacks the sealed final-test HLT cache")
    unary = [
        row
        for row in rows
        if row.get("run_id") == "RPT_SELECTED_UNARY"
        and row.get("configuration_role") == "semantic_control"
        and row.get("relational_selection_eligible") is False
    ]
    if len(unary) != 1:
        raise ValueError("finalist lock lacks its trained unary semantic control")
    return digest


__all__ = [
    "CONFIRMATION_REGISTRY_CONTRACT",
    "CONFIRMATION_SEEDS",
    "CONFIRMATION_SUMMARY_CONTRACT",
    "LOCKED_FINALISTS_CONTRACT",
    "SCREENING_SUMMARY_CONTRACT",
    "SELECTED_UNION_MODEL_CONTRACT",
    "aggregate_confirmation",
    "build_confirmation_registry",
    "build_screening_summary",
    "build_selected_union_model_contract",
    "validate_confirmation_registry",
    "validate_locked_finalists",
    "validate_screening_summary",
]
