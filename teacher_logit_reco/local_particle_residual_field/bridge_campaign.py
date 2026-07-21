"""Declarative Step 1 registry and storage preflight for the bridge pilot."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bridge_contracts import (
    canonical_json_bytes,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .bridge_splits import (
    PREDICTION_ANCHORED_SPLIT_CONTRACT,
    child_split_summary,
)


PREDICTION_ANCHORED_REGISTRY_CONTRACT = "prediction_anchored_campaign_registry_v1"
PREDICTION_ANCHORED_RUN_CONTRACT = "prediction_anchored_campaign_run_v1"
PREDICTION_ANCHORED_STORAGE_PROJECTION_CONTRACT = (
    "prediction_anchored_provisional_storage_projection_v1"
)
PREDICTION_ANCHORED_STEP1_REPORT_CONTRACT = "prediction_anchored_step1_report_v1"
PREDICTION_ANCHORED_STORAGE_FORMULA_CONTRACT = (
    "prediction_anchored_provisional_storage_formula_v1"
)

PAIRED_SEED_IDS = (101, 202, 303)
RETAINED_STATE_RULE = "metrics_all_seeds__weights_ordered_median_only"
REGISTRY_CONFIGURATION_COUNT = 54
RECONSTRUCTION_BREADTH_COUNT = 46
POST_TEACHER_CONFIGURATION_COUNT = 45
NORMAL_PILOT_BUDGET_BYTES = 5 * 1024**3
PAIRED3_HARD_CEILING_BYTES = 6 * 1024**3
MEASUREMENT_UNMEASURED = "UNMEASURED"
MEASUREMENT_MEASURED = "MEASURED"

_METRICS_CAP_BYTES = 1 * 1024**2
_LOG_CAP_BYTES = 2 * 1024**2
_RUN_MANIFEST_CAP_BYTES = 256 * 1024
_SERIALIZATION_HEADROOM_FRACTION = 0.05
_WEIGHT_DTYPE_BYTES = 4

# These are deliberately formula inputs, not instantiated placeholder models.
# Steps 3 and 5-8 replace them with measured serialized state sizes.
PROVISIONAL_PARAMETER_UPPER_BOUNDS: dict[str, int] = {
    "tagger": 12_000_000,
    "c0": 8_000_000,
    "particle_capacity": 20_000_000,
    "multiscale_local": 12_000_000,
    "regions": 18_000_000,
    "hlg": 24_000_000,
    "hlg_refine": 30_000_000,
    "direct_classifier": 30_000_000,
}


@dataclass(frozen=True)
class CampaignRunDefinition:
    canonical_run_id: str
    aliases: tuple[str, ...]
    family: str
    stage: str
    scientific_role: str
    selectable_for_primary_deployment: bool
    storage_class: str
    reconstruction_breadth: bool
    post_teacher_configuration: bool
    requires_selected_teacher: bool
    conditional_parent: str | None


def _definition(
    run_id: str,
    *,
    aliases: Sequence[str] = (),
    family: str,
    stage: str,
    scientific_role: str,
    selectable: bool,
    storage_class: str,
    reconstruction: bool,
    post_teacher: bool,
    requires_selected_teacher: bool,
    conditional_parent: str | None = None,
) -> CampaignRunDefinition:
    return CampaignRunDefinition(
        canonical_run_id=run_id,
        aliases=tuple(aliases),
        family=family,
        stage=stage,
        scientific_role=scientific_role,
        selectable_for_primary_deployment=bool(selectable),
        storage_class=storage_class,
        reconstruction_breadth=bool(reconstruction),
        post_teacher_configuration=bool(post_teacher),
        requires_selected_teacher=bool(requires_selected_teacher),
        conditional_parent=conditional_parent,
    )


def campaign_run_definitions() -> tuple[CampaignRunDefinition, ...]:
    """Return the locked canonical 54-row inventory.

    Aliases are identifiers, not additional configurations.  Every definition
    states its scientific role and selectability explicitly.
    """

    baseline_role = "baseline_or_teacher_control"
    loss_role = "confirmatory_physical45_loss_candidate"
    arch_role = "confirmatory_bounded_physical45_architecture_candidate"
    interaction_role = "confirmatory_bounded_a3_loss_schedule_candidate"

    rows: list[CampaignRunDefinition] = [
        _definition(
            "A0_C250",
            family="upstream",
            stage="B3",
            scientific_role=baseline_role,
            selectable=False,
            storage_class="tagger",
            reconstruction=False,
            post_teacher=False,
            requires_selected_teacher=False,
        ),
        _definition(
            "A0_C250_LONG",
            family="upstream",
            stage="B3",
            scientific_role=baseline_role,
            selectable=False,
            storage_class="tagger",
            reconstruction=False,
            post_teacher=False,
            requires_selected_teacher=False,
        ),
        _definition(
            "A0_S500",
            family="upstream",
            stage="B3",
            scientific_role=baseline_role,
            selectable=False,
            storage_class="tagger",
            reconstruction=False,
            post_teacher=False,
            requires_selected_teacher=False,
        ),
        _definition(
            "Tpred",
            family="upstream",
            stage="B3",
            scientific_role=baseline_role,
            selectable=False,
            storage_class="tagger",
            reconstruction=False,
            post_teacher=False,
            requires_selected_teacher=False,
        ),
        _definition(
            "Tpred_continue",
            family="upstream",
            stage="B3",
            scientific_role=baseline_role,
            selectable=False,
            storage_class="tagger",
            reconstruction=False,
            post_teacher=False,
            requires_selected_teacher=False,
        ),
        _definition(
            "T10_clean",
            family="upstream",
            stage="B3",
            scientific_role=baseline_role,
            selectable=False,
            storage_class="tagger",
            reconstruction=False,
            post_teacher=False,
            requires_selected_teacher=False,
        ),
        _definition(
            "T10_robust",
            family="upstream",
            stage="B3",
            scientific_role=baseline_role,
            selectable=False,
            storage_class="tagger",
            reconstruction=False,
            post_teacher=False,
            requires_selected_teacher=False,
        ),
        _definition(
            "T10_all50_clean",
            family="upstream",
            stage="B3",
            scientific_role=baseline_role,
            selectable=False,
            storage_class="tagger",
            reconstruction=False,
            post_teacher=False,
            requires_selected_teacher=False,
        ),
    ]

    loss_ids = (
        "D10_L0_bridge_only",
        "D10_L1_ce_only",
        "D10_L2_kd_only",
        "D10_L3_kd_ce",
        "D10_L4_kd_bridge",
        "D10_L5_ce_bridge",
        "D10_L6_kd_ce_bridge",
        "D10_L7_plus_anchor",
        "D10_L8_full_c0",
        "D10_L9_full_true_target",
        "D10_L10_no_trust",
    )
    for run_id in loss_ids:
        early = run_id == "D10_L0_bridge_only"
        no_trust = run_id == "D10_L10_no_trust"
        rows.append(
            _definition(
                run_id,
                aliases=("D10_A0_c0_delta",) if run_id == "D10_L8_full_c0" else (),
                family="loss",
                stage="B3" if early else "B6",
                scientific_role=("unbounded_safety_diagnostic" if no_trust else loss_role),
                selectable=not no_trust,
                storage_class="c0",
                reconstruction=True,
                post_teacher=not early,
                requires_selected_teacher=not early,
            )
        )

    architecture_rows = (
        ("D10_A0M_capacity_particle", arch_role, True, "particle_capacity"),
        ("D10_A1_multiscale_local", arch_role, True, "multiscale_local"),
        ("D10_A1H_hard_radius", arch_role, True, "multiscale_local"),
        ("D10_A2_regions_no_global", arch_role, True, "regions"),
        ("D10_A3_hlg_primary", arch_role, True, "hlg"),
        ("D10_A4_hlg_refine", arch_role, True, "hlg_refine"),
        ("D10_A5_hlg_absolute_conditioned", "absolute_output_diagnostic", False, "hlg"),
        ("D10_A5S_hlg_scratch_physical45", "absolute_output_diagnostic", False, "hlg"),
        ("D10_A6_hlg_no_pair_bias", arch_role, True, "hlg"),
        ("D10_A7_hlg_no_h0", arch_role, True, "hlg"),
        ("D10_A7F_hlg_no_f0", arch_role, True, "hlg"),
        ("D10_A7X_hlg_no_raw_skip", arch_role, True, "hlg"),
        ("D10_A8_hlg_fused_radius_heads", arch_role, True, "hlg"),
        ("D10_A9_hlg_group_gate", arch_role, True, "hlg"),
        ("D10_AS_hlg_regions_2_2_1", "exploratory_hierarchy_diagnostic", False, "hlg"),
        ("D10_AL_hlg_regions_8_8_4", "exploratory_hierarchy_diagnostic", False, "hlg"),
        ("D10_AFIX_hlg_fixed_assignment", "exploratory_hierarchy_diagnostic", False, "hlg"),
        ("D10_ASAME_hlg_same_scale_only", "exploratory_hierarchy_diagnostic", False, "hlg"),
        ("D10_AGLOBAL_hlg_one_global_token", "exploratory_hierarchy_diagnostic", False, "hlg"),
        ("A0_CAP500_direct_hlt", baseline_role, False, "direct_classifier"),
        ("A0_CAP500_r0rep_direct", baseline_role, False, "direct_classifier"),
    )
    for run_id, role, selectable, storage_class in architecture_rows:
        rows.append(
            _definition(
                run_id,
                aliases=("D10_XA3_full_primary",) if run_id == "D10_A3_hlg_primary" else (),
                family="architecture",
                stage="B6",
                scientific_role=role,
                selectable=selectable,
                storage_class=storage_class,
                reconstruction=True,
                post_teacher=True,
                requires_selected_teacher=not run_id.startswith("A0_CAP500_"),
            )
        )

    for run_id in (
        "D10_XA3_bridge_only",
        "D10_XA3_ce_only",
        "D10_XA3_kd_only",
        "D10_XA3_kd_bridge",
        "D10_XA3_kd_ce",
        "D10_XA3_full_no_warmup",
        "D10_XA3_full_no_smooth",
    ):
        rows.append(
            _definition(
                run_id,
                family="a3_interaction",
                stage="B6",
                scientific_role=interaction_role,
                selectable=True,
                storage_class="hlg",
                reconstruction=True,
                post_teacher=True,
                requires_selected_teacher=True,
            )
        )

    for run_id in ("D10_B1_all50_fullhead", "D10_B2_all50_physical45_only"):
        rows.append(
            _definition(
                run_id,
                family="all50",
                stage="B6",
                scientific_role="all50_semantic_diagnostic",
                selectable=False,
                storage_class="hlg",
                reconstruction=True,
                post_teacher=True,
                requires_selected_teacher=False,
            )
        )

    rows.append(
        _definition(
            "D10_TALT_A3",
            family="alternate_teacher",
            stage="B6",
            scientific_role="alternate_teacher_diagnostic",
            selectable=False,
            storage_class="hlg",
            reconstruction=True,
            post_teacher=True,
            requires_selected_teacher=True,
            conditional_parent="alternate_teacher_valid",
        )
    )

    for run_id in (
        "D10_N0_shuffled_logit_kd",
        "D10_N1_shuffled_bridge_field",
        "D10_N2_shuffled_primary",
        "D10_N3_nonprivileged_teacher_kd",
    ):
        rows.append(
            _definition(
                run_id,
                family="negative_control",
                stage="B6",
                scientific_role="negative_control",
                selectable=False,
                storage_class="hlg",
                reconstruction=True,
                post_teacher=True,
                requires_selected_teacher=True,
            )
        )
    return tuple(rows)


def provisional_storage_categories(storage_class: str) -> dict[str, int]:
    try:
        parameter_upper_bound = PROVISIONAL_PARAMETER_UPPER_BOUNDS[storage_class]
    except KeyError as exc:
        raise ValueError(f"unknown provisional storage class: {storage_class}") from exc
    weights = int(parameter_upper_bound * _WEIGHT_DTYPE_BYTES)
    headroom = int(math.ceil(weights * _SERIALIZATION_HEADROOM_FRACTION))
    return {
        "retained_median_weights": weights,
        "serialization_headroom": headroom,
        "aggregate_seed_metrics": _METRICS_CAP_BYTES,
        "bounded_logs": _LOG_CAP_BYTES,
        "run_manifest": _RUN_MANIFEST_CAP_BYTES,
        "nonmedian_replica_weights": 0,
        "optimizer_scheduler_state": 0,
        "generated_dense_fields": 0,
    }


def _seed_replicas(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "seed_id": seed,
            "replica_id": f"{run_id}__seed{seed}",
            "persistent_metrics": True,
            "persistent_weights": "ordered_median_only_after_aggregate",
        }
        for seed in PAIRED_SEED_IDS
    ]


def build_campaign_registry(*, alternate_teacher_valid: bool = False) -> dict[str, Any]:
    definitions = campaign_run_definitions()
    runs: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for definition in definitions:
        for alias in definition.aliases:
            aliases[alias] = definition.canonical_run_id
        categories = provisional_storage_categories(definition.storage_class)
        execution_status = (
            "SKIPPED_INVALID_PARENT"
            if definition.conditional_parent == "alternate_teacher_valid"
            and not alternate_teacher_valid
            else "RUNNABLE"
        )
        run = with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_RUN_CONTRACT,
                "canonical_run_id": definition.canonical_run_id,
                "aliases": list(definition.aliases),
                "family": definition.family,
                "stage": definition.stage,
                "scientific_role": definition.scientific_role,
                "selectable_for_primary_deployment": (
                    definition.selectable_for_primary_deployment
                ),
                "storage_class": definition.storage_class,
                "reconstruction_breadth": definition.reconstruction_breadth,
                "post_teacher_configuration": definition.post_teacher_configuration,
                "requires_selected_teacher": definition.requires_selected_teacher,
                "conditional_parent": definition.conditional_parent,
                "execution_status": execution_status,
                "paired_seed_ids": list(PAIRED_SEED_IDS),
                "seed_replicas": _seed_replicas(definition.canonical_run_id),
                "retained_state_rule": RETAINED_STATE_RULE,
                "measurement_status": MEASUREMENT_UNMEASURED,
                "measured_state_bytes": None,
                "measured_retained_bytes": None,
                "provisional_formula_inputs": {
                    "parameter_upper_bound": PROVISIONAL_PARAMETER_UPPER_BOUNDS[
                        definition.storage_class
                    ],
                    "weight_dtype_bytes": _WEIGHT_DTYPE_BYTES,
                    "serialization_headroom_fraction": (
                        _SERIALIZATION_HEADROOM_FRACTION
                    ),
                },
                "provisional_byte_categories": categories,
                "provisional_bytes": sum(categories.values()),
            }
        )
        runs.append(run)

    registry = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_REGISTRY_CONTRACT,
            "profile": "paired3",
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "retained_state_rule": RETAINED_STATE_RULE,
            "alternate_teacher_valid": bool(alternate_teacher_valid),
            "configuration_count": len(runs),
            "reconstruction_breadth_count": sum(
                bool(run["reconstruction_breadth"]) for run in runs
            ),
            "post_teacher_configuration_count": sum(
                bool(run["post_teacher_configuration"]) for run in runs
            ),
            "runnable_configuration_count": sum(
                run["execution_status"] == "RUNNABLE" for run in runs
            ),
            "alias_to_canonical": aliases,
            "storage_formula": {
                "contract": PREDICTION_ANCHORED_STORAGE_FORMULA_CONTRACT,
                "parameter_upper_bounds": dict(PROVISIONAL_PARAMETER_UPPER_BOUNDS),
                "weight_dtype_bytes": _WEIGHT_DTYPE_BYTES,
                "serialization_headroom_fraction": _SERIALIZATION_HEADROOM_FRACTION,
                "aggregate_seed_metrics_cap_bytes": _METRICS_CAP_BYTES,
                "bounded_log_cap_bytes": _LOG_CAP_BYTES,
                "run_manifest_cap_bytes": _RUN_MANIFEST_CAP_BYTES,
                "nonmedian_weights_persisted": False,
                "optimizer_state_persisted": False,
            },
            "runs": runs,
        }
    )
    validate_campaign_registry(registry)
    return registry


def validate_campaign_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_hash(payload, expected_contract=PREDICTION_ANCHORED_REGISTRY_CONTRACT)
    if payload.get("profile") != "paired3" or payload.get("paired_seed_ids") != list(
        PAIRED_SEED_IDS
    ):
        raise ValueError("campaign registry does not use the locked paired3 seed set")
    if payload.get("retained_state_rule") != RETAINED_STATE_RULE:
        raise ValueError("campaign registry retained-state rule changed")

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("campaign registry runs must be a list")
    definitions = {row.canonical_run_id: row for row in campaign_run_definitions()}
    if len(runs) != REGISTRY_CONFIGURATION_COUNT or set(
        run.get("canonical_run_id") for run in runs if isinstance(run, Mapping)
    ) != set(definitions):
        raise ValueError("campaign registry does not contain the locked 54 canonical rows")

    observed_aliases: dict[str, str] = {}
    reconstruction_count = 0
    post_teacher_count = 0
    runnable_count = 0
    alternate_valid = bool(payload.get("alternate_teacher_valid"))
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("campaign registry contains a non-object run")
        validate_content_hash(run, expected_contract=PREDICTION_ANCHORED_RUN_CONTRACT)
        run_id = str(run["canonical_run_id"])
        definition = definitions[run_id]
        locked_fields = {
            "contract": PREDICTION_ANCHORED_RUN_CONTRACT,
            "aliases": list(definition.aliases),
            "family": definition.family,
            "stage": definition.stage,
            "scientific_role": definition.scientific_role,
            "selectable_for_primary_deployment": (
                definition.selectable_for_primary_deployment
            ),
            "storage_class": definition.storage_class,
            "reconstruction_breadth": definition.reconstruction_breadth,
            "post_teacher_configuration": definition.post_teacher_configuration,
            "requires_selected_teacher": definition.requires_selected_teacher,
            "conditional_parent": definition.conditional_parent,
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "retained_state_rule": RETAINED_STATE_RULE,
        }
        for name, expected in locked_fields.items():
            if run.get(name) != expected:
                raise ValueError(f"registry run {run_id} changed locked field {name}")
        expected_status = (
            "SKIPPED_INVALID_PARENT"
            if definition.conditional_parent == "alternate_teacher_valid"
            and not alternate_valid
            else "RUNNABLE"
        )
        if run.get("execution_status") != expected_status:
            raise ValueError(f"registry run {run_id} has invalid conditional status")
        if not isinstance(run.get("scientific_role"), str) or not isinstance(
            run.get("selectable_for_primary_deployment"), bool
        ):
            raise ValueError(f"registry run {run_id} lacks explicit role/selectability")
        if run.get("seed_replicas") != _seed_replicas(run_id):
            raise ValueError(f"registry run {run_id} has invalid seed replica declarations")
        categories = run.get("provisional_byte_categories")
        if not isinstance(categories, Mapping) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in categories.values()
        ):
            raise ValueError(f"registry run {run_id} has invalid provisional categories")
        if int(run.get("provisional_bytes", -1)) != sum(categories.values()):
            raise ValueError(f"registry run {run_id} provisional byte sum mismatch")
        expected_categories = provisional_storage_categories(definition.storage_class)
        if dict(categories) != expected_categories:
            raise ValueError(f"registry run {run_id} changed provisional byte categories")
        expected_formula_inputs = {
            "parameter_upper_bound": PROVISIONAL_PARAMETER_UPPER_BOUNDS[
                definition.storage_class
            ],
            "weight_dtype_bytes": _WEIGHT_DTYPE_BYTES,
            "serialization_headroom_fraction": _SERIALIZATION_HEADROOM_FRACTION,
        }
        if run.get("provisional_formula_inputs") != expected_formula_inputs:
            raise ValueError(f"registry run {run_id} changed provisional formula inputs")
        measurement_status = run.get("measurement_status")
        if measurement_status not in {MEASUREMENT_UNMEASURED, MEASUREMENT_MEASURED}:
            raise ValueError(f"registry run {run_id} has invalid measurement status")
        if measurement_status == MEASUREMENT_UNMEASURED:
            if run.get("measured_state_bytes") is not None or run.get(
                "measured_retained_bytes"
            ) is not None:
                raise ValueError(f"unmeasured registry run {run_id} claims measured bytes")
        else:
            state_bytes = run.get("measured_state_bytes")
            retained_bytes = run.get("measured_retained_bytes")
            if (
                not isinstance(state_bytes, int)
                or isinstance(state_bytes, bool)
                or state_bytes <= 0
                or not isinstance(retained_bytes, int)
                or retained_bytes < state_bytes
            ):
                raise ValueError(f"measured registry run {run_id} has invalid byte counts")
        for alias in definition.aliases:
            if alias in definitions or alias in observed_aliases:
                raise ValueError(f"duplicate or colliding registry alias: {alias}")
            observed_aliases[alias] = run_id
        reconstruction_count += int(definition.reconstruction_breadth)
        post_teacher_count += int(definition.post_teacher_configuration)
        runnable_count += int(expected_status == "RUNNABLE")

    expected_counts = {
        "configuration_count": REGISTRY_CONFIGURATION_COUNT,
        "reconstruction_breadth_count": RECONSTRUCTION_BREADTH_COUNT,
        "post_teacher_configuration_count": POST_TEACHER_CONFIGURATION_COUNT,
        "runnable_configuration_count": runnable_count,
    }
    if reconstruction_count != RECONSTRUCTION_BREADTH_COUNT:
        raise AssertionError("internal reconstruction inventory arithmetic changed")
    if post_teacher_count != POST_TEACHER_CONFIGURATION_COUNT:
        raise AssertionError("internal post-teacher inventory arithmetic changed")
    for name, expected in expected_counts.items():
        if int(payload.get(name, -1)) != expected:
            raise ValueError(f"campaign registry count mismatch for {name}")
    if payload.get("alias_to_canonical") != observed_aliases:
        raise ValueError("campaign registry alias map mismatch")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        **expected_counts,
        "alias_count": len(observed_aliases),
    }


def resolve_registry_run(payload: Mapping[str, Any], run_id: str) -> Mapping[str, Any]:
    validate_campaign_registry(payload)
    canonical = payload["alias_to_canonical"].get(run_id, run_id)
    for run in payload["runs"]:
        if run["canonical_run_id"] == canonical:
            return run
    raise KeyError(f"unknown campaign run ID: {run_id}")


def dense_field_cache_projection(
    *,
    event_count: int,
    particle_width: int,
    channels: int = 50,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    values = {
        "event_count": event_count,
        "particle_width": particle_width,
        "channels": channels,
        "dtype_bytes": dtype_bytes,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
        raise ValueError("dense cache projection inputs must be positive integers")
    total = event_count * particle_width * channels * dtype_bytes
    return {
        **values,
        "projected_bytes": total,
        "projected_gib": total / 1024**3,
        "includes_masks_or_metadata": False,
        "production_persistence_allowed": False,
    }


def _weights_with_headroom(parameter_count: int) -> int:
    raw = int(parameter_count) * _WEIGHT_DTYPE_BYTES
    return raw + int(math.ceil(raw * _SERIALIZATION_HEADROOM_FRACTION))


def build_provisional_storage_projection(
    registry: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
    *,
    particle_width: int,
    selected_budget_bytes: int = NORMAL_PILOT_BUDGET_BYTES,
) -> dict[str, Any]:
    validate_campaign_registry(registry)
    validate_content_hash(
        child_manifest, expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    if selected_budget_bytes not in {
        NORMAL_PILOT_BUDGET_BYTES,
        PAIRED3_HARD_CEILING_BYTES,
    }:
        raise ValueError("selected budget must be the locked 5 or 6 GiB mode")
    if not isinstance(particle_width, int) or isinstance(particle_width, bool) or particle_width <= 0:
        raise ValueError("particle_width must be a positive integer")

    run_bytes = sum(int(run["provisional_bytes"]) for run in registry["runs"])
    stack_train_distill_count = int(
        child_manifest["children"]["stack_train_distill"]["count"]
    )
    target_namespace_count = 4 if registry["alternate_teacher_valid"] else 3
    target_logits = stack_train_distill_count * 10 * 4 * target_namespace_count
    child_manifest_bytes = len(canonical_json_bytes(child_manifest)) + 1
    r0_weights = _weights_with_headroom(16_000_000)
    final_bundle = _weights_with_headroom(16_000_000 + 24_000_000 + 12_000_000)
    fixed_categories = {
        "child_split_manifest": child_manifest_bytes,
        "r0_weights": r0_weights,
        "target_logit_namespaces": target_logits,
        "recipes_bindings_reports": 32 * 1024**2,
        "final_deployable_bundle": final_bundle,
    }
    projected = run_bytes + sum(fixed_categories.values())
    parent_counts = child_manifest["split_config"]["parent_split_counts"]
    non_final_count = sum(
        int(parent_counts[name])
        for name in ("model_train", "model_val", "stack_train", "stack_val")
    )
    dense_projection = dense_field_cache_projection(
        event_count=non_final_count,
        particle_width=particle_width,
    )
    unmeasured = [
        run["canonical_run_id"]
        for run in registry["runs"]
        if run["execution_status"] == "RUNNABLE"
        and run["measurement_status"] != MEASUREMENT_MEASURED
    ]
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STORAGE_PROJECTION_CONTRACT,
            "phase": "PROVISIONAL",
            "registry_sha256": registry["content_hash"],
            "child_manifest_sha256": child_manifest["content_hash"],
            "selected_budget_bytes": int(selected_budget_bytes),
            "run_provisional_bytes": run_bytes,
            "fixed_provisional_byte_categories": fixed_categories,
            "projected_persistent_bytes": projected,
            "provisional_budget_ok": projected <= selected_budget_bytes,
            "unmeasured_runnable_run_ids": unmeasured,
            "production_submission_allowed": False,
            "production_blocker": "runnable registry rows remain UNMEASURED",
            "dense_field_cache_projection": dense_projection,
        }
    )


def record_registry_measurements(
    registry: Mapping[str, Any],
    measured_state_bytes: Mapping[str, int],
) -> dict[str, Any]:
    """Return a re-hashed registry with supplied canonical rows marked measured."""

    validate_campaign_registry(registry)
    output = deepcopy(dict(registry))
    output.pop("content_hash", None)
    known = {run["canonical_run_id"] for run in output["runs"]}
    unknown = set(measured_state_bytes) - known
    if unknown:
        raise KeyError(f"measurements contain unknown canonical run IDs: {sorted(unknown)}")
    for run in output["runs"]:
        run_id = run["canonical_run_id"]
        if run_id not in measured_state_bytes:
            continue
        state_bytes = measured_state_bytes[run_id]
        if not isinstance(state_bytes, int) or isinstance(state_bytes, bool) or state_bytes <= 0:
            raise ValueError(f"measurement for {run_id} must be a positive integer")
        run.pop("content_hash", None)
        non_state_caps = (
            _METRICS_CAP_BYTES + _LOG_CAP_BYTES + _RUN_MANIFEST_CAP_BYTES
        )
        run["measurement_status"] = MEASUREMENT_MEASURED
        run["measured_state_bytes"] = state_bytes
        run["measured_retained_bytes"] = (
            state_bytes
            + int(math.ceil(state_bytes * _SERIALIZATION_HEADROOM_FRACTION))
            + non_state_caps
        )
        run.update(with_content_hash(run))
    output = with_content_hash(output)
    validate_campaign_registry(output)
    return output


def require_production_ready(
    registry: Mapping[str, Any],
    *,
    fixed_persistent_bytes: int,
    selected_budget_bytes: int,
) -> dict[str, Any]:
    """Reject provisional rows; later steps call this before production submit."""

    validate_campaign_registry(registry)
    if selected_budget_bytes not in {
        NORMAL_PILOT_BUDGET_BYTES,
        PAIRED3_HARD_CEILING_BYTES,
    }:
        raise ValueError("selected budget must be the locked 5 or 6 GiB mode")
    if not isinstance(fixed_persistent_bytes, int) or fixed_persistent_bytes < 0:
        raise ValueError("fixed_persistent_bytes must be a non-negative integer")
    unmeasured = [
        run["canonical_run_id"]
        for run in registry["runs"]
        if run["execution_status"] == "RUNNABLE"
        and run["measurement_status"] != MEASUREMENT_MEASURED
    ]
    if unmeasured:
        raise PermissionError(
            "production submission forbidden while runnable rows are UNMEASURED: "
            + ", ".join(unmeasured)
        )
    retained = sum(
        int(run["measured_retained_bytes"])
        for run in registry["runs"]
        if run["execution_status"] == "RUNNABLE"
    )
    projected = retained + fixed_persistent_bytes
    if projected > selected_budget_bytes:
        raise PermissionError(
            f"measured storage projection {projected} exceeds budget {selected_budget_bytes}"
        )
    return {
        "ok": True,
        "production_submission_allowed": True,
        "registry_sha256": registry["content_hash"],
        "measured_run_bytes": retained,
        "fixed_persistent_bytes": fixed_persistent_bytes,
        "projected_persistent_bytes": projected,
        "selected_budget_bytes": selected_budget_bytes,
    }


def build_step1_report(
    *,
    registry: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
    storage_projection: Mapping[str, Any],
) -> dict[str, Any]:
    registry_audit = validate_campaign_registry(registry)
    validate_content_hash(
        storage_projection,
        expected_contract=PREDICTION_ANCHORED_STORAGE_PROJECTION_CONTRACT,
    )
    if storage_projection.get("registry_sha256") != registry["content_hash"]:
        raise ValueError("storage projection is bound to a different registry")
    if storage_projection.get("child_manifest_sha256") != child_manifest["content_hash"]:
        raise ValueError("storage projection is bound to different child splits")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP1_REPORT_CONTRACT,
            "ok": True,
            "step": 1,
            "scope": "contracts_splits_registry_provisional_quota",
            "registry_sha256": registry["content_hash"],
            "child_manifest_sha256": child_manifest["content_hash"],
            "storage_projection_sha256": storage_projection["content_hash"],
            "counts": registry_audit,
            "child_splits": child_split_summary(child_manifest),
            "runs": [
                {
                    "canonical_run_id": run["canonical_run_id"],
                    "aliases": run["aliases"],
                    "stage": run["stage"],
                    "scientific_role": run["scientific_role"],
                    "selectable_for_primary_deployment": run[
                        "selectable_for_primary_deployment"
                    ],
                    "execution_status": run["execution_status"],
                    "seed_replicas": run["seed_replicas"],
                    "retained_state_rule": run["retained_state_rule"],
                    "measurement_status": run["measurement_status"],
                    "provisional_byte_categories": run[
                        "provisional_byte_categories"
                    ],
                    "provisional_bytes": run["provisional_bytes"],
                    "run_contract_sha256": run["content_hash"],
                }
                for run in registry["runs"]
            ],
            "storage": dict(storage_projection),
            "production_submission_allowed": False,
            "production_blockers": list(
                storage_projection["unmeasured_runnable_run_ids"]
            ),
            "placeholder_models_instantiated": False,
        }
    )


def write_step1_artifacts(
    output_dir: str | Path,
    *,
    child_manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    storage_projection: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_campaign_registry(registry)
    validate_content_hash(
        child_manifest, expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    validate_content_hash(
        storage_projection,
        expected_contract=PREDICTION_ANCHORED_STORAGE_PROJECTION_CONTRACT,
    )
    validate_content_hash(report, expected_contract=PREDICTION_ANCHORED_STEP1_REPORT_CONTRACT)
    expected_bindings = {
        "registry_sha256": registry["content_hash"],
        "child_manifest_sha256": child_manifest["content_hash"],
        "storage_projection_sha256": storage_projection["content_hash"],
    }
    for name, expected in expected_bindings.items():
        if report.get(name) != expected:
            raise ValueError(f"Step 1 report binding mismatch for {name}")
    if storage_projection.get("registry_sha256") != registry["content_hash"] or storage_projection.get(
        "child_manifest_sha256"
    ) != child_manifest["content_hash"]:
        raise ValueError("Step 1 storage projection lineage mismatch")
    if report.get("production_submission_allowed") is not False:
        raise ValueError("Step 1 report must not authorize production submission")

    root = Path(output_dir)
    paths = {
        "child_manifest": root / "contracts" / "split_children.json",
        "registry": root / "contracts" / "campaign_registry.json",
        "storage_projection": root / "contracts" / "provisional_storage.json",
        "step1_report": root / "reports" / "step1_dry_run_report.json",
    }
    receipts = {
        "child_manifest": write_immutable_json(paths["child_manifest"], child_manifest),
        "registry": write_immutable_json(paths["registry"], registry),
        "storage_projection": write_immutable_json(
            paths["storage_projection"], storage_projection
        ),
        "step1_report": write_immutable_json(paths["step1_report"], report),
    }
    return {
        "ok": True,
        "contract": "prediction_anchored_step1_publication_v1",
        "output_dir": str(root.resolve()),
        "receipts": receipts,
    }


__all__ = [
    "CampaignRunDefinition",
    "MEASUREMENT_MEASURED",
    "MEASUREMENT_UNMEASURED",
    "NORMAL_PILOT_BUDGET_BYTES",
    "PAIRED3_HARD_CEILING_BYTES",
    "PAIRED_SEED_IDS",
    "POST_TEACHER_CONFIGURATION_COUNT",
    "PREDICTION_ANCHORED_REGISTRY_CONTRACT",
    "PREDICTION_ANCHORED_RUN_CONTRACT",
    "PREDICTION_ANCHORED_STEP1_REPORT_CONTRACT",
    "PREDICTION_ANCHORED_STORAGE_FORMULA_CONTRACT",
    "PREDICTION_ANCHORED_STORAGE_PROJECTION_CONTRACT",
    "PROVISIONAL_PARAMETER_UPPER_BOUNDS",
    "RECONSTRUCTION_BREADTH_COUNT",
    "REGISTRY_CONFIGURATION_COUNT",
    "RETAINED_STATE_RULE",
    "build_campaign_registry",
    "build_provisional_storage_projection",
    "build_step1_report",
    "campaign_run_definitions",
    "dense_field_cache_projection",
    "provisional_storage_categories",
    "record_registry_measurements",
    "require_production_ready",
    "resolve_registry_run",
    "validate_campaign_registry",
    "write_step1_artifacts",
]
