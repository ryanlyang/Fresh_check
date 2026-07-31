"""Fail-closed validation for the canonical Stage-K evidence bundle."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import (
    SEMANTIC_CONTROL_POLICY,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
)
from .late_plan_factories import SEMANTIC_CONTROL_KINDS
from .predictor_bundle import PIPELINE_SEEDS
from .registry import EXPERT_ORDER
from .step7 import STAGE_E_SHAPES


SEMANTIC_CONTROLS_CONTRACT = "retb_stage_k_semantic_controls_bundle_v5"
RECONSTRUCTION_CONDITIONS = frozenset(
    {"frozen_reconstruction", "token_refiner", "unrestricted_fusion"}
)
CLASSIFICATION_METRICS = frozenset(
    {"accuracy", "cross_entropy", "macro_per_class_accuracy"}
)
PREDICTOR_METRICS = frozenset(
    {"accuracy", "cross_entropy", "normalized_token_error"}
)
RELATION_EXPERTS = ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")
CONSUMER_KINDS = ("HF_UNRESTRICTED", "HF_ADAPTER")
EVIDENCE_FAMILIES = (
    "relation_and_predictor_wave",
    "token_substitutions",
    "native_bypass",
    "reconstruction_comparison",
    "fixed_vs_learned_bias_scale",
    "predictor_direct_gated_logit_only",
)


def _exact_keys(payload: Any, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"Stage-K {name} schema differs")
    return payload


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Stage-K {name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stage-K {name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"Stage-K {name} must be a finite real number")
    return result


def _metrics(payload: Any, names: frozenset[str], *, name: str) -> dict[str, float]:
    row = _exact_keys(payload, set(names), name=name)
    return {key: _finite(row[key], name=f"{name}.{key}") for key in names}


def _validate_delta(
    control: Any,
    reference: Any,
    delta: Any,
    metric_names: frozenset[str],
    *,
    name: str,
) -> None:
    values = _metrics(control, metric_names, name=f"{name}.metrics")
    references = _metrics(
        reference, metric_names, name=f"{name}.reference_metrics"
    )
    expected_keys = {f"{key}_control_minus_reference" for key in metric_names}
    serialized = _exact_keys(delta, expected_keys, name=f"{name}.metric_deltas")
    for key in metric_names:
        observed = _finite(
            serialized[f"{key}_control_minus_reference"],
            name=f"{name}.metric_deltas.{key}",
        )
        expected = values[key] - references[key]
        if observed != expected:
            raise ValueError(f"Stage-K {name} metric delta differs")


def _sha(value: Any, *, name: str) -> str:
    return require_sha256(value, name=f"Stage-K {name}")


def _eligible_population(payload: Any, *, name: str) -> None:
    row = _exact_keys(
        payload,
        {
            "total_event_count",
            "evaluated_event_count",
            "evaluated_fraction",
            "excluded_singleton_stratum_event_count",
            "excluded_identity_order_sha256",
        },
        name=name,
    )
    total = int(row["total_event_count"])
    evaluated = int(row["evaluated_event_count"])
    excluded = int(row["excluded_singleton_stratum_event_count"])
    fraction = _finite(row["evaluated_fraction"], name=f"{name}.evaluated_fraction")
    _sha(row["excluded_identity_order_sha256"], name=f"{name}.excluded_identity_order")
    if total <= 0 or evaluated <= 0 or evaluated + excluded != total or fraction != evaluated / total:
        raise ValueError(f"Stage-K {name} accounting differs")


def _comparison(
    record: Mapping[str, Any],
    *,
    coordinate_keys: set[str],
    metric_names: frozenset[str] = CLASSIFICATION_METRICS,
    name: str,
) -> None:
    expected = {
        "record_type",
        *coordinate_keys,
        "condition_id",
        "metrics",
        "reference_metrics",
        "metric_deltas",
        "source_artifact_sha256",
    }
    row = _exact_keys(record, expected, name=name)
    if row["record_type"] != "metric_comparison" or not row["condition_id"]:
        raise ValueError(f"Stage-K {name} comparison semantics differ")
    _validate_delta(
        row["metrics"], row["reference_metrics"], row["metric_deltas"],
        metric_names, name=name,
    )
    _sha(row["source_artifact_sha256"], name=f"{name}.source_artifact_sha256")


def _relation_or_predictor_records(
    control_id: str, records: Sequence[Mapping[str, Any]]
) -> None:
    conditions = {
        "RELATION_WITHIN_JET_SHUFFLE": "RELATION__within_jet_cyclic",
        "RELATION_WRONG_EVENT_MATCHED_MULTIPLICITY": (
            "RELATION__wrong_event_matched_multiplicity"
        ),
        "RELATION_DIRECTIONAL_ENDPOINT_SWAP": "RELATION__directional_endpoint_swap",
        "PREDICTOR_ZERO_HLT_EVIDENCE": "PREDICTOR__zero_hlt_evidence",
        "PREDICTOR_SHUFFLE_HLT_EVIDENCE": (
            "PREDICTOR__shuffle_hlt_evidence_between_events"
        ),
        "PREDICTOR_REMOVE_NATIVE_PARTICLE_CONTEXT": (
            "PREDICTOR__remove_native_particle_context"
        ),
        "PREDICTOR_REMOVE_NONCORRESPONDING_BANKS": (
            "PREDICTOR__remove_noncorresponding_expert_banks"
        ),
    }
    expected = (
        {(role, seed, expert, f"RELATION__zero__{expert}")
         for role in STAGE_E_SHAPES for seed in PIPELINE_SEEDS
         for expert in RELATION_EXPERTS}
        if control_id == "RELATION_ZERO"
        else {(role, seed, None, conditions[control_id])
              for role in STAGE_E_SHAPES for seed in PIPELINE_SEEDS}
    )
    observed = set()
    for index, record in enumerate(records):
        name = f"{control_id}[{index}]"
        _comparison(
            record,
            coordinate_keys={
                "shape_role", "pipeline_seed", "relation_expert",
                "event_count", "eligible_population",
            },
            name=name,
        )
        if int(record["event_count"]) <= 0:
            raise ValueError(f"Stage-K {name} event count differs")
        _eligible_population(record["eligible_population"], name=f"{name}.eligible_population")
        observed.add((
            str(record["shape_role"]), int(record["pipeline_seed"]),
            record["relation_expert"], str(record["condition_id"]),
        ))
    if observed != expected or len(records) != len(expected):
        raise ValueError(f"Stage-K {control_id} coordinate coverage differs")


def _token_records(control_id: str, records: Sequence[Mapping[str, Any]]) -> None:
    fixed = {
        "TOKEN_SLOT_PERMUTE": "SLOT_PERMUTED_TARGETS",
        "TOKEN_MATCHED_GAUSSIAN_NOISE": "MATCHED_GAUSSIAN_NOISE",
    }
    prefixes = {
        "TOKEN_WITHIN_CLASS_MEAN_BANK": "WITHIN_CLASS_MEAN_TARGETS__",
        "TOKEN_WRONG_EVENT_SAME_CLASS_BANK": "WITHIN_CLASS_WRONG_EVENT_BANK__",
        "TOKEN_WRONG_EVENT_UNMATCHED_BANK": "WRONG_EVENT_BANK__",
        "TOKEN_ZERO_BANK": "ZERO_ORACLE_BANK__",
    }
    expected = (
        {(seed, fixed[control_id]) for seed in PIPELINE_SEEDS}
        if control_id in fixed
        else {(seed, f"{prefixes[control_id]}{expert}")
              for seed in PIPELINE_SEEDS for expert in EXPERT_ORDER}
    )
    observed = set()
    for index, record in enumerate(records):
        _comparison(
            record,
            coordinate_keys={"pipeline_seed"},
            name=f"{control_id}[{index}]",
        )
        observed.add((int(record["pipeline_seed"]), str(record["condition_id"])))
    if observed != expected or len(records) != len(expected):
        raise ValueError(f"Stage-K {control_id} coordinate coverage differs")


def _bank_diagnostics(records: Sequence[Mapping[str, Any]]) -> None:
    observed = set()
    for index, record in enumerate(records):
        name = f"TOKEN_NORM_COVARIANCE[{index}]"
        row = _exact_keys(
            record,
            {
                "record_type", "pipeline_seed", "predicted_bank_diagnostics",
                "oracle_bank_diagnostics", "source_artifact_sha256",
            },
            name=name,
        )
        if row["record_type"] != "bank_diagnostic":
            raise ValueError(f"Stage-K {name} record type differs")
        for bank_name in ("predicted_bank_diagnostics", "oracle_bank_diagnostics"):
            banks = _exact_keys(row[bank_name], set(EXPERT_ORDER), name=f"{name}.{bank_name}")
            for expert, diagnostic in banks.items():
                values = _exact_keys(
                    diagnostic, {"mean_event_l2_norm", "covariance_trace"},
                    name=f"{name}.{bank_name}.{expert}",
                )
                if any(_finite(value, name=f"{name}.{key}") < 0 for key, value in values.items()):
                    raise ValueError(f"Stage-K {name} diagnostic differs")
        _sha(row["source_artifact_sha256"], name=f"{name}.source_artifact_sha256")
        observed.add(int(row["pipeline_seed"]))
    if observed != set(PIPELINE_SEEDS) or len(records) != len(PIPELINE_SEEDS):
        raise ValueError("Stage-K TOKEN_NORM_COVARIANCE coordinate coverage differs")


def _bypass_records(control_id: str, records: Sequence[Mapping[str, Any]]) -> None:
    names = {
        "BYPASS_NATIVE_REMOVED": "NATIVE_BRANCH_REMOVED",
        "BYPASS_RECONSTRUCTED_REMOVED": "RECONSTRUCTED_BRANCH_REMOVED",
        "BYPASS_NATIVE_DROPPED_EVAL": "NATIVE_BRANCH_DROPPED_AT_EVALUATION",
        "BYPASS_GAMMA_ZERO": "RESIDUAL_GAMMA_ZERO",
        "BYPASS_SOURCE_EMBEDDINGS_SWAPPED": "SOURCE_EMBEDDINGS_SWAPPED",
    }
    consumers = ("HF_ADAPTER",) if control_id == "BYPASS_GAMMA_ZERO" else CONSUMER_KINDS
    expected = {(role, seed, consumer, names[control_id])
                for role in STAGE_E_SHAPES for seed in PIPELINE_SEEDS
                for consumer in consumers}
    observed = set()
    for index, record in enumerate(records):
        _comparison(
            record,
            coordinate_keys={"shape_role", "pipeline_seed", "consumer_kind"},
            name=f"{control_id}[{index}]",
        )
        observed.add((str(record["shape_role"]), int(record["pipeline_seed"]),
                      str(record["consumer_kind"]), str(record["condition_id"])))
    if observed != expected or len(records) != len(expected):
        raise ValueError(f"Stage-K {control_id} coordinate coverage differs")


def _bias_records(records: Sequence[Mapping[str, Any]]) -> None:
    observed = set()
    for index, record in enumerate(records):
        name = f"RELATION_FIXED_VS_LEARNED_SCALE[{index}]"
        _comparison(
            record,
            coordinate_keys={"expert_id", "fixed_run_id", "learned_run_id"},
            name=name,
        )
        if record["condition_id"] != "B_DUAL_GATED_VS_B_DUAL_FIXED":
            raise ValueError(f"Stage-K {name} condition differs")
        observed.add(str(record["expert_id"]))
    if observed != set(EXPERT_ORDER) or len(records) != len(EXPERT_ORDER):
        raise ValueError("Stage-K fixed-versus-learned coordinate coverage differs")


def _predictor_evaluation(record: Mapping[str, Any], *, name: str) -> None:
    row = _exact_keys(
        record,
        {
            "record_type", "candidate_id", "pipeline_seed", "architecture",
            "objective_id", "metrics", "bundle_input_index_sha256",
            "selector_configuration_sha256", "candidate_manifest_sha256",
            "materialized_run_sha256", "inference_manifest_sha256",
            "metric_artifact_sha256",
        },
        name=name,
    )
    if row["record_type"] != "predictor_evaluation" or not row["candidate_id"]:
        raise ValueError(f"Stage-K {name} predictor evaluation differs")
    _metrics(row["metrics"], PREDICTOR_METRICS, name=f"{name}.metrics")
    for key in (
        "bundle_input_index_sha256", "selector_configuration_sha256",
        "candidate_manifest_sha256", "materialized_run_sha256",
        "inference_manifest_sha256", "metric_artifact_sha256",
    ):
        _sha(row[key], name=f"{name}.{key}")


def _mean(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        key: sum(float(row["metrics"][key]) for row in records) / len(records)
        for key in PREDICTOR_METRICS
    }


def _architecture_evidence(
    payload: Any, row_map: Mapping[str, Mapping[str, Any]]
) -> None:
    expected_keys = {
        "bundle_input_index_sha256", "selector_configuration_sha256",
        "candidate_manifest_hashes", "coordinate_manifest_hashes",
        "direct_evaluations", "gated_evaluations", "W_LOGIT_ONLY_evaluations",
        "non_logit_evaluations", "direct_evaluations_sha256",
        "gated_evaluations_sha256", "W_LOGIT_ONLY_evaluations_sha256",
        "non_logit_evaluations_sha256",
        "all_evidence_is_materialized_val_design_inference",
    }
    evidence = _exact_keys(payload, expected_keys, name="predictor_architecture_evidence")
    index_hash = _sha(evidence["bundle_input_index_sha256"], name="architecture.index")
    config_hash = _sha(evidence["selector_configuration_sha256"], name="architecture.configuration")
    candidate_hashes = evidence["candidate_manifest_hashes"]
    coordinate_hashes = evidence["coordinate_manifest_hashes"]
    if (
        evidence["all_evidence_is_materialized_val_design_inference"] is not True
        or not isinstance(candidate_hashes, list) or not candidate_hashes
        or not isinstance(coordinate_hashes, list) or not coordinate_hashes
        or len(set(candidate_hashes)) != len(candidate_hashes)
        or len(set(coordinate_hashes)) != len(coordinate_hashes)
    ):
        raise ValueError("Stage-K predictor architecture evidence differs")
    for index, value in enumerate([*candidate_hashes, *coordinate_hashes]):
        _sha(value, name=f"architecture.manifest_hash[{index}]")
    groups = {
        "direct_evaluations": "direct_evaluations_sha256",
        "gated_evaluations": "gated_evaluations_sha256",
        "W_LOGIT_ONLY_evaluations": "W_LOGIT_ONLY_evaluations_sha256",
        "non_logit_evaluations": "non_logit_evaluations_sha256",
    }
    all_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for group_name, hash_name in groups.items():
        records = evidence[group_name]
        if not isinstance(records, list) or not records or canonical_sha256(records) != evidence[hash_name]:
            raise ValueError(f"Stage-K {group_name} hash differs")
        _sha(evidence[hash_name], name=hash_name)
        group_coordinates = set()
        for position, record in enumerate(records):
            _predictor_evaluation(record, name=f"{group_name}[{position}]")
            if record["bundle_input_index_sha256"] != index_hash or record["selector_configuration_sha256"] != config_hash:
                raise ValueError("Stage-K predictor evaluation selector lineage differs")
            key = (str(record["candidate_id"]), int(record["pipeline_seed"]))
            group_coordinates.add(key)
            previous = all_by_key.setdefault(key, record)
            if previous != record:
                raise ValueError("Stage-K predictor evaluation duplicate differs")
            if group_name == "direct_evaluations" and record["architecture"] != "A3_SLOT_DECODER_DIRECT":
                raise ValueError("Stage-K direct architecture evidence differs")
            if group_name == "gated_evaluations" and record["architecture"] != "A4_SLOT_DECODER_GATED":
                raise ValueError("Stage-K gated architecture evidence differs")
            if group_name == "W_LOGIT_ONLY_evaluations" and record["objective_id"] != "W_LOGIT_ONLY":
                raise ValueError("Stage-K logit-only objective evidence differs")
            if group_name == "non_logit_evaluations" and record["objective_id"] == "W_LOGIT_ONLY":
                raise ValueError("Stage-K non-logit objective evidence differs")
        candidates = {candidate for candidate, _ in group_coordinates}
        if (
            len(records) != len(group_coordinates)
            or group_coordinates
            != {(candidate, seed) for candidate in candidates for seed in PIPELINE_SEEDS}
        ):
            raise ValueError(f"Stage-K {group_name} seed coverage differs")
    logit_keys = {(row["candidate_id"], int(row["pipeline_seed"])) for row in evidence["W_LOGIT_ONLY_evaluations"]}
    non_logit_keys = {(row["candidate_id"], int(row["pipeline_seed"])) for row in evidence["non_logit_evaluations"]}
    if logit_keys & non_logit_keys or {
        row["candidate_manifest_sha256"] for row in all_by_key.values()
    } != set(candidate_hashes):
        raise ValueError("Stage-K predictor candidate-manifest coverage differs")

    comparisons = (
        ("PREDICTOR_DIRECT_VS_GATED", evidence["direct_evaluations"], evidence["gated_evaluations"], "GATED_MEAN_VS_DIRECT_MEAN"),
        ("PREDICTOR_W_LOGIT_ONLY", evidence["non_logit_evaluations"], evidence["W_LOGIT_ONLY_evaluations"], "W_LOGIT_ONLY_MEAN_VS_NON_LOGIT_MEAN"),
    )
    for control_id, reference_rows, control_rows, condition_id in comparisons:
        aggregate = {
            "record_type": "aggregate_metric_comparison",
            "condition_id": condition_id,
            "metrics": _mean(control_rows),
            "reference_metrics": _mean(reference_rows),
            "metric_deltas": {
                f"{key}_control_minus_reference": _mean(control_rows)[key] - _mean(reference_rows)[key]
                for key in PREDICTOR_METRICS
            },
            "aggregation": "arithmetic_mean_over_predeclared_val_design_rows",
        }
        expected_records = (
            [*reference_rows, *control_rows, aggregate]
            if control_id == "PREDICTOR_DIRECT_VS_GATED"
            else [*control_rows, aggregate]
        )
        if row_map[control_id]["metric_records"] != expected_records:
            raise ValueError(f"Stage-K {control_id} evidence records differ")


def _reconstruction(payload: Any) -> str:
    if not isinstance(payload, list):
        raise ValueError("Stage-K reconstruction records differ")
    expected_coordinates = {(role, seed) for role in STAGE_E_SHAPES for seed in PIPELINE_SEEDS}
    actual_coordinates = set()
    composite_hashes = []
    for index, row in enumerate(payload):
        name = f"reconstruction_metric_records[{index}]"
        record = _exact_keys(
            row,
            {
                "shape_role", "pipeline_seed", "condition_metrics",
                "condition_metric_artifact_hashes",
                "condition_metric_deltas_vs_frozen_reconstruction",
            },
            name=name,
        )
        coordinate = (str(record["shape_role"]), int(record["pipeline_seed"]))
        if coordinate in actual_coordinates:
            raise ValueError("Stage-K reconstruction coordinate is duplicated")
        conditions = _exact_keys(record["condition_metrics"], set(RECONSTRUCTION_CONDITIONS), name=f"{name}.conditions")
        hashes = _exact_keys(record["condition_metric_artifact_hashes"], set(RECONSTRUCTION_CONDITIONS), name=f"{name}.hashes")
        deltas = _exact_keys(record["condition_metric_deltas_vs_frozen_reconstruction"], {"token_refiner", "unrestricted_fusion"}, name=f"{name}.deltas")
        for condition in RECONSTRUCTION_CONDITIONS:
            _metrics(conditions[condition], CLASSIFICATION_METRICS, name=f"{name}.{condition}")
            _sha(hashes[condition], name=f"{name}.{condition}.artifact")
        for condition in ("token_refiner", "unrestricted_fusion"):
            _validate_delta(
                conditions[condition], conditions["frozen_reconstruction"],
                deltas[condition], CLASSIFICATION_METRICS,
                name=f"{name}.{condition}",
            )
        composite_hashes.append(canonical_sha256(dict(hashes)))
        actual_coordinates.add(coordinate)
    if actual_coordinates != expected_coordinates or len(payload) != len(expected_coordinates):
        raise ValueError("Stage-K reconstruction coordinate coverage differs")
    return canonical_sha256(composite_hashes)


def validate_stage_k_semantic_controls(
    payload: Mapping[str, Any],
    *,
    expected_source: Mapping[str, Any],
    expected_policy: Mapping[str, Any] = SEMANTIC_CONTROL_POLICY,
) -> str:
    """Reject incomplete, placeholder, stale, or internally inconsistent evidence."""

    digest = validate_content_hash(payload, expected_contract=SEMANTIC_CONTROLS_CONTRACT)
    expected_ids = list(SEMANTIC_CONTROL_KINDS)
    rows = payload.get("rows")
    reconstruction = payload.get("reconstruction_metric_records")
    evidence = payload.get("evidence_families")
    if (
        int(payload.get("schema_version", -1)) != 5
        or payload.get("source") != expected_source
        or payload.get("semantic_control_policy") != expected_policy
        or payload.get("semantic_control_policy") != SEMANTIC_CONTROL_POLICY
        or payload.get("control_ids") != expected_ids
        or payload.get("complete_coverage") is not True
        or payload.get("all_section_28_controls_have_real_evaluation_evidence") is not True
        or payload.get("scientific_underperformance_blocks_continuation") is not False
        or payload.get("stack_val_consumed") is not False
        or payload.get("final_test_consumed") is not False
        or not isinstance(rows, list) or len(rows) != len(expected_ids)
        or not isinstance(evidence, Mapping) or set(evidence) != set(EVIDENCE_FAMILIES)
    ):
        raise ValueError("Stage-K semantic-controls bundle semantics differ")
    for family, value in evidence.items():
        _sha(value, name=f"evidence_families.{family}")
    row_map: dict[str, Mapping[str, Any]] = {}
    expected_family = {}
    for control_id in expected_ids:
        expected_family[control_id] = (
            "relation_and_predictor_wave"
            if control_id.startswith(("RELATION_ZERO", "RELATION_WITHIN", "RELATION_WRONG", "RELATION_DIRECTIONAL", "PREDICTOR_ZERO", "PREDICTOR_SHUFFLE", "PREDICTOR_REMOVE"))
            else "fixed_vs_learned_bias_scale"
            if control_id == "RELATION_FIXED_VS_LEARNED_SCALE"
            else "predictor_direct_gated_logit_only"
            if control_id in {"PREDICTOR_DIRECT_VS_GATED", "PREDICTOR_W_LOGIT_ONLY"}
            else "native_bypass" if control_id.startswith("BYPASS_")
            else "token_substitutions"
        )
    for position, row in enumerate(rows):
        control = _exact_keys(
            row,
            {"control_id", "evidence_family", "artifact_sha256", "metric_records", "metric_record_count"},
            name=f"rows[{position}]",
        )
        control_id = str(control["control_id"])
        records = control["metric_records"]
        if (
            control_id in row_map or control_id != expected_ids[position]
            or control["evidence_family"] != expected_family.get(control_id)
            or control["artifact_sha256"] != evidence.get(control["evidence_family"])
            or not isinstance(records, list) or not records
            or int(control["metric_record_count"]) != len(records)
        ):
            raise ValueError("Stage-K per-control evidence binding differs")
        _sha(control["artifact_sha256"], name=f"rows[{position}].artifact_sha256")
        row_map[control_id] = control

    relation_controls = {
        "RELATION_ZERO", "RELATION_WITHIN_JET_SHUFFLE",
        "RELATION_WRONG_EVENT_MATCHED_MULTIPLICITY",
        "RELATION_DIRECTIONAL_ENDPOINT_SWAP", "PREDICTOR_ZERO_HLT_EVIDENCE",
        "PREDICTOR_SHUFFLE_HLT_EVIDENCE",
        "PREDICTOR_REMOVE_NATIVE_PARTICLE_CONTEXT",
        "PREDICTOR_REMOVE_NONCORRESPONDING_BANKS",
    }
    for control_id in relation_controls:
        _relation_or_predictor_records(control_id, row_map[control_id]["metric_records"])
    for control_id in {
        "TOKEN_SLOT_PERMUTE", "TOKEN_WITHIN_CLASS_MEAN_BANK",
        "TOKEN_WRONG_EVENT_SAME_CLASS_BANK", "TOKEN_WRONG_EVENT_UNMATCHED_BANK",
        "TOKEN_ZERO_BANK", "TOKEN_MATCHED_GAUSSIAN_NOISE",
    }:
        _token_records(control_id, row_map[control_id]["metric_records"])
    _bank_diagnostics(row_map["TOKEN_NORM_COVARIANCE"]["metric_records"])
    for control_id in {
        "BYPASS_NATIVE_REMOVED", "BYPASS_RECONSTRUCTED_REMOVED",
        "BYPASS_NATIVE_DROPPED_EVAL", "BYPASS_GAMMA_ZERO",
        "BYPASS_SOURCE_EMBEDDINGS_SWAPPED",
    }:
        _bypass_records(control_id, row_map[control_id]["metric_records"])
    _bias_records(row_map["RELATION_FIXED_VS_LEARNED_SCALE"]["metric_records"])
    _architecture_evidence(payload.get("predictor_architecture_evidence"), row_map)

    relation_sources = {}
    for control_id in relation_controls:
        for record in row_map[control_id]["metric_records"]:
            key = (record["shape_role"], int(record["pipeline_seed"]))
            relation_sources.setdefault(key, set()).add(record["source_artifact_sha256"])
    if set(relation_sources) != {(role, seed) for role in STAGE_E_SHAPES for seed in PIPELINE_SEEDS} or any(len(values) != 1 for values in relation_sources.values()):
        raise ValueError("Stage-K relation/predictor source coverage differs")
    relation_hashes = [next(iter(relation_sources[(role, seed)])) for role in STAGE_E_SHAPES for seed in PIPELINE_SEEDS]
    if canonical_sha256(relation_hashes) != evidence["relation_and_predictor_wave"]:
        raise ValueError("Stage-K relation/predictor evidence-family hash differs")

    token_sources = {}
    for control_id in {
        "TOKEN_SLOT_PERMUTE", "TOKEN_WITHIN_CLASS_MEAN_BANK",
        "TOKEN_WRONG_EVENT_SAME_CLASS_BANK", "TOKEN_WRONG_EVENT_UNMATCHED_BANK",
        "TOKEN_ZERO_BANK", "TOKEN_MATCHED_GAUSSIAN_NOISE", "TOKEN_NORM_COVARIANCE",
    }:
        for record in row_map[control_id]["metric_records"]:
            token_sources.setdefault(int(record["pipeline_seed"]), set()).add(record["source_artifact_sha256"])
    if set(token_sources) != set(PIPELINE_SEEDS) or any(len(values) != 1 for values in token_sources.values()) or canonical_sha256([next(iter(token_sources[seed])) for seed in PIPELINE_SEEDS]) != evidence["token_substitutions"]:
        raise ValueError("Stage-K token evidence-family hash differs")

    bypass_sources = {}
    for control_id in {key for key in expected_ids if key.startswith("BYPASS_")}:
        for record in row_map[control_id]["metric_records"]:
            key = (record["shape_role"], int(record["pipeline_seed"]), record["consumer_kind"])
            bypass_sources.setdefault(key, set()).add(record["source_artifact_sha256"])
    expected_bypass = {(role, seed, consumer) for role in STAGE_E_SHAPES for seed in PIPELINE_SEEDS for consumer in CONSUMER_KINDS}
    if set(bypass_sources) != expected_bypass or any(len(values) != 1 for values in bypass_sources.values()):
        raise ValueError("Stage-K bypass source coverage differs")
    bypass_hashes = [next(iter(bypass_sources[(role, seed, consumer)])) for role in STAGE_E_SHAPES for seed in PIPELINE_SEEDS for consumer in CONSUMER_KINDS]
    if canonical_sha256(bypass_hashes) != evidence["native_bypass"]:
        raise ValueError("Stage-K bypass evidence-family hash differs")

    bias_sources = {record["source_artifact_sha256"] for record in row_map["RELATION_FIXED_VS_LEARNED_SCALE"]["metric_records"]}
    if bias_sources != {evidence["fixed_vs_learned_bias_scale"]}:
        raise ValueError("Stage-K bias-scale evidence-family hash differs")
    if canonical_sha256(payload["predictor_architecture_evidence"]) != evidence["predictor_direct_gated_logit_only"]:
        raise ValueError("Stage-K predictor architecture family hash differs")
    if _reconstruction(reconstruction) != evidence["reconstruction_comparison"]:
        raise ValueError("Stage-K reconstruction evidence-family hash differs")
    return digest


__all__ = [
    "CLASSIFICATION_METRICS", "PREDICTOR_METRICS",
    "RECONSTRUCTION_CONDITIONS", "SEMANTIC_CONTROLS_CONTRACT",
    "validate_stage_k_semantic_controls",
]
