from __future__ import annotations

import hashlib
from typing import Any, Mapping

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    SEMANTIC_CONTROL_POLICY,
    canonical_sha256,
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import (
    SEMANTIC_CONTROL_KINDS,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import PIPELINE_SEEDS
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.step7 import STAGE_E_SHAPES


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metrics(offset: float = 0.0) -> dict[str, float]:
    return {
        "accuracy": 0.5 + offset,
        "cross_entropy": 1.0 - offset,
        "macro_per_class_accuracy": 0.45 + offset,
    }


def _deltas(control: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    return {
        f"{name}_control_minus_reference": float(control[name]) - float(reference[name])
        for name in control
    }


def _comparison(*, condition_id: str, source: str, coordinates: Mapping[str, Any]) -> dict[str, Any]:
    reference = _metrics()
    control = _metrics(-0.05)
    return {
        "record_type": "metric_comparison",
        **dict(coordinates),
        "condition_id": condition_id,
        "metrics": control,
        "reference_metrics": reference,
        "metric_deltas": _deltas(control, reference),
        "source_artifact_sha256": source,
    }


def _predictor_record(candidate: str, seed: int, architecture: str, objective: str,
                      *, index_hash: str, config_hash: str) -> dict[str, Any]:
    return {
        "record_type": "predictor_evaluation",
        "candidate_id": candidate,
        "pipeline_seed": seed,
        "architecture": architecture,
        "objective_id": objective,
        "metrics": {
            "accuracy": 0.5 + seed / 1_000_000,
            "cross_entropy": 1.0 - seed / 1_000_000,
            "normalized_token_error": 0.25 + seed / 1_000_000,
        },
        "bundle_input_index_sha256": index_hash,
        "selector_configuration_sha256": config_hash,
        "candidate_manifest_sha256": _digest(f"candidate:{candidate}"),
        "materialized_run_sha256": _digest(f"run:{candidate}:{seed}"),
        "inference_manifest_sha256": _digest(f"inference:{candidate}:{seed}"),
        "metric_artifact_sha256": _digest(f"metric:{candidate}:{seed}"),
    }


def _mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        name: sum(row["metrics"][name] for row in rows) / len(rows)
        for name in ("accuracy", "cross_entropy", "normalized_token_error")
    }


def build_valid_semantic_controls(source_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {
        control_id: [] for control_id in SEMANTIC_CONTROL_KINDS
    }
    relation_hashes = []
    bypass_hashes = []
    for role in STAGE_E_SHAPES:
        for seed in PIPELINE_SEEDS:
            relation_hash = _digest(f"relation:{role}:{seed}")
            relation_hashes.append(relation_hash)
            eligible = {
                "total_event_count": 10,
                "evaluated_event_count": 10,
                "evaluated_fraction": 1.0,
                "excluded_singleton_stratum_event_count": 0,
                "excluded_identity_order_sha256": _digest(""),
            }
            for expert in ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"):
                records["RELATION_ZERO"].append(_comparison(
                    condition_id=f"RELATION__zero__{expert}", source=relation_hash,
                    coordinates={"shape_role": role, "pipeline_seed": seed,
                                 "relation_expert": expert, "event_count": 10,
                                 "eligible_population": dict(eligible)},
                ))
            relation_conditions = {
                "RELATION_WITHIN_JET_SHUFFLE": "RELATION__within_jet_cyclic",
                "RELATION_WRONG_EVENT_MATCHED_MULTIPLICITY": "RELATION__wrong_event_matched_multiplicity",
                "RELATION_DIRECTIONAL_ENDPOINT_SWAP": "RELATION__directional_endpoint_swap",
                "PREDICTOR_ZERO_HLT_EVIDENCE": "PREDICTOR__zero_hlt_evidence",
                "PREDICTOR_SHUFFLE_HLT_EVIDENCE": "PREDICTOR__shuffle_hlt_evidence_between_events",
                "PREDICTOR_REMOVE_NATIVE_PARTICLE_CONTEXT": "PREDICTOR__remove_native_particle_context",
                "PREDICTOR_REMOVE_NONCORRESPONDING_BANKS": "PREDICTOR__remove_noncorresponding_expert_banks",
            }
            for control_id, condition in relation_conditions.items():
                records[control_id].append(_comparison(
                    condition_id=condition, source=relation_hash,
                    coordinates={"shape_role": role, "pipeline_seed": seed,
                                 "relation_expert": None, "event_count": 10,
                                 "eligible_population": dict(eligible)},
                ))
            for consumer in ("HF_UNRESTRICTED", "HF_ADAPTER"):
                bypass_hash = _digest(f"bypass:{role}:{seed}:{consumer}")
                bypass_hashes.append(bypass_hash)
                bypass_conditions = {
                    "BYPASS_NATIVE_REMOVED": "NATIVE_BRANCH_REMOVED",
                    "BYPASS_RECONSTRUCTED_REMOVED": "RECONSTRUCTED_BRANCH_REMOVED",
                    "BYPASS_NATIVE_DROPPED_EVAL": "NATIVE_BRANCH_DROPPED_AT_EVALUATION",
                    "BYPASS_SOURCE_EMBEDDINGS_SWAPPED": "SOURCE_EMBEDDINGS_SWAPPED",
                }
                if consumer == "HF_ADAPTER":
                    bypass_conditions["BYPASS_GAMMA_ZERO"] = "RESIDUAL_GAMMA_ZERO"
                for control_id, condition in bypass_conditions.items():
                    records[control_id].append(_comparison(
                        condition_id=condition, source=bypass_hash,
                        coordinates={"shape_role": role, "pipeline_seed": seed,
                                     "consumer_kind": consumer},
                    ))

    token_hashes = []
    for seed in PIPELINE_SEEDS:
        source = _digest(f"token:{seed}")
        token_hashes.append(source)
        for control_id, condition in {
            "TOKEN_SLOT_PERMUTE": "SLOT_PERMUTED_TARGETS",
            "TOKEN_MATCHED_GAUSSIAN_NOISE": "MATCHED_GAUSSIAN_NOISE",
        }.items():
            records[control_id].append(_comparison(
                condition_id=condition, source=source,
                coordinates={"pipeline_seed": seed},
            ))
        for expert in EXPERT_ORDER:
            for control_id, prefix in {
                "TOKEN_WITHIN_CLASS_MEAN_BANK": "WITHIN_CLASS_MEAN_TARGETS__",
                "TOKEN_WRONG_EVENT_SAME_CLASS_BANK": "WITHIN_CLASS_WRONG_EVENT_BANK__",
                "TOKEN_WRONG_EVENT_UNMATCHED_BANK": "WRONG_EVENT_BANK__",
                "TOKEN_ZERO_BANK": "ZERO_ORACLE_BANK__",
            }.items():
                records[control_id].append(_comparison(
                    condition_id=f"{prefix}{expert}", source=source,
                    coordinates={"pipeline_seed": seed},
                ))
        diagnostic = {
            expert: {"mean_event_l2_norm": 1.0, "covariance_trace": 2.0}
            for expert in EXPERT_ORDER
        }
        records["TOKEN_NORM_COVARIANCE"].append({
            "record_type": "bank_diagnostic",
            "pipeline_seed": seed,
            "predicted_bank_diagnostics": diagnostic,
            "oracle_bank_diagnostics": diagnostic,
            "source_artifact_sha256": source,
        })

    bias_hash = _digest("bias-scale")
    for expert in EXPERT_ORDER:
        records["RELATION_FIXED_VS_LEARNED_SCALE"].append(_comparison(
            condition_id="B_DUAL_GATED_VS_B_DUAL_FIXED", source=bias_hash,
            coordinates={"expert_id": expert, "fixed_run_id": f"fixed:{expert}",
                         "learned_run_id": f"learned:{expert}"},
        ))

    index_hash, config_hash = _digest("index"), _digest("configuration")
    direct = [_predictor_record("direct", seed, "A3_SLOT_DECODER_DIRECT", "W_CANONICAL",
                                index_hash=index_hash, config_hash=config_hash)
              for seed in PIPELINE_SEEDS]
    gated = [_predictor_record("gated", seed, "A4_SLOT_DECODER_GATED", "W_CANONICAL",
                               index_hash=index_hash, config_hash=config_hash)
             for seed in PIPELINE_SEEDS]
    logit = [_predictor_record("logit", seed, "A0_AFFINE", "W_LOGIT_ONLY",
                               index_hash=index_hash, config_hash=config_hash)
             for seed in PIPELINE_SEEDS]
    non_logit = [*direct, *gated]
    architecture = {
        "bundle_input_index_sha256": index_hash,
        "selector_configuration_sha256": config_hash,
        "candidate_manifest_hashes": [_digest(f"candidate:{name}") for name in ("direct", "gated", "logit")],
        "coordinate_manifest_hashes": [_digest("coordinate")],
        "direct_evaluations": direct,
        "gated_evaluations": gated,
        "W_LOGIT_ONLY_evaluations": logit,
        "non_logit_evaluations": non_logit,
        "direct_evaluations_sha256": canonical_sha256(direct),
        "gated_evaluations_sha256": canonical_sha256(gated),
        "W_LOGIT_ONLY_evaluations_sha256": canonical_sha256(logit),
        "non_logit_evaluations_sha256": canonical_sha256(non_logit),
        "all_evidence_is_materialized_val_design_inference": True,
    }
    for control_id, reference_rows, control_rows, condition in (
        ("PREDICTOR_DIRECT_VS_GATED", direct, gated, "GATED_MEAN_VS_DIRECT_MEAN"),
        ("PREDICTOR_W_LOGIT_ONLY", non_logit, logit, "W_LOGIT_ONLY_MEAN_VS_NON_LOGIT_MEAN"),
    ):
        reference, control = _mean(reference_rows), _mean(control_rows)
        if control_id == "PREDICTOR_DIRECT_VS_GATED":
            records[control_id].extend([*reference_rows, *control_rows])
        else:
            records[control_id].extend(control_rows)
        records[control_id].append({
            "record_type": "aggregate_metric_comparison",
            "condition_id": condition,
            "metrics": control,
            "reference_metrics": reference,
            "metric_deltas": _deltas(control, reference),
            "aggregation": "arithmetic_mean_over_predeclared_val_design_rows",
        })

    reconstruction_hashes = []
    reconstruction = []
    for role in STAGE_E_SHAPES:
        for seed in PIPELINE_SEEDS:
            condition_metrics = {
                "frozen_reconstruction": _metrics(),
                "token_refiner": _metrics(0.01),
                "unrestricted_fusion": _metrics(0.02),
            }
            hashes = {name: _digest(f"reconstruction:{role}:{seed}:{name}")
                      for name in condition_metrics}
            reconstruction_hashes.append(canonical_sha256(hashes))
            reconstruction.append({
                "shape_role": role,
                "pipeline_seed": seed,
                "condition_metrics": condition_metrics,
                "condition_metric_artifact_hashes": hashes,
                "condition_metric_deltas_vs_frozen_reconstruction": {
                    name: _deltas(values, condition_metrics["frozen_reconstruction"])
                    for name, values in condition_metrics.items()
                    if name != "frozen_reconstruction"
                },
            })
    evidence = {
        "relation_and_predictor_wave": canonical_sha256(relation_hashes),
        "token_substitutions": canonical_sha256(token_hashes),
        "native_bypass": canonical_sha256(bypass_hashes),
        "reconstruction_comparison": canonical_sha256(reconstruction_hashes),
        "fixed_vs_learned_bias_scale": bias_hash,
        "predictor_direct_gated_logit_only": canonical_sha256(architecture),
    }
    def family(control_id: str) -> str:
        if control_id.startswith(("RELATION_ZERO", "RELATION_WITHIN", "RELATION_WRONG", "RELATION_DIRECTIONAL", "PREDICTOR_ZERO", "PREDICTOR_SHUFFLE", "PREDICTOR_REMOVE")):
            return "relation_and_predictor_wave"
        if control_id == "RELATION_FIXED_VS_LEARNED_SCALE":
            return "fixed_vs_learned_bias_scale"
        if control_id in {"PREDICTOR_DIRECT_VS_GATED", "PREDICTOR_W_LOGIT_ONLY"}:
            return "predictor_direct_gated_logit_only"
        if control_id.startswith("BYPASS_"):
            return "native_bypass"
        return "token_substitutions"
    return with_content_hash({
        "contract": "retb_stage_k_semantic_controls_bundle_v5",
        "schema_version": 5,
        "control_ids": list(SEMANTIC_CONTROL_KINDS),
        "semantic_control_policy": dict(SEMANTIC_CONTROL_POLICY),
        "rows": [{
            "control_id": control_id,
            "evidence_family": family(control_id),
            "artifact_sha256": evidence[family(control_id)],
            "metric_records": records[control_id],
            "metric_record_count": len(records[control_id]),
        } for control_id in SEMANTIC_CONTROL_KINDS],
        "evidence_families": evidence,
        "predictor_architecture_evidence": architecture,
        "reconstruction_metric_records": reconstruction,
        "complete_coverage": True,
        "all_section_28_controls_have_real_evaluation_evidence": True,
        "scientific_underperformance_blocks_continuation": False,
        "stack_val_consumed": False,
        "final_test_consumed": False,
        "source": dict(source_snapshot),
    })
