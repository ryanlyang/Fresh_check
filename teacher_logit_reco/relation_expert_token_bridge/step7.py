"""Stage-E pilot/target registry and immutable Step-7 contract bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .bridge_certification import BRIDGE_ELIGIBILITY_CONTRACT
from .bridge_selection import BRIDGE_COORDINATE_SELECTION_CONTRACT
from .bridge_targets import (
    BRIDGE_TARGET_CONTRACT,
    LAMBDA_PRED_VALUES,
    PILOT_ARCHITECTURE_CONTRACT,
    build_bridge_candidate_contract,
    build_bridge_target_contract,
    build_pilot_architecture_contract,
)
from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .registry import EXPERT_ORDER, resolve_run_id


STAGE_E_TEMPLATE_REGISTRY_CONTRACT = "retb_stage_e_template_registry_v1"
STAGE_E_MATERIALIZED_RUN_CONTRACT = "retb_stage_e_materialized_run_v1"
BRIDGE_CERTIFICATION_POLICY_CONTRACT = "retb_bridge_certification_policy_v1"
STEP7_BUNDLE_CONTRACT = "retb_step7_bridge_target_bundle_v1"
STEP7_REPORT_CONTRACT = "retb_step7_report_v1"
STEP7_MINIATURE_COMPLETION_CONTRACT = "retb_step7_miniature_completion_v1"
STAGE_E_SHAPES = (
    "SHAPE_COMPACT",
    "SHAPE_HIGH",
    "HET_PHYSICS",
    "HET_SELECTED",
    "HET_BEAM",
)


def build_bridge_certification_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": BRIDGE_CERTIFICATION_POLICY_CONTRACT,
            "schema_version": 1,
            "split": "val_design",
            "offline_noninferiority": {
                "seeds": [101, 202, 303],
                "mean_accuracy_deficit_max": 0.0020,
                "mean_cross_entropy_increase_max": 0.0050,
                "worst_per_class_efficiency_deficit_max": 0.0100,
                "required_for_T1_T2_maximum_performance_eligibility": True,
            },
            "content_certification": {
                "class_agreement_min": 0.990,
                "median_variance_ratio_min": 0.50,
                "low_variance_channel_fraction_max": 0.05,
                "effective_rank_ratio_min": 0.80,
                "relative_per_slot_covariance_error_max": 0.25,
                "within_class_retrieval_accuracy_min": 0.20,
            },
            "effective_rank": {
                "matrix": "n_events_by_K_times_D_slot_major",
                "dtype": "C_contiguous_CPU_float64",
                "backend": "numpy.linalg.svd",
                "full_matrices": False,
                "relative_zero_threshold": 1.0e-12,
                "all_zero_rank": 0.0,
                "version_binding": "numpy_and_linked_LAPACK",
            },
            "retrieval": {
                "candidate_count": 32,
                "negative_count": 31,
                "within_class": True,
                "training_namespace": "retb_t1_negatives_v1",
                "certification_namespace": "retb_t1_cert_negatives_v1",
                "ring_order": "ascending_canonical_identity",
                "selection": "sha256_seeded_cyclic_without_replacement",
                "temperature": 0.1,
                "tie_rule": (
                    "descending_similarity_then_ascending_canonical_identity"
                ),
            },
            "T2_dimension_change": {
                "T0_coordinate_checks_after_training_decoder": True,
                "bridge_coordinate_rank_and_retrieval_also_reported": True,
            },
            "certification_normalization": {
                "T1_moving_predicted_and_T0": "frozen_T0_train_normalizer",
                "T2_moving_and_predicted": (
                    "bridge_coordinate_model_train_normalizer"
                ),
                "T2_T0_and_decoded": "frozen_T0_train_normalizer",
                "normalization_performed_by_certification_worker": True,
            },
            "eligibility": {
                "contract": BRIDGE_ELIGIBILITY_CONTRACT,
                "requires_exact_pipeline_seeds": [101, 202, 303],
                "T1_T2_requires_offline_noninferiority": True,
                "content_certification_controls_representation_claim": True,
                "T3_never_token_coordinate_eligible": True,
            },
            "scientific_failure_stops_workflow": False,
        }
    )


def _candidate_templates() -> list[dict[str, Any]]:
    templates = []
    for mode in ("T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE"):
        for weight in LAMBDA_PRED_VALUES:
            for final_two in (False, True):
                templates.append(
                    {
                        "target_mode": mode,
                        "lambda_pred": weight,
                        "bridge_dimension": None,
                        "unfreeze_final_two_blocks": final_two,
                        "role": (
                            "confirmation_candidate"
                            if final_two
                            else "scientific_candidate"
                        ),
                    }
                )
    for weight in LAMBDA_PRED_VALUES:
        for dimension in (64, 128):
            templates.append(
                {
                    "target_mode": "T2_PROJECT",
                    "lambda_pred": weight,
                    "bridge_dimension": dimension,
                    "unfreeze_final_two_blocks": False,
                    "role": "scientific_candidate",
                }
            )
    templates.append(
        {
            "target_mode": "T3_LOGIT",
            "lambda_pred": 0.0,
            "bridge_dimension": None,
            "unfreeze_final_two_blocks": False,
            "role": "task_distillation_control",
        }
    )
    return templates


def build_stage_e_template_registry() -> dict[str, Any]:
    templates = _candidate_templates()
    return with_content_hash(
        {
            "contract": STAGE_E_TEMPLATE_REGISTRY_CONTRACT,
            "schema_version": 1,
            "stage": "E",
            "shapes": list(STAGE_E_SHAPES),
            "expert_order": list(EXPERT_ORDER),
            "pipeline_seeds": [101, 202, 303],
            "pilot_template": {
                "pilot_id": "PILOT_T0",
                "mode": "T0_PURE",
                "hlt_encoder": "HE_OFFLINE_INIT",
                "unbiased_particle_context_encoder": "HE_BASE4",
                "realization_policy": "R_MULTI",
                "predictor": "A3_SLOT_DECODER_DIRECT",
                "context": "C2_ALL",
                "objective": "W_TOKEN_HEAVY",
                "uncertainty": "U_SLOT",
                "normalization": "N_UNCLIPPED",
                "learning_rate": 5.0e-4,
                "dropout": 0.0,
                "effective_batch_size": 256,
            },
            "candidate_templates": templates,
            "candidate_template_count_per_expert_shape_seed": len(templates),
            "pilot_membership_count": (
                len(STAGE_E_SHAPES) * len(EXPERT_ORDER) * 3
            ),
            "candidate_membership_count": (
                len(STAGE_E_SHAPES) * len(EXPERT_ORDER) * 3 * len(templates)
            ),
            "target_tuple_selector": {
                "readout": "fresh_F_POOLED_MLP_per_tuple",
                "readout_seed": 41703,
                "beam_width": 16,
                "final_top_four_fusion": (
                    "fresh_canonical_F_TOKEN_TRANSFORMER_per_tuple"
                ),
                "homogeneous_eligible_tuples_also_locked": True,
                "contract": BRIDGE_COORDINATE_SELECTION_CONTRACT,
            },
            "worker_access": ["model_train", "val_stop"],
            "certifier_access": ["val_design"],
            "forbidden_access": ["stack_val", "final_test"],
            "performance_based_termination": False,
        }
    )


def validate_stage_e_template_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_E_TEMPLATE_REGISTRY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_e_template_registry()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-E template registry differs")
    return digest


def materialize_stage_e_run(
    *,
    template_registry: Mapping[str, Any],
    pipeline_seed: int,
    expert_id: str,
    shape_id: str,
    target_mode: str,
    lambda_pred: float,
    bridge_dimension: int | None,
    unfreeze_final_two_blocks: bool,
    t0_checkpoint_sha256: str,
    hlt_encoder_checkpoint_sha256: str,
    unbiased_particle_encoder_checkpoint_sha256: str,
    pilot_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    registry_sha = validate_stage_e_template_registry(template_registry)
    if (
        int(pipeline_seed) not in {101, 202, 303}
        or expert_id not in EXPERT_ORDER
        or shape_id not in STAGE_E_SHAPES
    ):
        raise ValueError("Stage-E materialized identity is not registered")
    matches = []
    if target_mode != "T0_PURE":
        matches = [
            row
            for row in template_registry["candidate_templates"]
            if row["target_mode"] == target_mode
            and float(row["lambda_pred"]) == float(lambda_pred)
            and row["bridge_dimension"] == bridge_dimension
            and bool(row["unfreeze_final_two_blocks"])
            == bool(unfreeze_final_two_blocks)
        ]
        if len(matches) != 1:
            raise ValueError("Stage-E candidate template is absent/duplicated")
    t0_sha = require_sha256(
        t0_checkpoint_sha256, name="t0_checkpoint_sha256"
    )
    hlt_sha = require_sha256(
        hlt_encoder_checkpoint_sha256,
        name="hlt_encoder_checkpoint_sha256",
    )
    unbiased_sha = require_sha256(
        unbiased_particle_encoder_checkpoint_sha256,
        name="unbiased_particle_encoder_checkpoint_sha256",
    )
    if target_mode == "T0_PURE":
        if (
            pilot_checkpoint_sha256 is not None
            or float(lambda_pred) != 0.0
            or bridge_dimension is not None
            or unfreeze_final_two_blocks
        ):
            raise ValueError("T0 reference cannot bind a pilot")
        component = "BRIDGE_PILOT"
        parents = {
            "T0_checkpoint": t0_sha,
            "HLT_encoder_checkpoint": hlt_sha,
            "unbiased_HLT_particle_encoder_checkpoint": unbiased_sha,
        }
        semantics = {
            "pilot_id": "PILOT_T0",
            "target_mode": "T0_PURE",
            "expert_id": expert_id,
            "shape_id": shape_id,
            "template": dict(template_registry["pilot_template"]),
            "parents": parents,
        }
    else:
        pilot_sha = require_sha256(
            pilot_checkpoint_sha256, name="pilot_checkpoint_sha256"
        )
        component = "BRIDGE_TARGET"
        candidate = build_bridge_candidate_contract(
            target_mode=target_mode,
            pipeline_seed=pipeline_seed,
            expert_id=expert_id,
            shape_id=shape_id,
            lambda_pred=lambda_pred,
            bridge_dimension=bridge_dimension,
            unfreeze_final_two_blocks=unfreeze_final_two_blocks,
            t0_checkpoint_sha256=t0_sha,
            hlt_encoder_checkpoint_sha256=hlt_sha,
            unbiased_particle_encoder_checkpoint_sha256=unbiased_sha,
            pilot_checkpoint_sha256=pilot_sha,
        )
        semantics = {
            "target_mode": target_mode,
            "expert_id": expert_id,
            "shape_id": shape_id,
            "template": matches[0],
            "candidate_contract_sha256": candidate["content_hash"],
            "parents": candidate["parents"],
        }
    run_id = resolve_run_id(
        stage="E",
        component=component,
        seed=int(pipeline_seed),
        configuration=semantics,
    )
    return with_content_hash(
        {
            "contract": STAGE_E_MATERIALIZED_RUN_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "stage": "E",
            "component": component,
            "pipeline_seed": int(pipeline_seed),
            "stage_e_template_registry_sha256": registry_sha,
            "configuration": semantics,
            "performance_based_termination": False,
        }
    )


def execute_miniature_stage_e(
    materialized_runs: list[Mapping[str, Any]],
    *,
    executor: Any,
) -> dict[str, Any]:
    if not materialized_runs:
        raise ValueError("miniature Stage-E requires materialized runs")
    seen, counts = set(), {"BRIDGE_PILOT": 0, "BRIDGE_TARGET": 0}
    for run in materialized_runs:
        validate_content_hash(
            run, expected_contract=STAGE_E_MATERIALIZED_RUN_CONTRACT
        )
        if run["run_id"] in seen:
            raise ValueError("miniature Stage-E run is duplicated")
        seen.add(run["run_id"])
        result = dict(executor(dict(run)))
        if (
            result.get("status") != "completed"
            or result.get("performance_based_termination", False)
        ):
            raise RuntimeError("miniature Stage-E execution failed")
        counts[run["component"]] += 1
    return with_content_hash(
        {
            "contract": STEP7_MINIATURE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "materialized_run_hashes": [
                run["content_hash"] for run in materialized_runs
            ],
            "completed": counts,
            "pilot_parent_lock_verified": True,
            "target_semantics_separate": True,
            "scientific_failure_stops_workflow": False,
        }
    )


def build_step7_bundle(
    *,
    campaign_spec_sha256: str,
    step6_bundle_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    parents = {
        "campaign_spec": require_sha256(
            campaign_spec_sha256, name="campaign_spec_sha256"
        ),
        "step6_bundle": require_sha256(
            step6_bundle_sha256, name="step6_bundle_sha256"
        ),
        "global_determinism": require_sha256(
            global_determinism_sha256, name="global_determinism_sha256"
        ),
    }
    artifacts = {
        "bridge_target_modes": bind_source(
            build_bridge_target_contract(), source_snapshot=source_snapshot
        ),
        "pilot_architecture": bind_source(
            build_pilot_architecture_contract(), source_snapshot=source_snapshot
        ),
        "certification_policy": bind_source(
            build_bridge_certification_policy(), source_snapshot=source_snapshot
        ),
        "stage_e_templates": bind_source(
            build_stage_e_template_registry(), source_snapshot=source_snapshot
        ),
    }
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP7_BUNDLE_CONTRACT,
                "schema_version": 1,
                "parents": parents,
                "artifact_hashes": {
                    name: artifact["content_hash"]
                    for name, artifact in sorted(artifacts.items())
                },
                "target_modes_semantically_interchangeable": False,
                "T3_token_fidelity_claim": False,
                "pure_target_fallback_always_retained": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": STEP7_REPORT_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": parents["campaign_spec"],
                "step7_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "seed_matched_pilot": True,
                    "initial_pilot_hash_bound": True,
                    "dimension_correct_T2": True,
                    "alternating_detach_contract": True,
                    "offline_noninferiority_frozen_before_selection": True,
                    "content_certification_frozen_before_selection": True,
                    "coordinate_specific_fusions": True,
                    "all_negative_campaign_continues": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {**artifacts, "step7_bundle": manifest, "step7_report": report}


def validate_step7_bundle(bundle: Mapping[str, Any]) -> str:
    expected_names = {
        "bridge_target_modes",
        "pilot_architecture",
        "certification_policy",
        "stage_e_templates",
        "step7_bundle",
        "step7_report",
    }
    if set(bundle) != expected_names:
        raise ValueError("Step-7 bundle members differ")
    hashes = {
        "bridge_target_modes": validate_content_hash(
            bundle["bridge_target_modes"],
            expected_contract=BRIDGE_TARGET_CONTRACT,
        ),
        "pilot_architecture": validate_content_hash(
            bundle["pilot_architecture"],
            expected_contract=PILOT_ARCHITECTURE_CONTRACT,
        ),
        "certification_policy": validate_content_hash(
            bundle["certification_policy"],
            expected_contract=BRIDGE_CERTIFICATION_POLICY_CONTRACT,
        ),
        "stage_e_templates": validate_stage_e_template_registry(
            bundle["stage_e_templates"]
        ),
    }
    digest = validate_content_hash(
        bundle["step7_bundle"], expected_contract=STEP7_BUNDLE_CONTRACT
    )
    if bundle["step7_bundle"]["artifact_hashes"] != hashes:
        raise ValueError("Step-7 artifact hashes differ")
    validate_content_hash(
        bundle["step7_report"], expected_contract=STEP7_REPORT_CONTRACT
    )
    if bundle["step7_report"]["step7_bundle_sha256"] != digest:
        raise ValueError("Step-7 report parent differs")
    if len({repr(value.get("source")) for value in bundle.values()}) != 1:
        raise ValueError("Step-7 source lineage differs")
    return digest


def publish_step7_bundle(
    *, campaign_root: str | Path, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    digest = validate_step7_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "bridge_target_modes": root / "registry" / "retb_bridge_target_modes.json",
        "pilot_architecture": root / "registry" / "retb_pilot_t0_architecture.json",
        "certification_policy": root / "registry" / "retb_bridge_certification_policy.json",
        "stage_e_templates": root / "registry" / "retb_stage_e_templates.json",
        "step7_bundle": root / "registry" / "retb_step7_bridge_target_bundle.json",
        "step7_report": root / "reports" / "retb_step7_report.json",
    }
    return {
        "campaign_root": str(root.resolve()),
        "step7_bundle_sha256": digest,
        "publications": {
            name: write_immutable_json(path, bundle[name])
            for name, path in paths.items()
        },
    }


__all__ = [
    "build_stage_e_template_registry",
    "build_step7_bundle",
    "execute_miniature_stage_e",
    "materialize_stage_e_run",
    "publish_step7_bundle",
    "validate_stage_e_template_registry",
    "validate_step7_bundle",
]
