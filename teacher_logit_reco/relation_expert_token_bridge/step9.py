"""Step-9 predictor campaign registries, contracts, and publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .predictor_losses import (
    build_predictor_objective_contract,
    validate_predictor_objective_contract,
)
from .predictors import (
    build_predictor_architecture_contract,
    validate_predictor_architecture_contract,
)
from .registry import EXPERT_ORDER


STAGE_F_REGISTRY_CONTRACT = "retb_stage_f_predictor_screen_registry_v1"
STAGE_G_REGISTRY_CONTRACT = "retb_stage_g_predictor_loss_screen_registry_v1"
STAGE_H_POLICY_CONTRACT = "retb_stage_h_predictor_confirmation_policy_v1"
STEP9_BUNDLE_CONTRACT = "retb_step9_predictor_bundle_v1"
STEP9_REPORT_CONTRACT = "retb_step9_report_v1"
PREDICTOR_RUN_CONTRACT = "retb_materialized_predictor_run_v1"

REPRESENTATIVE_EXPERTS = ("BASE4", "PT", "TRACK", "REGION")
SCREEN_SHAPES = ("SHAPE_COMPACT", "SHAPE_HIGH")
ARCHITECTURE_CONTEXT_ROWS = (
    ("A0_AFFINE", "C0_SELF"),
    ("A1_RESMLP", "C0_SELF"),
    ("A2_TOKEN_ENCODER", "C0_SELF"),
    ("A3_SLOT_DECODER_DIRECT", "C0_SELF"),
    ("A3_SLOT_DECODER_DIRECT", "C1_NATIVE"),
    ("A3_SLOT_DECODER_DIRECT", "C2_ALL"),
    ("A4_SLOT_DECODER_GATED", "C0_SELF"),
    ("A4_SLOT_DECODER_GATED", "C1_NATIVE"),
    ("A4_SLOT_DECODER_GATED", "C2_ALL"),
)
LOSS_IDS = (
    "W_TOKEN_ONLY",
    "W_TOKEN_HEAVY",
    "W_CANONICAL",
    "W_TASK_HEAVY",
    "W_LOGIT_ONLY",
    "W_GRADNORM",
)


def _run_id(*parts: Any) -> str:
    return "_".join(str(value) for value in parts)


def build_stage_f_registry() -> dict[str, Any]:
    rows = []
    for expert in REPRESENTATIVE_EXPERTS:
        for shape in SCREEN_SHAPES:
            for architecture, context in ARCHITECTURE_CONTEXT_ROWS:
                rows.append(
                    {
                        "run_id": _run_id(
                            "RETB",
                            "F",
                            expert,
                            shape,
                            architecture,
                            context,
                            "S101",
                        ),
                        "stage": "F",
                        "pipeline_seed": 101,
                        "expert_id": expert,
                        "shape_alias": shape,
                        "architecture": architecture,
                        "context": context,
                        "objective_id": "W_CANONICAL",
                        "uncertainty_head": "U_SLOT",
                        "normalization_mode": "N_UNCLIPPED",
                        "target_mode": "T0_PURE",
                        "hlt_evidence_mode": "selected_native_HLT_evidence",
                        "learning_rate": 5.0e-4,
                        "dropout": 0.1,
                        "role": (
                            "architecture_control"
                            if architecture in {"A0_AFFINE", "A1_RESMLP"}
                            else "scientific_candidate"
                        ),
                    }
                )
    return with_content_hash(
        {
            "contract": STAGE_F_REGISTRY_CONTRACT,
            "schema_version": 1,
            "representative_experts": list(REPRESENTATIVE_EXPERTS),
            "shape_aliases": list(SCREEN_SHAPES),
            "architecture_context_rows": [
                list(row) for row in ARCHITECTURE_CONTEXT_ROWS
            ],
            "membership_count": len(rows),
            "rows": rows,
            "optimizer_followup": {
                "experts": ["PT", "TRACK"],
                "families": ["selected_direct", "selected_gated"],
                "learning_rates": [2.0e-4, 5.0e-4, 1.0e-3],
                "dropouts": [0.0, 0.1],
                "simplicity_control": {
                    "learning_rate": 5.0e-4,
                    "dropout": 0.0,
                },
            },
            "selection": [
                "maximum_frozen_hybrid_val_design_accuracy",
                "global_window_0.0001",
                "lower_normalized_token_error",
                "lower_parameter_count",
                "lexicographic_run_id",
            ],
            "all_negative_campaign_still_selects": True,
            "performance_based_termination": False,
        }
    )


def build_stage_g_registry() -> dict[str, Any]:
    rows = []
    for family in ("SELECTED_DIRECT", "SELECTED_GATED"):
        for expert in ("PT", "TRACK"):
            for shape in SCREEN_SHAPES:
                for objective in LOSS_IDS:
                    for uncertainty in (
                        "U_SLOT",
                        "U_GROUP4",
                        "U_DIAGONAL",
                    ):
                        for normalization in (
                            "N_UNCLIPPED",
                            "N_CLIP16",
                            "N_CLIP8",
                        ):
                            rows.append(
                                {
                                    "template_id": _run_id(
                                        "RETB",
                                        "G",
                                        family,
                                        expert,
                                        shape,
                                        objective,
                                        uncertainty,
                                        normalization,
                                        "S101",
                                    ),
                                    "stage": "G",
                                    "pipeline_seed": 101,
                                    "architecture_family": family,
                                    "expert_id": expert,
                                    "shape_alias": shape,
                                    "objective_id": objective,
                                    "uncertainty_head": uncertainty,
                                    "normalization_mode": normalization,
                                    "role": (
                                        "semantic_control"
                                        if objective == "W_LOGIT_ONLY"
                                        else "scientific_candidate"
                                    ),
                                    "GradNorm_adaptation_split": (
                                        "model_train"
                                        if objective == "W_GRADNORM"
                                        else None
                                    ),
                                }
                            )
    return with_content_hash(
        {
            "contract": STAGE_G_REGISTRY_CONTRACT,
            "schema_version": 1,
            "architecture_families": [
                "SELECTED_DIRECT",
                "SELECTED_GATED",
            ],
            "experts": ["PT", "TRACK"],
            "shape_aliases": list(SCREEN_SHAPES),
            "objective_ids": list(LOSS_IDS),
            "uncertainty_heads": ["U_SLOT", "U_GROUP4", "U_DIAGONAL"],
            "normalization_modes": [
                "N_UNCLIPPED",
                "N_CLIP16",
                "N_CLIP8",
            ],
            "membership_count": len(rows),
            "templates": rows,
            "validation_metrics_may_adapt_GradNorm": False,
            "hard_clipping_is_primary": False,
            "W_LOGIT_ONLY_faithful_token_claim_eligible": False,
            "performance_based_termination": False,
        }
    )


def build_stage_h_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": STAGE_H_POLICY_CONTRACT,
            "schema_version": 1,
            "pipeline_seeds": [101, 202, 303],
            "expert_order": list(EXPERT_ORDER),
            "shapes": [
                "SHAPE_COMPACT",
                "SHAPE_HIGH",
                "HET_PHYSICS_per_expert",
                "HET_SELECTED_per_expert",
                "HET_BEAM_per_expert",
            ],
            "families": ["selected_direct", "selected_gated"],
            "confirmation_controls": {
                "C3_ALL_PARTICLE": ["PT", "TRACK", "REGION"],
                "matched_widened_residual_MLP": True,
                "zero_evidence_decoder": True,
                "target_modes": [
                    "T0_PURE",
                    "eligible_T1_ANCHORED_BRIDGE",
                    "eligible_T1_TASK_BRIDGE",
                    "eligible_T2_PROJECT",
                ],
                "HLT_evidence_modes": [
                    "selected_HE_SCRATCH_CE",
                    "selected_HE_OFFLINE_INIT",
                    "selected_HE_DUAL_OBJECTIVE",
                ],
            },
            "capacity_report_required": [
                "parameter_count",
                "analytical_FLOPs",
                "measured_latency",
                "peak_memory",
            ],
            "val_design_outputs": [
                "canonical_identities",
                "predicted_tokens",
                "log_variance",
                "expert_logits",
            ],
            "uncertainty_calibration": {
                "fit_split": "val_design",
                "labels_consumed": False,
                "frozen_before_stack_val": True,
            },
            "configuration_shared_across_pipeline_seeds": True,
            "cross_seed_member_substitution_permitted": False,
            "performance_based_termination": False,
        }
    )


def materialize_predictor_run(
    *,
    run_id: str,
    stage: str,
    pipeline_seed: int,
    expert_id: str,
    shape_id: str,
    token_count: int,
    token_dimension: int,
    architecture: str,
    context: str,
    objective_id: str,
    uncertainty_head: str,
    normalization_mode: str,
    target_mode: str,
    hlt_evidence_mode: str,
    learning_rate: float,
    dropout: float,
    role: str,
    parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    from .predictor_losses import FIXED_WEIGHTS
    from .predictors import (
        ARCHITECTURES,
        CONTEXTS,
        NORMALIZATION_MODES,
        UNCERTAINTY_HEADS,
    )

    required = {
        "step9_bundle",
        "model_train_target_cache",
        "val_stop_target_cache",
        "val_design_target_cache",
        "target_normalizer",
        "slot_queries",
        "offline_target_checkpoint",
        "offline_fusion",
        "native_hlt_expert",
        "model_train_hlt_evidence_cache",
        "val_stop_hlt_evidence_cache",
        "val_design_hlt_evidence_cache",
        "model_train_identity_manifest",
        "val_stop_identity_manifest",
        "val_design_identity_manifest",
    }
    if (
        stage not in {"F", "G", "H"}
        or int(pipeline_seed) not in {101, 202, 303}
        or expert_id not in EXPERT_ORDER
        or int(token_count) not in {1, 2, 4, 8, 16}
        or int(token_dimension) not in {64, 128}
        or architecture not in ARCHITECTURES
        or context not in CONTEXTS
        or objective_id not in {*FIXED_WEIGHTS, "W_GRADNORM"}
        or uncertainty_head not in UNCERTAINTY_HEADS
        or normalization_mode not in NORMALIZATION_MODES
        or target_mode
        not in {
            "T0_PURE",
            "T1_ANCHORED_BRIDGE",
            "T1_TASK_BRIDGE",
            "T2_PROJECT",
        }
        or float(learning_rate) not in {2.0e-4, 5.0e-4, 1.0e-3}
        or float(dropout) not in {0.0, 0.1}
        or role
        not in {
            "architecture_control",
            "scientific_candidate",
            "semantic_control",
            "capacity_control",
        }
        or set(parent_hashes) != required
    ):
        raise ValueError("materialized predictor run semantics differ")
    if architecture in {"A0_AFFINE", "A1_RESMLP", "A2_TOKEN_ENCODER"} and (
        context != "C0_SELF"
    ):
        raise ValueError("A0-A2 predictor run requires C0_SELF")
    if (
        (stage in {"F", "G"} and int(pipeline_seed) != 101)
        or (stage != "H" and context == "C3_ALL_PARTICLE")
        or (
            stage == "F"
            and (
                expert_id not in REPRESENTATIVE_EXPERTS
                or (architecture, context) not in ARCHITECTURE_CONTEXT_ROWS
                or objective_id != "W_CANONICAL"
                or uncertainty_head != "U_SLOT"
                or normalization_mode != "N_UNCLIPPED"
                or target_mode != "T0_PURE"
                or hlt_evidence_mode != "selected_native_HLT_evidence"
                or role
                != (
                    "architecture_control"
                    if architecture in {"A0_AFFINE", "A1_RESMLP"}
                    else "scientific_candidate"
                )
            )
        )
        or (
            stage == "G"
            and (
                expert_id not in {"PT", "TRACK"}
                or objective_id not in LOSS_IDS
                or architecture
                not in {
                    "A3_SLOT_DECODER_DIRECT",
                    "A4_SLOT_DECODER_GATED",
                }
                or context not in {"C0_SELF", "C1_NATIVE", "C2_ALL"}
                or target_mode != "T0_PURE"
                or hlt_evidence_mode != "selected_native_HLT_evidence"
                or role
                != (
                    "semantic_control"
                    if objective_id == "W_LOGIT_ONLY"
                    else "scientific_candidate"
                )
            )
        )
        or (
            stage == "H"
            and (
                hlt_evidence_mode
                not in {
                    "HE_SCRATCH_CE",
                    "HE_OFFLINE_INIT",
                    "HE_DUAL_OBJECTIVE",
                }
                or (
                    context == "C3_ALL_PARTICLE"
                    and expert_id not in {"PT", "TRACK", "REGION"}
                )
            )
        )
        or (objective_id == "W_LOGIT_ONLY" and role != "semantic_control")
    ):
        raise ValueError("materialized predictor stage restrictions differ")
    return with_content_hash(
        {
            "contract": PREDICTOR_RUN_CONTRACT,
            "schema_version": 1,
            "run_id": str(run_id),
            "stage": stage,
            "pipeline_seed": int(pipeline_seed),
            "expert_id": expert_id,
            "shape_id": str(shape_id),
            "token_count": int(token_count),
            "token_dimension": int(token_dimension),
            "architecture": architecture,
            "context": context,
            "objective_id": objective_id,
            "uncertainty_head": uncertainty_head,
            "normalization_mode": normalization_mode,
            "target_mode": target_mode,
            "hlt_evidence_mode": str(hlt_evidence_mode),
            "learning_rate": float(learning_rate),
            "dropout": float(dropout),
            "role": role,
            "parent_hashes": {
                name: require_sha256(
                    parent_hashes[name], name=f"parent_hashes.{name}"
                )
                for name in sorted(parent_hashes)
            },
            "fixed_budget": True,
            "performance_based_termination": False,
        }
    )


def validate_materialized_predictor_run(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=PREDICTOR_RUN_CONTRACT
    )
    expected = materialize_predictor_run(
        run_id=payload.get("run_id"),
        stage=payload.get("stage"),
        pipeline_seed=int(payload.get("pipeline_seed", -1)),
        expert_id=payload.get("expert_id"),
        shape_id=payload.get("shape_id"),
        token_count=int(payload.get("token_count", -1)),
        token_dimension=int(payload.get("token_dimension", -1)),
        architecture=payload.get("architecture"),
        context=payload.get("context"),
        objective_id=payload.get("objective_id"),
        uncertainty_head=payload.get("uncertainty_head"),
        normalization_mode=payload.get("normalization_mode"),
        target_mode=payload.get("target_mode"),
        hlt_evidence_mode=payload.get("hlt_evidence_mode"),
        learning_rate=float(payload.get("learning_rate", -1.0)),
        dropout=float(payload.get("dropout", -1.0)),
        role=payload.get("role"),
        parent_hashes=payload.get("parent_hashes", {}),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("materialized predictor run semantics differ")
    return digest


def validate_stage_f_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_F_REGISTRY_CONTRACT
    )
    expected = build_stage_f_registry()
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-F predictor registry semantics differ")
    return digest


def validate_stage_g_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_G_REGISTRY_CONTRACT
    )
    expected = build_stage_g_registry()
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-G predictor registry semantics differ")
    return digest


def validate_stage_h_policy(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_H_POLICY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_h_policy()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-H predictor policy semantics differ")
    return digest


def build_step9_bundle(
    *,
    campaign_spec_sha256: str,
    step8_bundle_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    architecture = bind_source(
        build_predictor_architecture_contract(),
        source_snapshot=source_snapshot,
    )
    objective = bind_source(
        build_predictor_objective_contract(), source_snapshot=source_snapshot
    )
    stage_f = bind_source(
        build_stage_f_registry(), source_snapshot=source_snapshot
    )
    stage_g = bind_source(
        build_stage_g_registry(), source_snapshot=source_snapshot
    )
    stage_h = bind_source(
        build_stage_h_policy(), source_snapshot=source_snapshot
    )
    parents = {
        "campaign_spec": require_sha256(
            campaign_spec_sha256, name="campaign_spec_sha256"
        ),
        "step8_bundle": require_sha256(
            step8_bundle_sha256, name="step8_bundle_sha256"
        ),
        "global_determinism": require_sha256(
            global_determinism_sha256, name="global_determinism_sha256"
        ),
    }
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP9_BUNDLE_CONTRACT,
                "schema_version": 1,
                "parents": parents,
                "artifact_hashes": {
                    "predictor_architectures": architecture["content_hash"],
                    "predictor_objectives": objective["content_hash"],
                    "stage_f_registry": stage_f["content_hash"],
                    "stage_g_registry": stage_g["content_hash"],
                    "stage_h_policy": stage_h["content_hash"],
                },
                "offline_teachers_frozen": True,
                "predictor_forward_is_HLT_only": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": STEP9_REPORT_CONTRACT,
                "schema_version": 1,
                "step9_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "A0_through_A4": True,
                    "C0_through_C3": True,
                    "three_uncertainty_heads": True,
                    "label_free_val_design_calibration": True,
                    "three_normalization_modes": True,
                    "fixed_and_GradNorm_objectives": True,
                    "direct_and_gated_outputs": True,
                    "capacity_controls": True,
                    "fixed_budget_resume": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {
        "predictor_architectures": architecture,
        "predictor_objectives": objective,
        "stage_f_registry": stage_f,
        "stage_g_registry": stage_g,
        "stage_h_policy": stage_h,
        "step9_bundle": manifest,
        "step9_report": report,
    }


def validate_step9_bundle(bundle: Mapping[str, Any]) -> str:
    names = {
        "predictor_architectures",
        "predictor_objectives",
        "stage_f_registry",
        "stage_g_registry",
        "stage_h_policy",
        "step9_bundle",
        "step9_report",
    }
    if set(bundle) != names:
        raise ValueError("Step-9 bundle members differ")
    hashes = {
        "predictor_architectures": validate_predictor_architecture_contract(
            bundle["predictor_architectures"]
        ),
        "predictor_objectives": validate_predictor_objective_contract(
            bundle["predictor_objectives"]
        ),
        "stage_f_registry": validate_stage_f_registry(
            bundle["stage_f_registry"]
        ),
        "stage_g_registry": validate_stage_g_registry(
            bundle["stage_g_registry"]
        ),
        "stage_h_policy": validate_stage_h_policy(bundle["stage_h_policy"]),
    }
    digest = validate_content_hash(
        bundle["step9_bundle"], expected_contract=STEP9_BUNDLE_CONTRACT
    )
    if bundle["step9_bundle"]["artifact_hashes"] != hashes:
        raise ValueError("Step-9 artifact hashes differ")
    if set(bundle["step9_bundle"]["parents"]) != {
        "campaign_spec",
        "step8_bundle",
        "global_determinism",
    }:
        raise ValueError("Step-9 parent coverage differs")
    for name, value in bundle["step9_bundle"]["parents"].items():
        require_sha256(value, name=f"step9_bundle.parents.{name}")
    validate_content_hash(
        bundle["step9_report"], expected_contract=STEP9_REPORT_CONTRACT
    )
    if bundle["step9_report"]["step9_bundle_sha256"] != digest:
        raise ValueError("Step-9 report parent differs")
    if (
        not bundle["step9_bundle"]["offline_teachers_frozen"]
        or not bundle["step9_bundle"]["predictor_forward_is_HLT_only"]
        or bundle["step9_bundle"]["performance_based_termination"]
        or bundle["step9_report"]["checks"]
        != {
            "A0_through_A4": True,
            "C0_through_C3": True,
            "three_uncertainty_heads": True,
            "label_free_val_design_calibration": True,
            "three_normalization_modes": True,
            "fixed_and_GradNorm_objectives": True,
            "direct_and_gated_outputs": True,
            "capacity_controls": True,
            "fixed_budget_resume": True,
        }
        or bundle["step9_report"]["scientific_results_inspected"]
    ):
        raise ValueError("Step-9 bundle/report safety semantics differ")
    if len({repr(row.get("source")) for row in bundle.values()}) != 1:
        raise ValueError("Step-9 source lineage differs")
    return digest


def publish_step9_bundle(
    *, campaign_root: str | Path, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    digest = validate_step9_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "predictor_architectures": (
            root / "registry" / "retb_predictor_architectures.json"
        ),
        "predictor_objectives": (
            root / "registry" / "retb_predictor_objectives.json"
        ),
        "stage_f_registry": (
            root / "registry" / "retb_stage_f_predictor_screen.json"
        ),
        "stage_g_registry": (
            root / "registry" / "retb_stage_g_predictor_loss_screen.json"
        ),
        "stage_h_policy": (
            root / "registry" / "retb_stage_h_predictor_confirmation.json"
        ),
        "step9_bundle": (
            root / "registry" / "retb_step9_predictor_bundle.json"
        ),
        "step9_report": root / "reports" / "retb_step9_report.json",
    }
    return {
        "step9_bundle_sha256": digest,
        "publications": {
            name: write_immutable_json(paths[name], bundle[name])
            for name in paths
        },
    }


__all__ = [
    "ARCHITECTURE_CONTEXT_ROWS",
    "LOSS_IDS",
    "PREDICTOR_RUN_CONTRACT",
    "REPRESENTATIVE_EXPERTS",
    "SCREEN_SHAPES",
    "STAGE_F_REGISTRY_CONTRACT",
    "STAGE_G_REGISTRY_CONTRACT",
    "STAGE_H_POLICY_CONTRACT",
    "STEP9_BUNDLE_CONTRACT",
    "STEP9_REPORT_CONTRACT",
    "build_stage_f_registry",
    "build_stage_g_registry",
    "build_stage_h_policy",
    "build_step9_bundle",
    "materialize_predictor_run",
    "publish_step9_bundle",
    "validate_stage_f_registry",
    "validate_stage_g_registry",
    "validate_stage_h_policy",
    "validate_materialized_predictor_run",
    "validate_step9_bundle",
]
