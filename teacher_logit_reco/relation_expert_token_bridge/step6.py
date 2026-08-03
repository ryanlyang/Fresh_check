"""Stage-D native-HLT run registry and immutable Step-6 bundle."""

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
from .hlt_experts import (
    DUAL_WEIGHTS,
    HLT_EVIDENCE_MODE_CONTRACT,
    HLT_MODES,
    NativeHLTExpertTrainingConfig,
    build_hlt_evidence_mode_contract,
)
from .native_fusion import (
    NATIVE_FUSION_VARIANTS,
    build_native_fusion_contract,
)
from .registry import EXPERT_ORDER, resolve_run_id
from .replicas import REALIZATION_POLICIES


STAGE_D_RUN_REGISTRY_CONTRACT = "retb_stage_d_run_registry_v2"
HLT_MATCHED_CONTROL_CONTRACT = "retb_native_hlt_matched_controls_v1"
STEP6_BUNDLE_CONTRACT = "retb_step6_native_hlt_bundle_v2"
STEP6_REPORT_CONTRACT = "retb_step6_report_v2"
STEP6_MINIATURE_COMPLETION_CONTRACT = "retb_step6_miniature_completion_v1"
STAGE_D_CONFIRMATION_REGISTRY_CONTRACT = (
    "retb_stage_d_confirmation_registry_v1"
)
STAGE_D_SHAPES = (
    "S1_128",
    "SHAPE_COMPACT",
    "SHAPE_HIGH",
    "HET_PHYSICS",
    "HET_SELECTED",
    "HET_BEAM",
)
SELECTED_UNIFORM_SHAPES = ("SHAPE_COMPACT", "SHAPE_HIGH")


def build_hlt_matched_control_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": HLT_MATCHED_CONTROL_CONTRACT,
            "schema_version": 1,
            "controls": {
                "H_BASE": {
                    "architecture": "exact_standard_base_size_ParticleTransformer",
                    "relation_input": "standard_four",
                    "measurement_embedding": False,
                    "initialization": "scratch",
                },
                "H_WIDE": {
                    "architecture": "completed_RPT_pair_encoder_wide_control",
                    "scientific_meaning": "pair_encoder_capacity_not_complete_graph_match",
                },
                "H_7X_UNBIASED_ENSEMBLE": {
                    "expert_count": 7,
                    "relation_for_every_expert": "BASE4",
                    "combiner": "logit_mean",
                },
                "H_7X_UNBIASED_TOKEN_FUSION": {
                    "expert_count": 7,
                    "relation_for_every_expert": "BASE4",
                    "combiner": "native_token_transformer",
                },
            },
        "same_hlt_inputs": True,
            "same_shared_hlt_normalizer": True,
            "same_fixed_40_epoch_protocol": True,
            "primary_realization_policy": "R_MULTI",
            "measurement_embedding_absent_from_ordinary_baselines": True,
            "performance_based_termination": False,
        }
    )


def _run(
    *,
    component: str,
    seed: int,
    role: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": resolve_run_id(
            stage="D",
            component=component,
            seed=seed,
            configuration=configuration,
        ),
        "stage": "D",
        "component": component,
        "seed": int(seed),
        "role": role,
        "selection_eligible": role == "scientific_candidate",
        "configuration": dict(configuration),
    }


def _expert_configuration(
    *,
    expert: str,
    shape_id: str,
    mode: str,
    measurement_embedding: bool,
    realization_policy: str,
    dual_weights: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    return {
        "kind": "NATIVE_HLT_EXPERT",
        "expert_id": expert,
        "shape_id": shape_id,
        "mode": mode,
        "measurement_embedding": bool(measurement_embedding),
        "realization_policy": realization_policy,
        "lambda_token": float(dual_weights[0]),
        "lambda_logit": float(dual_weights[1]),
        "shared_hlt_normalizer": True,
        "expert_specific_normalizer": False,
        "offline_targets_consumed": mode == "HE_DUAL_OBJECTIVE",
        "privileged_training": mode == "HE_DUAL_OBJECTIVE",
        "fixed_epochs": 40,
        "performance_based_termination": False,
        "architecture_binding": (
            "same_relation_identity_token_shape_tokenizer_and_classifier_"
            "topology_as_corresponding_selected_offline_expert"
        ),
    }


def build_scratch_expert_rows() -> list[dict[str, Any]]:
    return [
        _run(
            component="HLT_EXPERT",
            seed=101,
            role="reference_baseline",
            configuration=_expert_configuration(
                expert=expert,
                shape_id=shape_id,
                mode="HE_SCRATCH_CE",
                measurement_embedding=False,
                realization_policy="R_MULTI",
            ),
        )
        for shape_id in STAGE_D_SHAPES
        for expert in EXPERT_ORDER
    ]


def build_encoder_screen_rows() -> list[dict[str, Any]]:
    rows = []
    mode_weights = [
        ("HE_SCRATCH_CE", (0.0, 0.0)),
        ("HE_OFFLINE_INIT", (0.0, 0.0)),
        *[("HE_DUAL_OBJECTIVE", weights) for weights in DUAL_WEIGHTS],
    ]
    for shape_id in SELECTED_UNIFORM_SHAPES:
        for expert in EXPERT_ORDER:
            for mode, weights in mode_weights:
                for measurement in (False, True):
                    for policy in REALIZATION_POLICIES:
                        rows.append(
                            _run(
                                component="HLT_EXPERT",
                                seed=101,
                                role=(
                                    "reference_baseline"
                                    if mode == "HE_SCRATCH_CE"
                                    else "scientific_candidate"
                                ),
                                configuration=_expert_configuration(
                                    expert=expert,
                                    shape_id=shape_id,
                                    mode=mode,
                                    measurement_embedding=measurement,
                                    realization_policy=policy,
                                    dual_weights=weights,
                                ),
                            )
                        )
    if len(rows) != 420:
        raise RuntimeError("Stage-D encoder screen must contain 420 memberships")
    return rows


def build_bridge_parent_expert_rows() -> list[dict[str, Any]]:
    """Predeclare the seed-matched HE_OFFLINE_INIT banks Stage E consumes."""

    rows = [
        _run(
            component="HLT_EXPERT",
            seed=seed,
            role="bridge_parent",
            configuration=_expert_configuration(
                expert=expert,
                shape_id=shape_id,
                mode="HE_OFFLINE_INIT",
                measurement_embedding=False,
                realization_policy="R_MULTI",
            ),
        )
        for shape_id in (
            "SHAPE_COMPACT",
            "SHAPE_HIGH",
            "HET_PHYSICS",
            "HET_SELECTED",
            "HET_BEAM",
        )
        for expert in EXPERT_ORDER
        for seed in (101, 202, 303)
    ]
    if len(rows) != 105:
        raise RuntimeError("Stage-E HLT parent bank must contain 105 memberships")
    return rows


def build_native_fusion_rows() -> list[dict[str, Any]]:
    rows = []
    roles = {
        "HF_NATIVE": "scientific_candidate",
        "HF_LOGIT_MEAN": "architecture_control",
        "HF_TRAINED_LOGIT": "architecture_control",
        "HF_7X_UNBIASED_LOGIT_MEAN": "capacity_control",
        "HF_7X_UNBIASED_TOKEN_FUSION": "capacity_control",
    }
    for shape_id in STAGE_D_SHAPES:
        for variant in NATIVE_FUSION_VARIANTS:
            rows.append(
                _run(
                    component="NATIVE_HLT_FUSION",
                    seed=101,
                    role=roles[variant],
                    configuration={
                        "kind": "NATIVE_HLT_FUSION",
                        "shape_id": shape_id,
                        "fusion_variant": variant,
                        "realization_policy": "R_MULTI",
                        "offline_reconstruction": False,
                        "offline_targets_consumed": False,
                        "experts_frozen": True,
                        "fixed_epochs": (
                            0
                            if variant
                            in {
                                "HF_LOGIT_MEAN",
                                "HF_7X_UNBIASED_LOGIT_MEAN",
                            }
                            else 40
                        ),
                        "performance_based_termination": False,
                    },
                )
            )
    return rows


def build_baseline_rows() -> list[dict[str, Any]]:
    return [
        _run(
            component="HLT_EXPERT",
            seed=101,
            role=(
                "reference_baseline"
                if control == "H_BASE"
                else "capacity_control"
            ),
            configuration={
                "kind": "NATIVE_HLT_MATCHED_CONTROL",
                "control_id": control,
                "realization_policy": "R_MULTI",
                "shared_hlt_normalizer": True,
                "fixed_epochs": 40,
                "performance_based_termination": False,
            },
        )
        for control in (
            "H_BASE",
            "H_WIDE",
            "H_7X_UNBIASED_ENSEMBLE",
            "H_7X_UNBIASED_TOKEN_FUSION",
        )
    ]


def build_stage_d_run_registry() -> dict[str, Any]:
    scratch = build_scratch_expert_rows()
    screen = build_encoder_screen_rows()
    bridge_parents = build_bridge_parent_expert_rows()
    fusion = build_native_fusion_rows()
    baselines = build_baseline_rows()
    return with_content_hash(
        {
            "contract": STAGE_D_RUN_REGISTRY_CONTRACT,
            "schema_version": 2,
            "stage": "D",
            "screen_seed": 101,
            "confirmation_seeds": [101, 202, 303],
            "expert_order": list(EXPERT_ORDER),
            "shape_aliases": list(STAGE_D_SHAPES),
            "worker_access": ["model_train", "val_stop"],
            "design_worker_access": ["val_design"],
            "forbidden_access": ["stack_val", "final_test"],
            "scratch_expert_rows": scratch,
            "encoder_screen_rows": screen,
            "bridge_parent_expert_rows": bridge_parents,
            "native_fusion_rows": fusion,
            "baseline_rows": baselines,
            "row_counts": {
                "scratch_expert_memberships": len(scratch),
                "encoder_screen_memberships": len(screen),
                "bridge_parent_expert_memberships": len(bridge_parents),
                "native_fusion_memberships": len(fusion),
                "matched_baselines": len(baselines),
            },
            "selection": {
                "split": "val_design",
                "global_accuracy_window": 0.0001,
                "always_emits_if_all_candidates_worse": True,
                "scientific_underperformance_blocks_DAG": False,
            },
            "confirmation_template": {
                "seeds": [101, 202, 303],
                "members": [
                    "H_BASE",
                    "H_WIDE",
                    "selected_individual_HLT_experts",
                    "selected_HF_NATIVE",
                    "selected_HF_TRAINED_LOGIT",
                    "chosen_HLT_encoder_modes",
                ],
                "configuration_shared_across_seeds": True,
                "seed_matched_offline_initialization": True,
            },
            "graph_level_joint_deployment_realization_policy": "R_MULTI",
        }
    )


def validate_stage_d_run_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_D_RUN_REGISTRY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_d_run_registry()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-D run registry differs")
    return digest


def resolve_stage_d_run(
    registry: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    validate_stage_d_run_registry(registry)
    matches = [
        dict(row)
        for section in (
            "scratch_expert_rows",
            "encoder_screen_rows",
            "bridge_parent_expert_rows",
            "native_fusion_rows",
            "baseline_rows",
        )
        for row in registry[section]
        if row["run_id"] == run_id
    ]
    if not matches:
        raise ValueError("Stage-D run ID is not registered")
    if len({repr(row["configuration"]) for row in matches}) != 1:
        raise ValueError("Stage-D run ID has conflicting semantics")
    return {**matches[0], "registry_membership_count": len(matches)}


def materialize_stage_d_confirmation_rows(
    registry: Mapping[str, Any],
    *,
    selected_run_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Freeze selected screen semantics identically across all three seeds."""
    validate_stage_d_run_registry(registry)
    if not selected_run_ids or len(selected_run_ids) != len(set(selected_run_ids)):
        raise ValueError("Stage-D confirmation selections are empty/duplicated")
    selected = [
        resolve_stage_d_run(registry, run_id=run_id)
        for run_id in selected_run_ids
    ]
    if any(row["seed"] != 101 for row in selected):
        raise ValueError("Stage-D confirmation must originate at screen seed 101")
    experts = {
        row["configuration"].get("expert_id")
        for row in selected
        if row["configuration"].get("kind") == "NATIVE_HLT_EXPERT"
    }
    fusions = {
        row["configuration"].get("fusion_variant")
        for row in selected
        if row["configuration"].get("kind") == "NATIVE_HLT_FUSION"
    }
    if experts != set(EXPERT_ORDER):
        raise ValueError("Stage-D confirmation lacks a selected seven-expert bank")
    if not {"HF_NATIVE", "HF_TRAINED_LOGIT"}.issubset(fusions):
        raise ValueError("Stage-D confirmation lacks both selected native fusions")
    baseline_by_control = {
        row["configuration"]["control_id"]: row
        for row in registry["baseline_rows"]
    }
    source_rows = [
        baseline_by_control["H_BASE"],
        baseline_by_control["H_WIDE"],
        *selected,
    ]
    rows = []
    for source in source_rows:
        for seed in (101, 202, 303):
            rows.append(
                _run(
                    component=source["component"],
                    seed=seed,
                    role="confirmation",
                    configuration=source["configuration"],
                )
            )
    return with_content_hash(
        {
            "contract": STAGE_D_CONFIRMATION_REGISTRY_CONTRACT,
            "schema_version": 1,
            "stage_d_run_registry_sha256": registry["content_hash"],
            "screen_seed": 101,
            "confirmation_seeds": [101, 202, 303],
            "selected_screen_run_ids": list(selected_run_ids),
            "rows": rows,
            "configuration_shared_across_seeds": True,
            "seed_matched_offline_initialization": True,
            "scientific_underperformance_blocks_confirmation": False,
        }
    )


def validate_stage_d_confirmation_registry(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_D_CONFIRMATION_REGISTRY_CONTRACT
    )
    require_sha256(
        payload.get("stage_d_run_registry_sha256"),
        name="stage_d_run_registry_sha256",
    )
    if payload.get("confirmation_seeds") != [101, 202, 303]:
        raise ValueError("Stage-D confirmation seeds differ")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage-D confirmation rows are absent")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Stage-D confirmation run IDs are duplicated")
    for row in rows:
        expected = _run(
            component=row["component"],
            seed=int(row["seed"]),
            role="confirmation",
            configuration=row["configuration"],
        )
        if row != expected or row["seed"] not in {101, 202, 303}:
            raise ValueError("Stage-D confirmation row semantics differ")
    return digest


def resolve_stage_d_confirmation_run(
    registry: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    validate_stage_d_confirmation_registry(registry)
    matches = [row for row in registry["rows"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("Stage-D confirmation run ID is absent/duplicated")
    return dict(matches[0])


def execute_miniature_stage_d(
    registry: Mapping[str, Any],
    *,
    expert_executor: Any,
    fusion_executor: Any,
    baseline_executor: Any,
) -> dict[str, Any]:
    validate_stage_d_run_registry(registry)
    results = {"expert": 0, "fusion": 0, "baseline": 0}
    seen = set()
    for section, executor, kind in (
        ("scratch_expert_rows", expert_executor, "expert"),
        ("encoder_screen_rows", expert_executor, "expert"),
        ("bridge_parent_expert_rows", expert_executor, "expert"),
        ("native_fusion_rows", fusion_executor, "fusion"),
        ("baseline_rows", baseline_executor, "baseline"),
    ):
        for row in registry[section]:
            if row["run_id"] in seen:
                continue
            seen.add(row["run_id"])
            result = dict(executor(dict(row)))
            if (
                result.get("status") != "completed"
                or result.get("performance_based_termination", False)
            ):
                raise RuntimeError("miniature Stage-D row failed")
            results[kind] += 1
    return with_content_hash(
        {
            "contract": STEP6_MINIATURE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "stage_d_run_registry_sha256": registry["content_hash"],
            "unique_rows_completed": results,
            "native_specialization_measurable": True,
            "offline_initialization_measurable": True,
            "privileged_alignment_measurable": True,
            "performance_based_termination": False,
        }
    )


def build_step6_bundle(
    *,
    campaign_spec_sha256: str,
    step5_bundle_sha256: str,
    global_determinism_sha256: str,
    hlt_replica_manifest_sha256: str,
    shared_hlt_normalizer_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    parents = {
        "campaign_spec": require_sha256(
            campaign_spec_sha256, name="campaign_spec_sha256"
        ),
        "step5_bundle": require_sha256(
            step5_bundle_sha256, name="step5_bundle_sha256"
        ),
        "global_determinism": require_sha256(
            global_determinism_sha256, name="global_determinism_sha256"
        ),
        "hlt_replica_manifest": require_sha256(
            hlt_replica_manifest_sha256,
            name="hlt_replica_manifest_sha256",
        ),
        "shared_hlt_normalizer": require_sha256(
            shared_hlt_normalizer_sha256,
            name="shared_hlt_normalizer_sha256",
        ),
    }
    evidence = bind_source(
        build_hlt_evidence_mode_contract(), source_snapshot=source_snapshot
    )
    fusion = bind_source(
        build_native_fusion_contract(), source_snapshot=source_snapshot
    )
    controls = bind_source(
        build_hlt_matched_control_contract(), source_snapshot=source_snapshot
    )
    runs = bind_source(
        build_stage_d_run_registry(), source_snapshot=source_snapshot
    )
    training = bind_source(
        NativeHLTExpertTrainingConfig(
            seed=101, mode="HE_SCRATCH_CE"
        ).artifact(
            global_determinism_sha256=parents["global_determinism"],
            evidence_mode_contract_sha256=evidence["content_hash"],
        ),
        source_snapshot=source_snapshot,
    )
    artifacts = {
        "hlt_evidence_modes": evidence,
        "native_fusion": fusion,
        "matched_controls": controls,
        "stage_d_run_registry": runs,
        "hlt_expert_training": training,
    }
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP6_BUNDLE_CONTRACT,
                "schema_version": 2,
                "parents": parents,
                "artifact_hashes": {
                    name: artifact["content_hash"]
                    for name, artifact in sorted(artifacts.items())
                },
                "native_hlt_modes": list(HLT_MODES),
                "realization_policies": list(REALIZATION_POLICIES),
                "native_fusion_variants": list(NATIVE_FUSION_VARIANTS),
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": STEP6_REPORT_CONTRACT,
                "schema_version": 2,
                "campaign_spec_sha256": parents["campaign_spec"],
                "step6_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "scratch_is_target_free": True,
                    "offline_init_is_target_free": True,
                    "dual_objective_is_privileged": True,
                    "measurement_embedding_is_separate_control": True,
                    "replica_selection_identity_epoch_bound": True,
                    "evaluation_replica_zero": True,
                    "shared_hlt_normalizer_only": True,
                    "native_fusion_has_no_reconstruction": True,
                    "matched_baselines_registered": True,
                    "scientific_underperformance_does_not_block": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {**artifacts, "step6_bundle": manifest, "step6_report": report}


def validate_step6_bundle(bundle: Mapping[str, Any]) -> str:
    names = {
        "hlt_evidence_modes",
        "native_fusion",
        "matched_controls",
        "stage_d_run_registry",
        "hlt_expert_training",
        "step6_bundle",
        "step6_report",
    }
    if set(bundle) != names:
        raise ValueError("Step-6 bundle members differ")
    artifact_hashes = {
        "hlt_evidence_modes": validate_content_hash(
            bundle["hlt_evidence_modes"],
            expected_contract=HLT_EVIDENCE_MODE_CONTRACT,
        ),
        "native_fusion": validate_content_hash(
            bundle["native_fusion"],
            expected_contract="retb_native_hlt_fusion_architecture_v1",
        ),
        "matched_controls": validate_content_hash(
            bundle["matched_controls"],
            expected_contract=HLT_MATCHED_CONTROL_CONTRACT,
        ),
        "stage_d_run_registry": validate_stage_d_run_registry(
            bundle["stage_d_run_registry"]
        ),
        "hlt_expert_training": validate_content_hash(
            bundle["hlt_expert_training"],
            expected_contract="retb_native_hlt_expert_training_v2",
        ),
    }
    digest = validate_content_hash(
        bundle["step6_bundle"], expected_contract=STEP6_BUNDLE_CONTRACT
    )
    if bundle["step6_bundle"]["artifact_hashes"] != artifact_hashes:
        raise ValueError("Step-6 bundle artifact hashes differ")
    validate_content_hash(
        bundle["step6_report"], expected_contract=STEP6_REPORT_CONTRACT
    )
    if bundle["step6_report"]["step6_bundle_sha256"] != digest:
        raise ValueError("Step-6 report parent differs")
    sources = {repr(value.get("source")) for value in bundle.values()}
    if len(sources) != 1:
        raise ValueError("Step-6 source lineage differs")
    return digest


def publish_step6_bundle(
    *, campaign_root: str | Path, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    digest = validate_step6_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "hlt_evidence_modes": root / "registry" / "retb_hlt_evidence_modes.json",
        "native_fusion": root / "registry" / "retb_native_hlt_fusion.json",
        "matched_controls": root / "registry" / "retb_hlt_matched_controls.json",
        "stage_d_run_registry": root / "registry" / "retb_stage_d_runs.json",
        "hlt_expert_training": root / "registry" / "retb_hlt_expert_training.json",
        "step6_bundle": root / "registry" / "retb_step6_native_hlt_bundle.json",
        "step6_report": root / "reports" / "retb_step6_report.json",
    }
    publications = {
        name: write_immutable_json(path, bundle[name])
        for name, path in paths.items()
    }
    return {
        "campaign_root": str(root.resolve()),
        "step6_bundle_sha256": digest,
        "publications": publications,
    }


__all__ = [
    "STAGE_D_RUN_REGISTRY_CONTRACT",
    "STAGE_D_CONFIRMATION_REGISTRY_CONTRACT",
    "STEP6_BUNDLE_CONTRACT",
    "build_stage_d_run_registry",
    "build_step6_bundle",
    "execute_miniature_stage_d",
    "materialize_stage_d_confirmation_rows",
    "publish_step6_bundle",
    "resolve_stage_d_run",
    "resolve_stage_d_confirmation_run",
    "validate_stage_d_confirmation_registry",
    "validate_stage_d_run_registry",
    "validate_step6_bundle",
]
