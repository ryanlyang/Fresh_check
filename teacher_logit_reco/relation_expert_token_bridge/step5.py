"""Immutable Stage-C offline-fusion campaign registry and Step-5 bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .complementarity import (
    COMPLEMENTARITY_CONTRACT,
    SUBSET_READOUT_CONTRACT,
)
from .capacity import build_capacity_control_registry
from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .fusion import (
    FUSION_VARIANTS,
    build_grouped_head_relation_contract,
    build_offline_fusion_architecture_contract,
    build_relation_specialization_contract,
    validate_offline_fusion_architecture_contract,
)
from .fusion_cache import FROZEN_TOKEN_CACHE_CONTRACT
from .fusion_training import OfflineFusionTrainingConfig
from .registry import EXPERT_ORDER, TOKEN_SHAPES, resolve_run_id
from .selection import (
    HETEROGENEOUS_SELECTION_CONTRACT,
    JOINT_EXPERT_LOSS_SELECTION_CONTRACT,
    OFFLINE_SHAPE_SELECTION_CONTRACT,
    UNIFORM_SHAPE_METRICS_CONTRACT,
)


STAGE_C_RUN_REGISTRY_CONTRACT = "retb_stage_c_run_registry_v1"
STEP5_BUNDLE_CONTRACT = "retb_step5_offline_fusion_bundle_v1"
STEP5_REPORT_CONTRACT = "retb_step5_report_v1"
STEP5_MINIATURE_COMPLETION_CONTRACT = "retb_step5_miniature_completion_v1"
PIPELINE_SEEDS = (101, 202, 303)


def _run(
    *,
    component: str,
    seed: int,
    role: str,
    configuration: Mapping[str, Any],
    reuse_run_id: str | None = None,
) -> dict[str, Any]:
    run_id = reuse_run_id or resolve_run_id(
        stage="C",
        component=component,
        seed=seed,
        configuration=configuration,
    )
    return {
        "run_id": run_id,
        "stage": "C",
        "component": component,
        "seed": int(seed),
        "role": role,
        "selection_eligible": role == "scientific_candidate",
        "configuration": dict(configuration),
        "reused_physical_run": reuse_run_id is not None,
    }


def build_expert_confirmation_rows() -> list[dict[str, Any]]:
    rows = []
    for shape_id in TOKEN_SHAPES:
        for seed in PIPELINE_SEEDS:
            for expert in EXPERT_ORDER:
                rows.append(
                    _run(
                        component="OFFLINE_FUSION",
                        seed=seed,
                        role="scientific_candidate",
                        configuration={
                            "kind": "PURE_OFFLINE_EXPERT_CONFIRMATION",
                            "expert_id": expert,
                            "shape_id": shape_id,
                            "loss_id": "ELOSS_CE",
                            "topology": "B_CONCAT",
                            "pipeline_seed": seed,
                            "fixed_epochs": 40,
                            "performance_based_termination": False,
                        },
                    )
                )
    if len(rows) != 147 or len({row["run_id"] for row in rows}) != 147:
        raise RuntimeError("Stage-C expert confirmation must contain 147 rows")
    return rows


def build_canonical_fusion_rows() -> list[dict[str, Any]]:
    rows = []
    for shape_id in TOKEN_SHAPES:
        shape = TOKEN_SHAPES[shape_id]
        for seed in PIPELINE_SEEDS:
            rows.append(
                _run(
                    component="OFFLINE_FUSION",
                    seed=seed,
                    role="scientific_candidate",
                    configuration={
                        "kind": "CANONICAL_UNIFORM_FUSION_CONFIRMATION",
                        "fusion_variant": "F_TOKEN_TRANSFORMER",
                        "shape_id": shape_id,
                        "allocation": {
                            expert: [shape["K"], shape["D"]]
                            for expert in EXPERT_ORDER
                        },
                        "pipeline_seed": seed,
                        "expert_state": "frozen",
                        "fixed_epochs": 40,
                        "whole_bank_dropout": 0.0,
                        "performance_based_termination": False,
                    },
                )
            )
    if len(rows) != 21:
        raise RuntimeError("Stage-C canonical fusion confirmation has wrong size")
    return rows


def build_uniform_control_rows(
    canonical: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    canonical_rows = build_canonical_fusion_rows() if canonical is None else canonical
    lookup = {
        row["configuration"]["shape_id"]: row
        for row in canonical_rows
        if row["seed"] == 101
    }
    rows = []
    for shape_id in TOKEN_SHAPES:
        for variant in (
            "F_BEST_SINGLE",
            "F_UNIFORM_LOGIT_MEAN",
            "F_TRAINED_LOGIT_LINEAR",
            "F_POOLED_MLP",
            "F_TOKEN_TRANSFORMER",
        ):
            configuration = {
                "kind": "UNIFORM_FUSION_CONTROL",
                "fusion_variant": variant,
                "shape_id": shape_id,
                "pipeline_seed": 101,
                "expert_state": "frozen",
                "fixed_epochs": 0
                if variant in {"F_BEST_SINGLE", "F_UNIFORM_LOGIT_MEAN"}
                else 40,
                "whole_bank_dropout": 0.0,
                "performance_based_termination": False,
            }
            rows.append(
                _run(
                    component="OFFLINE_FUSION",
                    seed=101,
                    role=(
                        "scientific_candidate"
                        if variant == "F_TOKEN_TRANSFORMER"
                        else "architecture_control"
                    ),
                    configuration=(
                        lookup[shape_id]["configuration"]
                        if variant == "F_TOKEN_TRANSFORMER"
                        else configuration
                    ),
                    reuse_run_id=(
                        lookup[shape_id]["run_id"]
                        if variant == "F_TOKEN_TRANSFORMER"
                        else None
                    ),
                )
            )
    return rows


def build_stage_c_run_registry() -> dict[str, Any]:
    experts = build_expert_confirmation_rows()
    canonical = build_canonical_fusion_rows()
    uniform_controls = build_uniform_control_rows(canonical)
    return with_content_hash(
        {
            "contract": STAGE_C_RUN_REGISTRY_CONTRACT,
            "schema_version": 1,
            "stage": "C",
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "expert_order": list(EXPERT_ORDER),
            "uniform_shape_order": list(TOKEN_SHAPES),
            "worker_access": ["model_train", "val_stop"],
            "design_worker_access": ["val_design"],
            "forbidden_access": ["stack_val", "final_test"],
            "row_counts": {
                "expert_shape_seed_confirmation": len(experts),
                "canonical_fusion_shape_seed_confirmation": len(canonical),
                "uniform_seed101_control_memberships": len(uniform_controls),
                "required_complete_shape_seed_metrics": 21,
                "subset_readouts_per_shape_seed": 128,
            },
            "expert_confirmation_rows": experts,
            "canonical_fusion_rows": canonical,
            "uniform_control_rows": uniform_controls,
            "dynamic_control_templates": {
                "at_S8_128_and_seed101_best_uniform": [
                    "F_TOKEN_TRANSFORMER_LIGHT_FINETUNE",
                    "F_TOKEN_TRANSFORMER_FULL_FINETUNE",
                    "O_GROUPED_HEAD_REL",
                    "O_7X_UNBIASED_ENSEMBLE",
                    "O_7X_UNBIASED_TOKEN_FUSION",
                    "S0_NATURAL",
                    "S1_FIXED_SCALE",
                    "S2_BOUNDED_SCALE",
                    "S3_RELATION_AUX",
                    "S4_RESTRICTED_FIELDS",
                    "S5_CROSSCOV",
                ],
                "all_128_subset_readouts": True,
                "whole_bank_dropout_primary": False,
            },
            "post_shape_lock_templates": {
                "shapes": ["SHAPE_HIGH", "SHAPE_COMPACT"],
                "all_carried_expert_loss_candidates": True,
                "joint_loss_selector": JOINT_EXPERT_LOSS_SELECTION_CONTRACT,
                "heterogeneous": [
                    "HET_PHYSICS",
                    "HET_SELECTED",
                    "HET_BEAM",
                ],
            },
            "selectors": {
                "uniform_metrics": UNIFORM_SHAPE_METRICS_CONTRACT,
                "shape_selection": OFFLINE_SHAPE_SELECTION_CONTRACT,
                "heterogeneous_selection": HETEROGENEOUS_SELECTION_CONTRACT,
                "poor_performance_blocks_future_stages": False,
            },
        }
    )


def validate_stage_c_run_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_C_RUN_REGISTRY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_c_run_registry()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-C run registry differs")
    return digest


def resolve_stage_c_run(
    registry: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    validate_stage_c_run_registry(registry)
    rows = [
        dict(row)
        for section in (
            "expert_confirmation_rows",
            "canonical_fusion_rows",
            "uniform_control_rows",
        )
        for row in registry[section]
        if row["run_id"] == str(run_id)
    ]
    if not rows:
        raise ValueError(f"Stage-C run ID is not registered: {run_id!r}")
    configurations = {repr(row["configuration"]) for row in rows}
    if len(configurations) != 1:
        raise ValueError("Stage-C physical run has conflicting configurations")
    return {
        **rows[0],
        "registry_memberships": [
            {
                "role": row["role"],
                "selection_eligible": row["selection_eligible"],
            }
            for row in rows
        ],
    }


def execute_miniature_stage_c(
    registry: Mapping[str, Any],
    *,
    expert_executor: Any,
    fusion_executor: Any,
) -> dict[str, Any]:
    validate_stage_c_run_registry(registry)
    expert_results = []
    for row in registry["expert_confirmation_rows"]:
        result = dict(expert_executor(dict(row)))
        if result.get("status") != "completed":
            raise RuntimeError("miniature Stage-C expert row failed")
        expert_results.append(row["run_id"])
    fusion_results = []
    for row in registry["canonical_fusion_rows"]:
        result = dict(fusion_executor(dict(row)))
        if (
            result.get("status") != "completed"
            or result.get("performance_based_termination", False)
        ):
            raise RuntimeError("miniature Stage-C fusion row failed")
        fusion_results.append(row["run_id"])
    return with_content_hash(
        {
            "contract": STEP5_MINIATURE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "stage_c_run_registry_sha256": registry["content_hash"],
            "expert_rows_completed": len(expert_results),
            "canonical_fusion_rows_completed": len(fusion_results),
            "expert_147_complete": len(expert_results) == 147,
            "fusion_21_complete": len(fusion_results) == 21,
            "performance_based_termination": False,
        }
    )


def build_step5_bundle(
    *,
    campaign_spec_sha256: str,
    step4_bundle_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_sha = require_sha256(
        campaign_spec_sha256, name="campaign_spec_sha256"
    )
    step4_sha = require_sha256(
        step4_bundle_sha256, name="step4_bundle_sha256"
    )
    determinism_sha = require_sha256(
        global_determinism_sha256, name="global_determinism_sha256"
    )
    fusion = bind_source(
        build_offline_fusion_architecture_contract(),
        source_snapshot=source_snapshot,
    )
    grouped = bind_source(
        build_grouped_head_relation_contract(),
        source_snapshot=source_snapshot,
    )
    specialization = bind_source(
        build_relation_specialization_contract(),
        source_snapshot=source_snapshot,
    )
    capacity = bind_source(
        build_capacity_control_registry(),
        source_snapshot=source_snapshot,
    )
    runs = bind_source(
        build_stage_c_run_registry(),
        source_snapshot=source_snapshot,
    )
    training = bind_source(
        OfflineFusionTrainingConfig(seed=101).artifact(
            global_determinism_sha256=determinism_sha,
            fusion_architecture_sha256=fusion["content_hash"],
        ),
        source_snapshot=source_snapshot,
    )
    artifacts = {
        "fusion_architecture": fusion,
        "capacity_controls": capacity,
        "grouped_head_relation": grouped,
        "relation_specialization": specialization,
        "stage_c_run_registry": runs,
        "fusion_training_protocol": training,
    }
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP5_BUNDLE_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign_sha,
                "step4_bundle_sha256": step4_sha,
                "global_determinism_sha256": determinism_sha,
                "artifact_hashes": {
                    name: value["content_hash"]
                    for name, value in sorted(artifacts.items())
                },
                "cache_contract": FROZEN_TOKEN_CACHE_CONTRACT,
                "complementarity_contract": COMPLEMENTARITY_CONTRACT,
                "subset_readout_contract": SUBSET_READOUT_CONTRACT,
                "fusion_variants": list(FUSION_VARIANTS),
                "expert_confirmation_rows": 147,
                "canonical_fusion_rows": 21,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": STEP5_REPORT_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign_sha,
                "step5_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "frozen_cache_identity_bound": True,
                    "all_seven_fusion_variants_registered": True,
                    "canonical_expert_order_locked": True,
                    "primary_experts_frozen": True,
                    "primary_expert_dropout_disabled": True,
                    "subset_readouts_cover_128_subsets": True,
                    "joint_loss_beam_width_16": True,
                    "heterogeneous_greedy_and_beam_locked": True,
                    "slot_budget_56": True,
                    "complete_seven_shapes_three_seeds_required": True,
                    "negative_campaign_still_selects": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {**artifacts, "step5_bundle": manifest, "step5_report": report}


def validate_step5_bundle(bundle: Mapping[str, Any]) -> str:
    names = {
        "fusion_architecture",
        "capacity_controls",
        "grouped_head_relation",
        "relation_specialization",
        "stage_c_run_registry",
        "fusion_training_protocol",
        "step5_bundle",
        "step5_report",
    }
    if set(bundle) != names:
        raise ValueError("Step-5 bundle members differ")
    hashes = {
        "fusion_architecture": validate_offline_fusion_architecture_contract(
            bundle["fusion_architecture"]
        ),
        "capacity_controls": validate_content_hash(
            bundle["capacity_controls"],
            expected_contract="retb_capacity_control_registry_v1",
        ),
        "grouped_head_relation": validate_content_hash(
            bundle["grouped_head_relation"],
            expected_contract="retb_grouped_head_relation_control_v1",
        ),
        "relation_specialization": validate_content_hash(
            bundle["relation_specialization"],
            expected_contract="retb_relation_specialization_controls_v1",
        ),
        "stage_c_run_registry": validate_stage_c_run_registry(
            bundle["stage_c_run_registry"]
        ),
        "fusion_training_protocol": validate_content_hash(
            bundle["fusion_training_protocol"],
            expected_contract="retb_offline_fusion_training_v1",
        ),
    }
    manifest_sha = validate_content_hash(
        bundle["step5_bundle"], expected_contract=STEP5_BUNDLE_CONTRACT
    )
    validate_content_hash(
        bundle["step5_report"], expected_contract=STEP5_REPORT_CONTRACT
    )
    if bundle["step5_bundle"]["artifact_hashes"] != {
        name: value for name, value in sorted(hashes.items())
    }:
        raise ValueError("Step-5 artifact hashes differ")
    if bundle["step5_report"]["step5_bundle_sha256"] != manifest_sha:
        raise ValueError("Step-5 report belongs to another bundle")
    source = bundle["step5_bundle"].get("source")
    expected = build_step5_bundle(
        campaign_spec_sha256=bundle["step5_bundle"]["campaign_spec_sha256"],
        step4_bundle_sha256=bundle["step5_bundle"]["step4_bundle_sha256"],
        global_determinism_sha256=bundle["step5_bundle"][
            "global_determinism_sha256"
        ],
        source_snapshot={
            "source_commit": source["commit"],
            "source_status_sha256": source["status_sha256"],
            "source_dirty": source["dirty"],
        },
    )
    if dict(bundle) != expected:
        raise ValueError("Step-5 bundle differs from deterministic rebuild")
    return manifest_sha


def publish_step5_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    digest = validate_step5_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "fusion_architecture": root / "registry" / "retb_offline_fusion.json",
        "capacity_controls": root / "registry" / "retb_capacity_controls.json",
        "grouped_head_relation": root
        / "registry"
        / "retb_grouped_head_relation.json",
        "relation_specialization": root
        / "registry"
        / "retb_relation_specialization.json",
        "stage_c_run_registry": root / "registry" / "retb_stage_c_runs.json",
        "fusion_training_protocol": root
        / "registry"
        / "retb_offline_fusion_training.json",
        "step5_bundle": root
        / "registry"
        / "retb_step5_offline_fusion_bundle.json",
        "step5_report": root / "reports" / "retb_step5_report.json",
    }
    publications = {
        name: write_immutable_json(paths[name], bundle[name])
        for name in sorted(paths)
    }
    return {
        "campaign_root": str(root.resolve()),
        "step5_bundle_sha256": digest,
        "publications": publications,
    }


__all__ = [
    "STAGE_C_RUN_REGISTRY_CONTRACT",
    "STEP5_BUNDLE_CONTRACT",
    "STEP5_MINIATURE_COMPLETION_CONTRACT",
    "STEP5_REPORT_CONTRACT",
    "build_canonical_fusion_rows",
    "build_expert_confirmation_rows",
    "build_stage_c_run_registry",
    "build_step5_bundle",
    "build_uniform_control_rows",
    "execute_miniature_stage_c",
    "publish_step5_bundle",
    "resolve_stage_c_run",
    "validate_stage_c_run_registry",
    "validate_step5_bundle",
]
