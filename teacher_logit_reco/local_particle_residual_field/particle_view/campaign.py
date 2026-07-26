"""Authoritative low-data campaign inventory for particle-view distillation.

This module freezes scientific run identities independently of the Slurm
packing used to execute them. Numerically resolved winners are represented by
predeclared role slots, so weak results can never erase downstream work.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .contracts import validate_content_hash, with_content_hash
from .controls import (
    STRUCTURAL_CONTROL_IDS,
    TRAINED_CONTROL_IDS,
    build_step8_control_registry,
)
from .distillation import build_target_loss_interaction_campaign
from .predictor import (
    NONSELECTABLE_PREDICTOR_ARCHITECTURES,
    PARTICLE_VIEW_PREDICTOR_ARCHITECTURES,
)
from .registry import (
    PARTICLE_VIEW_SEEDS,
    ParticleViewRunSpec,
    build_particle_view_registry,
    validate_particle_view_registry,
)
from .splits import PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT


PARTICLE_VIEW_LOW_DATA_INVENTORY_CONTRACT = (
    "particle_view_low_data_campaign_inventory_v1"
)
PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID = "particle_view_500k_full_pilot_v1"

TARGET_SCREEN_IDS = (
    "VGEN_TAP_RAW",
    "VGEN_TAP_MID",
    "VGEN_TAP_PENULT",
    "VGEN_TAP_FINAL",
    "VGEN_TAP_MIX3",
    "VGEN_QUERY_RAW",
    "VGEN_QUERY_EMBED",
    "VGEN_QUERY_MID",
    "VGEN_QUERY_PENULT",
    "VGEN_QUERY_MIX3",
    "VGEN_XATTN1",
    "VGEN_XATTN2",
    "VGEN_XATTN4",
    "VGEN_NO_PAIR",
    "VGEN_PAIR",
    "VGEN_LOCAL02",
    "VGEN_LOCAL04",
    "VGEN_NO_NULL",
    "VGEN_NULL",
    "VGEN_CENTERED",
    "VGEN_UNCENTERED",
    "VGEN_DIM1",
    "VGEN_DIM2",
    "VGEN_DIM4",
    "VGEN_DIM8",
    "VGEN_KD000",
    "VGEN_KD025",
    "VGEN_KD050",
    "VGEN_KD100",
    "VGEN_NO_RATE",
    "VGEN_RECODESIGN",
    "VGEN_TEACHER_LARGE",
    "VGEN_TEACHER_EXISTING",
    "VGEN_TEACHER_MIX2",
    "VGEN_MEMORY_HLT",
    "VGEN_MEMORY_HLT_SELFMASK",
)

NONSELECTABLE_TARGET_SCREEN_IDS = frozenset(
    {
        "VGEN_LOCAL02",
        "VGEN_LOCAL04",
        "VGEN_UNCENTERED",
        "VGEN_NO_RATE",
        "VGEN_MEMORY_HLT_SELFMASK",
    }
)

CONSUMER_SCREEN_IDS = (
    "C_RAWCAT",
    "C_EMBED",
    "C_PAIR",
    "C_EMBED_PAIR",
    "C_INJECT0",
    "C_INJECT1",
    "C_INJECTMID",
    "C_FIXED_TRUST",
    "C_GATED_TRUST",
    "C_CLEAN",
    "C_DROPOUT",
    "C_ROBUST_MIX",
)

CONFIRMATION_ROLE_IDS = (
    "CANONICAL_PREDECLARED",
    "BEST_ARCHITECTURE",
    "BEST_NO_CE",
    "BEST_SMALL_CE",
    "CE_ONLY_UPPER_BOUND",
    "REPRESENTATION_ONLY",
    "DIRECT_PARAMETER_CONTROL",
    "DIRECT_FLOP_CONTROL",
    "BEST_ALTERNATIVE_TARGET",
    "RECOVERABILITY_CODESIGNED",
    "HLT_MEMORY_CONTROL",
    "DVIEW_JOINT",
    "DVIEW_JOINT_CE_ONLY",
)

FOCUSED_INTERACTION_IDS = (
    "DIM1_VIEW_ALL",
    "DIM1_KD_VIEW_REL",
    "DIM2_VIEW_ALL",
    "DIM2_KD_VIEW_REL",
    "DIM4_VIEW_ALL",
    "DIM4_KD_VIEW_REL",
    "DIM8_VIEW_ALL",
    "DIM8_KD_VIEW_REL",
    "TAP_PENULT_PRIMARY",
    "TAP_MIX3_PRIMARY",
    "CENTERED_DIM2",
    "CENTERED_DIM4",
    "CENTERED_DIM8",
    "UNCENTERED_DIM2",
    "UNCENTERED_DIM4",
    "UNCENTERED_DIM8",
    "STANDARD_DIM2",
    "STANDARD_DIM4",
    "STANDARD_DIM8",
    "RECODESIGN_DIM2",
    "RECODESIGN_DIM4",
    "RECODESIGN_DIM8",
)

_BASELINE_IDS = (
    "A0_VIEW",
    "TOFF_VIEW_BASE",
    "TOFF_VIEW_LARGE",
    "TOFF_VIEW_EXISTING",
    "STAGE_A_PARAMETER_MATCH",
    "STAGE_A_FLOP_MATCH",
)

_VIEW_PUBLICATION_IDS = (
    "SELECTED_COORDINATE_BINDING",
    "SELECTED_VIEW_CACHE",
    "FINAL_CLEAN_CONSUMER",
)

_REPRESENTATION_IDS = (
    "PVIEW0",
    "RESIDUAL_SAMPLER",
    "ROBUST_CONSUMER",
)

_PRE_STAGE_TRAINED_CONTROLS = tuple(
    control_id
    for control_id in TRAINED_CONTROL_IDS
    if control_id
    not in {
        "STAGE_A_PARAMETER_MATCH",
        "STAGE_A_FLOP_MATCH",
        "SELECTED_PARAMETER_MATCH",
        "SELECTED_FLOP_MATCH",
        "A0_VIEW_LONG_DEPLOY",
        "A0_VIEW_TOTAL_LABEL_BUDGET",
    }
)

_FAIRNESS_CONTROL_IDS = (
    "A0_VIEW_LONG_DEPLOY",
    "A0_VIEW_TOTAL_LABEL_BUDGET",
    "SELECTED_PARAMETER_MATCH",
    "SELECTED_FLOP_MATCH",
)

_WINNER_FAMILIES = (
    "PRIVILEGED_SCIENTIFIC",
    "PRE_STAGE_G_DEPLOYABLE",
)

_STACK_STATIC_IDS = (
    "MATCHED_CE_ONLY_COMPARATOR",
    "A0_A0_PAIR_101_202",
    "A0_A0_PAIR_202_303",
    "A0_A0_PAIR_303_101",
    "PRIVILEGED_LOGIT_AVERAGE",
    "PRIVILEGED_LINEAR_FUSION",
    "DEPLOYABLE_LOGIT_AVERAGE",
    "DEPLOYABLE_LINEAR_FUSION",
    "OPTIONAL_P7B_FUSION",
)

_REPORT_EXPORT_IDS = (
    "AGGREGATE_REPORT",
    "EXPORT_PRIVILEGED_WINNER",
    "EXPORT_DEPLOYABLE_WINNER",
    "RELOAD_PRIVILEGED_WINNER",
    "RELOAD_DEPLOYABLE_WINNER",
    "FINAL_PERMIT_PRIVILEGED",
    "FINAL_PERMIT_DEPLOYABLE",
)

EXPECTED_LOW_DATA_CATEGORY_COUNTS = {
    "source": 1,
    "baseline": len(_BASELINE_IDS),
    "target_generator": len(TARGET_SCREEN_IDS),
    "consumer_interface": len(CONSUMER_SCREEN_IDS),
    "target_selection": 1,
    "view_publication": len(_VIEW_PUBLICATION_IDS),
    "representation": len(_REPRESENTATION_IDS),
    "predictor_architecture": len(PARTICLE_VIEW_PREDICTOR_ARCHITECTURES),
    "distillation": 52,
    "focused_interaction": len(FOCUSED_INTERACTION_IDS),
    "trained_control": len(_PRE_STAGE_TRAINED_CONTROLS),
    "structural_control": len(STRUCTURAL_CONTROL_IDS),
    "confirmation_role": len(CONFIRMATION_ROLE_IDS),
    "winner_selection": 1,
    "fairness_ledger": 1,
    "fairness_control": len(_WINNER_FAMILIES) * len(_FAIRNESS_CONTROL_IDS),
    "stack_winner": len(_WINNER_FAMILIES),
    "stack_fairness_control": (
        len(_WINNER_FAMILIES) * len(_FAIRNESS_CONTROL_IDS)
    ),
    "stack_static": len(_STACK_STATIC_IDS),
    "report_export": len(_REPORT_EXPORT_IDS),
    "final_test": len(_WINNER_FAMILIES),
}


def _category(scientific_role: str) -> str:
    return scientific_role.split(":", 1)[0]


def _distillation_rows() -> list[Mapping[str, Any]]:
    campaign = build_target_loss_interaction_campaign(
        target_ids=("TARGET_CANONICAL_SELECTED", "TARGET_ALTERNATE_SELECTED"),
        canonical_target_id="TARGET_CANONICAL_SELECTED",
        alternate_target_id="TARGET_ALTERNATE_SELECTED",
    )
    if campaign["row_count"] != EXPECTED_LOW_DATA_CATEGORY_COUNTS["distillation"]:
        raise RuntimeError("distillation campaign inventory drifted")
    return list(campaign["rows"])


def build_low_data_campaign_registry(
    *,
    unified_split_manifest: Mapping[str, Any],
    existing_teacher_compatible: bool = False,
    teacher_mix_compatible: bool = False,
    campaign_id: str = PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID,
) -> dict[str, Any]:
    """Build the complete predeclared 500k pilot registry.

    Compatibility flags affect selection permissions only. They never remove
    a target row or alter the amount of queued work.
    """

    validate_content_hash(
        unified_split_manifest,
        expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    )
    train_identity = unified_split_manifest["logical_splits"]["train"][
        "ordered_identity_sha256"
    ]
    specs: list[ParticleViewRunSpec] = []

    def add(
        run_id: str,
        *,
        stage: str,
        category: str,
        detail: str,
        parents: tuple[str, ...] = (),
        seeds: tuple[int, ...] = (101,),
        uses_labels: bool = True,
        selectable: bool = False,
        diagnostic: bool = False,
        family: str = "infrastructure",
        clean_paired: bool = False,
        robust_paired: bool = False,
        stack: bool = False,
        final: bool = False,
    ) -> None:
        specs.append(
            ParticleViewRunSpec(
                run_id=run_id,
                stage=stage,
                scientific_role=f"{category}:{detail}",
                selection_family=family,
                seed_ids=seeds,
                parent_run_ids=parents,
                uses_labels=uses_labels,
                train_split="train" if uses_labels else None,
                selectable=selectable,
                diagnostic=diagnostic,
                clean_consumer_paired=clean_paired,
                robust_consumer_paired=robust_paired,
                stack_val_eligible=stack,
                final_test_eligible=final,
            )
        )

    add(
        "PV_SOURCE_PREFLIGHT",
        stage="source",
        category="source",
        detail="unified_manifest_sources_storage",
        uses_labels=False,
    )

    for baseline_id in _BASELINE_IDS:
        diagnostic = baseline_id == "TOFF_VIEW_EXISTING"
        seeds = (
            PARTICLE_VIEW_SEEDS
            if baseline_id
            in {"A0_VIEW", "STAGE_A_PARAMETER_MATCH", "STAGE_A_FLOP_MATCH"}
            else (101,)
        )
        add(
            baseline_id,
            stage="baseline",
            category="baseline",
            detail=baseline_id,
            parents=("PV_SOURCE_PREFLIGHT",),
            seeds=seeds,
            diagnostic=diagnostic,
            family="diagnostic" if diagnostic else "infrastructure",
        )

    target_run_ids = []
    for target_id in TARGET_SCREEN_IDS:
        diagnostic = target_id in NONSELECTABLE_TARGET_SCREEN_IDS
        compatible = (
            existing_teacher_compatible
            if target_id == "VGEN_TEACHER_EXISTING"
            else teacher_mix_compatible
            if target_id == "VGEN_TEACHER_MIX2"
            else True
        )
        diagnostic = diagnostic or not compatible
        family = (
            "pre_stage_g_deployable"
            if target_id == "VGEN_MEMORY_HLT"
            else "diagnostic"
            if diagnostic
            else "privileged_scientific"
        )
        seeds = (
            PARTICLE_VIEW_SEEDS
            if target_id == "VGEN_MEMORY_HLT"
            else (101,)
        )
        if target_id == "VGEN_TEACHER_LARGE":
            target_parents = ("A0_VIEW", "TOFF_VIEW_LARGE")
        elif target_id == "VGEN_TEACHER_EXISTING":
            target_parents = ("A0_VIEW", "TOFF_VIEW_EXISTING")
        elif target_id == "VGEN_TEACHER_MIX2":
            target_parents = (
                "A0_VIEW",
                "TOFF_VIEW_BASE",
                "TOFF_VIEW_LARGE",
            )
        elif target_id in {
            "VGEN_MEMORY_HLT",
            "VGEN_MEMORY_HLT_SELFMASK",
        }:
            target_parents = ("A0_VIEW",)
        else:
            target_parents = ("A0_VIEW", "TOFF_VIEW_BASE")
        add(
            target_id,
            stage="target_screen",
            category="target_generator",
            detail=target_id,
            parents=target_parents,
            seeds=seeds,
            selectable=not diagnostic,
            diagnostic=diagnostic,
            family=family,
        )
        target_run_ids.append(target_id)

    consumer_run_ids = []
    for consumer_id in CONSUMER_SCREEN_IDS:
        run_id = f"SCREEN_{consumer_id}"
        add(
            run_id,
            stage="target_screen",
            category="consumer_interface",
            detail=consumer_id,
            parents=("VGEN_TAP_PENULT",),
            selectable=True,
            family="privileged_scientific",
            clean_paired=consumer_id == "C_CLEAN",
            robust_paired=consumer_id == "C_ROBUST_MIX",
        )
        consumer_run_ids.append(run_id)

    add(
        "SELECT_TARGET_DEFINITIONS",
        stage="target_screen",
        category="target_selection",
        detail="rank_top_two_plus_canonical",
        parents=tuple(target_run_ids + consumer_run_ids),
        uses_labels=False,
    )

    previous = "SELECT_TARGET_DEFINITIONS"
    for publication_id in _VIEW_PUBLICATION_IDS:
        uses_labels = publication_id == "FINAL_CLEAN_CONSUMER"
        add(
            publication_id,
            stage="view_publication",
            category="view_publication",
            detail=publication_id,
            parents=(previous,),
            uses_labels=uses_labels,
        )
        previous = publication_id

    previous = "FINAL_CLEAN_CONSUMER"
    for representation_id in _REPRESENTATION_IDS:
        uses_labels = representation_id != "RESIDUAL_SAMPLER"
        add(
            representation_id,
            stage="representation",
            category="representation",
            detail=representation_id,
            parents=(previous,),
            uses_labels=uses_labels,
        )
        previous = representation_id

    predictor_run_ids = []
    for architecture_id in PARTICLE_VIEW_PREDICTOR_ARCHITECTURES:
        diagnostic = architecture_id in NONSELECTABLE_PREDICTOR_ARCHITECTURES
        run_id = f"ARCH_{architecture_id}"
        add(
            run_id,
            stage="predictor",
            category="predictor_architecture",
            detail=architecture_id,
            parents=("ROBUST_CONSUMER",),
            selectable=not diagnostic,
            diagnostic=diagnostic,
            family="diagnostic" if diagnostic else "privileged_scientific",
        )
        predictor_run_ids.append(run_id)

    for index, row in enumerate(_distillation_rows()):
        run_id = f"DISTILL_{index:03d}"
        privileged = bool(row["privileged_claim_eligible"])
        add(
            run_id,
            stage="predictor",
            category="distillation",
            detail=str(row["row_id"]),
            parents=("ARCH_P_HIER_DECODER_REFINE",),
            selectable=True,
            family=(
                "privileged_scientific"
                if privileged
                else "pre_stage_g_deployable"
            ),
            clean_paired=row["consumer_id"] == "C_CLEAN",
            robust_paired=row["consumer_id"] == "C_ROBUST_MIX",
        )
        predictor_run_ids.append(run_id)

    for interaction_id in FOCUSED_INTERACTION_IDS:
        diagnostic = interaction_id.startswith("UNCENTERED_")
        run_id = f"INTERACTION_{interaction_id}"
        add(
            run_id,
            stage="predictor",
            category="focused_interaction",
            detail=interaction_id,
            parents=("ARCH_P_HIER_DECODER_REFINE",),
            selectable=not diagnostic,
            diagnostic=diagnostic,
            family="diagnostic" if diagnostic else "privileged_scientific",
        )
        predictor_run_ids.append(run_id)

    control_registry = build_step8_control_registry()
    controls_by_id = {
        row["control_id"]: row for row in control_registry["rows"]
    }
    for control_id in _PRE_STAGE_TRAINED_CONTROLS:
        row = controls_by_id[control_id]
        run_id = f"TRAINED_CONTROL_{control_id}"
        privileged = bool(row["privileged_claim_eligible"])
        selectable = bool(row["selectable"])
        add(
            run_id,
            stage="predictor",
            category="trained_control",
            detail=control_id,
            parents=("ROBUST_CONSUMER",),
            selectable=selectable,
            diagnostic=not selectable,
            family=(
                "privileged_scientific"
                if privileged
                else "pre_stage_g_deployable"
                if selectable
                else "diagnostic"
            ),
        )
        predictor_run_ids.append(run_id)

    for control_id in STRUCTURAL_CONTROL_IDS:
        add(
            f"STRUCTURAL_CONTROL_{control_id}",
            stage="confirmation",
            category="structural_control",
            detail=control_id,
            parents=tuple(predictor_run_ids),
            uses_labels=False,
            diagnostic=True,
            family="diagnostic",
        )

    confirmation_ids = []
    for role_id in CONFIRMATION_ROLE_IDS:
        privileged = role_id not in {
            "CE_ONLY_UPPER_BOUND",
            "DIRECT_PARAMETER_CONTROL",
            "DIRECT_FLOP_CONTROL",
            "HLT_MEMORY_CONTROL",
            "DVIEW_JOINT_CE_ONLY",
        }
        diagnostic = role_id in {
            "DIRECT_PARAMETER_CONTROL",
            "DIRECT_FLOP_CONTROL",
        }
        run_id = f"CONFIRM_{role_id}"
        add(
            run_id,
            stage="confirmation",
            category="confirmation_role",
            detail=role_id,
            parents=tuple(predictor_run_ids),
            seeds=PARTICLE_VIEW_SEEDS,
            selectable=not diagnostic,
            diagnostic=diagnostic,
            family=(
                "diagnostic"
                if diagnostic
                else "privileged_scientific"
                if privileged
                else "pre_stage_g_deployable"
            ),
            clean_paired=role_id == "CANONICAL_PREDECLARED",
            robust_paired=role_id != "CANONICAL_PREDECLARED",
        )
        confirmation_ids.append(run_id)

    add(
        "SELECT_WINNER_FAMILIES",
        stage="confirmation",
        category="winner_selection",
        detail="privileged_pre_stage_g_and_diagnostic",
        parents=tuple(confirmation_ids),
        uses_labels=False,
    )

    add(
        "SELECTED_PATH_FAIRNESS_LEDGER",
        stage="fairness",
        category="fairness_ledger",
        detail="two_winner_families",
        parents=("SELECT_WINNER_FAMILIES",),
        uses_labels=False,
    )
    fairness_run_ids = []
    for family_id in _WINNER_FAMILIES:
        for control_id in _FAIRNESS_CONTROL_IDS:
            run_id = f"FAIR_{family_id}_{control_id}"
            add(
                run_id,
                stage="fairness",
                category="fairness_control",
                detail=f"{family_id}/{control_id}",
                parents=("SELECTED_PATH_FAIRNESS_LEDGER",),
                seeds=PARTICLE_VIEW_SEEDS,
            )
            fairness_run_ids.append(run_id)

    for family_id in _WINNER_FAMILIES:
        add(
            f"STACK_WINNER_{family_id}",
            stage="stack",
            category="stack_winner",
            detail=family_id,
            parents=tuple(fairness_run_ids),
            seeds=PARTICLE_VIEW_SEEDS,
            uses_labels=False,
            stack=True,
        )
        for control_id in _FAIRNESS_CONTROL_IDS:
            add(
                f"STACK_FAIR_{family_id}_{control_id}",
                stage="stack",
                category="stack_fairness_control",
                detail=f"{family_id}/{control_id}",
                parents=(f"FAIR_{family_id}_{control_id}",),
                seeds=PARTICLE_VIEW_SEEDS,
                uses_labels=False,
                stack=True,
            )

    for stack_id in _STACK_STATIC_IDS:
        add(
            f"STACK_{stack_id}",
            stage="stack",
            category="stack_static",
            detail=stack_id,
            parents=tuple(fairness_run_ids),
            uses_labels=False,
            stack=True,
        )

    stack_parents = tuple(
        spec.run_id for spec in specs if spec.stage == "stack"
    )
    for report_id in _REPORT_EXPORT_IDS:
        add(
            f"REPORT_{report_id}",
            stage="report_export",
            category="report_export",
            detail=report_id,
            parents=stack_parents,
            uses_labels=False,
        )

    for family_id, family, permit_parent in (
        (
            "PRIVILEGED_SCIENTIFIC",
            "privileged_scientific",
            "REPORT_FINAL_PERMIT_PRIVILEGED",
        ),
        (
            "PRE_STAGE_G_DEPLOYABLE",
            "pre_stage_g_deployable",
            "REPORT_FINAL_PERMIT_DEPLOYABLE",
        ),
    ):
        add(
            f"FINAL_{family_id}",
            stage="final_test",
            category="final_test",
            detail=family_id,
            parents=(permit_parent,),
            uses_labels=False,
            selectable=True,
            family=family,
            final=True,
        )

    registry = build_particle_view_registry(
        unified_split_manifest_sha256=unified_split_manifest["content_hash"],
        train_identity_sha256=train_identity,
        run_specs=specs,
        campaign_id=campaign_id,
    )
    validate_low_data_campaign_registry(registry)
    return registry


def build_low_data_campaign_inventory(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize and authenticate the complete generated inventory."""

    audit = validate_particle_view_registry(registry)
    counts = Counter(_category(row["scientific_role"]) for row in registry["runs"])
    stage_counts = Counter(str(row["stage"]) for row in registry["runs"])
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_LOW_DATA_INVENTORY_CONTRACT,
            "registry_sha256": registry["content_hash"],
            "campaign_id": registry["campaign_id"],
            "category_counts": {
                key: counts[key] for key in sorted(counts)
            },
            "expected_category_counts": {
                key: EXPECTED_LOW_DATA_CATEGORY_COUNTS[key]
                for key in sorted(EXPECTED_LOW_DATA_CATEGORY_COUNTS)
            },
            "stage_counts": {
                key: stage_counts[key] for key in sorted(stage_counts)
            },
            "declared_run_count": audit["run_count"],
            "seed_expanded_replica_count": sum(
                len(row["seed_ids"]) for row in registry["runs"]
            ),
            "target_screen_count": len(TARGET_SCREEN_IDS),
            "consumer_screen_count": len(CONSUMER_SCREEN_IDS),
            "predictor_architecture_count": len(
                PARTICLE_VIEW_PREDICTOR_ARCHITECTURES
            ),
            "loss_interaction_count": len(_distillation_rows()),
            "structural_control_count": len(STRUCTURAL_CONTROL_IDS),
            "trained_control_count": len(TRAINED_CONTROL_IDS),
            "focused_interaction_count": len(FOCUSED_INTERACTION_IDS),
            "confirmation_role_count": len(CONFIRMATION_ROLE_IDS),
            "single_training_pool": audit["single_training_pool"],
            "quality_gates": False,
        }
    )


def validate_low_data_campaign_registry(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when any locked low-data run identity disappears."""

    audit = validate_particle_view_registry(registry)
    inventory = build_low_data_campaign_inventory(registry)
    if inventory["category_counts"] != inventory["expected_category_counts"]:
        raise ValueError(
            "low-data campaign category inventory mismatch: "
            f"actual={inventory['category_counts']} "
            f"expected={inventory['expected_category_counts']}"
        )
    if inventory["target_screen_count"] != 36:
        raise ValueError("low-data target-screen inventory changed")
    if inventory["consumer_screen_count"] != 12:
        raise ValueError("low-data consumer inventory changed")
    if inventory["predictor_architecture_count"] != 17:
        raise ValueError("low-data predictor inventory changed")
    if inventory["loss_interaction_count"] != 52:
        raise ValueError("low-data loss-interaction inventory changed")
    return {
        **audit,
        "inventory_sha256": inventory["content_hash"],
        "category_counts": inventory["category_counts"],
        "seed_expanded_replica_count": inventory[
            "seed_expanded_replica_count"
        ],
    }


__all__ = [
    "CONFIRMATION_ROLE_IDS",
    "CONSUMER_SCREEN_IDS",
    "EXPECTED_LOW_DATA_CATEGORY_COUNTS",
    "FOCUSED_INTERACTION_IDS",
    "NONSELECTABLE_TARGET_SCREEN_IDS",
    "PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID",
    "PARTICLE_VIEW_LOW_DATA_INVENTORY_CONTRACT",
    "TARGET_SCREEN_IDS",
    "build_low_data_campaign_inventory",
    "build_low_data_campaign_registry",
    "validate_low_data_campaign_registry",
]
