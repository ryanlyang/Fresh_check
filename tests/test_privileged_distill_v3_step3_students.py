from __future__ import annotations

from pathlib import Path

import pytest

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
)
from teacher_logit_reco.privileged_distill_v3 import (
    PDV3_LOSS_CE,
    PDV3_LOSS_LOGIT_KD,
    PDV3_LOSS_LOGIT_REP_KD,
    PDV3_STUDENT_DEFAULT_VARIANTS,
    PDV3_STUDENT_FAMILY_FEATURE_MLP,
    PDV3_STUDENT_FAMILY_HLT_PART,
    PDV3_STUDENT_FAMILY_LC_MLP_DELTA,
    PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP,
    PDV3_STUDENT_FEATURE_MLP_CE,
    PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD,
    PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
    PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START,
    PDV3_STUDENT_HLT_PART_CE,
    PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
    PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD,
    PDV3_STUDENT_LC_MLP_DELTA_CE,
    PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
    PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
    PDV3_STUDENT_REGISTRY_CONTRACT,
    PDV3_STUDENT_REGISTRY_STEP,
    PDV3_STUDENT_VARIANTS,
    PDV3_TEACHER_NONE,
    PDV3_TEACHER_V1_DUAL_VIEW,
    PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW,
    PDV3_V1_DUAL_VIEW_TEACHER_NAME,
    PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME,
    PDV3StudentVariantSpec,
    normalize_pdv3_student_variant,
    pdv3_student_registry_manifest,
    pdv3_student_variant_spec,
    pdv3_student_variant_specs,
    pdv3_student_variants,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


def test_step3_student_registry_contains_planned_matrix_in_order():
    assert PDV3_STUDENT_VARIANTS == (
        PDV3_STUDENT_HLT_PART_CE,
        PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
        PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD,
        PDV3_STUDENT_FEATURE_MLP_CE,
        PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD,
        PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
        PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START,
        PDV3_STUDENT_LC_MLP_DELTA_CE,
        PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
        PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START,
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT,
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED,
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT,
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
    )
    assert PDV3_STUDENT_DEFAULT_VARIANTS == PDV3_STUDENT_VARIANTS
    assert pdv3_student_variants() == PDV3_STUDENT_VARIANTS
    assert PDV3_STUDENT_HLT_PART_CE not in pdv3_student_variants(candidates_only=True)
    assert PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD in pdv3_student_variants(candidates_only=True)


def test_step3_aliases_resolve_to_canonical_student_variants():
    assert normalize_pdv3_student_variant("baseline") == PDV3_STUDENT_HLT_PART_CE
    assert normalize_pdv3_student_variant("feature-mlp-v2") == PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD
    assert normalize_pdv3_student_variant("best_expected") == PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED
    assert normalize_pdv3_student_variant("combined_v2_staged") == (
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED
    )
    assert normalize_pdv3_student_variant("input_delta_v2_frozen") == (
        PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START
    )
    with pytest.raises(ValueError, match="Unknown PDV3 student variant"):
        normalize_pdv3_student_variant("pdv3_not_a_real_variant")


def test_step3_specs_map_to_existing_av10_model_implementations():
    specs = pdv3_student_variant_specs()

    for variant in (
        PDV3_STUDENT_HLT_PART_CE,
        PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
        PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD,
    ):
        spec = specs[variant]
        assert spec.student_family == PDV3_STUDENT_FAMILY_HLT_PART
        assert spec.architecture_view_variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK
        assert spec.architecture_adapter_type == "none"
        assert spec.freeze_part_epochs == 0

    for variant in (
        PDV3_STUDENT_FEATURE_MLP_CE,
        PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD,
        PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
        PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START,
    ):
        spec = specs[variant]
        assert spec.student_family == PDV3_STUDENT_FAMILY_FEATURE_MLP
        assert spec.architecture_view_variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER
        assert spec.architecture_adapter_type == "feature_mlp_context"
        assert spec.freeze_part_epochs >= 1

    for variant in (
        PDV3_STUDENT_LC_MLP_DELTA_CE,
        PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
        PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START,
    ):
        spec = specs[variant]
        assert spec.student_family == PDV3_STUDENT_FAMILY_LC_MLP_DELTA
        assert spec.architecture_view_variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES
        assert spec.architecture_adapter_type == "feature_mlp_delta_F"
        assert spec.freeze_part_epochs >= 1

    for variant in (
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT,
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED,
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT,
        PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
    ):
        spec = specs[variant]
        assert spec.student_family == PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP
        assert spec.architecture_view_variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER
        assert spec.architecture_adapter_type == "feature_mlp_delta_F_plus_feature_mlp_context"
        assert spec.combined_adapter


def test_step3_specs_encode_ce_v1_and_v2_kd_requirements():
    ce = pdv3_student_variant_spec(PDV3_STUDENT_FEATURE_MLP_CE)
    assert ce.teacher_family == PDV3_TEACHER_NONE
    assert ce.loss_mode == PDV3_LOSS_CE
    assert ce.kd_alpha == 0.0
    assert ce.rep_beta == 0.0
    assert not ce.requires_teacher_logits
    assert not ce.requires_teacher_representations

    v1 = pdv3_student_variant_spec(PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD)
    assert v1.teacher_family == PDV3_TEACHER_V1_DUAL_VIEW
    assert v1.loss_mode == PDV3_LOSS_LOGIT_KD
    assert v1.teacher_logit_name == PDV3_V1_DUAL_VIEW_TEACHER_NAME
    assert v1.teacher_representation_name == ""
    assert v1.kd_temperature == 2.0
    assert v1.kd_alpha == 0.5
    assert v1.rep_beta == 0.0
    assert v1.requires_teacher_logits
    assert not v1.requires_teacher_representations

    v2 = pdv3_student_variant_spec(PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD)
    assert v2.teacher_family == PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW
    assert v2.loss_mode == PDV3_LOSS_LOGIT_REP_KD
    assert v2.teacher_logit_name == PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME
    assert v2.teacher_representation_name == PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME
    assert v2.kd_alpha == 0.5
    assert v2.rep_beta == 0.10
    assert v2.kd_warmup_epochs == 1
    assert v2.rep_warmup_epochs == 2
    assert v2.requires_teacher_logits
    assert v2.requires_teacher_representations


def test_step3_specs_track_expected_ranking_and_frozen_start_policy():
    best = pdv3_student_variant_spec(PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED)
    assert best.expected_rank == 1
    assert best.freeze_part_epochs == 4
    assert best.freeze_policy == "combined_staged"
    assert best.training_schedule == "staged"
    assert best.combined_adapter

    frozen_feature = pdv3_student_variant_spec(PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START)
    assert frozen_feature.expected_rank == 4
    assert frozen_feature.freeze_part_epochs == 2
    assert frozen_feature.freeze_policy == "frozen_start"

    frozen_lc = pdv3_student_variant_spec(PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START)
    assert frozen_lc.expected_rank == 5
    assert frozen_lc.freeze_part_epochs == 2
    assert frozen_lc.freeze_policy == "frozen_start"

    baseline = pdv3_student_variant_spec(PDV3_STUDENT_HLT_PART_CE)
    assert baseline.is_baseline
    assert not baseline.is_candidate
    assert baseline.expected_rank == 14


def test_step3_architecture_train_overrides_are_ready_for_step4_trainer():
    feature = pdv3_student_variant_spec(PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD)
    overrides = feature.architecture_train_overrides()
    assert overrides["variant"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER
    assert overrides["freeze_part_epochs"] == 1
    assert overrides["adapter_lr"] > overrides["part_lr"]
    assert overrides["weight_decay"] > 0.0

    hlt = pdv3_student_variant_spec(PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD)
    assert hlt.architecture_train_overrides()["variant"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK
    assert hlt.architecture_train_overrides()["freeze_part_epochs"] == 0


def test_step3_registry_manifest_is_serializable_and_complete():
    manifest = pdv3_student_registry_manifest()
    assert manifest["step"] == PDV3_STUDENT_REGISTRY_STEP
    assert manifest["contract"] == PDV3_STUDENT_REGISTRY_CONTRACT
    assert manifest["student_variants"] == list(PDV3_STUDENT_VARIANTS)
    assert manifest["default_student_variants"] == list(PDV3_STUDENT_DEFAULT_VARIANTS)
    assert manifest["teacher_names"]["v1_dual_view"] == PDV3_V1_DUAL_VIEW_TEACHER_NAME
    assert manifest["teacher_names"]["v2_particle_dual_view"] == PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME
    assert set(manifest["variants"]) == set(PDV3_STUDENT_VARIANTS)
    assert manifest["variants"][PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD]["requires_teacher_logits"]
    assert manifest["variants"][PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD]["requires_teacher_representations"]


def test_step3_spec_validation_rejects_inconsistent_kd_recipes():
    with pytest.raises(ValueError, match="CE-only variants must have zero KD weights"):
        PDV3StudentVariantSpec(
            name=PDV3_STUDENT_HLT_PART_CE,
            student_family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            kd_alpha=0.2,
        )
    with pytest.raises(ValueError, match="V1 dual-view variants require teacher_logit_name"):
        PDV3StudentVariantSpec(
            name=PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD,
            student_family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            teacher_family=PDV3_TEACHER_V1_DUAL_VIEW,
            loss_mode=PDV3_LOSS_LOGIT_KD,
            kd_alpha=0.5,
        )
    with pytest.raises(ValueError, match="V2 variants require both teacher logits and representations"):
        PDV3StudentVariantSpec(
            name=PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
            student_family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            teacher_family=PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW,
            loss_mode=PDV3_LOSS_LOGIT_REP_KD,
            teacher_logit_name=PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME,
            kd_alpha=0.5,
            rep_beta=0.1,
        )


def test_step3_sbatch_defaults_list_exact_student_matrix():
    common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")

    assert "PDV3_STUDENTS_DIR:=${PDV3_ROOT}/students" in common
    expected = " ".join(PDV3_STUDENT_VARIANTS)
    assert f"PDV3_STUDENT_VARIANTS:={expected}" in common
