from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier
from teacher_logit_reco.local_compression_part import LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES
from teacher_logit_reco.privileged_distill_v3 import (
    PDV3_HLT_DEGRADATION_STRENGTH,
    PDV3_HLT_PROFILE,
    PDV3_LABEL_FILTER,
    PDV3_LABEL_NAMES,
    PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
    PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START,
    PDV3_STUDENT_HLT_PART_CE,
    PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
    PDV3_STUDENT_TRAIN_CONTRACT,
    PDV3_TEACHER_NONE,
    PDV3_TEACHER_V1_DUAL_VIEW,
    PDV3StudentTrainConfig,
    apply_pdv3_student_training_phase,
    pdv3_effective_weight,
    pdv3_student_training_phase_plan,
    pdv3_student_representation_source_from_output,
)
from teacher_logit_reco.privileged_distill_v3.train import _pdv3_expected_hlt_params, _verify_pdv3_hlt_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


class _PhaseDummyPart(ParticleTransformerHLTClassifier):
    def __init__(self, embed_dim: int = 128, num_classes: int = 10) -> None:
        torch = pytest.importorskip("torch")
        torch.nn.Module.__init__(self)
        self.config = {"dummy_pdv3_phase_part": True, "embed_dim": int(embed_dim), "num_classes": int(num_classes)}
        self.mod = torch.nn.Module()
        self.mod.embed = torch.nn.Linear(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(embed_dim))
        self.mod.fc = torch.nn.Linear(int(embed_dim), int(num_classes))

    def no_weight_decay(self):
        return set()

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        rows = features.transpose(1, 2).contiguous()
        embedded = self.mod.embed(rows)
        particle_mask = mask.squeeze(1).to(dtype=embedded.dtype)
        pooled = (embedded * particle_mask[:, :, None]).sum(dim=1)
        pooled = pooled / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.mod.fc(pooled)


def _base_config(**overrides) -> PDV3StudentTrainConfig:
    payload = {
        "output_dir": "out",
        "manifest_path": "manifest.json.gz",
        "hlt_cache_dir": "hlt_cache",
        "baseline_checkpoint": "baseline.pt",
        "student_variant": PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
        "teacher_logit_root": "teacher_logits",
        "teacher_representation_root": "teacher_reps",
        "confirm_split_settings": True,
        "confirm_final_test": True,
        "max_train_jets": 32,
        "max_val_jets": 16,
        "max_final_test_jets": 16,
    }
    payload.update(overrides)
    return PDV3StudentTrainConfig(**payload)


def test_step4_train_config_maps_student_spec_to_av10_config():
    config = _base_config()
    arch = config.architecture_config()

    assert config.student_variant == PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD
    assert config.teacher_target == "particle_dual_view"
    assert arch.variant == config.spec.architecture_view_variant
    assert arch.freeze_part_epochs == config.spec.freeze_part_epochs
    assert arch.adapter_lr == config.spec.adapter_lr
    assert arch.part_lr == config.spec.part_lr
    assert arch.label_names == PDV3_LABEL_NAMES
    assert arch.label_filter == PDV3_LABEL_FILTER
    assert arch.num_classes == 10
    assert config.expected_hlt_profile == PDV3_HLT_PROFILE
    assert arch.expected_hlt_degradation_strength == PDV3_HLT_DEGRADATION_STRENGTH


def test_step4_hlt_contract_verifier_rejects_wrong_profile():
    metadata = {
        "hlt_profile": PDV3_HLT_PROFILE,
        "hlt_degradation_strength": PDV3_HLT_DEGRADATION_STRENGTH,
        "hlt_params": _pdv3_expected_hlt_params(PDV3_HLT_PROFILE, PDV3_HLT_DEGRADATION_STRENGTH),
    }
    ok = _verify_pdv3_hlt_metadata(
        metadata,
        split="model_train",
        expected_profile=PDV3_HLT_PROFILE,
        expected_strength=PDV3_HLT_DEGRADATION_STRENGTH,
        required=True,
    )
    assert ok["ok"]

    wrong_profile = "fixed_hlt_v2_realistic" if PDV3_HLT_PROFILE != "fixed_hlt_v2_realistic" else "fixed_hlt_v1"
    bad = {**metadata, "hlt_profile": wrong_profile}
    with pytest.raises(ValueError, match="HLT profile"):
        _verify_pdv3_hlt_metadata(
            bad,
            split="model_train",
            expected_profile=PDV3_HLT_PROFILE,
            expected_strength=PDV3_HLT_DEGRADATION_STRENGTH,
            required=True,
        )


def test_step4_ce_and_v1_configs_require_the_right_teacher_payloads():
    ce = _base_config(
        student_variant=PDV3_STUDENT_HLT_PART_CE,
        teacher_logit_root="",
        teacher_representation_root="",
    )
    assert ce.spec.teacher_family == PDV3_TEACHER_NONE
    assert ce.teacher_logit_cache_root == ""
    assert ce.teacher_representation_cache_root == ""

    v1 = _base_config(
        student_variant=PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
        teacher_representation_root="",
    )
    assert v1.spec.teacher_family == PDV3_TEACHER_V1_DUAL_VIEW
    assert v1.teacher_target == "dual_view"
    assert v1.teacher_logit_cache_root == "teacher_logits"
    assert v1.teacher_representation_cache_root == ""

    with pytest.raises(ValueError, match="requires teacher_logit_root"):
        _base_config(
            student_variant=PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
            teacher_logit_root="",
            teacher_representation_root="",
        ).teacher_logit_cache_root


def test_step4_baseline_can_train_from_scratch_when_checkpoint_is_missing():
    ce = _base_config(
        student_variant=PDV3_STUDENT_HLT_PART_CE,
        baseline_checkpoint="",
        teacher_logit_root="",
        teacher_representation_root="",
    )

    assert ce.trains_baseline_from_scratch
    assert ce.part_lr_value == 1.0e-3

    kd = _base_config(
        baseline_checkpoint="",
        student_variant=PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
    )
    assert not kd.trains_baseline_from_scratch


def test_step4_v2_can_disable_one_teacher_signal_for_diagnostic_modes():
    logit_only = _base_config(use_teacher_representations=False)
    assert logit_only.uses_teacher_logits
    assert not logit_only.uses_teacher_representations
    assert logit_only.teacher_logit_cache_root == "teacher_logits"
    assert logit_only.teacher_representation_cache_root == ""

    rep_only = _base_config(use_teacher_logits=False)
    assert not rep_only.uses_teacher_logits
    assert rep_only.uses_teacher_representations
    assert rep_only.teacher_logit_cache_root == ""
    assert rep_only.teacher_representation_cache_root == "teacher_reps"

    with pytest.raises(ValueError, match="at least one teacher signal"):
        _base_config(use_teacher_logits=False, use_teacher_representations=False)


def test_step4_kd_weight_warmup_is_linear_and_clamped():
    assert pdv3_effective_weight(0.5, 0, 1) == 0.5
    assert pdv3_effective_weight(0.5, 2, 1) == 0.25
    assert pdv3_effective_weight(0.5, 2, 2) == 0.5
    assert pdv3_effective_weight(0.5, 2, 5) == 0.5
    assert pdv3_effective_weight(0.0, 2, 1) == 0.0


def test_step4_combined_adapter_uses_staged_train_phase_schedule():
    torch = pytest.importorskip("torch")
    from teacher_logit_reco.architecture_view_part import ArchitectureViewResidualParT

    config = _base_config(student_variant=PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED)
    arch = config.architecture_config()
    model = ArchitectureViewResidualParT(
        arch.model_config(),
        variant=arch.variant,
        part_model=_PhaseDummyPart(embed_dim=config.part_embed_dim),
    )
    projector = torch.nn.Linear(config.part_embed_dim, config.representation_dim)

    plan = pdv3_student_training_phase_plan(config.spec)
    assert [phase["phase_name"] for phase in plan] == [
        "input_repair_warmup",
        "embedding_residual_warmup",
        "joint_finetune",
    ]

    phase1 = apply_pdv3_student_training_phase(model, projector, config.spec, epoch=1)
    assert phase1["phase_name"] == "input_repair_warmup"
    assert not phase1["part_model_trainable"]
    assert not phase1["classifier_head_trainable"]
    assert phase1["module_group_trainability"]["input_delta_adapter"]
    assert not phase1["module_group_trainability"]["embedding_delta_adapter"]
    assert phase1["trainable_adapter_module_names"] == ["feature_delta_adapter"]
    assert not any(param.requires_grad for param in model.context_control.parameters())

    phase3 = apply_pdv3_student_training_phase(model, projector, config.spec, epoch=3)
    assert phase3["phase_name"] == "embedding_residual_warmup"
    assert not phase3["part_model_trainable"]
    assert phase3["module_group_trainability"]["embedding_delta_adapter"]
    assert not phase3["module_group_trainability"]["input_delta_adapter"]
    assert phase3["trainable_adapter_module_names"] == ["context_control", "context_control_gate"]
    assert not any(param.requires_grad for param in model.feature_delta_adapter.parameters())

    phase5 = apply_pdv3_student_training_phase(model, projector, config.spec, epoch=5)
    assert phase5["phase_name"] == "joint_finetune"
    assert phase5["part_model_trainable"]
    assert phase5["classifier_head_trainable"]
    assert phase5["module_group_trainability"]["input_delta_adapter"]
    assert phase5["module_group_trainability"]["embedding_delta_adapter"]
    assert phase5["trainable_adapter_module_names"] == [
        "feature_delta_adapter",
        "context_control",
        "context_control_gate",
    ]
    assert any(param.requires_grad for param in model.part_model.parameters())


def test_step4_noncombined_frozen_start_trains_adapter_and_classifier_head():
    torch = pytest.importorskip("torch")
    from teacher_logit_reco.architecture_view_part import ArchitectureViewResidualParT

    config = _base_config(student_variant=PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START)
    arch = config.architecture_config()
    model = ArchitectureViewResidualParT(
        arch.model_config(),
        variant=arch.variant,
        part_model=_PhaseDummyPart(embed_dim=config.part_embed_dim),
    )
    projector = torch.nn.Linear(config.part_embed_dim, config.representation_dim)

    phase1 = apply_pdv3_student_training_phase(model, projector, config.spec, epoch=1)

    assert phase1["phase_name"] == "adapter_warmup"
    assert not phase1["part_model_trainable"]
    assert phase1["classifier_head_trainable"]
    assert phase1["module_group_trainability"]["classifier_head"]
    assert phase1["trainable_adapter_module_names"] == ["context_control", "context_control_gate"]
    assert not any(param.requires_grad for param in model.part_model.mod.embed.parameters())
    assert any(param.requires_grad for param in model.part_model.mod.fc.parameters())


def test_step4_representation_source_uses_pooled_part_embedding():
    torch = pytest.importorskip("torch")
    part_embedding = torch.randn(3, 5, 11)
    mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    canonical = SimpleNamespace(particle_mask=mask)
    output = SimpleNamespace(canonical_inputs=canonical, part_embedding=part_embedding)

    source = pdv3_student_representation_source_from_output(output)

    assert tuple(source.shape) == (3, 11)
    assert torch.allclose(source[0], part_embedding[0, :2].mean(dim=0))
    assert torch.allclose(source[1], part_embedding[1, :3].mean(dim=0))
    assert torch.allclose(source[2], part_embedding[2, :1].mean(dim=0))


def test_step4_training_contract_and_runner_are_exposed():
    runner = (SBATCH_DIR / "run_pdv3_train_student.sh").read_text(encoding="utf-8")
    common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")

    assert PDV3_STUDENT_TRAIN_CONTRACT == "pdv3_av10_adapter_student_kd_train_v1"
    assert "scripts/train_pdv3_student.py" in runner
    assert "--student-variant" in runner
    assert "--teacher-logit-root" in runner
    assert "--teacher-representation-root" in runner
    assert "--expected-hlt-profile" in runner
    assert "--require-baseline-split-manifest-hash" in runner
    assert "--disable-baseline-from-scratch" in runner
    assert "--final-test-teacher-diagnostics" in runner
    assert "PDV3_STUDENT_BASELINE_CHECKPOINT" in common
    assert "PDV3_STUDENT_REPRESENTATION_DIM:=128" in common
    assert "PDV3_STUDENT_ALLOW_BASELINE_FROM_SCRATCH:=1" in common
    assert "PDV3_STUDENT_FINAL_TEST_TEACHER_DIAGNOSTICS:=0" in common

    submitter = (SBATCH_DIR / "submit_pdv3_full_experiment.sh").read_text(encoding="utf-8")
    assert "pdv3_student_output_complete" in submitter
    assert "archive_incomplete_student_output" in submitter
    assert "found incomplete existing PDV3 student output" in submitter
    assert 'found complete existing output: ${output_dir}' in submitter
