from __future__ import annotations

import json
from pathlib import Path

import torch

from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    FIELD_GATE_MODE_LEARNED_SIGMOID,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldCurriculumJointConfig,
    LocalResidualFieldCurriculumJointModel,
    LocalResidualFieldReconstructorConfig,
    LocalResidualFieldTaggerConfig,
    reset_or_scale_student_residual_projection,
)
from teacher_logit_reco.local_particle_residual_field.oracle import (
    FrozenLocalResidualFieldOracleConsumer,
    FrozenOracleConsumerConfig,
)
import teacher_logit_reco.local_particle_residual_field.tagger as tagger_module


FIELD_NAMES = (
    "r0p02.delta_log_pt_sum",
    "r0p02.delta_pt_frac",
    "r0p02.delta_log_n",
    "flag.is_merged_token",
)
FIELD_GROUPS = {
    "pt_density": [0, 1],
    "multiplicity": [2],
    "reliability": [3],
}


class FakePart(torch.nn.Module):
    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.config = {"input_dim": int(input_dim), "num_classes": int(num_classes)}
        self.proj = torch.nn.Conv1d(int(input_dim), int(num_classes), kernel_size=1)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        logits_per_particle = self.proj(features)
        weights = mask.to(dtype=features.dtype)
        return (logits_per_particle * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def fake_build_hlt_classifier(*, num_classes: int, model_size: str = "base", overrides=None):
    del model_size
    input_dim = int((overrides or {}).get("input_dim", len(PF_FEATURE_NAMES)))
    return FakePart(input_dim=input_dim, num_classes=int(num_classes))


def _batch(batch: int = 2, particles: int = 5) -> dict[str, torch.Tensor]:
    tokens = torch.zeros(batch, particles, 14)
    raw_mask = torch.zeros(batch, particles, dtype=torch.bool)
    raw_mask[:, :3] = True
    for jet in range(batch):
        tokens[jet, :3, 0] = torch.tensor([10.0 + jet, 5.0, 2.0])
        tokens[jet, :3, 1] = torch.tensor([0.0, 0.04, 0.08])
        tokens[jet, :3, 2] = torch.tensor([0.0, 0.03, 0.10])
        tokens[jet, :3, 3] = tokens[jet, :3, 0] * 1.1
        tokens[jet, :3, 5] = 1.0
    features = torch.randn(batch, len(PF_FEATURE_NAMES), particles)
    features = features * raw_mask[:, None, :].to(dtype=features.dtype)
    return {
        "points": torch.zeros(batch, 2, particles),
        "features": features,
        "lorentz_vectors": torch.zeros(batch, 4, particles),
        "mask": raw_mask[:, None, :],
        "tokens": tokens,
        "raw_mask": raw_mask,
        "indices": torch.arange(batch),
        "target_fields": torch.randn(batch, particles, len(FIELD_NAMES)) * raw_mask[:, :, None],
    }


def _reco_config() -> LocalResidualFieldReconstructorConfig:
    return LocalResidualFieldReconstructorConfig(
        field_dim=len(FIELD_NAMES),
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        d_model=24,
        num_heads=4,
        num_layers=1,
        context_layers=1,
        dropout=0.0,
        attention_dropout=0.0,
        max_particles=8,
    )


def _student_config() -> LocalResidualFieldTaggerConfig:
    return LocalResidualFieldTaggerConfig(
        num_classes=3,
        field_dim=len(FIELD_NAMES),
        field_source="zero",
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
    )


def _student() -> LocalResidualFieldAugmentedParT:
    config = _student_config()
    return LocalResidualFieldAugmentedParT(
        config,
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )


def _write_fake_oracle_checkpoint(path: Path) -> None:
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source="oracle_scaled",
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "config": {"train_split": "model_train"},
            "metrics": {"accuracy": 0.74},
            "selected_field_indices": list(range(len(FIELD_NAMES))),
            "selected_field_names": list(FIELD_NAMES),
        },
        path,
    )
    (path.parent / "teacher_config.json").write_text(
        json.dumps(
            {
                "contract": "local_residual_field_oracle_teacher_config_v1",
                "teacher_id": "Ofull",
                "field_source": "oracle_scaled",
                "oracle_field_alpha": 1.0,
                "selected_field_indices": list(range(len(FIELD_NAMES))),
                "selected_field_names": list(FIELD_NAMES),
            }
        ),
        encoding="utf-8",
    )


def test_step4_joint_model_returns_planned_outputs_and_oracle_free_payload():
    cfg = LocalResidualFieldCurriculumJointConfig(
        reconstructor_config=_reco_config(),
        student_config=_student_config(),
        field_gate_mode=FIELD_GATE_MODE_LEARNED_SIGMOID,
        initial_gate_bias_prob=0.25,
        residual_projection_reset="scale",
        residual_projection_scale=0.1,
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        normalization_metadata={"target_mean": [0.0] * len(FIELD_NAMES)},
        provenance_hashes={"target_content_hash": "abc"},
    )
    model = LocalResidualFieldCurriculumJointModel(cfg, student=_student())
    batch = _batch()

    output = model(**batch)

    assert output.pred_fields_raw.shape == batch["target_fields"].shape
    assert output.pred_fields_effective.shape == batch["target_fields"].shape
    assert output.field_gate.shape == batch["target_fields"].shape
    assert output.student_logits.shape == (2, 3)
    assert output.oracle_pred_logits is None
    assert output.diagnostics["field_gate_mode"] == "learned_sigmoid"
    valid_gate = output.field_gate[batch["raw_mask"]]
    assert torch.allclose(valid_gate, torch.full_like(valid_gate, 0.25), atol=1.0e-5)

    payload = model.deployable_checkpoint_payload(extra_metadata={"run_id": "P7a"})
    assert payload["oracle_consumer_included"] is False
    assert payload["model_config"]["oracle_consumer_config"] is None
    assert "oracle" not in payload
    assert payload["field_names"] == list(FIELD_NAMES)
    assert payload["metadata"]["run_id"] == "P7a"


def test_step4_joint_model_routes_predicted_fields_through_frozen_oracle_consumer(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    checkpoint = tmp_path / "Ofull" / "best_model_val.pt"
    _write_fake_oracle_checkpoint(checkpoint)
    consumer = FrozenLocalResidualFieldOracleConsumer(
        FrozenOracleConsumerConfig(checkpoint=str(checkpoint), consumer_id="Ofull", alpha=0.5),
        device="cpu",
    )
    model = LocalResidualFieldCurriculumJointModel(
        LocalResidualFieldCurriculumJointConfig(
            reconstructor_config=_reco_config(),
            student_config=_student_config(),
            oracle_consumer_config={"checkpoint": str(checkpoint), "consumer_id": "Ofull"},
            field_gate_mode="none",
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        student=_student(),
        oracle_consumer=consumer,
    )
    batch = _batch()

    output = model(**batch, oracle_alpha=0.5)

    assert output.oracle_true_logits is not None
    assert output.oracle_pred_logits is not None
    assert output.oracle_true_logits.requires_grad is False
    assert output.oracle_pred_logits.requires_grad is True
    output.oracle_pred_logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.reconstructor.parameters())
    assert output.diagnostics["oracle_consumer"]["consumer_id"] == "Ofull"
    assert output.diagnostics["oracle_consumer"]["alpha"] == 0.5


def test_step4_residual_projection_reset_edits_only_residual_input_columns():
    student = _student()
    base_dim = int(student.config.base_feature_dim)
    augmented_dim = int(student.config.augmented_feature_dim)
    with torch.no_grad():
        student.part_model.proj.weight.fill_(1.0)

    report = reset_or_scale_student_residual_projection(student, mode="scale", scale=0.2)

    assert report["matched_parameter_count"] == 1
    weight = student.part_model.proj.weight.detach()
    assert torch.allclose(weight[:, :base_dim, :], torch.ones_like(weight[:, :base_dim, :]))
    assert torch.allclose(weight[:, base_dim:augmented_dim, :], torch.full_like(weight[:, base_dim:augmented_dim, :], 0.2))

    report = reset_or_scale_student_residual_projection(student, mode="reset")

    assert report["matched_parameter_count"] == 1
    weight = student.part_model.proj.weight.detach()
    assert torch.allclose(weight[:, base_dim:augmented_dim, :], torch.zeros_like(weight[:, base_dim:augmented_dim, :]))


def test_step4_freeze_phase_reports_trainable_groups():
    model = LocalResidualFieldCurriculumJointModel(
        LocalResidualFieldCurriculumJointConfig(
            reconstructor_config=_reco_config(),
            student_config=_student_config(),
            field_gate_mode="learned_sigmoid",
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        student=_student(),
    )

    phase1 = model.apply_freeze_phase("residual_path_warmup")

    assert phase1["phase"] == "residual_path_warmup"
    assert phase1["reconstructor"]["trainable"] > 0
    assert phase1["field_gate"]["trainable"] > 0
    assert 0 < phase1["student"]["trainable"] < phase1["student"]["total"]

    phase3 = model.apply_freeze_phase("full_gentle_unfreeze")

    assert phase3["phase"] == "full_gentle_unfreeze"
    assert phase3["student"]["trainable"] == phase3["student"]["total"]
