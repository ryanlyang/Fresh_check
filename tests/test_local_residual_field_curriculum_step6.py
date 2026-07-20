from __future__ import annotations

import torch

from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    FIELD_GATE_MODE_LEARNED_SIGMOID,
    FIELD_GATE_MODE_NONE,
    FIELD_GATE_MODE_SUPERVISED_RELIABILITY,
    FIELD_GATE_MODE_UNCERTAINTY_INVERSE,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldCurriculumJointConfig,
    LocalResidualFieldCurriculumJointModel,
    LocalResidualFieldReconstructorConfig,
    LocalResidualFieldTaggerConfig,
    compute_confidence_gate_loss,
    confidence_reliability_target,
)


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
        self.proj = torch.nn.Conv1d(int(input_dim), int(num_classes), kernel_size=1)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        logits_per_particle = self.proj(features)
        weights = mask.to(dtype=features.dtype)
        return (logits_per_particle * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def _batch(batch: int = 2, particles: int = 5) -> dict[str, torch.Tensor]:
    tokens = torch.zeros(batch, particles, 14)
    raw_mask = torch.zeros(batch, particles, dtype=torch.bool)
    raw_mask[:, :3] = True
    tokens[:, :3, 0] = torch.tensor([10.0, 5.0, 2.0])
    tokens[:, :3, 1] = torch.tensor([0.0, 0.04, 0.08])
    tokens[:, :3, 2] = torch.tensor([0.0, 0.03, 0.10])
    tokens[:, :3, 3] = tokens[:, :3, 0] * 1.1
    tokens[:, :3, 5] = 1.0
    features = torch.randn(batch, len(PF_FEATURE_NAMES), particles)
    features = features * raw_mask[:, None, :].to(dtype=features.dtype)
    target_fields = torch.zeros(batch, particles, len(FIELD_NAMES))
    target_fields[:, :3, 0] = 0.25
    target_fields[:, :3, 1] = -0.10
    target_fields[:, :3, 2] = 0.50
    target_fields[:, :3, 3] = 1.00
    return {
        "points": torch.zeros(batch, 2, particles),
        "features": features,
        "lorentz_vectors": torch.zeros(batch, 4, particles),
        "mask": raw_mask[:, None, :],
        "tokens": tokens,
        "raw_mask": raw_mask,
        "indices": torch.arange(batch),
        "target_fields": target_fields,
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


def _student() -> LocalResidualFieldAugmentedParT:
    config = LocalResidualFieldTaggerConfig(
        num_classes=3,
        field_dim=len(FIELD_NAMES),
        field_source="zero",
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
    )
    return LocalResidualFieldAugmentedParT(
        config,
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )


def test_step6_joint_config_defaults_to_learned_gate_with_light_reliability_supervision():
    cfg = LocalResidualFieldCurriculumJointConfig(
        reconstructor_config=_reco_config(),
        student_config=_student().config,
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
    )

    assert cfg.field_gate_mode == FIELD_GATE_MODE_LEARNED_SIGMOID
    assert cfg.initial_gate_bias_prob == 0.1
    assert cfg.gate_reliability_loss_weight == 0.05


def test_step6_forward_emits_delta_log_var_gate_and_reliability_loss():
    model = LocalResidualFieldCurriculumJointModel(
        LocalResidualFieldCurriculumJointConfig(
            reconstructor_config=_reco_config(),
            student_config=_student().config,
            field_gate_mode=FIELD_GATE_MODE_SUPERVISED_RELIABILITY,
            initial_gate_bias_prob=0.2,
            gate_reliability_loss_weight=0.25,
            gate_reliability_error_scale=0.5,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        student=_student(),
    )
    batch = _batch()

    output = model(**batch)

    assert output.field_delta.shape == batch["target_fields"].shape
    assert output.field_log_var.shape == batch["target_fields"].shape
    assert output.field_gate.shape == batch["target_fields"].shape
    assert output.field_gate_loss is not None
    assert output.field_gate_loss.requires_grad is True
    assert output.field_reliability_target is not None
    assert output.diagnostics["confidence_heads"]["field_delta_head"] is True
    assert output.diagnostics["confidence_heads"]["field_log_var_head"] is True
    assert output.diagnostics["confidence_heads"]["field_gate_head"] is True
    assert output.diagnostics["gate_supervision"]["gate_supervision_enabled"] is True

    payload = model.deployable_checkpoint_payload()
    assert "confidence_heads_state_dict" in payload
    assert any(key.startswith("field_delta_head.") for key in payload["confidence_heads_state_dict"])
    assert any(key.startswith("field_log_var_head.") for key in payload["confidence_heads_state_dict"])
    assert any(key.startswith("field_gate_head.") for key in payload["confidence_heads_state_dict"])


def test_step6_uncertainty_inverse_gate_is_bounded_and_uses_log_var_head():
    model = LocalResidualFieldCurriculumJointModel(
        LocalResidualFieldCurriculumJointConfig(
            reconstructor_config=_reco_config(),
            student_config=_student().config,
            field_gate_mode=FIELD_GATE_MODE_UNCERTAINTY_INVERSE,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        student=_student(),
    )
    batch = _batch()

    output = model(**batch)
    valid_gate = output.field_gate[batch["raw_mask"]]

    assert torch.all(valid_gate >= 0.0)
    assert torch.all(valid_gate <= 1.0)
    assert output.field_log_var is not None
    assert output.field_uncertainty is not None
    assert output.diagnostics["field_gate_mode"] == FIELD_GATE_MODE_UNCERTAINTY_INVERSE


def test_step6_reliability_target_and_loss_are_masked():
    gate = torch.full((1, 3, 2), 0.5, requires_grad=True)
    pred = torch.tensor([[[0.0, 1.0], [0.5, 0.5], [99.0, 99.0]]])
    target = torch.tensor([[[0.0, 0.0], [1.5, 0.5], [0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])

    reliability = confidence_reliability_target(
        pred_fields=pred,
        target_fields=target,
        mask=mask,
        error_scale=1.0,
    )
    loss, target_reliability, diagnostics = compute_confidence_gate_loss(
        gate=gate,
        pred_fields=pred,
        target_fields=target,
        mask=mask,
        mode=FIELD_GATE_MODE_SUPERVISED_RELIABILITY,
        loss_weight=0.5,
        error_scale=1.0,
    )

    assert torch.allclose(reliability[:, 2, :], torch.zeros_like(reliability[:, 2, :]))
    assert target_reliability is not None
    assert loss is not None
    loss.backward()
    assert gate.grad is not None
    assert torch.allclose(gate.grad[:, 2, :], torch.zeros_like(gate.grad[:, 2, :]))
    assert diagnostics["valid_gate_values"] == 4


def test_step6_reliability_is_groupwise_detached_and_uses_equal_group_mse():
    pred = torch.tensor([[[0.0, 2.0, 4.0]]], requires_grad=True)
    target = torch.zeros_like(pred)
    gate = torch.tensor([[[0.2, 0.6, 0.5]]], requires_grad=True)
    mask = torch.tensor([[True]])
    groups = {"paired": [0, 1], "single": [2]}

    reliability = confidence_reliability_target(
        pred_fields=pred,
        target_fields=target,
        mask=mask,
        error_scale=1.0,
        field_groups=groups,
    )
    loss, loss_target, diagnostics = compute_confidence_gate_loss(
        gate=gate,
        pred_fields=pred,
        target_fields=target,
        mask=mask,
        mode=FIELD_GATE_MODE_SUPERVISED_RELIABILITY,
        loss_weight=0.25,
        error_scale=1.0,
        field_groups=groups,
    )

    expected_paired = torch.exp(torch.tensor(-1.0))
    expected_single = torch.exp(torch.tensor(-4.0))
    expected_unweighted = (((torch.tensor(0.4) - expected_paired) ** 2) + ((torch.tensor(0.5) - expected_single) ** 2)) / 2.0
    assert torch.allclose(reliability[0, 0, :2], expected_paired.expand(2))
    assert torch.allclose(reliability[0, 0, 2], expected_single)
    assert reliability.requires_grad is False
    assert loss_target is not None and loss_target.requires_grad is False
    assert loss is not None
    assert torch.allclose(loss, expected_unweighted * 0.25)
    assert diagnostics["gate_loss_type"] == "mse"
    assert diagnostics["gate_group_count"] == 2
    assert diagnostics["valid_gate_values"] == 2

    loss.backward()
    assert pred.grad is None
    assert gate.grad is not None


def test_step6_learned_gate_is_tied_within_field_groups():
    model = LocalResidualFieldCurriculumJointModel(
        LocalResidualFieldCurriculumJointConfig(
            reconstructor_config=_reco_config(),
            student_config=_student().config,
            field_gate_mode=FIELD_GATE_MODE_LEARNED_SIGMOID,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        student=_student(),
    )
    with torch.no_grad():
        model.confidence_heads.field_gate_head[-1].weight.zero_()
        model.confidence_heads.field_gate_head[-1].bias.copy_(torch.tensor([-3.0, 1.0, 2.0, -2.0]))

    batch = _batch()
    output = model(**batch)

    assert torch.allclose(output.field_gate[..., 0], output.field_gate[..., 1])
    assert torch.allclose(
        output.field_gate[batch["raw_mask"]][:, 0],
        torch.sigmoid(torch.tensor(-1.0)).expand(output.field_gate[batch["raw_mask"]].shape[0]),
    )


def test_step6_none_gate_is_identity_and_disables_gate_supervision():
    model = LocalResidualFieldCurriculumJointModel(
        LocalResidualFieldCurriculumJointConfig(
            reconstructor_config=_reco_config(),
            student_config=_student().config,
            field_gate_mode=FIELD_GATE_MODE_NONE,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        student=_student(),
    )
    batch = _batch()

    output = model(**batch)

    assert torch.allclose(output.field_gate[batch["raw_mask"]], torch.ones_like(output.field_gate[batch["raw_mask"]]))
    assert torch.allclose(output.field_gate[~batch["raw_mask"]], torch.zeros_like(output.field_gate[~batch["raw_mask"]]))
    assert output.field_gate_loss is None
    assert output.field_reliability_target is None
    assert output.diagnostics["gate_supervision"]["gate_supervision_enabled"] is False


def test_step6_effective_field_is_clipped_before_gate_application():
    model = LocalResidualFieldCurriculumJointModel(
        LocalResidualFieldCurriculumJointConfig(
            reconstructor_config=_reco_config(),
            student_config=_student().config,
            field_gate_mode=FIELD_GATE_MODE_LEARNED_SIGMOID,
            initial_gate_bias_prob=0.5,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        student=_student(),
    )
    with torch.no_grad():
        model.confidence_heads.field_delta_head[-1].weight.zero_()
        model.confidence_heads.field_delta_head[-1].bias.fill_(100.0)
        model.confidence_heads.field_gate_head[-1].weight.zero_()
        model.confidence_heads.field_gate_head[-1].bias.zero_()

    batch = _batch()
    output = model(**batch)

    valid_effective = output.pred_fields_effective[batch["raw_mask"]]
    assert torch.allclose(valid_effective, torch.full_like(valid_effective, 4.0))
    assert output.diagnostics["pred_fields_clip_value"] == 8.0
    assert output.diagnostics["pred_fields_clipped_value_count"] == int(batch["raw_mask"].sum()) * len(FIELD_NAMES)
