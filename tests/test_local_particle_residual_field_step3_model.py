from __future__ import annotations

import pytest
import torch

from teacher_logit_reco.local_particle_residual_field import (
    LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS,
    RECONSTRUCTOR_VARIANT_C2_NO_GEOMETRY,
    RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY,
    RECONSTRUCTOR_VARIANT_C6_CONSISTENCY,
    LocalResidualFieldReconstructorConfig,
    build_local_residual_field_reconstructor,
    normalize_local_residual_reconstructor_variant,
)


def _tokens(batch: int = 2, particles: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.zeros(batch, particles, 14)
    mask = torch.zeros(batch, particles, dtype=torch.bool)
    mask[:, :3] = True
    tokens[:, :3, 0] = torch.tensor([[10.0, 5.0, 2.0], [8.0, 4.0, 1.0]])[:batch]
    tokens[:, :3, 1] = torch.tensor([[0.0, 0.04, 0.08], [0.1, 0.2, 0.25]])[:batch]
    tokens[:, :3, 2] = torch.tensor([[0.0, 0.03, 0.10], [0.1, 0.18, 0.26]])[:batch]
    tokens[:, :3, 3] = tokens[:, :3, 0] * 1.1
    tokens[:, :3, 5] = 1.0
    return tokens, mask


@pytest.mark.parametrize("variant", LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS)
def test_step3_reconstructor_variants_return_masked_field_tensors(variant: str):
    config = LocalResidualFieldReconstructorConfig(
        variant=variant,
        field_dim=12,
        d_model=32,
        num_heads=4,
        num_layers=2,
        context_dim=7,
        field_groups={"pt": [0, 1, 2, 3], "other": list(range(4, 12))},
    )
    model = build_local_residual_field_reconstructor(config)
    tokens, mask = _tokens()
    context = torch.randn(tokens.shape[0], 3, 7)
    context_mask = torch.ones(tokens.shape[0], 3, dtype=torch.bool)

    output = model(tokens, mask, context_tokens=context, context_mask=context_mask)

    assert output.predicted_fields.shape == (2, 5, 12)
    assert output.field_mask.shape == (2, 5)
    assert output.hidden.shape == (2, 5, 32)
    assert torch.isfinite(output.predicted_fields).all()
    assert torch.all(output.predicted_fields[:, 3:, :] == 0.0)
    assert output.diagnostics["variant"] == variant


def test_step3_zero_init_starts_as_no_correction():
    config = LocalResidualFieldReconstructorConfig(field_dim=6, d_model=24, num_heads=3, num_layers=1)
    model = build_local_residual_field_reconstructor(config)
    tokens, mask = _tokens()

    output = model(tokens, mask)

    assert torch.allclose(output.predicted_fields, torch.zeros_like(output.predicted_fields))


def test_step3_uncertainty_variant_returns_log_sigma():
    config = LocalResidualFieldReconstructorConfig(
        variant=RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY,
        field_dim=8,
        d_model=32,
        num_heads=4,
        num_layers=1,
    )
    model = build_local_residual_field_reconstructor(config)
    tokens, mask = _tokens()

    output = model(tokens, mask)

    assert output.log_sigma is not None
    assert output.log_sigma.shape == output.predicted_fields.shape
    assert output.diagnostics["log_sigma_present"] is True


def test_step3_no_geometry_variant_disables_geometry_bias():
    config = LocalResidualFieldReconstructorConfig(
        variant=RECONSTRUCTOR_VARIANT_C2_NO_GEOMETRY,
        field_dim=8,
        d_model=32,
        num_heads=4,
        num_layers=1,
    )
    model = build_local_residual_field_reconstructor(config)
    tokens, mask = _tokens()

    output = model(tokens, mask)

    assert output.diagnostics["geometry_bias_enabled"] is False


def test_step3_consistency_variant_reports_global_prediction():
    config = LocalResidualFieldReconstructorConfig(
        variant=RECONSTRUCTOR_VARIANT_C6_CONSISTENCY,
        field_dim=8,
        d_model=32,
        num_heads=4,
        num_layers=1,
    )
    model = build_local_residual_field_reconstructor(config)
    tokens, mask = _tokens()

    output = model(tokens, mask)

    assert "global_consistency_prediction" in output.diagnostics
    assert output.diagnostics["global_consistency_prediction"].shape == (2, 4)


def test_step3_variant_aliases_are_supported():
    assert normalize_local_residual_reconstructor_variant("C0") in LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS
    assert normalize_local_residual_reconstructor_variant("uncertainty") == RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY
    with pytest.raises(ValueError):
        normalize_local_residual_reconstructor_variant("not_a_variant")
