import pytest
import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_PREDICTOR_CONTRACT,
    CANONICAL_STATE_PREDICTOR_VARIANTS,
    PREDICTOR_VARIANT_DEEPSETS,
    PREDICTOR_VARIANT_GEOMETRY_BIASED,
    PREDICTOR_VARIANT_HARD_LOCALITY,
    PREDICTOR_VARIANT_NO_GEOMETRY_BIAS,
    PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION,
    PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES,
    PREDICTOR_VARIANT_STATE_ONLY,
    PREDICTOR_VARIANT_UNCERTAINTY,
    CanonicalStateResidualPredictorConfig,
    build_canonical_state_residual_predictor,
    default_canonical_jet_state_layout,
    normalize_predictor_variant,
)


def _small_config(variant: str) -> dict[str, object]:
    return {
        "variant": variant,
        "d_model": 32,
        "num_heads": 4,
        "particle_encoder_layers": 1,
        "decoder_layers": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "max_particles": 16,
        "max_slots": 16,
    }


def _sample_batch(batch_size: int = 2, n_particles: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = default_canonical_jet_state_layout()
    particles = torch.zeros((batch_size, n_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    particles[..., 0] = torch.tensor([0.42, 0.31, 0.18, 0.08, 0.04, 0.02])[:n_particles]
    particles[..., 1] = torch.linspace(-0.35, 0.35, n_particles)
    particles[..., 2] = torch.linspace(-0.4, 0.4, n_particles)
    particles[..., 3] = torch.log1p(particles[..., 0])
    particles[..., 4] = particles[..., 0] * 1.2
    mask = torch.ones((batch_size, n_particles), dtype=torch.bool)
    mask[0, -1] = False
    particles[0, -1, 0] = 0.0
    phi_hlt = torch.randn((batch_size, layout.k_state, layout.d_phi), dtype=torch.float32) * 0.03
    state_mask = torch.ones((batch_size, layout.k_state), dtype=torch.bool)
    state_mask[1, -2:] = False
    return particles, mask, phi_hlt, state_mask


@pytest.mark.parametrize("variant", CANONICAL_STATE_PREDICTOR_VARIANTS)
def test_predictor_variants_shape_and_zero_init(variant: str) -> None:
    particles, mask, phi_hlt, state_mask = _sample_batch()
    model = build_canonical_state_residual_predictor(**_small_config(variant))
    model.eval()

    with torch.no_grad():
        output = model(particles, mask, phi_hlt, state_mask, return_attention=True)

    assert output.delta_phi.shape == phi_hlt.shape
    assert output.raw_delta.shape == phi_hlt.shape
    assert output.phi_pred.shape == phi_hlt.shape
    assert torch.allclose(output.delta_phi, torch.zeros_like(output.delta_phi), atol=1e-7)
    assert torch.allclose(output.phi_pred, phi_hlt, atol=1e-7)
    assert output.diagnostics["contract"] == CANONICAL_STATE_PREDICTOR_CONTRACT
    assert output.diagnostics["variant"] == variant
    if variant == PREDICTOR_VARIANT_UNCERTAINTY:
        assert output.log_sigma is not None
        assert output.log_sigma.shape == phi_hlt.shape
    else:
        assert output.log_sigma is None


def test_geometry_bias_is_nontrivial_and_can_be_disabled() -> None:
    particles, mask, phi_hlt, state_mask = _sample_batch()
    model = build_canonical_state_residual_predictor(**_small_config(PREDICTOR_VARIANT_GEOMETRY_BIASED))
    no_geo = build_canonical_state_residual_predictor(**_small_config(PREDICTOR_VARIANT_NO_GEOMETRY_BIAS))
    model.eval()
    no_geo.eval()

    with torch.no_grad():
        bias, local_mask = model.geometry_bias_and_mask(particles, mask)
        geo_output = model(particles, mask, phi_hlt, state_mask, return_attention=True)
        no_geo_output = no_geo(particles, mask, phi_hlt, state_mask, return_attention=True)

    assert bias.shape == (particles.shape[0], phi_hlt.shape[1], particles.shape[1])
    assert local_mask.shape == bias.shape
    assert float(bias.min()) < 0.0
    assert float(bias.max()) <= 0.0
    assert geo_output.diagnostics["geometry_bias_applied"] is True
    assert no_geo_output.diagnostics["geometry_bias_applied"] is False


def test_invalid_particles_are_masked_from_attention() -> None:
    particles, mask, phi_hlt, state_mask = _sample_batch()
    model = build_canonical_state_residual_predictor(**_small_config(PREDICTOR_VARIANT_GEOMETRY_BIASED))
    model.eval()

    with torch.no_grad():
        output = model(particles, mask, phi_hlt, state_mask, return_attention=True)

    assert output.diagnostics["cross_attention_shape"] == [
        particles.shape[0],
        4,
        phi_hlt.shape[1],
        particles.shape[1],
    ]
    assert output.diagnostics["invalid_particle_attention_mass_max"] <= 1.0e-6


def test_hard_locality_variant_uses_local_mask() -> None:
    particles, mask, phi_hlt, state_mask = _sample_batch()
    model = build_canonical_state_residual_predictor(**_small_config(PREDICTOR_VARIANT_HARD_LOCALITY))
    model.eval()

    with torch.no_grad():
        _, local_mask = model.geometry_bias_and_mask(particles, mask)
        output = model(particles, mask, phi_hlt, state_mask, return_attention=True)

    assert output.diagnostics["hard_locality_applied"] is True
    assert bool(local_mask.any())
    assert bool((~local_mask).any())


@pytest.mark.parametrize(
    ("variant", "expected_geometry", "expected_self_attention"),
    [
        (PREDICTOR_VARIANT_DEEPSETS, False, True),
        (PREDICTOR_VARIANT_STATE_ONLY, False, True),
        (PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES, False, True),
        (PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION, True, False),
    ],
)
def test_variant_switches_report_active_paths(
    variant: str,
    expected_geometry: bool,
    expected_self_attention: bool,
) -> None:
    particles, mask, phi_hlt, state_mask = _sample_batch()
    model = build_canonical_state_residual_predictor(**_small_config(variant))
    model.eval()

    with torch.no_grad():
        output = model(particles, mask, phi_hlt, state_mask, return_attention=True)

    assert output.diagnostics["geometry_bias_applied"] is expected_geometry
    assert output.diagnostics["state_self_attention_enabled"] is expected_self_attention


def test_all_invalid_particle_row_is_safe() -> None:
    particles, mask, phi_hlt, state_mask = _sample_batch()
    mask[1, :] = False
    particles[1, :, 0] = 0.0
    model = build_canonical_state_residual_predictor(**_small_config(PREDICTOR_VARIANT_GEOMETRY_BIASED))
    model.eval()

    with torch.no_grad():
        output = model(particles, mask, phi_hlt, state_mask, return_attention=True)

    assert torch.isfinite(output.phi_pred).all()
    assert output.diagnostics["invalid_particle_attention_mass_max"] <= 1.0e-6


def test_predictor_config_and_alias_validation() -> None:
    assert normalize_predictor_variant("P0") == PREDICTOR_VARIANT_GEOMETRY_BIASED
    assert normalize_predictor_variant("no_geometry") == PREDICTOR_VARIANT_NO_GEOMETRY_BIAS
    with pytest.raises(ValueError):
        normalize_predictor_variant("missing")
    with pytest.raises(ValueError):
        CanonicalStateResidualPredictorConfig(variant=PREDICTOR_VARIANT_GEOMETRY_BIASED, d_model=30, num_heads=8)
    with pytest.raises(ValueError):
        CanonicalStateResidualPredictorConfig(variant=PREDICTOR_VARIANT_GEOMETRY_BIASED, zero_init_delta_projection=False)
