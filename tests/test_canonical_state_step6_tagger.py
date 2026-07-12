import pytest
import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.local_compression_part.config import LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES
from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_TAGGER_CONTRACT,
    STATE_CONTEXT_ALL,
    STATE_CONTEXT_DELTA_PHI,
    STATE_CONTEXT_FEATURE_MLP_PLUS_STATE,
    STATE_CONTEXT_NOISE,
    STATE_CONTEXT_ORACLE_PHI_OFF,
    STATE_CONTEXT_PARTICLES_ONLY,
    STATE_CONTEXT_PHI_HLT,
    STATE_CONTEXT_PHI_PRED,
    STATE_CONTEXT_SHUFFLED,
    STATE_CONTEXT_STATE_ONLY,
    CanonicalStateConditionedParT,
    CanonicalStateTaggerConfig,
    default_canonical_jet_state_layout,
    normalize_state_tagger_mode,
)


class _FakeParT(torch.nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES),
        embed_dim: int = 32,
        num_classes: int = 10,
        trim_to: int | None = None,
    ) -> None:
        super().__init__()
        self.trim_to = trim_to
        self.mod = torch.nn.Module()
        self.mod.embed = torch.nn.Linear(feature_dim, embed_dim)
        self.mod.fc = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, points, features, lorentz_vectors, mask):  # noqa: ANN001
        del points, lorentz_vectors
        rows = features.transpose(1, 2).contiguous()
        if self.trim_to is not None:
            rows = rows[:, : self.trim_to, :]
            mask = mask[:, :, : self.trim_to]
        embeddings = self.mod.embed(rows)
        valid = mask[:, 0, :].bool()[:, :, None].to(dtype=embeddings.dtype)
        pooled = (embeddings * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.mod.fc(pooled)


def _config(mode: str) -> CanonicalStateTaggerConfig:
    return CanonicalStateTaggerConfig(
        mode=mode,
        part_embed_dim=32,
        state_dim=32,
        state_layers=1,
        state_heads=4,
        state_mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
        max_constits=16,
        predictor_config={
            "d_model": 32,
            "num_heads": 4,
            "particle_encoder_layers": 1,
            "decoder_layers": 1,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "max_particles": 16,
        },
    )


def _sample_inputs(batch_size: int = 2, n_particles: int = 7):
    layout = default_canonical_jet_state_layout()
    tokens = torch.zeros((batch_size, n_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    pt = torch.tensor([0.55, 0.36, 0.22, 0.14, 0.09, 0.05, 0.02])[:n_particles]
    tokens[..., 0] = pt
    tokens[..., 1] = torch.linspace(-0.45, 0.45, n_particles)
    tokens[..., 2] = torch.linspace(-0.35, 0.35, n_particles)
    tokens[..., 3] = torch.log1p(tokens[..., 0])
    tokens[..., 4] = tokens[..., 0] * 1.4
    tokens[..., 5] = 1.0
    mask = torch.ones((batch_size, n_particles), dtype=torch.bool)
    mask[0, -1] = False
    tokens[0, -1, 0] = 0.0
    phi_hlt = torch.randn((batch_size, layout.k_state, layout.d_phi), dtype=torch.float32) * 0.05
    delta_phi = torch.randn_like(phi_hlt) * 0.01
    phi_pred = phi_hlt + delta_phi
    phi_off = phi_hlt + 2.0 * delta_phi
    return tokens, mask, phi_hlt, delta_phi, phi_pred, phi_off


def _model(mode: str) -> CanonicalStateConditionedParT:
    return CanonicalStateConditionedParT(_config(mode), part_model=_FakeParT())


def test_zero_init_state_adapter_preserves_baseline_logits() -> None:
    tokens, mask, phi_hlt, _, _, _ = _sample_inputs()
    model = _model(STATE_CONTEXT_PHI_HLT)
    model.eval()
    canonical = model.build_canonical_inputs(tokens, mask)
    with torch.no_grad():
        baseline = model.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)
        output = model(tokens, mask, phi_hlt=phi_hlt)

    assert torch.allclose(output.logits, baseline, atol=1.0e-6)
    assert output.output_contract == CANONICAL_STATE_TAGGER_CONTRACT
    assert output.diagnostics["injection"]["injection_applied"] is True
    assert output.diagnostics["injection"]["state_adapter"]["delta_h_norm_max"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "mode",
    [
        STATE_CONTEXT_PARTICLES_ONLY,
        STATE_CONTEXT_PHI_HLT,
        STATE_CONTEXT_DELTA_PHI,
        STATE_CONTEXT_PHI_PRED,
        STATE_CONTEXT_ALL,
        STATE_CONTEXT_STATE_ONLY,
        STATE_CONTEXT_FEATURE_MLP_PLUS_STATE,
    ],
)
def test_state_tagger_modes_emit_logits_and_diagnostics(mode: str) -> None:
    tokens, mask, phi_hlt, delta_phi, phi_pred, _ = _sample_inputs()
    model = _model(mode)
    model.eval()

    with torch.no_grad():
        output = model(tokens, mask, phi_hlt=phi_hlt, delta_phi=delta_phi, phi_pred=phi_pred)

    assert output.logits.shape == (tokens.shape[0], 10)
    assert torch.isfinite(output.logits).all()
    assert output.diagnostics["contract"] == CANONICAL_STATE_TAGGER_CONTRACT
    assert output.diagnostics["mode"] == mode
    if mode == STATE_CONTEXT_PARTICLES_ONLY:
        assert output.state_context is None
        assert output.delta_h is None
    elif mode == STATE_CONTEXT_STATE_ONLY:
        assert output.state_context is not None
        assert output.delta_h is None
    else:
        assert output.state_context is not None
        assert output.delta_h is not None


def test_state_family_embeddings_create_context_tokens() -> None:
    tokens, mask, phi_hlt, _, _, _ = _sample_inputs()
    model = _model(STATE_CONTEXT_PHI_HLT)
    model.eval()

    with torch.no_grad():
        output = model(tokens, mask, phi_hlt=phi_hlt)

    assert output.state_context is not None
    assert output.state_context.shape[:2] == phi_hlt.shape[:2]
    assert output.state_context.abs().sum().item() > 0.0
    assert output.diagnostics["state_context_ids"] == [0]


def test_shuffled_and_noise_controls_break_state_semantics() -> None:
    tokens, mask, phi_hlt, _, _, _ = _sample_inputs()
    shuffled = _model(STATE_CONTEXT_SHUFFLED)
    noise = _model(STATE_CONTEXT_NOISE)
    shuffled.eval()
    noise.eval()

    with torch.no_grad():
        shuffled_output = shuffled(tokens, mask, phi_hlt=phi_hlt)
        noise_output = noise(tokens, mask, phi_hlt=phi_hlt)

    assert shuffled_output.diagnostics["state_semantics_broken"] is True
    assert noise_output.diagnostics["state_semantics_broken"] is True
    assert shuffled_output.state_values is not None
    assert noise_output.state_values is not None
    assert not torch.allclose(shuffled_output.state_values, phi_hlt)
    assert not torch.allclose(noise_output.state_values, phi_hlt)


def test_oracle_phi_off_context_is_blocked_on_primary_final_test() -> None:
    tokens, mask, phi_hlt, _, _, phi_off = _sample_inputs()
    model = _model(STATE_CONTEXT_ORACLE_PHI_OFF)
    model.eval()

    with pytest.raises(ValueError, match="blocked"):
        model(tokens, mask, phi_hlt=phi_hlt, phi_off=phi_off, split="final_test")

    with torch.no_grad():
        output = model(tokens, mask, phi_hlt=phi_hlt, phi_off=phi_off, split="final_test", allow_oracle_context=True)

    assert output.diagnostics["oracle_context_used"] is True
    assert output.logits.shape == (tokens.shape[0], 10)


def test_delta_phi_derives_phi_pred_when_only_delta_supplied() -> None:
    tokens, mask, phi_hlt, delta_phi, _, _ = _sample_inputs()
    model = _model(STATE_CONTEXT_PHI_PRED)
    model.eval()

    with torch.no_grad():
        output = model(tokens, mask, phi_hlt=phi_hlt, delta_phi=delta_phi)

    assert output.state_values is not None
    assert output.state_values.shape[1] == 2 * phi_hlt.shape[1]
    assert torch.allclose(output.state_values[:, : phi_hlt.shape[1]], phi_hlt, atol=1.0e-6)
    assert torch.allclose(output.state_values[:, phi_hlt.shape[1] :], phi_hlt + delta_phi, atol=1.0e-6)
    assert output.diagnostics["state_context_ids"] == [0, 2]
    assert output.predictor_output is None


def test_state_mask_is_propagated_to_repeated_state_context() -> None:
    tokens, mask, phi_hlt, delta_phi, _, _ = _sample_inputs()
    layout = default_canonical_jet_state_layout()
    state_mask = torch.ones((tokens.shape[0], layout.k_state), dtype=torch.bool)
    state_mask[:, -3:] = False
    model = _model(STATE_CONTEXT_DELTA_PHI)
    model.eval()

    with torch.no_grad():
        output = model(tokens, mask, phi_hlt=phi_hlt, delta_phi=delta_phi, state_mask=state_mask)

    assert output.state_mask is not None
    assert output.state_mask.shape[1] == 2 * state_mask.shape[1]
    assert torch.equal(output.state_mask[:, : state_mask.shape[1]], state_mask)
    assert torch.equal(output.state_mask[:, state_mask.shape[1] :], state_mask)
    assert output.diagnostics["state_valid_count_mean"] == pytest.approx(float(state_mask.sum(dim=1).float().mean().item()) * 2.0)


def test_delta_context_can_use_separate_delta_state_mask() -> None:
    tokens, mask, phi_hlt, delta_phi, _, _ = _sample_inputs()
    layout = default_canonical_jet_state_layout()
    hlt_state_mask = torch.ones((tokens.shape[0], layout.k_state), dtype=torch.bool)
    delta_state_mask = hlt_state_mask.clone()
    hlt_state_mask[:, -1] = False
    delta_state_mask[:, -1] = True
    model = _model(STATE_CONTEXT_DELTA_PHI)
    model.eval()

    with torch.no_grad():
        output = model(
            tokens,
            mask,
            phi_hlt=phi_hlt,
            delta_phi=delta_phi,
            state_mask=hlt_state_mask,
            delta_state_mask=delta_state_mask,
        )

    assert output.state_mask is not None
    assert output.state_mask[:, layout.k_state - 1].eq(False).all()
    assert output.state_mask[:, 2 * layout.k_state - 1].eq(True).all()
    assert output.diagnostics["state_mask_policy"] == "hlt_mask_for_phi_hlt__separate_mask_for_delta_phi"


def test_feature_mlp_plus_state_runs_both_zero_init_adapters() -> None:
    tokens, mask, phi_hlt, delta_phi, phi_pred, _ = _sample_inputs()
    model = _model(STATE_CONTEXT_FEATURE_MLP_PLUS_STATE)
    model.eval()

    with torch.no_grad():
        output = model(tokens, mask, phi_hlt=phi_hlt, delta_phi=delta_phi, phi_pred=phi_pred)

    assert output.delta_h is not None
    assert output.diagnostics["feature_mlp_adapter"]["feature_mlp_adapter_active"] is True
    assert output.diagnostics["feature_mlp_adapter"]["delta_h_norm_max"] == pytest.approx(0.0)
    assert output.diagnostics["injection"]["state_adapter"]["delta_h_norm_max"] == pytest.approx(0.0)


def test_state_injection_aligns_mask_to_trimmed_part_embedding_rows() -> None:
    tokens, mask, phi_hlt, _, _, _ = _sample_inputs(n_particles=7)
    model = CanonicalStateConditionedParT(_config(STATE_CONTEXT_PHI_HLT), part_model=_FakeParT(trim_to=5))
    model.eval()

    with torch.no_grad():
        output = model(tokens, mask, phi_hlt=phi_hlt)

    assert output.logits.shape == (tokens.shape[0], 10)
    assert output.delta_h is not None
    assert output.delta_h.shape[1] == 5
    assert output.diagnostics["injection"]["hook_particle_count"] == 5
    assert output.diagnostics["injection"]["input_particle_count"] == tokens.shape[1]


def test_state_tagger_mode_alias_and_config_validation() -> None:
    assert normalize_state_tagger_mode("all") == STATE_CONTEXT_ALL
    assert normalize_state_tagger_mode("oracle") == STATE_CONTEXT_ORACLE_PHI_OFF
    with pytest.raises(ValueError):
        normalize_state_tagger_mode("not_a_mode")
    with pytest.raises(ValueError):
        CanonicalStateTaggerConfig(mode=STATE_CONTEXT_PHI_HLT, state_dim=30, state_heads=8)
