from __future__ import annotations

import pytest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_BRANCHES,
    ARCHITECTURE_VIEW_BRANCH_PCNN,
    ARCHITECTURE_VIEW_BRANCH_PFN,
    ARCHITECTURE_VIEW_BRANCH_PN,
    ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ArchitectureViewBranchBank,
    ArchitectureViewConfig,
    ArchitectureViewFusion,
    ArchitectureViewParticleViews,
    architecture_view_config_manifest,
    build_architecture_view_branch,
    enabled_views_for_variant,
    normalize_architecture_view_variant,
    sanitize_architecture_view_tokens,
)


def _toy_tokens(batch: int = 3, particles: int = 9):
    torch = require_torch()
    generator = torch.Generator().manual_seed(123)
    tokens = torch.randn(batch, particles, RAW_TOKEN_DIM, generator=generator) * 0.2
    tokens[:, :, 0] = torch.linspace(80.0, 5.0, particles)[None, :]
    tokens[:, :, 1] = torch.linspace(-1.0, 1.0, particles)[None, :]
    tokens[:, :, 2] = torch.linspace(-3.0, 3.0, particles)[None, :]
    tokens[:, :, 3] = tokens[:, :, 0] * 1.3
    tokens[:, :, 4] = torch.sign(tokens[:, :, 4])
    tokens[:, :, 5:10] = 0.0
    tokens[:, :, 5] = 1.0
    tokens[:, 1::3, 5:10] = 0.0
    tokens[:, 1::3, 7] = 1.0
    tokens[:, :, 11] = tokens[:, :, 11].abs()
    tokens[:, :, 13] = tokens[:, :, 13].abs()
    mask = torch.ones(batch, particles, dtype=torch.bool)
    mask[0, -2:] = False
    mask[1, -1] = False
    tokens = tokens.clone()
    tokens[0, -1, 0] = float("nan")
    tokens.requires_grad_(True)
    return tokens, mask


def _small_config(**overrides):
    return ArchitectureViewConfig(
        view_dim=12,
        hidden_dim=24,
        pn_k=4,
        pn_layers=2,
        pfn_hidden_dim=24,
        pcnn_channels=24,
        pcnn_layers=2,
        fusion_hidden_dim=32,
        part_embed_dim=40,
        dropout=0.0,
        attention_dropout=0.0,
        **overrides,
    )


def test_architecture_view_config_and_variant_registry():
    config = _small_config()
    manifest = architecture_view_config_manifest(config)
    assert manifest["hlt_degradation_strength"] == 0.6
    assert normalize_architecture_view_variant("full") == ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS
    assert enabled_views_for_variant(ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS) == ARCHITECTURE_VIEW_BRANCHES
    assert enabled_views_for_variant(ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK) == ()
    with pytest.raises(ValueError):
        ArchitectureViewConfig(enabled_views=("pn", "pn"))


def test_raw_hlt_input_sanitization_preserves_quality_signal():
    torch = require_torch()
    tokens, mask = _toy_tokens()
    tokens = tokens.detach()
    tokens[0, 0, 2] = float("inf")
    prepared = sanitize_architecture_view_tokens(tokens, mask)
    assert prepared.tokens.shape == tokens.shape
    assert prepared.mask.shape == mask.shape
    assert torch.isfinite(prepared.tokens).all()
    assert not bool(prepared.mask[0, 0])
    assert not bool(prepared.original_finite_mask[0, 0])
    assert torch.allclose(prepared.tokens[~prepared.mask], torch.zeros_like(prepared.tokens[~prepared.mask]))
    assert prepared.quality_features.shape[:2] == mask.shape


@pytest.mark.parametrize(
    "branch_name",
    [ARCHITECTURE_VIEW_BRANCH_PN, ARCHITECTURE_VIEW_BRANCH_PFN, ARCHITECTURE_VIEW_BRANCH_PCNN],
)
def test_each_architecture_view_branch_runs_independently_with_finite_gradients(branch_name):
    torch = require_torch()
    tokens, mask = _toy_tokens()
    config = _small_config(enabled_views=(branch_name,))
    branch = build_architecture_view_branch(branch_name, config)
    output = branch(tokens, mask)
    assert output.embeddings.shape == (tokens.shape[0], tokens.shape[1], config.view_dim)
    assert output.mask.shape == mask.shape
    assert torch.isfinite(output.embeddings).all()
    assert torch.allclose(output.embeddings[~output.mask], torch.zeros_like(output.embeddings[~output.mask]))
    loss = output.embeddings[output.mask].pow(2).mean()
    loss.backward()
    finite_grads = [
        param.grad
        for param in branch.parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert finite_grads
    assert all(torch.isfinite(grad).all().item() for grad in finite_grads)


def test_branch_bank_outputs_all_enabled_views():
    tokens, mask = _toy_tokens()
    config = _small_config()
    bank = ArchitectureViewBranchBank(config)
    outputs = bank(tokens, mask)
    assert tuple(outputs) == ARCHITECTURE_VIEW_BRANCHES
    for name, output in outputs.items():
        assert output.name == name
        assert output.embeddings.shape[-1] == config.view_dim


def test_view_fusion_zero_init_is_baseline_safe_and_masks_invalid_particles():
    torch = require_torch()
    tokens, mask = _toy_tokens()
    config = _small_config()
    bank = ArchitectureViewBranchBank(config)
    branch_outputs = bank(tokens, mask)
    fusion = ArchitectureViewFusion(config)
    output = fusion(branch_outputs, mask)
    assert output.combined_view.shape == (tokens.shape[0], tokens.shape[1], config.view_dim * 3)
    assert output.delta_h.shape == (tokens.shape[0], tokens.shape[1], config.part_embed_dim)
    assert output.gate.shape == (tokens.shape[0], tokens.shape[1], 1)
    assert torch.isfinite(output.combined_view).all()
    assert torch.isfinite(output.delta_h).all()
    assert torch.allclose(output.delta_h, torch.zeros_like(output.delta_h))
    assert torch.allclose(output.gate[~mask], torch.zeros_like(output.gate[~mask]))


def test_step1_particle_view_module_supports_single_view_and_empty_baseline():
    torch = require_torch()
    tokens, mask = _toy_tokens()
    config = _small_config()
    single = ArchitectureViewParticleViews(config, enabled_views=(ARCHITECTURE_VIEW_BRANCH_PN,))
    single_output = single(tokens, mask)
    assert list(single_output.view_embeddings) == [ARCHITECTURE_VIEW_BRANCH_PN]
    assert single_output.combined_view.shape[-1] == config.view_dim
    assert single_output.delta_h.shape[-1] == config.part_embed_dim
    baseline = ArchitectureViewParticleViews(config, enabled_views=())
    baseline_output = baseline(tokens, mask)
    assert baseline_output.combined_view.shape[-1] == 0
    assert torch.allclose(baseline_output.delta_h, torch.zeros_like(baseline_output.delta_h))
