from __future__ import annotations

import math

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    ArchitectureViewConfig,
    ArchitectureViewResidualParT,
    architecture_view_variant_is_baseline_recheck,
    architecture_view_variant_is_runnable,
)
from teacher_logit_reco.local_compression_part import LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES


torch = require_torch()


class _BatchFirstEmbed(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, rows):
        return self.linear(rows)


class _FakeParTMod(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embed = _BatchFirstEmbed(in_dim, embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, features, v=None, mask=None):
        del v
        rows = features.transpose(1, 2).contiguous()
        embedded = self.embed(rows)
        particle_mask = mask.squeeze(1).to(dtype=embedded.dtype)
        pooled = (embedded * particle_mask[:, :, None]).sum(dim=1)
        pooled = pooled / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(self.norm(pooled))


class DummyEmbeddingPart(ParticleTransformerHLTClassifier):
    def __init__(self, embed_dim: int = 16, num_classes: int = 10) -> None:
        torch.nn.Module.__init__(self)
        self.config = {"dummy_embedding_part": True, "embed_dim": int(embed_dim), "num_classes": int(num_classes)}
        self.mod = _FakeParTMod(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(embed_dim), int(num_classes))

    def no_weight_decay(self):
        return set()

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


def _tokens(batch: int = 2, particles: int = 6):
    tokens = torch.zeros((batch, particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((batch, particles), dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :4] = True
    for batch_index in range(batch):
        for particle_index in range(int(mask[batch_index].sum().item())):
            pt = 20.0 + particle_index + batch_index
            eta = -0.2 + 0.08 * particle_index
            phi = -math.pi + 0.3 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.2
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
    return tokens, mask


def _config() -> ArchitectureViewConfig:
    return ArchitectureViewConfig(
        view_dim=8,
        hidden_dim=16,
        pn_k=3,
        pn_layers=1,
        pfn_hidden_dim=16,
        pcnn_channels=16,
        pcnn_layers=1,
        fusion_hidden_dim=20,
        part_embed_dim=16,
        num_classes=10,
        dropout=0.0,
        attention_dropout=0.0,
    )


def _baseline_logits(model: ArchitectureViewResidualParT, tokens, mask):
    canonical = model.build_canonical_inputs(tokens, mask, max_constits=tokens.shape[1])
    return model.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)


def test_step2_part_only_adapter_is_zero_initialized_and_runnable():
    tokens, mask = _tokens()
    model = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
        part_model=DummyEmbeddingPart(embed_dim=16, num_classes=10),
    )
    model.eval()
    with torch.no_grad():
        baseline = _baseline_logits(model, tokens, mask)
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    assert architecture_view_variant_is_runnable(ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER)
    assert output.diagnostics()["embed_injection"]["adapter_kind"] == "part_embedding_mlp"
    assert output.diagnostics()["variant_behavior"]["uses_part_only_adapter"]
    assert torch.allclose(output.logits, baseline, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(output.view_output.delta_h, torch.zeros_like(output.view_output.delta_h))


def test_step2_extra_part_block_is_zero_initialized_and_runnable():
    tokens, mask = _tokens()
    model = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
        part_model=DummyEmbeddingPart(embed_dim=16, num_classes=10),
    )
    model.eval()
    with torch.no_grad():
        baseline = _baseline_logits(model, tokens, mask)
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    assert architecture_view_variant_is_runnable(ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK)
    assert output.diagnostics()["embed_injection"]["adapter_kind"] == "extra_part_block"
    assert output.diagnostics()["variant_behavior"]["uses_extra_part_block"]
    assert torch.allclose(output.logits, baseline, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(output.view_output.delta_h, torch.zeros_like(output.view_output.delta_h))


def test_step2_larger_part_is_training_variant_not_eval_only_baseline():
    assert architecture_view_variant_is_runnable(ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART)
    assert not architecture_view_variant_is_baseline_recheck(ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART)

    model = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
        part_model=DummyEmbeddingPart(embed_dim=16, num_classes=10),
    )
    behavior = model.variant_behavior()
    assert behavior["uses_larger_part_backbone"]
    assert behavior["forces_zero_delta"]
    assert behavior["injects_embedding_delta"] is False
