from __future__ import annotations

import math

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ArchitectureViewConfig,
    ArchitectureViewResidualParT,
    architecture_view_variant_is_runnable,
)
from teacher_logit_reco.architecture_view_part.train import _delta_l2_mean_from_output
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


class DummyTenClassPart(ParticleTransformerHLTClassifier):
    def __init__(self, embed_dim: int = 16) -> None:
        torch.nn.Module.__init__(self)
        self.config = {"dummy_embedding_part": True, "embed_dim": int(embed_dim), "num_classes": 10}
        self.mod = _FakeParTMod(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(embed_dim), 10)

    def no_weight_decay(self):
        return set()

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


def _config(**overrides) -> ArchitectureViewConfig:
    payload = {
        "view_dim": 8,
        "hidden_dim": 16,
        "pn_k": 3,
        "pn_layers": 1,
        "pfn_hidden_dim": 16,
        "pcnn_channels": 16,
        "pcnn_layers": 1,
        "fusion_hidden_dim": 20,
        "part_embed_dim": 16,
        "num_classes": 10,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    payload.update(overrides)
    return ArchitectureViewConfig(**payload)


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
            tokens[batch_index, particle_index, 10] = 0.01 * particle_index
            tokens[batch_index, particle_index, 11] = 0.03
            tokens[batch_index, particle_index, 12] = -0.02 * particle_index
            tokens[batch_index, particle_index, 13] = 0.04
    return tokens, mask


def _baseline_logits(model: ArchitectureViewResidualParT, tokens, mask):
    canonical = model.build_canonical_inputs(tokens, mask, max_constits=tokens.shape[1])
    return model.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)


def test_step6_lc_mlp_delta_starts_as_exact_input_feature_identity():
    tokens, mask = _tokens()
    model = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
        part_model=DummyTenClassPart(),
    )
    model.eval()

    with torch.no_grad():
        baseline = _baseline_logits(model, tokens, mask)
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    assert architecture_view_variant_is_runnable(ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES)
    assert torch.allclose(output.logits, baseline, atol=1.0e-6, rtol=1.0e-6)
    assert output.feature_delta_output is not None
    assert torch.allclose(
        output.feature_delta_output.delta_F_rows,
        torch.zeros_like(output.feature_delta_output.delta_F_rows),
    )
    assert output.diagnostics()["variant_behavior"]["uses_lc_mlp_delta_features"]
    assert output.diagnostics()["variant_behavior"]["adapts_input_features"]
    assert output.diagnostics()["variant_behavior"]["injects_embedding_delta"] is False
    assert output.diagnostics()["embed_injection"]["adapter_kind"] == "no_injection"
    assert output.diagnostics()["feature_delta"]["diagnostics"]["adapter_kind"] == "lc_mlp_delta_features"
    assert output.diagnostics()["feature_delta"]["diagnostics"]["delta_F_l2_mean"] == 0.0
    accounting = model.parameter_accounting()
    assert accounting["active_adapter_module_names"] == ["feature_delta_adapter"]
    assert accounting["dormant_adapter_params"] > 0
    assert accounting["adapter_params"] < accounting["all_adapter_params"]


def test_step6_parameter_accounting_is_variant_active():
    baseline = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
        part_model=DummyTenClassPart(),
    )
    feature = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        part_model=DummyTenClassPart(),
    )
    lc_delta = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
        part_model=DummyTenClassPart(),
    )

    baseline_accounting = baseline.parameter_accounting()
    feature_accounting = feature.parameter_accounting()
    lc_accounting = lc_delta.parameter_accounting()

    assert baseline_accounting["adapter_params"] == 0
    assert baseline_accounting["active_adapter_module_names"] == []
    assert feature_accounting["active_adapter_module_names"] == ["context_control", "context_control_gate"]
    assert lc_accounting["active_adapter_module_names"] == ["feature_delta_adapter"]
    assert feature_accounting["adapter_params"] < feature_accounting["all_adapter_params"]
    assert lc_accounting["adapter_params"] < lc_accounting["all_adapter_params"]


def test_step6_delta_l2_regularizer_uses_delta_f_for_input_delta_variant():
    tokens, mask = _tokens()
    model = ArchitectureViewResidualParT(
        _config(input_delta_scale=0.5),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
        part_model=DummyTenClassPart(),
    )
    with torch.no_grad():
        final = model.feature_delta_adapter.projector[-1]
        final.bias.fill_(0.25)
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    delta = output.feature_delta_output.delta_F_rows
    expected = delta.square().sum(dim=-1)[output.feature_delta_output.mask].mean()
    assert expected.item() > 0.0
    assert torch.allclose(_delta_l2_mean_from_output(output), expected)
    assert torch.all(output.feature_delta_output.delta_F_rows.abs() <= output.feature_delta_output.feature_delta_scales.view(1, 1, -1) + 1.0e-6)


def test_step6_input_delta_can_freeze_pid_and_geometry_channels():
    tokens, mask = _tokens()
    model = ArchitectureViewResidualParT(
        _config(freeze_input_delta_pid=True, freeze_input_delta_geometry=True),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
        part_model=DummyTenClassPart(),
    )
    with torch.no_grad():
        model.feature_delta_adapter.projector[-1].bias.fill_(1.0)
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    names = tuple(output.feature_delta_output.feature_names)
    frozen = {
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
        "part_deltaR",
        "part_deta",
        "part_dphi",
    }
    for name in frozen:
        index = names.index(name)
        assert float(output.feature_delta_output.delta_F_rows[:, :, index].abs().max().item()) == 0.0
