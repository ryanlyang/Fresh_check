from __future__ import annotations

import copy
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    build_campaign_spec,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_model import (
    RetbExpertModel,
    RetbParticleEncoder,
    expert_relation_family,
)
from teacher_logit_reco.relation_expert_token_bridge.layerwise_pair_bias import (
    LAYERWISE_PAIR_BIAS_CONTRACT,
    LayerwisePairBiasProvider,
    build_layerwise_pair_bias_contract,
    validate_layerwise_pair_bias_contract,
)
from teacher_logit_reco.relation_expert_token_bridge.particle_tap import (
    MeasurementStateEmbedding,
    ReferenceParticleStateTap,
    build_measurement_embedding_contract,
    build_particle_tap_contract,
    derive_measurement_states_torch,
)
from teacher_logit_reco.relation_expert_token_bridge.step3 import (
    build_step3_bundle,
    publish_step3_bundle,
    validate_step3_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.summary_tokens import (
    CanonicalSummaryTokenizer,
    MultiDepthSummaryTokenizer,
    TokenOnlyExpertHead,
    build_summary_tokenizer_contract,
    build_token_only_head_contract,
)
from teacher_logit_reco.relation_expert_token_bridge.token_shape_registry import (
    HET_PHYSICS,
    build_token_shape_contract,
    resolve_expert_shapes,
    resolve_uniform_shape,
    validate_heterogeneous_allocation,
)
from teacher_logit_reco.relational_part.attention import DirectionalPairStem
from teacher_logit_reco.relational_part.model import exact_rpt_base_config


EXPERT_ORDER = ("BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")
UNIFORM_SHAPES = (
    "S1_128",
    "S2_128",
    "S4_128",
    "S8_128",
    "S16_128",
    "S8_64",
    "S16_64",
)


class _FakePairEmbed(torch.nn.Module):
    def __init__(self, input_dimension: int, heads: int) -> None:
        super().__init__()
        self.pairwise_lv_dim = 0
        self.pairwise_input_dim = int(input_dimension)
        self.out_dim = int(heads)
        self.remove_self_pair = False
        self.fts_embed = torch.nn.Sequential(
            torch.nn.BatchNorm1d(int(input_dimension)),
            torch.nn.Conv1d(int(input_dimension), 12, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(12, int(heads), 1),
        )


class _FakeParticleBlock(torch.nn.Module):
    def __init__(self, dimension: int = 128, heads: int = 8) -> None:
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(dimension)

    def forward(self, x, padding_mask=None, attn_mask=None):
        if attn_mask is not None and attn_mask.ndim == 4:
            attn_mask = attn_mask.flatten(0, 1)
        update, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.norm(x + update)


class _FakeClassBlock(torch.nn.Module):
    def __init__(self, dimension: int = 128, heads: int = 8) -> None:
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )

    def forward(self, x, x_cls=None, padding_mask=None):
        context = torch.cat((x_cls, x), dim=1)
        cls_padding = torch.zeros(
            padding_mask.shape[0],
            1,
            dtype=torch.bool,
            device=padding_mask.device,
        )
        update, _ = self.attention(
            x_cls,
            context,
            context,
            key_padding_mask=torch.cat((cls_padding, padding_mask), dim=1),
            need_weights=False,
        )
        return x_cls + update


class _FakeArchitectureTransformer(torch.nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        self.pair_embed = _FakePairEmbed(
            int(config["pair_extra_dim"]), int(config["num_heads"])
        )
        self.input = torch.nn.Linear(17, 128)
        self.blocks = torch.nn.ModuleList(
            [_FakeParticleBlock(128, int(config["num_heads"])) for _ in range(8)]
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.cls_blocks = torch.nn.ModuleList(
            [_FakeClassBlock(128, int(config["num_heads"])) for _ in range(2)]
        )
        self.norm = torch.nn.LayerNorm(128)
        self.fc = torch.nn.Linear(128, 10)
        self.include_global_token = False

    def embed(self, features):
        return self.input(features.transpose(1, 2))


def _fake_weaver():
    def transformer(**config):
        return _FakeArchitectureTransformer(**config)

    def pairwise(xi, xj, num_outputs=4):
        base = xi[:, :1] + xj[:, :1]
        return torch.cat([base + index for index in range(num_outputs)], dim=1)

    return SimpleNamespace(
        ParticleTransformer=transformer,
        pairwise_lv_fts=pairwise,
    )


def _batch(batch: int = 2, particles: int = 5):
    torch.manual_seed(9001)
    features = torch.randn(batch, 17, particles)
    vectors = torch.randn(batch, 4, particles)
    mask = torch.ones(batch, 1, particles, dtype=torch.bool)
    mask[0, 0, -2:] = False
    mask[1, 0, -1:] = False
    raw = torch.zeros(batch, particles, 14)
    raw[:, :, 0] = torch.linspace(20.0, 1.0, particles)
    raw[:, :, 1] = torch.linspace(-1.8, 1.8, particles)
    raw[:, :, 2] = torch.linspace(-2.0, 2.0, particles)
    raw[:, :, 3] = raw[:, :, 0] * torch.cosh(raw[:, :, 1])
    raw[:, :, 5] = 1.0
    raw[:, :, 4] = 1.0
    raw[:, :, 10] = 0.1
    raw[:, :, 11] = 0.02
    raw[:, :, 12] = 0.2
    raw[:, :, 13] = 0.04
    raw[0, 1, 10:14] = 0.0
    raw[1, 2, 5] = 0.0
    raw[1, 2, 6] = 1.0
    raw[1, 2, 4] = 0.0
    features = features.masked_fill(~mask, 0.0)
    vectors = vectors.masked_fill(~mask, 0.0)
    raw = raw.masked_fill(~mask.transpose(1, 2), 0.0)
    return {
        "features": features,
        "vectors": vectors,
        "mask": mask,
        "raw_tokens": raw,
    }


class _ReferenceTransformer(torch.nn.Module):
    """Small official-interface analogue for tap parity."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(17, 32)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    32,
                    4,
                    dim_feedforward=64,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(2)
            ]
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 32))
        self.cls_attention = torch.nn.MultiheadAttention(
            32, 4, dropout=0.0, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(32)
        self.fc = torch.nn.Linear(32, 10)
        self.include_global_token = False

    def _forward_encoder(self, x, v=None, mask=None, uu=None, uu_idx=None):
        del v, uu, uu_idx
        padding = ~mask[:, 0].bool()
        states = self.embed(x.transpose(1, 2)).masked_fill(
            padding.unsqueeze(-1), 0.0
        )
        for block in self.blocks:
            states = block(states, src_key_padding_mask=padding)
        return states, padding

    def _forward_aggregator(self, x, padding_mask):
        query = self.cls_token.expand(x.shape[0], -1, -1)
        update, _ = self.cls_attention(
            query,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.norm(update[:, 0])

    def forward(self, x, v=None, mask=None, uu=None):
        states, padding = self._forward_encoder(x, v=v, mask=mask, uu=uu)
        return self.fc(self._forward_aggregator(states, padding))


def _pair_stem(input_dimension: int = 4) -> DirectionalPairStem:
    return DirectionalPairStem(
        _FakePairEmbed(input_dimension, 8),
        input_dimension=input_dimension,
    )


def test_step3_contract_bundle_and_candidate_registration(tmp_path: Path) -> None:
    source = {
        "source_commit": "1" * 40,
        "source_status_sha256": "2" * 64,
        "source_dirty": True,
    }
    bundle = build_step3_bundle(
        campaign_spec_sha256="3" * 64,
        source_snapshot=source,
    )
    digest = validate_step3_bundle(bundle)
    assert bundle["particle_tap"]["contract"] == "retb_particle_state_tap_v1"
    assert (
        bundle["layerwise_pair_bias"]["contract"]
        == LAYERWISE_PAIR_BIAS_CONTRACT
    )
    assert bundle["candidate_registry"]["registered_particle_view_candidate"][
        "id"
    ] == "V_MEASUREMENT_EMBED"
    assert bundle["candidate_registry"]["classification_bypass_allowed"] is False
    result = publish_step3_bundle(campaign_root=tmp_path, bundle=bundle)
    assert result["step3_bundle_sha256"] == digest
    assert (
        tmp_path / "registry" / "retb_step3_architecture_bundle.json"
    ).is_file()
    repeated = publish_step3_bundle(campaign_root=tmp_path, bundle=bundle)
    assert repeated["publications"]["step3_bundle"]["status"] == "already_present"
    tampered = copy.deepcopy(bundle)
    candidate = dict(tampered["candidate_registry"])
    candidate.pop("content_hash")
    candidate["classification_bypass_allowed"] = True
    tampered["candidate_registry"] = with_content_hash(candidate)
    manifest = dict(tampered["step3_bundle"])
    manifest.pop("content_hash")
    manifest["artifact_hashes"]["candidate_registry"] = tampered[
        "candidate_registry"
    ]["content_hash"]
    tampered["step3_bundle"] = with_content_hash(manifest)
    report = dict(tampered["step3_report"])
    report.pop("content_hash")
    report["step3_bundle_sha256"] = tampered["step3_bundle"]["content_hash"]
    tampered["step3_report"] = with_content_hash(report)
    with pytest.raises(ValueError, match="locked definition"):
        validate_step3_bundle(tampered)


def test_individual_step3_contracts_are_locked() -> None:
    assert build_particle_tap_contract()["particle_state_shape"] == [
        "B",
        "N",
        128,
    ]
    assert build_measurement_embedding_contract()["states"] == [
        "not_track_domain",
        "track_measurement_available",
        "track_measurement_missing",
    ]
    layerwise = build_layerwise_pair_bias_contract()
    assert validate_layerwise_pair_bias_contract(layerwise) == layerwise[
        "content_hash"
    ]
    assert layerwise["materialize_B_L_H_N_N"] is False
    assert build_summary_tokenizer_contract()["canonical_blocks"] == 2
    assert (
        build_token_only_head_contract()["input"] == "summary_tokens_only"
    )
    shapes = build_token_shape_contract()
    assert shapes["equal_scalar_budget_comparison"]["scalar_count"] == 1024


def test_measurement_states_and_embedding_distinguish_three_domains() -> None:
    batch = _batch()
    states = derive_measurement_states_torch(
        batch["raw_tokens"], batch["mask"]
    )
    assert states[0, 0].item() == 1
    assert states[0, 1].item() == 2
    assert states[1, 2].item() == 0
    embedding = MeasurementStateEmbedding().eval()
    output = embedding(states, batch["mask"])
    assert output.shape == (2, 5, 128)
    assert torch.count_nonzero(output[~batch["mask"][:, 0]]) == 0
    assert not torch.equal(output[0, 0], output[0, 1])
    output.sum().backward()
    assert embedding.embedding.weight.grad is not None


def test_reference_particle_tap_preserves_logits_and_gradients() -> None:
    torch.manual_seed(14)
    direct = _ReferenceTransformer().eval()
    tapped_model = copy.deepcopy(direct).eval()
    tap = ReferenceParticleStateTap(tapped_model)
    features = torch.randn(3, 17, 6, requires_grad=True)
    tapped_features = features.detach().clone().requires_grad_(True)
    mask = torch.tensor(
        [
            [[True, True, True, True, False, False]],
            [[True, True, True, False, False, False]],
            [[True, True, True, True, True, False]],
        ]
    )
    expected = direct(features, mask=mask)
    actual = tap(tapped_features, mask=mask, return_states=True)
    torch.testing.assert_close(actual["logits"], expected, atol=0, rtol=0)
    assert actual["particle_states"].shape == (3, 6, 32)
    assert torch.equal(actual["particle_mask"], mask[:, 0])
    expected.square().sum().backward()
    actual["logits"].square().sum().backward()
    torch.testing.assert_close(
        tapped_features.grad, features.grad, atol=0, rtol=0
    )
    for (expected_name, expected_parameter), (
        actual_name,
        actual_parameter,
    ) in zip(direct.named_parameters(), tapped_model.named_parameters()):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=0,
            rtol=0,
        )


def test_real_weaver_particle_tap_parity_when_available() -> None:
    try:
        weaver_module = importlib.import_module(
            "weaver.nn.model.ParticleTransformer"
        )
    except ImportError:
        pytest.skip("real Weaver is not installed in this environment")
    torch.manual_seed(1403)
    direct = weaver_module.ParticleTransformer(
        **exact_rpt_base_config()
    ).float().eval()
    tapped_model = copy.deepcopy(direct).float().eval()
    direct.trimmer.enabled = False
    tapped_model.trimmer.enabled = False
    tap = ReferenceParticleStateTap(tapped_model)
    batch, particles = 2, 7
    mask = torch.tensor(
        [
            [[True, True, True, True, True, False, False]],
            [[True, True, True, True, False, False, False]],
        ]
    )
    features = torch.randn(batch, 17, particles).masked_fill(
        ~mask, 0.0
    ).requires_grad_(True)
    tapped_features = features.detach().clone().requires_grad_(True)
    pt = torch.rand(batch, particles) + 0.5
    eta = torch.randn(batch, particles) * 0.4
    phi = torch.randn(batch, particles)
    mass = torch.rand(batch, particles) * 0.2
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px.square() + py.square() + pz.square() + mass.square())
    vectors = torch.stack((px, py, pz, energy), dim=1).masked_fill(
        ~mask, 0.0
    )
    expected = direct(features, v=vectors, mask=mask)
    actual = tap(
        tapped_features,
        vectors=vectors,
        mask=mask,
        return_states=True,
    )
    torch.testing.assert_close(
        actual["logits"], expected, atol=2.0e-6, rtol=2.0e-6
    )
    assert actual["particle_states"].shape == (batch, particles, 128)
    assert torch.equal(actual["particle_mask"], mask[:, 0])
    expected.square().sum().backward()
    actual["logits"].square().sum().backward()
    torch.testing.assert_close(
        tapped_features.grad, features.grad, atol=2.0e-6, rtol=2.0e-6
    )
    direct_parameters = dict(direct.named_parameters())
    tapped_parameters = dict(tapped_model.named_parameters())
    assert direct_parameters.keys() == tapped_parameters.keys()
    for name in direct_parameters:
        expected_gradient = direct_parameters[name].grad
        actual_gradient = tapped_parameters[name].grad
        assert (expected_gradient is None) == (actual_gradient is None)
        if expected_gradient is not None:
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=2.0e-6,
                rtol=2.0e-6,
            )


def test_uniform_and_heterogeneous_shape_resolution_is_exact() -> None:
    expected = {
        "S1_128": (1, 128),
        "S2_128": (2, 128),
        "S4_128": (4, 128),
        "S8_128": (8, 128),
        "S16_128": (16, 128),
        "S8_64": (8, 64),
        "S16_64": (16, 64),
    }
    assert {shape: resolve_uniform_shape(shape) for shape in UNIFORM_SHAPES} == expected
    assert expected["S8_128"][0] * expected["S8_128"][1] == (
        expected["S16_64"][0] * expected["S16_64"][1]
    )
    assert sum(validate_heterogeneous_allocation(HET_PHYSICS).values()) == 56
    resolved = resolve_expert_shapes(heterogeneous_allocation=HET_PHYSICS)
    assert resolved["TRACK"] == (16, 128)
    assert resolved["BASE4"] == (4, 128)
    with pytest.raises(ValueError, match="56-slot"):
        validate_heterogeneous_allocation(
            {expert: 16 for expert in EXPERT_ORDER}
        )


@pytest.mark.parametrize("shape_id", UNIFORM_SHAPES)
def test_every_uniform_token_shape_backpropagates_and_respects_padding(
    shape_id: str,
) -> None:
    token_count, token_dimension = resolve_uniform_shape(shape_id)
    torch.manual_seed(33)
    tokenizer = CanonicalSummaryTokenizer(
        expert_id="PT",
        token_count=token_count,
        token_dimension=token_dimension,
    ).eval()
    head = TokenOnlyExpertHead(token_dimension=token_dimension).eval()
    states = torch.randn(2, 6, 128, requires_grad=True)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
        ]
    )
    changed_padding = states.detach().clone()
    changed_padding[~mask] = torch.randn_like(changed_padding[~mask]) * 1000
    with torch.no_grad():
        expected = tokenizer(states.detach(), mask)
        actual = tokenizer(changed_padding, mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    tokens = tokenizer(states, mask)
    assert tokens.shape == (2, token_count, token_dimension)
    logits = head(tokens)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert states.grad is not None
    assert tokenizer.slot_queries.grad is not None
    assert head.class_query.grad is not None


def test_every_heterogeneous_physics_bank_instantiates_and_backpropagates() -> None:
    states = {
        expert: torch.randn(1, 4, 128, requires_grad=True)
        for expert in EXPERT_ORDER
    }
    mask = torch.tensor([[True, True, True, False]])
    losses = []
    for expert in EXPERT_ORDER:
        tokenizer = CanonicalSummaryTokenizer(
            expert_id=expert,
            token_count=HET_PHYSICS[expert],
            token_dimension=128,
        )
        head = TokenOnlyExpertHead(token_dimension=128)
        tokens = tokenizer(states[expert], mask)
        assert tokens.shape == (1, HET_PHYSICS[expert], 128)
        losses.append(head(tokens).square().mean())
    sum(losses).backward()
    assert all(value.grad is not None for value in states.values())


def test_multidepth_reads_block4_and_block8_with_both_masks() -> None:
    torch.manual_seed(87)
    tokenizer = MultiDepthSummaryTokenizer(
        expert_id="TRACK",
        token_count=4,
        token_dimension=128,
    ).eval()
    block4 = torch.randn(2, 5, 128, requires_grad=True)
    block8 = torch.randn(2, 5, 128, requires_grad=True)
    mask4 = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]]
    )
    mask8 = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]]
    )
    tokens = tokenizer(block4, block8, mask4, mask8)
    assert tokens.shape == (2, 4, 128)
    tokens.square().mean().backward()
    assert block4.grad is not None and torch.count_nonzero(block4.grad) > 0
    assert block8.grad is not None and torch.count_nonzero(block8.grad) > 0
    assert tokenizer.depth_block_numbers == (4, 8)


def test_token_only_head_has_no_particle_or_raw_bypass() -> None:
    signature = inspect.signature(TokenOnlyExpertHead.forward)
    assert list(signature.parameters) == ["self", "tokens"]
    head = TokenOnlyExpertHead(token_dimension=64)
    tokens = torch.randn(3, 8, 64, requires_grad=True)
    logits = head(tokens)
    assert logits.shape == (3, 10)
    with pytest.raises(TypeError):
        head(tokens, torch.randn(3, 4, 128))


def test_no_self_attention_controls_preserve_two_block_depth() -> None:
    from teacher_logit_reco.relation_expert_token_bridge.summary_tokens import (
        build_summary_tokenizer,
    )

    one = build_summary_tokenizer(
        mode="TOK_ONE_QUERY_NO_SELF",
        expert_id="BASE4",
        shape_id="S1_128",
    )
    many = build_summary_tokenizer(
        mode="TOK_K_QUERY_NO_SELF",
        expert_id="PT",
        shape_id="S4_128",
    )
    assert len(one.blocks) == len(many.blocks) == 2
    assert all(not block.enable_slot_self_attention for block in one.blocks)
    assert all(not block.enable_slot_self_attention for block in many.blocks)


def test_dual_provider_exact_sum_gates_masks_and_streaming() -> None:
    torch.manual_seed(112)
    provider = LayerwisePairBiasProvider(
        base_stem=_pair_stem(4),
        relation_stem=_pair_stem(6),
        num_layers=3,
        num_heads=8,
        topology="B_DUAL_GATED",
    )
    base_features = torch.randn(2, 4, 4, 4, requires_grad=True)
    relation_features = torch.randn(2, 6, 4, 4, requires_grad=True)
    mask = torch.tensor(
        [[[True, True, True, False]], [[True, True, False, False]]]
    )
    base_latent, relation_latent = provider.build_latents(
        base_features, relation_features, mask
    )
    provider.bind(base_latent, relation_latent, mask)
    assert torch.equal(provider.relation_scales(), torch.ones(3, 8))
    emitted = []
    for layer in range(3):
        bias = provider.bias_for_layer(layer)
        expected_base = provider.base_projections._project(
            provider.base_projections.projections[layer], base_latent
        )
        expected_relation = provider.relation_projections._project(
            provider.relation_projections.projections[layer], relation_latent
        )
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        torch.testing.assert_close(
            bias,
            (expected_base + expected_relation).masked_fill(~pair_mask, 0.0),
        )
        assert bias.shape == (2, 8, 4, 4)
        assert torch.count_nonzero(bias.masked_select(~pair_mask)) == 0
        emitted.append(bias)
    assert (
        provider.base_projections.projections[0].weight.data_ptr()
        != provider.base_projections.projections[1].weight.data_ptr()
    )
    assert (
        provider.relation_projections.projections[0].weight.data_ptr()
        != provider.relation_projections.projections[1].weight.data_ptr()
    )
    diagnostics = provider.diagnostics()
    assert diagnostics["emitted_layer_count"] == 3
    assert diagnostics["materialized_B_L_H_N_N"] is False
    assert all(len(row["shape"]) == 4 for row in diagnostics["emitted_layers"])
    sum(value.square().mean() for value in emitted).backward()
    assert provider.relation_gate_logits.grad is not None
    provider.clear()


def test_zero_relation_provider_is_exact_base_control() -> None:
    torch.manual_seed(501)
    provider = LayerwisePairBiasProvider(
        base_stem=_pair_stem(4),
        relation_stem=_pair_stem(4),
        num_layers=2,
        num_heads=8,
        topology="B_DUAL_FIXED",
        force_zero_relation=True,
    )
    base = torch.randn(1, 4, 3, 3, requires_grad=True)
    relation = torch.randn(1, 4, 3, 3, requires_grad=True)
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    base_latent, relation_latent = provider.build_latents(base, relation, mask)
    provider.bind(base_latent, relation_latent, mask)
    for layer in range(2):
        bias = provider.bias_for_layer(layer)
        expected = provider.base_projections._project(
            provider.base_projections.projections[layer], base_latent
        )
        torch.testing.assert_close(bias, expected, atol=0, rtol=0)
    provider.clear()


def test_zero_relation_fixed_and_gated_models_match_logits_and_gradients() -> None:
    def build(topology: str) -> RetbExpertModel:
        torch.manual_seed(1507)
        return RetbExpertModel(
            particle_encoder=RetbParticleEncoder(
                expert_id="BASE4",
                topology=topology,
                weaver_module=_fake_weaver(),
                force_zero_relation=True,
                activation_checkpointing=False,
            ),
            shape_id="S2_128",
        ).eval()

    fixed = build("B_DUAL_FIXED")
    gated = build("B_DUAL_GATED")
    fixed_batch = _batch()
    gated_batch = {
        name: value.detach().clone()
        if isinstance(value, torch.Tensor)
        else copy.deepcopy(value)
        for name, value in fixed_batch.items()
    }
    fixed_batch["features"].requires_grad_(True)
    gated_batch["features"].requires_grad_(True)
    fixed_output = fixed(return_details=True, **fixed_batch)
    gated_output = gated(return_details=True, **gated_batch)
    torch.testing.assert_close(
        gated_output["logits"], fixed_output["logits"], atol=0, rtol=0
    )
    assert torch.equal(
        gated_output["particle_mask"], fixed_output["particle_mask"]
    )
    fixed_output["logits"].square().sum().backward()
    gated_output["logits"].square().sum().backward()
    torch.testing.assert_close(
        gated_batch["features"].grad,
        fixed_batch["features"].grad,
        atol=0,
        rtol=0,
    )
    fixed_parameters = dict(fixed.named_parameters())
    gated_parameters = dict(gated.named_parameters())
    shared_names = sorted(set(fixed_parameters) & set(gated_parameters))
    assert shared_names
    for name in shared_names:
        fixed_gradient = fixed_parameters[name].grad
        gated_gradient = gated_parameters[name].grad
        assert (fixed_gradient is None) == (gated_gradient is None), name
        if fixed_gradient is not None:
            torch.testing.assert_close(
                gated_gradient, fixed_gradient, atol=0, rtol=0
            )
    gate_gradient = (
        gated.particle_encoder.pair_bias_provider.relation_gate_logits.grad
    )
    assert gate_gradient is not None
    assert torch.count_nonzero(gate_gradient) == 0


def test_end_to_end_encoder_ignores_every_padded_input_value() -> None:
    torch.manual_seed(1601)
    model = RetbExpertModel(
        particle_encoder=RetbParticleEncoder(
            expert_id="BASE4",
            topology="B_CONCAT",
            weaver_module=_fake_weaver(),
            measurement_embedding=True,
            activation_checkpointing=False,
        ),
        shape_id="S2_128",
    ).eval()
    clean = _batch()
    altered = {
        name: value.detach().clone()
        if isinstance(value, torch.Tensor)
        else copy.deepcopy(value)
        for name, value in clean.items()
    }
    padded = ~altered["mask"]
    altered["features"] = altered["features"].masked_fill(padded, 1.0e4)
    altered["vectors"] = altered["vectors"].masked_fill(padded, -1.0e4)
    raw_padding = ~altered["mask"].transpose(1, 2)
    altered["raw_tokens"] = altered["raw_tokens"].masked_fill(
        raw_padding, 123.0
    )
    with torch.no_grad():
        expected = model(return_details=True, **clean)
        actual = model(return_details=True, **altered)
    torch.testing.assert_close(actual["tokens"], expected["tokens"], atol=0, rtol=0)
    torch.testing.assert_close(actual["logits"], expected["logits"], atol=0, rtol=0)
    assert torch.equal(actual["particle_mask"], expected["particle_mask"])


@pytest.mark.parametrize(
    ("topology", "control"),
    [
        ("B_CONCAT", None),
        ("B_DUAL_FIXED", "base4"),
        ("B_DUAL_GATED", "base4"),
        ("B_DUAL_GATED", "zero"),
    ],
)
def test_end_to_end_base_expert_shapes_gradients_and_no_class_bypass(
    topology: str,
    control: str | None,
) -> None:
    torch.manual_seed(701)
    encoder = RetbParticleEncoder(
        expert_id="BASE4",
        topology=topology,
        weaver_module=_fake_weaver(),
        measurement_embedding=True,
        dual_base4_capacity_control=control == "base4",
        force_zero_relation=control == "zero",
        activation_checkpointing=True,
    )
    model = RetbExpertModel(
        particle_encoder=encoder,
        shape_id="S2_128",
        tokenizer_mode="TOK_MULTI_DEPTH",
    )
    batch = _batch()
    batch["features"].requires_grad_(True)
    output = model(return_details=True, **batch)
    assert output["particle_states"].shape == (2, 5, 128)
    assert output["intermediate_particle_states"].shape == (2, 5, 128)
    assert output["tokens"].shape == (2, 2, 128)
    assert output["logits"].shape == (2, 10)
    output["logits"].square().mean().backward()
    assert batch["features"].grad is not None
    assert encoder.measurement_state_embedding.embedding.weight.grad is not None
    assert not any(
        name.startswith("particle_encoder.mod.cls")
        or name.startswith("particle_encoder.mod.fc")
        for name, _ in model.named_parameters()
    )
    diagnostics = encoder.diagnostics()
    assert diagnostics["materialized_B_L_H_N_N"] is False
    if topology != "B_CONCAT":
        provider = diagnostics["pair_bias_provider"]
        assert provider["emitted_layer_count"] == 8
        assert all(len(row["shape"]) == 4 for row in provider["emitted_layers"])


def test_expert_relation_mapping_is_exactly_one_family() -> None:
    assert expert_relation_family("BASE4") is None
    for family in EXPERT_ORDER[1:]:
        assert expert_relation_family(family) == family
    with pytest.raises(ValueError, match="unknown"):
        expert_relation_family("PT_TRACK")


def test_versioned_state_dictionary_rejects_old_or_different_semantics() -> None:
    torch.manual_seed(919)
    model = RetbExpertModel(
        particle_encoder=RetbParticleEncoder(
            expert_id="BASE4",
            topology="B_CONCAT",
            weaver_module=_fake_weaver(),
            activation_checkpointing=False,
        ),
        shape_id="S1_128",
    )
    state = model.state_dict()
    clone = RetbExpertModel(
        particle_encoder=RetbParticleEncoder(
            expert_id="BASE4",
            topology="B_CONCAT",
            weaver_module=_fake_weaver(),
            activation_checkpointing=False,
        ),
        shape_id="S1_128",
    )
    clone.load_state_dict(state, strict=True)
    old = {name: value for name, value in state.items() if not name.endswith("_extra_state")}
    with pytest.raises(RuntimeError):
        clone.load_state_dict(old, strict=True)
    different = RetbExpertModel(
        particle_encoder=RetbParticleEncoder(
            expert_id="BASE4",
            topology="B_CONCAT",
            weaver_module=_fake_weaver(),
            activation_checkpointing=False,
        ),
        shape_id="S2_128",
    )
    with pytest.raises(RuntimeError):
        different.load_state_dict(state, strict=True)


def test_tokenizer_and_head_are_finite_in_fp32_and_bfloat16() -> None:
    states = torch.randn(2, 4, 128)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    for dtype in (torch.float32, torch.bfloat16):
        tokenizer = CanonicalSummaryTokenizer(
            expert_id="DENSITY",
            token_count=2,
            token_dimension=128,
        ).to(dtype=dtype).eval()
        head = TokenOnlyExpertHead(token_dimension=128).to(dtype=dtype).eval()
        tokens = tokenizer(states.to(dtype), mask)
        logits = head(tokens)
        assert torch.isfinite(tokens).all()
        assert torch.isfinite(logits).all()


def test_step3_cli_dry_run_is_nonmutating(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.build_retb_step3_contracts import main
    from teacher_logit_reco.relation_expert_token_bridge.provenance import (
        source_snapshot,
    )

    repo_root = Path(__file__).resolve().parents[1]
    snapshot = source_snapshot(repo_root)
    parent_names = (
        "artifact_layout",
        "final_select_label_manifest",
        "global_determinism",
        "hlt_replica_manifest",
        "raw_input_schema",
        "scale_train_manifest",
        "split_audit",
        "split_manifest",
        "storage_measurements",
        "validation_partition_manifest",
    )
    campaign = build_campaign_spec(
        campaign_id="retb-step3-dry-run",
        campaign_profile="miniature_test",
        source_snapshot=snapshot,
        parent_artifact_hashes={
            name: "0123456789abcdef"[index] * 64
            for index, name in enumerate(parent_names)
        },
        run_registry_hashes={"runs": "f" * 64},
    )
    write_immutable_json(tmp_path / "campaign_spec.json", campaign)
    assert main(["--campaign-root", str(tmp_path), "--dry-run"]) == 0
    assert not (tmp_path / "registry" / "retb_particle_state_tap.json").exists()
    assert "step3_bundle_sha256" in capsys.readouterr().out
