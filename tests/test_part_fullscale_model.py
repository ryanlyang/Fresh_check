"""Real pinned-Weaver parity and isolated shared-bias EdgeValue semantics."""

import copy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("weaver.nn.model.ParticleTransformer")
from part_ev_fullscale import model as M


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(batch=3, particles=11):
    generator = torch.Generator().manual_seed(1123)
    mask = torch.arange(particles)[None, None, :] < torch.tensor([particles - i * 2 for i in range(batch)])[:, None, None]
    features = torch.randn(batch, 17, particles, generator=generator) * mask
    momentum = torch.randn(batch, 3, particles, generator=generator)
    energy = (momentum.square().sum(dim=1, keepdim=True) + 0.25).sqrt()
    vectors = torch.cat((momentum, energy), dim=1) * mask
    points = torch.randn(batch, 2, particles, generator=generator) * mask
    return points, features, vectors, mask


def seeded_model(architecture, **kwargs):
    torch.manual_seed(319)
    return M.build_model(architecture, **kwargs)


def reference_state_from_ev(model):
    result = {}
    prefix_size = len(model.mod.pair_embed.reference.embed.prefix)
    for key, value in model.state_dict().items():
        if key.endswith("edge_projection"):
            continue
        if key.startswith("mod.pair_embed.reference.embed.prefix."):
            key = key.replace("mod.pair_embed.reference.embed.prefix.", "mod.pair_embed.embed.")
        elif key.startswith("mod.pair_embed.reference.embed.projection."):
            suffix = key.removeprefix("mod.pair_embed.reference.embed.projection.")
            index, tail = suffix.split(".", 1)
            key = "mod.pair_embed.embed.{}.{}".format(int(index) + prefix_size, tail)
        key = key.replace(".attn.reference.", ".attn.")
        result[key] = value
    return result


def test_native_wrapper_matches_official_get_model_and_parameter_names():
    path = Path(__file__).resolve().parents[1] / "particle_transformer/networks/example_ParticleTransformer.py"
    specification = importlib.util.spec_from_file_location("official_part_network", path)
    official = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(official)
    config = SimpleNamespace(
        input_dicts={"pf_features": list(range(17))}, label_value=list(range(10)),
        input_names=["pf_features"], input_shapes={"pf_features": (None, 17, 128)},
    )
    torch.manual_seed(319)
    original, _ = official.get_model(config, trim=False)
    model = seeded_model("part", trim=False)
    assert model.state_dict().keys() == original.state_dict().keys()
    for key in model.state_dict():
        torch.testing.assert_close(model.state_dict()[key], original.state_dict()[key], rtol=0, atol=0)
    model.eval()
    original.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(*inputs()), original(*inputs()), rtol=0, atol=0)
    assert model.no_weight_decay() == {"mod.cls_token"}
    assert len(model.mod.blocks) == 8
    assert len(model.mod.cls_blocks) == 2
    assert model.mod.blocks[0].attn.dropout == 0.1
    assert model.mod.cls_blocks[0].attn.dropout == 0


def test_shared_initial_state_rng_and_exact_parameter_increment():
    base = seeded_model("part")
    baseline_rng = torch.random.get_rng_state().clone()
    ev = seeded_model("part_ev")
    assert torch.equal(baseline_rng, torch.random.get_rng_state())
    reference = reference_state_from_ev(ev)
    assert reference.keys() == base.state_dict().keys()
    for key, value in base.state_dict().items():
        torch.testing.assert_close(reference[key], value, rtol=0, atol=0)
    increment = sum(p.numel() for p in ev.parameters()) - sum(p.numel() for p in base.parameters())
    assert increment == 8 * 8 * 16 * 64 == 65536
    assert all(block.attn.edge_projection.shape == (8, 16, 64) for block in ev.mod.blocks)
    assert all(type(block.attn) is nn.MultiheadAttention for block in ev.mod.cls_blocks)
    assert ev.mod.pair_embed.reference.is_symmetric
    assert not ev.mod.pair_embed.reference.remove_self_pair


@pytest.mark.parametrize("training", [False, True])
def test_zero_edgevalue_matches_native_forward_including_training_dropout(training):
    base = seeded_model("part", trim=False).train(training)
    ev = seeded_model("part_ev", trim=False).train(training)
    with torch.no_grad():
        for block in ev.mod.blocks:
            block.attn.edge_projection.zero_()
        torch.manual_seed(812)
        expected = base(*inputs())
        torch.manual_seed(812)
        actual = ev(*inputs())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for block in ev.mod.blocks:
        assert block.attn._relation_stem is None
        assert block.attn._query_valid is None
    assert ev.mod.pair_embed._trimmed_mask is None


def test_mask_follows_native_random_trim_and_shared_bias_population():
    base = seeded_model("part", trim=True).train()
    ev = seeded_model("part_ev", trim=True).train()
    base.mod.trimmer._counter = ev.mod.trimmer._counter = 5
    base.mod.trimmer.target = ev.mod.trimmer.target = (0.5, 0.5)
    observed = []

    def check_stem(module, args):
        stem, mask = module._relation_stem, module._query_valid
        observed.append(stem.shape[-1])
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        assert torch.count_nonzero(stem.masked_select(~pair_mask)) == 0

    handle = ev.mod.blocks[0].attn.register_forward_pre_hook(check_stem)
    with torch.no_grad():
        for block in ev.mod.blocks:
            block.attn.edge_projection.zero_()
        torch.manual_seed(41)
        random.seed(41)
        expected = base(*inputs())
        torch.manual_seed(41)
        random.seed(41)
        actual = ev(*inputs())
    handle.remove()
    assert observed == [9]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference = reference_state_from_ev(ev)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(reference[key], value, rtol=0, atol=0)


def test_edge_projection_and_native_parameters_receive_gradients_even_from_zero_ev():
    model = seeded_model("part_ev", trim=False).train()
    with torch.no_grad():
        for block in model.mod.blocks:
            block.attn.edge_projection.zero_()
    logits = model(*inputs())
    F.cross_entropy(logits, torch.tensor([1, 4, 8])).backward()
    for block in model.mod.blocks:
        assert block.attn.edge_projection.grad is not None
        assert torch.isfinite(block.attn.edge_projection.grad).all()
        assert torch.count_nonzero(block.attn.edge_projection.grad) > 0
        assert torch.count_nonzero(block.attn.reference.in_proj_weight.grad) > 0
    stem = model.mod.pair_embed.reference.embed.prefix
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in stem.parameters())


def test_nonzero_edgevalue_matches_explicit_message_and_same_dropout_weights():
    torch.manual_seed(93)
    attention = nn.MultiheadAttention(16, 4, dropout=0.35)
    ordinary = copy.deepcopy(attention)
    edge = M.EdgeValueAttention(attention, relation_width=7).train()
    query = torch.randn(6, 2, 16)
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    stem = torch.randn(2, 7, 6, 6)
    stem = stem.masked_fill(~(mask[:, None, :, None] & mask[:, None, None, :]), 0)
    edge.bind(stem, mask)
    torch.manual_seed(591)
    output, used_weights = edge(query, query, query, key_padding_mask=~mask, average_attn_weights=False)
    torch.manual_seed(591)
    base_output, expected_weights = ordinary(query, query, query, key_padding_mask=~mask, average_attn_weights=False)
    torch.testing.assert_close(used_weights, expected_weights, rtol=0, atol=0)
    assert torch.count_nonzero(used_weights == 0) > 0
    pair_values = torch.einsum("hod,bdij->bhijo", edge.edge_projection, stem)
    heads = (used_weights.unsqueeze(-1) * pair_values).sum(dim=3)
    extra = heads.permute(2, 0, 1, 3).reshape(6, 2, 16)
    expected = base_output + F.linear(extra, ordinary.out_proj.weight) * mask.T.unsqueeze(-1)
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(output[:, 1][~mask[1]], base_output[:, 1][~mask[1]], rtol=0, atol=0)


def test_forward_error_clears_transient_pair_context(monkeypatch):
    model = seeded_model("part_ev", trim=False)

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected attention failure")

    monkeypatch.setattr(model.mod.blocks[2], "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model(*inputs())
    assert model.mod.pair_embed._trimmed_mask is None
    assert all(block.attn._relation_stem is None for block in model.mod.blocks)


def test_syncbn_conversion_preserves_native_pair_and_stem_layout():
    base = seeded_model("part")
    ev = seeded_model("part_ev")
    base_bn = sum(isinstance(module, nn.BatchNorm1d) for module in base.modules())
    ev_bn = sum(isinstance(module, nn.BatchNorm1d) for module in ev.modules())
    assert base_bn == ev_bn
    converted = nn.SyncBatchNorm.convert_sync_batchnorm(ev)
    assert sum(isinstance(module, nn.SyncBatchNorm) for module in converted.modules()) == base_bn
    assert not any(isinstance(module, nn.BatchNorm1d) for module in converted.modules())
    assert converted.mod.trimmer._counter == 0


def test_state_dict_roundtrip_rebinds_pair_context_and_preserves_outputs():
    original = seeded_model("part_ev", trim=True).eval()
    restored = seeded_model("part_ev", trim=True).eval()
    original.mod.trimmer._counter = 7
    restored.load_state_dict(original.state_dict(), strict=True)
    # Native counter is intentionally external to state_dict, as documented.
    assert restored.mod.trimmer._counter == 0
    restored.mod.trimmer._counter = original.mod.trimmer._counter
    with torch.no_grad():
        torch.testing.assert_close(original(*inputs()), restored(*inputs()), rtol=0, atol=0)
    assert all("_trimmed_mask" not in key and "_relation_stem" not in key
               for key in original.state_dict())


def test_runtime_metadata_and_amp_setting():
    metadata = M.reference_metadata()
    assert metadata["weaver_version"] == "0.4.17"
    assert metadata["weaver_particle_transformer_sha256"] == M.WEAVER_SOURCE_SHA256
    assert M.build_model("part_ev", use_amp=True).mod.use_amp is True
    with pytest.raises(ValueError, match="architecture"):
        M.build_model("selected")

