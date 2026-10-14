"""Native shared-bias ParT capacity control matched to the EV parameter count."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("weaver.nn.model.ParticleTransformer")
from weaver.nn.model.ParticleTransformer import Block, PairEmbed
from part_ev_fullscale import model as M
from part_ev_fullscale.checkpoint import atomic_torch_save, trusted_load


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def build(architecture):
    torch.manual_seed(319)
    return M.build_model(architecture, trim=False)


def inputs():
    generator = torch.Generator().manual_seed(729)
    mask = torch.arange(11)[None, None, :] < torch.tensor([11, 9, 7])[:, None, None]
    points = torch.randn(3, 2, 11, generator=generator) * mask
    features = torch.randn(3, 17, 11, generator=generator) * mask
    momentum = torch.randn(3, 3, 11, generator=generator)
    energy = (momentum.square().sum(1, keepdim=True) + 0.25).sqrt()
    vectors = torch.cat((momentum, energy), dim=1) * mask
    return points, features, vectors, mask


def is_particle_ffn(name):
    if not name.startswith("mod.blocks."):
        return False
    return name.split(".")[3] in {"fc1", "post_fc_norm", "fc2"}


def test_matched_count_and_nearest_uniform_width_are_exactly_documented():
    metadata = M.parameter_match_metadata()
    counts = {
        name: sum(parameter.numel() for parameter in build(name).parameters())
        for name in ("part", "part_ev", "part_matched")
    }
    assert counts == {"part": 2143486, "part_ev": 2209022, "part_matched": 2209790}
    assert metadata["baseline_parameters"] == counts["part"]
    assert metadata["edgevalue_parameters"] == counts["part_ev"]
    assert metadata["matched_parameters"] == counts["part_matched"]
    assert metadata["delta_vs_edgevalue_parameters"] == 768
    assert metadata["relative_delta_vs_edgevalue"] == pytest.approx(768 / 2209022)
    assert metadata["particle_ffn_hidden_dim"] == 544
    assert metadata["class_ffn_hidden_dim"] == 512
    per_hidden_unit = 8 * (128 + 1 + 2 + 128)
    assert counts["part_matched"] - counts["part"] == 32 * per_hidden_unit
    for neighbor in (543, 545):
        candidate = counts["part"] + (neighbor - 512) * per_hidden_unit
        assert abs(counts["part_matched"] - counts["part_ev"]) < abs(candidate - counts["part_ev"])


def test_only_particle_ffn_shapes_change_and_every_native_module_is_preserved():
    baseline = build("part")
    baseline_rng = torch.random.get_rng_state().clone()
    matched = build("part_matched")
    assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    assert matched.state_dict().keys() == baseline.state_dict().keys()
    assert not any(isinstance(module, M.EdgeValueAttention) for module in matched.modules())
    assert type(matched.mod.pair_embed) is PairEmbed
    assert matched.mod.pair_embed.out_dim == 8
    assert matched.mod.pair_embed.pairwise_lv_dim == 4
    assert len(matched.mod.blocks) == 8
    assert len(matched.mod.cls_blocks) == 2
    for block in matched.mod.blocks:
        assert type(block) is Block
        assert type(block.attn) is nn.MultiheadAttention
        assert block.embed_dim == 128 and block.num_heads == 8
        assert block.ffn_dim == 544
        assert block.fc1.in_features == 128 and block.fc1.out_features == 544
        assert block.post_fc_norm.normalized_shape == (544,)
        assert block.post_fc_norm.eps == 1e-5
        assert block.fc2.in_features == 544 and block.fc2.out_features == 128
        assert type(block.act) is nn.GELU
        assert block.dropout.p == block.act_dropout.p == block.attn.dropout == 0.1
    for block in matched.mod.cls_blocks:
        assert type(block) is Block
        assert block.ffn_dim == 512
        assert block.fc1.out_features == block.fc2.in_features == 512
        assert block.post_fc_norm.normalized_shape == (512,)
    for name, value in matched.state_dict().items():
        if not is_particle_ffn(name):
            torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
    assert matched.no_weight_decay() == baseline.no_weight_decay() == {"mod.cls_token"}


def test_matched_keeps_real_shared_pair_bias_at_every_particle_block():
    model = build("part_matched").eval()
    biases = []

    def observe(module, args, kwargs):
        bias = kwargs["attn_mask"]
        assert bias is not None
        biases.append(bias)

    handles = [block.attn.register_forward_pre_hook(observe, with_kwargs=True)
               for block in model.mod.blocks]
    try:
        with torch.no_grad():
            logits = model(*inputs())
    finally:
        for handle in handles:
            handle.remove()
    assert logits.shape == (3, 10) and torch.isfinite(logits).all()
    assert len(biases) == 8
    assert biases[0].shape == (3 * 8, 11, 11)
    assert all(bias is biases[0] for bias in biases)
    assert torch.count_nonzero(biases[0]) > 0


def test_every_added_ffn_parameter_is_active_and_receives_finite_gradients():
    model = build("part_matched").train()
    loss = F.cross_entropy(model(*inputs()), torch.tensor([1, 5, 8]))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for block in model.mod.blocks:
        for added_gradients in (
            block.fc1.weight.grad[512:], block.fc1.bias.grad[512:],
            block.post_fc_norm.weight.grad[512:], block.post_fc_norm.bias.grad[512:],
            block.fc2.weight.grad[:, 512:],
        ):
            assert added_gradients.numel() > 0
            assert torch.count_nonzero(added_gradients) > 0


def test_matched_checkpoint_roundtrip_restores_exact_outputs_and_compatible_layout(tmp_path):
    original = build("part_matched").eval()
    path = tmp_path / "matched.pt"
    atomic_torch_save(original.state_dict(), path)
    restored = build("part_matched").eval()
    restored.load_state_dict(trusted_load(path), strict=True)
    with torch.no_grad():
        expected = original(*inputs())
        torch.testing.assert_close(restored(*inputs()), expected, rtol=0, atol=0)
        torch.testing.assert_close(copy.deepcopy(original)(*inputs()), expected, rtol=0, atol=0)
    # The model is precisely native ParT with only native FFN modules widened;
    # no custom block forward, padding route or checkpoint key remapping exists.
    native = build("part").eval()
    for destination, source in zip(native.mod.blocks, original.mod.blocks):
        destination.ffn_dim = source.ffn_dim
        destination.fc1 = copy.deepcopy(source.fc1)
        destination.post_fc_norm = copy.deepcopy(source.post_fc_norm)
        destination.fc2 = copy.deepcopy(source.fc2)
    native.load_state_dict(original.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(native(*inputs()), expected, rtol=0, atol=0)


@pytest.mark.parametrize("architecture,count", [
    ("part", 2143486), ("part_ev", 2209022),
    ("part_plain", 2133918), ("part_plain_ev", 2208486),
])
def test_existing_architectures_remain_unmodified_after_matched_construction(architecture, count):
    before = build(architecture)
    before_rng = torch.random.get_rng_state().clone()
    build("part_matched")
    after = build(architecture)
    assert sum(p.numel() for p in before.parameters()) == count
    assert sum(p.numel() for p in after.parameters()) == count
    assert torch.equal(before_rng, torch.random.get_rng_state())
    for name, parameter in before.state_dict().items():
        torch.testing.assert_close(after.state_dict()[name], parameter, rtol=0, atol=0)
    assert all(block.ffn_dim == 512 for block in after.mod.blocks)
