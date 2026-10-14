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


def test_plain_matches_native_bias_free_configuration_and_has_no_pair_parameters():
    from weaver.nn.model.ParticleTransformer import ParticleTransformer

    plain = seeded_model("part_plain", trim=False).eval()
    config = M.reference_config(trim=False)
    config.update(pair_input_dim=0, pair_embed_dims=None)
    native = ParticleTransformer(**config).eval()
    native.load_state_dict(plain.mod.state_dict(), strict=True)
    assert plain.mod.pair_embed is None
    assert not any("pair_embed" in key for key in plain.state_dict())
    points, features, vectors, mask = inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            plain(points, features, vectors, mask),
            native(features, v=vectors, mask=mask), rtol=0, atol=0,
        )
    # Only the learned particle inputs matter in plain attention. Changing the
    # independent four-vector argument must not change its predictions.
    changed = vectors.clone()
    changed[:, :3] *= 1.3
    changed[:, 3:] = (changed[:, :3].square().sum(1, keepdim=True) + 0.5).sqrt() * mask
    with torch.no_grad():
        torch.testing.assert_close(
            plain(points, features, vectors, mask),
            plain(points, features, changed, mask), rtol=0, atol=0,
        )


def test_plain_shared_backbone_initialization_and_rng_match_all_four_arms():
    baseline = seeded_model("part", trim=False)
    expected_rng = torch.random.get_rng_state().clone()
    for architecture in ("part_plain", "part_plain_ev"):
        model = seeded_model(architecture, trim=False)
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
        for key, value in model.mod.state_dict().items():
            native_key = "mod." + key.replace(".attn.reference.", ".attn.")
            if not key.endswith("edge_projection"):
                torch.testing.assert_close(
                    value, baseline.state_dict()[native_key], rtol=0, atol=0,
                )
        assert all(type(block.attn) is nn.MultiheadAttention for block in model.mod.cls_blocks)
    biased_ev = seeded_model("part_ev", trim=False)
    plain_ev = seeded_model("part_plain_ev", trim=False)
    for left, right in zip(biased_ev.mod.blocks, plain_ev.mod.blocks):
        torch.testing.assert_close(left.attn.edge_projection, right.attn.edge_projection, rtol=0, atol=0)
    biased_stem = biased_ev.mod.pair_embed.reference.embed.prefix.state_dict()
    plain_stem = plain_ev.edge_pair_values.reference.embed.state_dict()
    assert biased_stem.keys() == plain_stem.keys()
    for key in biased_stem:
        torch.testing.assert_close(biased_stem[key], plain_stem[key], rtol=0, atol=0)
    # Plain+EV removes the entire final 64->8 Conv/BN head rather than retaining
    # dormant trainable weights (which would also break default DDP reduction).
    assert plain_ev.mod.pair_embed is None
    assert plain_ev.edge_pair_values.reference.out_dim == 64
    assert not any("projection" in key for key in plain_stem)
    head_params = sum(p.numel() for p in biased_ev.mod.pair_embed.reference.embed.projection.parameters())
    assert sum(p.numel() for p in biased_ev.parameters()) - sum(p.numel() for p in plain_ev.parameters()) == head_params


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("trim", [False, True])
def test_zero_plain_ev_matches_plain_including_dropout_and_trimming(training, trim):
    plain = seeded_model("part_plain", trim=trim).train(training)
    ev = seeded_model("part_plain_ev", trim=trim).train(training)
    plain.mod.trimmer._counter = ev.mod.trimmer._counter = 5
    plain.mod.trimmer.target = ev.mod.trimmer.target = (0.5, 0.5)
    observed = []

    def observe_attention(module, args, kwargs):
        assert kwargs.get("attn_mask") is None
        stem, valid = module._relation_stem, module._query_valid
        mask = valid[:, None, :, None] & valid[:, None, None, :]
        assert torch.count_nonzero(stem.masked_select(~mask)) == 0
        observed.append(stem.shape[-1])

    handles = [
        block.attn.register_forward_pre_hook(observe_attention, with_kwargs=True)
        for block in ev.mod.blocks
    ]
    with torch.no_grad():
        for block in ev.mod.blocks:
            block.attn.edge_projection.zero_()
        torch.manual_seed(812)
        random.seed(812)
        expected = plain(*inputs())
        torch.manual_seed(812)
        random.seed(812)
        actual = ev(*inputs())
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(observed) == 8
    if trim and training:
        assert observed == [9] * 8
    assert ev.edge_pair_values._trimmed_vectors is None
    assert ev.edge_pair_values._trimmed_mask is None
    assert all(block.attn._relation_stem is None for block in ev.mod.blocks)


def test_plain_ev_uses_pair_features_only_in_value_messages():
    ev = seeded_model("part_plain_ev", trim=False).eval()
    points, features, vectors, mask = inputs()
    changed = vectors.clone()
    changed[:, :3] *= 1.3
    changed[:, 3:] = (changed[:, :3].square().sum(1, keepdim=True) + 0.5).sqrt() * mask
    first_attention_outputs = []

    def observe_native_attention(module, args, kwargs, output):
        assert kwargs.get("attn_mask") is None
        first_attention_outputs.append(output[0].detach().clone())

    handle = ev.mod.blocks[0].attn.reference.register_forward_hook(
        observe_native_attention, with_kwargs=True,
    )
    with torch.no_grad():
        original = ev(points, features, vectors, mask)
        modified = ev(points, features, changed, mask)
    handle.remove()
    # The first block's ordinary attention has identical Q/K/V and no pairwise
    # bias; later blocks may change because their previous EV messages changed.
    torch.testing.assert_close(first_attention_outputs[0], first_attention_outputs[1], rtol=0, atol=0)
    assert not torch.allclose(original, modified)


@pytest.mark.parametrize("architecture", ["part_plain", "part_plain_ev"])
def test_plain_arms_have_no_unused_trainable_parameters(architecture):
    model = seeded_model(architecture, trim=False).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(*inputs()), torch.tensor([1, 4, 8]))
        loss.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        assert not missing, missing
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        if architecture == "part_plain_ev":
            assert any(torch.count_nonzero(p.grad) for p in model.edge_pair_values.parameters())
        optimizer.step()


def test_plain_ev_syncbn_roundtrip_and_error_cleanup(monkeypatch):
    original = seeded_model("part_plain_ev", trim=True).eval()
    restored = seeded_model("part_plain_ev", trim=True).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    copied = copy.deepcopy(original)
    assert copied.edge_pair_values is not original.edge_pair_values
    assert all(
        bound is block.attn
        for bound, block in zip(copied.edge_pair_values._edges, copied.mod.blocks)
    )
    with torch.no_grad():
        torch.testing.assert_close(original(*inputs()), restored(*inputs()), rtol=0, atol=0)
        torch.testing.assert_close(original(*inputs()), copied(*inputs()), rtol=0, atol=0)
    count = sum(isinstance(module, nn.BatchNorm1d) for module in restored.modules())
    converted = nn.SyncBatchNorm.convert_sync_batchnorm(restored)
    assert sum(isinstance(module, nn.SyncBatchNorm) for module in converted.modules()) == count
    assert not any(isinstance(module, nn.BatchNorm1d) for module in converted.modules())
    with torch.no_grad():
        torch.testing.assert_close(original(*inputs()), converted(*inputs()), rtol=0, atol=0)

    def fail(*args, **kwargs):
        raise RuntimeError("injected plain attention failure")

    monkeypatch.setattr(original.mod.blocks[2], "forward", fail)
    with pytest.raises(RuntimeError, match="injected plain"):
        original(*inputs())
    assert original.edge_pair_values._trimmed_vectors is None
    assert original.edge_pair_values._trimmed_mask is None
    assert all(block.attn._relation_stem is None for block in original.mod.blocks)


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="CPU Gloo distributed runtime unavailable",
)
@pytest.mark.parametrize("architecture", ["part_plain", "part_plain_ev"])
def test_plain_arms_default_ddp_complete_two_iterations(architecture, tmp_path):
    from datetime import timedelta
    from torch.nn.parallel import DistributedDataParallel

    if torch.distributed.is_initialized():
        pytest.skip("An external process group already owns the test runtime")
    torch.distributed.init_process_group(
        backend="gloo", init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0, world_size=1, timeout=timedelta(seconds=30),
    )
    try:
        # The second forward checks that the previous gradient reduction
        # finished without find_unused_parameters or static-graph workarounds.
        model = DistributedDataParallel(seeded_model(architecture, trim=False).train())
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(*inputs()), torch.tensor([1, 4, 8])).backward()
    finally:
        torch.distributed.destroy_process_group()

