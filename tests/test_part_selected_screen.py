import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from part_selected_screen import data, protocol as P
from part_selected_screen.model import build_model, CONFIGS
from part_selected_screen.normalization import identity_normalization
from part_selected_screen.region import trees_for, pack, lowest_common_ancestors, region_features
from part_selected_screen.features import Relations
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, restore_rank_state
from part_ev_fullscale.model import build_model as reference_model
from part_ev_screen.protocol import recipes
from part_ev_fullscale.optimizer import build_optimizer
from part_screen_common.engine import training_step
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads
from tests.test_part_fullscale_data import write_root, fixture_inventory


def selected_inputs(batch=3, particles=11):
    points, f, v, mask = inputs(batch, particles)
    f[:, 5:11] = 0
    f[:, 5:7] = 1  # charged hadrons
    tracks = torch.stack((f[:, 0]*.02, torch.full_like(f[:, 0], .03),
                          f[:, 1]*.04, torch.full_like(f[:, 0], .05)), 1)
    return points, torch.cat((f, tracks), 1), v, mask


def model_for(arch, trim=False):
    return build_model(arch, normalization=identity_normalization(), backend='reference_for_tests', trim=trim)


def test_extended_yaml_preserves_native_inputs_and_unclipped_raw_measurements(tmp_path):
    from part_ev_screen.data import finite_validation_loader as original, batch_fingerprint
    root = tmp_path/'val_5M/HToBB_120.root'
    columns = write_root(root, 23)
    inventory = fixture_inventory('val', [root])
    config = tmp_path/'selected.yaml'
    data.write_config(config)
    old = list(original(inventory, inventory['splits']['val']['files'][0], batch_size=7))
    new = list(data.finite_validation_loader(inventory, inventory['splits']['val']['files'][0], batch_size=7, config=config))
    assert [batch_fingerprint(b) for b in old] == [data.batch_fingerprint(b) for b in new]
    f = new[0][0]['pf_features']
    assert f.shape[1] == 21
    assert f[2, 12].max() == 1  # native error clipped
    assert f[2, 18].max() > 1   # raw error retained
    assert f[2, 17].max() > 1   # raw d0 not tanh transformed
    np.testing.assert_array_equal(f[2, 18], np.asarray(columns['part_d0err'][2])[:128])
    assert [len(b[1]['_label_']) for b in new] == [7, 7, 7, 2]


@pytest.mark.parametrize('workers', [0, 2])
def test_stream_parity_and_resume_even_with_raw_features(tmp_path, workers):
    from part_ev_fullscale.data import build_loader as original
    from part_ev_screen.data import batch_fingerprint
    files = [tmp_path/f'val_5M/HToBB_{120+i}.root' for i in range(2)]
    for p in files:
        write_root(p, 80)
    inventory = fixture_inventory('val', files)
    config = tmp_path/'selected.yaml'
    data.write_config(config)
    old, _ = original('val', inventory, 4, workers=workers)
    new, _ = data.build_loader('val', inventory, 4, config=config, workers=workers)
    try:
        a, b = iter(old), iter(new)
        expected = [batch_fingerprint(next(a)) for _ in range(9)]
        assert [data.batch_fingerprint(next(b)) for _ in range(9)] == expected
    finally:
        data.shutdown_loader(old)
        data.shutdown_loader(new)
    again, _ = data.build_loader('val', inventory, 4, config=config, workers=workers)
    try:
        it = iter(again)
        for _ in range(5):
            next(it)
        assert [data.batch_fingerprint(next(it)) for _ in range(4)] == expected[5:]
    finally:
        data.shutdown_loader(again)


def parent_walk(tree, i, j):
    left, right = tree['leaf_to_node'][i], tree['leaf_to_node'][j]
    ancestors = set()
    while left >= 0:
        ancestors.add(left)
        left = tree['parent'][left]
    while right not in ancestors:
        right = tree['parent'][right]
    return right


def test_lca_matches_parent_walk_including_unequal_depths_and_legacy_regression():
    from teacher_logit_reco.relational_part.relation_region import _batched_lca
    _, f, v, m = selected_inputs(3, 23)
    r = data.raw_tokens(f, v, m)
    trees = trees_for(v, r, m, 'reference_for_tests')
    packed = pack(trees, 23, 'cpu')
    actual = lowest_common_ancestors(packed)
    legacy = _batched_lca(packed)
    differs = False
    for b, tree in enumerate(trees):
        valid = torch.where(m[b, 0])[0].tolist()
        for i in valid:
            for j in valid:
                assert actual[b, i, j] == parent_walk(tree, i, j)
                differs |= bool(actual[b, i, j] != legacy[b, i, j])
    assert differs  # historical >= instead of bit-test is a real regression


def test_region_channel_formulas_match_legacy_with_correct_ancestor_lookup(monkeypatch):
    from teacher_logit_reco.relational_part import relation_region as old
    _, f, v, m = selected_inputs(3, 15)
    r = data.raw_tokens(f, v, m)
    trees = trees_for(v, r, m, 'reference_for_tests')
    monkeypatch.setattr(old, '_batched_lca', lowest_common_ancestors)
    expected = old.build_batched_region_raw_features(trees, r, m)
    actual = region_features(trees, r, m)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('arch', CONFIGS)
def test_native_backbone_rng_parameter_counts_and_all_added_modules_train(arch):
    from scripts.run_part_selected_screen import initial_parity
    seed_everything(101)
    reference_model('part_ev' if CONFIGS[arch][1] else 'part')
    expected_rng = torch.get_rng_state()
    seed_everything(101)
    model = model_for(arch).train()
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == P.COUNTS[arch]
    assert initial_parity(model, dict(seed=101, architecture=arch), None, torch.device('cpu'))['passed']
    values = selected_inputs()
    logits = model(*values)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2, 3])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name in ('pt', 'track_node', 'track_pair', 'region'):
        assert any(p.grad.count_nonzero() for p in getattr(model.mod.pair_embed.relations, name).parameters())
    assert len(model.mod.pair_embed.heads) == (8 if CONFIGS[arch][0] else 1)
    assert model.mod.pair_embed._raw is None
    assert all(b.attn._bias is None for b in model.mod.blocks)


@pytest.mark.parametrize('arch', CONFIGS)
def test_permutation_padding_and_actual_native_trim_alignment(arch):
    model = model_for(arch, trim=True).eval()
    p, f, v, m = selected_inputs()
    order = torch.randperm(f.shape[-1])
    with torch.no_grad():
        a = model(p, f, v, m)
        b = model(p[:, :, order], f[:, :, order], v[:, :, order], m[:, :, order])
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    observed = []
    def capture(_mod, _inputs, outputs):
        x, vv, mm, _ = outputs
        assert x.shape[1] == 17
        raw = model.mod.pair_embed._raw
        torch.testing.assert_close(raw[:, :, 0], torch.hypot(vv[:, 0], vv[:, 1]).masked_fill(~mm[:, 0], 0))
        observed.append(raw.shape[1])
    hook = model.mod.trimmer.register_forward_hook(capture)
    model.train()
    model.mod.trimmer._counter = 5
    model.mod.trimmer.target = (.5, .5)
    model(p, f, v, m)
    hook.remove()
    assert observed and observed[0] <= f.shape[-1]


@pytest.mark.parametrize('arch', CONFIGS)
def test_resume_uses_fresh_rng_or_exact_checkpoint_state(arch):
    from part_ev_screen.precision import model_precision
    seed_everything(101)
    model = model_for(arch, trim=True).train()
    model.mod.trimmer._counter = 5
    opt = build_optimizer(model, recipes()['ranger_short_v1'])
    values = selected_inputs()
    def step(m, o):
        o.zero_grad()
        m(*values).square().mean().backward()
        o.step()
    step(model, opt)
    state = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), capture_rank_state(model)
    step(model, opt)
    after = copy.deepcopy(model.state_dict())
    seed_everything(101)
    other = model_for(arch, trim=True).train()
    opt2 = build_optimizer(other, recipes()['ranger_short_v1'])
    other.load_state_dict(state[0]); opt2.load_state_dict(state[1]); restore_rank_state(other, state[2])
    step(other, opt2)
    for k, value in after.items():
        torch.testing.assert_close(value, other.state_dict()[k], rtol=0, atol=0)


def test_contrasts_are_matched_not_confounded_baseline_only():
    from part_selected_screen.reporting import contrasts
    values = dict(zip(P.SHARED+P.LAYERWISE+list(P.MATCHED), [.80, .81, .82, .83, .84, .86, .87, .90]))
    r = contrasts(values)
    assert r['SELECTED_EV_LAYERWISE_minus_EV_LAYERWISE_RANGER'] == pytest.approx(.07)
    assert r['selected_ev_by_layerwise_interaction'] == pytest.approx(.01)


def test_training_only_normalization_fit_and_reuse(tmp_path, monkeypatch):
    from part_selected_screen import normalization as norm
    from part_selected_screen.features import Normalizer
    p, f, v, m = selected_inputs()
    batch = ({'pf_points': p, 'pf_features': f, 'pf_vectors': v, 'pf_mask': m},
             {'_label_': torch.tensor([0, 1, 2])}, {})
    calls = []
    def loader(split, *a, **kw):
        calls.append(split)
        return [batch], None
    monkeypatch.setattr(norm, 'build_loader', loader)
    monkeypatch.setattr(norm, 'shutdown_loader', lambda loader: None)
    monkeypatch.setattr(norm, 'SETTINGS', dict(norm.SETTINGS, jets=3))
    donor = dict(inventory={'inventory_sha256': 'train'}, input_fingerprints={'0': data.batch_fingerprint(batch)})
    result = norm.fit(tmp_path, donor, 'reference_for_tests')
    assert calls == ['train'] and result['population'] == 'training_only'
    assert result['floors']['d0'] == pytest.approx(.03)
    for family in result['families']:
        Normalizer(family, result)
    before = (tmp_path/'inputs/normalization.json').read_bytes()
    assert norm.fit(tmp_path, donor, 'reference_for_tests') == result
    assert calls == ['train'] and before == (tmp_path/'inputs/normalization.json').read_bytes()
    donor['inventory']['inventory_sha256'] = 'different'
    with pytest.raises(ValueError): norm.fit(tmp_path, donor, 'reference_for_tests')


def test_cleanup_after_failed_forward_and_directionality():
    model = model_for('selected_layerwise_ev')
    def fail(*a): raise RuntimeError('synthetic attention failure')
    hook = model.mod.blocks[0].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError): model(*selected_inputs())
    hook.remove()
    assert model.mod.pair_embed._raw is None
    assert all(b.attn._bias is None and b.attn.attention._relation_stem is None for b in model.mod.blocks)
    p, f, v, m = selected_inputs()
    encoded = model.mod.pair_embed.relations(v, data.raw_tokens(f, v, m), m)
    assert not torch.allclose(encoded, encoded.transpose(-1, -2))
    pair_mask = m[:, :, :, None] & m[:, :, None, :]
    assert not encoded.masked_select(~pair_mask.expand_as(encoded)).count_nonzero()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires cluster GPU and precision runtime')
@pytest.mark.parametrize('arch', CONFIGS)
def test_cuda_fp16_and_fp32(arch):
    from part_ev_screen.precision import model_precision, audit_precision
    device = torch.device('cuda')
    model = model_for(arch).to(device)
    values = [x.to(device) for x in selected_inputs()]
    for training, dtype in ((True, torch.float16), (False, torch.float32)):
        model.train(training)
        with model_precision(model, device, training=training), audit_precision(model, expected_dtype=dtype) as audit:
            logits = model(*values)
            assert torch.isfinite(logits).all()
        assert audit['passed']
