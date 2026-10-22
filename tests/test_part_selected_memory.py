"""Allocation savings without changing model capacity or the screen recipe."""
import copy
import numpy as np
import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks

from part_selected_screen.model import build_model, packed_ev_stem, CONFIGS
from part_selected_screen.normalization import identity_normalization
from part_selected_screen.memory import execution_parity, compare, PARITY
from part_selected_screen import protocol as P, data
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state
from part_ev_fullscale.checkpoint import restore_rank_state
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_screen.protocol import recipes
from tests.test_part_selected_screen import selected_inputs
from tests.test_part_fullscale_model import bounded_cpu_threads


@pytest.mark.parametrize('particles', [1, 7])
def test_packed_stem_values_mask_gradient_and_nonaliasing(particles):
    stem = torch.randn(2, 64, particles*particles, requires_grad=True)
    before = stem.detach().clone()
    valid = torch.ones(2, particles, dtype=torch.bool)
    valid[1, -1] = False
    old = stem.reshape(2, 64, particles, particles).masked_fill(
        ~(valid[:, None, :, None] & valid[:, None, None, :]), 0)
    new = packed_ev_stem(stem, valid)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert new.untyped_storage().data_ptr() != stem.untyped_storage().data_ptr()
    torch.testing.assert_close(stem, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(old.square().sum(), stem)[0],
                               torch.autograd.grad(new.square().sum(), stem)[0], rtol=0, atol=0)


def test_eight_ev_blocks_share_one_saved_stem_not_eight_copies():
    b, h, n, d = 2, 8, 13, 64
    stem = torch.randn(b, d, n*n, requires_grad=True)
    weights = torch.randn(b, h, n, n, requires_grad=True)
    valid = torch.ones(b, n, dtype=torch.bool)
    storage_sets = []
    for packed in (False, True):
        x = packed_ev_stem(stem, valid) if packed else stem.reshape(b, d, n, n).clone()
        saved = []
        def observe(t):
            if t.shape == (b*n, n, d):
                saved.append(t)
            return t
        with saved_tensors_hooks(observe, lambda t: t):
            outputs = [torch.einsum('bhij,bdij->bhid', weights, x) for _ in range(8)]
        assert len(saved) == 8
        assert all(t.stride() == (n*d, d, 1) for t in saved)
        pointers = {t.untyped_storage().data_ptr() for t in saved}
        if packed:
            assert pointers == {x.untyped_storage().data_ptr()}
        storage_sets.append(len(pointers))
        sum(y.square().mean() for y in outputs).backward()
    assert storage_sets == [8, 1]


def normalized_model(arch, *, device='cpu', optimized=True):
    normal = identity_normalization()
    # Exercise actual robust scaling/clipping rather than only identity scales.
    for family in normal['families'].values():
        family['center'] = [i*.13-.5 for i in range(len(family['center']))]
        family['scale'] = [.2+i*.07 for i in range(len(family['scale']))]
    return build_model(arch, normalization=normal, backend='reference_for_tests',
                       trim=True, memory_optimized=optimized).to(device)


def batch_for(device='cpu', particles=19):
    p, f, v, m = selected_inputs(3, particles)
    # Neutral/invalid tracks and padding must remain masked identically.
    f[0, 5:7, 1:4] = 0
    f[1, 18, 1:4] = 0
    return ({k: t.to(device) for k, t in zip(
        ('pf_points', 'pf_features', 'pf_vectors', 'pf_mask'), (p, f, v, m))},
        {'_label_': torch.tensor([0, 1, 2], device=device)}, {})


def assert_rng_equal(first, second):
    assert first['python'] == second['python']
    assert first['numpy'][0] == second['numpy'][0]
    np.testing.assert_array_equal(first['numpy'][1], second['numpy'][1])
    assert first['numpy'][2:] == second['numpy'][2:]
    torch.testing.assert_close(first['torch'], second['torch'], rtol=0, atol=0)
    assert first['trimmers'] == second['trimmers']
    assert len(first['cuda']) == len(second['cuda'])
    for a, b in zip(first['cuda'], second['cuda']):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('arch', CONFIGS)
def test_fp32_fp16_outputs_gradients_counts_and_diagnostic_state_restoration(arch):
    seed_everything(101)
    model = normalized_model(arch).eval()
    model.mod.trimmer._counter = 9
    batch = batch_for()
    before = copy.deepcopy(model.state_dict())
    rng = capture_rank_state(model)
    result = execution_parity(model, batch, torch.device('cpu'))
    assert result['passed'] and result['events'] == 3
    assert set(result['modes']) == {'fp32_eval', 'fp32_train', 'fp16_train'}
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert not model.training and model.mod.trimmer.enabled
    assert model.mod.pair_embed.memory_optimized and model.mod.pair_embed.relations.memory_optimized
    assert all(p.grad is None for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == P.COUNTS[arch]
    assert_rng_equal(rng, capture_rank_state(model))


def test_diagnostic_restores_on_failure(monkeypatch):
    import part_selected_screen.memory as memory
    model = normalized_model('selected_shared_ev').train()
    batch = batch_for()
    before = copy.deepcopy(model.state_dict())
    rng = capture_rank_state(model)
    def fail(*args, **kwargs):
        raise ValueError('injected mismatch')
    monkeypatch.setattr(memory, 'compare', fail)
    with pytest.raises(ValueError, match='injected mismatch'):
        execution_parity(model, batch, torch.device('cpu'))
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    assert model.training and model.mod.trimmer.enabled
    assert model.mod.pair_embed._raw is None
    assert model.mod.pair_embed.memory_optimized and model.mod.pair_embed.relations.memory_optimized
    assert_rng_equal(rng, capture_rank_state(model))


@pytest.mark.parametrize('gradients', [False, True])
def test_parity_rejects_wrong_or_nonfinite_values(gradients):
    for value in (1.1, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            compare(torch.tensor([value]), torch.ones(1), PARITY['fp16'], gradients=gradients)


@pytest.mark.parametrize('arch', ['selected_shared', 'selected_layerwise'])
def test_non_ev_never_constructs_ev_message(arch, monkeypatch):
    import part_selected_screen.model as module
    model = normalized_model(arch).eval()
    def fail(*args):
        raise AssertionError('Non-EV model must not pack EV stem')
    monkeypatch.setattr(module, 'packed_ev_stem', fail)
    with torch.no_grad():
        model(*selected_inputs())


def test_amp_track_linear_effective_input_identical_without_fp32_pair_cat():
    model = normalized_model('selected_shared').eval()
    _, f, v, m = selected_inputs()
    relation = model.mod.pair_embed.relations
    observed = []
    hook = relation.track_pair[0].register_forward_pre_hook(
        lambda module, args: observed.append((args[0].dtype, args[0].half().detach().clone())))
    try:
        for optimized in (False, True):
            relation.memory_optimized = optimized
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.float16):
                relation(v, data.raw_tokens(f, v, m), m)
    finally:
        hook.remove()
    assert observed[0][0] == torch.float32 and observed[1][0] == torch.float16
    torch.testing.assert_close(observed[0][1], observed[1][1], rtol=0, atol=0)


def test_ranger_updates_and_lookahead_remain_equivalent():
    seed_everything(101)
    model = normalized_model('selected_layerwise_ev').train()
    before = copy.deepcopy(model.state_dict())
    state = capture_rank_state(model)
    batch = batch_for()
    values = [batch[0][k] for k in ('pf_points', 'pf_features', 'pf_vectors', 'pf_mask')]
    after = []
    for optimized in (False, True):
        model.load_state_dict(before)
        restore_rank_state(model, state)
        model.mod.pair_embed.memory_optimized = model.mod.pair_embed.relations.memory_optimized = optimized
        optimizer = build_optimizer(model, recipes()['ranger_short_v1'])
        # Include the six-step Lookahead synchronization and native trimming.
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(*values), batch[1]['_label_'])
            loss.backward()
            optimizer.step()
        after.append(copy.deepcopy(model.state_dict()))
    for key in after[0]:
        torch.testing.assert_close(after[0][key], after[1][key], rtol=2e-5, atol=2e-6, msg=key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires cluster GPU')
@pytest.mark.parametrize('arch', CONFIGS)
def test_cuda_execution_parity(arch):
    seed_everything(101)
    model = normalized_model(arch, device='cuda')
    assert execution_parity(model, batch_for(particles=128), torch.device('cuda'))['passed']
