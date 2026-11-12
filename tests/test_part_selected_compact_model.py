import pytest
import torch
from part_selected_compact import model as M
from part_selected_screen.normalization import identity_normalization
from part_selected_fast.trees import PackedTrees
from tests.test_part_selected_screen import selected_inputs
from tests.test_part_fullscale_model import bounded_cpu_threads


def factory(arch='compact', *, fast=True, optimized=True, trim=False):
    torch.manual_seed(101)
    provider = PackedTrees(None, oracle='reference_for_tests') if fast else None
    return M.build_model(arch, normalization=identity_normalization(), backend='reference_for_tests',
                         provider=provider, optimized=optimized, trim=trim)


def batch(n=11):
    values = selected_inputs(batch=3, particles=n)
    return ({k: v for k, v in zip(('pf_points', 'pf_features', 'pf_vectors', 'pf_mask'), values)},
            {'_label_': torch.arange(3)}, {})


def test_counts_structure_rng_and_shared_baseline_control():
    from part_selected_screen.model import build_model
    torch.manual_seed(101)
    build_model('selected_shared', normalization=identity_normalization(), backend='reference_for_tests')
    expected = torch.get_rng_state()
    for arch in M.COUNTS:
        model = factory(arch)
        assert M.validate_model(model, arch)['parameters'] == M.COUNTS[arch]
        assert torch.equal(torch.get_rng_state(), expected)
    model = factory('matched')
    assert len(model.mod.pair_embed.heads) == 1
    assert all(type(b.attn.attention) is torch.nn.MultiheadAttention for b in model.mod.blocks)
    assert not any('state.' in n or 'edge_projection' in n for n, _ in model.named_parameters())
    # Each integer particle-FFN width adds 2072 parameters; 570 is the nearest.
    assert abs(M.COUNTS['matched']-M.COUNTS['compact']) == 712
    assert 712 < abs(M.COUNTS['matched']-2072-M.COUNTS['compact'])
    compact = factory()
    assert sum(n == 'shared_state_edge.weight' for n, _ in compact.named_parameters()) == 1
    assert all(b.attn.attention.state.receiver.out_features == 16 for b in compact.mod.blocks)


def test_directed_cache_never_mirrors_upper_triangle_and_preserves_gradient():
    projection = torch.nn.Linear(64, 16, bias=False).double()
    cache = M.DirectedPairCache(projection, True)
    stem = torch.randn(2, 64, 5, 5, dtype=torch.double, requires_grad=True)
    packed, lookup = cache.get(stem)
    actual = packed[:, lookup]
    expected = projection(M.normalize(stem.permute(0, 2, 3, 1)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual[:, 0, 1], actual[:, 1, 0])
    a = torch.autograd.grad(actual.square().sum(), stem, retain_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), stem)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert cache.get(stem)[0] is packed


@pytest.mark.parametrize('n', [1, 11, 65, 128])
def test_nonzero_directed_execution_and_gradients(n):
    candidate, oracle = factory(), factory(fast=False, optimized=False)
    with torch.no_grad():
        for block in candidate.mod.blocks:
            block.attn.attention.state.output.normal_(0, .03)
    oracle.load_state_dict(candidate.state_dict())
    if n == 1:
        # The generic padded fixture masks all particles in its last two jets
        # at n=1. Native attention does not support entirely empty jets.
        from part_selected_compact.flops import synthetic
        x = synthetic(3, 1, torch.device('cpu'))
    else:
        x = selected_inputs(batch=3, particles=n)
    observations = []
    for model in (candidate, oracle):
        torch.manual_seed(818)
        out = model(*x)
        out.square().mean().backward()
        observations.append((out, {k: p.grad.clone() for k, p in model.named_parameters()}))
    torch.testing.assert_close(observations[0][0], observations[1][0], rtol=1e-4, atol=1e-5)
    for name, grad in observations[0][1].items():
        torch.testing.assert_close(grad, observations[1][1][name], rtol=3e-3, atol=3e-5, msg=lambda s: name+': '+s)
    assert candidate._compact_cache.stem is None


def test_zero_state_matches_selected_layerwise_ev():
    from part_selected_screen.model import build_model
    candidate = factory().eval()
    ordinary = build_model('selected_layerwise_ev', normalization=identity_normalization(),
                           backend='reference_for_tests', trim=False).eval()
    ordinary.load_state_dict({k: v for k, v in candidate.state_dict().items()
                             if '.state.' not in k and not k.startswith('shared_state_edge.')})
    x = selected_inputs()
    with torch.no_grad():
        torch.testing.assert_close(candidate(*x), ordinary(*x), rtol=1e-4, atol=1e-5)


def test_cache_once_backward_exception_cleanup_and_permutation():
    model = factory()
    calls = []
    hook = model.shared_state_edge.register_forward_hook(lambda *_: calls.append(1))
    model(*selected_inputs()).sum().backward()
    assert calls == [1]
    hook.remove()
    def fail(*_): raise RuntimeError('intentional')
    hook = model.mod.blocks[2].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match='intentional'):
        model(*selected_inputs())
    assert model._compact_cache.stem is None
    assert all(b.attn.attention.stem is None for b in model.mod.blocks)
    hook.remove()
    model.eval()
    x = selected_inputs()
    perm = torch.randperm(11)
    with torch.no_grad():
        torch.testing.assert_close(model(*x), model(*(v[:, :, perm] for v in x)), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('arch', M.COUNTS)
def test_composed_execution_and_exact_restart(arch):
    from part_selected_compact.parity import execution, restart
    assert execution(factory, arch, batch(), torch.device('cpu'))['passed']
    result = restart(factory(arch, trim=True), batch(), torch.device('cpu'))
    assert result['passed'] and result['exact_each_update'] and result['optimizer_and_rng_exact']
