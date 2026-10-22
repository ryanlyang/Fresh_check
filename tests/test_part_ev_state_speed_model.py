import copy
import pytest
import torch
from part_ev_message_study import model as old
from part_ev_fullscale.model import build_model as baseline
from part_ev_state_speed import model as M
from part_ev_state_speed.benchmark import gradient_equivalence
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads


@pytest.mark.parametrize("name", M.CONFIGS)
def test_counts_layers_and_initial_control_identity(name):
    torch.manual_seed(101)
    reference = baseline("part_ev", trim=False)
    rng = torch.get_rng_state()
    torch.manual_seed(101)
    model = M.build_model(name, trim=False)
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum(p.numel() for p in model.parameters()) == M.COUNTS[name]
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for i, block in enumerate(model.mod.blocks):
        assert (block.attn.state is not None) == (i in M.CONFIGS[name]["layers"])
        if block.attn.state is not None:
            assert block.attn.state.output.shape == (8, 16, M.CONFIGS[name]["rank"])
    for training in (False, True):
        reference.train(training)
        model.train(training)
        torch.manual_seed(409)
        expected = reference(*inputs())
        after = torch.get_rng_state()
        torch.manual_seed(409)
        actual = model(*inputs())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), after)
        assert model._state_pair_cache.normalized is None
        actual.sum().backward()  # cache was already cleared; checkpoint replay still works
        for block in model.mod.blocks:
            if block.attn.state is not None:
                assert block.attn.state.output.grad.abs().sum() > 0


def test_full_trained_state_compatible_nonzero_outputs_gradients_and_cache(monkeypatch):
    torch.manual_seed(101)
    reference = old.build_model("ev_state", trim=False)
    with torch.no_grad():
        for block in reference.mod.blocks:
            block.attn.state.output.normal_(std=.02)
    fast = M.build_model("state_fast", trim=False)
    assert reference.state_dict().keys() == fast.state_dict().keys()
    fast.load_state_dict(reference.state_dict(), strict=True)
    norm = old.normalize
    calls = []
    def recorded(x):
        if x.ndim == 4 and x.shape[-1] == 64:
            calls.append(x.shape)
        return norm(x)
    monkeypatch.setattr(old, "normalize", recorded)
    reference.eval()
    fast.eval()
    data = inputs(particles=71)
    expected = reference(*data)
    calls.clear()
    actual = fast(*data)
    assert len(calls) == 1  # shared across all eight blocks
    assert fast._state_pair_cache.stem is None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, a), (key, b) in zip(fast.named_parameters(), reference.named_parameters()):
        assert name == key
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5)
    assert fast._state_pair_cache.stem is None
    with torch.no_grad():
        fast(*inputs(particles=19))
    assert fast._state_pair_cache.stem is None


@pytest.mark.parametrize("name", M.CONFIGS)
def test_nonzero_branch_gpu_probe_math_on_cpu(name):
    result = gradient_equivalence(name, torch.device("cpu"))
    assert result["passed"] and result["tensors"] > 0 and result["checkpoint_backward"]


def test_cache_cleanup_on_exception():
    model = M.build_model("state_fast", trim=False).eval()
    def fail(_m, _args, _result):
        assert model._state_pair_cache.normalized is not None
        raise RuntimeError("deliberate failure after state block")
    handle = model.mod.blocks[0].register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        model(*inputs())
    assert model._state_pair_cache.stem is None
    assert model._state_pair_cache.normalized is None
    assert model.mod.blocks[0].attn._relation_stem is None
    handle.remove()
    assert torch.isfinite(model(*inputs())).all()


def test_shared_cache_preserves_double_precision_gradients_and_padding():
    torch.manual_seed(7)
    reference = old.StateMessage(width=7, relations=4, rank=3, heads=2, channels=2, chunk=2).double()
    with torch.no_grad():
        reference.output.normal_(std=.1)
    fast = M.FastStateMessage(copy.deepcopy(reference), M.PairCache(), chunk=4)
    args = (torch.rand(2,2,9,9,dtype=torch.double), torch.randn(2,4,9,9,dtype=torch.double),
            torch.randn(9,2,7,dtype=torch.double))
    valid = torch.tensor([[1]*9, [1]*5+[0]*4], dtype=torch.bool)
    results = []
    for branch in (reference, fast):
        xs = [x.clone().requires_grad_() for x in args]
        y = branch(*xs, valid)
        if isinstance(branch, M.FastStateMessage):
            branch.cache.clear()
        y.square().sum().backward()
        results.append((y, [x.grad for x in xs] + [p.grad for p in branch.parameters()]))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-10, atol=1e-12)
    assert results[1][0][1,:,5:].count_nonzero() == 0
    for a,b in zip(results[1][1], results[0][1]):
        torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
