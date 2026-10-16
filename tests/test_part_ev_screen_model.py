import copy
import pytest
import torch
from part_ev_screen.model import (COUNTS, JetContext, RelationGate, ScreenAttention, build_model)
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads


def make(name, **kwargs):
    torch.manual_seed(101)
    return build_model(name, **kwargs)


@pytest.mark.parametrize("training", [False, True])
def test_identity_initialization_counts_and_matched_rng(training):
    ev = make("part_ev", trim=False).train(training)
    initial_rng = torch.get_rng_state().clone()
    for name in list(COUNTS)[2:]:
        ev = make("part_ev", trim=False).train(training)
        model = make(name, trim=False).train(training)
        assert torch.equal(torch.get_rng_state(), initial_rng)
        for key, value in ev.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        torch.manual_seed(181)
        expected = ev(*inputs())
        torch.manual_seed(181)
        actual = model(*inputs())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert sum(p.numel() for p in model.parameters()) == COUNTS[name]


def test_context_pooling_padding_permutation_and_gradient():
    context = JetContext(width=8, hidden=3, relations=5)
    query = torch.randn(4, 2, 8, requires_grad=True)
    valid = torch.tensor([[1,1,1,0], [1,1,0,0]], dtype=torch.bool)
    assert torch.equal(context(query, valid), torch.ones(2,5))
    context(query, valid).sum().backward()
    assert context.last.weight.grad.abs().sum() > 0
    with torch.no_grad():
        context.last.weight.normal_()
    changed = query.detach().clone().masked_fill(~valid.T[...,None], 1000)
    torch.testing.assert_close(context(query, valid), context(changed, valid))
    perm = torch.tensor([2,0,3,1])
    torch.testing.assert_close(context(query, valid), context(query[perm], valid[:,perm]))
    assert ((context(query, valid) >= 0) & (context(query, valid) <= 2)).all()
    context.zero_grad()
    context(query,valid).square().sum().backward()
    assert context.first.weight.grad.abs().sum() > 0
    assert query.grad.abs().sum() > 0


def test_gate_factored_matches_explicit_forward_and_gradients():
    torch.manual_seed(45)
    gate = RelationGate(heads=2, rank=2, channels=3, relations=4, query_chunk=2).double()
    with torch.no_grad():
        gate.basis.normal_(std=.2)
    explicit = copy.deepcopy(gate)
    w = torch.rand(2,2,5,5, dtype=torch.double, requires_grad=True)
    e = torch.randn(2,4,5,5, dtype=torch.double, requires_grad=True)
    v = torch.randn(2,2,5,3, dtype=torch.double, requires_grad=True)
    w2,e2,v2 = [x.detach().clone().requires_grad_() for x in (w,e,v)]
    valid = torch.tensor([[1,1,1,1,1], [1,1,1,0,0]], dtype=torch.bool)
    actual = gate(w,e,v,valid)
    coefficients = explicit.coefficients(e2, valid[:,:,None] & valid[:,None,:])
    gates = torch.einsum("bhijr,hrd->bhijd", coefficients, explicit.basis.tanh()) / 2
    expected = (w2[...,None] * gates * v2[:,:,None,:,:]).sum(3)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward(); expected.square().sum().backward()
    for a,b in zip((w,e,v,*gate.parameters()), (w2,e2,v2,*explicit.parameters())):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-8)
    assert torch.count_nonzero(actual[1,:,3:]) == 0


def test_zero_gate_learns_basis_then_coefficients():
    gate = RelationGate(heads=2, rank=2, channels=3, relations=4)
    args = (torch.rand(2,2,3,3), torch.randn(2,4,3,3), torch.randn(2,2,3,3), torch.ones(2,3,dtype=torch.bool))
    first = gate(*args)
    assert torch.count_nonzero(first) == 0
    first.sum().backward()
    assert gate.basis.grad.abs().sum() > 0
    assert torch.count_nonzero(gate.coefficient.weight.grad) == 0
    with torch.no_grad():
        gate.basis.add_(gate.basis.grad, alpha=-.1)
    gate.zero_grad()
    gate(*args).sum().backward()
    assert gate.coefficient.weight.grad.abs().sum() > 0


def test_combined_components_match_individual_initialization_and_backward():
    both = make("part_ev_context_relation_gate", trim=False)
    context = make("part_ev_context", trim=False)
    gate = make("part_ev_relation_gate", trim=False)
    for key, value in both.state_dict().items():
        if ".context." in key:
            torch.testing.assert_close(value, context.state_dict()[key], rtol=0, atol=0)
        if ".gate." in key:
            torch.testing.assert_close(value, gate.state_dict()[key], rtol=0, atol=0)
    both(*inputs()).square().sum().backward()
    for block in both.mod.blocks:
        assert block.attn.context.last.weight.grad.abs().sum() > 0
        assert block.attn.gate.basis.grad.abs().sum() > 0
        assert block.attn._relation_stem is None


@pytest.mark.parametrize('training', [False,True])
def test_combined_matches_explicit_pair_values_with_native_bias_and_dropout(training):
    from part_ev_fullscale.model import EdgeValueAttention
    from torch.nn import functional as F
    torch.manual_seed(7)
    attention = ScreenAttention(EdgeValueAttention(torch.nn.MultiheadAttention(128,8,dropout=.3)),
                                context=True,gate=True,seed=101,layer=0).train(training)
    with torch.no_grad():
        attention.context.last.weight.normal_(std=.05)
        attention.gate.basis.normal_(std=.1)
        attention.reference.in_proj_bias.normal_(std=.1)
        attention.reference.out_proj.bias.normal_(std=.1)
    reference = copy.deepcopy(attention)
    q = torch.randn(5,2,128,requires_grad=True)
    e = torch.randn(2,64,5,5,requires_grad=True)
    q2,e2 = [v.detach().clone().requires_grad_() for v in (q,e)]
    valid = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
    mask = valid[:,:,None] & valid[:,None,:]
    stem = e.masked_fill(~mask[:,None],0)
    stem2 = e2.masked_fill(~mask[:,None],0)
    attention.bind(stem,valid)
    torch.manual_seed(90)
    actual,_ = attention(q,q,q,key_padding_mask=~valid,average_attn_weights=False)
    torch.manual_seed(90)
    ordinary, weights = reference.reference(q2,q2,q2,key_padding_mask=~valid,need_weights=True,average_attn_weights=False)
    values = F.linear(q2,reference.reference.in_proj_weight[256:],reference.reference.in_proj_bias[256:]).reshape(5,2,8,16).permute(1,2,0,3)
    scale = reference.context(q2,valid)
    edge_values = torch.einsum('hdf,bfij->bhijd', reference.edge_projection,stem2 * scale[:,:,None,None])
    coeff = reference.gate.coefficients(stem2,mask)
    gates = torch.einsum('bhijr,hrd->bhijd',coeff,reference.gate.basis.tanh())/4
    extra_heads = (weights[...,None] * (edge_values + gates * values[:,:,None,:,:])).sum(3)
    joined = extra_heads.permute(2,0,1,3).reshape(5,2,128)
    expected = ordinary + F.linear(joined,reference.reference.out_proj.weight,None) * valid.T[...,None]
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    actual.square().sum().backward(); expected.square().sum().backward()
    for a,b in zip((q,e,*attention.parameters()),(q2,e2,*reference.parameters())):
        torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-5)


def test_gate_remains_active_with_ev_zero_and_full_model_is_permutation_invariant():
    baseline = make('part',trim=False).eval()
    extended = make('part_ev_context_relation_gate',trim=False).eval()
    with torch.no_grad():
        for block in extended.mod.blocks: block.attn.edge_projection.zero_()
        sample = inputs()
        torch.testing.assert_close(extended(*sample),baseline(*sample),rtol=0,atol=0)
        for block in extended.mod.blocks: block.attn.gate.basis.normal_(std=.1)
        result = extended(*sample)
        assert not torch.allclose(result,baseline(*sample))
        permutation = torch.tensor([3,0,10,4,1,5,6,7,2,8,9])
        permuted = extended(*(t[:,:,permutation] for t in sample))
        torch.testing.assert_close(permuted,result,rtol=1e-4,atol=2e-6)
