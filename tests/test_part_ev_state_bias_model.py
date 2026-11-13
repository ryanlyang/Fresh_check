import copy
import pytest
import torch
from torch.nn import functional as F

from part_ev_state_bias import model as M
from part_ev_state_compare.model import build_model as reference
from part_ev_gpu_scaling.benchmark import fixed_batch, tensors
from part_ev_fullscale.checkpoint import seed_everything
from tests.test_part_fullscale_model import bounded_cpu_threads


def make(*, old=False, **kwargs):
    seed_everything(101)
    return reference('state_layerwise', **kwargs) if old else M.build_model(**kwargs)


def data(n=9):
    values, y = tensors(fixed_batch(3, n), torch.device('cpu'))
    values[-1][1, :, -2:] = False
    return values, y


def active(model, bias=True):
    with torch.no_grad():
        for a in M.attentions(model):
            a.state.output.normal_(std=.02)
            if bias:
                a.bias_head.normal_(std=.03)


def test_count_old_initialization_rng_and_zero_head():
    old = make(old=True, trim=False)
    rng = torch.get_rng_state()
    new = make(trim=False)
    assert sum(p.numel() for p in new.parameters()) == 2329510
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    heads = [p for n, p in new.named_parameters() if n.endswith('bias_head')]
    assert len(heads) == 8 and sum(p.numel() for p in heads) == 2048
    assert all(torch.count_nonzero(p) == 0 for p in heads)


@pytest.mark.parametrize('training', [False, True])
def test_identity_output_gradients_with_active_state_and_dropout(training):
    new, old = make(trim=False), make(old=True, trim=False)
    active(new, bias=False)
    old.load_state_dict({k:v for k,v in new.state_dict().items() if not k.endswith('bias_head')})
    values, y = data()
    outputs = []
    for model in (old, new):
        model.train(training)
        seed_everything(823)
        out = model(*values)
        F.cross_entropy(out, y).backward()
        outputs.append(out)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    params = dict(new.named_parameters())
    for name, p in old.named_parameters():
        # Added zero-gradient path changes shared-stem reduction ordering.
        # Float64 diagnosis reduces the discrepancy to machine precision.
        # Reuse the pre-existing State-EV gradient contract, not fitted limits.
        torch.testing.assert_close(p.grad, params[name].grad, rtol=1e-4, atol=1e-5)
    for a in M.attentions(new):
        assert a.bias_head.grad.isfinite().all() and a.bias_head.grad.count_nonzero() > 0


def test_active_chunk_checkpoint_gradients_match_unchunked_oracle():
    fast, oracle = make(trim=False, chunk=4), make(trim=False, chunk=128, checkpointed=False)
    active(fast)
    oracle.load_state_dict(fast.state_dict())
    values, y = data(11)
    results = []
    for model in (fast, oracle):
        seed_everything(912)
        out = model(*values)
        F.cross_entropy(out, y).backward()
        results.append(out)
        assert model._state_pair_cache.normalized is None
        assert all(a._relation_stem is None for a in M.attentions(model))
    torch.testing.assert_close(*results, rtol=2e-5, atol=2e-6)
    for (name,p), (other,q) in zip(fast.named_parameters(), oracle.named_parameters()):
        assert name == other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-6)


def test_bias_only_gradient_reaches_state_projections_direction_and_mask():
    model = make(trim=False)
    a = M.attentions(model)[0]
    with torch.no_grad(): a.bias_head.normal_(std=.03)
    query = torch.randn(7, 2, 128, requires_grad=True)
    stem = torch.randn(2, 64, 7, 7, requires_grad=True)
    valid = torch.ones(2, 7, dtype=torch.bool); valid[1,-2:] = False
    delta = a.correction(query, stem, valid)
    assert delta.abs().max() <= 1
    assert not torch.allclose(delta, delta.transpose(-1,-2))
    assert delta[1,:,-2:].count_nonzero() == delta[1,:,:,-2:].count_nonzero() == 0
    delta.square().sum().backward()
    for layer in (a.state.receiver, a.state.sender, a.state.edge):
        assert layer.weight.grad.isfinite().all() and layer.weight.grad.count_nonzero() > 0


def test_one_mha_corrected_weights_for_all_paths_and_no_double_dropout():
    model = make(trim=False)
    a = M.attentions(model)[0].train()
    active(model)
    q = torch.randn(7, 2, 128)
    valid = torch.ones(2,7,dtype=torch.bool); valid[1,-2:] = False
    stem = torch.randn(2,64,7,7).masked_fill(~(valid[:,None,:,None]&valid[:,None,None,:]),0)
    a.bind(stem,valid)
    mask = torch.randn(16,7,7)
    delta = a.correction(q,stem,valid).reshape_as(mask)
    expected_attention = copy.deepcopy(a.reference)
    calls = []
    handle = a.reference.register_forward_hook(lambda *args: calls.append(1))
    seed_everything(982)
    actual, weights = a(q,q,q,key_padding_mask=~valid,attn_mask=mask,average_attn_weights=False)
    seed_everything(982)
    ordinary, expected_weights = expected_attention(q,q,q,key_padding_mask=~valid,
        attn_mask=mask+delta,need_weights=True,average_attn_weights=False)
    handle.remove()
    assert calls == [1]
    torch.testing.assert_close(weights,expected_weights,rtol=0,atol=0)
    ev = torch.einsum('hod,bhid->bhio',a.edge_projection,
                      torch.einsum('bhij,bdij->bhid',weights,stem))
    state = a.state(weights,stem,q,valid)
    extra = F.linear((ev+state).permute(2,0,1,3).reshape_as(q),a.reference.out_proj.weight)
    torch.testing.assert_close(actual,ordinary+extra*valid.T[:,:,None],rtol=0,atol=0)


def test_permutation_padding_empty_and_exception_cleanup(monkeypatch):
    model = make(trim=False).eval(); active(model)
    values, _ = data()
    perm = torch.tensor([8,3,0,1,6,2,7,4,5])
    with torch.no_grad():
        out = model(*values)
        torch.testing.assert_close(model(*(v[:,:,perm] for v in values)),out,rtol=2e-5,atol=2e-6)
    bad = [v.clone() for v in values]; bad[-1][0] = False
    with pytest.raises(ValueError,match='valid particle'): model(*bad)
    def fail(*args,**kwargs): raise RuntimeError('injected')
    monkeypatch.setattr(model.mod.blocks[-1],'forward',fail)
    with pytest.raises(RuntimeError,match='injected'): model(*values)
    assert model._state_pair_cache.normalized is None
    assert all(b.attn._bias is None and b.attn.attention._relation_stem is None for b in model.mod.blocks)


def test_native_trimming_remains_aligned_with_zero_identity_and_active_oracle():
    new, old = make(trim=True), make(old=True,trim=True)
    active(new,bias=False)
    old.load_state_dict({k:v for k,v in new.state_dict().items() if not k.endswith('bias_head')})
    oracle=make(trim=True,chunk=128,checkpointed=False)
    values,y=data(17)
    values[-1][:,:,10:]=False
    for module in (new,old):
        module.mod.trimmer._counter=10
    outputs=[]
    lengths=[]
    def record(_module,inputs): lengths.append(inputs[0].shape[0])
    for model in (old,new):
        handle=model.mod.blocks[0].register_forward_pre_hook(record)
        seed_everything(43);outputs.append(model(*values));handle.remove()
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    assert lengths[0]==lengths[1] and lengths[0]<=10
    active(new)
    oracle.load_state_dict(new.state_dict())
    oracle.mod.trimmer._counter=new.mod.trimmer._counter
    outputs=[]
    for model in (new,oracle):
        seed_everything(194);out=model(*values);F.cross_entropy(out,y).backward();outputs.append(out)
    torch.testing.assert_close(*outputs,rtol=2e-5,atol=2e-6)


def test_valid_self_pair_and_padding_values_are_isolated_in_eval():
    model=make(trim=False).eval();active(model)
    values,_=data(9)
    padded=[v.clone() for v in values]
    for tensor in padded[:-1]: tensor[1,:,-2:]=37
    with torch.no_grad():
        torch.testing.assert_close(model(*values),model(*padded),rtol=2e-5,atol=2e-6)
        single=[v[:1,:,:1] for v in values]
        assert model(*single).isfinite().all()
    a=M.attentions(model)[0]
    valid=torch.ones(1,1,dtype=torch.bool)
    delta=a.correction(torch.randn(1,1,128),torch.randn(1,64,1,1),valid)
    assert delta.count_nonzero()>0  # Self pairs are not mistakenly removed.
