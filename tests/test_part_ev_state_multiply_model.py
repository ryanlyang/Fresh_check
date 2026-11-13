import copy
import pytest
import torch
from torch.nn import functional as F

from part_ev_state_multiply import model as M
from part_ev_state_compare.model import build_model as reference
from part_ev_message_study.model import MessageAttention
from part_ev_gpu_scaling.benchmark import fixed_batch,tensors
from part_ev_fullscale.checkpoint import seed_everything
from tests.test_part_fullscale_model import bounded_cpu_threads


def make(*,old=False,**kwargs):
    seed_everything(101)
    return reference('state_layerwise',**kwargs) if old else M.build_model(**kwargs)


def data(n=9):
    values,y=tensors(fixed_batch(3,n),torch.device('cpu'))
    values[-1][1,:,-2:]=False
    return values,y


def active(model,gate=True):
    with torch.no_grad():
        for s in M.states(model):
            s.output.normal_(std=.02)
            if gate:
                s.receiver_gate.normal_(std=.03);s.sender_gate.normal_(std=.03)


def test_parameters_initialization_rng_and_native_attention_are_preserved():
    old=make(old=True,trim=False);rng=torch.get_rng_state()
    model=make(trim=False)
    assert sum(p.numel() for p in model.parameters())==2343846
    assert sum(p.numel() for p in M.gate_parameters(model))==16384
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.state_dict().items():assert torch.equal(v,model.state_dict()[k]),k
    assert all(not p.count_nonzero() for p in M.gate_parameters(model))
    assert all(type(b.attn.attention) is MessageAttention for b in model.mod.blocks)
    assert not any('bias_head' in n for n,_ in model.named_parameters())


@pytest.mark.parametrize('training',[False,True])
def test_zero_modulation_identity_active_state_dropout_gradients(training):
    model,old=make(trim=False),make(old=True,trim=False)
    active(model,gate=False)
    old.load_state_dict({k:v for k,v in model.state_dict().items() if not k.endswith(M.GATE_SUFFIXES)})
    values,y=data();outputs=[]
    for target in (old,model):
        target.train(training);seed_everything(810)
        out=target(*values);F.cross_entropy(out,y).backward();outputs.append(out)
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    params=dict(model.named_parameters())
    for n,p in old.named_parameters():torch.testing.assert_close(p.grad,params[n].grad,rtol=1e-4,atol=1e-5)
    assert all(p.grad.isfinite().all() and p.grad.count_nonzero() for p in M.gate_parameters(model))


def test_delayed_activation_is_not_a_dead_double_zero_adapter():
    model=make(trim=False)
    values,y=data()
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    F.cross_entropy(model(*values),y).backward()
    assert all(s.output.grad.count_nonzero() for s in M.states(model))
    assert all(p.grad is not None and p.grad.count_nonzero()==0 for p in M.gate_parameters(model))
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(*values),y).backward()
    assert all(p.grad.isfinite().all() and p.grad.count_nonzero() for p in M.gate_parameters(model))
    optimizer.step()
    assert all(p.detach().count_nonzero() for p in M.gate_parameters(model))


def test_active_chunk_checkpoint_backward_matches_unchunked():
    fast,oracle=make(trim=False,chunk=4),make(trim=False,chunk=128,checkpointed=False)
    active(fast);oracle.load_state_dict(fast.state_dict())
    values,y=data(11);outputs=[]
    for target in (fast,oracle):
        seed_everything(282);out=target(*values);F.cross_entropy(out,y).backward();outputs.append(out)
        assert target._state_pair_cache.normalized is None
    torch.testing.assert_close(*outputs,rtol=1e-5,atol=1e-6)
    for (n,p),(other,q) in zip(fast.named_parameters(),oracle.named_parameters()):
        assert n==other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5)


def test_dense_formula_and_gradients_independent_of_optimized_chunk_code():
    model=make(trim=False);active(model)
    state=M.states(model)[0].double()
    oracle=copy.deepcopy(state)
    q=torch.randn(7,2,128,dtype=torch.float64,requires_grad=True)
    stem=torch.randn(2,64,7,7,dtype=torch.float64,requires_grad=True)
    weights=torch.randn(2,8,7,7,dtype=torch.float64).softmax(-1).requires_grad_()
    valid=torch.ones(2,7,dtype=torch.bool);valid[1,-2:]=False
    normalized=F.layer_norm(q.transpose(0,1),(128,),eps=1e-5)
    p=oracle.receiver(normalized);s=oracle.sender(normalized)
    pairs=F.layer_norm(stem.permute(0,2,3,1),(64,),eps=1e-5)
    g=oracle.edge(pairs)
    gamma=(F.linear(p,oracle.receiver_gate)[:,:,None]+F.linear(s,oracle.sender_gate)[:,None]).tanh()
    h=g+p[:,:,None]+s[:,None]+p[:,:,None]*s[:,None]
    features=F.gelu(F.layer_norm(h+g*gamma,(32,),eps=1e-5))
    features=features.masked_fill(~(valid[:,:,None]&valid[:,None,:])[...,None],0)
    expected=torch.einsum('hor,bhir->bhio',oracle.output,torch.einsum('bhij,bijr->bhir',weights,features))
    actual=state(weights,stem,q,valid)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ag=torch.autograd.grad(actual.square().sum(),[q,stem,weights,*state.parameters()])
    eg=torch.autograd.grad(expected.square().sum(),[q,stem,weights,*oracle.parameters()])
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)


def test_per_particle_projections_direction_mask_and_nonzero_gradients(monkeypatch):
    model=make(trim=False,chunk=3);active(model);state=M.states(model)[0]
    q=torch.randn(7,2,128,requires_grad=True);stem=torch.randn(2,64,7,7,requires_grad=True)
    weights=torch.randn(2,8,7,7).softmax(-1)
    valid=torch.ones(2,7,dtype=torch.bool);valid[1,-2:]=False
    calls=[];linear=M.F.linear
    def observe(x,w,bias=None):
        if w is state.receiver_gate or w is state.sender_gate:calls.append(tuple(x.shape))
        return linear(x,w,bias)
    monkeypatch.setattr(M.F,'linear',observe)
    out=state(weights,stem,q,valid)
    assert calls==[(2,7,32),(2,7,32)]  # Independent of three query chunks.
    out.square().sum().backward()
    for p in (state.receiver.weight,state.sender.weight,state.edge.weight,state.receiver_gate,state.sender_gate):
        assert p.grad.isfinite().all() and p.grad.count_nonzero()
    assert q.grad.isfinite().all() and stem.grad.isfinite().all()
    assert out[1,:,-2:].count_nonzero()==0
    p,s=state.receiver(M.normalize(q.transpose(0,1))),state.sender(M.normalize(q.transpose(0,1)))
    a,c=state.gate_endpoints(p,s)
    _,gate,_,_=state.preactivation(M.normalize(stem.permute(0,2,3,1)),p,s,a,c)
    assert gate.abs().max()<=1
    assert not torch.allclose(gate,gate.transpose(1,2))
    assert gate[:,0,0].count_nonzero()>0  # Self-pair modulation is retained.


def test_native_attention_and_ordinary_ev_unchanged_at_identical_input():
    from part_ev_state_multiply.benchmark import attention_invariance
    assert attention_invariance(torch.device('cpu'))['passed']


def test_permutation_padding_single_particle_empty_and_error_cleanup(monkeypatch):
    model=make(trim=False).eval();active(model)
    values,_=data();perm=torch.tensor([8,3,0,1,6,2,7,4,5])
    padded=[v.clone() for v in values]
    for v in padded[:-1]:v[1,:,-2:]=37
    with torch.no_grad():
        out=model(*values)
        torch.testing.assert_close(model(*(v[:,:,perm] for v in values)),out,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(model(*padded),out,rtol=2e-5,atol=2e-6)
        assert model(*(v[:1,:,:1] for v in values)).isfinite().all()
    bad=[v.clone() for v in values];bad[-1][0]=False
    with pytest.raises(ValueError,match='valid particle'):model(*bad)
    def fail(*args,**kwargs):raise RuntimeError('injected')
    monkeypatch.setattr(model.mod.blocks[-1],'forward',fail)
    with pytest.raises(RuntimeError,match='injected'):model(*values)
    assert model._state_pair_cache.normalized is None
    assert all(b.attn._bias is None and b.attn.attention._relation_stem is None for b in model.mod.blocks)


def test_native_trimming_alignment_in_zero_gate_and_nonzero_chunk_paths():
    model,old=make(trim=True),make(old=True,trim=True);active(model,gate=False)
    old.load_state_dict({k:v for k,v in model.state_dict().items() if not k.endswith(M.GATE_SUFFIXES)})
    values,y=data(17);values[-1][:,:,10:]=False
    for m in (model,old):m.mod.trimmer._counter=10
    outputs=[];lengths=[]
    for m in (old,model):
        hook=m.mod.blocks[0].register_forward_pre_hook(lambda _,args:lengths.append(args[0].shape[0]))
        seed_everything(67);outputs.append(m(*values));hook.remove()
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    assert lengths[0]==lengths[1] and lengths[0]<=10
    active(model);oracle=make(trim=True,chunk=128,checkpointed=False)
    oracle.load_state_dict(model.state_dict());oracle.mod.trimmer._counter=model.mod.trimmer._counter
    outputs=[]
    for m in (model,oracle):
        seed_everything(173);out=m(*values);F.cross_entropy(out,y).backward();outputs.append(out)
    torch.testing.assert_close(*outputs,rtol=1e-5,atol=1e-6)
