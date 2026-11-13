import copy
import pytest
import torch
from torch.nn import functional as F
from part_ev_pair_memory import model as M
from part_ev_state_compare.model import build_model as reference
from part_ev_gpu_scaling.benchmark import fixed_batch,tensors
from part_ev_fullscale.checkpoint import seed_everything
from tests.test_part_fullscale_model import bounded_cpu_threads


def make(old=False,**kw):
    seed_everything(101)
    return reference('state_layerwise',**kw) if old else M.build_model(**kw)


def data(n=9):
    values,y=tensors(fixed_batch(3,n),torch.device('cpu'))
    values[-1][1,:,-2:]=False
    return values,y


def active(model,readers=True):
    with torch.no_grad():
        for s in M.states(model):
            s.output.normal_(std=.02)
            if readers and s.memory_reader is not None:s.memory_reader.normal_(std=.02)
            if readers and s.memory_gate is not None:s.memory_gate[:3].normal_(std=.1)


def test_count_initialization_rng_and_ordering():
    old=make(old=True,trim=False);rng=torch.get_rng_state()
    m=make(trim=False)
    assert sum(p.numel() for p in m.parameters())==2364294
    assert sum(p.numel() for n,p in m.named_parameters() if M.memory_key(n))==36832
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k
    assert len(M.readers(m))==7 and all(not p.count_nonzero() for p in M.readers(m))
    s=M.states(m)
    assert s[0].memory_reader is None and s[0].memory_gate is None and s[7].memory_writer is None
    assert sum(x.memory_writer is not None for x in s)==7
    assert sum(x.memory_gate is not None for x in s)==6
    for x in s[1:7]:
        torch.testing.assert_close(x.memory_gate[3].sigmoid(),torch.tensor([.2]*16+[.5]*16))


@pytest.mark.parametrize('training',[False,True])
def test_zero_reader_identity_with_active_old_state_and_gradients(training):
    m,old=make(trim=False),make(old=True,trim=False);active(m,False)
    old.load_state_dict({k:v for k,v in m.state_dict().items() if not M.memory_key(k)})
    values,y=data();outs=[];rngs=[]
    for target in (old,m):
        target.train(training);seed_everything(810)
        out=target(*values);F.cross_entropy(out,y).backward();outs.append(out)
        rngs.append(torch.get_rng_state())
    torch.testing.assert_close(*outs,rtol=0,atol=0)
    assert torch.equal(*rngs)
    params=dict(m.named_parameters())
    for n,p in old.named_parameters():torch.testing.assert_close(p.grad,params[n].grad,rtol=1e-4,atol=1e-5)
    for n,b in old.named_buffers():torch.testing.assert_close(b,dict(m.named_buffers())[n],rtol=0,atol=0)
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in M.writers_and_gates(m))
    assert all(p.grad.isfinite().all() and p.grad.count_nonzero() for p in M.readers(m))


def test_first_readers_then_all_writers_and_gates_learn():
    m=make(trim=False);values,y=data();opt=torch.optim.SGD(m.parameters(),lr=.01)
    F.cross_entropy(m(*values),y).backward()
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in M.writers_and_gates(m))
    assert all(p.grad.count_nonzero() for p in M.readers(m))
    opt.step();opt.zero_grad(set_to_none=True)
    F.cross_entropy(m(*values),y).backward()
    assert all(p.grad.isfinite().all() and p.grad.count_nonzero() for p in M.writers_and_gates(m))


def test_independent_dense_multilayer_recurrence_and_history_gradients():
    from part_ev_pair_memory.oracle import verify
    assert verify(torch.device('cpu'))['passed']


def test_chunk_checkpoint_all_gradients_after_cache_cleanup():
    m,old=make(trim=False,chunk=4),make(trim=False,chunk=128,checkpointed=False)
    active(m);old.load_state_dict(m.state_dict());values,y=data(11);outs=[]
    for target in (old,m):
        seed_everything(92);out=target(*values)
        assert target._state_pair_cache.normalized is None
        F.cross_entropy(out,y).backward();outs.append(out)
    torch.testing.assert_close(*outs,rtol=1e-5,atol=1e-6)
    for (n,p),(k,q) in zip(m.named_parameters(),old.named_parameters()):
        assert n==k and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5)


def test_gate_limits_bounds_padding_direction_and_self_pairs():
    m=make(trim=False);s=M.states(m)[2]
    o=torch.randn(2,5,5,32);memory=torch.rand_like(o)*2-1
    valid=torch.ones(2,5,dtype=torch.bool);valid[1,-2:]=False
    pair=valid[:,:,None]&valid[:,None,:]
    with torch.no_grad():s.memory_gate[:3].zero_();s.memory_gate[3].fill_(-1000)
    kept,c,u,ok=s.write(o,memory,pair)
    assert ok and u.max()==0
    torch.testing.assert_close(kept,memory.masked_fill(~pair[...,None],0),rtol=0,atol=0)
    with torch.no_grad():s.memory_gate[3].fill_(1000)
    replaced,c,u,ok=s.write(o,memory,pair)
    torch.testing.assert_close(replaced,c,rtol=0,atol=0)
    assert c.abs().max()<=1 and c[:,0,0].count_nonzero()
    assert not torch.allclose(c,c.transpose(1,2))
    assert not c.masked_select(~pair[...,None]).count_nonzero()
    first=M.states(m)[0];last=M.states(m)[7]
    updated,c,u,ok=first.write(o,None,pair)
    assert updated is c and u is None
    assert last.write(o,memory,pair)[:3]==(None,None,None)


def test_read_before_write_not_candidate_or_attention_weighted_write():
    m=make(trim=False);active(m);s=M.states(m)[2]
    weights=torch.randn(2,8,5,5).softmax(-1);pairs=torch.randn(2,5,5,64)
    p,q=torch.randn(2,5,32),torch.randn(2,5,32);valid=torch.ones(2,5,5,dtype=torch.bool)
    memory=torch.rand(2,5,5,32)*2-1
    a=s.normalized_chunk(weights,pairs,p,q,valid,memory)
    b=s.normalized_chunk(weights,pairs*3,p*4,q*2,valid,memory)
    torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
    assert not torch.allclose(a[2],b[2])
    c=s.normalized_chunk(weights*0,pairs,p,q,valid,memory)
    torch.testing.assert_close(a[2],c[2],rtol=0,atol=0)


def test_overflow_before_saturation_is_detected():
    m=make(trim=False);s=M.states(m)[1]
    o=torch.randn(2,3,3,32);pair=torch.ones(2,3,3,dtype=torch.bool);memory=torch.ones_like(o)
    with torch.no_grad():s.memory_writer.bias.fill_(float('inf'))
    updated,c,gate,finite=s.write(o,memory,pair)
    assert not finite and c.isfinite().all()  # tanh would otherwise hide failure.
    with torch.no_grad():s.memory_writer.bias.zero_();s.memory_gate[3].fill_(float('inf'))
    assert not s.write(o,memory,pair)[3]


def test_projection_autocast_but_memory_and_gate_fp32():
    # CPU autocast exercises cast boundaries; CUDA FP16 is separately benchmarked.
    m=make(trim=False);s=M.states(m)[1]
    o=torch.randn(2,3,3,32);pair=torch.ones(2,3,3,dtype=torch.bool);memory=torch.rand_like(o)
    seen=[];hook=s.memory_writer.register_forward_hook(lambda _,args,out:seen.append(out.dtype))
    with torch.autocast('cpu',dtype=torch.bfloat16):updated,candidate,gate,ok=s.write(o,memory,pair)
    hook.remove()
    assert seen==[torch.bfloat16] and ok
    assert updated.dtype==candidate.dtype==gate.dtype==torch.float32


def test_health_error_is_not_hidden_by_zero_reader():
    m=make(trim=False);values,y=data()
    with torch.no_grad():M.states(m)[2].memory_writer.bias.fill_(float('inf'))
    with pytest.raises(FloatingPointError,match='preactivation'):m(*values)
    assert m._state_pair_cache.normalized is None
    assert all(b.attn._bias is None for b in m.mod.blocks)


def test_two_outstanding_training_forwards_and_eval_interleaving():
    m=make(trim=False,chunk=4);active(m);oracle=make(trim=False,chunk=128,checkpointed=False)
    oracle.load_state_dict(m.state_dict());values,y=data(11);grads=[]
    for target in (m,oracle):
        target.train();seed_everything(516);a=target(*values)
        target.eval()
        with torch.no_grad():target(*(v[:1,:,:1] for v in values))
        target.train();b=target(*values)
        (F.cross_entropy(a,y)+F.cross_entropy(b,y)).backward()
        grads.append({n:p.grad.clone() for n,p in target.named_parameters()})
    for name,g in grads[0].items():torch.testing.assert_close(g,grads[1][name],rtol=1e-4,atol=1e-5)


def test_permutation_padding_single_particle_empty_and_lifetime(monkeypatch):
    m=make(trim=False).eval();active(m);values,_=data();perm=torch.randperm(9)
    keys=set(m.state_dict());padded=[v.clone() for v in values]
    for v in padded[:-1]:v[1,:,-2:]=37
    with torch.no_grad():
        out=m(*values)
        torch.testing.assert_close(m(*(v[:,:,perm] for v in values)),out,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(m(*padded),out,rtol=2e-5,atol=2e-6)
        assert m(*(v[:1,:,:1] for v in values)).isfinite().all()
        torch.testing.assert_close(m(*values),out,rtol=0,atol=0)
    bad=[v.clone() for v in values];bad[-1][0]=False
    with pytest.raises(ValueError,match='valid particle'):m(*bad)
    def fail(*args,**kw):raise RuntimeError('injected')
    with monkeypatch.context() as patch:
        patch.setattr(m.mod.blocks[5].attn.attention.state,'forward',fail)
        with pytest.raises(RuntimeError,match='injected'):m(*values)
    assert m._state_pair_cache.normalized is None
    assert all(b.attn._bias is None and b.attn.attention._relation_stem is None for b in m.mod.blocks)
    assert set(m.state_dict())==keys
    assert not any('current_memory' in vars(module) for module in m.modules())
    with torch.no_grad():torch.testing.assert_close(m(*values),out,rtol=0,atol=0)


def test_one_native_trim_and_one_attention_each_with_actual_permutation():
    m,old=make(trim=True),make(old=True,trim=True);active(m,False)
    old.load_state_dict({k:v for k,v in m.state_dict().items() if not M.memory_key(k)})
    values,y=data(17);values[-1][:,:,10:]=False;outs=[]
    for target in (old,m):
        target.mod.trimmer._counter=10;calls=[];lengths=[]
        handles=[target.mod.trimmer.register_forward_hook(lambda *a:calls.append('trim'))]
        handles += [b.attn.attention.reference.register_forward_hook(lambda _,args,result:lengths.append(args[0].shape[0])) for b in target.mod.blocks]
        seed_everything(67);outs.append(target(*values))
        for h in handles:h.remove()
        assert calls==['trim'] and len(lengths)==8 and max(lengths)<=10
    torch.testing.assert_close(*outs,rtol=0,atol=0)
