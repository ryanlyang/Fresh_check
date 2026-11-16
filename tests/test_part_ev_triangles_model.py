import pytest
import torch
from torch.nn import functional as F
from part_ev_triangles import model as M
from part_ev_triangles.benchmark import active, boundaries, particle_path
from part_ev_gpu_scaling.benchmark import fixed_batch,tensors
from part_ev_fullscale.checkpoint import seed_everything
from tests.test_part_fullscale_model import bounded_cpu_threads


def make(old=False,**kw):
    seed_everything(101)
    return M.reference_model('state_layerwise',**kw) if old else M.build_model(**kw)


def data(n=9):
    values,y=tensors(fixed_batch(3,n),torch.device('cpu'))
    values[-1][1,:,-2:]=False
    return values,y


def test_count_rng_and_no_duplicate_registration():
    old=make(trim=False);rng=torch.get_rng_state();m=make(trim=False)
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in m.parameters())==2334662
    assert sum(p.numel() for n,p in m.named_parameters() if M.triangle_key(n))==7200
    for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k
    names=list(m.named_parameters(remove_duplicate=False))
    assert len(names)==len({id(p) for _,p in names})
    assert all(not p.count_nonzero() for p in M.output_parameters(m))


@pytest.mark.parametrize('training',[False,True])
def test_identity_active_base_gradients_buffers_rng(training):
    m,old=make(trim=False),make(old=True,trim=False);active(m,triangle=False)
    old.load_state_dict({k:v for k,v in m.state_dict().items() if not M.triangle_key(k)})
    values,y=data();outs=[];rngs=[]
    for target in (old,m):
        target.train(training);seed_everything(87)
        out=target(*values);F.cross_entropy(out,y).backward();outs.append(out);rngs.append(torch.get_rng_state())
    torch.testing.assert_close(*outs,rtol=0,atol=0);assert torch.equal(*rngs)
    new=dict(m.named_parameters())
    for n,p in old.named_parameters():torch.testing.assert_close(p.grad,new[n].grad,rtol=1e-4,atol=1e-5)
    for n,b in old.named_buffers():torch.testing.assert_close(b,dict(m.named_buffers())[n],rtol=0,atol=0)
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in M.upstream_parameters(m))
    assert all(p.grad.isfinite().all() and p.grad.count_nonzero() for p in M.output_parameters(m))


def test_output_then_upstream_learning_and_optimizer_membership():
    m=make(trim=False);values,y=data();opt=torch.optim.SGD(m.parameters(),lr=.01)
    F.cross_entropy(m(*values),y).backward()
    assert all(not p.grad.count_nonzero() for p in M.upstream_parameters(m))
    opt.step();opt.zero_grad(set_to_none=True)
    F.cross_entropy(m(*values),y).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero() for p in (p for n,p in m.named_parameters() if M.triangle_key(n)))


@pytest.mark.parametrize('chunk',[1,4,64])
def test_active_chunk_checkpoint_all_gradients(chunk):
    m=make(trim=False,chunk=chunk);oracle=make(trim=False,chunk=128,checkpointed=False,backend='dense_oracle')
    active(m);oracle.load_state_dict(m.state_dict());values,y=data(11);outs=[]
    for target in (oracle,m):
        seed_everything(91);out=target(*values)
        assert target._state_pair_cache.normalized is None
        F.cross_entropy(out,y).backward();outs.append(out)
    torch.testing.assert_close(*outs,rtol=1e-5,atol=1e-6)
    for (n,p),(k,q) in zip(m.named_parameters(),oracle.named_parameters()):
        assert n==k and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5)


@pytest.mark.parametrize('count',[1,2])
def test_no_third_particle_graph_with_active_triangle_outputs(count):
    m=make(trim=False).eval();active(m);values,y=data()
    values[-1][:]=False;values[-1][:,:,2:2+count]=True
    old=make(old=True,trim=False).eval()
    old.load_state_dict({k:v for k,v in m.state_dict().items() if not M.triangle_key(k)})
    torch.testing.assert_close(m(*values),old(*values),rtol=0,atol=0)
    F.cross_entropy(m(*values),y).backward()
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in (p for n,p in m.named_parameters() if M.triangle_key(n)))


def test_permutation_overflow_and_native_call_counts():
    assert boundaries(torch.device('cpu'))['passed']
    assert particle_path(torch.device('cpu'))['passed']


def test_two_outstanding_forwards_and_exception_cleanup(monkeypatch):
    m=make(trim=False,chunk=4);active(m);oracle=make(trim=False,chunk=128,checkpointed=False)
    oracle.load_state_dict(m.state_dict());values,y=data(11);grads=[]
    for target in (m,oracle):
        seed_everything(19);a=target(*values);b=target(*values)
        (F.cross_entropy(a,y)+F.cross_entropy(b,y)).backward()
        grads.append({n:p.grad.clone() for n,p in target.named_parameters()})
    for n,g in grads[0].items():torch.testing.assert_close(g,grads[1][n],rtol=1e-4,atol=1e-5)
    def fail(*args,**kw):raise RuntimeError('injected')
    with monkeypatch.context() as patch:
        patch.setattr(M.branches(m)[0],'project',fail)
        with pytest.raises(RuntimeError,match='injected'):m(*values)
    assert m._state_pair_cache.normalized is None and all(b.attn._bias is None for b in m.mod.blocks)
    assert not any('last_summary' in vars(module) for module in m.modules())
    assert m(*values).isfinite().all()


def test_invalid_and_unsupported_inputs():
    m=make(trim=False);values,_=data();values[-1][0]=False
    with pytest.raises(ValueError,match='valid particle'):m(*values)
    values,_=data()
    with pytest.raises(ValueError,match='standard relations'):m.mod(values[1],v=values[2],mask=values[3],uu=torch.zeros(1))
    with pytest.raises(ValueError):M.build_model(chunk=0)


def test_batch_independence_active_model():
    m=make(trim=False).eval();active(m);values,_=data(11)
    with torch.no_grad():
        together=m(*values)
        separate=torch.cat([m(*(v[i:i+1] for v in values)) for i in range(3)])
    torch.testing.assert_close(together,separate,rtol=1e-5,atol=1e-6)
