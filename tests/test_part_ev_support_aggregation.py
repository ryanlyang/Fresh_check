import copy
import pytest
import torch
from torch.nn import functional as F
from part_ev_support import aggregation as A,oracle as O
from part_ev_support.model import Support
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_explicit_oracle():assert O.verify(torch.device('cpu'))['passed']


def test_constant_gates_layout_replication_and_floor():
    logits=torch.zeros(1,2,3,4,dtype=torch.float64)
    values=torch.arange(16,dtype=torch.float64).expand(1,2,3,16)
    mask=torch.tensor([[[1,0,1],[0,0,0]]],dtype=torch.bool)
    c,s,m,h=A.statistics(logits,values,mask);f,ok=A.finish(c,s,m,h)
    assert ok and c[0,0].tolist()==[1.]*4 and m.tolist()==[[2,0]]
    channels=f.reshape(1,2,4,18)
    torch.testing.assert_close(channels[0,0,:,:16],torch.arange(16,dtype=torch.float64).expand(4,16))
    torch.testing.assert_close(channels[0,0,:,16],torch.full((4,),2.,dtype=torch.float64).log())
    assert channels[0,0,:,17].tolist()==[.5]*4 and not f[0,1].count_nonzero()
    c2,s2,m2,h2=A.statistics(logits.repeat_interleave(3,2),values.repeat_interleave(3,2),mask.repeat_interleave(3,2))
    f2,_=A.finish(c2,s2,m2,h2);f2=f2.reshape_as(channels)
    torch.testing.assert_close(c2,c*3);torch.testing.assert_close(s2,s*3)
    torch.testing.assert_close(f2[...,:16],channels[...,:16]);torch.testing.assert_close(f2[...,17],channels[...,17])
    torch.testing.assert_close(f2[...,16],c2.log1p())


@pytest.mark.parametrize('logit',[-1000.,-18.,-14.,0.,1000.])
def test_extreme_finite_logits_underflow_and_zero_neighbors(logit):
    logits=torch.full((2,3,4,4),logit,requires_grad=True)
    values=torch.randn(2,3,4,16,requires_grad=True)
    valid=torch.ones(2,3,4,dtype=torch.bool);valid[1]=False
    stats=A.statistics(logits,values,valid);f,h=A.finish(*stats)
    assert h and f.isfinite().all() and not f[1].count_nonzero()
    f.square().sum().backward()
    assert logits.grad.isfinite().all() and values.grad.isfinite().all()
    assert not logits.grad[1].count_nonzero() and not values.grad[1].count_nonzero()
    if logit==-1000:assert not f.count_nonzero()


def test_raw_merge_before_normalization_and_not_mean_of_means():
    logits=torch.tensor([[[[-4.]*4,[3.]*4,[1.]*4]]],dtype=torch.float64)
    values=torch.tensor([[[[-2.]*16,[3.]*16,[7.]*16]]],dtype=torch.float64)
    mask=torch.ones(1,1,3,dtype=torch.bool)
    full=A.statistics(logits,values,mask)
    left=A.statistics(logits[:,:,:1],values[:,:,:1],mask[:,:,:1])
    right=A.statistics(logits[:,:,1:],values[:,:,1:],mask[:,:,1:])
    merged=A.merge(left,right)
    for a,b in zip(full,merged):torch.testing.assert_close(a,b)
    f,_=A.finish(*full);g,_=A.finish(*merged);torch.testing.assert_close(f,g)
    wrong=(A.finish(*left)[0]+A.finish(*right)[0])/2
    assert not torch.allclose(f,wrong)


def test_gradcheck_near_floor_and_at_kink():
    logits=torch.randn(1,2,3,4,dtype=torch.float64,requires_grad=True)
    values=torch.randn(1,2,3,16,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[[1,0,1],[0,0,0]]],dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda l,v:A.finish(*A.statistics(l,v,mask))[0],(logits,values),fast_mode=True)
    for count in (.1e-6,1e-6,2e-6):
        c=torch.full((1,1,4),count,dtype=torch.float64,requires_grad=True)
        s=(c[...,None]*torch.ones(1,1,4,16,dtype=torch.float64))
        f,h=A.finish(c,s,torch.ones(1,1,dtype=torch.long),torch.tensor(True))
        assert h;f.sum().backward();assert c.grad.isfinite().all()
        expected=min(1.,count/A.EPSILON)
        torch.testing.assert_close(f.reshape(1,1,4,18)[...,:16],torch.full((1,1,4,16),expected,dtype=torch.float64))


def test_projected_features_full_parameter_input_gradient_oracle():
    torch.manual_seed(811);m=Support().double()
    with torch.no_grad():m.output.weight.normal_(std=.02);m.hidden.bias.fill_(.1)
    other=copy.deepcopy(m);features=torch.randn(2,5,5,32,dtype=torch.float64)
    valid=torch.tensor([[1,0,1,1,1],[0,0,1,0,0]],dtype=torch.bool)
    mask=valid[:,:,None]&valid[:,None,:]&~torch.eye(5,dtype=torch.bool)
    outputs=[];gradients=[]
    for target,oracle in ((m,False),(other,True)):
        r=features.clone().requires_grad_()
        if oracle:
            u=F.layer_norm(r.masked_fill(~mask[...,None],0),(32,),eps=1e-5)
            logits=F.linear(u,target.gate.weight,target.gate.bias);values=F.linear(u,target.value.weight)
            c,s,degree,_=O.explicit(logits,values,mask)
            packed=torch.cat((s/c.clamp_min(1e-6)[...,None],c.log1p()[...,None],
                              (c/degree.clamp_min(1)[...,None])[...,None]),-1).flatten(-2)
            packed=packed.masked_fill(~(degree>0)[...,None],0)
            out=F.linear(F.gelu(F.linear(packed,target.hidden.weight,target.hidden.bias)),target.output.weight)
            out=out.masked_fill(~(degree>0)[...,None],0)
        else:out,h=target(r,mask);assert h
        out.square().sum().backward();outputs.append(out)
        gradients.append([r.grad,*[p.grad for p in target.parameters()]])
        assert not r.grad[~mask].count_nonzero() and not out[1].count_nonzero()
    torch.testing.assert_close(*outputs,rtol=1e-10,atol=1e-11)
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)


def test_excluded_nonfinite_features_and_fp32_autocast_reductions():
    m=Support();r=torch.randn(2,5,5,32)
    mask=torch.ones(2,5,5,dtype=torch.bool)&~torch.eye(5,dtype=torch.bool)
    mask[1,-2:]=False;mask[1,:,-2:]=False
    with torch.autocast('cpu',dtype=torch.bfloat16):
        logits,values,h=m.project(r,mask);stats=A.statistics(logits,values,mask);f,ok=A.finish(*stats)
        assert h and ok and all(x.dtype==torch.float32 for x in (stats[0],stats[1],f))
        changed=r.clone();changed[~mask]=float('nan')
        l,v,h=m.project(changed,mask);g,ok=A.finish(*A.statistics(l,v,mask))
        assert h and ok;torch.testing.assert_close(f,g,rtol=0,atol=0)
    with torch.no_grad():m.gate.bias.fill_(float('inf'))
    _,healthy=m(r,mask);assert not healthy  # sigmoid may be finite, preactivation isn't.


def test_directed_permutation_and_distinct_learned_counts():
    torch.manual_seed(11);m=Support().double();r=torch.randn(1,6,6,32,dtype=torch.float64)
    mask=~torch.eye(6,dtype=torch.bool)[None];permutation=torch.tensor([4,2,5,0,1,3])
    def f(x,v):
        l,c,_=m.project(x,v);return A.finish(*A.statistics(l,c,v))[0]
    expected=f(r,mask)
    torch.testing.assert_close(f(r[:,permutation][:,:,permutation],mask[:,permutation][:,:,permutation]),
                               expected[:,permutation],rtol=1e-10,atol=1e-11)
    assert not torch.allclose(f(r.transpose(1,2),mask),expected)
    counts=expected.reshape(1,6,4,18)[...,16].expm1()
    assert not torch.allclose(counts[...,0],counts[...,1])
    assert counts.std()>0 and not torch.allclose(counts.sum(-1),torch.full((1,6),5.,dtype=torch.float64))
