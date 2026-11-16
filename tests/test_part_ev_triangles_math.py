import pytest
import torch
from part_ev_triangles import contraction as C,oracle as O
from part_ev_triangles.neighbors import weights_from_distances,pack,NeighborBuilder
from part_ev_triangles.model import Triangle
from part_ev_gpu_scaling.benchmark import fixed_batch,tensors
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_explicit_directed_oracle():assert O.verify(torch.device('cpu'))['passed']


@pytest.mark.parametrize('n',[1,2,3,8,9,10,17])
@pytest.mark.parametrize('ties',[True,False])
def test_sparse_dense_gradient_and_tiling(n,ties):
    values,graph=O.fixture(torch.device('cpu'),n,ties=ties)
    a=C.dense(*values,graph)
    b=C.sparse(*values,graph,receiver_tile=2,sender_tile=3,intermediary_tile=1,tie_tile=2)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
    for x,y in zip(torch.autograd.grad(a.square().sum(),values),torch.autograd.grad(b.square().sum(),values)):
        torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)
    if n<3:assert not b.count_nonzero()


def test_fractional_cutoff_no_index_tiebreak_or_backfill():
    n=12;valid=torch.ones(1,n,dtype=torch.bool)
    distances=torch.ones(1,n,n);distances[0,0,1:7]=.5
    nu,ok=weights_from_distances(distances,valid)
    torch.testing.assert_close(nu[0,0,1:7],torch.ones(6))
    torch.testing.assert_close(nu[0,0,7:],torch.full((5,),.4))
    assert nu[0,0].sum()==8
    # Removing sender 7 leaves 7.6 weighted support, NOT 8, nor 10 contributors.
    torch.testing.assert_close(nu[0,0].sum()-nu[0,0,7],torch.tensor(7.6))
    perm=torch.tensor([11,3,0,1,2,7,6,8,4,9,5,10])
    other,_=weights_from_distances(distances[:,perm][:,:,perm],valid[:,perm])
    torch.testing.assert_close(other,nu[:,perm][:,:,perm],rtol=0,atol=0)
    graph=pack(nu,valid,ok)
    assert len(graph.tie_rows)==n and graph.tie_indices.shape==(n,n-1)
    assert not graph.candidate_weights.count_nonzero()


def test_geometry_wrapped_phi_padding_and_health():
    values,_=tensors(fixed_batch(2,11),torch.device('cpu'));v=values[2];valid=values[-1].squeeze(1)
    valid[1,::3]=False
    graph=NeighborBuilder()(v,valid)
    polluted=v.clone();polluted.masked_fill_(~valid[:,None],float('nan'))
    other=NeighborBuilder()(polluted,valid)
    assert other.healthy and not other.weights.requires_grad
    torch.testing.assert_close(graph.weights,other.weights,rtol=0,atol=0)
    polluted[0,0,0]=float('nan');assert not NeighborBuilder()(polluted,valid).healthy
    # Azimuth discontinuity is wrapped, making +/-pi nearby.
    phi=torch.tensor([3.13,-3.13,0.]);v=torch.stack([phi.cos(),phi.sin(),torch.zeros(3),torch.ones(3)*2])[None]
    g=NeighborBuilder(budget=1)(v,torch.ones(1,3,dtype=torch.bool))
    assert g.weights[0,0,1]==1 and g.weights[0,1,0]==1


def test_gradcheck_and_bounded_index_select(monkeypatch):
    values,graph=O.fixture(torch.device('cpu'),3)
    assert torch.autograd.gradcheck(lambda *xs:C.sparse(*xs,graph,checkpointed=False),tuple(values),fast_mode=True)
    sizes=[];original=torch.Tensor.index_select
    def audit(self,dim,index):
        # Flat pair caches only, never a receiver-expanded sender cache.
        assert self.ndim==2
        sizes.append(index.numel()*self.shape[1]);return original(self,dim,index)
    monkeypatch.setattr(torch.Tensor,'index_select',audit)
    values,graph=O.fixture(torch.device('cpu'),17)
    C.sparse(*values,graph,receiver_tile=2,sender_tile=3,intermediary_tile=2,tie_tile=3,checkpointed=False)
    assert max(sizes)<=2*2*2*3*3


def test_projection_overflow_is_not_hidden_by_saturating_factors():
    t=Triangle();r=torch.randn(2,4,4,32);valid=torch.ones(2,4,4,dtype=torch.bool)
    with torch.no_grad():t.gate.bias.fill_(float('inf'))
    a,b,g,healthy=t.project(r,valid)
    assert g.isfinite().all() and not healthy


def test_independent_factor_norm_projections_output_and_input_gradients():
    import copy
    from torch.nn import functional as F
    torch.manual_seed(71)
    module=Triangle().double()
    with torch.no_grad():module.output.normal_(std=.02)
    reference=copy.deepcopy(module)
    values,graph=O.fixture(torch.device('cpu'),5)
    mask=graph.valid[:,:,None]&graph.valid[:,None,:]&~torch.eye(5,dtype=torch.bool)
    features=torch.randn(2,5,5,32,dtype=torch.float64)
    attention=torch.rand(2,8,5,5,dtype=torch.float64)
    outputs=[];gradients=[]
    for m,independent in ((module,False),(reference,True)):
        r=features.clone().requires_grad_();alpha=attention.clone().requires_grad_()
        if independent:
            u=F.layer_norm(r.masked_fill(~mask[...,None],0),(32,),eps=1e-5)
            a=F.linear(u,m.left.weight).tanh().masked_fill(~mask[...,None],0)
            b=F.linear(u,m.right.weight).tanh().masked_fill(~mask[...,None],0)
            gate=F.linear(u,m.gate.weight,m.gate.bias).sigmoid().masked_fill(~mask[...,None],0)
            pooled=O.explicit(a,b,gate,alpha,graph)
        else:
            a,b,gate,healthy=m.project(r,mask);assert healthy
            pooled=C.sparse(a,b,gate,alpha,graph,receiver_tile=2,sender_tile=2,intermediary_tile=2)
        out=torch.einsum('hor,bhir->bhio',m.output,pooled)
        out.square().sum().backward();outputs.append(out)
        gradients.append([r.grad,alpha.grad,*[p.grad for p in m.parameters()]])
        assert not r.grad[~mask].count_nonzero()
    torch.testing.assert_close(*outputs,rtol=1e-10,atol=1e-11)
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)


def test_fp32_factor_caches_and_contraction_inside_autocast():
    t=Triangle();r=torch.randn(2,11,11,32)
    _,graph=O.fixture(torch.device('cpu'),11,dtype=torch.float32)
    mask=graph.valid[:,:,None]&graph.valid[:,None,:]&~torch.eye(11,dtype=torch.bool)
    alpha=torch.rand(2,8,11,11)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        a,b,gate,healthy=t.project(r,mask)
        pooled=C.sparse(a,b,gate,alpha,graph)
    assert healthy and all(x.dtype==torch.float32 for x in (a,b,gate,pooled))
