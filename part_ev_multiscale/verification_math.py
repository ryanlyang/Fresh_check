"""Independent mathematical admission on the actual target device."""
from dataclasses import replace
import torch
from .attention import Attention
from .model import MultiscaleStage
from .geometry import build_geometry
from .pooling import pool
from . import oracles as O


def fixture(device,n=21,b=2):
    rng=torch.Generator().manual_seed(431+n)
    p=torch.randn(b,n,4,generator=rng);p[...,3]=p[...,:3].square().sum(-1).sqrt()+2
    mask=torch.ones(b,n,dtype=torch.bool)
    if n>4:mask[1,1::4]=False
    return p.to(device),mask.to(device)


def pooling_oracle(device):
    p,valid=fixture(device);g=build_geometry(p,valid)
    x=torch.randn(2,21,128,device=device,dtype=torch.float64);x[...,:4]=1
    x.requires_grad_();actual=pool(x,g,valid);expected=O.pooling(x,g,valid)
    params=(x,)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    ga=torch.autograd.grad(sum(z.square().sum() for z in actual),params,retain_graph=True)[0]
    ge=torch.autograd.grad(sum(z.square().sum() for z in expected),params)[0]
    torch.testing.assert_close(ga,ge,rtol=1e-9,atol=1e-10)
    if not bool(g.healthy):raise ValueError('Geometry unhealthy')
    return dict(passed=True,tie_gradients=True,independent_loop=True,precision='fp64',backend='scatter_reduce_amax')


def geometry_oracle(device):
    cases=[]
    for n in (1,2,4,16,17,128):
        p,valid=fixture(device,n)
        if n>=16:p[0,2]=p[0,0];p[0,3]=p[0,0]
        g=build_geometry(p,valid);reference=O.tree(p,valid)
        physical=O.physical_descriptors(p,valid,reference)
        for scale in ('fine','coarse'):
            torch.testing.assert_close(getattr(g,scale+'_descriptors').cpu(),physical[scale],rtol=1e-5,atol=1e-5)
        if not bool(g.healthy):raise ValueError('Unhealthy tree')
        for b,row in enumerate(reference):
            for scale in ('fine','coarse'):
                membership=getattr(g,scale)[b].cpu().bool();vectors=getattr(g,scale+'_vectors')[b].cpu()
                for j,(_,vector,members) in enumerate(row[scale]):
                    if set(torch.where(membership[:,j])[0].tolist())!=members:raise ValueError('Tree CPU oracle membership differs')
                    torch.testing.assert_close(vectors[j],vector,rtol=1e-5,atol=1e-5)
        perm=torch.arange(n-1,-1,-1,device=device);gp=build_geometry(p[:,perm],valid[:,perm])
        for name in ('fine','coarse'):
            if not torch.equal(getattr(gp,name),getattr(g,name)[:,perm]):raise ValueError('Noncanonical permuted cut')
        cases.append(dict(n=n,passed=True,min_margin=float(g.min_margin.min()),ties=int(g.exact_ties.sum())))
    # Equal angular coordinates, distinct momenta: genuine exact-distance ties.
    p=torch.zeros(2,21,4,device=device);p[...,0]=torch.arange(1,22,device=device);p[...,3]=p[...,0]+2
    valid=torch.ones(2,21,device=device,dtype=torch.bool);g=build_geometry(p,valid)
    if not bool((g.exact_ties>0).all()):raise ValueError('Tie fixture did not exercise ties')
    # All duplicate atoms must remain indivisible even with 128 particles.
    d=p[:,:1].expand(-1,128,-1);gd=build_geometry(d,torch.ones(2,128,device=device,dtype=torch.bool))
    if not torch.equal(gd.atom_count,torch.ones_like(gd.atom_count)):raise ValueError('Duplicate coalescing')
    return dict(passed=True,independent_tree=True,permutation=True,nesting=True,duplicates=True,
                exact_ties=True,fp32=True,independent_descriptors=True,cases=cases)


def stage_oracle(device):
    stage=MultiscaleStage().to(device).double()
    with torch.no_grad():stage.feedback_output.weight.normal_(std=.02)
    p,valid=fixture(device);g=build_geometry(p,valid)
    h=torch.randn(2,21,128,device=device,dtype=torch.float64,requires_grad=True)
    actual=stage(h,g,valid);expected=O.stage(stage,h,g,valid)
    for a,b in zip(actual[:3],expected):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    params=(h,*stage.parameters())
    ga=torch.autograd.grad(sum(x.square().mean() for x in actual[:3]),params,retain_graph=True)
    ge=torch.autograd.grad(sum(x.square().mean() for x in expected),params)
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    perm=torch.arange(20,-1,-1,device=device)
    pg=replace(g,fine=g.fine[:,perm],coarse=g.coarse[:,perm],fine_ids=g.fine_ids[:,perm],
               coarse_ids=g.coarse_ids[:,perm],atom_ids=g.atom_ids[:,perm])
    hp,fp,cp,_=stage(h[:,perm],pg,valid[:,perm])
    for a,b in ((hp,actual[0][:,perm]),(fp,actual[1]),(cp,actual[2])):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    # No event can attend another event. Rebuild only the selected event's map.
    one_g=build_geometry(p[:1],valid[:1]);one=stage(h[:1],one_g,valid[:1])
    for a,b in zip(one[:3],actual[:3]):torch.testing.assert_close(a,b[:1],rtol=1e-9,atol=1e-10)
    junk=h.detach().clone();junk[~valid]=float('nan');out=stage(junk,g,valid)
    for a,b in ((out[0][valid],actual[0][valid]),(out[1],actual[1]),(out[2],actual[2])):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    if not bool(actual[3]):raise ValueError('Stage oracle unhealthy')
    return dict(passed=True,precision='fp64',independent_stage=True,gradients=True,
                permutation=True,padding=True,event_isolation=True)


def oracle(device):
    a=Attention().to(device).double()
    q=torch.randn(2,3,128,device=device,dtype=torch.float64,requires_grad=True)
    s=torch.randn(2,5,128,device=device,dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([[1,0,1,1,0],[1,1,0,1,0]],device=device,dtype=torch.bool)
    qv=torch.tensor([[1,0,1],[1,1,0]],device=device,dtype=torch.bool)
    allowed=qv[:,:,None]&valid[:,None,:];allowed[:,0,2:]=False
    actual,ok=a(q,s,valid,allowed,qv);expected=O.attention(a,q,s,valid,allowed,qv)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    params=(q,s,*a.parameters())
    ga=torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
    ge=torch.autograd.grad(expected.square().sum(),params)
    for x,y in zip(ga,ge):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)
    if not bool(ok):raise ValueError('Attention oracle unhealthy')
    return dict(passed=True,precision='fp64',independent_head_loop=True,gradients=True,
        full_stage=stage_oracle(device),pooling=pooling_oracle(device),geometry=geometry_oracle(device))
