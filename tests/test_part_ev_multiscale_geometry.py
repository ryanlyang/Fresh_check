"""Geometry/descriptor/segmented-pooling edge cases and independent formulas."""
from dataclasses import replace
import pytest
import torch
from part_ev_multiscale.geometry import build_geometry,coordinates,pairwise_sum,mapping_digest,observe
from part_ev_multiscale.verification_math import fixture,pooling_oracle
from part_ev_multiscale.oracles import tree
from part_ev_multiscale.pooling import pool

torch.set_num_threads(1)


@pytest.mark.parametrize('n',[1,2,4,16,17,31,128])
def test_nested_cuts_preserve_every_particle_and_permutation(n):
    p,valid=fixture(torch.device('cpu'),n)
    if n>4:p[:,2]=p[:,0];p[:,3]=p[:,0]
    g=build_geometry(p,valid)
    assert g.healthy and not g.fine.requires_grad
    assert torch.equal(g.fine@g.parent,g.coarse)
    assert torch.equal(g.fine.sum(-1),valid.float())
    assert torch.equal(g.coarse.sum(-1),valid.float())
    assert (g.fine_ids[~valid]==-1).all() and (g.coarse_ids[~valid]==-1).all()
    assert torch.equal(g.parent.sum(-1),g.fine_valid.float())
    torch.testing.assert_close(g.fine_vectors.sum(1),p.masked_fill(~valid[...,None],0).sum(1),rtol=1e-5,atol=1e-5)
    perm=torch.arange(n-1,-1,-1)
    other=build_geometry(p[:,perm],valid[:,perm])
    for name in ('fine','coarse','atom_ids'):
        assert torch.equal(getattr(other,name),getattr(g,name)[:,perm])
    for name in ('fine_vectors','coarse_vectors','fine_count','coarse_count','parent'):
        assert torch.equal(getattr(other,name),getattr(g,name))


def test_exact_tie_tree_and_indivisible_duplicate_atoms():
    p=torch.zeros(2,21,4);p[...,0]=torch.arange(1,22);p[...,3]=p[...,0]+2
    valid=torch.ones(2,21,dtype=torch.bool);p[0,1]=p[0,0]
    g=build_geometry(p,valid);reference=tree(p,valid)
    assert g.exact_ties.sum()>0 and g.fine_ids[0,0]==g.fine_ids[0,1]
    for b,row in enumerate(reference):
        for scale in ('fine','coarse'):
            for j,(_,vector,members) in enumerate(row[scale]):
                assert set(torch.where(getattr(g,scale)[b,:,j])[0].tolist())==members
                torch.testing.assert_close(getattr(g,scale+'_vectors')[b,j],vector,rtol=0,atol=0)
    duplicate=p[:,:1].expand(-1,128,-1)
    d=build_geometry(duplicate,torch.ones(2,128,dtype=torch.bool))
    assert (d.atom_count==1).all() and (d.fine_count[:,0]==128).all()


def test_padding_nonprefix_signed_zero_wrap_and_health():
    p,valid=fixture(torch.device('cpu'),21);p[~valid]=float('nan')
    p[0,0]=torch.tensor([1.,0.,0.,2.]);p[0,1]=torch.tensor([1.,-0.,-0.,2.])
    g=build_geometry(p,valid);assert g.healthy and g.atom_ids[0,0]==g.atom_ids[0,1]
    p[0,2,3]=-1;assert not build_geometry(p,valid).healthy
    p[0,2,3]=float('inf');assert not build_geometry(p,valid).healthy
    assert not build_geometry(torch.zeros_like(p),torch.zeros_like(valid)).healthy
    q=torch.tensor([[-1.,1e-6,0.,2.],[-1.,-1e-6,0.,2.]])
    _,y,phi=coordinates(q);delta=phi[0]-phi[1]
    assert torch.atan2(delta.sin(),delta.cos()).abs()<3e-6


def test_descriptors_from_original_particles_not_fine_child_approximation():
    p,valid=fixture(torch.device('cpu'),31);g=build_geometry(p,valid)
    # Independent event/region loops with prescribed FP32 mass arithmetic.
    for b in range(2):
        ordered=sorted({tuple(float(x) for x in row) for row in p[b,valid[b]]})
        atom=torch.stack([torch.tensor(t)*(p[b,valid[b]]==torch.tensor(t)).all(-1).sum() for t in ordered])
        jet=pairwise_sum(atom[None])[0];_,jy,jphi=coordinates(jet)
        sorted_p=torch.tensor(sorted([tuple(float(x) for x in row) for row in p[b,valid[b]]]))
        scalar_pt=pairwise_sum(coordinates(sorted_p)[0][None,:,None])[0,0]
        for scale in ('fine','coarse'):
            for j in torch.where(getattr(g,scale+'_valid')[b])[0]:
                members=p[b,getattr(g,scale)[b,:,j].bool()];v=getattr(g,scale+'_vectors')[b,j]
                pt,y,phi=coordinates(members);gp,gy,gphi=coordinates(v)
                angle=torch.atan2((gphi-jphi).sin(),(gphi-jphi).cos())
                dphi=torch.atan2((phi-gphi).sin(),(phi-gphi).cos())
                width=(sum(pt*((y-gy).square()+dphi.square()))/sum(pt).clamp_min(1e-6)).clamp_min(0).sqrt()
                mass=(v[3].square()-v[:3].square().sum()).clamp_min(0).sqrt()
                expected=torch.stack(((gp/scalar_pt.clamp_min(1e-6)).clamp_min(1e-6).log(),mass.log1p(),
                    gy-jy,angle.sin(),angle.cos(),torch.tensor(float(len(members))).log1p(),width.log1p()))
                torch.testing.assert_close(getattr(g,scale+'_descriptors')[b,j],expected,rtol=1e-5,atol=1e-5)


def test_segmented_max_values_and_equal_tie_input_gradients():
    assert pooling_oracle(torch.device('cpu'))['tie_gradients']
    p=torch.tensor([[[1.,0.,0.,2.]]]).expand(1,5,4);valid=torch.ones(1,5,dtype=torch.bool)
    g=build_geometry(p,valid);h=torch.ones(1,5,3,dtype=torch.float64,requires_grad=True)
    _,maximum=pool(h,g,valid);maximum.sum().backward()
    torch.testing.assert_close(h.grad,torch.full_like(h,1/5),rtol=0,atol=0)


def test_finite_overflow_is_false_health_not_out_of_bounds_mapping():
    p=torch.full((2,21,4),1e38);p[...,0]=torch.linspace(1e37,1e38,21)
    valid=torch.ones(2,21,dtype=torch.bool);g=build_geometry(p,valid)
    assert not g.healthy and torch.equal(g.fine@g.parent,g.coarse)
    assert (g.fine_ids>=0).all() and (g.fine_ids<16).all()


def test_observation_is_opt_in_one_build_and_no_backward_geometry():
    p,v=fixture(torch.device('cpu'),17)
    with observe() as rows:g=build_geometry(p,v)
    assert len(rows)==1 and rows[0]['mapping_sha256']==mapping_digest(g)
    h=torch.randn(2,17,128,requires_grad=True)
    with observe() as rows:
        sum(x.sum() for x in pool(h,g,v)).backward()
    assert rows==[] and h.grad is not None
