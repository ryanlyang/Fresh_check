import copy
import pytest
import torch
from part_ev_masked_predictive import runtime as R,views,verification as V
from part_ev_masked_predictive.restart import compare
from part_ev_fullscale.checkpoint import capture_rank_state

torch.set_num_threads(1);D=torch.device('cpu')


def test_independent_visible_formula_cardinality_alignment_and_permutation():assert V.view_oracle(D)['visible_rebase']


@pytest.mark.parametrize('n',[1,2,3,4,5,7,8,17,128])
def test_exact_mask_count_nominal_modes_and_generator_replay(n):
    x,_=R.tensors(R.fixed_batch(8,n),D);g=views.Generators(0);before=copy.deepcopy(g.state_dict())
    native=capture_rank_state(torch.nn.Identity());a=views.create((x[1],x[2],x[3]),g)
    compare(capture_rank_state(torch.nn.Identity()),native,exact=True)
    expected=0 if n<4 else min(n-2,max(1,n//4))
    assert (a.query_valid.sum(-1)==expected).all() and len(a.selected.unique())==4 and a.modes.sum()==2
    assert g.draws==1 and a.healthy
    g.load_state_dict(before);b=views.create((x[1],x[2],x[3]),g);assert a.fingerprint()==b.fingerprint()
    if n>=4:assert (a.visible.sum(-1)>=2).all()
    else:assert not a.eligible.any() and views.slot(a,0,4) is None


def test_generator_rank_and_state_identity_are_strict():
    a=views.Generators(0);b=views.Generators(1)
    assert not torch.equal(a.draw(8)[1],b.draw(8)[1])
    with pytest.raises(ValueError):a.load_state_dict(b.state_dict())
    old=a.state_dict();old['generators'].pop('ties')
    with pytest.raises(ValueError):a.load_state_dict(old)


def test_pack_wraps_only_visible_particles_and_preserves_inverse():
    p=torch.arange(2*4*7).reshape(2,4,7).float();f=torch.arange(2*17*7).reshape(2,17,7).float();points=f[:,:2]
    visible=torch.tensor([[1,0,0,1,0,0,0],[0,1,0,1,0,1,1]],dtype=torch.bool)
    packed,inverse=views.pack(points,f,p,visible)
    assert packed[3].shape==(2,1,4) and packed[3][0].sum()==2
    assert torch.equal(packed[2][0],p[0][:,[0,3,0,3]])
    assert inverse[0,0]==0 and inverse[0,3]==1 and inverse[0,1]==7
    assert torch.equal(packed[2][1],p[1][:,[1,3,5,6]])


def test_degenerate_visible_reference_is_ineligible_but_valid_nan_is_error():
    f=torch.zeros(2,17,4);p=torch.zeros(2,4,4);p[:,3]=1;p[0,0]=torch.tensor([1.,-1.,1.,-1.])
    valid=torch.ones(2,4,dtype=torch.bool);visible=valid.clone()
    _,_,eligible,healthy=views.rebase(f,p,valid,visible);assert healthy and not eligible.any()
    p[0,0,0]=float('nan');assert not views.rebase(f,p,valid,visible)[-1]


def test_all_duplicate_content_remains_distribution_equivariant():
    x,_=R.tensors(R.fixed_batch(8,17),D);x[1][:]=x[1][:,:,:1].clone();x[2][:]=x[2][:,:,:1].clone()
    a=views.create((x[1],x[2],x[3]),views.Generators(0))
    b=views.create(tuple(v.flip(-1) for v in (x[1],x[2],x[3])),views.Generators(0))
    for view in (a,b):
        item=views.slot(view,0,4);assert (item['visible'][3].sum(-1)==13).all()
    torch.testing.assert_close(a.locations,b.locations,rtol=0,atol=0)
