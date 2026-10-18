import copy
import pytest
import torch
from part_ev_fullscale.checkpoint import seed_everything
from part_ev_fullscale.model import build_model as baseline
from part_group_screen.model import build_model
from part_group_screen.config import ARCHITECTURES, CONFIG
from part_group_screen.grouping import fixed_assignment, random_assignment, SoftRouter
from part_group_screen.geometry import descriptors, locality_penalty
from part_group_screen.metadata import random_order
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads


def sample(n=13):
    points, features, vectors, mask = inputs(batch=2, particles=n)
    return features.transpose(1, 2), vectors.transpose(1, 2), mask[:, 0].bool()


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_initialization_disabled_parity_and_backward(architecture):
    seed_everything(101)
    reference = baseline('part', trim=False).eval()
    expected_rng = torch.get_rng_state().clone()
    seed_everything(101)
    model = build_model(architecture, trim=False).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, tensor in reference.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name
    args = inputs(batch=2, particles=13)
    model.group_branch.disabled = True
    torch.testing.assert_close(model(*args), reference(*args), rtol=0, atol=0)
    model.group_branch.disabled = False
    result = model(*args)
    assert torch.isfinite(result).all()
    result.square().sum().backward()
    for name, p in model.group_branch.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model._group_context is None


def test_common_branch_router_initialization():
    models = {}
    for name in ARCHITECTURES:
        seed_everything(101)
        models[name] = build_model(name)
    for key, value in models['fixed'].group_branch.state_dict().items():
        for model in models.values():
            assert torch.equal(model.group_branch.state_dict()[key], value), key
    for key, value in models['learned'].group_branch.state_dict().items():
        assert torch.equal(models['hybrid'].group_branch.state_dict()[key], value)


@pytest.mark.parametrize('assignment', [fixed_assignment, random_assignment])
@pytest.mark.parametrize('n', [1, 5, 8, 13, 32])
def test_hard_masks_sizes_permutation_and_rng(assignment, n):
    f, v, valid = sample(n)
    before = torch.get_rng_state().clone()
    a, active = assignment(v, f, valid)
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(a.sum(-1), valid.float(), rtol=0, atol=0)
    assert (a * (~active[:, None])).sum() == 0
    fixed, _ = fixed_assignment(v, f, valid)
    assert torch.equal(a.sum(1).sort().values, fixed.sum(1).sort().values)
    perm = torch.randperm(n)
    shuffled, active2 = assignment(v[:, perm], f[:, perm], valid[:, perm])
    torch.testing.assert_close(a[:, perm], shuffled, rtol=0, atol=0)
    assert torch.equal(active, active2)
    groups = torch.einsum('bnk,bnd->bkd', a, v)
    torch.testing.assert_close(groups.sum(1), (v * valid[..., None]).sum(1))


def test_fixed_hand_fixture():
    # Two close pairs and one isolated particle, clustered to three groups.
    phi = torch.tensor([[0., .01, 1., 1.02, 2.5]])
    v = torch.stack((phi.cos(), phi.sin(), torch.zeros_like(phi), torch.ones_like(phi)), -1)
    f, valid = torch.zeros(1, 5, 17), torch.ones(1, 5, dtype=torch.bool)
    a, _ = fixed_assignment(v, f, valid, groups=3)
    pair = a @ a.transpose(1, 2)
    assert pair[0, 0, 1] == 1 and pair[0, 2, 3] == 1
    assert pair[0, 0, 2] == 0 and pair[0, 4, :4].sum() == 0


def test_fixed_against_independent_small_cpu_oracle():
    import math
    f,v,valid=sample(16)
    a,_=fixed_assignment(v,f,valid,groups=4)
    for b in range(2):
        groups=[([i],v[b,i].double().tolist()) for i in range(16) if valid[b,i]]
        def coord(vec):
            px,py,pz,e=vec
            return .5*math.log(max(e+pz,1e-6)/max(e-pz,1e-6)),math.atan2(py,px)
        while len(groups)>4:
            candidates=[]
            for i in range(len(groups)):
                yi,pi=coord(groups[i][1])
                for j in range(i+1,len(groups)):
                    yj,pj=coord(groups[j][1])
                    dp=math.atan2(math.sin(pi-pj),math.cos(pi-pj))
                    candidates.append(((yi-yj)**2+dp**2,i,j))
            _,i,j=min(candidates)
            groups[i]=(groups[i][0]+groups[j][0],[x+y for x,y in zip(groups[i][1],groups[j][1])])
            groups.pop(j)
        expected={frozenset(ids) for ids,_ in groups}
        actual={frozenset(torch.where(a[b,:,k]>0)[0].tolist()) for k in range(4)}
        assert actual==expected


def test_random_duplicate_inputs_and_masked_features():
    f,v,valid=sample(16)
    f[:,1]=f[:,0];v[:,1]=v[:,0]
    a,_=random_assignment(v,f,valid)
    changed=f.clone();changed[~valid]=1e9
    b,_=random_assignment(v,changed,valid)
    assert torch.equal(a,b)
    perm=torch.randperm(16)
    c,_=random_assignment(v[:,perm],f[:,perm],valid[:,perm])
    # Truly duplicate constituents can exchange keys; physical group contents
    # still agree, and distinct inputs retain full canonical tie breaking.
    for batch in range(v.shape[0]):
        for group in range(a.shape[-1]):
            original = v[batch, a[batch, :, group].bool()].tolist()
            shuffled = v[batch, perm][c[batch, :, group].bool()].tolist()
            assert sorted(original) == sorted(shuffled)
    # Reduction order can change FP32 last bits even with identical membership.
    torch.testing.assert_close(torch.einsum('bnk,bnd->bkd',a,v),
                               torch.einsum('bnk,bnd->bkd',c,v[:,perm]))


def test_hybrid_zero_equivalence_and_gradient():
    f, v, valid = sample()
    h = torch.randn(2, 13, 128, requires_grad=True)
    learned = SoftRouter(CONFIG)
    hybrid = copy.deepcopy(learned)
    a, _ = learned(h, v, valid)
    b, _ = hybrid(h, v, valid)
    assert torch.equal(a, b)
    g1 = torch.autograd.grad(a.square().sum(), h, retain_graph=True)[0]
    g2 = torch.autograd.grad(b.square().sum(), h)[0]
    assert torch.equal(g1, g2)
    hybrid.locality = 1.
    b, _ = hybrid(h, v, valid)
    b.square().sum().backward()
    for name, p in hybrid.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_degenerate_geometry_finite_gradients():
    v = torch.tensor([[[1., 0., 0., 1.]] * 4, [[0., 0., 0., 0.]] * 4])
    valid = torch.ones(2, 4, dtype=torch.bool)
    raw = torch.zeros(2, 4, 8, requires_grad=True)
    a = raw.softmax(-1)
    n, groups, f = descriptors(a, v, valid)
    penalty = locality_penalty(a, v, valid)
    assert torch.isfinite(f).all() and torch.isfinite(penalty).all()
    (f.square().sum() + penalty.square().sum()).backward()
    assert torch.isfinite(raw.grad).all()


def test_trimming_context_cleanup_and_permutation():
    seed_everything(101)
    model = build_model('hybrid', trim=True).eval()
    args = inputs(batch=2, particles=13)
    model.mod.trimmer._counter = 5
    before = model(*args)
    perm = torch.randperm(13)
    torch.testing.assert_close(before, model(*(x[..., perm] for x in args)), rtol=2e-5, atol=2e-6)
    assert model._group_context is None
    with pytest.raises(ValueError, match='at least one'):
        model(*args[:-1], torch.zeros_like(args[-1]))
    assert model._group_context is None
