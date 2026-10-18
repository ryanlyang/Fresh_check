import copy
import pytest
import torch
from torch.nn import functional as F
from part_ev_message_study import model as M
from part_ev_fullscale.model import build_model as reference
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads


def make(name, **kwargs):
    torch.manual_seed(101)
    return M.build_model(name, **kwargs)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("name", M.COUNTS)
def test_initial_identity_backbone_counts_rng_and_cleanup(name, training):
    torch.manual_seed(101)
    ev = reference("part_ev",trim=False).train(training)
    rng = torch.get_rng_state().clone()
    model = make(name,trim=False).train(training)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in ev.state_dict().items():
        torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    assert sum(p.numel() for p in model.parameters()) == M.COUNTS[name]
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})
    torch.manual_seed(918)
    expected = ev(*inputs())
    after = torch.get_rng_state().clone()
    torch.manual_seed(918)
    actual = model(*inputs())
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(after,torch.get_rng_state())
    actual.sum().backward()
    assert all(block.attn._relation_stem is None for block in model.mod.blocks)
    assert model.mod.pair_embed._trimmed_mask is None
    for block in model.mod.blocks:
        branch = block.attn.state or block.attn.routing or block.attn.variance
        if branch is not None:
            parameter = branch.projection.weight if isinstance(branch,M.ValueRouting) else branch.output
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_separate_stem_copies_buffers_but_not_storage_and_combination_matches():
    separate = make("ev_separate",trim=False)
    both = make("ev_state_separate",trim=False)
    state = make("ev_state",trim=False)
    stem = separate.mod.pair_embed.reference.embed
    for left,right in zip(stem.prefix.state_dict().values(),stem.value_prefix.state_dict().values()):
        assert left.data_ptr() != right.data_ptr()
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    for key,value in both.state_dict().items():
        if ".state." in key:
            torch.testing.assert_close(value,state.state_dict()[key],rtol=0,atol=0)
        elif ".value_prefix." in key:
            torch.testing.assert_close(value,separate.state_dict()[key],rtol=0,atol=0)
    separate(*inputs()).square().sum().backward()
    assert any(p.grad.abs().sum() > 0 for p in stem.prefix.parameters())
    assert any(p.grad.abs().sum() > 0 for p in stem.value_prefix.parameters())
    for a,b in zip(stem.prefix.modules(),stem.value_prefix.modules()):
        if isinstance(a,torch.nn.BatchNorm1d):
            assert a.num_batches_tracked == b.num_batches_tracked == 1


def test_state_explicit_directed_chunk_forward_and_gradients():
    torch.manual_seed(717)
    branch = M.StateMessage(width=7,relations=4,rank=3,heads=2,channels=2,chunk=2).double()
    with torch.no_grad():
        branch.output.normal_(std=.2)
    other = copy.deepcopy(branch)
    w = torch.rand(2,2,5,5,dtype=torch.double,requires_grad=True)
    z = torch.randn(2,4,5,5,dtype=torch.double,requires_grad=True)
    x = torch.randn(5,2,7,dtype=torch.double,requires_grad=True)
    w2,z2,x2 = [v.detach().clone().requires_grad_() for v in (w,z,x)]
    valid = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
    actual = branch(w,z,x,valid)
    p,q = other.receiver(M.normalize(x2.transpose(0,1))),other.sender(M.normalize(x2.transpose(0,1)))
    features = other.features(z2,p,q,valid[:,:,None]&valid[:,None,:])
    assert not torch.allclose(features[0],features[0].transpose(0,1))
    edge_messages = torch.einsum("hor,bijr->bhijo",other.output,features)
    expected = (w2[...,None]*edge_messages).sum(3)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a,b in zip((w,z,x,*branch.parameters()),(w2,z2,x2,*other.parameters())):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-8,atol=1e-10)
    assert actual[1,:,3:].count_nonzero() == 0
    branch.chunk_size = 5
    torch.testing.assert_close(branch(w,z,x,valid),expected)


def test_state_zero_output_learns_upstream_after_first_step():
    branch = M.StateMessage(width=7,relations=4,rank=3,heads=2,channels=2)
    args = (torch.rand(2,2,4,4),torch.randn(2,4,4,4),
            torch.randn(4,2,7),torch.ones(2,4,dtype=torch.bool))
    branch(*args).sum().backward()
    assert branch.output.grad.abs().sum() > 0
    assert branch.receiver.weight.grad.count_nonzero() == 0
    with torch.no_grad():
        branch.output.add_(branch.output.grad,alpha=-.01)
    branch.zero_grad()
    branch(*args).sum().backward()
    assert all(p.grad.abs().sum() > 0 for p in branch.parameters())


def test_variance_explicit_reference_gradients_zero_rows_and_single_key():
    branch = M.VarianceMessage(heads=2,channels=3,relations=4,chunk=2).double()
    with torch.no_grad():
        branch.output.normal_(std=.1)
    other = copy.deepcopy(branch)
    w = torch.rand(2,2,5,5,dtype=torch.double,requires_grad=True)
    z = torch.randn(2,4,5,5,dtype=torch.double,requires_grad=True)
    w2,z2 = [v.detach().clone().requires_grad_() for v in (w,z)]
    valid = torch.tensor([[1,1,1,1,1],[1,0,0,0,0]],dtype=torch.bool)
    with torch.no_grad():
        w[0,:,0] = 0
        w2[0,:,0] = 0
    actual = branch(w,z,valid)
    pair = valid[:,:,None]&valid[:,None,:]
    weights = w2.masked_fill(~pair[:,None],0)
    total = weights.sum(-1,keepdim=True)
    pi = weights/torch.where(total>0,total,torch.ones_like(total))
    features = M.normalize(z2.permute(0,2,3,1))
    mean = (pi[...,None]*features[:,None]).sum(3)
    variance = (pi[...,None]*(features[:,None]-mean.unsqueeze(3)).square()).sum(3)
    expected = torch.einsum("hod,bhid->bhio",other.output,variance)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a,b in zip((w,z,*branch.parameters()),(w2,z2,*other.parameters())):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-10,rtol=1e-8)
        assert torch.isfinite(a.grad).all()
    assert actual[0,:,0].count_nonzero() == actual[1].count_nonzero() == 0
    constant = torch.randn(2,4,5,1,dtype=torch.double).expand(2,4,5,5)
    torch.testing.assert_close(branch(w,constant,valid),torch.zeros_like(actual),atol=1e-12,rtol=0)


@pytest.mark.parametrize("training",[False,True])
def test_routing_matches_shifted_logits_same_dropout_and_gradients(training):
    torch.manual_seed(88)
    route = M.ValueRouting(relations=4,heads=2,chunk=2).double()
    with torch.no_grad():
        route.projection.weight.normal_(std=.1)
    other = copy.deepcopy(route)
    q = torch.randn(4,3,5,dtype=torch.double,requires_grad=True)
    k = torch.randn(4,4,5,dtype=torch.double,requires_grad=True)
    z = torch.randn(2,4,3,4,dtype=torch.double,requires_grad=True)
    q2,k2,z2 = [v.detach().clone().requires_grad_() for v in (q,k,z)]
    mask = torch.zeros(4,3,4,dtype=torch.double)
    mask[:,:,-1] = -torch.inf
    score = torch.baddbmm(mask,q*(1/5)**.5,k.transpose(1,2))
    alpha = score.softmax(-1).reshape(2,2,3,4)
    dropout = (torch.rand_like(alpha) > .3).double()/.7 if training else torch.ones_like(alpha)
    actual = route.chunk(alpha*dropout,z,q,k,mask)
    delta = 2*other.projection(M.normalize(z2.permute(0,2,3,1))).permute(0,3,1,2).tanh()
    score2 = torch.baddbmm(mask,q2*(1/5)**.5,k2.transpose(1,2)).reshape_as(alpha)
    expected = (score2+delta).softmax(-1)*dropout
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a,b in zip((q,k,z,*route.parameters()),(q2,k2,z2,*other.parameters())):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-10,rtol=1e-8)


@pytest.mark.parametrize("name",M.COUNTS)
def test_nonzero_mechanisms_permutation_masks_and_chunk_invariance(name):
    model = make(name,trim=False).eval()
    with torch.no_grad():
        for block in model.mod.blocks:
            for branch in (block.attn.state,block.attn.variance):
                if branch is not None:
                    branch.output.normal_(std=.03)
            if block.attn.routing is not None:
                block.attn.routing.projection.weight.normal_(std=.03)
        data = inputs()
        result = model(*data)
        order = torch.randperm(data[0].shape[-1])
        torch.testing.assert_close(model(*(x[:,:,order] for x in data)),result,atol=3e-6,rtol=2e-4)
        for block in model.mod.blocks:
            for branch in (block.attn.state,block.attn.routing,block.attn.variance):
                if branch is not None:
                    branch.chunk_size = 3
        torch.testing.assert_close(model(*data),result,atol=3e-6,rtol=2e-4)
        assert torch.isfinite(model(*inputs(batch=1,particles=1))).all()


@pytest.mark.parametrize("name",M.COUNTS)
def test_cpu_autocast_identity_and_backward_dtype_compatibility(name):
    # Local dtype-path check only; it does not certify CUDA memory or numerics.
    torch.manual_seed(101)
    ev = reference("part_ev",trim=False).train()
    model = make(name,trim=False).train()
    torch.manual_seed(91)
    with torch.autocast("cpu",dtype=torch.float16):
        expected = ev(*inputs(batch=2,particles=7))
    torch.manual_seed(91)
    with torch.autocast("cpu",dtype=torch.float16):
        actual = model(*inputs(batch=2,particles=7))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.float().square().sum().backward()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
