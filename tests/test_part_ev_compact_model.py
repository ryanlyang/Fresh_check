import pytest
import torch
from torch import nn
from part_ev_compact import model as M
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads


def build(arch='compact', **kw):
    torch.manual_seed(101)
    return M.build_model(arch, trim=False, **kw)


def test_counts_structure_and_rng():
    from part_ev_fullscale.model import build_model as native
    torch.manual_seed(101); native('part', trim=False); expected=torch.get_rng_state()
    for arch in M.COUNTS:
        model=build(arch)
        assert sum(p.numel() for p in model.parameters()) == M.COUNTS[arch]
        assert torch.equal(expected,torch.get_rng_state())
        assert all(b.ffn_dim==512 for b in model.mod.cls_blocks)
    compact=build()
    assert len(compact.mod.blocks)==8
    assert len([n for n,p in compact.named_parameters() if n=='shared_state_edge.weight'])==1
    assert len([n for n in compact.state_dict() if n.endswith('receiver.weight')])==8
    assert len(compact.mod.pair_embed.reference.embed.heads)==8
    matched=build('matched')
    assert all(type(b.attn) is nn.MultiheadAttention and b.ffn_dim==570 for b in matched.mod.blocks)
    assert not any('state' in n or 'edge_projection' in n for n,p in matched.named_parameters())


@pytest.mark.parametrize('n,batch',[(1,1),(11,3),(65,2),(128,2)])
@pytest.mark.parametrize('training',[False,True])
def test_nonzero_execution_forward_and_all_gradients(n,batch,training):
    candidate,oracle=build(),build(optimized=False)
    with torch.no_grad():
        for block in candidate.mod.blocks: block.attn.attention.state.output.normal_(0,.03)
    oracle.load_state_dict(candidate.state_dict())
    candidate.train(training); oracle.train(training)
    # Native BatchNorm cannot train on a single particle in a single jet.
    if n == 1 and training:
        from part_ev_compact.flops import synthetic
        x=synthetic(2,1,torch.device('cpu'))
    else:
        x=inputs(batch=batch,particles=n)
    torch.manual_seed(20); a=candidate(*x)
    torch.manual_seed(20); b=oracle(*x)
    torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
    a.square().mean().backward(); b.square().mean().backward()
    for (name,p),(other,q) in zip(candidate.named_parameters(),oracle.named_parameters()):
        assert name==other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-3,msg=lambda s:name+': '+s)
    assert candidate._compact_cache.stem is None
    if n>1: assert candidate.shared_state_edge.weight.grad.abs().sum()>0


def test_zero_state_matches_ordinary_layerwise_ev():
    from part_layerwise_screen.model import build_model
    candidate=build().eval()
    torch.manual_seed(101); ordinary=build_model('part_ev_layerwise',trim=False).eval()
    ordinary.load_state_dict({k:v for k,v in candidate.state_dict().items()
                             if '.state.' not in k and not k.startswith('shared_state_edge.')})
    with torch.no_grad():
        torch.testing.assert_close(candidate(*inputs()),ordinary(*inputs()),atol=3e-6,rtol=3e-5)


def test_shared_projection_once_per_forward_and_exception_cleanup():
    model=build().train(); calls=[]
    handle=model.shared_state_edge.register_forward_hook(lambda *_: calls.append(1))
    model(*inputs()).sum().backward()
    assert len(calls)==1
    handle.remove()
    def fail(*args): raise RuntimeError('intentional')
    hook=model.mod.blocks[2].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError,match='intentional'): model(*inputs())
    assert model._compact_cache.stem is None
    assert all(b.attn.attention.stem is None for b in model.mod.blocks)
    hook.remove()
    assert torch.isfinite(model(*inputs())).all()


def test_permutation_equivariance_eval():
    model=build().eval()
    x=inputs(); perm=torch.randperm(11)
    with torch.no_grad():
        torch.testing.assert_close(model(*x),model(*(v[:,:,perm] for v in x)),rtol=2e-5,atol=2e-6)


def test_explicit_attention_bool_and_float_masks_double():
    ref=nn.MultiheadAttention(128,8,dropout=0).double().eval()
    q=torch.randn(7,2,128,dtype=torch.float64,requires_grad=True)
    pad=torch.zeros(2,7,dtype=torch.bool);pad[1,-2:]=True
    for mask in (torch.zeros(16,7,7,dtype=torch.float64),torch.zeros(16,7,7,dtype=torch.bool)):
        pre,w=M.pre_output_attention(ref,q,pad,mask)
        a=torch.nn.functional.linear(pre,ref.out_proj.weight,ref.out_proj.bias)
        b,wb=ref(q,q,q,key_padding_mask=pad,attn_mask=mask,average_attn_weights=False)
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(w,wb,atol=1e-12,rtol=1e-12)


def test_cpu_integration_and_complete_restart():
    from part_ev_compact.parity import execution,restart
    x=inputs(batch=4,particles=11)
    batch=({k:v for k,v in zip(('pf_points','pf_features','pf_vectors','pf_mask'),x)},
           {'_label_':torch.arange(4)}, {})
    assert execution(batch,torch.device('cpu'))['passed']
    for arch in M.COUNTS:
        assert restart(build(arch),batch,torch.device('cpu'))['exact_each_update']
