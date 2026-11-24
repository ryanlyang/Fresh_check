import copy
import pytest
import torch
from torch import nn
from part_ev_multiscale import model as M
from part_ev_multiscale import runtime as R,verification as V
from part_ev_multiscale.config import COUNTS
from part_ev_multiscale.geometry import observe

torch.set_num_threads(1)
D=torch.device('cpu')


def test_counts_norms_exactly_five_zero_outputs_and_native_rng():
    torch.manual_seed(101);base=M.build_model('control',trim=False);rng=torch.get_rng_state()
    torch.manual_seed(101);m=M.build_model(trim=False)
    assert torch.equal(torch.get_rng_state(),rng)
    assert sum(p.numel() for p in m.parameters())==COUNTS['multiscale']==5366950
    assert sum(p.numel() for p in m.mod.multiscale.parameters())==3039488
    for n,v in base.state_dict().items():assert torch.equal(v,m.state_dict()[n])
    assert len(set(s.particle_read.query.weight.data_ptr() for s in m.mod.multiscale.stages))==4
    for s in m.mod.multiscale.stages:
        assert sum(p.numel() for p in s.parameters())==714240
        assert sum(isinstance(x,nn.LayerNorm) for x in s.modules())==14
    zero=[n for n,x in m.mod.multiscale.named_modules() if isinstance(x,nn.Linear) and not x.weight.any()]
    assert zero==[f'stages.{i}.feedback_output' for i in range(4)]+['final_read.fusion_output']
    assert V.initial_identity(D,0,1)['rng_exact']


def test_first_ce_gradients_open_all_five_outputs_and_all_graph_paths_connected():
    m=M.build_model(trim=False);values,y=R.tensors(R.fixed_batch(8,37),D)
    torch.nn.functional.cross_entropy(m(*values),y).backward()
    for n,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        if n.endswith(('feedback_output.weight','fusion_output.weight')):assert p.grad.abs().max()>0,n


def test_checkpointed_active_backward_constructs_geometry_only_once():
    m=M.build_model(trim=False);V.activate(m);other=copy.deepcopy(m);other.mod.multiscale_checkpointed=False
    values,y=R.tensors(R.fixed_batch(8,21),D);records=[]
    for model in (m,other):
        torch.manual_seed(31)
        with observe() as rows:
            logits=model(*values);assert model._state_pair_cache.stem is None
            logits.square().mean().backward()
        assert len(rows)==1
        records.append(logits.detach())
    torch.testing.assert_close(*records,rtol=1e-5,atol=1e-6)
    for (n,p),(_,q) in zip(m.named_parameters(),other.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5,msg=n)


@pytest.mark.parametrize('kind',['one','duplicate','different_lengths'])
def test_degenerate_graph_no_unused_parameters(kind):
    m=M.build_model(trim=False);V.activate(m)
    values,y=R.tensors(R.fixed_batch(8,1 if kind=='one' else 21),D)
    if kind=='duplicate':values[2][:]=values[2][:,:,:1].clone()
    if kind=='different_lengths':values[3][0,:,1:]=False
    logits=m(*values);assert torch.isfinite(logits).all();logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_exception_and_repeated_forward_cleanup(monkeypatch):
    m=M.build_model(trim=False);values,_=R.tensors(R.fixed_batch(8,17),D)
    def broken(*a,**k):raise RuntimeError('stage fixture')
    with monkeypatch.context() as patch:
        patch.setattr(m.mod.multiscale.stages[1],'forward',broken)
        with pytest.raises(RuntimeError,match='stage fixture'):m(*values)
    assert m._state_pair_cache.stem is None
    m.eval()
    with torch.no_grad():a=m(*values);b=m(*values)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not any('geometry' in n or 'last_regions' in n for n in m.state_dict())


def test_attention_invalid_queries_zero_after_bias_and_empty_active_rejected():
    from part_ev_multiscale.attention import Attention
    a=Attention();a.output.bias.data.fill_(4)
    q=torch.randn(2,3,128);s=torch.randn(2,5,128);qv=torch.tensor([[1,0,1],[0,1,0]],dtype=torch.bool)
    valid=torch.ones(2,5,dtype=torch.bool);allowed=qv[:,:,None]&valid[:,None,:]
    q[~qv]=float('nan');output,ok=a(q,s,valid,allowed,qv)
    assert ok and (output[~qv]==0).all()
    allowed[:,0]=False;output,ok=a(q,s,valid,allowed,qv)
    assert not ok and torch.isfinite(output).all()
