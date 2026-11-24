"""Independent functional/particle-loop oracles, not zero-output self-tests."""
import copy
import pytest
import torch
from torch.nn import functional as F
from part_ev_ffn_experts.model import SpecialistBank
from part_ev_ffn_experts.losses import from_global,balancing

torch.set_num_threads(1)


def active(bank):
    with torch.no_grad():
        for e in bank.experts:e.output.weight.normal_(std=.02);e.output.bias.normal_(std=.01)


def oracle(bank,u,mask):
    result=torch.zeros_like(u);all_p=[];all_ids=[]
    for b in range(u.shape[1]):
        valid=[i for i in range(u.shape[0]) if mask[b,i]]
        mean=torch.stack([u[i,b] for i in valid]).sum(0)/len(valid)
        g=F.layer_norm(mean,(128,),bank.context_norm.weight,bank.context_norm.bias,1e-5)
        for i in valid:
            x=u[i,b]
            z=F.linear(F.gelu(F.linear(torch.cat((x,g,x*g)),bank.router[0].weight,bank.router[0].bias)),
                bank.router[2].weight,bank.router[2].bias)
            ids=sorted(range(4),key=lambda e:(-float(z[e].detach()),e))[:2]
            w=z[ids].softmax(0);parts=[]
            for e in ids:
                expert=bank.experts[e]
                hidden=F.gelu(F.linear(x,expert.input.weight,expert.input.bias))
                hidden=F.layer_norm(hidden,(256,),expert.norm.weight,expert.norm.bias,1e-5)
                parts.append(F.linear(hidden,expert.output.weight,expert.output.bias))
            result[i,b]=w[0]*parts[0]+w[1]*parts[1]
            all_p.append(z.softmax(0));all_ids.append(ids)
    return result,torch.stack(all_p),torch.tensor(all_ids)


def test_fp64_context_dispatch_and_all_gradients_match_particle_loop():
    torch.manual_seed(18);bank=SpecialistBank(checkpointed=False).double();active(bank)
    u=torch.randn(5,3,128,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[1,0,1,1,0],[0,1,0,0,1],[1,1,1,1,1]],dtype=torch.bool)
    actual,stats,healthy=bank(u,mask,details=True)
    expected,p,ids=oracle(bank,u,mask)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(stats['probability_sum'],p.sum(0),rtol=1e-9,atol=1e-10)
    assert torch.equal(stats['details']['ids'],ids)
    assert healthy and stats['requested'].sum()==2*mask.sum()
    assert torch.equal(stats['requested'],stats['executed'])
    params=(u,*bank.parameters())
    a=torch.autograd.grad(actual.square().sum()+stats['probability_sum'].square().sum(),params,retain_graph=True)
    b=torch.autograd.grad(expected.square().sum()+p.sum(0).square().sum(),params,allow_unused=True)
    for x,y in zip(a,b):
        if y is None:y=torch.zeros_like(x)
        torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('checkpointed',[False,True])
def test_exact_ties_overload_empty_experts_have_zero_not_missing_gradients(checkpointed):
    bank=SpecialistBank(checkpointed=checkpointed)
    with torch.no_grad():bank.router[2].weight.zero_();bank.router[2].bias.zero_()
    u=torch.randn(5,3,128,requires_grad=True);mask=torch.ones(3,5,dtype=torch.bool)
    out,s,healthy=bank(u,mask,details=True)
    assert torch.equal(s['details']['ids'],torch.tensor([[0,1]]*15))
    assert torch.equal(s['requested'],torch.tensor([15,15,0,0]))
    assert torch.equal(s['requested'],s['executed']) and healthy
    (out.sum()+s['probability_sum'].square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bank.parameters())
    assert all(p.grad.count_nonzero()==0 for e in bank.experts[2:] for p in e.parameters())
    torch.testing.assert_close(s['details']['weights'],torch.full((15,2),.5))


def test_padding_permutation_jet_isolation_and_one_particle():
    torch.manual_seed(33);bank=SpecialistBank().double();active(bank)
    u=torch.randn(7,3,128,dtype=torch.float64);mask=torch.rand(3,7)>.3;mask[:,0]=True
    a,s,_=bank(u,mask,details=True)
    assert s['min_margin']>1e-7
    perm=torch.tensor([6,0,3,2,1,5,4]);b,t,_=bank(u[perm],mask[:,perm],details=True)
    torch.testing.assert_close(a[perm],b,rtol=1e-9,atol=1e-10)
    assert torch.equal(s['requested'],t['requested'])
    junk=u.clone();junk[~mask.T]=float('nan')
    c,_,ok=bank(junk,mask);assert ok
    torch.testing.assert_close(a,c,rtol=1e-9,atol=1e-10)
    one,_,_=bank(u[:,:1],mask[:1]);torch.testing.assert_close(one,a[:,:1],rtol=1e-9,atol=1e-10)
    out,_,ok=bank(u[:1],torch.ones(3,1,dtype=torch.bool));assert ok and out.shape==(1,3,128)


def test_uniform_balance_is_one_and_six_bank_mean_not_sum():
    sums=torch.full((6,4),2.,requires_grad=True)
    counts=torch.full((6,4),4,dtype=torch.int64);tokens=torch.full((6,),8,dtype=torch.int64)
    loss,values=from_global(sums,sums.detach(),counts,tokens,1)
    assert loss==1 and torch.equal(values,torch.ones(6))
    loss.backward();torch.testing.assert_close(sums.grad,torch.full((6,4),1/48))


def test_unequal_population_ddp_proxy_parameter_and_input_gradients():
    torch.manual_seed(44);global_bank=SpecialistBank(checkpointed=False).double();active(global_bank)
    local_banks=[copy.deepcopy(global_bank),copy.deepcopy(global_bank)]
    u=torch.randn(7,4,128,dtype=torch.float64,requires_grad=True)
    mask=torch.ones(4,7,dtype=torch.bool);mask[:2,1:]=False
    _,_,_=global_bank(u,mask)
    _,p,ids=oracle(global_bank,u,mask)
    counts=torch.bincount(ids.flatten(),minlength=4).double()
    direct=4*((p.sum(0)/mask.sum())*(counts/(2*mask.sum()))).sum()
    direct.backward();expected=[p.grad.clone() for p in global_bank.parameters() if p.grad is not None]
    # Only router/context contribute to this auxiliary-only comparison.
    expected={n:p.grad.clone() for n,p in global_bank.named_parameters() if p.grad is not None}
    locals=[];naive=[]
    for rank,bank in enumerate(local_banks):
        x=u.detach()[:,rank*2:(rank+1)*2].clone().requires_grad_()
        _,stats,_=bank(x,mask[rank*2:(rank+1)*2]);locals.append((x,stats))
        naive.append(4*(stats['probability_sum']/stats['tokens']*(stats['requested']/(2*stats['tokens']))).sum())
    gs=sum(s['probability_sum'].detach() for _,s in locals)
    gc=sum(s['requested'] for _,s in locals);gt=sum(s['tokens'] for _,s in locals)
    assert int(locals[0][1]['tokens'])!=int(locals[1][1]['tokens'])
    for x,s in locals:
        loss,value=from_global(s['probability_sum'][None],gs[None],gc[None],gt[None],2)
        torch.testing.assert_close(value[0],direct.detach(),rtol=1e-9,atol=1e-10)
        loss.backward(retain_graph=True)
    for name,gradient in expected.items():
        average=sum(dict(b.named_parameters())[name].grad for b in local_banks)/2
        torch.testing.assert_close(average,gradient,rtol=1e-9,atol=1e-10)
    for rank,(x,s) in enumerate(locals):
        torch.testing.assert_close(x.grad/2,u.grad[:,rank*2:(rank+1)*2],rtol=1e-9,atol=1e-10)
    assert abs(float((sum(naive)/2-direct).detach()))>1e-6


def test_valid_nonfinite_router_or_expert_is_not_hidden_by_zero_output():
    bank=SpecialistBank();u=torch.randn(3,2,128);mask=torch.ones(2,3,dtype=torch.bool)
    with torch.no_grad():bank.router[2].weight.zero_();bank.router[2].bias.zero_();bank.experts[0].input.weight[0,0]=float('inf')
    _,_,ok=bank(u,mask);assert not ok
    _,_,ok=SpecialistBank()(u,torch.zeros_like(mask));assert not ok
