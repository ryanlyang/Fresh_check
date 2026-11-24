"""Bounded independent functional/particle-loop math, used on target devices."""
import copy
import torch
from torch.nn import functional as F
from .model import SpecialistBank
from .losses import from_global


def bank_reference(bank,u,mask):
    result=torch.zeros_like(u);probabilities=[];routes=[]
    for b in range(u.shape[1]):
        positions=mask[b].nonzero(as_tuple=False).flatten()
        mean=u.index_select(0,positions)[:,b].sum(0)/len(positions)
        g=F.layer_norm(mean,(128,),bank.context_norm.weight,bank.context_norm.bias,1e-5)
        for i in positions.tolist():
            x=u[i,b]
            logits=F.linear(F.gelu(F.linear(torch.cat((x,g,x*g)),bank.router[0].weight,bank.router[0].bias)),
                bank.router[2].weight,bank.router[2].bias)
            ids=sorted(range(4),key=lambda e:(-float(logits[e].detach()),e))[:2]
            weights=logits[ids].softmax(0);parts=[]
            for e in ids:
                expert=bank.experts[e]
                h=F.gelu(F.linear(x,expert.input.weight,expert.input.bias))
                h=F.layer_norm(h,(256,),expert.norm.weight,expert.norm.bias,1e-5)
                parts.append(F.linear(h,expert.output.weight,expert.output.bias))
            result[i,b]=weights[0]*parts[0]+weights[1]*parts[1]
            probabilities.append(logits.softmax(0));routes.append(ids)
    return result,torch.stack(probabilities),torch.tensor(routes,device=u.device)


def check(device):
    torch.manual_seed(811);bank=SpecialistBank(checkpointed=False).to(device).double()
    with torch.no_grad():
        for e in bank.experts:e.output.weight.normal_(std=.02)
    u=torch.randn(4,3,128,device=device,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[1,0,1,1],[0,1,0,0],[1,1,1,1]],device=device,dtype=torch.bool)
    actual,stats,health=bank(u,mask,details=True);expected,p,ids=bank_reference(bank,u,mask)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(stats['probability_sum'],p.sum(0),rtol=1e-9,atol=1e-10)
    if not torch.equal(stats['details']['ids'],ids) or not health:raise ValueError('Route oracle mismatch')
    params=(u,*bank.parameters())
    a=torch.autograd.grad(actual.square().sum()+stats['probability_sum'].square().sum(),params,retain_graph=True)
    b=torch.autograd.grad(expected.square().sum()+p.sum(0).square().sum(),params,allow_unused=True)
    for x,y in zip(a,b):torch.testing.assert_close(x,torch.zeros_like(x) if y is None else y,rtol=1e-9,atol=1e-10)
    # Execute the installed device's stable-sort/empty-gradient kernels, not
    # just a CPU unit test or an assumption about the library API contract.
    for dtype in (torch.float32,torch.float64):
        tied=SpecialistBank().to(device=device,dtype=dtype)
        with torch.no_grad():tied.router[2].weight.zero_();tied.router[2].bias.zero_()
        hidden=torch.randn(3,2,128,device=device,dtype=dtype)
        valid=torch.ones(2,3,device=device,dtype=torch.bool)
        correction,s,ok=tied(hidden,valid,details=True)
        if not ok or not torch.equal(s['details']['ids'],torch.tensor([[0,1]]*6,device=device)):
            raise ValueError('Installed stable expert-ID tie ordering failed')
        (correction.sum()+s['probability_sum'].square().sum()).backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in tied.parameters()):
            raise ValueError('Device empty-expert path lost parameter gradients')
        if any(p.grad.count_nonzero() for e in tied.experts[2:] for p in e.parameters()):
            raise ValueError('Device globally empty expert current gradient nonzero')
        with torch.no_grad():
            tied.router[2].bias[2]=1e-6
            _,s,_=tied(hidden,valid,details=True)
        if not torch.equal(s['details']['ids'],torch.tensor([[2,0]]*6,device=device)):
            raise ValueError('Installed near-tie route ordering failed')
    # Auxiliary gradient oracle with unequal counts, distinct per-rank usage,
    # shared router/context parameters and world-scaled unreduced input grads.
    global_bank=copy.deepcopy(bank)
    # Deliberately unequal expert populations, not a random negative control
    # that could accidentally yield uniform assignments on another device.
    with torch.no_grad():
        global_bank.router[0].weight.zero_();global_bank.router[0].bias.zero_()
        global_bank.router[0].weight[0,0]=1.;global_bank.router[0].weight[1,0]=-1.
        global_bank.router[0].weight[0,129]=.01;global_bank.router[0].weight[1,257]=.01
        global_bank.router[2].weight.zero_();global_bank.router[2].bias.zero_()
        global_bank.router[2].weight[0,0]=1.;global_bank.router[2].weight[1,0]=.5
        global_bank.router[2].weight[2,1]=1.;global_bank.router[2].weight[3,1]=.5
    x=torch.randn(5,4,128,device=device,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():x[:,:2,0]=2.;x[:,2:,0]=-2.
    valid=torch.ones(4,5,device=device,dtype=torch.bool);valid[:2,1:]=False
    _,pg,ig=bank_reference(global_bank,x,valid)
    cg=torch.bincount(ig.flatten(),minlength=4)
    direct=4*(pg.mean(0)*cg/(2*len(pg))).sum();direct.backward()
    gradients={n:p.grad.clone() for n,p in global_bank.named_parameters() if p.grad is not None}
    locals=[]
    for rank in range(2):
        local=copy.deepcopy(global_bank);local.zero_grad(set_to_none=True)
        xi=x.detach()[:,rank*2:rank*2+2].clone().requires_grad_()
        _,s,_=local(xi,valid[rank*2:rank*2+2]);locals.append((local,xi,s))
    sums=sum(s['probability_sum'].detach() for _,_,s in locals)
    counts=sum(s['requested'] for _,_,s in locals);tokens=sum(s['tokens'] for _,_,s in locals)
    naive=[]
    for local,xi,s in locals:
        proxy,value=from_global(s['probability_sum'][None],sums[None],counts[None],tokens[None],2)
        torch.testing.assert_close(value[0],direct.detach(),rtol=1e-9,atol=1e-10)
        naive.append(4*(s['probability_sum']/s['tokens']*s['requested']/(2*s['tokens'])).sum())
        proxy.backward()
    for name,expected in gradients.items():
        actual=sum(dict(m.named_parameters())[name].grad for m,_,_ in locals)/2
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    for rank,(_,xi,s) in enumerate(locals):
        torch.testing.assert_close(xi.grad/2,x.grad[:,rank*2:rank*2+2],rtol=1e-9,atol=1e-10)
    if abs(float((sum(naive)/2-direct).detach()))<=1e-8:raise ValueError('Unequal-count negative control inconclusive')
    return dict(passed=True,fp64=True,independent_particle_loop=True,gradients=True,exact_routes=True,
        unequal_count_proxy=True,shared_parameter_gradients=True,input_gradient_world_factor=True,
        naive_average_rejected=True,stable_ties=True,device_empty_gradients=True)
