"""Target-device native, expert, actual-DDP and empty-population proofs."""
import copy
import torch
from torch import nn
from . import runtime as R
from .model import build_model,banks,SpecialistBank
from .losses import balancing
from .precision import model_precision
from .config import SCIENCE,PARITY
from .restart import compare
from .oracles import check as oracle
from part_ev_state_compare.model import state_branches
from part_ev_fullscale.checkpoint import capture_rank_state,restore_rank_state
from part_ev_fullscale.optimizer import Ranger


def activate(base,*,separated=False):
    with torch.no_grad():
        for b in state_branches(base):b.output.normal_(std=.02)
        for bank in banks(base):
            for e in bank.experts:e.output.weight.normal_(std=.02)
            if separated:bank.router[2].bias.copy_(torch.tensor([2.,1.,-1.,-2.],device=bank.router[2].bias.device))


def dropout_off(base):
    for m in base.modules():
        if isinstance(m,nn.Dropout):m.p=0.
        if isinstance(getattr(m,'dropout',None),float):m.dropout=0.


def close(a,b,*,gradient=False,amp=False,ddp=False):
    if ddp:rtol=PARITY['ddp_gradient_rtol' if gradient else 'ddp_forward_rtol'];atol=PARITY['ddp_atol']
    elif amp:rtol=PARITY['amp_gradient_rtol' if gradient else 'amp_forward_rtol'];atol=PARITY['amp_gradient_atol' if gradient else 'amp_forward_atol']
    else:rtol=PARITY['gradient_rtol' if gradient else 'forward_rtol'];atol=PARITY['gradient_atol' if gradient else 'forward_atol']
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)


def loss(base,model,values,y,world,*,amp=False,details=False):
    with model_precision(base,y.device,training=amp):
        out=model(*values,return_router_stats=True,return_route_details=details)
        ce=torch.nn.functional.cross_entropy(out['logits'],y)
    if base.experts_architecture=='experts':
        auxiliary,_=balancing(out['banks'],world,y.device);ce=ce+SCIENCE['alpha']*auxiliary
    return ce,out


def initial_identity(device,rank,world):
    control,_,_,_=R.fresh('control',device,world,trim=True)
    candidate,_,_,_=R.fresh('experts',device,world,trim=True)
    for m in (control,candidate):m.mod.trimmer._counter=5
    with torch.no_grad():
        for branch in state_branches(control):branch.output.normal_(std=.02)
        for block in control.mod.blocks:block.c_attn.uniform_(.7,1.3);block.w_resid.uniform_(.7,1.3)
    state=candidate.state_dict();state.update(control.state_dict());candidate.load_state_dict(state)
    batch=R.fixed_batch(8,13)
    batch[0]['pf_mask'][::2,:,1]=False
    batch[0]['pf_mask'][1::2,:,-4:]=False
    values,y=R.tensors(R.split_batch(batch,rank,world),device)
    embedded=[]
    h1=control.mod.embed.register_forward_pre_hook(lambda m,a:embedded.append(a[0].detach().clone()))
    h2=candidate.mod.embed.register_forward_pre_hook(lambda m,a:embedded.append(a[0].detach().clone()))
    modules=dict(trimmer=candidate.mod.trimmer,embedding=candidate.mod.embed,pairs=candidate.mod.pair_embed,
                 classifier=candidate.mod.fc)
    modules.update({f'attention_{i}':b.attn for i,b in enumerate(candidate.mod.blocks)})
    modules.update({f'fc1_{i}':b.fc1 for i,b in enumerate(candidate.mod.blocks)})
    modules.update({f'fc2_{i}':b.fc2 for i,b in enumerate(candidate.mod.blocks)})
    modules.update({f'class_{i}':b.attn for i,b in enumerate(candidate.mod.cls_blocks)})
    modules.update({f'bank_{i}':b for i,b in enumerate(banks(candidate))})
    counts={};handles=[m.register_forward_hook(lambda m,a,o,n=n:counts.__setitem__(n,counts.get(n,0)+1)) for n,m in modules.items()]
    R.seed_everything(601+rank);rng=capture_rank_state(candidate)
    with torch.no_grad(),model_precision(control,device,training=False):expected=control(*values)
    after=capture_rank_state(control);restore_rank_state(candidate,rng)
    with torch.no_grad(),model_precision(candidate,device,training=False):actual=candidate(*values)
    h1.remove();h2.remove()
    if len(embedded)!=2 or not torch.equal(embedded[0],embedded[1]):raise ValueError('Native trimmed inputs changed')
    for h in handles:h.remove()
    close(actual,expected);compare(capture_rank_state(candidate),after,exact=True)
    if counts!={n:1 for n in modules}:raise ValueError('Native pipeline repeated or missing')
    for base in (control,candidate):
        base.zero_grad(set_to_none=True);R.seed_everything(73+rank)
        with model_precision(base,device,training=False):torch.nn.functional.cross_entropy(base(*values),y).backward()
    for n,p in control.named_parameters():close(p.grad,dict(candidate.named_parameters())[n].grad,gradient=True)
    return dict(passed=True,native_calls=counts,rng_exact=True,trimmed_inputs_exact=True,
        nonzero_state=True,shared_ce_gradient=True)


def active_parity(arm,device,rank,world,*,amp=False):
    fast,_,_,_=R.fresh(arm,device,world,trim=False);activate(fast,separated=True)
    slow=copy.deepcopy(fast)
    for bank in banks(slow):bank.checkpointed=False
    values,y=R.tensors(R.split_batch(R.fixed_batch(8,11),rank,world),device)
    records=[]
    for base in (slow,fast):
        base.train();base.zero_grad(set_to_none=True);R.seed_everything(81+rank)
        inputs=[v.detach().clone().requires_grad_(i==1) if v.is_floating_point() else v.clone() for i,v in enumerate(values)]
        total,out=loss(base,base,inputs,y,world,amp=amp,details=True);total.backward()
        records.append((out,{n:p.grad for n,p in base.named_parameters()},inputs[1].grad))
    close(records[0][0]['logits'],records[1][0]['logits'],amp=amp)
    for a,b in zip(records[0][0]['banks'],records[1][0]['banks']):
        if not torch.equal(a['details']['ids'],b['details']['ids']):raise ValueError('Checkpoint rerouted tokens')
    for name,p in records[1][1].items():
        if p is None or not torch.isfinite(p).all():raise ValueError('Missing/nonfinite active gradient: '+name)
        close(p,records[0][1][name],gradient=True,amp=amp)
    close(records[0][2],records[1][2],gradient=True,amp=amp)
    if fast._state_pair_cache.stem is not None:raise ValueError('Leaked pair cache')
    return dict(passed=True,gradients=len(records[0][1]),input_gradients=True,amp=amp,
                checkpoint_backward=True,exact_routes=True)


def distributed(arm,device,rank,world):
    base,model,opt,scaler=R.fresh(arm,device,world,trim=False);activate(base,separated=True);dropout_off(base)
    reference=build_model(arm,trim=False).to(device);reference.load_state_dict(base.state_dict());dropout_off(reference)
    reference.train();batch=R.fixed_batch(8,13);batch[0]['pf_mask'][:4,:,1:]=False
    values,y=R.tensors(batch,device);values[1]=values[1].detach().clone().requires_grad_()
    total,expected=loss(reference,reference,values,y,1);total.backward()
    local,truth=R.tensors(R.split_batch(batch,rank,world),device);local[1]=local[1].detach().clone().requires_grad_()
    total,actual=loss(base,model,local,truth,world);total.backward()
    close(actual['logits'],expected['logits'][rank*8//world:(rank+1)*8//world],ddp=True)
    for n,p in base.named_parameters():
        if p.grad is None:raise ValueError('DDP missing gradient: '+n)
        close(p.grad,dict(reference.named_parameters())[n].grad,gradient=True,ddp=True)
    close(local[1].grad/world,values[1].grad[rank*8//world:(rank+1)*8//world],gradient=True,ddp=True)
    for n in (1,1 if rank==0 else 9):
        x,y=R.tensors(R.split_batch(R.fixed_batch(8,n),rank,world),device)
        R.step(base,model,opt,scaler,x,y,0,world,amp=False)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in base.parameters()):raise ValueError('Degenerate DDP graph')
    return dict(passed=True,world_size=world,global_events=8,fp32=True,dropout=False,
        one_particle=True,unequal_lengths=True,unequal_tokens=True,input_gradient_world_factor=True,find_unused_parameters=False)


def configure_coverage(bank):
    with torch.no_grad():
        bank.router[0].weight.zero_();bank.router[0].bias.zero_()
        bank.router[0].weight[0,0]=1.;bank.router[0].weight[1,0]=-1.
        bank.router[0].weight[0,129]=.01;bank.router[0].weight[1,257]=.01
        bank.router[2].weight.zero_();bank.router[2].bias.zero_()
        bank.router[2].weight[0,0]=1.;bank.router[2].weight[1,0]=.5
        bank.router[2].weight[2,1]=1.;bank.router[2].weight[3,1]=.5


class BankFixture(nn.Module):
    """Diagnostic-only upstream layer, one bank and a ten-logit head."""
    def __init__(self):
        super().__init__();self.upstream=nn.Linear(128,128);self.bank=SpecialistBank();self.head=nn.Linear(128,10)
        with torch.no_grad():self.upstream.weight.copy_(torch.eye(128));self.upstream.bias.zero_()
        configure_coverage(self.bank)
        with torch.no_grad():
            for e in self.bank.experts:e.output.weight.normal_(std=.02)
    def forward(self,x,valid):
        u=self.upstream(x);correction,stats,health=self.bank(u,valid)
        pooled=(u+correction).transpose(0,1).masked_fill(~valid[...,None],0).sum(1)/valid.sum(1)[:,None]
        return self.head(pooled),stats,health


def bank_ddp(device,rank,world):
    R.seed_everything(991);base=BankFixture().to(device)
    model=(torch.nn.parallel.DistributedDataParallel(base,device_ids=[device.index],
        find_unused_parameters=False,gradient_as_bucket_view=True) if world==2 and device.type=='cuda' else
        torch.nn.parallel.DistributedDataParallel(base,find_unused_parameters=False) if world==2 else base)
    optimizer=Ranger(base.parameters());records=[]
    for phase in ('empty_local','empty_global'):
        reference=copy.deepcopy(base);reference.zero_grad(set_to_none=True)
        g=torch.Generator().manual_seed(71)
        x=torch.randn(7,4,128,generator=g).to(device);x[:,:2,0]=2.;x[:,2:,0]=-2. if phase=='empty_local' else 2.
        valid=torch.ones(4,7,dtype=torch.bool,device=device);valid[:2,1:]=False
        y=torch.arange(4,device=device)
        global_x=x.detach().clone().requires_grad_();expected,stat,_=reference(global_x,valid)
        aux,_=balancing([stat]*6,1,device)
        (torch.nn.functional.cross_entropy(expected,y)+.001*aux).backward()
        local_x=x[:,rank*4//world:(rank+1)*4//world].detach().clone().requires_grad_()
        local_valid=valid[rank*4//world:(rank+1)*4//world]
        optimizer.zero_grad(set_to_none=True)
        output,s,ok=model(local_x,local_valid)
        aux,global_stats=balancing([s]*6,world,device)
        (torch.nn.functional.cross_entropy(output,y[rank*4//world:(rank+1)*4//world])+.001*aux).backward()
        if not ok:raise ValueError('Bank fixture unhealthy')
        for name,p in base.named_parameters():
            if p.grad is None:raise ValueError('Empty expert omitted DDP gradient')
            close(p.grad,dict(reference.named_parameters())[name].grad,gradient=True,ddp=True)
        close(local_x.grad/world,global_x.grad[:,rank*4//world:(rank+1)*4//world],gradient=True,ddp=True)
        if phase=='empty_global' and any(p.grad.count_nonzero() for e in base.bank.experts[2:] for p in e.parameters()):
            raise ValueError('Globally empty expert has nonzero current gradient')
        optimizer.step()
        if len(set(R.gather(R.parameter_digest(base),world)))!=1:raise ValueError('Reduced expert parameters diverged')
        records.append(dict(phase=phase,global_tokens=int(global_stats['tokens'][0]),
            counts=global_stats['requested'][0].cpu().tolist(),zero_drops=True))
    return dict(passed=True,world_size=world,actual_data_parallel=world==2,unequal_tokens=True,
        local_empty=True,global_empty=True,all_gradients=True,shared_parameter_gradients=True,
        input_gradient_world_factor=True,phases=records)


def learning(arm,device,rank,world):
    base,model,opt,scaler=R.fresh(arm,device,world,trim=False)
    before={n:p.detach().clone() for n,p in base.named_parameters() if '.specialists.' in n}
    values,y=R.tensors(R.split_batch(R.fixed_batch(8,9),rank,world),device)
    successes=skips=0
    for attempt in range(20):
        result=R.step(base,model,opt,scaler,values,y,attempt,world)
        skips+=result['skipped'];successes+=1-result['skipped']
        if successes==8:break
    if successes!=8:raise ValueError('Learning probe did not settle within twenty attempts')
    if arm=='experts':
        for i,bank in zip(SCIENCE['blocks'],banks(base)):
            if all(torch.equal(before[f'mod.blocks.{i}.specialists.experts.{e}.output.weight'],b.output.weight)
                   for e,b in enumerate(bank.experts)):raise ValueError('No selected expert output learned')
    return base,model,opt,scaler,dict(passed=True,successful_updates=8,attempts=attempt+1,skips=skips,
        alpha=.001 if arm=='experts' else 0.,auxiliary_included=arm=='experts')


def coverage(device):
    records=[]
    for block in SCIENCE['blocks']:
        R.seed_everything(912+block);bank=SpecialistBank(block=block).to(device);configure_coverage(bank)
        optimizer=Ranger(bank.parameters())
        u=torch.randn(5,4,128,device=device);u[:,:2,0]=2.;u[:,2:,0]=-2.
        valid=torch.ones(4,5,dtype=torch.bool,device=device);valid[2:,1:]=False
        target=torch.arange(4,device=device)
        head=torch.randn(128,10,device=device)/12
        before={n:p.detach().clone() for n,p in bank.named_parameters()}
        norms=None
        for step in range(8):
            optimizer.zero_grad(set_to_none=True);out,s,_=bank(u,valid)
            pooled=(u+out).transpose(0,1).masked_fill(~valid[...,None],0).sum(1)/valid.sum(1)[:,None]
            ce=torch.nn.functional.cross_entropy(pooled@head,target)
            aux,_=balancing([s]*6,1,device)
            if step==7:
                router=[p for n,p in bank.named_parameters() if n.startswith(('router.','context_norm.'))]
                c=torch.autograd.grad(ce,router,retain_graph=True);a=torch.autograd.grad(aux,router,retain_graph=True)
                norms=dict(ce=float(torch.stack([g.norm() for g in c]).norm()),auxiliary=float(torch.stack([g.norm() for g in a]).norm()))
            ce.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in bank.parameters()):raise ValueError('Coverage gradient missing')
            optimizer.step()
        for e,expert in enumerate(bank.experts):
            for part in ('input.weight','output.weight','norm.weight'):
                name=f'experts.{e}.{part}'
                if torch.equal(before[name],dict(bank.named_parameters())[name]):raise ValueError('Expert failed CE coverage: '+name)
        if not norms or min(norms.values())<=0:raise ValueError('CE/auxiliary router gradient proof missing')
        records.append(dict(block=block,all_four_learned=True,successful_updates=8,gradient_norms=norms))
    return dict(passed=True,fixture='controlled_postnorm_two_sign_coverage_ce_only_v1',banks=records)


def rejection(arm,device,rank,world):
    kinds=['empty_input','nonfinite']+(['bad_stats'] if arm=='experts' else [])
    for kind in kinds:
        base,model,opt,scaler=R.fresh(arm,device,world,trim=False)
        x,y=R.tensors(R.split_batch(R.fixed_batch(8,7),rank,world),device)
        if rank==0:
            if kind=='empty_input':x[-1][0]=False
            elif kind=='nonfinite':
                with torch.no_grad():
                    p=base.mod.blocks[2].specialists.router[0].weight if arm=='experts' else base.mod.fc[0].weight
                    p[0,0]=float('inf')
            else:
                def corrupt(m,a,o):o[1]['requested'][0]+=1
                base.mod.blocks[2].specialists.register_forward_hook(corrupt)
        rejected=False
        try:R.step(base,model,opt,scaler,x,y,0,world)
        except (FloatingPointError,ValueError):rejected=True
        if not R.agree(rejected,device,world):raise ValueError('Failure not coordinated')
    rejected=False
    try:R.all_call(lambda:(_ for _ in ()).throw(OSError('fixture')) if rank==0 else None,world)
    except RuntimeError:rejected=True
    if not R.agree(rejected,device,world):raise ValueError('Rank-local I/O not propagated')
    previous=R.STOP
    try:
        R.STOP=rank==0
        if not R.stopping(device,world):raise ValueError('Stop not coordinated')
    finally:R.STOP=previous
    return dict(passed=True,empty_input=True,nonfinite=True,bad_stats=arm=='experts',rank_local_io=True,stop=True)
