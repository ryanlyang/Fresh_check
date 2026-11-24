"""Disposable target-device mechanism/math/DDP gates, with no accuracy claim."""
import copy
import torch
from torch import nn
from . import runtime as R
from .model import build_model, MultiscaleStage
from .attention import Attention
from .config import PARITY
from part_ev_state_compare.model import state_branches
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state
from part_ev_screen.precision import model_precision
from .restart import compare
from torch.nn import functional as F


def activate(base):
    with torch.no_grad():
        for branch in state_branches(base): branch.output.normal_(std=.02)
        for n,p in base.named_parameters():
            if n.endswith(('feedback_output.weight','final_read.fusion_output.weight')): p.normal_(std=.02)


def dropout_off(base):
    for m in base.modules():
        if isinstance(m,nn.Dropout):m.p=0.
        if isinstance(getattr(m,'dropout',None),float):m.dropout=0.


def close(a,b,*,gradient=False,amp=False,ddp=False):
    if ddp: rtol=PARITY['ddp_gradient_rtol' if gradient else 'ddp_forward_rtol'];atol=PARITY['ddp_atol']
    elif amp: rtol=PARITY['amp_gradient_rtol' if gradient else 'amp_forward_rtol'];atol=PARITY['amp_gradient_atol' if gradient else 'amp_forward_atol']
    else:rtol=PARITY['gradient_rtol' if gradient else 'forward_rtol'];atol=PARITY['gradient_atol' if gradient else 'forward_atol']
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)


from .verification_math import oracle


def initial_identity(device,rank,world):
    control,_,_,_=R.fresh('control',device,world,trim=False)
    candidate,_,_,_=R.fresh('multiscale',device,world,trim=False)
    with torch.no_grad():
        for s in state_branches(control):s.output.normal_(std=.02)
    state=candidate.state_dict();state.update(control.state_dict());candidate.load_state_dict(state,strict=True)
    values,_=R.tensors(R.split_batch(R.fixed_batch(8,9),rank,world),device)
    count={};native={};handles=[]
    modules={'trimmer':candidate.mod.trimmer,'embedding':candidate.mod.embed,
             'pairs':candidate.mod.pair_embed,'classifier':candidate.mod.fc}
    modules.update({f'class_attention_{i}':b.attn for i,b in enumerate(candidate.mod.cls_blocks)})
    for name,module in modules.items():
        handles.append(module.register_forward_hook(
            lambda m,a,o,name=name:native.__setitem__(name,native.get(name,0)+1)))
    for i,block in enumerate(candidate.mod.blocks):
        handles.append(block.attn.register_forward_hook(lambda m,a,o,i=i:count.__setitem__(i,count.get(i,0)+1)))
    for base in (control,candidate):base.train()
    R.seed_everything(601+rank);rng=capture_rank_state(candidate)
    with torch.no_grad(),model_precision(control,device,training=False):expected=control(*values)
    after=capture_rank_state(control);restore_rank_state(candidate,rng)
    with torch.no_grad(),model_precision(candidate,device,training=False):actual=candidate(*values)
    close(actual,expected);compare(capture_rank_state(candidate),after,exact=True)
    for handle in handles:handle.remove()
    if count!={i:1 for i in range(8)} or native!={n:1 for n in modules}:
        raise ValueError('Native preprocessing/attention repeated/missing')
    gradients=[]
    for base in (control,candidate):
        base.zero_grad(set_to_none=True);R.seed_everything(894+rank)
        inputs=list(values);inputs[1]=values[1].detach().clone().requires_grad_(True)
        with model_precision(base,device,training=False):
            pred=base(*inputs);F.cross_entropy(pred,torch.arange(len(pred),device=device)%10).backward()
        gradients.append((inputs[1].grad,{n:p.grad for n,p in base.named_parameters()}))
    close(gradients[0][0],gradients[1][0],gradient=True)
    for n,p in gradients[0][1].items():close(p,gradients[1][1][n],gradient=True)
    return dict(passed=True,native_calls=count,native_pipeline=native,nonzero_state=True,rng_exact=True,ce_gradients=True)


def active_parity(arm,device,rank,world,*,amp=False):
    fast,_,_,_=R.fresh(arm,device,world,trim=False);activate(fast)
    slow=copy.deepcopy(fast)
    if arm=='multiscale':slow.mod.multiscale_checkpointed=False
    values,_=R.tensors(R.split_batch(R.fixed_batch(8,21),rank,world),device)
    records=[]
    for base in (slow,fast):
        base.train();base.zero_grad(set_to_none=True);R.seed_everything(75+rank)
        inputs=list(values);inputs[1]=values[1].detach().clone().requires_grad_(True)
        with model_precision(base,device,training=amp):
            pred=base(*inputs);loss=pred.float().square().mean()
        loss.backward();records.append((pred.detach(),{k:p.grad for k,p in base.named_parameters()},inputs[1].grad))
    close(records[1][0],records[0][0],amp=amp)
    for k,p in records[1][1].items():
        if p is None or not torch.isfinite(p).all():raise ValueError('Missing/nonfinite active gradient: '+k)
        close(p,records[0][1][k],gradient=True,amp=amp)
    if fast._state_pair_cache.stem is not None:raise ValueError('Leaked native cache')
    close(records[1][2],records[0][2],gradient=True,amp=amp)
    return dict(passed=True,gradients=len(records[0][1]),amp=amp,checkpoint_backward=True,input_gradients=True)


def distributed(arm,device,rank,world):
    base,model,opt,scaler=R.fresh(arm,device,world,trim=False);activate(base);dropout_off(base)
    R.seed_everything(101);reference=build_model(arm,trim=False).to(device)
    reference.load_state_dict(base.state_dict());dropout_off(reference);reference.train()
    batch=R.fixed_batch(8,13);values,y=R.tensors(batch,device)
    with model_precision(reference,device,training=False):
        expected=reference(*values);torch.nn.functional.cross_entropy(expected,y).backward()
    values,y=R.tensors(R.split_batch(batch,rank,world),device);base.train()
    with model_precision(base,device,training=False):
        actual=model(*values);torch.nn.functional.cross_entropy(actual,y).backward()
    close(actual,expected[rank*8//world:(rank+1)*8//world],ddp=True)
    for n,p in base.named_parameters():close(p.grad,dict(reference.named_parameters())[n].grad,gradient=True,ddp=True)
    # Same graph on N=1 and unequal rank lengths. No altered model/loss recipe.
    for label,n in [('one_particle',1),('unequal_lengths',1 if rank==0 else 9)]:
        values,y=R.tensors(R.split_batch(R.fixed_batch(8,n),rank,world),device)
        R.step(base,model,opt,scaler,values,y,0,world,amp=False)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in base.parameters()):
            raise ValueError('Degenerate graph missing gradients')
    return dict(passed=True,world_size=world,global_events=8,fp32=True,dropout=False,
                one_particle=True,unequal_lengths=True,find_unused_parameters=False)


def learning(arm,device,rank,world):
    base,model,opt,scaler=R.fresh(arm,device,world,trim=False)
    before={n:p.detach().clone() for n,p in base.named_parameters() if 'multiscale' in n}
    values,y=R.tensors(R.split_batch(R.fixed_batch(8,37),rank,world),device)
    successes=skips=0
    for attempt in range(20):
        result=R.step(base,model,opt,scaler,values,y,attempt,world)
        skips+=result['skipped'];successes+=1-result['skipped']
        if successes==8:break
    if successes!=8:raise ValueError('Learning probe did not settle within twenty attempts')
    if arm=='multiscale':
        names=dict(base.named_parameters())
        required=['mod.multiscale.final_read.fusion_output.weight','mod.multiscale.final_read.fine_read.query.weight','mod.multiscale.final_read.coarse_read.query.weight']
        required += [f'mod.multiscale.stages.{i}.{part}.weight' for i in range(4)
                     for part in ('particle_read.query','fine_local.query','child_read.query','coarse_global.query','topdown.query','feedback_output','fine_ffn.0','coarse_ffn.0')]
        if any(torch.equal(before[n],names[n]) for n in required):raise ValueError('Multiscale path failed to learn')
    return base,model,opt,scaler,dict(passed=True,successful_updates=successes,attempts=attempt+1,skips=skips,
        nondegenerate_particles=37,paths_moved=len(required) if arm=='multiscale' else 0)


def rejection(arm,device,rank,world):
    for kind in ('empty_input','nonfinite','overflow_geometry'):
        base,model,opt,scaler=R.fresh(arm,device,world,trim=False)
        values,y=R.tensors(R.split_batch(R.fixed_batch(8,7),rank,world),device)
        if rank==0:
            if kind=='empty_input':values[-1][0]=False
            elif kind=='overflow_geometry':
                values[2].fill_(1e38)
            else:
                with torch.no_grad():
                    p=(base.mod.multiscale.stages[0].particle_read.query.weight if arm=='multiscale'
                       else base.mod.fc[0].weight)
                    p[0,0]=float('inf')
        rejected=False
        try:R.step(base,model,opt,scaler,values,y,0,world)
        except (ValueError,FloatingPointError):rejected=True
        if not R.agree(rejected,device,world):raise ValueError('Failure was not coherently rejected')
    rejected=False
    try:R.all_call(lambda:(_ for _ in ()).throw(OSError('fixture')) if rank==0 else None,world)
    except RuntimeError:rejected=True
    if not R.agree(rejected,device,world):raise ValueError('Rank-local I/O not propagated')
    previous=R.STOP
    try:
        R.STOP=rank==0
        if not R.stopping(device,world):raise ValueError('Stop not coordinated')
    finally:R.STOP=previous
    return dict(passed=True,empty_input=True,nonfinite=True,rank_local_io=True,stop=True)

