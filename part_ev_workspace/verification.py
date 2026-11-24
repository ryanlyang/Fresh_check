"""Disposable target-device mechanism/math/DDP gates, with no accuracy claim."""
import copy
import torch
from torch import nn
from . import runtime as R
from .model import build_model, WorkspaceStage
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
            if n.endswith(('write.output.weight','class_read.attention.output.weight')): p.normal_(std=.02)


def dropout_off(base):
    for m in base.modules():
        if isinstance(m,nn.Dropout):m.p=0.
        if isinstance(getattr(m,'dropout',None),float):m.dropout=0.


def close(a,b,*,gradient=False,amp=False,ddp=False):
    if ddp: rtol=PARITY['ddp_gradient_rtol' if gradient else 'ddp_forward_rtol'];atol=PARITY['ddp_atol']
    elif amp: rtol=PARITY['amp_gradient_rtol' if gradient else 'amp_forward_rtol'];atol=PARITY['amp_gradient_atol' if gradient else 'amp_forward_atol']
    else:rtol=PARITY['gradient_rtol' if gradient else 'forward_rtol'];atol=PARITY['gradient_atol' if gradient else 'forward_atol']
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)


def reference_attention(module, query, source, valid=None):
    """Independent event/head loops; never call the production attention forward."""
    rows=[]
    for b in range(query.shape[0]):
        x=source[b] if valid is None else source[b,valid[b]]
        q=F.linear(query[b],module.query.weight,module.query.bias)
        k=F.linear(x,module.key.weight,module.key.bias)
        v=F.linear(x,module.value.weight,module.value.bias)
        heads=[]
        for h in range(8):
            region=slice(16*h,16*(h+1))
            heads.append((q[:,region]@k[:,region].T/4).softmax(-1)@v[:,region])
        rows.append(F.linear(torch.cat(heads,-1),module.output.weight,module.output.bias))
    return torch.stack(rows)


def reference_stage(stage,h,z,valid):
    def norm(module,x):
        return F.layer_norm(x,(128,),module.weight,module.bias,1e-5)
    safe=h.masked_fill(~valid[...,None],0)
    z1=z+reference_attention(stage.read,norm(stage.read_query,z),norm(stage.read_kv,safe),valid)
    x=norm(stage.mix_norm,z1)
    z2=z1+reference_attention(stage.mix,x,x)
    z3=z2+F.linear(F.gelu(F.linear(norm(stage.ffn_norm,z2),stage.ffn[0].weight,stage.ffn[0].bias)),
                    stage.ffn[2].weight,stage.ffn[2].bias)
    feedback=reference_attention(stage.write,norm(stage.write_query,safe),norm(stage.write_kv,z3))
    return h+feedback.masked_fill(~valid[...,None],0),z3


def stage_oracle(device):
    stage=WorkspaceStage().to(device).double()
    with torch.no_grad():stage.write.output.weight.normal_(std=.02)
    h=torch.randn(2,5,128,device=device,dtype=torch.float64,requires_grad=True)
    z=torch.randn(2,8,128,device=device,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[1,0,1,1,0],[0,1,0,0,0]],device=device,dtype=torch.bool)
    actual,workspace,ok=stage(h,z,mask)
    expected,zref=reference_stage(stage,h,z,mask)
    for a,b in ((actual,expected),(workspace,zref)):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    params=(h,z,*stage.parameters())
    ga=torch.autograd.grad(actual.square().mean()+workspace.square().mean(),params,retain_graph=True)
    ge=torch.autograd.grad(expected.square().mean()+zref.square().mean(),params)
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    perm=torch.tensor([4,1,3,0,2],device=device)
    hp,zp,_=stage(h[:,perm],z,mask[:,perm])
    torch.testing.assert_close(hp,actual[:,perm],rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(zp,workspace,rtol=1e-9,atol=1e-10)
    single,zsingle,_=stage(h[:1],z[:1],mask[:1])
    torch.testing.assert_close(single,actual[:1],rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(zsingle,workspace[:1],rtol=1e-9,atol=1e-10)
    junk=h.detach().clone();junk[~mask]=123456.
    hj,zj,_=stage(junk,z,mask)
    torch.testing.assert_close(hj[mask],actual[mask],rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(zj,workspace,rtol=1e-9,atol=1e-10)
    if not ok:raise ValueError('Stage oracle unhealthy')
    return dict(passed=True,precision='fp64',independent_stage=True,gradients=True,
                permutation=True,padding=True,event_isolation=True)


def oracle(device):
    a=Attention().to(device).double()
    q=torch.randn(2,3,128,device=device,dtype=torch.float64,requires_grad=True)
    x=torch.randn(2,5,128,device=device,dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([[1,0,1,1,0],[0,1,0,0,0]],device=device,dtype=torch.bool)
    actual,ok=a(q,x,valid);rows=[]
    for b in range(2):
        qs=a.query(q[b]);ks=a.key(x[b,valid[b]]);vs=a.value(x[b,valid[b]])
        parts=[]
        for h in range(8):
            region=slice(h*16,(h+1)*16)
            parts.append((qs[:,region]@ks[:,region].T/4).softmax(-1)@vs[:,region])
        rows.append(a.output(torch.cat(parts,-1)))
    expected=torch.stack(rows);torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    params=(q,x,*a.parameters())
    ga=torch.autograd.grad(actual.square().mean(),params,retain_graph=True)
    ge=torch.autograd.grad(expected.square().mean(),params)
    for p,v in zip(ga,ge):torch.testing.assert_close(p,v,rtol=1e-9,atol=1e-10)
    if not ok:raise ValueError('Attention oracle unhealthy')
    return dict(passed=True,precision='fp64',independent_head_loop=True,gradients=True,
                full_stage=stage_oracle(device))


def initial_identity(device,rank,world):
    control,_,_,_=R.fresh('control',device,world,trim=False)
    candidate,_,_,_=R.fresh('workspace',device,world,trim=False)
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
    return dict(passed=True,native_calls=count,native_pipeline=native,nonzero_state=True,rng_exact=True)


def active_parity(arm,device,rank,world,*,amp=False):
    fast,_,_,_=R.fresh(arm,device,world,trim=False);activate(fast)
    slow=copy.deepcopy(fast)
    if arm=='workspace':slow.mod.workspace_checkpointed=False
    values,_=R.tensors(R.split_batch(R.fixed_batch(8,11),rank,world),device)
    records=[]
    for base in (slow,fast):
        base.train();base.zero_grad(set_to_none=True);R.seed_everything(75+rank)
        with model_precision(base,device,training=amp):
            pred=base(*values);loss=pred.float().square().mean()
        loss.backward();records.append((pred.detach(),{k:p.grad for k,p in base.named_parameters()}))
    close(records[1][0],records[0][0],amp=amp)
    for k,p in records[1][1].items():
        if p is None or not torch.isfinite(p).all():raise ValueError('Missing/nonfinite active gradient: '+k)
        close(p,records[0][1][k],gradient=True,amp=amp)
    if fast._state_pair_cache.stem is not None:raise ValueError('Leaked native cache')
    return dict(passed=True,gradients=len(records[0][1]),amp=amp,checkpoint_backward=True)


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
    before={n:p.detach().clone() for n,p in base.named_parameters() if 'global_workspace' in n}
    values,y=R.tensors(R.split_batch(R.fixed_batch(8,9),rank,world),device)
    successes=skips=0
    for attempt in range(20):
        result=R.step(base,model,opt,scaler,values,y,attempt,world)
        skips+=result['skipped'];successes+=1-result['skipped']
        if successes==8:break
    if successes!=8:raise ValueError('Learning probe did not settle within twenty attempts')
    if arm=='workspace':
        names=dict(base.named_parameters())
        required=['mod.global_workspace.initial_tokens','mod.global_workspace.class_read.attention.output.weight']
        required += [f'mod.global_workspace.stages.{i}.{part}.weight' for i in range(4)
                     for part in ('read.query','mix.query','write.output','ffn.0')]
        if any(torch.equal(before[n],names[n]) for n in required):raise ValueError('Workspace path failed to learn')
    return base,model,opt,scaler,dict(passed=True,successful_updates=successes,attempts=attempt+1,skips=skips)


def rejection(arm,device,rank,world):
    for kind in ('empty_input','nonfinite'):
        base,model,opt,scaler=R.fresh(arm,device,world,trim=False)
        values,y=R.tensors(R.split_batch(R.fixed_batch(8,7),rank,world),device)
        if rank==0:
            if kind=='empty_input':values[-1][0]=False
            else:
                with torch.no_grad():
                    p=(base.mod.global_workspace.stages[0].read.query.weight if arm=='workspace'
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
