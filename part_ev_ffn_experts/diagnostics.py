"""Bounded non-mutating routing, gradient and operator-cost evidence."""
from pathlib import Path
import torch
from . import runtime as R
from .config import SCIENCE
from .losses import balancing
from .model import banks
from .precision import model_precision
from part_ev_fullscale.checkpoint import capture_rank_state,restore_rank_state

SCOPES={'experts.context_router','experts.stable_selection','experts.packing','experts.mlp','experts.combination'}


def memory():
    result={};path=Path('/proc/self/status')
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(('VmRSS:','VmHWM:')):
                key,value,*_=line.split();result[key.rstrip(':')+'_bytes']=int(value)*1024
    for p in ('/sys/fs/cgroup/memory.peak','/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
        if Path(p).exists():
            value=Path(p).read_text().strip()
            if value.isdecimal():result['cgroup_peak_bytes']=int(value)
            break
    return result


def profile(arm,device):
    from .verification import activate,loss
    base,_,_,_=R.fresh(arm,device,1,trim=False);activate(base)
    x,y=R.tensors(R.fixed_batch(2,17),device);records=[]
    for training in (True,False):
        base.train(training);base.zero_grad(set_to_none=True)
        activities=[torch.profiler.ProfilerActivity.CPU]
        if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=False,
                profile_memory=False,with_stack=False,with_flops=True) as profiler:
            with torch.set_grad_enabled(training):
                if training:total,_=loss(base,base,x,y,1);total.backward()
                else:
                    with model_precision(base,device,training=False):base(*x)
            R.synchronize(device)
        stages={}
        for e in profiler.key_averages():
            if e.key.startswith('experts.'):
                stages[e.key]=dict(calls=e.count,cpu_us=e.cpu_time_total,device_us=getattr(e,'device_time_total',0.))
        if arm=='experts' and (set(stages)!=SCOPES or any(
                s['calls']<=0 or s['cpu_us']+s['device_us']<=0 for s in stages.values())):
            raise ValueError('Experts small-profile scope coverage incomplete')
        records.append(dict(training=training,events=2,particles=17,stages=stages,
            precision='fp32',profiler_visible_flops=sum(e.flops or 0 for e in profiler.key_averages()),
            flops_convention='2 FLOPs per MAC; visible operators only, not complete model or training FLOPs'))
    return records


def sample(base,device):
    if base.experts_architecture=='control':return dict(events=8,particles=17,experts=False)
    x,y=R.tensors(R.fixed_batch(8,17),device)
    rng=capture_rank_state(base);flags=[(m,m.training) for m in base.modules()]
    trim=base.mod.trimmer.enabled;handles=[];dense={};rows=[]
    output_parameters={n:p.detach().clone() for n,p in base.named_parameters()
        if '.specialists.experts.' in n and '.output.' in n}
    def dense_hook(i):
        def hook(m,a,out):dense[i]=out.detach()
        return hook
    def bank_hook(i):
        def hook(m,a,out):
            correction,s,ok=out;t=s['tokens'].to(torch.float32)
            rows.append(dict(block=i,requested=s['requested'].cpu().tolist(),executed=s['executed'].cpu().tolist(),
                tokens=int(t),probability_means=(s['probability_sum']/t).detach().cpu().tolist(),
                primary_usage=(s['primary']/t).cpu().tolist(),secondary_usage=(s['secondary']/t).cpu().tolist(),
                full_entropy=float(s['full_entropy_sum']/t),selected_entropy=float(s['selected_entropy_sum']/t),
                minimum_margin=float(s['min_margin']),
                correction_dense_rms_ratio=float(correction.float().square().mean().sqrt()/dense[i].float().square().mean().sqrt().clamp_min(1e-12)),
                normalized_input_rms=float(a[0].float().square().mean().sqrt()),healthy=bool(ok)))
        return hook
    try:
        base.eval();base.mod.trimmer.enabled=False
        for i in SCIENCE['blocks']:
            block=base.mod.blocks[i]
            handles.extend([block.fc2.register_forward_hook(dense_hook(i)),block.specialists.register_forward_hook(bank_hook(i))])
        with torch.no_grad(),model_precision(base,device,training=False):expected=base(*x)
        for h in handles:h.remove()
        handles=[]
        with torch.no_grad():
            for n,p in base.named_parameters():
                if n in output_parameters:p.zero_()
            with model_precision(base,device,training=False):disabled=base(*x)
            delta=float((disabled-expected).abs().max())
            for n,p in base.named_parameters():
                if n in output_parameters:p.copy_(output_parameters[n])
        with model_precision(base,device,training=False):out=base(*x,return_router_stats=True)
        ce=torch.nn.functional.cross_entropy(out['logits'],y)
        aux,global_stats=balancing(out['banks'],1,device)
        selected=[(n,p) for n,p in base.named_parameters() if '.specialists.' in n]
        cg=torch.autograd.grad(ce,[p for n,p in selected],retain_graph=True,allow_unused=True)
        ag=torch.autograd.grad(aux,[p for n,p in selected],allow_unused=True)
        gradients={}
        for group in ('router','experts'):
            index=[j for j,(n,p) in enumerate(selected) if ('.router.' in n or '.context_norm.' in n)==(group=='router')]
            gradients[group]={}
            for key,values in [('ce',cg),('auxiliary',ag)]:
                norms=[values[j].float().norm() for j in index if values[j] is not None]
                gradients[group][key]=float(torch.stack(norms).norm()) if norms else 0.
        return dict(events=8,particles=17,experts=True,banks=rows,gradient_norms=gradients,
            ce=float(ce.detach()),balance=float(global_stats['balance']),alpha=SCIENCE['alpha'],
            disabled_experts_logit_delta=delta)
    finally:
        for h in handles:h.remove()
        with torch.no_grad():
            for n,p in base.named_parameters():
                if n in output_parameters:p.copy_(output_parameters[n])
        for m,flag in flags:m.training=flag
        base.mod.trimmer.enabled=trim;restore_rank_state(base,rng)
