"""Small, isolated inspection: no activations/graphs cached on model modules."""
import copy
from contextlib import contextmanager
from pathlib import Path
import torch
from . import runtime as R
from .attention import Attention
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state
from part_ev_screen.precision import model_precision


def memory():
    result={}
    path=Path('/proc/self/status')
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



def required_scopes():
    keys={'multiscale.geometry','multiscale.canonical_atoms','multiscale.merge_history',
          'multiscale.maps','multiscale.descriptors','multiscale.pooling','multiscale.feedback','multiscale.fusion'}
    for i in range(4):
        keys.update(f'multiscale.stages.{i}.{p}' for p in ('particle_read','fine_local','child_read',
            'coarse_global','topdown','fine_ffn','coarse_ffn','fine_seed_linear','coarse_seed_linear'))
    return keys|{'multiscale.final_read.fine_read','multiscale.final_read.coarse_read'}


@contextmanager
def scopes(base,phase=None):
    handles=[];stacks={};counts=dict(forward={},recompute={})
    def before(name):
        def hook(*_):
            group=counts[(phase or {'value':'forward'})['value']]
            group[name]=group.get(name,0)+1
            s=torch.profiler.record_function(name);s.__enter__();stacks.setdefault(name,[]).append(s)
        return hook
    def after(name):
        def hook(*_):stacks[name].pop().__exit__(None,None,None)
        return hook
    if hasattr(base.mod,'multiscale'):
        for name,module in base.mod.multiscale.named_modules():
            key='multiscale.'+name
            if key in required_scopes():
                handles.extend([module.register_forward_pre_hook(before(key)),
                                module.register_forward_hook(after(key),always_call=True)])
    try:yield counts
    finally:
        for h in handles:h.remove()
        for rows in stacks.values():
            for s in reversed(rows):s.__exit__(None,None,None)


def profile(arm,device):
    base,_,_,_=R.fresh(arm,device,1,trim=False)
    from .verification import activate
    activate(base);values,_=R.tensors(R.fixed_batch(2,17),device);records=[]
    for training in (True,False):
        base.train(training);base.zero_grad(set_to_none=True)
        phase={'value':'forward'}
        activities=[torch.profiler.ProfilerActivity.CPU]
        if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with scopes(base,phase) as counts,torch.profiler.profile(activities=activities,record_shapes=False,
                profile_memory=False,with_stack=False,with_flops=True) as profiler:
            with torch.set_grad_enabled(training),model_precision(base,device,training=False):
                output=base(*values)
                if training:
                    phase['value']='recompute';output.square().mean().backward()
            R.synchronize(device)
        entries={e.key:dict(calls=e.count,cpu_us=e.cpu_time_total,device_us=getattr(e,'device_time_total',0.))
                 for e in profiler.key_averages() if e.key.startswith('multiscale.')}
        if arm=='multiscale' and (set(entries)!=required_scopes() or any(
                r['calls']<=0 or r['cpu_us']+r['device_us']<=0 for r in entries.values())):
            raise ValueError('Missing small-profile scopes')
        records.append(dict(training=training,events=2,particles=17,stages=entries,recompute_included=training,
            calls_by_phase=counts,
            profiler_visible_flops=sum(e.flops or 0 for e in profiler.key_averages()),
            flops_convention='2 FLOPs per MAC; profiler-visible operators only; geometry comparisons omitted, not a complete FLOP count'))
    return records


def sample(base,device):
    if not hasattr(base.mod,'multiscale'):return dict(events=8,particles=17,multiscale=False)
    from .geometry import observe
    values,_=R.tensors(R.fixed_batch(8,17),device)
    rng=capture_rank_state(base);flags=[(m,m.training) for m in base.modules()];trim=base.mod.trimmer.enabled
    params={n:p.detach().clone() for n,p in base.named_parameters() if 'mod.multiscale.' in n}
    layers=[];attention={};handles=[]
    def stage_hook(i):
        def hook(module,args,out):
            h,f,c,ok=out;g=args[1]
            layers.append(dict(stage=i,fine_rms=float(f.float()[g.fine_valid].square().mean().sqrt()),
                coarse_rms=float(c.float()[g.coarse_valid].square().mean().sqrt()),
                feedback_ratio=float((h-args[0]).float().norm()/args[0].float().norm().clamp_min(1e-12))))
        return hook
    def attention_hook(name):
        def hook(module,args,kwargs,out):
            q,x=args[:2]
            valid=args[2] if len(args)>2 else kwargs.get('valid')
            allowed=args[3] if len(args)>3 else kwargs.get('allowed')
            qv=args[4] if len(args)>4 else kwargs.get('query_valid')
            if valid is None:valid=torch.ones(x.shape[:2],device=device,dtype=torch.bool)
            if qv is None:qv=torch.ones(q.shape[:2],device=device,dtype=torch.bool)
            if allowed is None:allowed=qv[:,:,None]&valid[:,None,:]
            qs=module.query(q.masked_fill(~qv[...,None],0)).reshape(len(q),q.shape[1],8,16).transpose(1,2).float()
            ks=module.key(x.masked_fill(~valid[...,None],0)).reshape(len(x),x.shape[1],8,16).transpose(1,2).float()
            scores=(qs@ks.transpose(-1,-2)/4).masked_fill(~allowed[:,None],-torch.inf)
            scores=torch.where(qv[:,None,:,None],scores,torch.zeros_like(scores));w=scores.softmax(-1)
            entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)
            attention[name]=float(entropy.transpose(1,2)[qv].mean())
        return hook
    try:
        base.eval();base.mod.trimmer.enabled=False
        for i,s in enumerate(base.mod.multiscale.stages):handles.append(s.register_forward_hook(stage_hook(i)))
        for name,m in base.mod.multiscale.named_modules():
            if isinstance(m,Attention):handles.append(m.register_forward_hook(attention_hook(name),with_kwargs=True))
        with observe() as geometry,torch.no_grad(),model_precision(base,device,training=False):expected=base(*values)
        for h in handles:h.remove()
        handles=[];effects={}
        for mode in ('feedback','topdown','final_read'):
            with torch.no_grad():
                for n,p in base.named_parameters():
                    target='feedback_output.' in n if mode=='feedback' else (
                        'topdown.output.' in n if mode=='topdown' else 'final_read.fusion_output.' in n)
                    if 'mod.multiscale.' in n and target:p.zero_()
                with model_precision(base,device,training=False):changed=base(*values)
                effects[mode]=float((changed-expected).abs().max())
                for n,p in base.named_parameters():
                    if n in params:p.copy_(params[n])
        return dict(events=8,particles=17,multiscale=True,layers=layers,geometry=geometry,
                    attention_entropy=attention,disabled_logit_changes=effects)
    finally:
        for h in handles:h.remove()
        with torch.no_grad():
            for n,p in base.named_parameters():
                if n in params:p.copy_(params[n])
        for m,flag in flags:m.training=flag
        base.mod.trimmer.enabled=trim;restore_rank_state(base,rng)

