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


@contextmanager
def scopes(base):
    """Bound record_function scopes including checkpoint recomputation."""
    handles=[];stacks={}
    def before(name):
        def hook(*_):
            scope=torch.profiler.record_function(name);scope.__enter__();stacks.setdefault(name,[]).append(scope)
        return hook
    def after(name):
        def hook(*_):stacks[name].pop().__exit__(None,None,None)
        return hook
    if hasattr(base.mod,'global_workspace'):
        for name,module in base.mod.global_workspace.named_modules():
            if isinstance(module,Attention) or name.endswith('.ffn'):
                key='workspace.'+name
                handles.extend([module.register_forward_pre_hook(before(key)),
                                module.register_forward_hook(after(key),always_call=True)])
    try:yield
    finally:
        for h in handles:h.remove()
        for rows in stacks.values():
            for s in reversed(rows):s.__exit__(None,None,None)


def profile(arm,device):
    base,_,_,_=R.fresh(arm,device,1,trim=False)
    from .verification import activate
    activate(base)
    values,_=R.tensors(R.fixed_batch(2,17),device)
    records=[]
    for training in (True,False):
        base.train(training);base.zero_grad(set_to_none=True)
        activities=[torch.profiler.ProfilerActivity.CPU]
        if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with scopes(base),torch.profiler.profile(activities=activities,record_shapes=False,
                profile_memory=False,with_stack=False,with_flops=True) as profiler:
            with torch.set_grad_enabled(training),model_precision(base,device,training=False):
                output=base(*values)
                if training:output.square().mean().backward()
            R.synchronize(device)
        entries={}
        for event in profiler.key_averages():
            if event.key.startswith('workspace.'):
                entries[event.key]=dict(calls=event.count,cpu_us=event.cpu_time_total,
                    device_us=getattr(event,'device_time_total',0.))
        if arm=='workspace':
            required={f'workspace.stages.{i}.{part}' for i in range(4) for part in ('read','mix','write','ffn')}
            required.add('workspace.class_read.attention')
            if set(entries)!=required or any(r['calls']<=0 or r['cpu_us']+r['device_us']<=0 for r in entries.values()):
                raise ValueError('Missing small-profile scopes')
        records.append(dict(training=training,events=2,particles=17,stages=entries,
            profiler_visible_flops=sum(e.flops or 0 for e in profiler.key_averages()),
            flops_convention='2 FLOPs per MAC; profiler-visible operators only, not a complete model/training FLOP count'))
    return records


def sample(base,device):
    if not hasattr(base.mod,'global_workspace'): return dict(events=8,particles=17,workspace=False)
    values,_=R.tensors(R.fixed_batch(8,17),device)
    rng=capture_rank_state(base);flags=[(m,m.training) for m in base.modules()]
    trim=base.mod.trimmer.enabled;params={n:p.detach().clone() for n,p in base.named_parameters() if 'global_workspace' in n}
    layers=[];attention={};handles=[]
    def stage_hook(i):
        def hook(module,args,out):
            h,z,ok=out
            unit=torch.nn.functional.normalize(z.float(),dim=-1)
            pairs=unit@unit.transpose(-1,-2)
            off=~torch.eye(8,dtype=torch.bool,device=device)
            layers.append(dict(stage=i,workspace_rms=float(z.float().square().mean().sqrt()),
                token_cosine=float(pairs[:,off].mean()),jet_variation=float(z.float().var(0,unbiased=False).mean()),
                feedback_ratio=float((h-args[0]).float().norm()/args[0].float().norm().clamp_min(1e-12))))
        return hook
    def attention_hook(name):
        def hook(module,args,out):
            q,x=args[:2];valid=args[2] if len(args)>2 else None
            b,n,_=q.shape;d=16
            qs=module.query(q).reshape(b,n,8,d).transpose(1,2)
            ks=module.key(x).reshape(b,x.shape[1],8,d).transpose(1,2)
            scores=qs@ks.transpose(-1,-2)/4
            if valid is not None:scores=scores.masked_fill(~valid[:,None,None,:],-torch.inf)
            weights=scores.softmax(-1)
            attention[name]=float(-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean())
        return hook
    try:
        base.eval();base.mod.trimmer.enabled=False
        for i,stage in enumerate(base.mod.global_workspace.stages):handles.append(stage.register_forward_hook(stage_hook(i)))
        for name,m in base.mod.global_workspace.named_modules():
            if isinstance(m,Attention):handles.append(m.register_forward_hook(attention_hook(name)))
        with torch.no_grad(),model_precision(base,device,training=False):expected=base(*values)
        for h in handles:h.remove()
        handles=[]
        effects={}
        for mode in ('feedback','class_read'):
            with torch.no_grad():
                for n,p in base.named_parameters():
                    target=('write.output.' in n) if mode=='feedback' else ('class_read.attention.output.' in n)
                    if 'global_workspace' in n and target:p.zero_()
                with model_precision(base,device,training=False):changed=base(*values)
                effects[mode]=float((changed-expected).abs().max())
                for n,p in base.named_parameters():
                    if n in params:p.copy_(params[n])
        return dict(events=8,particles=17,workspace=True,layers=layers,
                    attention_entropy=attention,disabled_logit_changes=effects)
    finally:
        for h in handles:h.remove()
        with torch.no_grad():
            for n,p in base.named_parameters():
                if n in params:p.copy_(params[n])
        for m,flag in flags:m.training=flag
        base.mod.trimmer.enabled=trim;restore_rank_state(base,rng)
