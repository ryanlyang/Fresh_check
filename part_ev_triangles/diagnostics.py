"""Small detached reports; no pair caches or profiler traces persisted."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from . import model as M, protocol as P
from part_ev_gpu_scaling import benchmark as B
from part_ev_fullscale.checkpoint import capture_rank_state,restore_rank_state
from part_ev_screen.precision import model_precision


def tied_batch(events=512,particles=128):
    batch=B.fixed_batch(events,particles)
    # Identical geometry, but original heterogeneous particle features/labels.
    batch[0]['pf_vectors']=batch[0]['pf_vectors'][:,:,:1].expand(-1,-1,particles).clone()
    return batch


class NeighborStats:
    def __init__(self,base):
        self.phase=None;self.data={}
        self.handle=base.mod.triangle_neighbors.register_forward_hook(self.observe)

    @torch.no_grad()
    def observe(self,module,args,graph):
        if self.phase is None:return
        counts=graph.counts.masked_fill(~graph.valid,128)
        hist=torch.bincount(counts.flatten(),minlength=129)[:128]
        mass=graph.weights.sum(dtype=torch.float64)
        if self.phase in self.data:
            h,m=self.data[self.phase];hist=h+hist;mass=m+mass
        self.data[self.phase]=(hist,mass)

    def report(self):
        result={}
        for phase,(h,m) in self.data.items():
            hist=h.cpu().tolist();total=sum(hist)
            if not total:raise ValueError('No valid receiver statistics')
            def quantile(q):
                cumulative=0
                for i,n in enumerate(hist):
                    cumulative+=n
                    if cumulative>=q*total:return i
            result[phase]=dict(histogram=hist,receivers=total,tie_fraction=sum(hist[9:])/total,
                maximum_support=max(i for i,n in enumerate(hist) if n),
                support_p50=quantile(.5),support_p95=quantile(.95),support_p99=quantile(.99),
                mean_weight_mass=float(m)/total)
        return result

    def close(self):self.handle.remove()


def gradient_norms(base):
    return {n:float(p.grad.float().norm()) for n,p in base.named_parameters()
            if M.triangle_key(n) and p.grad is not None}


def diagnostic_sample(base,values,device):
    rng=capture_rank_state(base);flags=[(m,m.training) for m in base.modules()]
    trims=[(m,m.enabled) for m in base.modules() if type(m).__name__=='SequenceTrimmer']
    try:
        base.eval()
        for m,_ in trims:m.enabled=False
        with torch.no_grad(),model_precision(base,device,training=False):
            logits,info=base.mod.forward_details(values[1],values[2],values[3],diagnostic=True)
            disabled,_=base.mod.forward_details(values[1],values[2],values[3],disabled=True)
            return dict(healthy=bool(info['healthy'] and logits.isfinite().all()),layers=info['layers'],
                events=len(values[0]),particles=values[0].shape[-1],
                disabled_logit_rms_change=float((disabled-logits).square().mean().sqrt()))
    finally:
        base.mod.pair_embed.clear()
        for m,v in flags:m.training=v
        for m,v in trims:m.enabled=v
        restore_rank_state(base,rng)


def host_memory():
    """Per-process RSS, not the whole Slurm job or its data-loader workers."""
    row = dict(pid=os.getpid(), scope='process_only_excludes_workers',
               rss_bytes=None, peak_rss_bytes=None)
    status = Path('/proc/self/status')
    if sys.platform.startswith('linux') and status.is_file():
        for line in status.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in ('VmRSS:', 'VmHWM:'):
                row['rss_bytes' if fields[0] == 'VmRSS:' else 'peak_rss_bytes'] = int(fields[1])*1024
    return row


def profile_sample(values, truth):
    """Bound trace event counts even for the all-ties fixture; never a speed estimate."""
    source_events, source_particles = values[0].shape[0], values[0].shape[-1]
    if (source_events < 1 or source_particles < 1 or len(truth) != source_events
            or any(v.ndim != 3 or v.shape[0] != source_events or v.shape[-1] != source_particles for v in values)):
        raise ValueError('Invalid profiling inputs')
    events = min(source_events, P.PROFILE_POLICY['max_events'])
    particles = min(source_particles, P.PROFILE_POLICY['max_particles'])
    bounded = [v[:events, :, :particles].detach().clone().contiguous() for v in values]
    if not bounded[-1].bool().any(-1).all():
        raise ValueError('Profiling slice has an empty jet')
    return bounded, truth[:events].detach().clone(), dict(source_events=source_events,
        source_particles=source_particles, events=events, particles=particles)


def stage_profile(base,wrapped,values,truth,device,deadline,*,training=True,label='diagnostic'):
    """One bounded diagnostic forward/backward, separate from full-size admission.

    Both DDP ranks must participate in training profiles (including SyncBN).
    Only inference profiles may run on rank zero alone. No optimizer step.
    """
    if B.STOP or time.monotonic()>=deadline:raise TimeoutError('Profiling budget exhausted')
    values,truth,sample=profile_sample(values,truth)
    sample['training']=training
    before=host_memory()
    def log(phase, memory):
        print(json.dumps(dict(stage='triangle_profile',label=label,phase=phase,
            rank=int(os.environ.get('RANK',0)),sample=sample,policy=P.PROFILE_POLICY,
            host_memory=memory)),flush=True)
    log('start',before)
    rng=capture_rank_state(base);flags=[(m,m.training) for m in base.modules()]
    buffers={n:b.detach().clone() for n,b in base.named_buffers()}
    trims=[(m,m.enabled) for m in base.modules() if type(m).__name__=='SequenceTrimmer']
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    complete=False;prof=None
    try:
        base.train(training);base.zero_grad(set_to_none=True)
        # The bounds describe the actual profiled tensor; no stochastic trimming.
        for m,_ in trims:m.enabled=False
        with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
            with torch.set_grad_enabled(training),model_precision(base,device,training=training):
                prediction=wrapped(*values);loss=F.cross_entropy(prediction,truth)
            if training:loss.backward()
            if device.type=='cuda':torch.cuda.synchronize(device)
        if not prediction.isfinite().all():raise ValueError('Nonfinite profiling forward')
        stages = {event.key:dict(calls=event.count,cpu_us=float(event.cpu_time_total),
                              device_us=float(getattr(event,'device_time_total',0)))
                for event in prof.key_averages() if event.key.startswith('triangles.')}
        complete=True
    finally:
        # Do not keep an event history alive through restart/validation/inference.
        del prof
        gc.collect()
        with torch.no_grad():
            for n,b in base.named_buffers():b.copy_(buffers[n])
        base.zero_grad(set_to_none=True)
        for m,v in flags:m.training=v
        for m,v in trims:m.enabled=v
        restore_rank_state(base,rng)
        after=host_memory()
        log('complete' if complete else 'failed',after)
    return dict(policy=dict(P.PROFILE_POLICY),sample=sample,stages=stages,
                host_memory=dict(before=before,after=after))
