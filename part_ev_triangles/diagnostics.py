"""Small detached reports; no pair caches or profiler traces persisted."""
import time
import torch
from torch.nn import functional as F
from . import model as M
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


def stage_profile(base,wrapped,values,truth,device,deadline,*,training=True):
    """Separate from throughput. Restore buffers/RNG; do not step the optimizer."""
    if B.STOP or time.monotonic()>=deadline:raise TimeoutError('Profiling budget exhausted')
    rng=capture_rank_state(base);flags=[(m,m.training) for m in base.modules()]
    buffers={n:b.detach().clone() for n,b in base.named_buffers()}
    trims=[(m,m.enabled) for m in base.modules() if type(m).__name__=='SequenceTrimmer']
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        base.train(training);base.zero_grad(set_to_none=True)
        for m,_ in trims:m.enabled=True
        with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
            with torch.set_grad_enabled(training),model_precision(base,device,training=training):
                prediction=wrapped(*values);loss=F.cross_entropy(prediction,truth)
            if training:loss.backward()
            if device.type=='cuda':torch.cuda.synchronize(device)
        if not prediction.isfinite().all():raise ValueError('Nonfinite profiling forward')
        return {event.key:dict(calls=event.count,cpu_us=float(event.cpu_time_total),
                              device_us=float(getattr(event,'device_time_total',0)))
                for event in prof.key_averages() if event.key.startswith('triangles.')}
    finally:
        with torch.no_grad():
            for n,b in base.named_buffers():b.copy_(buffers[n])
        base.zero_grad(set_to_none=True)
        for m,v in flags:m.training=v
        for m,v in trims:m.enabled=v
        restore_rank_state(base,rng)
