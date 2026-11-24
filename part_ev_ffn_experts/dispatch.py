"""Dropless fixed-ID execution and unique assignment-slot combination."""
import torch
from torch.utils.checkpoint import checkpoint
from .routing import reduction_dtype


def dispatch(packed,ids,weights,experts,*,checkpointed):
    tokens=packed.shape[0]
    flat=ids.reshape(-1)
    dtype=reduction_dtype(packed)
    slots=torch.zeros((tokens*2,128),dtype=dtype,device=packed.device)
    health=torch.ones((),dtype=torch.bool,device=packed.device);executed=[]
    for e,expert in enumerate(experts):
        with torch.profiler.record_function('experts.packing'):
            positions=(flat==e).nonzero(as_tuple=False).flatten()
            x=packed.index_select(0,positions//2)
        if checkpointed and torch.is_grad_enabled():
            output=checkpoint(expert,x,use_reentrant=False,preserve_rng_state=False)
        else:output=expert(x)
        health=health & torch.isfinite(output).all()
        # Empty index_copy still retains the expert's autograd dependency;
        # no synthetic token and no atomic many-writer particle accumulation.
        slots=slots.index_copy(0,positions,output.to(dtype))
        executed.append(positions.numel())
    with torch.profiler.record_function('experts.combination'),torch.autocast(packed.device.type,enabled=False):
        slots=slots.reshape(tokens,2,128)
        correction=weights[:,0,None]*slots[:,0]+weights[:,1,None]*slots[:,1]
        health=health & torch.isfinite(correction).all()
    return correction,torch.tensor(executed,dtype=torch.int64,device=packed.device),health
