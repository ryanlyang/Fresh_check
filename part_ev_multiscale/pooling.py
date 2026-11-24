"""Linear-size segmented pooling; amax shares gradients across every tied member."""
import torch
from .attention import reduction_dtype


def pool(hidden,geometry,valid):
    with torch.profiler.record_function('multiscale.pooling'),torch.autocast(hidden.device.type,enabled=False):
        x=hidden.to(reduction_dtype(hidden)).masked_fill(~valid[...,None],0)
        membership=geometry.fine.to(x.dtype)
        mean=membership.transpose(1,2)@x/geometry.fine_count.clamp_min(1)[...,None]
        index=geometry.fine_ids.clamp_min(0)[...,None].expand_as(x)
        source=x.masked_fill(~valid[...,None],-torch.inf)
        maximum=torch.full((x.shape[0],16,x.shape[-1]),-torch.inf,device=x.device,dtype=x.dtype)
        maximum=maximum.scatter_reduce(1,index,source,reduce='amax',include_self=True)
        maximum=maximum.masked_fill(~geometry.fine_valid[...,None],0)
        return mean,maximum
