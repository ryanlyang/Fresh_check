"""Per-jet contextual routing; no randomness, collectives or retained state."""
import torch
from torch import nn
from torch.nn import functional as F


def reduction_dtype(x):return torch.float64 if x.dtype==torch.float64 else torch.float32


class StableNorm(nn.LayerNorm):
    def __init__(self,width):super().__init__(width,eps=1e-5)
    def forward(self,x):
        with torch.autocast(x.device.type,enabled=False):
            return F.layer_norm(x.to(reduction_dtype(x)),self.normalized_shape,self.weight,self.bias,self.eps)


def route(u,valid,context_norm,router):
    """Return packed valid states, probabilities and stable token-major routes."""
    if u.ndim!=3 or valid.shape!=u.shape[:2] or valid.dtype!=torch.bool:
        raise ValueError('Expected batch-first states and exact post-trim boolean mask')
    with torch.autocast(u.device.type,enabled=False):
        with torch.profiler.record_function('experts.context_router'):
            values=u.to(reduction_dtype(u));safe=values.masked_fill(~valid[...,None],0)
            counts=valid.sum(1)
            # Empty jets are invalid; this safe denominator never admits them.
            context=context_norm(safe.sum(1)/counts.clamp_min(1)[:,None])
            positions=valid.flatten().nonzero(as_tuple=False).flatten()
            packed=values.reshape(-1,128).index_select(0,positions)
            g=context.index_select(0,positions//u.shape[1])
            logits=router(torch.cat((packed,g,packed*g),-1))
            probabilities=logits.softmax(-1)
        with torch.profiler.record_function('experts.stable_selection'):
            order=torch.argsort(logits,dim=-1,descending=True,stable=True)
            ids=order[:,:2]
            selected=logits.gather(1,ids).softmax(-1)
            requested=torch.bincount(ids.flatten(),minlength=4)
        healthy=((counts>0).all() & torch.isfinite(safe).all() & torch.isfinite(context).all()
            & torch.isfinite(logits).all() & torch.isfinite(probabilities).all() & torch.isfinite(selected).all())
        margin=(logits.gather(1,order[:,1:2])-logits.gather(1,order[:,2:3])).flatten()
        return dict(packed=packed,positions=positions,probabilities=probabilities,ids=ids,
            weights=selected,requested=requested,health=healthy,logits=logits,margin=margin)
