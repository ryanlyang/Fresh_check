"""Masked additive support statistics. No attention normalization or 5-D products."""
import torch

EPSILON=1e-6


def statistics(logits,values,valid):
    dtype=torch.float64 if logits.dtype==torch.float64 else torch.float32
    with torch.autocast(logits.device.type,enabled=False):
        logits,values=logits.to(dtype),values.to(dtype)
        healthy=(torch.isfinite(logits).logical_or(~valid[...,None]).all()
                 & torch.isfinite(values).logical_or(~valid[...,None]).all())
        # Check BEFORE sigmoid: finite saturation is allowed, concealed Inf isn't.
        gates=logits.sigmoid().masked_fill(~valid[...,None],0)
        content=values.masked_fill(~valid[...,None],0)
        count=gates.sum(-2)
        total=torch.einsum('bqja,bqjd->bqad',gates,content)
        degree=valid.sum(-1)
        healthy=healthy & torch.isfinite(count).all() & torch.isfinite(total).all()
        return count,total,degree,healthy


def merge(a,b):
    """For math tests/future sender tiling. Never merge normalized summaries."""
    return a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]&b[3]


def finish(count,total,degree,healthy):
    with torch.autocast(count.device.type,enabled=False):
        mean=total/count.clamp_min(EPSILON)[...,None]
        log_count=count.log1p()
        fraction=count/degree.clamp_min(1)[...,None]
        features=torch.cat((mean,log_count[...,None],fraction[...,None]),-1).flatten(-2)
        features=features.masked_fill(~(degree>0)[...,None],0)
        return features,healthy & torch.isfinite(features).all()
