"""Explicit small attention: FP32 reductions, FP64 oracle support, tensor health."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def reduction_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


class StableNorm(nn.LayerNorm):
    def __init__(self, width=128):
        super().__init__(width, eps=1e-5)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return F.layer_norm(x.to(reduction_dtype(x)), self.normalized_shape,
                                self.weight, self.bias, self.eps)


class Attention(nn.Module):
    def __init__(self, width=128, heads=8):
        super().__init__()
        if width % heads: raise ValueError('Head width must divide embedding')
        self.width, self.heads = width, heads
        self.query, self.key = nn.Linear(width, width), nn.Linear(width, width)
        self.value, self.output = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, query, source, valid=None, allowed=None, query_valid=None):
        if query.ndim != 3 or source.ndim != 3 or query.shape[0] != source.shape[0]:
            raise ValueError('Expected batch-first query and source')
        if valid is not None:
            if valid.shape != source.shape[:2] or valid.dtype != torch.bool:
                raise ValueError('Boolean source mask shape mismatch')
            source = source.masked_fill(~valid[..., None], 0)
        b, nq, _ = query.shape
        if query_valid is None:query_valid=torch.ones((b,nq),device=query.device,dtype=torch.bool)
        if valid is None:valid=torch.ones(source.shape[:2],device=source.device,dtype=torch.bool)
        if query_valid.shape!=(b,nq) or query_valid.dtype!=torch.bool:raise ValueError('Query mask shape')
        if allowed is None:allowed=query_valid[:,:,None]&valid[:,None,:]
        if allowed.shape!=(b,nq,source.shape[1]) or allowed.dtype!=torch.bool:raise ValueError('Allowed mask shape')
        allowed=allowed&query_valid[:,:,None]&valid[:,None,:]
        query=query.masked_fill(~query_valid[...,None],0)
        ns, d = source.shape[1], self.width//self.heads
        q, k, v = self.query(query), self.key(source), self.value(source)
        healthy = torch.isfinite(q).all() & torch.isfinite(k).all() & torch.isfinite(v).all()
        def heads(x, n): return x.reshape(b, n, self.heads, d).transpose(1, 2)
        with torch.autocast(query.device.type, enabled=False):
            dtype = reduction_dtype(query)
            q, k, v = heads(q,nq).to(dtype), heads(k,ns).to(dtype), heads(v,ns).to(dtype)
            scores = (q @ k.transpose(-1,-2))/math.sqrt(d)
            healthy = healthy & torch.isfinite(scores).all()
            if valid is not None:
                nonempty = allowed.any(-1)
                healthy = healthy & (nonempty|~query_valid).all()
                scores = scores.masked_fill(~allowed[:,None,:,:], -torch.inf)
                # A malformed empty row never causes an unmatched rank exception.
                # Health is false and the outer model poisons logits for rejection.
                scores = torch.where(nonempty[:,None,:,None], scores, torch.zeros_like(scores))
            weights = scores.softmax(-1)
            result = (weights @ v).transpose(1,2).reshape(b,nq,self.width)
        result = self.output(result).masked_fill(~query_valid[...,None],0)
        healthy = healthy & torch.isfinite(result).all()
        return result, healthy
