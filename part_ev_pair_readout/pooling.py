"""Globally normalized, graph-connected online pair pooling. No module state."""
import torch


def statistics(scores, values, valid):
    """[B,A,P], [B,P,D], [B,P] -> present, detached max, mass, numerator, health."""
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    with torch.autocast(scores.device.type, enabled=False):
        scores, values = scores.to(dtype), values.to(dtype)
        mask = valid[:, None].expand_as(scores)
        health = (torch.isfinite(scores).logical_or(~mask).all()
                  & torch.isfinite(values).logical_or(~valid[..., None]).all())
        present = mask.any(-1)
        maximum = scores.masked_fill(~mask, -torch.inf).amax(-1)
        maximum = torch.where(present, maximum, torch.zeros_like(maximum)).detach()
        # Mask BEFORE exponentiation; empty chunks never evaluate inf-inf/exp(inf).
        shifted = (scores - maximum[..., None]).masked_fill(~mask, 0)
        weight = shifted.exp().masked_fill(~mask, 0)
        values = values.masked_fill(~valid[..., None], 0)
        mass = weight.sum(-1)
        numerator = torch.einsum('bap,bpd->bad', weight, values)
        return present, maximum, mass, numerator, health


def merge(a, b):
    pa, ma, da, za, ha = a
    pb, mb, db, zb, hb = b
    with torch.autocast(ma.device.type, enabled=False):
        present = pa | pb
        maximum = torch.where(pa & pb, torch.maximum(ma, mb), torch.where(pa, ma, mb))
        maximum = torch.where(present, maximum, torch.zeros_like(maximum)).detach()
        sa = torch.where(pa, ma - maximum, torch.zeros_like(ma)).exp() * pa
        sb = torch.where(pb, mb - maximum, torch.zeros_like(mb)).exp() * pb
        return present, maximum, sa*da+sb*db, sa[..., None]*za+sb[..., None]*zb, ha & hb


def finish(stats):
    present, _, mass, numerator, healthy = stats
    with torch.autocast(mass.device.type, enabled=False):
        result = numerator / torch.where(present, mass, torch.ones_like(mass))[..., None]
        healthy = healthy & torch.isfinite(result).all() & torch.isfinite(mass).all()
    return result, healthy


def dense_reference(scores, values, valid):
    """Small-input oracle using one dense softmax, independent of chunk merge."""
    mask = valid[:, None].expand_as(scores)
    present = mask.any(-1, keepdim=True)
    safe = torch.where(present, scores.masked_fill(~mask, -torch.inf), torch.zeros_like(scores))
    weights = safe.softmax(-1).masked_fill(~mask, 0)
    return torch.einsum('bap,bpd->bad', weights, values.masked_fill(~valid[..., None], 0))
