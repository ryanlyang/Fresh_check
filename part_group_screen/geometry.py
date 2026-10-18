"""Masked FP32 geometry. Soft assignments remain differentiable."""
from __future__ import annotations
import torch


def wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def positive_sqrt(value, eps=1e-6):
    # Unlike sqrt(clamp(x, 0)), the inactive branch has finite derivatives too.
    return torch.where(value > 0, value.clamp_min(eps * eps).sqrt(), torch.zeros_like(value))


def coordinates(vectors, eps=1e-6):
    px, py, pz, energy = vectors.unbind(-1)
    pt2 = px.square() + py.square()
    pt = positive_sqrt(pt2, eps)
    good = pt2 > eps * eps
    phi = torch.atan2(torch.where(good, py, torch.zeros_like(py)),
                      torch.where(good, px, torch.ones_like(px)))
    rapidity = .5 * ((energy + pz).clamp_min(eps).log() - (energy - pz).clamp_min(eps).log())
    return pt, rapidity, phi


def distance(y1, phi1, y2, phi2):
    return (y1 - y2).square() + wrap(phi1 - phi2).square()


def descriptors(assignment, vectors, valid, eps=1e-6):
    """Return occupancy, group four-vectors, and the study's seven features."""
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        a = assignment.float() * valid[..., None]
        v = vectors.float().masked_fill(~valid[..., None], 0)
        n = a.sum(1)
        groups = torch.einsum("bnk,bnd->bkd", a, v)
        pt, y, phi = coordinates(v, eps)
        gpt, gy, gphi = coordinates(groups, eps)
        _, jy, jphi = coordinates(v.sum(1), eps)
        mass = positive_sqrt(groups[..., 3].square() - groups[..., :3].square().sum(-1), eps)
        weights = a * pt[..., None]
        spread = distance(y[..., None], phi[..., None], gy[:, None], gphi[:, None])
        width = positive_sqrt((weights * spread).sum(1) / weights.sum(1).clamp_min(eps), eps)
        dphi = wrap(gphi - jphi[:, None])
        features = torch.stack(((gpt / pt.sum(1, keepdim=True).clamp_min(eps)).clamp_min(eps).log(),
            mass.log1p(), gy - jy[:, None], dphi.sin(), dphi.cos(), n.log1p(), width.log1p()), -1)
        return n, groups, features


def locality_penalty(assignment, vectors, valid, radius_floor=.05, eps=1e-6):
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        v = vectors.float().masked_fill(~valid[..., None], 0)
        pt, y, phi = coordinates(v, eps)
        _, jy, jphi = coordinates(v.sum(1), eps)
        w = assignment.float() * pt[..., None] * valid[..., None]
        denom = w.sum(1)
        cy = (w * y[..., None]).sum(1) / denom.clamp_min(eps)
        cy = torch.where(denom > eps, cy, jy[:, None])
        s = (w * phi.sin()[..., None]).sum(1)
        c = (w * phi.cos()[..., None]).sum(1)
        good = (s.square() + c.square() > eps * eps) & (denom > eps)
        # Keep atan2's unused branch finite as well as its returned value.
        cp = torch.atan2(torch.where(good, s, torch.zeros_like(s)),
                         torch.where(good, c, torch.ones_like(c)))
        cp = torch.where(good, cp, jphi[:, None])
        r2 = ((pt * distance(y, phi, jy[:, None], jphi[:, None])).sum(1)
              / pt.sum(1).clamp_min(eps)).clamp_min(radius_floor ** 2)
        return distance(y[..., None], phi[..., None], cy[:, None], cp[:, None]) / (2 * r2[:, None, None])
