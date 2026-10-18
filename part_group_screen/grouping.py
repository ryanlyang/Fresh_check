"""Four group-assignment strategies with one masking contract."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import coordinates, distance, locality_penalty
from .metadata import canonical_order, random_order


def active_slots(valid, groups):
    return torch.arange(groups, device=valid.device)[None] < valid.sum(1, keepdim=True).clamp_max(groups)


@torch.no_grad()
def fixed_assignment(vectors, features, valid, groups=8):
    """Batched exact fixed-count angular agglomeration, no per-merge host sync.

Only the merged cluster's distance row/column is updated. Initial canonical
ordering fixes ties for distinguishable inputs; slot names have no semantics.
"""
    bsize, length, _ = vectors.shape
    order = canonical_order(features, vectors, valid)
    v = vectors.float().gather(1, order[..., None].expand(-1, -1, 4)).clone()
    alive = valid.gather(1, order).clone()
    labels = torch.arange(length, device=v.device).expand(bsize, -1).clone()
    batch = torch.arange(bsize, device=v.device)
    indices = torch.arange(length, device=v.device)
    upper = indices[:, None] < indices[None, :]
    _, y, phi = coordinates(v)
    d = distance(y[:, :, None], phi[:, :, None], y[:, None, :], phi[:, None, :])
    for _ in range(max(0, length - groups)):
        do_merge = alive.sum(1) > groups
        distances = d.masked_fill(~(alive[:, :, None] & alive[:, None, :] & upper), float("inf"))
        chosen = distances.flatten(1).argmin(1)
        left, right = chosen // length, chosen % length
        combined = v[batch, left] + v[batch, right]
        v[batch, left] = torch.where(do_merge[:, None], combined, v[batch, left])
        alive[batch, right] &= ~do_merge
        labels = torch.where(do_merge[:, None] & (labels == right[:, None]), left[:, None], labels)
        _, y, phi = coordinates(v)
        row = distance(y[batch, left, None], phi[batch, left, None], y, phi)
        d[batch, left, :] = row
        d[batch, :, left] = row
    compact = (alive.long().cumsum(1) - 1).clamp_min(0)
    assigned = compact.gather(1, labels).clamp_max(groups - 1)
    a_sorted = F.one_hot(assigned, groups).float() * valid.gather(1, order)[..., None]
    a = torch.zeros_like(a_sorted).scatter(1, order[..., None].expand_as(a_sorted), a_sorted)
    return a, active_slots(valid, groups)


@torch.no_grad()
def random_assignment(vectors, features, valid, groups=8, seed=731019):
    fixed, active = fixed_assignment(vectors, features, valid, groups)
    sizes = fixed.sum(1).long()
    order = random_order(features, vectors, valid, seed)
    rank = torch.arange(valid.shape[1], device=vectors.device)[None, :, None]
    slot = (rank >= sizes.cumsum(1)[:, None, :]).sum(-1).clamp_max(groups - 1)
    ordered = F.one_hot(slot, groups).float() * valid.gather(1, order)[..., None]
    return torch.zeros_like(ordered).scatter(1, order[..., None].expand_as(ordered), ordered), active


class SoftRouter(nn.Module):
    def __init__(self, config, *, locality=0.):
        super().__init__()
        self.groups, self.width = config["groups"], config["width"]
        self.rounds, self.temperature = config["rounds"], config["temperature"]
        self.locality, self.radius_floor = float(locality), config["radius_floor"]
        self.eps = config["epsilon"]
        self.slots = nn.Parameter(torch.empty(self.groups, self.width))
        nn.init.normal_(self.slots, std=.02)
        self.input_norm = nn.LayerNorm(self.width, eps=config["norm_epsilon"])
        self.slot_norm = nn.LayerNorm(self.width, eps=config["norm_epsilon"])
        self.key = nn.Linear(self.width, self.width)
        self.value = nn.Linear(self.width, self.width)
        self.query = nn.Linear(self.width, self.width)
        self.update = nn.GRUCell(self.width, self.width)

    def forward(self, hidden, vectors, valid):
        active = active_slots(valid, self.groups)
        x = self.input_norm(hidden)
        key, value = self.key(x), self.value(x)
        slots = self.slots[None].expand(hidden.shape[0], -1, -1)
        a = None
        for round_index in range(self.rounds):
            query = self.query(self.slot_norm(slots))
            # Scores/softmax/reductions are FP32; learned projections retain AMP.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                scores = torch.einsum("bnd,bkd->bnk", key.float(), query.float()) / math.sqrt(self.width)
                if round_index and self.locality:
                    scores = scores - self.locality * locality_penalty(a, vectors, valid, self.radius_floor, self.eps)
                a = (scores / self.temperature).masked_fill(~active[:, None], -torch.inf).softmax(-1)
                a = a * valid[..., None]
                pooled = torch.einsum("bnk,bnd->bkd", a, value.float()) / (a.sum(1)[..., None] + self.eps)
            if round_index < self.rounds - 1:
                slots = self.update(pooled.reshape(-1, self.width), slots.reshape(-1, self.width)).reshape_as(slots)
        return a, active
