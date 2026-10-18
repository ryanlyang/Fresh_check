"""Class-free canonical input keys: no loader changes or global RNG draws.

Keys use the actual post-trim 17 features and four-vector bits, never hidden
states/labels/filenames. Equal keys for truly identical inputs are interchangeable
in evaluation; distinguishable hash collisions use full lexicographic input order.
"""
import torch


@torch.no_grad()
def canonical_order(features, vectors, valid):
    values = torch.cat((vectors.float(), features.float()), -1)
    order = torch.arange(values.shape[1], device=values.device).expand(values.shape[:2])
    for column in reversed(range(values.shape[-1])):
        keys = values[..., column].gather(1, order)
        order = order.gather(1, torch.argsort(keys, dim=1, stable=True))
    return order.gather(1, torch.argsort(~valid.gather(1, order), dim=1, stable=True))


@torch.no_grad()
def random_order(features, vectors, valid, seed):
    values = torch.cat((vectors.float(), features.float()), -1).contiguous()
    # Canonicalize signed zero before bit hashing; no floating-point arithmetic
    # is used in the hash. All integer products stay below int64's upper bound.
    values = torch.where(values == 0, torch.zeros_like(values), values).contiguous()
    bits = values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    prime = 2147483647
    h = torch.full(valid.shape, int(seed) % prime, dtype=torch.int64, device=values.device)
    for column in range(bits.shape[-1]):
        h = (h * 48271 + bits[..., column] + column * 101) % prime
    jet_key = (h * valid).sum(1, keepdim=True) % prime
    h = (h * 48271 + jet_key * 40699 + int(seed)) % prime
    h = h.masked_fill(~valid, prime)
    order = canonical_order(features, vectors, valid)
    return order.gather(1, torch.argsort(h.gather(1, order), dim=1, stable=True))
