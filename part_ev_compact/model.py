"""Compact all-eight layerwise State-EV and a shared-bias FFN capacity control.

Rank reduction and tying the pair projection are architectural changes. Packed
symmetric projection, caching and output-projection merging are execution
changes; ``optimized=False`` is their deliberately slow mathematical oracle.
No existing campaign/model configuration is modified.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from part_ev_fullscale.model import build_model as native_model
from part_layerwise_screen.model import build_model as layerwise_model

COUNTS = {"compact": 2262950, "matched": 2263662}
RANK = 16
WIDTH = 570


def normalize(x):
    with torch.autocast(x.device.type, enabled=False):
        return F.layer_norm(x if x.dtype == torch.float64 else x.float(), (x.shape[-1],), eps=1e-5)


class PairCache:
    def __init__(self, projection, optimized):
        self.projection, self.optimized = projection, optimized
        self.clear()

    def clear(self):
        self.stem = self.packed = self.lookup = None

    def get(self, stem):
        n = stem.shape[-1]
        if not self.optimized:
            projected = self.projection(normalize(stem.permute(0, 2, 3, 1)))
            return projected.flatten(1, 2), torch.arange(n*n, device=stem.device).reshape(n, n)
        if self.stem is not stem:
            self.stem = stem
            row, col = torch.tril_indices(n, n, device=stem.device)
            self.packed = self.projection(normalize(stem[:, :, row, col].transpose(1, 2)))
            indices = torch.arange(n, device=stem.device)
            hi = torch.maximum(indices[:, None], indices[None, :])
            lo = torch.minimum(indices[:, None], indices[None, :])
            self.lookup = hi*(hi+1)//2 + lo
        return self.packed, self.lookup


class StateMessage(nn.Module):
    def __init__(self, cache, chunk=64):
        super().__init__()
        self.receiver, self.sender = nn.Linear(128, RANK, bias=False), nn.Linear(128, RANK, bias=False)
        nn.init.xavier_uniform_(self.receiver.weight)
        nn.init.xavier_uniform_(self.sender.weight)
        self.output = nn.Parameter(torch.zeros(8, 16, RANK))
        self.cache, self.chunk = cache, chunk

    def pool(self, weights, packed, lookup, receiver, sender, valid):
        p, q = receiver[:, :, None], sender[:, None]
        features = F.gelu(normalize(packed[:, lookup] + p + q + p*q))
        features = features.masked_fill(~valid[..., None], 0)
        pooled = torch.einsum("bhij,bijr->bhir", weights, features)
        return torch.einsum("hor,bhir->bhio", self.output, pooled)

    def forward(self, weights, stem, query, valid):
        packed, lookup = self.cache.get(stem)
        states = normalize(query.transpose(0, 1))
        p, q = self.receiver(states), self.sender(states)
        pieces = []
        for start in range(0, query.shape[0], self.chunk):
            stop = start+self.chunk
            args = (weights[:, :, start:stop], packed, lookup[start:stop], p[:, start:stop], q,
                    valid[:, start:stop, None] & valid[:, None, :])
            # Backward never consults the cleared mutable forward cache.
            pieces.append(checkpoint(self.pool, *args, use_reentrant=False, preserve_rng_state=False)
                          if self.training and torch.is_grad_enabled() else self.pool(*args))
        return torch.cat(pieces, dim=2)


def pre_output_attention(ref, query, padding, bias):
    """Pinned need_weights=True MHA math, stopping just before its output GEMM."""
    n, b, e = query.shape
    if (ref.batch_first or ref.bias_k is not None or ref.bias_v is not None
            or ref.add_zero_attn or not ref._qkv_same_embed_dim or e != 128 or ref.num_heads != 8):
        raise ValueError("Unsupported native attention configuration")
    q, k, v = F._in_projection_packed(query, query, query, ref.in_proj_weight, ref.in_proj_bias)
    q, k, v = [x.view(n, b*8, 16).transpose(0, 1) for x in (q, k, v)]
    def additive(mask):
        return torch.zeros_like(mask, dtype=query.dtype).masked_fill(mask, float('-inf')) if mask.dtype == torch.bool else mask
    mask = additive(bias) if bias is not None else None
    if padding is not None:
        pad = additive(padding).view(b, 1, 1, n).expand(-1, 8, -1, -1).reshape(b*8, 1, n)
        mask = pad if mask is None else mask+pad
    scores = (torch.bmm(q*math.sqrt(1/16), k.transpose(-2, -1)) if mask is None else
              torch.baddbmm(mask, q*math.sqrt(1/16), k.transpose(-2, -1)))
    weights = F.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=ref.dropout, training=ref.training)
    result = torch.bmm(weights, v).transpose(0, 1).contiguous().view(n, b, e)
    return result, weights.view(b, 8, n, n)


class CompactAttention(nn.Module):
    def __init__(self, old, state, optimized):
        super().__init__()
        self.reference, self.edge_projection = old.reference, old.edge_projection
        self.state, self.optimized = state, optimized
        self.clear()

    def bind(self, stem, valid):
        self.stem, self.valid = stem, valid

    def clear(self):
        self.stem = self.valid = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if query is not key or key is not value or is_causal:
            raise ValueError("Only noncausal particle self-attention is supported")
        if self.stem is None or self.valid is None:
            raise RuntimeError("Missing native pair context")
        if self.optimized:
            ordinary, weights = pre_output_attention(self.reference, query, key_padding_mask, attn_mask)
        else:
            ordinary, weights = self.reference(query, key, value, key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        pooled = torch.einsum("bhij,bdij->bhid", weights, self.stem)
        heads = torch.einsum("hod,bhid->bhio", self.edge_projection, pooled)
        heads = heads + self.state(weights, self.stem, query, self.valid)
        joined = heads.permute(2, 0, 1, 3).reshape_as(ordinary)
        projection = self.reference.out_proj
        if self.optimized:
            result = F.linear(ordinary + joined*self.valid.T[..., None], projection.weight, projection.bias)
        else:
            result = ordinary + F.linear(joined, projection.weight)*self.valid.T[..., None]
        returned = weights.mean(1) if average_attn_weights else weights
        return result, returned if need_weights else None


def clear_before(model, _args):
    model._compact_cache.clear()


def clear_after(model, _args, _output):
    model._compact_cache.clear()


def build_model(architecture, *, seed=101, trim=True, use_amp=False, optimized=True):
    if architecture not in COUNTS:
        raise ValueError(f"Unknown compact campaign architecture: {architecture}")
    if architecture == "matched":
        model = native_model("part", trim=trim, use_amp=use_amp)
        with torch.random.fork_rng(devices=[]):
            for block in model.mod.blocks:
                if block.ffn_dim != 512: raise ValueError("Native FFN changed")
                block.ffn_dim = WIDTH
                block.fc1 = nn.Linear(128, WIDTH)
                block.post_fc_norm = nn.LayerNorm(WIDTH, eps=block.post_fc_norm.eps)
                block.fc2 = nn.Linear(WIDTH, 128)
    else:
        model = layerwise_model("part_ev_layerwise", trim=trim, use_amp=use_amp)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+81016)
            model.shared_state_edge = nn.Linear(64, RANK, bias=False)
            nn.init.xavier_uniform_(model.shared_state_edge.weight)
            cache = PairCache(model.shared_state_edge, optimized)
            for layer, block in enumerate(model.mod.blocks):
                torch.random.default_generator.manual_seed(seed+82016+layer)
                block.attn.attention = CompactAttention(block.attn.attention, StateMessage(cache), optimized)
        model._compact_cache = cache
        model.register_forward_pre_hook(clear_before)
        model.register_forward_hook(clear_after, always_call=True)
    model.compact_architecture = architecture
    if sum(p.numel() for p in model.parameters()) != COUNTS[architecture]:
        raise ValueError("Compact campaign parameter contract changed")
    return model
