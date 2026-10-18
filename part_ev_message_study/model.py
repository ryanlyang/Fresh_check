"""Identity-initialized message, routing and moment extensions of native EV."""
from __future__ import annotations

import copy
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from part_ev_fullscale.model import build_model as reference_model

COUNTS = dict(ev_state=2323710, ev_separate=2218054,
              ev_state_separate=2332742, ev_routing=2213118, ev_variance=2274558)
CONFIG = dict(rank=32, query_chunk=16, checkpoint_chunks=True, norm_epsilon=1e-5,
              routing_bound=2., initialization="fresh_EV_plus_identity_extensions",
              variance="FP32_normalized_post_dropout_diagonal_moment",
              routing="native_logits_plus_bounded_correction_same_dropout")


def normalize(x):
    # Double is retained for mathematical gradient tests; real parameters are FP32.
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(x.device.type, enabled=False):
        return F.layer_norm(x.to(dtype), (x.shape[-1],), eps=CONFIG["norm_epsilon"])


def mechanism_seed(seed, layer, name):
    value = f"ev-message-study-v1:{seed}:{layer}:{name}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63)


def chunk_call(module, fn, *args):
    if module.training and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
    return fn(*args)


class StateMessage(nn.Module):
    def __init__(self, width=128, relations=64, rank=32, heads=8, channels=16, chunk=16):
        super().__init__()
        self.receiver = nn.Linear(width, rank, bias=False)
        self.sender = nn.Linear(width, rank, bias=False)
        self.edge = nn.Linear(relations, rank, bias=False)
        self.output = nn.Parameter(torch.zeros(heads, channels, rank))
        self.chunk_size = chunk
        for layer in (self.receiver, self.sender, self.edge):
            nn.init.xavier_uniform_(layer.weight)

    def features(self, stem, receiver, sender, pair_valid):
        g = self.edge(normalize(stem.permute(0, 2, 3, 1)))
        p, q = receiver[:, :, None], sender[:, None]
        r = F.gelu(normalize(g + p + q + p*q))
        return r.masked_fill(~pair_valid[..., None], 0)

    def chunk(self, weights, stem, receiver, sender, pair_valid):
        r = self.features(stem, receiver, sender, pair_valid)
        pooled = torch.einsum("bhij,bijr->bhir", weights, r)
        return torch.einsum("hor,bhir->bhio", self.output, pooled)

    def forward(self, weights, stem, query, valid):
        states = normalize(query.transpose(0, 1))
        p, q = self.receiver(states), self.sender(states)
        outputs = []
        for start in range(0, weights.shape[2], self.chunk_size):
            stop = start + self.chunk_size
            pair = valid[:, start:stop, None] & valid[:, None, :]
            outputs.append(chunk_call(self, self.chunk, weights[:, :, start:stop],
                                      stem[:, :, start:stop], p[:, start:stop], q, pair))
        return torch.cat(outputs, 2)


class VarianceMessage(nn.Module):
    def __init__(self, heads=8, channels=16, relations=64, chunk=16):
        super().__init__()
        self.output = nn.Parameter(torch.zeros(heads, channels, relations))
        self.chunk_size = chunk

    @staticmethod
    def moments(weights, stem, pair_valid):
        with torch.autocast(stem.device.type, enabled=False):
            z = normalize(stem.permute(0, 2, 3, 1))
            w = weights.to(z.dtype).masked_fill(~pair_valid[:, None], 0)
            total = w.sum(-1, keepdim=True)
            pi = w / torch.where(total > 0, total, torch.ones_like(total))
            mean = torch.einsum("bhij,bijd->bhid", pi, z)
            second = torch.einsum("bhij,bijd->bhid", pi, z.square())
            return (second - mean.square()).clamp_min(0)

    def chunk(self, weights, stem, pair_valid):
        variance = self.moments(weights, stem, pair_valid)
        return torch.einsum("hod,bhid->bhio", self.output, variance)

    def forward(self, weights, stem, valid):
        outputs = []
        for start in range(0, weights.shape[2], self.chunk_size):
            stop = start + self.chunk_size
            pair = valid[:, start:stop, None] & valid[:, None, :]
            outputs.append(chunk_call(self, self.chunk, weights[:, :, start:stop],
                                      stem[:, :, start:stop], pair))
        return torch.cat(outputs, 2)


class ValueRouting(nn.Module):
    def __init__(self, relations=64, heads=8, chunk=16):
        super().__init__()
        self.projection = nn.Linear(relations, heads, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.chunk_size = chunk

    def chunk(self, weights, stem, q, k, mask):
        # Reproduce need_weights=True MHA's score arithmetic (including AMP).
        scores = torch.baddbmm(mask, q * math.sqrt(1. / q.shape[-1]), k.transpose(1, 2))
        alpha = scores.softmax(-1).reshape_as(weights)
        dtype = torch.float64 if weights.dtype == torch.float64 else torch.float32
        delta = CONFIG["routing_bound"] * self.projection(
            normalize(stem.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).to(dtype).tanh()
        with torch.autocast(weights.device.type, enabled=False):
            dtype = torch.float64 if weights.dtype == torch.float64 else torch.float32
            alpha = alpha.to(dtype)
            # Remove softmax rounding in the row sum. expm1 form makes delta=0
            # exactly identity, including FP16 native softmax rounding.
            alpha = alpha / alpha.sum(-1, keepdim=True)
            increment = torch.expm1(delta.to(dtype))
            normalizer = 1 + (alpha * increment).sum(-1, keepdim=True)
            adjusted = weights.to(dtype) * (1 + increment) / normalizer
        return adjusted.to(weights.dtype)

    def forward(self, weights, stem, query, key, value, attention, key_padding_mask, attn_mask):
        if query is not key or key is not value:
            raise ValueError("Value routing requires native self-attention")
        q, k, _ = F._in_projection_packed(query, key, value,
                                         attention.in_proj_weight, attention.in_proj_bias)
        n, batch, _ = query.shape
        heads, channels = attention.num_heads, attention.head_dim
        q = q.reshape(n, batch*heads, channels).transpose(0, 1)
        k = k.reshape(n, batch*heads, channels).transpose(0, 1)
        mask = torch.zeros(batch*heads, 1, n, device=query.device, dtype=query.dtype)
        if key_padding_mask is not None:
            padding = key_padding_mask
            if padding.dtype == torch.bool:
                padding = torch.zeros_like(padding, dtype=query.dtype).masked_fill(padding, -torch.inf)
            mask = mask + padding[:, None, None, :].expand(batch, heads, 1, n).reshape(batch*heads, 1, n)
        if attn_mask is not None:
            additive = attn_mask
            if additive.dtype == torch.bool:
                additive = torch.zeros_like(additive, dtype=query.dtype).masked_fill(additive, -torch.inf)
            mask = mask + additive
        mask = mask.expand(batch*heads, n, n)
        outputs = []
        for start in range(0, n, self.chunk_size):
            stop = start + self.chunk_size
            outputs.append(chunk_call(self, self.chunk, weights[:, :, start:stop],
                stem[:, :, start:stop], q[:, start:stop], k, mask[:, start:stop]))
        return torch.cat(outputs, 2)


class SeparateStem(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.prefix, self.projection = original.prefix, original.projection
        self.value_prefix = copy.deepcopy(original.prefix)
        self.collect_diagnostics = False
        self.last_diagnostics = None

    def forward(self, packed):
        bias, value = self.prefix(packed), self.value_prefix(packed)
        if self.collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = dict(
                    packed_value_to_bias_difference=float((value.float()-bias.float()).norm()/bias.float().norm().clamp_min(1e-12)),
                    independent_storage=all(a.data_ptr() != b.data_ptr() for a,b in
                        zip(self.prefix.parameters(), self.value_prefix.parameters())))
        return torch.cat((self.projection(bias), value), dim=1)


class MessageAttention(nn.Module):
    def __init__(self, original, *, mechanism, seed, layer):
        super().__init__()
        self.reference, self.edge_projection = original.reference, original.edge_projection
        self.embed_dim, self.num_heads, self.head_dim = original.embed_dim, original.num_heads, original.head_dim
        self._relation_stem = self._query_valid = None
        self.collect_diagnostics = self.disable_new = False
        self.last_diagnostics = None
        self.state = self.routing = self.variance = None
        if mechanism is not None:
            factory = dict(state=StateMessage, routing=ValueRouting, variance=VarianceMessage)[mechanism]
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(mechanism_seed(seed, layer, mechanism))
                setattr(self, mechanism, factory())

    def bind(self, stem, valid):
        self._relation_stem, self._query_valid = stem, valid

    def clear(self):
        self._relation_stem = self._query_valid = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        stem, valid = self._relation_stem, self._query_valid
        if stem is None or valid is None or is_causal:
            raise ValueError("Requires bound, noncausal native particle attention")
        if stem.shape != (query.shape[1], 64, query.shape[0], key.shape[0]):
            raise ValueError("Trimmed pair features do not match particle states")
        ordinary, weights = self.reference(query, key, value, key_padding_mask=key_padding_mask,
            need_weights=True, attn_mask=attn_mask, average_attn_weights=False, is_causal=False)
        routed = weights
        if self.routing is not None and not self.disable_new:
            routed = self.routing(weights, stem, query, key, value, self.reference, key_padding_mask, attn_mask)
        pooled = torch.einsum("bhij,bdij->bhid", routed, stem)
        ev = torch.einsum("hod,bhid->bhio", self.edge_projection, pooled)
        delta = torch.zeros_like(ev)
        if self.state is not None and not self.disable_new:
            delta = self.state(weights, stem, query, valid)
        if self.variance is not None and not self.disable_new:
            delta = self.variance(weights, stem, valid)
        joined = (ev+delta).permute(2, 0, 1, 3).reshape(query.shape)
        extra = F.linear(joined, self.reference.out_proj.weight, bias=None)
        extra = extra * valid.T.unsqueeze(-1).to(extra.dtype)
        if self.collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = dict(
                    new_to_ev_norm=float(delta.float().norm()/ev.float().norm().clamp_min(1e-12)),
                    correction_to_native_norm=float(extra.float().norm()/ordinary.float().norm().clamp_min(1e-12)),
                    routing_weight_mean_absolute_change=float((routed.float()-weights.float()).abs().mean()))
                if self.state is not None:
                    self.last_diagnostics["state_output_parameter_norm"] = float(self.state.output.float().norm())
                    # Small, fixed subset. Only this branch's endpoints change;
                    # attention, pair features and native output stay fixed.
                    small = min(8, query.shape[1])
                    x, v = query[:, :small], valid[:small]
                    mean = x.transpose(0, 1).float().masked_fill(~v[..., None], 0).sum(1) / v.sum(1, keepdim=True)
                    pooled = mean.unsqueeze(0).expand_as(x)
                    altered = self.state(weights[:small], stem[:small], pooled, v)
                    self.last_diagnostics["pooled_endpoint_message_difference"] = float(
                        (altered.float()-delta[:small].float()).norm()/delta[:small].float().norm().clamp_min(1e-12))
                if self.variance is not None:
                    self.last_diagnostics["variance_output_parameter_norm"] = float(self.variance.output.float().norm())
                    v = valid[:8]
                    moments = self.variance.moments(weights[:8], stem[:8], v[:,:,None]&v[:,None,:])
                    self.last_diagnostics["weighted_variance_mean"] = float(moments.masked_select(v[:,None,:,None]).mean())
                    self.last_diagnostics["zero_weight_rows"] = int(((weights.sum(-1) == 0)&valid[:,None]).sum())
                if self.routing is not None:
                    self.last_diagnostics["routing_parameter_norm"] = float(self.routing.projection.weight.float().norm())
                    a, b = weights[:8].float(), routed[:8].float()
                    v = valid[:8,None]
                    self.last_diagnostics["routing_total_variation"] = float(
                        (.5*(a-b).abs().sum(-1)).masked_select(v).mean())
                    for name, distribution in (("native",a),("value",b)):
                        entropy = -(distribution*distribution.clamp_min(1e-12).log()).sum(-1)
                        self.last_diagnostics[name+"_entropy"] = float(entropy.masked_select(v).mean())
        returned = weights.mean(1) if average_attn_weights else weights
        return ordinary+extra, returned if need_weights else None


def build_model(architecture, *, seed=101, trim=True, use_amp=False):
    if architecture not in COUNTS:
        raise ValueError(f"Unknown message study architecture: {architecture}")
    model = reference_model("part_ev", trim=trim, use_amp=use_amp)
    if "separate" in architecture:
        pair = model.mod.pair_embed.reference
        pair.embed = SeparateStem(pair.embed)
    mechanism = next((m for m in ("state", "routing", "variance") if m in architecture), None)
    edges = []
    for layer, block in enumerate(model.mod.blocks):
        block.attn = MessageAttention(block.attn, mechanism=mechanism, seed=seed, layer=layer)
        edges.append(block.attn)
    object.__setattr__(model.mod.pair_embed, "_edges", tuple(edges))
    model.study_architecture = architecture
    if sum(p.numel() for p in model.parameters()) != COUNTS[architecture]:
        raise ValueError("Model parameter contract changed")
    return model


def diagnostic_mode(model, enabled):
    for module in model.modules():
        if isinstance(module, (MessageAttention, SeparateStem)):
            module.collect_diagnostics = enabled
            module.last_diagnostics = None


def diagnostics(model):
    return {name: module.last_diagnostics for name, module in model.named_modules()
            if isinstance(module, (MessageAttention, SeparateStem)) and module.last_diagnostics is not None}
