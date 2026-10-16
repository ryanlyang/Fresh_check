"""Identity-initialized context modulation and rank-four relation value gates.

Native ParT, shared pair bias, class attention and existing EV initialization
are imported unchanged. Query-chunk checkpointing bounds gate activations;
there is never a full [batch,heads,N,N,head_dim] value tensor in production.
"""
from __future__ import annotations

import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from part_ev_fullscale.model import build_model as reference_model

COUNTS = {"part": 2143486, "part_ev": 2209022, "part_ev_context": 2258942,
          "part_ev_relation_gate": 2229758, "part_ev_context_relation_gate": 2279678}
CONFIG = {"context_hidden": 32, "gate_rank": 4, "gate_query_chunk": 16,
          "gate_checkpoint": True, "normalization_epsilon": 1e-5,
          "context_initialization": "identity", "gate_initialization": "identity"}


def mechanism_seed(seed, layer, mechanism):
    payload = f"part-ev-screen-v1:{seed}:{layer}:{mechanism}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


class JetContext(nn.Module):
    def __init__(self, width=128, hidden=32, relations=64):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-5)
        self.first = nn.Linear(width, hidden)
        self.last = nn.Linear(hidden, relations)
        nn.init.xavier_uniform_(self.first.weight)
        nn.init.zeros_(self.first.bias)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, query, valid):
        # query [N,B,C]; padding values must not enter even a learned mean.
        count = valid.sum(-1, keepdim=True)
        if bool((count == 0).any()):
            raise ValueError("Jet context requires at least one valid particle")
        pooled = query.transpose(0, 1).float().masked_fill(~valid[..., None], 0).sum(1) / count
        pooled = self.norm(pooled).to(self.first.weight.dtype)
        return 1 + self.last(F.gelu(self.first(pooled))).tanh()


class RelationGate(nn.Module):
    def __init__(self, heads=8, rank=4, channels=16, relations=64, query_chunk=16):
        super().__init__()
        self.heads, self.rank, self.channels = heads, rank, channels
        self.query_chunk = query_chunk
        self.norm = nn.LayerNorm(relations, elementwise_affine=False, eps=1e-5)
        self.coefficient = nn.Linear(relations, heads * rank)
        self.basis = nn.Parameter(torch.zeros(heads, rank, channels))
        nn.init.xavier_uniform_(self.coefficient.weight)
        nn.init.zeros_(self.coefficient.bias)

    def coefficients(self, stem, pair_valid):
        # [B,F,I,J] -> [B,H,I,J,R]; no value-channel axis on both I and J.
        features = self.norm(stem.permute(0, 2, 3, 1))
        shape = (*features.shape[:-1], self.heads, self.rank)
        coefficients = self.coefficient(features).reshape(shape).permute(0, 3, 1, 2, 4).tanh()
        return coefficients.masked_fill(~pair_valid[:, None, :, :, None], 0)

    def chunk(self, weights, stem, values, pair_valid):
        coefficients = self.coefficients(stem, pair_valid)
        basis = self.basis.tanh().to(values.dtype)
        result = torch.zeros((*weights.shape[:3], values.shape[-1]), device=values.device, dtype=values.dtype)
        for rank in range(self.rank):
            weighted = weights * coefficients[..., rank]
            result = result + torch.matmul(weighted, values) * basis[None, :, rank, None, :]
        return result / self.rank

    def forward(self, weights, stem, values, valid):
        outputs = []
        for start in range(0, weights.shape[2], self.query_chunk):
            stop = start + self.query_chunk
            pair_valid = valid[:, start:stop, None] & valid[:, None, :]
            args = (weights[:, :, start:stop], stem[:, :, start:stop], values, pair_valid)
            if self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self.chunk, *args, use_reentrant=False, preserve_rng_state=False))
            else:
                outputs.append(self.chunk(*args))
        return torch.cat(outputs, dim=2)


class ScreenAttention(nn.Module):
    def __init__(self, original, *, context, gate, seed, layer):
        super().__init__()
        self.reference = original.reference
        self.edge_projection = original.edge_projection
        self.embed_dim, self.num_heads, self.head_dim = original.embed_dim, original.num_heads, original.head_dim
        self._relation_stem = self._query_valid = None
        self.collect_diagnostics = False
        self.last_diagnostics = None
        self.context = self.gate = None
        # CPU-only construction: independent module RNGs leave backbone/training RNG intact.
        for name, enabled, factory in (("context", context, JetContext), ("gate", gate, RelationGate)):
            if enabled:
                with torch.random.fork_rng(devices=[]):
                    # manual_seed() also reseeds CUDA; this CPU factory must
                    # leave device RNGs alone, including an initialized GPU.
                    torch.random.default_generator.manual_seed(mechanism_seed(seed, layer, name))
                    setattr(self, name, factory())

    def bind(self, stem, valid):
        self._relation_stem, self._query_valid = stem, valid

    def clear(self):
        self._relation_stem = self._query_valid = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        stem, valid = self._relation_stem, self._query_valid
        if stem is None or valid is None:
            raise RuntimeError("Missing native trimmed pair/mask context")
        if query.shape != key.shape or query.shape != value.shape or stem.shape != (
                query.shape[1], 64, query.shape[0], query.shape[0]):
            raise ValueError("Screen extension supports native particle self-attention only")
        ordinary, weights = self.reference(query, key, value, key_padding_mask=key_padding_mask,
            need_weights=True, attn_mask=attn_mask, average_attn_weights=False, is_causal=is_causal)
        aggregated = torch.einsum("bhij,bdij->bhid", weights, stem)
        scale = self.context(query, valid) if self.context is not None else None
        if scale is not None:
            aggregated = aggregated * scale[:, None, None, :]
        ev = torch.einsum("hod,bhid->bhio", self.edge_projection, aggregated)
        correction = ev
        gate_delta = None
        if self.gate is not None:
            width = self.embed_dim
            bias = self.reference.in_proj_bias
            values = F.linear(value, self.reference.in_proj_weight[2*width:],
                              None if bias is None else bias[2*width:])
            values = values.reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            gate_delta = self.gate(weights, stem, values, valid)
            correction = correction + gate_delta
        joined = correction.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], self.embed_dim)
        extra = F.linear(joined, self.reference.out_proj.weight, bias=None)
        extra = extra * valid.T.unsqueeze(-1).to(extra.dtype)
        output = ordinary + extra
        if self.collect_diagnostics:
            with torch.no_grad():
                record = {"correction_to_native_norm": float(extra.float().norm() / ordinary.float().norm().clamp_min(1e-12))}
                if scale is not None:
                    record.update(context_mean=float(scale.float().mean()), context_std=float(scale.float().std(unbiased=False)),
                                  context_saturation=float(((scale < .05) | (scale > 1.95)).float().mean()))
                if self.gate is not None:
                    # Diagnostic subset only; never materialize this for the training population.
                    mask = valid[:, :8, None] & valid[:, None, :8]
                    coeff = self.gate.coefficients(stem[:, :, :8, :8], mask)
                    gates = 1 + torch.einsum("bhijr,hrd->bhijd", coeff, self.gate.basis.tanh().to(coeff.dtype)) / self.gate.rank
                    selected = gates.masked_select(mask[:, None, :, :, None].expand_as(gates)).float()
                    record.update(gate_mean=float(selected.mean()), gate_std=float(selected.std(unbiased=False)),
                                  gate_saturation=float(((selected < .05) | (selected > 1.95)).float().mean()),
                                  gate_to_ev_norm=float(gate_delta.float().norm() / ev.float().norm().clamp_min(1e-12)))
                self.last_diagnostics = record
        returned = weights.mean(1) if average_attn_weights else weights
        return output, returned if need_weights else None


def build_model(architecture, *, seed=101, trim=True, use_amp=False):
    if architecture not in COUNTS:
        raise ValueError(f"Unknown screen architecture: {architecture}")
    model = reference_model(architecture if architecture in ("part", "part_ev") else "part_ev",
                            trim=trim, use_amp=use_amp)
    if architecture not in ("part", "part_ev"):
        edges = []
        for layer, block in enumerate(model.mod.blocks):
            block.attn = ScreenAttention(block.attn, context="context" in architecture,
                gate="gate" in architecture, seed=seed, layer=layer)
            edges.append(block.attn)
        object.__setattr__(model.mod.pair_embed, "_edges", tuple(edges))
    model.screen_architecture = architecture
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != COUNTS[architecture]:
        raise ValueError(f"Parameter count changed: {actual} != {COUNTS[architecture]}")
    return model


def diagnostic_mode(model, enabled):
    for module in model.modules():
        if isinstance(module, ScreenAttention):
            module.collect_diagnostics = enabled
            module.last_diagnostics = None


def diagnostics(model):
    return {name: module.last_diagnostics for name, module in model.named_modules()
            if isinstance(module, ScreenAttention) and module.last_diagnostics is not None}
