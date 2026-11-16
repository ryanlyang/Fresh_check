"""Explicit final-block pair readout, preserving the pinned native backbone.

Pair summaries are returned tensors, never mutable hooks/caches. The first seven
particle blocks and both class blocks use their original forwards unchanged.
"""
from functools import partial
import math
import torch
from torch import nn
from torch.nn import functional as F
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from part_ev_fullscale.model import reference_metadata
from part_ev_state_compare.model import build_model as reference_model, state_branches
from part_ev_message_study.model import normalize, chunk_call, mechanism_seed
from . import pooling

PARAMETERS = 2349702
BASE_PARAMETERS = 2327462
CONFIG = dict(base='state_layerwise', state_blocks=list(range(8)), state_width=32,
    source_block=7, queries=2, key_width=16, value_width=32, fusion_hidden=64,
    query_chunk=64, checkpoint_chunks=True, pooling='global_directed_offdiagonal_softmax',
    precision='fp32_norm_scores_pool_merge_fp16_autocast_linears',
    initialization='zero_fusion_output_isolated_rng', norm_epsilon=1e-5,
    relations='standard', compact=False, pair_bias='layerwise')


def readout_key(name):
    return name.startswith('mod.pair_readout.')


class PairReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(2, 16))
        self.key = nn.Linear(32, 16, bias=False)
        self.value = nn.Linear(32, 32, bias=False)
        self.hidden = nn.Linear(192, 64)
        self.output = nn.Linear(64, 128)
        with torch.no_grad():
            self.queries.mul_(math.sqrt(16) / self.queries.norm(dim=-1, keepdim=True))
        for layer in (self.key, self.value, self.hidden):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def project(self, features, valid, *, diagnostic=False):
        u = normalize(features.masked_fill(~valid[..., None], 0))
        keys, values = self.key(u), self.value(u)
        dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        with torch.autocast(u.device.type, enabled=False):
            keys, values = keys.to(dtype), values.to(dtype)
            scores = torch.einsum('ar,bijr->baij', self.queries.to(dtype), keys) / 4
            health = torch.isfinite(keys).logical_or(~valid[..., None]).all()
            health = health & torch.isfinite(u).logical_or(~valid[..., None]).all()
        result = (scores.flatten(2), values.flatten(1, 2), health)
        return (*result, keys.flatten(1, 2)) if diagnostic else result

    def fuse(self, h, summaries, has_pairs):
        hidden = F.gelu(self.hidden(torch.cat((h, summaries.flatten(1)), -1)))
        delta = self.output(hidden).to(h.dtype) * has_pairs[:, None]
        health = torch.isfinite(hidden).all() & torch.isfinite(delta).all()
        return h + delta, delta, health


def joint_chunk(state, readout, weights, pairs, p, q, pair_valid, readout_valid, *, diagnostic=False):
    g = state.edge(pairs)
    p, q = p[:, :, None], q[:, None]
    r = F.gelu(normalize(g+p+q+p*q)).masked_fill(~pair_valid[..., None], 0)
    pooled = torch.einsum('bhij,bijr->bhir', weights, r)
    current = torch.einsum('hor,bhir->bhio', state.output, pooled)
    projected = readout.project(r, readout_valid, diagnostic=diagnostic)
    scores, values, healthy = projected[:3]
    present, maximum, mass, numerator, ok = pooling.statistics(scores, values, readout_valid.flatten(1))
    result = (current, present, maximum, mass, numerator, healthy & ok)
    return (*result, scores, values, projected[3]) if diagnostic else result


def state_with_readout(state, readout, weights, stem, query, valid, *, chunk, checkpointed, diagnostic=False):
    pairs = state.cache.get(stem)
    states = normalize(query.transpose(0, 1))
    p, q = state.receiver(states), state.sender(states)
    current, combined, records = [], None, []
    count = query.shape[0]
    sender = torch.arange(count, device=query.device)
    call = partial(joint_chunk, state, readout, diagnostic=diagnostic)
    for start in range(0, count, chunk):
        stop = min(count, start+chunk)
        pair = valid[:, start:stop, None] & valid[:, None, :]
        offdiag = torch.arange(start, stop, device=query.device)[:, None] != sender[None]
        args = (weights[:, :, start:stop], pairs[:, start:stop], p[:, start:stop], q, pair, pair & offdiag)
        row = chunk_call(state, call, *args) if checkpointed else call(*args)
        current.append(row[0])
        combined = row[1:6] if combined is None else pooling.merge(combined, row[1:6])
        if diagnostic: records.append((*row[6:], (pair & offdiag).flatten(1)))
    summary, healthy = pooling.finish(combined)
    diagnostics = None
    if diagnostic:
        from .diagnostics import projection_statistics
        diagnostics = projection_statistics(
            torch.cat([r[0] for r in records], 2), torch.cat([r[1] for r in records], 1),
            torch.cat([r[2] for r in records], 1), torch.cat([r[3] for r in records], 1))
    return torch.cat(current, 2), summary, healthy, diagnostics


def final_attention(block, query, padding_mask, readout, *, chunk, checkpointed, diagnostic=False):
    layer, attention = block.attn, block.attn.attention
    stem, valid = attention._relation_stem, attention._query_valid
    if stem is None or valid is None or layer._bias is None:
        raise ValueError('Missing native trimmed layerwise pair context')
    ordinary, weights = attention.reference(query, query, query, key_padding_mask=padding_mask,
        need_weights=True, attn_mask=layer._bias, average_attn_weights=False, is_causal=False)
    pooled = torch.einsum('bhij,bdij->bhid', weights, stem)
    ev = torch.einsum('hod,bhid->bhio', attention.edge_projection, pooled)
    state, summary, healthy, diagnostics = state_with_readout(attention.state, readout, weights, stem, query,
        valid, chunk=chunk, checkpointed=checkpointed, diagnostic=diagnostic)
    joined = (ev+state).permute(2, 0, 1, 3).reshape(query.shape)
    extra = F.linear(joined, attention.reference.out_proj.weight, bias=None)
    extra = extra * valid.T.unsqueeze(-1).to(extra.dtype)
    return ordinary+extra, summary, healthy, diagnostics


def final_block(block, x, padding_mask, readout, *, chunk, checkpointed, diagnostic=False):
    """Pinned Block.forward order, including its nontrivial c_attn layout."""
    residual = x
    x = block.pre_attn_norm(x)
    x, summary, healthy, diagnostics = final_attention(block, x, padding_mask, readout,
        chunk=chunk, checkpointed=checkpointed, diagnostic=diagnostic)
    if block.c_attn is not None:
        tgt_len = x.size(0)
        x = x.view(tgt_len, -1, block.num_heads, block.head_dim)
        x = torch.einsum('tbhd,h->tbdh', x, block.c_attn)
        x = x.reshape(tgt_len, -1, block.embed_dim)
    if block.post_attn_norm is not None: x = block.post_attn_norm(x)
    x = block.dropout(x)
    x += residual
    residual = x
    x = block.pre_fc_norm(x)
    x = block.act(block.fc1(x))
    x = block.act_dropout(x)
    if block.post_fc_norm is not None: x = block.post_fc_norm(x)
    x = block.fc2(x)
    x = block.dropout(x)
    if block.w_resid is not None: residual = torch.mul(block.w_resid, residual)
    x += residual
    return x, summary, healthy, diagnostics


class PairReadoutTransformer(ParticleTransformer):
    def forward_details(self, x, v=None, mask=None, uu=None, uu_idx=None, *, diagnostic=False):
        if uu is not None or uu_idx is not None or self.for_inference or self.fc is None:
            raise ValueError('Readout experiment requires standard relations and classification logits')
        if diagnostic and (self.training or torch.is_grad_enabled() or x.shape[0] > 8 or x.shape[-1] > 32):
            raise ValueError('Dense diagnostics require small isolated no-grad eval inputs')
        try:
            with torch.no_grad():
                x, v, mask, uu = self.trimmer(x, v, mask, uu)
                padding_mask = ~mask.squeeze(1)
                has_pairs = mask.squeeze(1).sum(-1) >= 2
            with torch.autocast(x.device.type, enabled=self.use_amp):
                x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)
                bias = self.pair_embed(v, uu)
                for block in self.blocks[:-1]:
                    x = block(x, padding_mask=padding_mask, attn_mask=bias)
                x, summary, healthy, diagnostics = final_block(self.blocks[-1], x, padding_mask, self.pair_readout,
                    chunk=self.readout_chunk, checkpointed=self.readout_checkpointed, diagnostic=diagnostic)
                cls_tokens = self.cls_token.expand(1, x.size(1), -1)
                for block in self.cls_blocks:
                    cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
                h = self.norm(cls_tokens).squeeze(0)
                fused, delta, ok = self.pair_readout.fuse(h, summary, has_pairs)
                logits = self.fc(fused)
                # No collectives here: eval runs on rank zero alone. Poison failed
                # health so the driver's existing coordinated finite-loss check
                # rejects all ranks BEFORE backward; rank-zero eval also checks.
                logits = torch.where(healthy & ok, logits, torch.full_like(logits, float('nan')))
                return logits, (h, summary, delta, has_pairs, healthy & ok, diagnostics)
        finally:
            self.pair_embed.clear()

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        return self.forward_details(x, v, mask, uu, uu_idx)[0]


def check_inputs(_model, inputs):
    mask = inputs[3]
    if mask.ndim != 3 or mask.shape[1] != 1 or mask.shape[2] < 1 or not mask.bool().any(-1).all().item():
        raise ValueError('Each jet must have at least one valid particle')


def build_model(*, seed=101, trim=True, use_amp=False, chunk=64, checkpointed=True):
    if type(chunk) is not int or chunk < 1: raise ValueError('Positive query chunk required')
    reference_metadata()
    model = reference_model('state_layerwise', seed=seed, trim=trim, use_amp=use_amp)
    if type(model.mod) is not ParticleTransformer: raise ValueError('Expected pinned native transformer')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(mechanism_seed(seed, 7, 'pair-readout-v1'))
        model.mod.pair_readout = PairReadout()
    model.mod.readout_chunk, model.mod.readout_checkpointed = chunk, checkpointed
    model.mod.__class__ = PairReadoutTransformer
    model.register_forward_pre_hook(check_inputs)
    model.study_architecture = 'state_pair_readout_layerwise'
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('Pair-readout parameter contract changed')
    return model


def output_parameters(model):
    return list(model.mod.pair_readout.output.parameters())


def upstream_parameters(model):
    return [p for n, p in model.named_parameters() if readout_key(n) and '.output.' not in n]
