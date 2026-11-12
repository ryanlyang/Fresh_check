"""Compact state messages on DIRECTED selected pairs, never triangular packing.

The selected encoders, normalization, dense pair BatchNorm and original particle
backbone are unchanged. Only the new arm adds compact state messages. The slow
oracle repeats the shared directed projection and keeps separate output GEMMs.
"""
import torch
from torch import nn
from part_ev_compact.model import (StateMessage, CompactAttention, normalize,
                                  clear_before, clear_after)
from part_selected_screen.model import build_model as selected_model
from part_selected_fast.model import build_model as fast_model
from part_selected_fullscale.model import widen

RANK = 16
WIDTH = 570
COUNTS = {'compact': 2272870, 'matched': 2273582}


class DirectedPairCache:
    def __init__(self, projection, optimized):
        self.projection, self.optimized = projection, optimized
        self.clear()

    def clear(self):
        self.stem = self.packed = self.lookup = None

    def get(self, stem):
        if stem.ndim != 4 or stem.shape[1] != 64 or stem.shape[2] != stem.shape[3]:
            raise ValueError('Require dense directed [batch,64,query,key] stem')
        n = stem.shape[-1]
        if not self.optimized or self.stem is not stem:
            # i->j and j->i may differ. Cache ALL n*n directed projections.
            projected = self.projection(normalize(stem.permute(0, 2, 3, 1)))
            packed = projected.flatten(1, 2)
            lookup = torch.arange(n*n, device=stem.device).reshape(n, n)
            if not self.optimized:
                return packed, lookup
            self.stem, self.packed, self.lookup = stem, packed, lookup
        return self.packed, self.lookup


def build_model(architecture, *, normalization, backend, provider=None,
                seed=101, trim=True, use_amp=False, optimized=True):
    if architecture not in COUNTS:
        raise ValueError('Unknown selected compact architecture: '+architecture)
    base = 'selected_layerwise_ev' if architecture == 'compact' else 'selected_shared'
    kwargs = dict(normalization=normalization, backend=backend, seed=seed,
                  trim=trim, use_amp=use_amp)
    model = (selected_model(base, **kwargs) if provider is None else
             fast_model(base, provider=provider, **kwargs))
    if architecture == 'matched':
        widen(model, WIDTH)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+81016)
            model.shared_state_edge = nn.Linear(64, RANK, bias=False)
            nn.init.xavier_uniform_(model.shared_state_edge.weight)
            cache = DirectedPairCache(model.shared_state_edge, optimized)
            for layer, block in enumerate(model.mod.blocks):
                torch.random.default_generator.manual_seed(seed+82016+layer)
                # Retain outer wrappers: SelectedPairs binds them every forward.
                block.attn.attention = CompactAttention(
                    block.attn.attention, StateMessage(cache), optimized)
        model._compact_cache = cache
        model.register_forward_pre_hook(clear_before)
        model.register_forward_hook(clear_after, always_call=True)
    model.selected_compact_architecture = architecture
    validate_model(model, architecture)
    return model


def validate_model(model, architecture):
    compact = architecture == 'compact'
    parameters = sum(p.numel() for p in model.parameters())
    pairs = model.mod.pair_embed
    states = [getattr(b.attn.attention, 'state', None) for b in model.mod.blocks]
    if (parameters != COUNTS[architecture] or len(model.mod.blocks) != 8
            or len(pairs.heads) != (8 if compact else 1) or pairs.ev != compact
            or pairs.layerwise != compact or pairs.prefix[1].in_channels != 36
            or sum(s is not None for s in states) != (8 if compact else 0)
            or any(b.ffn_dim != (512 if compact else WIDTH) for b in model.mod.blocks)
            or any(b.ffn_dim != 512 for b in model.mod.cls_blocks)):
        raise ValueError('Selected compact parameter/architecture contract changed')
    if compact and (model.shared_state_edge.out_features != RANK or
                    any(s.receiver.out_features != RANK for s in states)):
        raise ValueError('State rank changed')
    return dict(parameters=parameters, directed_pairs=True, layerwise=compact,
                ev=compact, state_blocks=8 if compact else 0)
