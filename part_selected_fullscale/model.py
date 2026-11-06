"""Compose existing numerical implementations without mutating their globals."""
import torch
from torch import nn
from part_ev_fullscale.model import build_model as native_model
from part_ev_state_speed.model import (build_model as state_model, PairCache,
    FastStateMessage, clear_cache_before, clear_cache_after)
from part_ev_message_study.model import MessageAttention
from part_layerwise_screen.model import build_model as layerwise_model
from part_selected_screen.model import build_model as selected_model
from part_selected_fast.model import build_model as fast_selected_model

CONFIGS = {
    'part_match_shared': dict(selected=False, width=599, state=False, layerwise=False, ev=False),
    'part_match_layerwise': dict(selected=False, width=601, state=False, layerwise=False, ev=False),
    'selected_match_layerwise': dict(selected=True, width=601, state=False, layerwise=False, ev=False),
    'state_shared': dict(selected=False, width=512, state=True, layerwise=False, ev=True),
    'ev_layerwise': dict(selected=False, width=512, state=False, layerwise=True, ev=True),
    'selected_shared': dict(selected=True, width=512, state=False, layerwise=False, ev=False),
    'selected_state_layerwise': dict(selected=True, width=512, state=True, layerwise=True, ev=True),
}
COUNTS = dict(zip(CONFIGS, (2323750, 2327894, 2337814, 2323710, 2212774, 2153406, 2337382)))


def widen(model, width):
    """Native FFN initialization; fork preserves the caller's training RNG."""
    with torch.random.fork_rng(devices=[]):
        for block in model.mod.blocks:
            if (block.embed_dim != 128 or block.ffn_dim != 512
                    or type(block.fc1) is not nn.Linear or type(block.fc2) is not nn.Linear
                    or type(block.post_fc_norm) is not nn.LayerNorm):
                raise ValueError('Pinned particle FFN changed')
            factory = dict(device=block.fc1.weight.device, dtype=block.fc1.weight.dtype)
            block.fc1 = nn.Linear(128, width, **factory)
            block.post_fc_norm = nn.LayerNorm(width, eps=block.post_fc_norm.eps, **factory)
            block.fc2 = nn.Linear(width, 128, **factory)
            block.ffn_dim = width


def state_branches(model):
    return [getattr(getattr(b.attn, 'attention', b.attn), 'state', None) for b in model.mod.blocks]


def add_selected_state(model, seed, optimized=True):
    cache = PairCache()
    for layer, block in enumerate(model.mod.blocks):
        # SelectedPairs retains these LayerwiseAttention objects; replace ONLY
        # their inner EV attention so selected stem binding still reaches state.
        block.attn.attention = MessageAttention(block.attn.attention, mechanism='state', seed=seed, layer=layer)
        if optimized:
            block.attn.attention.state = FastStateMessage(block.attn.attention.state, cache, chunk=64)
    model._state_pair_cache = cache
    model.register_forward_pre_hook(clear_cache_before)
    model.register_forward_hook(clear_cache_after, always_call=True)


def build_model(architecture, *, normalization=None, backend=None, provider=None,
                seed=101, trim=True, use_amp=False, optimized_state=True):
    config = CONFIGS[architecture]
    if config['selected']:
        base = 'selected_layerwise_ev' if config['state'] else 'selected_shared'
        kwargs = dict(normalization=normalization, backend=backend, seed=seed, trim=trim, use_amp=use_amp)
        model = (selected_model(base, **kwargs) if provider is None else
                 fast_selected_model(base, provider=provider, **kwargs))
        if config['state']:
            add_selected_state(model, seed, optimized_state)
    elif config['state']:
        model = state_model('state_fast', seed=seed, trim=trim, use_amp=use_amp, optimized=optimized_state)
    elif config['layerwise']:
        model = layerwise_model('part_ev_layerwise', seed=seed, trim=trim, use_amp=use_amp)
    else:
        model = native_model('part', trim=trim, use_amp=use_amp)
    if config['width'] != 512:
        widen(model, config['width'])
    model.study_architecture = architecture
    validate_model(model, architecture)
    return model


def validate_model(model, architecture):
    c = CONFIGS[architecture]
    count = sum(p.numel() for p in model.parameters())
    if count != COUNTS[architecture]:
        raise ValueError(f'{architecture}: expected {COUNTS[architecture]} parameters, found {count}')
    if (len(model.mod.blocks) != 8 or any(b.ffn_dim != c['width'] for b in model.mod.blocks)
            or any(b.ffn_dim != 512 for b in model.mod.cls_blocks)):
        raise ValueError('Particle/class FFN or depth contract changed')
    branches = state_branches(model)
    if sum(x is not None for x in branches) != (8 if c['state'] else 0):
        raise ValueError('State must be present in all eight blocks or absent')
    if c['selected']:
        pairs = model.mod.pair_embed
        if len(pairs.heads) != (8 if c['layerwise'] else 1) or pairs.ev != c['ev']:
            raise ValueError('Selected bias/value contract changed')
    elif not c['ev'] and any(type(b.attn) is not nn.MultiheadAttention for b in model.mod.blocks):
        raise ValueError('Capacity controls must retain ordinary shared-bias ParT attention')
    return dict(parameters=count, **c)
