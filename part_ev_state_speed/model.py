"""Same-math state EV optimization and two explicitly different cheaper models."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from part_ev_message_study import model as original

CONFIGS = {
    "state_fast": dict(layers=list(range(8)), rank=32, query_chunk=64),
    "state_late4": dict(layers=list(range(4, 8)), rank=32, query_chunk=64),
    "state_late4_r16": dict(layers=list(range(4, 8)), rank=16, query_chunk=64),
}
COUNTS = {name: 2209022 + len(c["layers"]) * c["rank"] * (128 + 128 + 64 + 128)
          for name, c in CONFIGS.items()}
CONFIG = dict(variants=CONFIGS, pair_normalization="shared_per_forward_with_autograd",
              checkpoint_chunks=True, norm_epsilon=1e-5, compiled=False,
              initialization="fresh_EV_plus_identity_extensions")


class PairCache:
    """Not persistent, not detached, not a parameter/buffer, never shared across jets."""

    def clear(self):
        self.stem = self.normalized = None

    def __init__(self):
        self.clear()

    def get(self, stem):
        if self.stem is not stem:
            self.stem = stem
            self.normalized = original.normalize(stem.permute(0, 2, 3, 1))
        return self.normalized


def clear_cache_before(model, _inputs):
    model._state_pair_cache.clear()


def clear_cache_after(model, _inputs, _output):
    model._state_pair_cache.clear()


class FastStateMessage(original.StateMessage):
    def __init__(self, reference, cache, chunk=64):
        nn.Module.__init__(self)
        self.receiver, self.sender, self.edge = reference.receiver, reference.sender, reference.edge
        self.output = reference.output
        self.cache, self.chunk_size = cache, chunk

    def normalized_chunk(self, weights, normalized_pairs, receiver, sender, pair_valid):
        g = self.edge(normalized_pairs)
        p, q = receiver[:, :, None], sender[:, None]
        features = F.gelu(original.normalize(g + p + q + p*q))
        features = features.masked_fill(~pair_valid[..., None], 0)
        pooled = torch.einsum("bhij,bijr->bhir", weights, features)
        return torch.einsum("hor,bhir->bhio", self.output, pooled)

    def forward(self, weights, stem, query, valid):
        normalized_pairs = self.cache.get(stem)
        states = original.normalize(query.transpose(0, 1))
        p, q = self.receiver(states), self.sender(states)
        outputs = []
        for start in range(0, weights.shape[2], self.chunk_size):
            stop = start + self.chunk_size
            pair = valid[:, start:stop, None] & valid[:, None, :]
            # Pass graph-connected tensors explicitly: recomputation after the
            # outer forward has cleared the cache must not consult mutable state.
            outputs.append(original.chunk_call(self, self.normalized_chunk,
                weights[:, :, start:stop], normalized_pairs[:, start:stop],
                p[:, start:stop], q, pair))
        return torch.cat(outputs, 2)


def build_model(architecture, *, seed=101, trim=True, use_amp=False, optimized=True):
    config = CONFIGS[architecture]
    model = original.build_model("ev_state", seed=seed, trim=trim, use_amp=use_amp)
    cache = PairCache()
    for layer, block in enumerate(model.mod.blocks):
        if layer not in config["layers"]:
            block.attn.state = None
            continue
        state = block.attn.state
        if config["rank"] != 32:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(original.mechanism_seed(seed, layer, "state"))
                state = original.StateMessage(rank=config["rank"])
        block.attn.state = FastStateMessage(state, cache, config["query_chunk"]) if optimized else state
    model._state_pair_cache = cache
    # Always clear on success AND exceptions; backward uses explicit saved args.
    model.register_forward_pre_hook(clear_cache_before)
    model.register_forward_hook(clear_cache_after, always_call=True)
    model.study_architecture = architecture
    if sum(p.numel() for p in model.parameters()) != COUNTS[architecture]:
        raise ValueError("State speed parameter contract changed")
    return model


diagnostic_mode = original.diagnostic_mode
diagnostics = original.diagnostics
