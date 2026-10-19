"""Separate bias heads over the unchanged shared native pair stem.

Only the final Conv1d/BatchNorm/GELU bias head is copied, not the pair encoder.
All copies start identical; their parameters and BN buffers are independent.
EV keeps its existing independent per-layer value projections. Native pair
packing, trimming, attention dropout, class attention and readout are retained.
"""
import copy
import torch
from torch import nn
from part_ev_fullscale.model import ParticleTransformerWrapper, _BiasAndStem

COUNTS = {"part_layerwise": 2147238, "part_ev_layerwise": 2212774}
BASE = {"part_layerwise": "part", "part_ev_layerwise": "part_ev"}


class LayerwiseAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self._bias = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if self._bias is None:
            raise RuntimeError("Layer-specific pair bias was not bound")
        return self.attention(query, key, value, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=self._bias,
            average_attn_weights=average_attn_weights, is_causal=is_causal)


class BiasHeads(nn.Module):
    def __init__(self, native_embed, layers, with_ev):
        super().__init__()
        split = native_embed if isinstance(native_embed, _BiasAndStem) else _BiasAndStem(native_embed)
        self.prefix = split.prefix
        self.heads = nn.ModuleList([split.projection] + [copy.deepcopy(split.projection) for _ in range(layers - 1)])
        self.with_ev = with_ev

    def forward(self, packed):
        stem = self.prefix(packed)
        outputs = [head(stem) for head in self.heads]
        if self.with_ev:
            outputs.append(stem)
        return torch.cat(outputs, dim=1)


class LayerwisePairs(nn.Module):
    def __init__(self, native, attentions, with_ev):
        super().__init__()
        self.reference = native
        self.layers, self.heads, self.with_ev = len(attentions), 8, with_ev
        if (native.pairwise_lv_dim != 4 or native.pairwise_input_dim != 0
                or not native.is_symmetric or native.remove_self_pair):
            raise ValueError("Expected the pinned symmetric four-feature pair encoder")
        native.embed = BiasHeads(native.embed, self.layers, with_ev)
        native.out_dim = self.layers * self.heads + (64 if with_ev else 0)
        object.__setattr__(self, "_attentions", tuple(attentions))
        self._trimmed_mask = None

    def capture_mask(self, module, inputs, output):
        self._trimmed_mask = output[2]

    def clear(self):
        self._trimmed_mask = None
        for block in self._attentions:
            block._bias = None
            if self.with_ev:
                block.attention.clear()

    def forward(self, vectors, uu=None):
        packed = self.reference(vectors, uu=uu)
        batch, _, count, _ = packed.shape
        if self._trimmed_mask is None or self._trimmed_mask.shape != (batch, 1, count):
            raise RuntimeError("Layerwise pairs and native trimmed mask disagree")
        valid = self._trimmed_mask[:, 0].bool()
        stem = None
        if self.with_ev:
            stem = packed[:, self.layers * self.heads:].masked_fill(
                ~(valid[:, None, :, None] & valid[:, None, None, :]), 0)
        for layer, attention in enumerate(self._attentions):
            bias = packed[:, layer*self.heads:(layer+1)*self.heads].contiguous()
            attention._bias = bias.reshape(batch*self.heads, count, count)
            if self.with_ev:
                attention.attention.bind(stem, valid)
        # The native backbone expects one mask; wrappers select their own mask.
        return packed[:, :self.heads].contiguous()


class LayerwiseParT(ParticleTransformerWrapper):
    def __init__(self, architecture, *, trim=True, use_amp=False):
        super().__init__(BASE[architecture], trim=trim, use_amp=use_amp)
        with_ev = BASE[architecture] == "part_ev"
        if with_ev:
            old_pair = self.mod.pair_embed
            native = old_pair.reference
            # Remove only the inherited pair-context hook, retaining native hooks.
            for key, callback in list(self.mod.trimmer._forward_hooks.items()):
                if getattr(callback, "__self__", None) is old_pair:
                    del self.mod.trimmer._forward_hooks[key]
        else:
            native = self.mod.pair_embed
        attentions = []
        for block in self.mod.blocks:
            block.attn = LayerwiseAttention(block.attn)
            attentions.append(block.attn)
        self.mod.pair_embed = LayerwisePairs(native, attentions, with_ev)
        self.mod.trimmer.register_forward_hook(self.mod.pair_embed.capture_mask)
        self.layerwise_architecture = architecture

    def forward(self, points, features, vectors, mask):
        try:
            return self.mod(features, v=vectors, mask=mask)
        finally:
            self.mod.pair_embed.clear()


def build_model(architecture, *, seed=101, trim=True, use_amp=False):
    # Caller seeds the unchanged reference; deepcopy adds no RNG draws.
    if architecture not in COUNTS:
        raise ValueError(f"Unknown layerwise architecture: {architecture}")
    model = LayerwiseParT(architecture, trim=trim, use_amp=use_amp)
    if sum(p.numel() for p in model.parameters()) != COUNTS[architecture]:
        raise ValueError("Layerwise parameter count differs from frozen design")
    return model
