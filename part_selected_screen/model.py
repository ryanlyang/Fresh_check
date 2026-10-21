"""Selected features with shared/layerwise bias and unchanged per-layer EV.

Directional pair descriptors require dense pair processing. In particular the
pair BatchNorm population is directed, unlike the native triangular control.
This is recorded as part of the architectural intervention, not hidden parity.
Particle features/trimming/blocks/class attention and the training RNG stay native.
"""
import copy
import torch
from torch import nn
from part_ev_fullscale.model import ParticleTransformerWrapper, _BiasAndStem
from part_layerwise_screen.model import LayerwiseAttention
from .data import raw_tokens
from .features import Relations

CONFIGS = {'selected_shared': (False, False), 'selected_shared_ev': (False, True),
           'selected_layerwise': (True, False), 'selected_layerwise_ev': (True, True)}


class SelectedPairs(nn.Module):
    def __init__(self, native, attentions, normalization, backend, layerwise, ev):
        super().__init__()
        split = native.embed if isinstance(native.embed, _BiasAndStem) else _BiasAndStem(native.embed)
        self.prefix = split.prefix
        self.heads = nn.ModuleList([split.projection] +
            [copy.deepcopy(split.projection) for _ in range(7 if layerwise else 0)])
        self.pair_features = native.pairwise_lv_fts
        self.ev, self.layerwise = ev, layerwise
        old_bn, old_conv = self.prefix[0], self.prefix[1]
        if not isinstance(old_bn, nn.BatchNorm1d) or old_conv.in_channels != 4:
            raise ValueError('Pinned pair stem changed')
        self.prefix[0] = nn.BatchNorm1d(36, eps=old_bn.eps, momentum=old_bn.momentum)
        self.prefix[1] = nn.Conv1d(36, 64, 1)
        with torch.no_grad():
            for name in ('weight', 'bias', 'running_mean', 'running_var'):
                getattr(self.prefix[0], name)[:4].copy_(getattr(old_bn, name))
            self.prefix[0].num_batches_tracked.copy_(old_bn.num_batches_tracked)
            self.prefix[1].weight[:, :4].copy_(old_conv.weight)
            self.prefix[1].bias.copy_(old_conv.bias)
        self.relations = Relations(normalization, backend)
        object.__setattr__(self, '_attentions', tuple(attentions))
        self._raw = self._mask = None

    def capture(self, module, inputs, outputs):
        x, v, mask, uu = outputs
        self._raw = raw_tokens(x, v, mask)
        self._mask = mask
        # This hook runs under native no_grad. Only raw information is captured;
        # all learned relation networks run later in pair_embed.forward.
        return x[:, :17].contiguous(), v, mask, uu

    def forward(self, vectors, uu=None):
        if uu is not None or self._raw is None or self._mask is None:
            raise ValueError('Selected pair context missing or unexpected external edges')
        b, _, n = vectors.shape
        with torch.no_grad(), torch.autocast(device_type=vectors.device.type, enabled=False):
            standard = self.pair_features(vectors.float().unsqueeze(-1), vectors.float().unsqueeze(-2))
        extra = self.relations(vectors, self._raw, self._mask)
        stem = self.prefix(torch.cat((standard, extra), 1).reshape(b, 36, n*n))
        biases = [head(stem).reshape(b*8, n, n) for head in self.heads]
        valid = self._mask[:, 0].bool()
        message = stem.reshape(b, 64, n, n).masked_fill(~(valid[:, None, :, None] & valid[:, None, None, :]), 0)
        for i, attention in enumerate(self._attentions):
            attention._bias = biases[i if self.layerwise else 0]
            if self.ev:
                attention.attention.bind(message, valid)
        return biases[0].reshape(b, 8, n, n)

    def clear(self):
        self._raw = self._mask = None
        for attention in self._attentions:
            attention._bias = None
            if self.ev:
                attention.attention.clear()


class SelectedParT(ParticleTransformerWrapper):
    def __init__(self, architecture, normalization, backend, *, seed=101, trim=True, use_amp=False):
        layerwise, ev = CONFIGS[architecture]
        super().__init__('part_ev' if ev else 'part', trim=trim, use_amp=use_amp)
        if ev:
            old = self.mod.pair_embed
            native = old.reference
            for key, hook in list(self.mod.trimmer._forward_hooks.items()):
                if getattr(hook, '__self__', None) is old:
                    del self.mod.trimmer._forward_hooks[key]
        else:
            native = self.mod.pair_embed
        attentions = []
        for block in self.mod.blocks:
            block.attn = LayerwiseAttention(block.attn)
            attentions.append(block.attn)
        with torch.random.fork_rng(devices=[]):
            # CPU construction only: do not reset an already initialized CUDA RNG.
            torch.random.default_generator.manual_seed(seed + 71031)
            self.mod.pair_embed = SelectedPairs(native, attentions, normalization, backend, layerwise, ev)
        self.mod.trimmer.register_forward_hook(self.mod.pair_embed.capture)

    def forward(self, points, features, vectors, mask):
        try:
            return self.mod(features, v=vectors, mask=mask)
        finally:
            self.mod.pair_embed.clear()


def build_model(architecture, *, normalization, backend, seed=101, trim=True, use_amp=False):
    if architecture not in CONFIGS:
        raise ValueError(f'Unknown selected configuration: {architecture}')
    return SelectedParT(architecture, normalization, backend, seed=seed, trim=trim, use_amp=use_amp)
