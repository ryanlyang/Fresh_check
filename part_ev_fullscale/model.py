"""Official JetClass ParT and its isolated shared-bias EdgeValue extension.

The baseline is the original ParticleTransformer wrapper, using the pinned
Weaver 0.4.17 implementation. The extension changes only particle-attention
value messages: pair features, triangular packing, shared attention bias,
normalizations, sequence trimming and the class-token readout remain native.
No previous campaign artifacts or relational-model packages are imported.
"""

import hashlib
import importlib
import importlib.metadata
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


WEAVER_VERSION = "0.4.17"
WEAVER_SOURCE_SHA256 = "246be11c969fabf87a94fa2f68522da82a7f002069b370f4e4d7e3030fab1ce2"
ARCHITECTURES = ("part", "part_ev")


def reference_metadata():
    """Identity of the actual installed reference used by this experiment."""
    version = importlib.metadata.version("weaver-core")
    if version != WEAVER_VERSION:
        raise RuntimeError("This experiment requires weaver-core=={}; found {}".format(
            WEAVER_VERSION, version))
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    source = Path(module.__file__)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if checksum != WEAVER_SOURCE_SHA256:
        raise RuntimeError("Installed Weaver ParticleTransformer source differs from the pinned reference")
    return {"weaver_version": version, "weaver_particle_transformer_sha256": checksum,
            "torch_version": torch.__version__}


def reference_config(*, trim=True, use_amp=False):
    """Configuration from particle_transformer/networks/example_ParticleTransformer.py."""
    return dict(
        input_dim=17, num_classes=10, pair_input_dim=4,
        use_pre_activation_pair=False, embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8, num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[], activation="gelu", trim=bool(trim),
        for_inference=False, use_amp=bool(use_amp),
    )


class EdgeValueAttention(nn.Module):
    """Native MHA plus U_lh sum_j(a_ijh z_ij), using ONE dropout realization.

The native call returns both its ordinary output and unaveraged, post-dropout
attention weights. The extra value contribution uses those same weights and
the native output projection (without adding its bias a second time).
"""

    def __init__(self, reference, relation_width=64):
        super().__init__()
        if (not isinstance(reference, nn.MultiheadAttention) or reference.batch_first
                or reference.bias_k is not None or reference.bias_v is not None
                or reference.add_zero_attn or not reference._qkv_same_embed_dim):
            raise TypeError("Expected the pinned reference's sequence-first self-attention")
        self.reference = reference
        self.embed_dim = reference.embed_dim
        self.num_heads = reference.num_heads
        self.head_dim = reference.head_dim
        self.dropout = reference.dropout
        self.edge_projection = nn.Parameter(torch.empty(
            self.num_heads, self.head_dim, relation_width))
        nn.init.xavier_uniform_(self.edge_projection)
        self._relation_stem = None
        self._query_valid = None

    def bind(self, stem, query_valid):
        self._relation_stem = stem
        self._query_valid = query_valid

    def clear(self):
        self._relation_stem = None
        self._query_valid = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        stem, valid = self._relation_stem, self._query_valid
        if stem is None or valid is None:
            raise RuntimeError("EdgeValue attention has no pair stem bound for this forward")
        if stem.shape != (query.shape[1], self.edge_projection.shape[-1],
                          query.shape[0], key.shape[0]) or valid.shape != (query.shape[1], query.shape[0]):
            raise ValueError("Trimmed pair stem and attention sequence shapes disagree")
        ordinary, weights = self.reference(
            query, key, value, key_padding_mask=key_padding_mask, need_weights=True,
            attn_mask=attn_mask, average_attn_weights=False, is_causal=is_causal,
        )
        # Avoid the much larger [B,H,N,N,head_dim] pair-value tensor.
        aggregated = torch.einsum("bhij,bdij->bhid", weights, stem)
        heads = torch.einsum("hod,bhid->bhio", self.edge_projection, aggregated)
        joined = heads.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], self.embed_dim)
        extra = F.linear(joined, self.reference.out_proj.weight, bias=None)
        output = ordinary + extra * valid.T.unsqueeze(-1).to(extra.dtype)
        returned = weights.mean(dim=1) if average_attn_weights else weights
        return output, returned if need_weights else None


class _BiasAndStem(nn.Module):
    """Return the native shared head projection alongside its 64-channel input."""

    def __init__(self, original):
        super().__init__()
        if not isinstance(original, nn.Sequential):
            raise TypeError("Expected native sequential pair embedding")
        modules = list(original.children())
        convolution_indices = [i for i, child in enumerate(modules) if isinstance(child, nn.Conv1d)]
        split = convolution_indices[-1]
        final = modules[split]
        if final.in_channels != 64 or final.out_channels != 8:
            raise ValueError("Reference pair embedding no longer has the expected 64-to-8 projection")
        self.prefix = nn.Sequential(*modules[:split])
        # Keep the original final Conv, BatchNorm AND activation unmodified.
        self.projection = nn.Sequential(*modules[split:])

    def forward(self, packed):
        stem = self.prefix(packed)
        return torch.cat((self.projection(stem), stem), dim=1)


class _SharedPairValues(nn.Module):
    """Use native symmetric pair packing/scattering for both bias and stem.

Weaver 0.4.17 PairEmbed has no mask argument, so a native-trimmer forward hook
supplies its exact post-permutation/truncation mask. Padding is removed from
the added message only; we do not alter the reference PairEmbed BN population.
"""

    def __init__(self, reference, edges):
        super().__init__()
        if (reference.pairwise_lv_dim != 4 or reference.pairwise_input_dim != 0
                or not reference.is_symmetric or reference.remove_self_pair
                or reference.out_dim != 8):
            raise ValueError("Expected the standard symmetric four-channel reference pair encoder")
        self.reference = reference
        self.reference.embed = _BiasAndStem(reference.embed)
        self.reference.out_dim = 72
        # Each edge module is registered exactly once, inside its particle block.
        object.__setattr__(self, "_edges", tuple(edges))
        self._trimmed_mask = None

    def capture_mask(self, module, inputs, output):
        self._trimmed_mask = output[2]

    def clear(self):
        self._trimmed_mask = None
        for edge in self._edges:
            edge.clear()

    def forward(self, vectors, uu=None):
        packed = self.reference(vectors, uu=uu)
        mask = self._trimmed_mask
        if mask is None or mask.shape != (packed.shape[0], 1, packed.shape[-1]):
            raise RuntimeError("Native sequence trimmer did not provide the matching pair mask")
        valid = mask.bool()
        pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        stem = packed[:, 8:].masked_fill(~pair_valid, 0)
        for edge in self._edges:
            edge.bind(stem, valid[:, 0])
        return packed[:, :8].contiguous()


class ParticleTransformerWrapper(nn.Module):
    """Original four-input interface; `part` state_dict names match the reference."""

    def __init__(self, architecture="part", *, trim=True, use_amp=False):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError("architecture must be 'part' or 'part_ev'")
        reference_metadata()
        reference = importlib.import_module("weaver.nn.model.ParticleTransformer")
        self.architecture = architecture
        self.mod = reference.ParticleTransformer(**reference_config(trim=trim, use_amp=use_amp))
        if architecture == "part_ev":
            # Construct EVERY native weight first: the same global seed gives
            # identical shared initial weights in the two architectures. Restore
            # the CPU RNG after the new projections so training RNG also starts
            # at the same point. Models are constructed on CPU, then moved/DDP'd.
            with torch.random.fork_rng(devices=[]):
                edges = []
                for block in self.mod.blocks:
                    block.attn = EdgeValueAttention(block.attn)
                    edges.append(block.attn)
            pair = _SharedPairValues(self.mod.pair_embed, edges)
            self.mod.pair_embed = pair
            self.mod.trimmer.register_forward_hook(pair.capture_mask)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, vectors, mask):
        # points are part of the reference loader interface, not a ParT input.
        try:
            return self.mod(features, v=vectors, mask=mask)
        finally:
            if self.architecture == "part_ev":
                self.mod.pair_embed.clear()


def build_model(architecture, *, trim=True, use_amp=False):
    """Build one CPU model; caller handles device, SyncBatchNorm, DDP and seeding.

Use one model per DDP process, not nn.DataParallel (which replicates mutable
hook context). The native trimmer's Python `_counter` must be saved/restored by
the trainer because the official module does not include it in state_dict.
"""
    return ParticleTransformerWrapper(architecture, trim=trim, use_amp=use_amp)
