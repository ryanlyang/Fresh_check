"""Follow-up offline factorial cells; original campaign definitions are untouched.

Shared+EV keeps Weaver's original forward/trimmer and original pair packing.
Only the value messages are added. In particular, standard pairs still use
the reference triangular/symmetric BatchNorm population, not directional pairs.
"""
from __future__ import annotations

import torch

from .attention import EdgeValueAttention
from .model import (RelationalParticleTransformer, RelationalFamilyParticleTransformer,
                    build_confirmation_architecture_model)

CONTRACT = "relational_part_offline_factorial_followup_v1"
SEEDS = (101, 202, 303)
SELECTED = ("PT", "TRACK", "REGION")
MODEL_SPECS = {
    "OFF_RPT_BASE_LAYERWISE": {"features": "standard", "bias": "layerwise", "edge_value": False},
    "OFF_RPT_BASE_SHARED_EDGEVALUE": {"features": "standard", "bias": "shared", "edge_value": True},
    "OFF_RPT_SELECTED_SHARED_EDGEVALUE": {"features": "selected", "bias": "shared", "edge_value": True},
    "OFF_RPT_SELECTED_SHARED": {"features": "selected", "bias": "shared", "edge_value": False},
}


class _BiasAndStem(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        if not isinstance(network, torch.nn.Sequential):
            raise TypeError("Shared EV requires a Sequential Weaver pair encoder")
        children = list(network.children())
        indices = [i for i, child in enumerate(children) if isinstance(child, torch.nn.Conv1d)]
        if not indices:
            raise TypeError("Shared EV requires a Conv1d final head projection")
        split = indices[-1]
        self.width = children[split].in_channels
        self.heads = children[split].out_channels
        self.prefix = torch.nn.Sequential(*children[:split])
        self.projection = torch.nn.Sequential(*children[split:])

    def forward(self, packed):
        stem = self.prefix(packed)
        return torch.cat((self.projection(stem), stem), dim=1)


class _SharedPairWithValues(torch.nn.Module):
    """Scatter bias and stem in one ORIGINAL PairEmbed call, after trimming."""
    def __init__(self, reference, edges):
        super().__init__()
        self.reference = reference
        network_name = "fts_embed" if hasattr(reference, "fts_embed") else "embed"
        network = _BiasAndStem(getattr(reference, network_name))
        self.heads, self.width = network.heads, network.width
        if reference.out_dim != self.heads:
            raise ValueError("Shared pair output width differs from attention heads")
        setattr(reference, network_name, network)
        reference.out_dim = self.heads + self.width
        object.__setattr__(self, "edges", edges)

    def forward(self, v, uu=None, mask=None):
        packed = self.reference(v, uu=uu, mask=mask)
        valid = mask.bool() if mask is not None else v.ne(0).any(dim=1, keepdim=True)
        if valid.shape != (packed.shape[0], 1, packed.shape[-1]):
            raise ValueError("Trimmed shared pair mask shape differs")
        pair_mask = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        stem = packed[:, self.heads:].masked_fill(~pair_mask, 0)
        for edge in self.edges:
            edge.bind(stem, valid[:, 0])
        return packed[:, :self.heads].contiguous()


class SharedBiasEdgeValue(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        self.families = getattr(reference, "families", ())
        pair = reference.mod.pair_embed
        network = getattr(pair, "fts_embed", getattr(pair, "embed", None))
        convolutions = [m for m in network.children() if isinstance(m, torch.nn.Conv1d)]
        if not convolutions:
            raise TypeError("Unsupported shared Weaver pair network")
        edges = []
        for block in reference.mod.blocks:
            edge = EdgeValueAttention(block.attn, relation_width=convolutions[-1].in_channels)
            block.attn = edge
            edges.append(edge)
        object.__setattr__(self, "edge_attention", edges)
        reference.mod.pair_embed = _SharedPairWithValues(pair, edges)

    def forward(self, points, features, lorentz_vectors, mask, raw_tokens=None, region_trees=None):
        try:
            if self.families:
                return self.reference(points, features, lorentz_vectors, mask, raw_tokens, region_trees)
            return self.reference(points, features, lorentz_vectors, mask)
        finally:
            for edge in self.edge_attention:
                edge.clear()


def build_ablation_model(run_id, *, normalization_artifact, region_normalization_artifact,
                         weaver_module=None):
    spec = MODEL_SPECS[run_id]
    if spec["bias"] == "layerwise":
        model = build_confirmation_architecture_model("RPT_BASE_LAYERWISE", weaver_module=weaver_module)
    else:
        if spec["features"] == "standard":
            model = RelationalParticleTransformer(weaver_module=weaver_module)
        else:
            model = RelationalFamilyParticleTransformer(
                SELECTED, normalization_artifact=normalization_artifact,
                region_normalization_artifact=region_normalization_artifact,
                weaver_module=weaver_module)
        if spec["edge_value"]:
            model = SharedBiasEdgeValue(model)
    model.run_id = run_id
    return model
