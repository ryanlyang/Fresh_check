"""Hierarchy-aware dual-stream Particle Transformer tagger for ABPH Step 10.

The trusted HLT branch remains an independently executable classifier.  Pseudo
particles and hierarchy memory can only modify that branch through explicitly
scaled residual paths, so setting the ReZero parameters to zero recovers the
HLT checkpoint exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Keep cache/report utilities importable without the training stack.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - research training requires torch
    torch = None
    nn = None

if nn is None:  # pragma: no cover - permits metadata/cache-only imports
    class _ModuleBase:
        pass
else:
    _ModuleBase = nn.Module

from .prediction_cache import DeployablePseudoViewBatch
from .schemas import ABPH_MAX_PARTICLES


ABPH_HIERARCHY_TAGGER_CONTRACT = "adaptive_binary_hierarchy_tagger_v1"
ABPH_PRIMARY_FUSION_LOCATIONS: tuple[int, ...] = (4, 8, 12)
ABPH_PRIMARY_REZERO_INIT = 1.0e-3
ABPH_PRIMARY_HIERARCHY_NAMES: tuple[str, ...] = (
    "exclusive_kt",
    "cambridge_aachen",
)


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("ABPH hierarchy-aware tagging requires PyTorch")
    return torch


def _masked_mean(values: Any, mask: Any, dim: int) -> Any:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def _safe_key_padding(tokens: Any, valid_mask: Any) -> tuple[Any, Any]:
    """Give all-empty rows one zero-valued dummy key to avoid attention NaNs."""

    safe_tokens = tokens
    safe_valid = valid_mask.bool()
    empty = ~safe_valid.any(dim=1)
    if bool(empty.any()):
        safe_tokens = tokens.clone()
        safe_valid = safe_valid.clone()
        safe_tokens[empty, 0] = 0.0
        safe_valid[empty, 0] = True
    return safe_tokens, ~safe_valid


def _attention_mask_for_mha(bias: Any | None) -> Any | None:
    if bias is None:
        return None
    if bias.ndim != 4:
        raise ValueError("attention bias must have shape [B, H, Q, K]")
    return bias.reshape(bias.shape[0] * bias.shape[1], bias.shape[2], bias.shape[3])


def _compatible_padding_mask(padding: Any, bias: Any | None, dtype: Any) -> Any:
    if bias is None:
        return padding
    return torch.zeros_like(padding, dtype=dtype).masked_fill(padding, float("-inf"))


def _batched_gather(values: Any, indices: Any) -> tuple[Any, Any]:
    """Gather ``values[B,N,...]`` with signed ``indices[B,P]``."""

    if values.ndim < 3 or indices.ndim != 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("batched gather received incompatible shapes")
    valid = (indices >= 0) & (indices < values.shape[1])
    safe = indices.clamp(0, max(int(values.shape[1]) - 1, 0)).long()
    gather_shape = (safe.shape[0], safe.shape[1]) + tuple(values.shape[2:])
    gather_index = safe.reshape(safe.shape + (1,) * (values.ndim - 2)).expand(gather_shape)
    gathered = torch.gather(values, 1, gather_index)
    gathered = gathered * valid.reshape(valid.shape + (1,) * (values.ndim - 2)).to(gathered.dtype)
    return gathered, valid


def _tensor_sha256(value: Any) -> str:
    array = value.detach().to("cpu").contiguous().numpy()
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _combined_tensor_sha256(*values: Any) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(_tensor_sha256(value).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


@dataclass(frozen=True)
class PseudoViewInputs:
    """Torch view of the stable Step-9 deployable pseudo cache schema."""

    arrays: Mapping[str, Any]
    view_names: tuple[str, ...]
    hierarchy_names: tuple[str, ...]
    frontier_depths: Mapping[str, int]
    diagnostics: Mapping[str, Any]

    @classmethod
    def from_deployable_batch(
        cls,
        batch: DeployablePseudoViewBatch,
        *,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> "PseudoViewInputs":
        _require_torch()
        batch.validate()
        converted: dict[str, Any] = {}
        for name, value in batch.arrays.items():
            tensor = torch.as_tensor(value, device=device)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            converted[str(name)] = tensor
        return cls(
            arrays=converted,
            view_names=tuple(batch.view_names),
            hierarchy_names=tuple(batch.hierarchy_names),
            frontier_depths={str(k): int(v) for k, v in batch.frontier_depths.items()},
            diagnostics=dict(batch.diagnostics),
        )

    def validate(self) -> None:
        if self.diagnostics.get("offline_inputs_loaded") is not False:
            raise ValueError("deployable tagger input must be HLT-only")
        if self.diagnostics.get("teacher_logits_loaded") is not False:
            raise ValueError("deployable tagger input cannot load teacher logits")
        if self.diagnostics.get("offline_target_selected_hypothesis") is not False:
            raise ValueError("deployable tagger input cannot select a hypothesis with offline truth")
        if "shared_root_ledger" not in self.arrays:
            raise KeyError("pseudo input is missing shared_root_ledger")
        root = self.arrays["shared_root_ledger"]
        if root.ndim != 2:
            raise ValueError("shared_root_ledger must have shape [B,R]")
        n_views = len(self.view_names)
        if n_views <= 0:
            raise ValueError("pseudo input has no hypotheses")
        if not self.hierarchy_names or len(set(self.hierarchy_names)) != len(
            self.hierarchy_names
        ):
            raise ValueError("pseudo input requires unique hierarchy names")
        for hierarchy in self.hierarchy_names:
            depth_count = int(self.frontier_depths.get(hierarchy, 0))
            if not 1 <= depth_count <= 6:
                raise ValueError(f"hierarchy {hierarchy} has an invalid frontier depth")
            mask_key = f"particle__{hierarchy}__mask"
            if mask_key not in self.arrays:
                raise KeyError(f"pseudo input is missing {mask_key}")
            mask = self.arrays[mask_key]
            if mask.shape[:2] != (root.shape[0], n_views):
                raise ValueError(f"{mask_key} has incompatible batch/view axes")
            if int(mask.shape[-1]) != ABPH_MAX_PARTICLES:
                raise ValueError(f"{mask_key} violates the fixed 128-token contract")
            root_prefix = f"frontier__{hierarchy}__depth_00__"
            for field in ("ledger", "uncertainty", "mask"):
                if root_prefix + field not in self.arrays:
                    raise KeyError(f"pseudo input is missing {root_prefix + field}")
            branch_root = self.arrays[root_prefix + "ledger"]
            if branch_root.shape != (root.shape[0], n_views, 1, root.shape[1]):
                raise ValueError(f"hierarchy {hierarchy} root frontier has the wrong shape")
            if self.arrays[root_prefix + "uncertainty"].shape != (
                root.shape[0],
                n_views,
                1,
            ):
                raise ValueError(f"hierarchy {hierarchy} root uncertainty has the wrong shape")
            expected_root = root[:, None, None, :].expand_as(branch_root)
            if not torch.equal(branch_root, expected_root):
                raise ValueError(
                    f"hierarchy {hierarchy} does not receive the exact shared compiled root"
                )
        if len(self.hierarchy_names) > 1:
            first = f"frontier__{self.hierarchy_names[0]}__depth_00__"
            for hierarchy in self.hierarchy_names[1:]:
                candidate = f"frontier__{hierarchy}__depth_00__"
                for field in ("uncertainty", "mask"):
                    if not torch.equal(
                        self.arrays[first + field], self.arrays[candidate + field]
                    ):
                        raise ValueError(
                            f"hierarchy roots disagree on shared {field}"
                        )

    def permute_views(self, order: Any) -> "PseudoViewInputs":
        order_tensor = torch.as_tensor(order, dtype=torch.long, device=self.arrays["shared_root_ledger"].device)
        if tuple(sorted(order_tensor.tolist())) != tuple(range(len(self.view_names))):
            raise ValueError("view permutation must contain every hypothesis exactly once")
        arrays: dict[str, Any] = {}
        for name, value in self.arrays.items():
            if value.ndim >= 2 and int(value.shape[1]) == len(self.view_names) and name != "shared_root_ledger":
                arrays[name] = value.index_select(1, order_tensor)
            else:
                arrays[name] = value
        names = tuple(self.view_names[int(index)] for index in order_tensor.tolist())
        return replace(self, arrays=arrays, view_names=names)


@dataclass(frozen=True)
class ParticleStageState:
    tokens: Any
    valid_mask: Any
    attention_bias: Any | None
    four_vector: Any


class _BiasedSelfAttentionBlock(_ModuleBase):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=float(dropout), batch_first=True
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, model_dim),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: Any, valid_mask: Any, bias: Any | None = None) -> Any:
        safe_tokens, padding = _safe_key_padding(tokens, valid_mask)
        query = self.norm1(safe_tokens)
        update, _ = self.attn(
            query,
            query,
            query,
            key_padding_mask=_compatible_padding_mask(padding, bias, query.dtype),
            attn_mask=_attention_mask_for_mha(bias),
            need_weights=False,
        )
        output = safe_tokens + self.dropout(update)
        output = output + self.dropout(self.ff(self.norm2(output)))
        return output * valid_mask.unsqueeze(-1).to(output.dtype)


class NativeStagewiseParticleTransformer(_ModuleBase):
    """Small stagewise ParT-compatible backbone used for tests and smoke pilots."""

    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int = 192,
        num_layers: int = 12,
        num_heads: int = 8,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(int(input_dim)), nn.Linear(int(input_dim), self.model_dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            _BiasedSelfAttentionBlock(
                self.model_dim, self.num_heads, self.model_dim * 4, float(dropout)
            )
            for _ in range(self.num_layers)
        )
        self.final_norm = nn.LayerNorm(self.model_dim)
        self.classifier = nn.Linear(self.model_dim, int(num_classes))

    def prepare(self, features: Any, four_vector: Any, mask: Any) -> ParticleStageState:
        if features.ndim != 3 or four_vector.ndim != 3 or mask.ndim != 3:
            raise ValueError("stagewise inputs must have shapes [B,C,P], [B,4,P], [B,1,P]")
        valid = mask[:, 0].bool()
        tokens = self.input_projection(features.transpose(1, 2))
        tokens = tokens * valid.unsqueeze(-1).to(tokens.dtype)
        return ParticleStageState(
            tokens=tokens,
            valid_mask=valid,
            attention_bias=None,
            four_vector=four_vector.transpose(1, 2),
        )

    def run_layers(
        self,
        state: ParticleStageState,
        start: int,
        stop: int,
        *,
        extra_bias: Any | None = None,
    ) -> ParticleStageState:
        if not 0 <= int(start) <= int(stop) <= self.num_layers:
            raise ValueError("invalid particle-layer interval")
        tokens = state.tokens
        for block in self.blocks[int(start) : int(stop)]:
            tokens = block(tokens, state.valid_mask, extra_bias)
        return replace(state, tokens=tokens)

    def pool_and_classify(self, state: ParticleStageState) -> tuple[Any, Any]:
        representation = self.final_norm(_masked_mean(state.tokens, state.valid_mask, 1))
        return representation, self.classifier(representation)

    def classify(self, representation: Any) -> Any:
        return self.classifier(representation)


class WeaverStagewiseParticleTransformer(_ModuleBase):
    """Stagewise adapter around Weaver's reference ``ParticleTransformer``.

    The adapter uses the reference embedding, pair embedding, particle blocks,
    class-attention blocks, normalization, and classifier.  In evaluation this
    is algebraically the normal Weaver forward pass, while exposing layers
    4/8/preclassification for Step-10 fusion.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.reference_model = model
        mod = getattr(model, "mod", model)
        required = ("embed", "blocks", "cls_token", "cls_blocks", "norm", "fc")
        missing = [name for name in required if not hasattr(mod, name)]
        if missing:
            raise TypeError(f"Weaver ParticleTransformer is missing stage modules: {missing}")
        self.num_layers = len(mod.blocks)
        self.model_dim = int(mod.cls_token.shape[-1])
        self.num_heads = int(getattr(mod.blocks[0], "num_heads", 1))

    @property
    def mod(self) -> Any:
        return getattr(self.reference_model, "mod", self.reference_model)

    def prepare(self, features: Any, four_vector: Any, mask: Any) -> ParticleStageState:
        valid = mask[:, 0].bool()
        pair_embed = getattr(self.mod, "pair_embed", None)
        pair_bias = None
        if pair_embed is not None:
            pair_bias = pair_embed(four_vector, uu=None, mask=mask)
        tokens = self.mod.embed(features)
        tokens = tokens * valid.unsqueeze(-1).to(tokens.dtype)
        return ParticleStageState(tokens, valid, pair_bias, four_vector.transpose(1, 2))

    def run_layers(
        self,
        state: ParticleStageState,
        start: int,
        stop: int,
        *,
        extra_bias: Any | None = None,
    ) -> ParticleStageState:
        if not 0 <= int(start) <= int(stop) <= self.num_layers:
            raise ValueError("invalid Weaver particle-layer interval")
        bias = state.attention_bias
        if extra_bias is not None:
            bias = extra_bias if bias is None else bias + extra_bias
        tokens = state.tokens
        padding = ~state.valid_mask
        for block in self.mod.blocks[int(start) : int(stop)]:
            tokens = block(tokens, padding_mask=padding, attn_mask=bias)
            tokens = tokens * state.valid_mask.unsqueeze(-1).to(tokens.dtype)
        return replace(state, tokens=tokens)

    def pool_and_classify(self, state: ParticleStageState) -> tuple[Any, Any]:
        cls_token = self.mod.cls_token.expand(state.tokens.shape[0], 1, -1)
        padding = ~state.valid_mask
        for block in self.mod.cls_blocks:
            cls_token = block(state.tokens, x_cls=cls_token, padding_mask=padding)
        representation = self.mod.norm(cls_token).squeeze(1)
        return representation, self.mod.fc(representation)

    def classify(self, representation: Any) -> Any:
        return self.mod.fc(representation)

    def reference_parity_report(
        self,
        features: Any,
        four_vector: Any,
        mask: Any,
        *,
        tolerance: float = 1.0e-6,
    ) -> dict[str, Any]:
        """Verify the staged evaluation path reproduces the unmodified ParT."""

        was_training = self.training
        self.eval()
        with torch.no_grad():
            reference = self.mod(features, v=four_vector, mask=mask)
            state = self.prepare(features, four_vector, mask)
            state = self.run_layers(state, 0, self.num_layers)
            _, staged = self.pool_and_classify(state)
        if was_training:
            self.train()
        difference = (reference - staged).abs()
        maximum = float(difference.max().cpu())
        return {
            "ok": bool(maximum <= float(tolerance)),
            "maximum_absolute_difference": maximum,
            "mean_absolute_difference": float(difference.mean().cpu()),
            "tolerance": float(tolerance),
        }


if nn is None:  # pragma: no cover - cannot instantiate without torch
    class _ZeroInitLazyLinear:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _require_torch()
else:
    class _ZeroInitLazyLinear(nn.LazyLinear):
        def initialize_parameters(self, input: Any) -> None:  # noqa: A002 - PyTorch API name
            super().initialize_parameters(input)
            if not self.has_uninitialized_params():
                nn.init.zeros_(self.weight)
                if self.bias is not None:
                    nn.init.zeros_(self.bias)


class TreeRelationBias(_ModuleBase):
    """Permutation-equivariant pseudo-particle attention relation encoder."""

    def __init__(self, num_heads: int, max_depth: int = 6) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_depth = int(max_depth)
        self.projection = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, self.num_heads, bias=False)
        )

    def forward(self, ancestor_indices: Any, uncertainty: Any, mask: Any) -> Any:
        if ancestor_indices.ndim != 3:
            raise ValueError("ancestor indices must have shape [B,P,D]")
        valid_ancestor = ancestor_indices >= 0
        same = (
            ancestor_indices[:, :, None, :] == ancestor_indices[:, None, :, :]
        ) & valid_ancestor[:, :, None, :] & valid_ancestor[:, None, :, :]
        depths = torch.arange(
            ancestor_indices.shape[-1], device=ancestor_indices.device, dtype=uncertainty.dtype
        )
        deepest = torch.where(
            same,
            depths.reshape(1, 1, 1, -1),
            torch.full_like(depths.reshape(1, 1, 1, -1), -1.0),
        ).amax(dim=-1)
        norm_depth = (deepest + 1.0) / max(float(ancestor_indices.shape[-1]), 1.0)
        path_distance = 2.0 * (1.0 - norm_depth)
        recent_support = same.to(uncertainty.dtype).mean(dim=-1)
        relative_uncertainty = torch.abs(uncertainty[:, :, None] - uncertainty[:, None, :])
        same_hypothesis = torch.ones_like(norm_depth)
        features = torch.stack(
            (norm_depth, path_distance, same_hypothesis, recent_support, relative_uncertainty),
            dim=-1,
        )
        bias = self.projection(features).permute(0, 3, 1, 2)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        return bias * pair_mask[:, None].to(bias.dtype)


@dataclass(frozen=True)
class HierarchyEncoding:
    tokens: Any
    mask: Any
    uncertainty: Any
    depth: Any
    support: Any
    levels: tuple[Any, ...]
    level_masks: tuple[Any, ...]
    level_uncertainty: tuple[Any, ...]
    parent_indices: tuple[Any, ...]


class HierarchyMemoryEncoder(_ModuleBase):
    def __init__(self, model_dim: int, max_depth: int = 6, max_hierarchies: int = 4) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.max_depth = int(max_depth)
        self.ledger_projection = nn.LazyLinear(self.model_dim)
        self.hidden_projection = nn.LazyLinear(self.model_dim)
        self.support_projection = nn.LazyLinear(self.model_dim)
        self.scalar_projection = nn.Linear(2, self.model_dim)
        self.root_projection = nn.LazyLinear(self.model_dim)
        self.depth_embedding = nn.Embedding(self.max_depth, self.model_dim)
        self.hierarchy_embedding = nn.Embedding(int(max_hierarchies), self.model_dim)
        self.norm = nn.LayerNorm(self.model_dim)

    def forward(
        self,
        pseudo: PseudoViewInputs,
        hierarchy: str,
        hierarchy_index: int,
        *,
        root_override: Any | None = None,
    ) -> HierarchyEncoding:
        arrays = pseudo.arrays
        levels: list[Any] = []
        masks: list[Any] = []
        uncertainties: list[Any] = []
        parents: list[Any] = []
        depth_vectors: list[Any] = []
        support_scores: list[Any] = []
        for depth in range(int(pseudo.frontier_depths[hierarchy])):
            prefix = f"frontier__{hierarchy}__depth_{depth:02d}__"
            ledger = arrays[prefix + "ledger"]
            hidden = arrays[prefix + "hidden"]
            support = arrays[prefix + "support"]
            uncertainty = arrays[prefix + "uncertainty"].to(ledger.dtype)
            topology = arrays[prefix + "topology"].to(ledger.dtype)
            mask = arrays[prefix + "mask"].bool()
            if depth == 0:
                root = arrays["shared_root_ledger"] if root_override is None else root_override
                if root.shape != arrays["shared_root_ledger"].shape:
                    raise ValueError("root override has the wrong compiled-ledger shape")
                token = self.root_projection(root)[:, None, None, :].expand(
                    -1, ledger.shape[1], 1, -1
                )
                token = (
                    token
                    + self.scalar_projection(
                        torch.stack(
                            (uncertainty, torch.zeros_like(uncertainty)), dim=-1
                        )
                    )
                    + self.depth_embedding.weight[depth]
                )
            else:
                token = (
                    self.ledger_projection(ledger)
                    + self.hidden_projection(hidden)
                    + self.support_projection(support)
                    + self.scalar_projection(torch.stack((uncertainty, topology), dim=-1))
                    + self.depth_embedding.weight[depth]
                    + self.hierarchy_embedding.weight[int(hierarchy_index)]
                )
            token = self.norm(token) * mask.unsqueeze(-1).to(token.dtype)
            levels.append(token)
            masks.append(mask)
            uncertainties.append(uncertainty * mask.to(uncertainty.dtype))
            parents.append(arrays[prefix + "parent_indices"].long())
            depth_vectors.append(torch.full_like(uncertainty, float(depth)))
            support_scores.append(torch.linalg.vector_norm(support, dim=-1))
        return HierarchyEncoding(
            tokens=torch.cat(levels, dim=2),
            mask=torch.cat(masks, dim=2),
            uncertainty=torch.cat(uncertainties, dim=2),
            depth=torch.cat(depth_vectors, dim=2),
            support=torch.cat(support_scores, dim=2),
            levels=tuple(levels),
            level_masks=tuple(masks),
            level_uncertainty=tuple(uncertainties),
            parent_indices=tuple(parents),
        )


def derive_particle_ancestors(group_indices: Any, hierarchy: HierarchyEncoding) -> Any:
    """Walk compiled parent links from final groups back to the shared root."""

    if group_indices.ndim != 3:
        raise ValueError("particle group indices must have shape [B,V,P]")
    depth_count = len(hierarchy.levels)
    ancestors = torch.full(
        group_indices.shape + (depth_count,),
        -1,
        dtype=torch.long,
        device=group_indices.device,
    )
    current = group_indices.long()
    ancestors[..., depth_count - 1] = current
    batch, views, particles = current.shape
    for depth in range(depth_count - 1, 0, -1):
        parent = hierarchy.parent_indices[depth].reshape(batch * views, -1)
        current_flat = current.reshape(batch * views, particles)
        gathered, valid = _batched_gather(parent.unsqueeze(-1), current_flat)
        current = gathered.squeeze(-1).long().reshape(batch, views, particles)
        valid = valid.reshape(batch, views, particles)
        current = torch.where(valid, current, torch.full_like(current, -1))
        ancestors[..., depth - 1] = current
    return ancestors


class AncestorInjection(_ModuleBase):
    def __init__(self, particle_dim: int, hierarchy_dim: int, max_depth: int = 6) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(hierarchy_dim, particle_dim) for _ in range(int(max_depth))
        )
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(particle_dim + 2, particle_dim), nn.GELU(), nn.Linear(particle_dim, 1))
            for _ in range(int(max_depth))
        )

    def forward(
        self,
        tokens: Any,
        particle_uncertainty: Any,
        particle_mask: Any,
        ancestors: Any,
        hierarchy: HierarchyEncoding,
    ) -> Any:
        batch_views, particles, _ = tokens.shape
        output = tokens
        for depth, level in enumerate(hierarchy.levels):
            level_flat = level.reshape(batch_views, level.shape[2], level.shape[3])
            unc_flat = hierarchy.level_uncertainty[depth].reshape(batch_views, level.shape[2])
            index_flat = ancestors[..., depth].reshape(batch_views, particles)
            ancestor, valid = _batched_gather(level_flat, index_flat)
            group_unc, unc_valid = _batched_gather(unc_flat.unsqueeze(-1), index_flat)
            projected = self.projections[depth](ancestor)
            gate_input = torch.cat(
                (projected, particle_uncertainty.unsqueeze(-1), group_unc), dim=-1
            )
            gate = torch.sigmoid(self.gates[depth](gate_input))
            active = valid & unc_valid & particle_mask
            output = output + gate * projected * active.unsqueeze(-1).to(output.dtype)
        return output * particle_mask.unsqueeze(-1).to(output.dtype)


class PermutationInvariantViewAggregator(_ModuleBase):
    """Set transformer over hypotheses with no position or hypothesis-index embedding."""

    def __init__(
        self,
        model_dim: int,
        latent_dim: int,
        *,
        num_heads: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_projection = nn.Linear(int(latent_dim) + 1, int(model_dim))
        self.blocks = nn.ModuleList(
            _BiasedSelfAttentionBlock(model_dim, num_heads, model_dim * 4, dropout)
            for _ in range(int(blocks))
        )
        self.score = nn.Linear(model_dim, 1)

    def forward(
        self,
        tokens: Any,
        mask: Any,
        latent: Any,
        prior_log_prob: Any,
        view_keep: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        if tokens.ndim != 4 or mask.shape != tokens.shape[:3]:
            raise ValueError("view tokens/mask must have shapes [B,V,P,D]/[B,V,P]")
        batch, views, particles, dim = tokens.shape
        if view_keep is not None:
            if view_keep.shape != (batch, views):
                raise ValueError("view_keep must have shape [B,V]")
            mask = mask & view_keep[:, :, None]
        hypothesis = self.latent_projection(
            torch.cat((latent, prior_log_prob.unsqueeze(-1)), dim=-1)
        )
        values = tokens + hypothesis[:, :, None, :]
        values = values.permute(0, 2, 1, 3).reshape(batch * particles, views, dim)
        valid = mask.permute(0, 2, 1).reshape(batch * particles, views)
        for block in self.blocks:
            values = block(values, valid)
        score = self.score(values).squeeze(-1).masked_fill(~valid, -1.0e4)
        weights = torch.softmax(score, dim=1) * valid.to(score.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        aggregate = (values * weights.unsqueeze(-1)).sum(dim=1)
        variance = ((values - aggregate[:, None]) ** 2 * weights.unsqueeze(-1)).sum(dim=1)
        aggregate = aggregate.reshape(batch, particles, dim)
        disagreement = variance.mean(dim=-1).clamp_min(1.0e-12).sqrt().reshape(
            batch, particles
        )
        aggregate_mask = mask.any(dim=1)
        aggregate = aggregate * aggregate_mask.unsqueeze(-1).to(aggregate.dtype)
        disagreement = disagreement * aggregate_mask.to(disagreement.dtype)
        return aggregate, aggregate_mask, disagreement


@dataclass(frozen=True)
class _PseudoBranch:
    name: str
    state: ParticleStageState
    hierarchy: HierarchyEncoding
    ancestors: Any
    uncertainty: Any
    support: Any
    latent: Any
    prior_log_prob: Any
    view_keep: Any


@dataclass(frozen=True)
class _CrossMemory:
    tokens: Any
    mask: Any
    four_vector: Any
    uncertainty: Any
    disagreement: Any
    support: Any
    depth: Any
    geometric_mask: Any


class CrossRelationBias(_ModuleBase):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.projection = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(), nn.Linear(32, self.num_heads, bias=False)
        )

    @staticmethod
    def _kinematics(four_vector: Any) -> tuple[Any, Any, Any]:
        px, py, pz, energy = four_vector.unbind(dim=-1)
        del energy
        pt2 = px.square() + py.square()
        geometric = pt2 > 1.0e-12
        # Hierarchy-memory keys have no physical four-vector.  Feeding literal
        # (0,0) to atan2 has an undefined backward even though its forward is
        # finite, so represent those keys at a neutral differentiable origin.
        safe_px = torch.where(geometric, px, torch.ones_like(px))
        safe_py = torch.where(geometric, py, torch.zeros_like(py))
        safe_pz = torch.where(geometric, pz, torch.zeros_like(pz))
        pt = torch.sqrt((safe_px.square() + safe_py.square()).clamp_min(1.0e-12))
        eta = torch.asinh(safe_pz / pt.clamp_min(1.0e-8))
        phi = torch.atan2(safe_py, safe_px)
        return pt, eta, phi

    def forward(self, query_p4: Any, memory: _CrossMemory) -> Any:
        qpt, qeta, qphi = self._kinematics(query_p4)
        kpt, keta, kphi = self._kinematics(memory.four_vector)
        deta = qeta[:, :, None] - keta[:, None, :]
        dphi = torch.remainder(qphi[:, :, None] - kphi[:, None, :] + torch.pi, 2 * torch.pi) - torch.pi
        dr = torch.sqrt((deta.square() + dphi.square()).clamp_min(1.0e-12))
        rel_log_pt = torch.log(qpt[:, :, None].clamp_min(1.0e-8)) - torch.log(
            kpt[:, None, :].clamp_min(1.0e-8)
        )
        shape = deta.shape
        features = torch.stack(
            (
                deta,
                dphi,
                dr,
                rel_log_pt,
                memory.uncertainty[:, None].expand(shape),
                memory.disagreement[:, None].expand(shape),
                memory.support[:, None].expand(shape),
                memory.depth[:, None].expand(shape),
            ),
            dim=-1,
        )
        return self.projection(features).permute(0, 3, 1, 2)

    @classmethod
    def local_support(cls, query_p4: Any, memory: _CrossMemory) -> Any:
        _, qeta, qphi = cls._kinematics(query_p4)
        _, keta, kphi = cls._kinematics(memory.four_vector)
        deta = qeta[:, :, None] - keta[:, None, :]
        dphi = torch.remainder(qphi[:, :, None] - kphi[:, None, :] + torch.pi, 2 * torch.pi) - torch.pi
        radial_weight = torch.exp(-(deta.square() + dphi.square()) / (2.0 * 0.16**2))
        valid = (memory.mask & memory.geometric_mask).to(radial_weight.dtype)
        return torch.log1p((radial_weight * valid[:, None]).sum(dim=-1))


class BidirectionalFusionBlock(_ModuleBase):
    def __init__(
        self,
        *,
        particle_dim: int,
        fusion_dim: int,
        num_heads: int,
        dropout: float,
        rezero_init: float,
        uncertainty_gates: bool = True,
        update_pseudo_stream: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.uncertainty_gates = bool(uncertainty_gates)
        self.update_pseudo_stream = bool(update_pseudo_stream)
        self.hlt_in = nn.Linear(particle_dim, fusion_dim)
        self.pseudo_in = nn.Linear(particle_dim, fusion_dim)
        self.hlt_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pseudo_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.hlt_norm = nn.LayerNorm(fusion_dim)
        self.pseudo_norm = nn.LayerNorm(fusion_dim)
        self.hlt_out = nn.Linear(fusion_dim, particle_dim)
        self.pseudo_out = nn.Linear(fusion_dim, particle_dim)
        self.trust_gate = nn.Linear(fusion_dim * 2 + 5, fusion_dim)
        nn.init.zeros_(self.trust_gate.bias)
        self.hlt_rezero = nn.Parameter(torch.tensor(float(rezero_init)))
        self.pseudo_rezero = nn.Parameter(torch.tensor(float(rezero_init)))
        self.relation_bias = CrossRelationBias(num_heads)

    def forward(
        self,
        hlt_state: ParticleStageState,
        branches: Sequence[_PseudoBranch],
        memory: _CrossMemory,
    ) -> tuple[ParticleStageState, tuple[_PseudoBranch, ...]]:
        safe_memory, memory_padding = _safe_key_padding(memory.tokens, memory.mask)
        hlt_query = self.hlt_in(hlt_state.tokens)
        relation = self.relation_bias(hlt_state.four_vector, memory)
        attended, _ = self.hlt_attention(
            hlt_query,
            safe_memory,
            safe_memory,
            key_padding_mask=_compatible_padding_mask(
                memory_padding, relation, hlt_query.dtype
            ),
            attn_mask=_attention_mask_for_mha(relation),
            need_weights=False,
        )
        key_weight = memory.mask.to(hlt_query.dtype)
        mean_unc = (memory.uncertainty * key_weight).sum(1) / key_weight.sum(1).clamp_min(1.0)
        mean_dis = (memory.disagreement * key_weight).sum(1) / key_weight.sum(1).clamp_min(1.0)
        mean_support = (memory.support * key_weight).sum(1) / key_weight.sum(1).clamp_min(1.0)
        mean_depth = (memory.depth * key_weight).sum(1) / key_weight.sum(1).clamp_min(1.0)
        local_support = self.relation_bias.local_support(hlt_state.four_vector, memory)
        scalar = torch.stack((mean_unc, mean_dis, mean_support, mean_depth), dim=-1)
        scalar = scalar[:, None].expand(-1, hlt_query.shape[1], -1)
        gate_input = torch.cat((hlt_query, attended, scalar, local_support.unsqueeze(-1)), dim=-1)
        trust = (
            torch.sigmoid(self.trust_gate(gate_input))
            if self.uncertainty_gates
            else torch.ones_like(attended)
        )
        hlt_update = self.hlt_out(self.hlt_norm(attended * trust))
        hlt_tokens = hlt_state.tokens + self.hlt_rezero * hlt_update
        hlt_tokens = hlt_tokens * hlt_state.valid_mask.unsqueeze(-1).to(hlt_tokens.dtype)
        if not self.update_pseudo_stream:
            return replace(hlt_state, tokens=hlt_tokens), tuple(branches)
        updated_branches: list[_PseudoBranch] = []
        for branch in branches:
            batch, views, particles = branch.uncertainty.shape
            pseudo_query = self.pseudo_in(branch.state.tokens)
            hlt_keys = hlt_query[:, None].expand(-1, views, -1, -1).reshape(
                batch * views, hlt_query.shape[1], hlt_query.shape[2]
            )
            hlt_valid = hlt_state.valid_mask[:, None].expand(-1, views, -1).reshape(
                batch * views, -1
            )
            safe_hlt, hlt_padding = _safe_key_padding(hlt_keys, hlt_valid)
            reverse, _ = self.pseudo_attention(
                pseudo_query,
                safe_hlt,
                safe_hlt,
                key_padding_mask=hlt_padding,
                need_weights=False,
            )
            pseudo_update = self.pseudo_out(self.pseudo_norm(reverse))
            pseudo_tokens = branch.state.tokens + self.pseudo_rezero * pseudo_update
            pseudo_tokens = pseudo_tokens * branch.state.valid_mask.unsqueeze(-1).to(pseudo_tokens.dtype)
            updated_branches.append(replace(branch, state=replace(branch.state, tokens=pseudo_tokens)))
        return replace(hlt_state, tokens=hlt_tokens), tuple(updated_branches)


@dataclass(frozen=True)
class HierarchyAwareTaggerOutput:
    logits: Any
    baseline_logits: Any
    representation: Any
    baseline_representation: Any
    diagnostics: Mapping[str, Any]
    auxiliary_logits: Mapping[str, Any] = field(default_factory=dict)


class HierarchyAwareDualStreamTagger(_ModuleBase):
    """Performance-first HLT + pseudo-particle + hierarchy-memory classifier."""

    def __init__(
        self,
        *,
        hlt_backbone: Any,
        pseudo_backbone: Any,
        num_classes: int = 10,
        fusion_dim: int = 256,
        fusion_heads: int = 8,
        fusion_blocks_per_location: int = 2,
        fusion_locations: Sequence[int] = ABPH_PRIMARY_FUSION_LOCATIONS,
        view_aggregator_blocks: int = 4,
        view_dropout: float = 0.15,
        hypothesis_latent_dim: int = 64,
        dropout: float = 0.1,
        rezero_init: float = ABPH_PRIMARY_REZERO_INIT,
        dual_hierarchy: bool = False,
        independent_roots: bool = False,
        fusion_kind: str = "bidirectional_dualcross",
        use_hierarchy_memory: bool = True,
        use_ancestor_injection: bool = True,
        use_tree_bias: bool = True,
        uncertainty_gates: bool = True,
    ) -> None:
        super().__init__()
        self.hlt_backbone = hlt_backbone
        self.pseudo_backbone = pseudo_backbone
        self.num_classes = int(num_classes)
        self.fusion_dim = int(fusion_dim)
        self.dual_hierarchy = bool(dual_hierarchy)
        self.independent_roots = bool(independent_roots)
        self.fusion_kind = str(fusion_kind)
        self.use_hierarchy_memory = bool(use_hierarchy_memory)
        self.use_ancestor_injection = bool(use_ancestor_injection)
        self.use_tree_bias = bool(use_tree_bias)
        self.uncertainty_gates = bool(uncertainty_gates)
        self.view_dropout = float(view_dropout)
        if not 0.0 <= self.view_dropout < 1.0:
            raise ValueError("view_dropout must be in [0,1)")
        if self.independent_roots and not self.dual_hierarchy:
            raise ValueError("independent roots are meaningful only for a dual hierarchy")
        if int(hlt_backbone.model_dim) != int(pseudo_backbone.model_dim):
            raise ValueError("HLT and pseudo ParT streams must share the locked embedding width")
        if int(hlt_backbone.num_layers) != int(pseudo_backbone.num_layers):
            raise ValueError("HLT and pseudo ParT streams must share the locked layer count")
        locations = tuple(int(value) for value in fusion_locations)
        non_cross_kinds = {"pseudo_only", "untrained_logit_mean", "late_representation"}
        if self.fusion_kind not in non_cross_kinds and not locations:
            raise ValueError("cross-fusion variants require at least one fusion location")
        if locations and locations[-1] > int(hlt_backbone.num_layers):
            raise ValueError("fusion location exceeds the particle backbone depth")
        if tuple(sorted(set(locations))) != locations:
            raise ValueError("fusion locations must be unique and increasing")
        self.fusion_locations = locations
        particle_dim = int(hlt_backbone.model_dim)
        self.side_adapter = _ZeroInitLazyLinear(particle_dim)
        self.hierarchy_encoder = HierarchyMemoryEncoder(self.fusion_dim)
        self.ancestor_injection = AncestorInjection(particle_dim, self.fusion_dim)
        self.tree_bias = TreeRelationBias(int(pseudo_backbone.num_heads))
        self.view_aggregator = PermutationInvariantViewAggregator(
            particle_dim,
            int(hypothesis_latent_dim),
            num_heads=int(fusion_heads),
            blocks=int(view_aggregator_blocks),
            dropout=float(dropout),
        )
        self.hierarchy_view_aggregator = PermutationInvariantViewAggregator(
            self.fusion_dim,
            int(hypothesis_latent_dim),
            num_heads=int(fusion_heads),
            blocks=int(view_aggregator_blocks),
            dropout=float(dropout),
        )
        self.pseudo_memory_projection = nn.Linear(particle_dim, self.fusion_dim)
        self.fusion_stacks = nn.ModuleList(
            nn.ModuleList(
                BidirectionalFusionBlock(
                    particle_dim=particle_dim,
                    fusion_dim=self.fusion_dim,
                    num_heads=int(fusion_heads),
                    dropout=float(dropout),
                    rezero_init=float(rezero_init),
                    uncertainty_gates=self.uncertainty_gates,
                    update_pseudo_stream=(self.fusion_kind != "single_cross_attention"),
                )
                for _ in range(int(fusion_blocks_per_location))
            )
            for _ in self.fusion_locations
        )
        joint_dim = particle_dim * 2 + self.fusion_dim
        self.delta_representation = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, particle_dim * 2),
            nn.GELU(),
            nn.Linear(particle_dim * 2, particle_dim),
            nn.Tanh(),
        )
        self.delta_logits = nn.Sequential(
            nn.LayerNorm(joint_dim), nn.Linear(joint_dim, self.num_classes), nn.Tanh()
        )
        self.late_representation = nn.Sequential(
            nn.LayerNorm(particle_dim * 2),
            nn.Linear(particle_dim * 2, particle_dim * 2),
            nn.GELU(),
            nn.Linear(particle_dim * 2, particle_dim),
        )
        self.hierarchy_aux_classifier = nn.Linear(self.fusion_dim, self.num_classes)
        self.alpha = nn.Parameter(torch.tensor(float(rezero_init)))
        self.beta = nn.Parameter(torch.tensor(float(rezero_init)))

    @staticmethod
    def _part_inputs(raw_tokens: Any, mask: Any) -> Mapping[str, Any]:
        from jetclass_fresh.dual_view import build_part_inputs_torch

        return build_part_inputs_torch(raw_tokens, mask, max_constits=ABPH_MAX_PARTICLES)

    @staticmethod
    def _pseudo_raw(canonical: Any) -> tuple[Any, Any]:
        if canonical.shape[-1] < 15:
            raise ValueError("pseudo canonical particle tensor is incomplete")
        # The offline ParT contract has five explicit PID flags.  The ABPH-only
        # `other` probability is carried by the residual side adapter instead.
        raw = torch.cat((canonical[..., :10], canonical[..., 11:15]), dim=-1)
        other_pid = canonical[..., 10:11]
        return raw, other_pid

    def _prepare_branch(
        self,
        pseudo: PseudoViewInputs,
        hierarchy_name: str,
        hierarchy_index: int,
        *,
        root_override: Any | None = None,
        view_keep: Any,
    ) -> _PseudoBranch:
        arrays = pseudo.arrays
        prefix = f"particle__{hierarchy_name}__"
        canonical = arrays[prefix + "canonical_features"]
        side = arrays[prefix + "side_channels"]
        mask = arrays[prefix + "mask"].bool()
        uncertainty = arrays[prefix + "uncertainty"].to(canonical.dtype)
        support = side[..., 2].to(canonical.dtype) * mask.to(canonical.dtype)
        raw, other_pid = self._pseudo_raw(canonical)
        batch, views, particles, _ = raw.shape
        part = self._part_inputs(raw.reshape(batch * views, particles, -1), mask.reshape(batch * views, particles))
        state = self.pseudo_backbone.prepare(
            part["features"], part["lorentz_vectors"], part["mask"]
        )
        latent = arrays["hypothesis_latent"].to(state.tokens.dtype)
        prior = arrays["hypothesis_prior_log_prob"].to(state.tokens.dtype)
        # Numeric hypothesis indices are deliberately removed.  The latent and
        # log-probability identify a view without introducing order semantics.
        clean_side = side.clone()
        clean_side[..., 1] = 0.0
        residual_features = torch.cat(
            (
                clean_side,
                other_pid,
                latent[:, :, None].expand(-1, -1, particles, -1),
                prior[:, :, None, None].expand(-1, -1, particles, -1),
            ),
            dim=-1,
        ).reshape(batch * views, particles, -1)
        residual = self.side_adapter(residual_features)
        state = replace(
            state,
            tokens=(state.tokens + residual) * state.valid_mask.unsqueeze(-1).to(state.tokens.dtype),
        )
        hierarchy = self.hierarchy_encoder(
            pseudo,
            hierarchy_name,
            hierarchy_index,
            root_override=root_override,
        )
        ancestors = derive_particle_ancestors(arrays[prefix + "group_indices"].long(), hierarchy)
        injected = (
            self.ancestor_injection(
                state.tokens,
                uncertainty.reshape(batch * views, particles),
                state.valid_mask,
                ancestors,
                hierarchy,
            )
            if self.use_ancestor_injection
            else state.tokens
        )
        return _PseudoBranch(
            name=hierarchy_name,
            state=replace(state, tokens=injected),
            hierarchy=hierarchy,
            ancestors=ancestors,
            uncertainty=uncertainty,
            support=support,
            latent=latent,
            prior_log_prob=prior,
            view_keep=view_keep,
        )

    def _aggregate_branch(self, branch: _PseudoBranch) -> tuple[Any, Any, Any, Any, Any]:
        batch, views, particles = branch.uncertainty.shape
        tokens = branch.state.tokens.reshape(batch, views, particles, -1)
        mask = branch.state.valid_mask.reshape(batch, views, particles)
        aggregate, aggregate_mask, disagreement = self.view_aggregator(
            tokens,
            mask,
            branch.latent,
            branch.prior_log_prob,
            branch.view_keep,
        )
        effective_mask = mask & branch.view_keep[:, :, None]
        weights = effective_mask.to(tokens.dtype)
        p4 = branch.state.four_vector.reshape(batch, views, particles, 4)
        p4 = (p4 * weights.unsqueeze(-1)).sum(1) / weights.sum(1).clamp_min(1.0).unsqueeze(-1)
        uncertainty = (branch.uncertainty * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        support = (branch.support * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return aggregate, aggregate_mask, disagreement, p4, torch.stack((uncertainty, support), dim=-1)

    def _cross_memory(self, branches: Sequence[_PseudoBranch]) -> _CrossMemory:
        tokens: list[Any] = []
        masks: list[Any] = []
        p4s: list[Any] = []
        uncertainties: list[Any] = []
        disagreements: list[Any] = []
        supports: list[Any] = []
        depths: list[Any] = []
        geometric_masks: list[Any] = []
        for branch in branches:
            aggregate, mask, disagreement, p4, statistics = self._aggregate_branch(branch)
            tokens.append(self.pseudo_memory_projection(aggregate))
            masks.append(mask)
            p4s.append(p4)
            uncertainties.append(statistics[..., 0])
            disagreements.append(disagreement)
            supports.append(statistics[..., 1])
            depths.append(torch.full_like(disagreement, float(len(branch.hierarchy.levels))))
            geometric_masks.append(mask)
            if not self.use_hierarchy_memory:
                continue
            hierarchy_tokens, hierarchy_mask, hierarchy_disagreement = self.hierarchy_view_aggregator(
                branch.hierarchy.tokens,
                branch.hierarchy.mask,
                branch.latent,
                branch.prior_log_prob,
                branch.view_keep,
            )
            tokens.append(hierarchy_tokens)
            masks.append(hierarchy_mask)
            p4s.append(torch.zeros(hierarchy_tokens.shape[:-1] + (4,), device=p4.device, dtype=p4.dtype))
            hierarchy_effective_mask = (
                branch.hierarchy.mask & branch.view_keep[:, :, None]
            )
            hweights = hierarchy_effective_mask.to(hierarchy_tokens.dtype)
            hunc = (branch.hierarchy.uncertainty * hweights).sum(1) / hweights.sum(1).clamp_min(1.0)
            hsupport = (branch.hierarchy.support * hweights).sum(1) / hweights.sum(1).clamp_min(1.0)
            hdepth = (branch.hierarchy.depth * hweights).sum(1) / hweights.sum(1).clamp_min(1.0)
            uncertainties.append(hunc)
            disagreements.append(hierarchy_disagreement)
            supports.append(hsupport)
            depths.append(hdepth)
            geometric_masks.append(torch.zeros_like(hierarchy_mask))
        return _CrossMemory(
            tokens=torch.cat(tokens, dim=1),
            mask=torch.cat(masks, dim=1),
            four_vector=torch.cat(p4s, dim=1),
            uncertainty=torch.cat(uncertainties, dim=1),
            disagreement=torch.cat(disagreements, dim=1),
            support=torch.cat(supports, dim=1),
            depth=torch.cat(depths, dim=1),
            geometric_mask=torch.cat(geometric_masks, dim=1),
        )

    def _pseudo_classification(
        self, branches: Sequence[_PseudoBranch]
    ) -> tuple[Any, Any]:
        representations = []
        logits = []
        for branch in branches:
            aggregate, aggregate_mask, _, p4, _ = self._aggregate_branch(branch)
            state = ParticleStageState(
                tokens=aggregate,
                valid_mask=aggregate_mask,
                attention_bias=None,
                four_vector=p4,
            )
            representation, branch_logits = self.pseudo_backbone.pool_and_classify(state)
            representations.append(representation)
            logits.append(branch_logits)
        return torch.stack(representations, dim=0).mean(0), torch.stack(logits, dim=0).mean(0)

    def _advance_pseudo_branches(
        self,
        branches: Sequence[_PseudoBranch],
        start: int,
        stop: int,
    ) -> tuple[_PseudoBranch, ...]:
        advanced = []
        for branch in branches:
            tree = None
            if self.use_tree_bias:
                tree = self.tree_bias(
                    branch.ancestors.reshape(
                        -1, branch.ancestors.shape[2], branch.ancestors.shape[3]
                    ),
                    branch.uncertainty.reshape(-1, branch.uncertainty.shape[2]),
                    branch.state.valid_mask,
                )
            state = self.pseudo_backbone.run_layers(
                branch.state, start, stop, extra_bias=tree
            )
            advanced.append(replace(branch, state=state))
        return tuple(advanced)

    def _sample_view_keep(self, pseudo: PseudoViewInputs) -> Any:
        prior = pseudo.arrays["hypothesis_prior_log_prob"]
        if not self.training or self.view_dropout == 0.0:
            return torch.ones_like(prior, dtype=torch.bool)
        keep = torch.rand_like(prior) >= self.view_dropout
        empty = ~keep.any(dim=1)
        if bool(empty.any()):
            fallback = prior.argmax(dim=1)
            keep[empty, fallback[empty]] = True
        return keep

    def _root_provenance(
        self,
        pseudo: PseudoViewInputs,
        independent_root_ledgers: Mapping[str, Any] | None,
        *,
        compute_hashes: bool,
    ) -> dict[str, Any]:
        if independent_root_ledgers:
            if not self.independent_roots:
                raise ValueError("only E11 may supply independent hierarchy roots")
            if set(independent_root_ledgers) != set(pseudo.hierarchy_names):
                raise ValueError("independent roots must cover every hierarchy")
            if not compute_hashes:
                return {
                    "shared_root": False,
                    "root_hashes": None,
                    "root_hash_count": len(independent_root_ledgers),
                    "hashes_deferred_during_training": True,
                }
            hashes = {
                name: _combined_tensor_sha256(
                    value,
                    pseudo.arrays[f"frontier__{name}__depth_00__uncertainty"],
                )
                for name, value in independent_root_ledgers.items()
            }
            return {
                "shared_root": False,
                "root_hashes": hashes,
                "root_hash_count": len(hashes),
                "unique_root_content_hash_count": len(set(hashes.values())),
            }
        if not compute_hashes:
            return {
                "shared_root": True,
                "root_hash": None,
                "root_hash_count": 1,
                "hashes_deferred_during_training": True,
            }
        branch_hashes: dict[str, str] = {}
        for hierarchy_name in pseudo.hierarchy_names:
            prefix = f"frontier__{hierarchy_name}__depth_00__"
            branch_hashes[hierarchy_name] = _combined_tensor_sha256(
                pseudo.arrays[prefix + "ledger"],
                pseudo.arrays[prefix + "uncertainty"],
                pseudo.arrays[prefix + "mask"],
            )
        first_hierarchy = pseudo.hierarchy_names[0]
        first_prefix = f"frontier__{first_hierarchy}__depth_00__"
        first_ledger = pseudo.arrays[first_prefix + "ledger"]
        shared_ledger = pseudo.arrays["shared_root_ledger"][
            :, None, None, :
        ].expand_as(first_ledger)
        root_hash = _combined_tensor_sha256(
            shared_ledger,
            pseudo.arrays[first_prefix + "uncertainty"],
            pseudo.arrays[first_prefix + "mask"],
        )
        return {
            "shared_root": True,
            "root_hash": root_hash,
            "branch_root_hashes": branch_hashes,
            "root_hash_count": len(set(branch_hashes.values())),
            "unique_root_content_hash_count": len(set(branch_hashes.values())),
            "branch_hashes_computed_independently": True,
        }

    def forward(
        self,
        hlt_tokens: Any,
        hlt_mask: Any,
        pseudo_views: PseudoViewInputs | DeployablePseudoViewBatch | None = None,
        *,
        independent_root_ledgers: Mapping[str, Any] | None = None,
    ) -> HierarchyAwareTaggerOutput:
        _require_torch()
        hlt_part = self._part_inputs(hlt_tokens, hlt_mask)
        base_state = self.hlt_backbone.prepare(
            hlt_part["features"], hlt_part["lorentz_vectors"], hlt_part["mask"]
        )
        base_state = self.hlt_backbone.run_layers(base_state, 0, self.hlt_backbone.num_layers)
        base_representation, baseline_logits = self.hlt_backbone.pool_and_classify(base_state)
        if pseudo_views is None:
            return HierarchyAwareTaggerOutput(
                logits=baseline_logits,
                baseline_logits=baseline_logits,
                representation=base_representation,
                baseline_representation=base_representation,
                diagnostics={
                    "contract": ABPH_HIERARCHY_TAGGER_CONTRACT,
                    "hlt_only": True,
                    "offline_inputs_loaded": False,
                    "teacher_logits_loaded": False,
                },
                auxiliary_logits={},
            )
        if isinstance(pseudo_views, DeployablePseudoViewBatch):
            pseudo = PseudoViewInputs.from_deployable_batch(
                pseudo_views, device=hlt_tokens.device, dtype=hlt_tokens.dtype
            )
        else:
            pseudo = pseudo_views
        pseudo.validate()
        if pseudo.arrays["shared_root_ledger"].shape[0] != hlt_tokens.shape[0]:
            raise ValueError("HLT and pseudo batches differ")
        if self.dual_hierarchy and set(pseudo.hierarchy_names) != set(ABPH_PRIMARY_HIERARCHY_NAMES):
            raise ValueError("E7/E11 require exclusive-kT and C/A hierarchy branches")
        root_provenance = self._root_provenance(
            pseudo,
            independent_root_ledgers,
            compute_hashes=not self.training,
        )
        view_keep = self._sample_view_keep(pseudo)
        branches = tuple(
            self._prepare_branch(
                pseudo,
                name,
                (
                    ABPH_PRIMARY_HIERARCHY_NAMES.index(name)
                    if name in ABPH_PRIMARY_HIERARCHY_NAMES
                    else index
                ),
                root_override=(
                    None
                    if independent_root_ledgers is None
                    else independent_root_ledgers[name]
                ),
                view_keep=view_keep,
            )
            for index, name in enumerate(pseudo.hierarchy_names)
        )
        if self.fusion_kind in {
            "pseudo_only",
            "untrained_logit_mean",
            "late_representation",
        }:
            branches = self._advance_pseudo_branches(
                branches, 0, self.pseudo_backbone.num_layers
            )
            pseudo_representation, pseudo_logits = self._pseudo_classification(branches)
            if self.fusion_kind == "pseudo_only":
                representation = pseudo_representation
                logits = pseudo_logits
            elif self.fusion_kind == "untrained_logit_mean":
                representation = 0.5 * (base_representation + pseudo_representation)
                logits = 0.5 * (baseline_logits + pseudo_logits)
            else:
                representation = self.late_representation(
                    torch.cat((base_representation, pseudo_representation), dim=-1)
                )
                logits = self.hlt_backbone.classify(representation)
            return HierarchyAwareTaggerOutput(
                logits=logits,
                baseline_logits=baseline_logits,
                representation=representation,
                baseline_representation=base_representation,
                diagnostics={
                    "contract": ABPH_HIERARCHY_TAGGER_CONTRACT,
                    "hlt_only": False,
                    "fusion_kind": self.fusion_kind,
                    "hierarchy_names": list(pseudo.hierarchy_names),
                    "offline_inputs_loaded": False,
                    "teacher_logits_loaded": False,
                    "root_provenance": root_provenance,
                },
                auxiliary_logits={"pseudo": pseudo_logits},
            )
        fused_state = self.hlt_backbone.prepare(
            hlt_part["features"], hlt_part["lorentz_vectors"], hlt_part["mask"]
        )
        start = 0
        for location_index, stop in enumerate(self.fusion_locations):
            fused_state = self.hlt_backbone.run_layers(fused_state, start, stop)
            branches = self._advance_pseudo_branches(branches, start, stop)
            for fusion in self.fusion_stacks[location_index]:
                memory = self._cross_memory(branches)
                fused_state, branches = fusion(fused_state, branches, memory)
            start = stop
        if start < self.hlt_backbone.num_layers:
            fused_state = self.hlt_backbone.run_layers(
                fused_state, start, self.hlt_backbone.num_layers
            )
            branches = self._advance_pseudo_branches(
                branches, start, self.pseudo_backbone.num_layers
            )
        fused_representation, _ = self.hlt_backbone.pool_and_classify(fused_state)
        memory = self._cross_memory(branches)
        memory_representation = _masked_mean(memory.tokens, memory.mask, 1)
        pseudo_representation, pseudo_logits = self._pseudo_classification(branches)
        joint = torch.cat(
            (base_representation, fused_representation, memory_representation), dim=-1
        )
        delta_representation = self.delta_representation(joint)
        representation = base_representation + self.alpha * delta_representation
        logits = self.hlt_backbone.classify(representation)
        logits = logits + self.beta * self.delta_logits(joint)
        diagnostics = {
            "contract": ABPH_HIERARCHY_TAGGER_CONTRACT,
            "hlt_only": False,
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "fusion_locations": list(self.fusion_locations),
            "fusion_blocks_per_location": [len(stack) for stack in self.fusion_stacks],
            "hypothesis_count": len(pseudo.view_names),
            "active_hypotheses_per_jet": (
                None
                if self.training
                else [
                    int(value)
                    for value in view_keep.sum(dim=1).detach().cpu().tolist()
                ]
            ),
            "view_dropout": self.view_dropout,
            "hierarchy_names": list(pseudo.hierarchy_names),
            "alpha": float(self.alpha.detach().cpu()),
            "beta": float(self.beta.detach().cpu()),
            "root_provenance": root_provenance,
            "fusion_kind": self.fusion_kind,
            "hierarchy_memory": self.use_hierarchy_memory,
            "ancestor_injection": self.use_ancestor_injection,
            "tree_bias": self.use_tree_bias,
            "uncertainty_gates": self.uncertainty_gates,
        }
        return HierarchyAwareTaggerOutput(
            logits=logits,
            baseline_logits=baseline_logits,
            representation=representation,
            baseline_representation=base_representation,
            diagnostics=diagnostics,
            auxiliary_logits={
                "pseudo": pseudo_logits,
                "hierarchy": self.hierarchy_aux_classifier(memory_representation),
            },
        )

    def set_all_residual_scales(self, value: float) -> None:
        with torch.no_grad():
            self.alpha.fill_(float(value))
            self.beta.fill_(float(value))
            for stack in self.fusion_stacks:
                for block in stack:
                    block.hlt_rezero.fill_(float(value))
                    block.pseudo_rezero.fill_(float(value))

    def calibration_report(
        self,
        hlt_tokens: Any,
        hlt_mask: Any,
        pseudo_views: PseudoViewInputs | DeployablePseudoViewBatch,
        *,
        tolerance: float = 1.0e-3,
    ) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self(hlt_tokens, hlt_mask, pseudo_views)
        if was_training:
            self.train()
        difference = (output.logits - output.baseline_logits).abs()
        mean_difference = float(difference.mean().cpu())
        top1_changes = int(
            (output.logits.argmax(dim=-1) != output.baseline_logits.argmax(dim=-1)).sum().cpu()
        )
        return {
            "ok": bool(mean_difference <= float(tolerance) and top1_changes == 0),
            "mean_absolute_logit_difference": mean_difference,
            "maximum_absolute_logit_difference": float(difference.max().cpu()),
            "top1_changes": top1_changes,
            "tolerance": float(tolerance),
        }

    def require_initial_calibration(
        self,
        hlt_tokens: Any,
        hlt_mask: Any,
        pseudo_views: PseudoViewInputs | DeployablePseudoViewBatch,
        *,
        tolerance: float = 1.0e-3,
    ) -> dict[str, Any]:
        report = self.calibration_report(
            hlt_tokens, hlt_mask, pseudo_views, tolerance=tolerance
        )
        if not report["ok"]:
            raise RuntimeError(
                "initial fused tagger violates baseline preservation: "
                f"mean_abs={report['mean_absolute_logit_difference']:.6g}, "
                f"top1_changes={report['top1_changes']}"
            )
        return report

    def fusion_gradient_report(self) -> dict[str, Any]:
        rows: dict[str, float] = {}
        for location, stack in zip(self.fusion_locations, self.fusion_stacks):
            for block_index, block in enumerate(stack):
                total = 0.0
                for parameter in block.parameters():
                    if parameter.grad is not None:
                        total += float(parameter.grad.detach().norm().cpu()) ** 2
                rows[f"layer_{location}.block_{block_index}"] = total ** 0.5
        return {
            "all_nonzero": bool(
                rows and all(np.isfinite(value) and value > 0.0 for value in rows.values())
            ),
            "grad_norms": rows,
        }


def _checkpoint_state_dict(path: str | Path) -> Mapping[str, Any]:
    _require_torch()
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older research-compute PyTorch
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint {path} is not a mapping")
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    if payload and all(isinstance(key, str) for key in payload):
        return payload
    raise KeyError(f"checkpoint {path} has no model state")


def _load_reference_state(backbone: Any, state: Mapping[str, Any], *, strict: bool) -> Any:
    module = getattr(backbone, "reference_model", backbone)
    target_keys = set(module.state_dict())
    candidates: list[dict[str, Any]] = [dict(state)]
    for prefix in ("module.", "model.", "tagger.", "hlt_branch.", "pseudo_branch."):
        candidates.append(
            {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        )
    best = max(candidates, key=lambda row: len(target_keys.intersection(row)))
    if not target_keys.intersection(best):
        raise ValueError("checkpoint has no parameters matching the requested ParT stream")
    return module.load_state_dict(best, strict=bool(strict))


def load_dual_stream_warm_starts(
    model: HierarchyAwareDualStreamTagger,
    *,
    hlt_checkpoint: str | Path,
    offline_checkpoint: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Load the selected A0 HLT and A4 offline checkpoints into separate streams."""

    hlt_result = _load_reference_state(
        model.hlt_backbone, _checkpoint_state_dict(hlt_checkpoint), strict=strict
    )
    pseudo_result = _load_reference_state(
        model.pseudo_backbone, _checkpoint_state_dict(offline_checkpoint), strict=strict
    )
    return {
        "hlt_checkpoint": str(Path(hlt_checkpoint).resolve()),
        "offline_checkpoint": str(Path(offline_checkpoint).resolve()),
        "hlt_missing_keys": list(hlt_result.missing_keys),
        "hlt_unexpected_keys": list(hlt_result.unexpected_keys),
        "pseudo_missing_keys": list(pseudo_result.missing_keys),
        "pseudo_unexpected_keys": list(pseudo_result.unexpected_keys),
        "pseudo_new_side_adapter_zero_initialized": True,
    }


def build_large_hierarchy_aware_tagger(
    *,
    num_classes: int = 10,
    dual_hierarchy: bool = False,
    independent_roots: bool = False,
) -> HierarchyAwareDualStreamTagger:
    """Build the locked 12-layer, 192-wide performance-first Step-10 model."""

    from jetclass_fresh.hlt_baseline import build_particle_transformer_classifier

    hlt = build_particle_transformer_classifier(num_classes=num_classes, model_size="large")
    pseudo = build_particle_transformer_classifier(num_classes=num_classes, model_size="large")
    return HierarchyAwareDualStreamTagger(
        hlt_backbone=WeaverStagewiseParticleTransformer(hlt),
        pseudo_backbone=WeaverStagewiseParticleTransformer(pseudo),
        num_classes=num_classes,
        fusion_dim=256,
        fusion_heads=8,
        fusion_blocks_per_location=2,
        fusion_locations=(4, 8, 12),
        view_aggregator_blocks=4,
        view_dropout=0.15,
        hypothesis_latent_dim=64,
        dropout=0.1,
        rezero_init=ABPH_PRIMARY_REZERO_INIT,
        dual_hierarchy=dual_hierarchy,
        independent_roots=independent_roots,
    )


def build_variant_hierarchy_aware_tagger(
    variant_name: str,
    *,
    num_classes: int = 10,
    smoke: bool = False,
) -> HierarchyAwareDualStreamTagger:
    """Build the concrete E/F/G0/G1 architecture declared by the registry."""

    from .variants import resolve_variant_config

    resolved = resolve_variant_config(variant_name)
    fusion = dict(resolved["model"]["fusion"])
    run_id = str(resolved["variant"]["run_id"])
    kind = str(fusion.get("kind", "bidirectional_dualcross"))
    if run_id == "E0":
        runtime_kind = "pseudo_only"
    elif kind in {"untrained_logit_mean", "late_representation"}:
        runtime_kind = kind
    elif kind == "single_cross_attention":
        runtime_kind = kind
    else:
        runtime_kind = "bidirectional_dualcross"
    if smoke:
        hlt_backbone = NativeStagewiseParticleTransformer(
            input_dim=17,
            model_dim=32,
            num_layers=12,
            num_heads=4,
            num_classes=num_classes,
        )
        pseudo_backbone = NativeStagewiseParticleTransformer(
            input_dim=17,
            model_dim=32,
            num_layers=12,
            num_heads=4,
            num_classes=num_classes,
        )
        fusion_dim = 32
        fusion_heads = 4
        aggregator_blocks = 1
    else:
        from jetclass_fresh.hlt_baseline import build_particle_transformer_classifier

        hlt = build_particle_transformer_classifier(
            num_classes=num_classes, model_size="large"
        )
        pseudo = build_particle_transformer_classifier(
            num_classes=num_classes, model_size="large"
        )
        hlt_backbone = WeaverStagewiseParticleTransformer(hlt)
        pseudo_backbone = WeaverStagewiseParticleTransformer(pseudo)
        fusion_dim = 256
        fusion_heads = 8
        aggregator_blocks = int(
            resolved["model"]["hierarchy_modules"]["view_aggregator_blocks"]
        )
    locations = tuple(
        int(hlt_backbone.num_layers) if value == "preclassification" else int(value)
        for value in fusion.get("locations", ())
    )
    if runtime_kind in {"pseudo_only", "untrained_logit_mean", "late_representation"}:
        locations = ()
    rezero = fusion.get("rezero_init", ABPH_PRIMARY_REZERO_INIT)
    if rezero is None:
        rezero = 1.0
    hierarchy_memory = bool(fusion.get("hierarchy_memory", True))
    use_hierarchy_structure = hierarchy_memory
    return HierarchyAwareDualStreamTagger(
        hlt_backbone=hlt_backbone,
        pseudo_backbone=pseudo_backbone,
        num_classes=num_classes,
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
        fusion_blocks_per_location=int(fusion.get("blocks_per_location", 2)),
        fusion_locations=locations,
        view_aggregator_blocks=aggregator_blocks,
        view_dropout=0.15,
        hypothesis_latent_dim=64,
        dropout=(0.0 if smoke else 0.1),
        rezero_init=float(rezero),
        dual_hierarchy=bool(fusion.get("dual_hierarchy", False)),
        independent_roots=(run_id == "E11"),
        fusion_kind=runtime_kind,
        use_hierarchy_memory=hierarchy_memory,
        use_ancestor_injection=use_hierarchy_structure,
        use_tree_bias=use_hierarchy_structure,
        uncertainty_gates=bool(fusion.get("uncertainty_gates", True)),
    )


__all__ = [
    "ABPH_HIERARCHY_TAGGER_CONTRACT",
    "ABPH_PRIMARY_FUSION_LOCATIONS",
    "ABPH_PRIMARY_HIERARCHY_NAMES",
    "ABPH_PRIMARY_REZERO_INIT",
    "HierarchyAwareDualStreamTagger",
    "HierarchyAwareTaggerOutput",
    "NativeStagewiseParticleTransformer",
    "ParticleStageState",
    "PseudoViewInputs",
    "TreeRelationBias",
    "WeaverStagewiseParticleTransformer",
    "build_large_hierarchy_aware_tagger",
    "build_variant_hierarchy_aware_tagger",
    "derive_particle_ancestors",
    "load_dual_stream_warm_starts",
]
