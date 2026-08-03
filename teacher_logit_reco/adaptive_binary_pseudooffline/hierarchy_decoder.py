"""Level-specific recursive decoder for adaptive pseudo-offline hierarchies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import require_torch

from .binary_accounting import (
    ABPH_AUXILIARY_ADDITIVE_NAMES,
    ABPH_BINARY_COUNT_SUPPORT,
    AccountingState,
    BinarySplitPrediction,
    CompiledBinarySplit,
    compile_binary_split,
)
from .root_transforms import ROOT_FEATURE_INDEX, ROOT_SHAPE_FEATURE_NAMES, wrap_phi_tensor
from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import (
    ABPH_LEVEL_CAPACITIES,
    ROOT_FEATURE_NAMES,
    TOPOLOGY_ACTIVE_SPLIT,
    TOPOLOGY_ACTIVE_TERMINAL,
    TOPOLOGY_PADDING,
)


try:  # Keep input/cache utilities importable without torch installed.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ABPH_HIERARCHY_DECODER_CONTRACT = "adaptive_binary_pseudooffline_hierarchy_decoder_v1"
ABPH_HLT_SUPPORT_FEATURE_NAMES: tuple[str, ...] = (
    "eta_hlt_relative",
    "phi_hlt_relative",
    "log_pt",
    *(f"pid_{name}" for name in ABPH_PID_CATEGORIES),
)
ABPH_GROUP_SUPPORT_NAMES: tuple[str, ...] = (
    "centroid_eta",
    "centroid_phi",
    "covariance_l11",
    "covariance_l21",
    "covariance_l22",
    "radial_q50",
    "radial_q80",
    "radial_q95",
    "maximum_member_radius",
    "principal_axis_sin",
    "principal_axis_cos",
)
ABPH_GROUP_SUPPORT_DIM = len(ABPH_GROUP_SUPPORT_NAMES)


@dataclass(frozen=True)
class RecursiveHierarchyDecoderConfig:
    hlt_input_dims: tuple[int, ...] = (192,)
    d_model: int = 256
    num_heads: int = 8
    ffn_dim: int = 1024
    blocks_per_level: int = 4
    dropout: float = 0.10
    attention_dropout: float = 0.10
    level_capacities: tuple[int, ...] = ABPH_LEVEL_CAPACITIES
    root_semantic_dim: int = 256
    latent_dim: int = 64
    maximum_attention_bias: float = 4.0

    def __post_init__(self) -> None:
        if not self.hlt_input_dims or any(int(value) <= 0 for value in self.hlt_input_dims):
            raise ValueError("hlt_input_dims must contain positive dimensions")
        for name in (
            "d_model",
            "num_heads",
            "ffn_dim",
            "blocks_per_level",
            "root_semantic_dim",
            "latent_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.d_model) % int(self.num_heads):
            raise ValueError("d_model must be divisible by num_heads")
        capacities = tuple(int(value) for value in self.level_capacities)
        if not capacities or capacities != ABPH_LEVEL_CAPACITIES[: len(capacities)]:
            raise ValueError(
                "level capacities must be a nonempty prefix of "
                f"{ABPH_LEVEL_CAPACITIES}"
            )
        for name in ("dropout", "attention_dropout"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if float(self.maximum_attention_bias) <= 0.0:
            raise ValueError("maximum_attention_bias must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "contract": ABPH_HIERARCHY_DECODER_CONTRACT,
                "hlt_support_features": list(ABPH_HLT_SUPPORT_FEATURE_NAMES),
                "group_support_features": list(ABPH_GROUP_SUPPORT_NAMES),
                "decoder_transitions": [
                    f"{1 if index == 0 else self.level_capacities[index - 1]}_to_{capacity}"
                    for index, capacity in enumerate(self.level_capacities)
                ],
                "global_hlt_access_at_every_level": True,
                "local_attention_policy": "bounded_support_bias_not_hard_crop",
            }
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["config_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload


@dataclass(frozen=True)
class HierarchyFrontier:
    ledger: Any
    hidden: Any
    support: Any
    uncertainty: Any
    mask: Any
    topology: Any
    parent_indices: Any
    source_child_indices: Any

    @property
    def batch_size(self) -> int:
        return int(self.mask.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.mask.shape[1])


@dataclass(frozen=True)
class HierarchyLevelOutput:
    depth: int
    parent_frontier: HierarchyFrontier
    parent_context: Any
    prediction: BinarySplitPrediction
    compiled: CompiledBinarySplit
    flat_parent_indices: Any
    next_frontier: HierarchyFrontier
    support_attention_bias: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RecursiveHierarchyOutput:
    mode: str
    root_frontier: HierarchyFrontier
    levels: tuple[HierarchyLevelOutput, ...]
    final_frontier: HierarchyFrontier
    diagnostics: Mapping[str, Any]


class _TypedHead(_ModuleBase):
    def __init__(self, d_model: int, ffn_dim: int, output_dim: int, dropout: float) -> None:
        torch = require_torch()
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, output_dim),
        )

    def forward(self, values: Any) -> Any:
        return self.network(values)


class _HierarchyDecoderBlock(_ModuleBase):
    def __init__(self, config: RecursiveHierarchyDecoderConfig) -> None:
        torch = require_torch()
        super().__init__()
        kwargs = {
            "embed_dim": config.d_model,
            "num_heads": config.num_heads,
            "dropout": config.attention_dropout,
            "batch_first": True,
        }
        self.num_heads = int(config.num_heads)
        self.peer_attention = torch.nn.MultiheadAttention(**kwargs)
        self.global_hlt_attention = torch.nn.MultiheadAttention(**kwargs)
        self.local_hlt_attention = torch.nn.MultiheadAttention(**kwargs)
        self.coarse_attention = torch.nn.MultiheadAttention(**kwargs)
        self.norm_peer = torch.nn.LayerNorm(config.d_model)
        self.norm_global = torch.nn.LayerNorm(config.d_model)
        self.norm_local = torch.nn.LayerNorm(config.d_model)
        self.norm_coarse = torch.nn.LayerNorm(config.d_model)
        self.norm_ffn = torch.nn.LayerNorm(config.d_model)
        self.gates = torch.nn.Parameter(torch.full((5,), 1.0e-3))
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.ffn_dim, config.d_model),
            torch.nn.Dropout(config.dropout),
        )

    def forward(
        self,
        parent: Any,
        parent_mask: Any,
        hlt: Any,
        hlt_mask: Any,
        coarse: Any,
        coarse_mask: Any,
        local_bias: Any,
    ) -> Any:
        mask_value = parent_mask.unsqueeze(-1).to(parent.dtype)
        query = self.norm_peer(parent)
        update, _ = self.peer_attention(
            query,
            query,
            query,
            key_padding_mask=~parent_mask,
            need_weights=False,
        )
        parent = (parent + self.gates[0] * update) * mask_value
        update, _ = self.global_hlt_attention(
            self.norm_global(parent),
            hlt,
            hlt,
            key_padding_mask=~hlt_mask,
            need_weights=False,
        )
        parent = (parent + self.gates[1] * update) * mask_value
        attention_mask = local_bias.repeat_interleave(self.num_heads, dim=0)
        local_padding_mask = require_torch().zeros(
            hlt_mask.shape, dtype=local_bias.dtype, device=hlt_mask.device
        ).masked_fill(~hlt_mask, float("-inf"))
        update, _ = self.local_hlt_attention(
            self.norm_local(parent),
            hlt,
            hlt,
            key_padding_mask=local_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        parent = (parent + self.gates[2] * update) * mask_value
        update, _ = self.coarse_attention(
            self.norm_coarse(parent),
            coarse,
            coarse,
            key_padding_mask=~coarse_mask,
            need_weights=False,
        )
        parent = (parent + self.gates[3] * update) * mask_value
        return (parent + self.gates[4] * self.ffn(self.norm_ffn(parent))) * mask_value


class _HierarchyLevelDecoder(_ModuleBase):
    def __init__(self, config: RecursiveHierarchyDecoderConfig, depth: int) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.depth = int(depth)
        self.blocks = torch.nn.ModuleList(
            [_HierarchyDecoderBlock(config) for _ in range(config.blocks_per_level)]
        )
        support_bias_dim = 1 + 2 + 1 + 1 + len(ABPH_PID_CATEGORIES) + 1 + 1
        self.support_bias_network = torch.nn.Sequential(
            torch.nn.Linear(support_bias_dim, config.d_model // 2),
            torch.nn.GELU(),
            torch.nn.Linear(config.d_model // 2, 1),
        )
        self.topology_head = _TypedHead(config.d_model, config.ffn_dim, 2, config.dropout)
        self.count_head = _TypedHead(
            config.d_model, config.ffn_dim, ABPH_BINARY_COUNT_SUPPORT, config.dropout
        )
        self.type_head = _TypedHead(
            config.d_model, config.ffn_dim, len(ABPH_PID_CATEGORIES), config.dropout
        )
        self.charge_head = _TypedHead(
            config.d_model, config.ffn_dim, 2 * ABPH_MAX_PARTICLES + 1, config.dropout
        )
        self.mass_head = _TypedHead(config.d_model, config.ffn_dim, 3, config.dropout)
        self.direction_head = _TypedHead(config.d_model, config.ffn_dim, 3, config.dropout)
        self.collinear_head = _TypedHead(config.d_model, config.ffn_dim, 1, config.dropout)
        self.auxiliary_head = _TypedHead(
            config.d_model,
            config.ffn_dim,
            len(ABPH_AUXILIARY_ADDITIVE_NAMES),
            config.dropout,
        )
        self.shape_head = _TypedHead(
            config.d_model,
            config.ffn_dim,
            2 * (len(ROOT_SHAPE_FEATURE_NAMES) + 1),
            config.dropout,
        )
        self.support_extra_head = _TypedHead(config.d_model, config.ffn_dim, 2 * 3, config.dropout)
        self.child_hidden_head = _TypedHead(
            config.d_model, config.ffn_dim, 2 * config.d_model, config.dropout
        )
        self.child_uncertainty_head = _TypedHead(
            config.d_model, config.ffn_dim, 2, config.dropout
        )
        self.child_ledger_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(len(ROOT_FEATURE_NAMES)),
            torch.nn.Linear(len(ROOT_FEATURE_NAMES), config.d_model),
        )

    def support_attention_bias(
        self,
        frontier: HierarchyFrontier,
        hlt_support_features: Any,
    ) -> Any:
        torch = require_torch()
        features = torch.as_tensor(hlt_support_features, device=frontier.hidden.device).float()
        if features.shape[:2] != (frontier.batch_size, hlt_support_features.shape[1]) or features.shape[-1] != len(ABPH_HLT_SUPPORT_FEATURE_NAMES):
            raise ValueError(
                f"HLT support features must have shape [B, N, {len(ABPH_HLT_SUPPORT_FEATURE_NAMES)}]"
            )
        support = frontier.support.float()
        delta_eta = features[:, None, :, 0] - support[:, :, None, 0]
        delta_phi = wrap_phi_tensor(features[:, None, :, 1] - support[:, :, None, 1])
        l11 = support[:, :, 2].abs().clamp_min(1.0e-3)[:, :, None]
        l21 = support[:, :, 3][:, :, None]
        l22 = support[:, :, 4].abs().clamp_min(1.0e-3)[:, :, None]
        first = delta_eta / l11
        second = (delta_phi - l21 * first) / l22
        mahalanobis = first.square() + second.square()
        delta_r = torch.sqrt(delta_eta.square() + delta_phi.square() + 1.0e-12)
        parent_px = frontier.ledger[:, :, ROOT_FEATURE_INDEX["px"]]
        parent_py = frontier.ledger[:, :, ROOT_FEATURE_INDEX["py"]]
        parent_log_pt = 0.5 * torch.log(parent_px.square() + parent_py.square() + 1.0e-8)
        relative_log_pt = features[:, None, :, 2] - parent_log_pt[:, :, None]
        pid = features[:, None, :, 3 : 3 + len(ABPH_PID_CATEGORIES)].expand(
            -1, frontier.capacity, -1, -1
        )
        depth = torch.full_like(delta_eta, float(self.depth) / len(ABPH_LEVEL_CAPACITIES))
        uncertainty = frontier.uncertainty[:, :, None].expand_as(delta_eta)
        raw = torch.cat(
            (
                mahalanobis[..., None],
                delta_eta[..., None],
                delta_phi[..., None],
                delta_r[..., None],
                relative_log_pt[..., None],
                pid,
                depth[..., None],
                uncertainty[..., None],
            ),
            dim=-1,
        )
        bounded = self.config.maximum_attention_bias * torch.tanh(
            self.support_bias_network(raw).squeeze(-1)
        )
        strength = 1.0 / (1.0 + frontier.uncertainty.clamp_min(0.0))
        return bounded * strength[:, :, None] * frontier.mask[:, :, None].to(bounded.dtype)

    def _prediction(self, flat_context: Any) -> tuple[BinarySplitPrediction, Any, Any, Any]:
        support_extra = self.support_extra_head(flat_context).reshape(-1, 2, 3)
        child_hidden = self.child_hidden_head(flat_context).reshape(
            -1, 2, self.config.d_model
        )
        child_uncertainty = require_torch().nn.functional.softplus(
            self.child_uncertainty_head(flat_context)
        )
        prediction = BinarySplitPrediction(
            topology_logits=self.topology_head(flat_context),
            count_logits=self.count_head(flat_context),
            type_allocation_logits=self.type_head(flat_context),
            charge_logits=self.charge_head(flat_context),
            mass_allocation_logits=self.mass_head(flat_context),
            direction_raw=self.direction_head(flat_context),
            collinear_fraction_raw=self.collinear_head(flat_context).squeeze(-1),
            auxiliary_fraction_logits=self.auxiliary_head(flat_context),
            child_shape_raw=self.shape_head(flat_context).reshape(
                -1, 2, len(ROOT_SHAPE_FEATURE_NAMES) + 1
            ),
        )
        return prediction, support_extra, child_hidden, child_uncertainty

    def forward(
        self,
        frontier: HierarchyFrontier,
        hlt_evidence: Any,
        hlt_mask: Any,
        hlt_support_features: Any,
        coarse_memory: Any,
        coarse_mask: Any,
        *,
        next_capacity: int,
        coherent_hypothesis_context: Any | None = None,
        topology_override: Any | None = None,
        final_frontier: bool = False,
    ) -> HierarchyLevelOutput:
        torch = require_torch()
        local_bias = self.support_attention_bias(frontier, hlt_support_features)
        context = frontier.hidden
        if coherent_hypothesis_context is not None:
            latent_context = torch.as_tensor(
                coherent_hypothesis_context,
                device=context.device,
                dtype=context.dtype,
            )
            if latent_context.shape != (frontier.batch_size, self.config.d_model):
                raise ValueError(
                    f"coherent hypothesis context must have shape [B, {self.config.d_model}]"
                )
            context = context + latent_context[:, None, :] * frontier.mask.unsqueeze(-1)
        for block in self.blocks:
            context = block(
                context,
                frontier.mask,
                hlt_evidence,
                hlt_mask,
                coarse_memory,
                coarse_mask,
                local_bias,
            )
        flat_parent_indices = torch.nonzero(frontier.mask, as_tuple=False)
        if not int(flat_parent_indices.shape[0]):
            raise ValueError("hierarchy level has no active parent")
        flat_context = context[frontier.mask]
        prediction, support_extra, child_hidden_delta, child_uncertainty = self._prediction(
            flat_context
        )
        parent = AccountingState.from_ledger(frontier.ledger[frontier.mask])
        flat_override = None
        if topology_override is not None:
            supplied = torch.as_tensor(topology_override, device=frontier.mask.device).to(torch.long)
            if supplied.shape != frontier.mask.shape:
                raise ValueError("topology override must match the parent frontier")
            flat_override = supplied[frontier.mask]
        else:
            # A hard stop is absorbing. Newly created children may choose their
            # own topology, but a terminal carried from an earlier depth cannot
            # silently reopen later in rollout.
            flat_override = prediction.topology_logits.argmax(dim=-1).to(torch.long) + 1
            flat_count = frontier.ledger[
                frontier.mask, ROOT_FEATURE_INDEX["constituent_count"]
            ].round().to(torch.long)
            carried_terminal = (
                (frontier.parent_indices[frontier.mask] >= 0)
                & (frontier.source_child_indices[frontier.mask] == -1)
                & (frontier.topology[frontier.mask] == int(TOPOLOGY_ACTIVE_TERMINAL))
            )
            flat_override = torch.where(
                (flat_count <= 1) | carried_terminal,
                torch.full_like(flat_override, int(TOPOLOGY_ACTIVE_TERMINAL)),
                flat_override,
            )
        compiled = compile_binary_split(parent, prediction, topology_override=flat_override)
        next_frontier = self._pack_next_frontier(
            frontier,
            context,
            compiled,
            flat_parent_indices,
            support_extra,
            child_hidden_delta,
            child_uncertainty,
            next_capacity=int(next_capacity),
            final_frontier=bool(final_frontier),
        )
        return HierarchyLevelOutput(
            depth=self.depth,
            parent_frontier=frontier,
            parent_context=context,
            prediction=prediction,
            compiled=compiled,
            flat_parent_indices=flat_parent_indices,
            next_frontier=next_frontier,
            support_attention_bias=local_bias,
            diagnostics={
                "depth": self.depth,
                "n_active_parents": int(frontier.mask.sum().detach().cpu()),
                "n_output_groups": int(next_frontier.mask.sum().detach().cpu()),
                "global_hlt_attention_used": True,
                "local_hlt_attention_used": True,
                "coarse_memory_attention_used": True,
                "coherent_jet_latent_injected_directly": coherent_hypothesis_context is not None,
                "node_local_noise_used": False,
                "carried_terminal_reopened": False,
                "compiler": compiled.diagnostics,
            },
        )

    def _pack_next_frontier(
        self,
        frontier: HierarchyFrontier,
        context: Any,
        compiled: CompiledBinarySplit,
        flat_parent_indices: Any,
        support_extra: Any,
        child_hidden_delta: Any,
        child_uncertainty: Any,
        *,
        next_capacity: int,
        final_frontier: bool,
    ) -> HierarchyFrontier:
        torch = require_torch()
        batch = frontier.batch_size
        device = frontier.ledger.device
        dtype = frontier.ledger.dtype
        ledger = torch.zeros(
            (batch, next_capacity, len(ROOT_FEATURE_NAMES)), dtype=dtype, device=device
        )
        hidden = torch.zeros(
            (batch, next_capacity, self.config.d_model), dtype=context.dtype, device=device
        )
        support = torch.zeros(
            (batch, next_capacity, ABPH_GROUP_SUPPORT_DIM), dtype=dtype, device=device
        )
        uncertainty = torch.zeros((batch, next_capacity), dtype=dtype, device=device)
        mask = torch.zeros((batch, next_capacity), dtype=torch.bool, device=device)
        topology = torch.full(
            (batch, next_capacity), int(TOPOLOGY_PADDING), dtype=torch.long, device=device
        )
        parent_indices = torch.full(
            (batch, next_capacity), -1, dtype=torch.long, device=device
        )
        source_child_indices = torch.full_like(parent_indices, -2)
        write_offsets = [0 for _ in range(batch)]
        for flat_index in range(int(flat_parent_indices.shape[0])):
            batch_index = int(flat_parent_indices[flat_index, 0])
            parent_index = int(flat_parent_indices[flat_index, 1])
            if bool(compiled.split_mask[flat_index]):
                for child_index in range(2):
                    output_index = write_offsets[batch_index]
                    if output_index >= next_capacity:
                        raise RuntimeError("recursive hierarchy exceeded its fixed level capacity")
                    child_ledger = compiled.child_ledger[flat_index, child_index]
                    ledger[batch_index, output_index] = child_ledger.to(ledger.dtype)
                    child_support = support_from_ledger(
                        child_ledger[None, :], support_extra[flat_index, child_index][None, :]
                    )[0]
                    support[batch_index, output_index] = child_support.to(support.dtype)
                    child_hidden = (
                        context[batch_index, parent_index]
                        + child_hidden_delta[flat_index, child_index]
                        + self.child_ledger_projection(child_ledger[None, :])[0]
                    )
                    hidden[batch_index, output_index] = child_hidden.to(hidden.dtype)
                    uncertainty[batch_index, output_index] = child_uncertainty[
                        flat_index, child_index
                    ].to(uncertainty.dtype)
                    count = int(compiled.child_constituent_count[flat_index, child_index])
                    topology[batch_index, output_index] = int(
                        TOPOLOGY_ACTIVE_TERMINAL
                        if final_frontier or count <= 1
                        else TOPOLOGY_ACTIVE_SPLIT
                    )
                    parent_indices[batch_index, output_index] = parent_index
                    source_child_indices[batch_index, output_index] = child_index
                    mask[batch_index, output_index] = True
                    write_offsets[batch_index] += 1
            else:
                output_index = write_offsets[batch_index]
                if output_index >= next_capacity:
                    raise RuntimeError("terminal carries exceeded their fixed level capacity")
                ledger[batch_index, output_index] = frontier.ledger[
                    batch_index, parent_index
                ].to(ledger.dtype)
                hidden[batch_index, output_index] = context[
                    batch_index, parent_index
                ].to(hidden.dtype)
                support[batch_index, output_index] = frontier.support[
                    batch_index, parent_index
                ].to(support.dtype)
                uncertainty[batch_index, output_index] = frontier.uncertainty[
                    batch_index, parent_index
                ].to(uncertainty.dtype)
                topology[batch_index, output_index] = int(TOPOLOGY_ACTIVE_TERMINAL)
                parent_indices[batch_index, output_index] = parent_index
                source_child_indices[batch_index, output_index] = -1
                mask[batch_index, output_index] = True
                write_offsets[batch_index] += 1
        return HierarchyFrontier(
            ledger=ledger,
            hidden=hidden * mask.unsqueeze(-1),
            support=support,
            uncertainty=uncertainty,
            mask=mask,
            topology=topology,
            parent_indices=parent_indices,
            source_child_indices=source_child_indices,
        )


def support_from_ledger(ledger: Any, extra: Any | None = None) -> Any:
    """Derive bounded group support from compiled shape state plus three extras."""

    torch = require_torch()
    values = torch.as_tensor(ledger)
    if values.ndim != 2 or values.shape[-1] != len(ROOT_FEATURE_NAMES):
        raise ValueError("ledger must have shape [B, root_feature_dim]")
    if extra is None:
        extras = torch.zeros((values.shape[0], 3), dtype=values.dtype, device=values.device)
    else:
        extras = torch.as_tensor(extra, device=values.device, dtype=values.dtype)
        if extras.shape != (values.shape[0], 3):
            raise ValueError("support extras must have shape [B, 3]")
    centroid_eta = values[:, ROOT_FEATURE_INDEX["eta_first_moment"]]
    centroid_phi = wrap_phi_tensor(values[:, ROOT_FEATURE_INDEX["phi_first_moment"]])
    covariance = torch.stack(
        tuple(
            values[:, ROOT_FEATURE_INDEX[f"covariance_cholesky[{index}]"]]
            for index in range(3)
        ),
        dim=-1,
    )
    radial = torch.stack(
        tuple(
            values[:, ROOT_FEATURE_INDEX[f"radial_quantiles[{index}]"]]
            for index in range(3)
        ),
        dim=-1,
    ).clamp_min(0.0)
    maximum_radius = radial[:, 2] + torch.nn.functional.softplus(extras[:, 0])
    orientation = extras[:, 1:3]
    orientation_norm = orientation.norm(dim=-1, keepdim=True)
    default = torch.cat((torch.zeros_like(orientation[:, :1]), torch.ones_like(orientation[:, :1])), dim=-1)
    orientation = torch.where(
        orientation_norm > 1.0e-8,
        orientation / orientation_norm.clamp_min(1.0e-8),
        default,
    )
    return torch.cat(
        (
            centroid_eta[:, None],
            centroid_phi[:, None],
            covariance,
            radial,
            maximum_radius[:, None],
            orientation,
        ),
        dim=-1,
    )


class RecursiveHierarchyDecoder(_ModuleBase):
    """Five specialized binary transitions with fresh HLT access at every depth."""

    def __init__(
        self,
        config: RecursiveHierarchyDecoderConfig | Mapping[str, Any] | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        if config is None:
            resolved = RecursiveHierarchyDecoderConfig()
        elif isinstance(config, RecursiveHierarchyDecoderConfig):
            resolved = config
        else:
            resolved = RecursiveHierarchyDecoderConfig(**dict(config))
        self.config = resolved
        self.hlt_projections = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(input_dim),
                    torch.nn.Linear(input_dim, resolved.d_model),
                )
                for input_dim in resolved.hlt_input_dims
            ]
        )
        self.hlt_source_embedding = torch.nn.Parameter(
            torch.zeros(len(resolved.hlt_input_dims), resolved.d_model)
        )
        torch.nn.init.trunc_normal_(self.hlt_source_embedding, std=0.02)
        self.root_hidden_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(resolved.d_model),
            torch.nn.Linear(resolved.d_model, resolved.d_model),
        )
        self.root_semantic_projection = torch.nn.Linear(
            resolved.root_semantic_dim, resolved.d_model
        )
        self.root_ledger_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(len(ROOT_FEATURE_NAMES)),
            torch.nn.Linear(len(ROOT_FEATURE_NAMES), resolved.d_model),
        )
        self.support_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(ABPH_GROUP_SUPPORT_DIM),
            torch.nn.Linear(ABPH_GROUP_SUPPORT_DIM, resolved.d_model),
        )
        self.uncertainty_projection = torch.nn.Linear(1, resolved.d_model)
        self.latent_projection = torch.nn.Linear(resolved.latent_dim, resolved.d_model)
        self.depth_embedding = torch.nn.Embedding(
            len(resolved.level_capacities) + 1, resolved.d_model
        )
        self.levels = torch.nn.ModuleList(
            [
                _HierarchyLevelDecoder(resolved, depth=index + 1)
                for index in range(len(resolved.level_capacities))
            ]
        )

    def _project_hlt(self, embeddings: Any, mask: Any) -> Any:
        torch = require_torch()
        sources = (embeddings,) if not isinstance(embeddings, (tuple, list)) else tuple(embeddings)
        if len(sources) != len(self.hlt_projections):
            raise ValueError(
                f"expected {len(self.hlt_projections)} HLT embedding depths, got {len(sources)}"
            )
        projected = []
        expected_prefix = None
        for index, (source, projection, input_dim) in enumerate(
            zip(sources, self.hlt_projections, self.config.hlt_input_dims)
        ):
            values = torch.as_tensor(source)
            if values.ndim != 3 or values.shape[-1] != input_dim:
                raise ValueError(f"HLT source {index} must have shape [B, N, {input_dim}]")
            if expected_prefix is None:
                expected_prefix = values.shape[:2]
            elif values.shape[:2] != expected_prefix:
                raise ValueError("all HLT embedding depths must share batch and particle axes")
            projected.append(projection(values) + self.hlt_source_embedding[index][None, None, :])
        hlt_mask = torch.as_tensor(mask, device=projected[0].device).bool()
        if tuple(hlt_mask.shape) != tuple(expected_prefix):
            raise ValueError("HLT mask does not match embedding axes")
        if not bool(hlt_mask.any(dim=1).all()):
            raise ValueError("every hierarchy example requires at least one valid HLT particle")
        return torch.stack(projected, dim=0).mean(dim=0) * hlt_mask.unsqueeze(-1)

    def _root_frontier(
        self,
        root_state: AccountingState,
        root_hidden: Any,
        hypothesis_latent: Any | None,
    ) -> HierarchyFrontier:
        torch = require_torch()
        batch = root_state.batch_size
        hidden = torch.as_tensor(root_hidden, device=root_state.ledger.device)
        if hidden.shape != (batch, self.config.d_model):
            raise ValueError(f"root_hidden must have shape [B, {self.config.d_model}]")
        support = support_from_ledger(root_state.ledger)
        hidden = self.root_hidden_projection(hidden) + self.root_ledger_projection(root_state.ledger)
        hidden = hidden + self.support_projection(support)
        if hypothesis_latent is not None:
            latent = torch.as_tensor(hypothesis_latent, device=hidden.device, dtype=hidden.dtype)
            if latent.shape != (batch, self.config.latent_dim):
                raise ValueError(f"hypothesis latent must have shape [B, {self.config.latent_dim}]")
            hidden = hidden + self.latent_projection(latent)
        mask = torch.ones((batch, 1), dtype=torch.bool, device=hidden.device)
        topology = torch.where(
            root_state.constituent_count[:, None] > 1,
            torch.full((batch, 1), int(TOPOLOGY_ACTIVE_SPLIT), dtype=torch.long, device=hidden.device),
            torch.full((batch, 1), int(TOPOLOGY_ACTIVE_TERMINAL), dtype=torch.long, device=hidden.device),
        )
        return HierarchyFrontier(
            ledger=root_state.ledger[:, None, :],
            hidden=hidden[:, None, :],
            support=support[:, None, :],
            uncertainty=torch.zeros((batch, 1), dtype=hidden.dtype, device=hidden.device),
            mask=mask,
            topology=topology,
            parent_indices=torch.full((batch, 1), -1, dtype=torch.long, device=hidden.device),
            source_child_indices=torch.full((batch, 1), -1, dtype=torch.long, device=hidden.device),
        )

    def _teacher_parent_frontier(
        self,
        ledger: Any,
        support: Any,
        mask: Any,
        topology: Any,
        parent_indices: Any,
        root_context: Any,
        depth: int,
    ) -> HierarchyFrontier:
        torch = require_torch()
        values = torch.as_tensor(ledger, device=root_context.device).float()
        supports = torch.as_tensor(support, device=root_context.device).float()
        valid = torch.as_tensor(mask, device=root_context.device).bool()
        states = torch.as_tensor(topology, device=root_context.device).to(torch.long)
        parents = torch.as_tensor(parent_indices, device=root_context.device).to(torch.long)
        if values.shape[:2] != valid.shape or values.shape[-1] != len(ROOT_FEATURE_NAMES):
            raise ValueError("teacher parent ledger/mask shapes are inconsistent")
        if supports.shape != (*valid.shape, ABPH_GROUP_SUPPORT_DIM):
            raise ValueError("teacher support shape is inconsistent")
        if states.shape != valid.shape or parents.shape != valid.shape:
            raise ValueError("teacher topology/parent shapes are inconsistent")
        hidden = (
            self.root_ledger_projection(values)
            + self.support_projection(supports)
            + root_context[:, None, :]
            + self.depth_embedding.weight[int(depth)][None, None, :]
        )
        hidden = hidden * valid.unsqueeze(-1)
        return HierarchyFrontier(
            ledger=values,
            hidden=hidden,
            support=supports,
            uncertainty=torch.zeros(valid.shape, dtype=values.dtype, device=values.device),
            mask=valid,
            topology=states,
            parent_indices=parents,
            source_child_indices=torch.full_like(parents, -1),
        )

    def forward(
        self,
        root_state: AccountingState,
        root_hidden: Any,
        root_semantic_tokens: Any,
        hlt_particle_embeddings: Any,
        hlt_particle_mask: Any,
        hlt_support_features: Any,
        *,
        mode: str = "rollout",
        hypothesis_latent: Any | None = None,
        teacher_parent_frontiers: Sequence[HierarchyFrontier] | None = None,
    ) -> RecursiveHierarchyOutput:
        torch = require_torch()
        resolved_mode = str(mode).strip().lower().replace("-", "_")
        if resolved_mode not in {"rollout", "teacher_forced"}:
            raise ValueError("mode must be rollout or teacher_forced")
        hlt_evidence = self._project_hlt(hlt_particle_embeddings, hlt_particle_mask)
        hlt_mask = torch.as_tensor(hlt_particle_mask, device=hlt_evidence.device).bool()
        hlt_support = torch.as_tensor(
            hlt_support_features, device=hlt_evidence.device, dtype=hlt_evidence.dtype
        )
        if hlt_support.shape != (
            hlt_evidence.shape[0],
            hlt_evidence.shape[1],
            len(ABPH_HLT_SUPPORT_FEATURE_NAMES),
        ):
            raise ValueError("HLT support feature shape does not match the locked contract")
        root_tokens = torch.as_tensor(root_semantic_tokens, device=hlt_evidence.device)
        if root_tokens.ndim != 3 or root_tokens.shape[0] != root_state.batch_size or root_tokens.shape[-1] != self.config.root_semantic_dim:
            raise ValueError(
                f"root semantic tokens must have shape [B, Q, {self.config.root_semantic_dim}]"
            )
        root_memory = self.root_semantic_projection(root_tokens)
        root_memory_mask = torch.ones(root_memory.shape[:2], dtype=torch.bool, device=root_memory.device)
        root_frontier = self._root_frontier(root_state, root_hidden, hypothesis_latent)
        coherent_hypothesis_context = None
        if hypothesis_latent is not None:
            latent = torch.as_tensor(
                hypothesis_latent,
                device=root_frontier.hidden.device,
                dtype=root_frontier.hidden.dtype,
            )
            if latent.shape != (root_state.batch_size, self.config.latent_dim):
                raise ValueError(
                    f"hypothesis latent must have shape [B, {self.config.latent_dim}]"
                )
            coherent_hypothesis_context = self.latent_projection(latent)
        if resolved_mode == "teacher_forced":
            if teacher_parent_frontiers is None or len(teacher_parent_frontiers) != len(self.levels):
                raise ValueError("teacher-forced mode requires one parent frontier per decoder level")
        elif teacher_parent_frontiers is not None:
            raise ValueError("rollout mode cannot consume teacher parent frontiers")
        frontier = root_frontier
        root_context = root_frontier.hidden[:, 0]
        memories = [root_memory]
        memory_masks = [root_memory_mask]
        outputs: list[HierarchyLevelOutput] = []
        hlt_access_count = 0
        for level_index, level in enumerate(self.levels):
            if resolved_mode == "teacher_forced":
                supplied = teacher_parent_frontiers[level_index]
                frontier = self._teacher_parent_frontier(
                    supplied.ledger,
                    supplied.support,
                    supplied.mask,
                    supplied.topology,
                    supplied.parent_indices,
                    root_context,
                    level_index,
                )
            coarse = torch.cat(memories, dim=1)
            coarse_mask = torch.cat(memory_masks, dim=1)
            override = frontier.topology if resolved_mode == "teacher_forced" else None
            output = level(
                frontier,
                hlt_evidence,
                hlt_mask,
                hlt_support,
                coarse,
                coarse_mask,
                next_capacity=self.config.level_capacities[level_index],
                coherent_hypothesis_context=coherent_hypothesis_context,
                topology_override=override,
                final_frontier=level_index == len(self.levels) - 1,
            )
            outputs.append(output)
            frontier = output.next_frontier
            memories.append(frontier.hidden)
            memory_masks.append(frontier.mask)
            hlt_access_count += int(output.diagnostics["global_hlt_attention_used"])
        return RecursiveHierarchyOutput(
            mode=resolved_mode,
            root_frontier=root_frontier,
            levels=tuple(outputs),
            final_frontier=frontier,
            diagnostics={
                "contract": ABPH_HIERARCHY_DECODER_CONTRACT,
                "mode": resolved_mode,
                "level_capacities": list(self.config.level_capacities),
                "hlt_evidence_access_count": hlt_access_count,
                "all_levels_accessed_original_hlt": hlt_access_count == len(self.levels),
                "teacher_inputs_consumed": resolved_mode == "teacher_forced",
                "offline_inputs_consumed_in_rollout": False,
                "coherent_latent_injected_at_every_level": (
                    coherent_hypothesis_context is not None
                    and all(
                        bool(level.diagnostics["coherent_jet_latent_injected_directly"])
                        for level in outputs
                    )
                ),
                "node_local_noise_used": False,
            },
        )


__all__ = [
    "ABPH_GROUP_SUPPORT_DIM",
    "ABPH_GROUP_SUPPORT_NAMES",
    "ABPH_HIERARCHY_DECODER_CONTRACT",
    "ABPH_HLT_SUPPORT_FEATURE_NAMES",
    "HierarchyFrontier",
    "HierarchyLevelOutput",
    "RecursiveHierarchyDecoder",
    "RecursiveHierarchyDecoderConfig",
    "RecursiveHierarchyOutput",
    "support_from_ledger",
]
