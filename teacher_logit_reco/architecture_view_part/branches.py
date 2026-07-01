"""PN/PFN/PCNN per-particle architecture-view branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import (
    ARCHITECTURE_VIEW_BRANCHES,
    ARCHITECTURE_VIEW_BRANCH_PCNN,
    ARCHITECTURE_VIEW_BRANCH_PFN,
    ARCHITECTURE_VIEW_BRANCH_PN,
    ArchitectureViewConfig,
    normalize_architecture_view_branch,
)
from .inputs import sanitize_architecture_view_tokens, wrap_architecture_view_phi

try:  # Keep package imports cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ARCHITECTURE_VIEW_BRANCHES_CONTRACT = "architecture_view_branches_v1"


@dataclass(frozen=True)
class ArchitectureViewBranchOutput:
    """One branch's per-particle view tensor."""

    name: str
    embeddings: Any
    mask: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        valid = self.mask.to(dtype=self.embeddings.dtype)
        denom = valid.sum().clamp_min(1.0)
        norm = self.embeddings.norm(dim=-1)
        return {
            "contract": ARCHITECTURE_VIEW_BRANCHES_CONTRACT,
            "name": self.name,
            "embeddings_shape": list(self.embeddings.shape),
            "valid_particle_count": int(self.mask.sum().detach().cpu().item()),
            "mean_embedding_norm": float(((norm * valid).sum() / denom).detach().cpu().item()),
            **self.diagnostics,
        }


def _mlp(sizes: tuple[int, ...], *, dropout: float = 0.0, final_activation: bool = False) -> Any:
    torch = require_torch()
    layers = []
    for index, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(torch.nn.Linear(int(din), int(dout)))
        is_last = index == len(sizes) - 2
        if (not is_last) or final_activation:
            layers.append(torch.nn.GELU())
            if float(dropout) > 0.0:
                layers.append(torch.nn.Dropout(float(dropout)))
    return torch.nn.Sequential(*layers)


def _masked_zero(value: Any, mask: Any) -> Any:
    return value * mask[:, :, None].to(dtype=value.dtype)


def _coerce_prepared_inputs(tokens: Any, mask: Any | None = None) -> tuple[Any, Any]:
    prepared = sanitize_architecture_view_tokens(tokens, mask)
    return prepared.tokens, prepared.mask


def _pairwise_knn(tokens: Any, mask: Any, config: ArchitectureViewConfig) -> tuple[Any, Any, Any, Any]:
    """Return neighbor indices/masks plus pairwise geometry for PN view."""

    torch = require_torch()
    batch_size, num_particles, _ = tokens.shape
    if int(num_particles) == 0:
        empty_idx = torch.empty(batch_size, 0, int(config.pn_k), dtype=torch.long, device=tokens.device)
        empty_mask = torch.empty(batch_size, 0, int(config.pn_k), dtype=torch.bool, device=tokens.device)
        empty_edge = tokens.new_empty(batch_size, 0, int(config.pn_k), 5)
        empty_dist = tokens.new_empty(batch_size, 0, int(config.pn_k))
        return empty_idx, empty_mask, empty_edge, empty_dist

    eta = tokens[:, :, int(config.eta_index)]
    phi = tokens[:, :, int(config.phi_index)]
    log_pt = torch.log(torch.clamp(tokens[:, :, int(config.pt_index)], min=float(config.eps)))
    deta = eta[:, :, None] - eta[:, None, :]
    dphi = wrap_architecture_view_phi(phi[:, :, None] - phi[:, None, :])
    delta_r = torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=0.0))
    delta_log_pt = log_pt[:, :, None] - log_pt[:, None, :]

    candidate_mask = mask[:, None, :] & mask[:, :, None]
    eye = torch.eye(num_particles, dtype=torch.bool, device=tokens.device)[None, :, :]
    candidate_mask = candidate_mask & ~eye
    large = torch.finfo(delta_r.dtype).max / 16.0
    masked_dist = delta_r.masked_fill(~candidate_mask, large)
    topk_count = min(int(config.pn_k), int(num_particles))
    _, indices = torch.topk(masked_dist, k=topk_count, dim=-1, largest=False, sorted=True)
    neighbor_mask = torch.gather(candidate_mask, dim=2, index=indices) & mask[:, :, None]
    fallback_index = candidate_mask.to(dtype=torch.long).argmax(dim=2, keepdim=True)
    indices = torch.where(neighbor_mask, indices, fallback_index.expand_as(indices))
    if topk_count < int(config.pn_k):
        pad = int(config.pn_k) - topk_count
        indices = torch.cat([indices, fallback_index.expand(-1, -1, pad)], dim=-1)
        neighbor_mask = torch.cat(
            [neighbor_mask, torch.zeros_like(neighbor_mask[:, :, :1]).expand(-1, -1, pad)],
            dim=-1,
        )
    indices = torch.where(mask[:, :, None], indices, torch.zeros_like(indices)).long()
    neighbor_mask = neighbor_mask & mask[:, :, None]
    selected_dist = torch.gather(delta_r, dim=2, index=indices)
    selected_edge = torch.stack(
        [
            torch.gather(deta, dim=2, index=indices),
            torch.gather(dphi, dim=2, index=indices),
            selected_dist,
            torch.log(torch.clamp(selected_dist, min=float(config.eps))),
            torch.gather(delta_log_pt, dim=2, index=indices),
        ],
        dim=-1,
    )
    selected_edge = torch.where(neighbor_mask[:, :, :, None], selected_edge, torch.zeros_like(selected_edge))
    selected_dist = torch.where(neighbor_mask, selected_dist, torch.zeros_like(selected_dist))
    return indices, neighbor_mask, selected_edge, selected_dist


def _gather_neighbors(state: Any, indices: Any) -> Any:
    batch_size, num_particles, hidden_dim = state.shape
    k = int(indices.shape[-1])
    expanded = state[:, None, :, :].expand(batch_size, num_particles, num_particles, hidden_dim)
    gather_index = indices[:, :, :, None].expand(batch_size, num_particles, k, hidden_dim)
    return torch_gather(expanded, 2, gather_index)


def torch_gather(value: Any, dim: int, index: Any) -> Any:
    torch = require_torch()
    return torch.gather(value, dim=dim, index=index)


class PNMessageLayer(_ModuleBase):
    """One EdgeConv-style local update layer."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        torch = require_torch()
        self.message = _mlp((hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim), dropout=dropout)
        self.update = _mlp((hidden_dim * 3, hidden_dim, hidden_dim), dropout=dropout)
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, state: Any, edge_features: Any, indices: Any, neighbor_mask: Any, mask: Any) -> Any:
        torch = require_torch()
        neighbors = _gather_neighbors(state, indices)
        center = state[:, :, None, :].expand_as(neighbors)
        message_input = torch.cat([center, neighbors - center, edge_features], dim=-1)
        messages = self.message(message_input)
        messages = torch.where(neighbor_mask[:, :, :, None], messages, torch.zeros_like(messages))
        valid_counts = neighbor_mask.sum(dim=-1, keepdim=True).to(dtype=state.dtype)
        mean_msg = messages.sum(dim=2) / valid_counts.clamp_min(1.0)
        very_negative = torch.finfo(messages.dtype).min / 4.0
        max_input = messages.masked_fill(~neighbor_mask[:, :, :, None], very_negative)
        max_msg = max_input.max(dim=2).values
        max_msg = torch.where(valid_counts > 0, max_msg, torch.zeros_like(max_msg))
        update = self.update(torch.cat([state, max_msg, mean_msg], dim=-1))
        return _masked_zero(self.norm(state + update), mask)


class PNViewBranch(_ModuleBase):
    """ParticleNet-style local eta-phi graph view branch."""

    def __init__(self, config: ArchitectureViewConfig | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = config or ArchitectureViewConfig()
        self.stem = _mlp(
            (int(self.config.raw_token_dim), int(self.config.hidden_dim), int(self.config.hidden_dim)),
            dropout=float(self.config.dropout),
        )
        self.layers = torch.nn.ModuleList(
            [
                PNMessageLayer(
                    hidden_dim=int(self.config.hidden_dim),
                    edge_dim=5,
                    dropout=float(self.config.dropout),
                )
                for _ in range(int(self.config.pn_layers))
            ]
        )
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.hidden_dim)),
            torch.nn.Linear(int(self.config.hidden_dim), int(self.config.view_dim)),
        )

    def forward(self, tokens: Any, mask: Any | None = None) -> ArchitectureViewBranchOutput:
        tokens, mask = _coerce_prepared_inputs(tokens, mask)
        indices, neighbor_mask, edge_features, distances = _pairwise_knn(tokens, mask, self.config)
        state = _masked_zero(self.stem(tokens), mask)
        for layer in self.layers:
            state = layer(state, edge_features, indices, neighbor_mask, mask)
        embeddings = _masked_zero(self.proj(state), mask)
        valid_queries = mask.sum().clamp_min(1).to(dtype=tokens.dtype)
        diagnostics = {
            "mean_valid_neighbors": float(
                (neighbor_mask.sum(dim=-1).to(dtype=tokens.dtype) * mask.to(dtype=tokens.dtype)).sum()
                .div(valid_queries)
                .detach()
                .cpu()
                .item()
            ),
            "mean_neighbor_delta_r": float(
                distances.sum()
                .div(neighbor_mask.sum().clamp_min(1).to(dtype=tokens.dtype))
                .detach()
                .cpu()
                .item()
            ),
        }
        return ArchitectureViewBranchOutput(
            name=ARCHITECTURE_VIEW_BRANCH_PN,
            embeddings=embeddings,
            mask=mask,
            diagnostics=diagnostics,
        )


class PFNViewBranch(_ModuleBase):
    """PFN-style particle MLP with optional global jet summary conditioning."""

    def __init__(self, config: ArchitectureViewConfig | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = config or ArchitectureViewConfig()
        hidden = int(self.config.pfn_hidden_dim)
        self.local_phi = _mlp((int(self.config.raw_token_dim), hidden, hidden), dropout=float(self.config.dropout))
        self.global_phi = _mlp((int(self.config.raw_token_dim), hidden, hidden), dropout=float(self.config.dropout))
        out_input = hidden * (2 if bool(self.config.pfn_use_global_context) else 1)
        self.out = torch.nn.Sequential(
            torch.nn.LayerNorm(out_input),
            torch.nn.Linear(out_input, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(hidden, int(self.config.view_dim)),
        )

    def forward(self, tokens: Any, mask: Any | None = None) -> ArchitectureViewBranchOutput:
        tokens, mask = _coerce_prepared_inputs(tokens, mask)
        local = _masked_zero(self.local_phi(tokens), mask)
        if bool(self.config.pfn_use_global_context):
            global_values = _masked_zero(self.global_phi(tokens), mask)
            counts = mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype).clamp_min(1.0)
            global_summary = global_values.sum(dim=1) / counts
            global_broadcast = global_summary[:, None, :].expand_as(local)
            features = torch_cat([local, global_broadcast], dim=-1)
        else:
            features = local
        embeddings = _masked_zero(self.out(features), mask)
        return ArchitectureViewBranchOutput(
            name=ARCHITECTURE_VIEW_BRANCH_PFN,
            embeddings=embeddings,
            mask=mask,
            diagnostics={"uses_global_context": bool(self.config.pfn_use_global_context)},
        )


def torch_cat(values: list[Any], dim: int) -> Any:
    torch = require_torch()
    return torch.cat(values, dim=dim)


class PCNNResidualBlock(_ModuleBase):
    """Small residual Conv1d block over canonical pT-ordered particles."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        torch = require_torch()
        padding = int(kernel_size) // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.norm = torch.nn.LayerNorm(channels)

    def forward(self, state: Any, mask: Any) -> Any:
        update = self.net(state.transpose(1, 2)).transpose(1, 2)
        state = self.norm(state + update)
        return _masked_zero(state, mask)


class PCNNViewBranch(_ModuleBase):
    """PCNN-style local sequence branch over the existing HLT particle order."""

    def __init__(self, config: ArchitectureViewConfig | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = config or ArchitectureViewConfig()
        channels = int(self.config.pcnn_channels)
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.raw_token_dim), channels),
            torch.nn.GELU(),
            torch.nn.LayerNorm(channels),
        )
        kernels = tuple(int(k) for k in self.config.pcnn_kernel_sizes)
        self.blocks = torch.nn.ModuleList(
            [
                PCNNResidualBlock(
                    channels=channels,
                    kernel_size=kernels[index % len(kernels)],
                    dropout=float(self.config.dropout),
                )
                for index in range(int(self.config.pcnn_layers))
            ]
        )
        self.proj = torch.nn.Linear(channels, int(self.config.view_dim))

    def forward(self, tokens: Any, mask: Any | None = None) -> ArchitectureViewBranchOutput:
        tokens, mask = _coerce_prepared_inputs(tokens, mask)
        state = _masked_zero(self.stem(tokens), mask)
        for block in self.blocks:
            state = block(state, mask)
        embeddings = _masked_zero(self.proj(state), mask)
        return ArchitectureViewBranchOutput(
            name=ARCHITECTURE_VIEW_BRANCH_PCNN,
            embeddings=embeddings,
            mask=mask,
            diagnostics={"kernel_sizes": tuple(int(k) for k in self.config.pcnn_kernel_sizes)},
        )


class ArchitectureViewBranchBank(_ModuleBase):
    """Construct and run any subset of architecture-view branches."""

    def __init__(
        self,
        config: ArchitectureViewConfig | None = None,
        *,
        enabled_views: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        torch = require_torch()
        self.config = config or ArchitectureViewConfig()
        views = self.config.enabled_views if enabled_views is None else enabled_views
        self.enabled_views = tuple(normalize_architecture_view_branch(v) for v in views)
        modules: dict[str, Any] = {}
        for name in self.enabled_views:
            if name == ARCHITECTURE_VIEW_BRANCH_PN:
                modules[name] = PNViewBranch(self.config)
            elif name == ARCHITECTURE_VIEW_BRANCH_PFN:
                modules[name] = PFNViewBranch(self.config)
            elif name == ARCHITECTURE_VIEW_BRANCH_PCNN:
                modules[name] = PCNNViewBranch(self.config)
            else:  # pragma: no cover - guarded by normalization
                raise ValueError(f"Unsupported branch {name!r}")
        self.branches = torch.nn.ModuleDict(modules)

    def forward(self, tokens: Any, mask: Any | None = None) -> dict[str, ArchitectureViewBranchOutput]:
        prepared = sanitize_architecture_view_tokens(tokens, mask)
        return {name: module(prepared.tokens, prepared.mask) for name, module in self.branches.items()}

    def empty_view_dict(self) -> dict[str, ArchitectureViewBranchOutput]:
        return {}


def build_architecture_view_branch(name: str, config: ArchitectureViewConfig | None = None) -> Any:
    normalized = normalize_architecture_view_branch(name)
    if normalized == ARCHITECTURE_VIEW_BRANCH_PN:
        return PNViewBranch(config)
    if normalized == ARCHITECTURE_VIEW_BRANCH_PFN:
        return PFNViewBranch(config)
    if normalized == ARCHITECTURE_VIEW_BRANCH_PCNN:
        return PCNNViewBranch(config)
    raise ValueError(f"Unsupported branch {name!r}")  # pragma: no cover


def branch_output_shapes(outputs: Mapping[str, ArchitectureViewBranchOutput]) -> dict[str, list[int]]:
    return {name: list(output.embeddings.shape) for name, output in outputs.items()}
