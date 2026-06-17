"""Pairwise geometry and attention-bias utilities for five-view taggers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from typing import Any

from jetclass_fresh.hlt_baseline import require_torch


PAIRWISE_GEOMETRY_FEATURE_NAMES = (
    "delta_eta",
    "sin_delta_phi",
    "cos_delta_phi",
    "delta_r",
    "delta_log_pt",
    "abs_delta_eta",
    "abs_delta_phi",
    "confidence_i",
    "confidence_j",
    "confidence_product",
    "same_view",
)
PAIRWISE_GEOMETRY_FEATURE_DIM = len(PAIRWISE_GEOMETRY_FEATURE_NAMES)

TOKEN_KIND_GLOBAL = 0
TOKEN_KIND_VIEW_SUMMARY = 1
TOKEN_KIND_PARTICLE = 2

RELATION_PARTICLE_SAME_VIEW = 0
RELATION_PARTICLE_DIFFERENT_VIEW = 1
RELATION_VIEW_SUMMARY = 2
RELATION_SUMMARY_PARTICLE = 3
RELATION_GLOBAL = 4
NUM_FIVE_VIEW_RELATION_TYPES = 5


if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    import torch as _torch

    _ModuleBase = _torch.nn.Module


def wrapped_delta_phi(delta_phi):
    """Wrap a delta-phi tensor to ``[-pi, pi]``."""

    torch = require_torch()
    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def _as_tensor(value: Any, *, dtype=None, device=None):
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor
    return torch.as_tensor(value, dtype=dtype, device=device)


def build_pairwise_view_pair_ids(token_view_ids, *, max_view_embeddings: int):
    """Return learned view-pair ids for all token-token pairs.

    Special tokens should use any nonnegative view id; the relation-type mask
    decides whether the view-pair embedding is actually applied.
    """

    token_view_ids = _as_tensor(token_view_ids, dtype=require_torch().long)
    max_view_embeddings = int(max_view_embeddings)
    if token_view_ids.ndim != 2:
        raise ValueError(f"token_view_ids must be [batch, tokens], got {tuple(token_view_ids.shape)}")
    if max_view_embeddings <= 0:
        raise ValueError("max_view_embeddings must be positive")
    safe_ids = token_view_ids.clamp(min=0, max=max_view_embeddings - 1)
    return safe_ids[:, :, None] * max_view_embeddings + safe_ids[:, None, :]


def build_pairwise_relation_type_ids(token_kinds, token_view_ids=None):
    """Build relation ids for global, summary, same-view, and cross-view pairs."""

    torch = require_torch()
    token_kinds = _as_tensor(token_kinds, dtype=torch.long)
    if token_kinds.ndim != 2:
        raise ValueError(f"token_kinds must be [batch, tokens], got {tuple(token_kinds.shape)}")
    left_kind = token_kinds[:, :, None]
    right_kind = token_kinds[:, None, :]
    relation_ids = torch.full(
        (int(token_kinds.shape[0]), int(token_kinds.shape[1]), int(token_kinds.shape[1])),
        RELATION_GLOBAL,
        dtype=torch.long,
        device=token_kinds.device,
    )

    left_particle = left_kind == TOKEN_KIND_PARTICLE
    right_particle = right_kind == TOKEN_KIND_PARTICLE
    both_particle = left_particle & right_particle
    if token_view_ids is None:
        same_view = torch.ones_like(both_particle, dtype=torch.bool)
    else:
        token_view_ids = _as_tensor(token_view_ids, dtype=torch.long, device=token_kinds.device)
        if tuple(token_view_ids.shape) != tuple(token_kinds.shape):
            raise ValueError("token_view_ids shape must match token_kinds")
        same_view = token_view_ids[:, :, None].eq(token_view_ids[:, None, :])
    relation_ids = torch.where(
        both_particle & same_view,
        torch.full_like(relation_ids, RELATION_PARTICLE_SAME_VIEW),
        relation_ids,
    )
    relation_ids = torch.where(
        both_particle & ~same_view,
        torch.full_like(relation_ids, RELATION_PARTICLE_DIFFERENT_VIEW),
        relation_ids,
    )

    left_summary = left_kind == TOKEN_KIND_VIEW_SUMMARY
    right_summary = right_kind == TOKEN_KIND_VIEW_SUMMARY
    both_summary = left_summary & right_summary
    summary_particle = (left_summary & right_particle) | (left_particle & right_summary)
    relation_ids = torch.where(
        both_summary,
        torch.full_like(relation_ids, RELATION_VIEW_SUMMARY),
        relation_ids,
    )
    relation_ids = torch.where(
        summary_particle,
        torch.full_like(relation_ids, RELATION_SUMMARY_PARTICLE),
        relation_ids,
    )
    return relation_ids


def build_pairwise_geometry_features(
    tokens,
    *,
    confidence=None,
    token_is_particle=None,
    token_view_ids=None,
    pt_index: int = 0,
    eta_index: int = 1,
    phi_index: int = 2,
    eps: float = 1.0e-6,
):
    """Build wraparound-aware pairwise geometry features.

    Non-particle pairs are zeroed so special-token behavior is handled by
    relation embeddings, not fake coordinates.
    """

    torch = require_torch()
    tokens = _as_tensor(tokens, dtype=torch.float32)
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [batch, tokens, features], got {tuple(tokens.shape)}")
    batch_size, n_tokens, feature_dim = (int(value) for value in tokens.shape)
    for name, index in (("pt_index", pt_index), ("eta_index", eta_index), ("phi_index", phi_index)):
        if int(index) < 0 or int(index) >= feature_dim:
            raise ValueError(f"{name}={index} is outside feature dimension {feature_dim}")
    if confidence is None:
        confidence = torch.ones((batch_size, n_tokens), dtype=tokens.dtype, device=tokens.device)
    else:
        confidence = _as_tensor(confidence, dtype=tokens.dtype, device=tokens.device)
        if tuple(confidence.shape) != (batch_size, n_tokens):
            raise ValueError("confidence shape must be [batch, tokens]")
    if token_is_particle is None:
        token_is_particle = torch.ones((batch_size, n_tokens), dtype=torch.bool, device=tokens.device)
    else:
        token_is_particle = _as_tensor(token_is_particle, dtype=torch.bool, device=tokens.device)
        if tuple(token_is_particle.shape) != (batch_size, n_tokens):
            raise ValueError("token_is_particle shape must be [batch, tokens]")
    if token_view_ids is not None:
        token_view_ids = _as_tensor(token_view_ids, dtype=torch.long, device=tokens.device)
        if tuple(token_view_ids.shape) != (batch_size, n_tokens):
            raise ValueError("token_view_ids shape must be [batch, tokens]")

    pt = torch.clamp(tokens[:, :, int(pt_index)].abs(), min=float(eps))
    eta = tokens[:, :, int(eta_index)]
    phi = tokens[:, :, int(phi_index)]
    delta_eta = eta[:, :, None] - eta[:, None, :]
    delta_phi = wrapped_delta_phi(phi[:, :, None] - phi[:, None, :])
    delta_r = torch.sqrt(torch.clamp(delta_eta * delta_eta + delta_phi * delta_phi, min=0.0))
    delta_log_pt = torch.log(pt[:, :, None]) - torch.log(pt[:, None, :])
    conf_i = confidence[:, :, None].expand(batch_size, n_tokens, n_tokens)
    conf_j = confidence[:, None, :].expand(batch_size, n_tokens, n_tokens)
    if token_view_ids is None:
        same_view = torch.ones_like(delta_eta)
    else:
        same_view = token_view_ids[:, :, None].eq(token_view_ids[:, None, :]).to(tokens.dtype)
    features = torch.stack(
        [
            torch.clamp(delta_eta / 5.0, -2.0, 2.0),
            torch.sin(delta_phi),
            torch.cos(delta_phi),
            torch.clamp(delta_r / 5.0, 0.0, 4.0),
            torch.clamp(delta_log_pt, -8.0, 8.0),
            torch.clamp(delta_eta.abs() / 5.0, 0.0, 4.0),
            torch.clamp(delta_phi.abs() / math.pi, 0.0, 1.0),
            conf_i,
            conf_j,
            conf_i * conf_j,
            same_view,
        ],
        dim=-1,
    )
    particle_pair = token_is_particle[:, :, None] & token_is_particle[:, None, :]
    return torch.where(particle_pair[:, :, :, None], features, torch.zeros_like(features))


@dataclass(frozen=True)
class PairwiseAttentionBiasConfig:
    """Small MLP configuration for additive attention bias."""

    num_heads: int
    hidden_dim: int = 64
    pair_feature_dim: int = PAIRWISE_GEOMETRY_FEATURE_DIM
    num_relation_types: int = NUM_FIVE_VIEW_RELATION_TYPES
    max_view_embeddings: int = 16
    dropout: float = 0.0
    use_view_pair_embedding: bool = True

    def __post_init__(self) -> None:
        if int(self.num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(self.pair_feature_dim) <= 0:
            raise ValueError("pair_feature_dim must be positive")
        if int(self.num_relation_types) <= 0:
            raise ValueError("num_relation_types must be positive")
        if int(self.max_view_embeddings) <= 0:
            raise ValueError("max_view_embeddings must be positive")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")


class PairwiseAttentionBias(_ModuleBase):
    """Convert geometry features and relation ids into per-head attention bias."""

    def __init__(self, config: PairwiseAttentionBiasConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear(int(config.pair_feature_dim), int(config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(int(config.hidden_dim), int(config.num_heads)),
        )
        self.relation_embedding = torch.nn.Embedding(int(config.num_relation_types), int(config.num_heads))
        if bool(config.use_view_pair_embedding):
            self.view_pair_embedding = torch.nn.Embedding(
                int(config.max_view_embeddings) * int(config.max_view_embeddings),
                int(config.num_heads),
            )
        else:
            self.view_pair_embedding = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch = require_torch()
        final = self.feature_mlp[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
        torch.nn.init.zeros_(self.relation_embedding.weight)
        if self.view_pair_embedding is not None:
            torch.nn.init.zeros_(self.view_pair_embedding.weight)

    def forward(self, pair_features, relation_type_ids, view_pair_ids=None):
        torch = require_torch()
        pair_features = _as_tensor(pair_features, dtype=torch.float32)
        relation_type_ids = _as_tensor(relation_type_ids, dtype=torch.long, device=pair_features.device)
        if pair_features.ndim != 4:
            raise ValueError("pair_features must be [batch, tokens, tokens, features]")
        if tuple(relation_type_ids.shape) != tuple(pair_features.shape[:3]):
            raise ValueError("relation_type_ids shape must match pair_features first three dimensions")
        if int(relation_type_ids.min().item()) < 0 or int(relation_type_ids.max().item()) >= int(self.config.num_relation_types):
            raise ValueError("relation_type_ids contain out-of-range values")
        bias = self.feature_mlp(torch.nan_to_num(pair_features, nan=0.0, posinf=0.0, neginf=0.0))
        bias = bias + self.relation_embedding(relation_type_ids)
        if self.view_pair_embedding is not None and view_pair_ids is not None:
            view_pair_ids = _as_tensor(view_pair_ids, dtype=torch.long, device=pair_features.device)
            if tuple(view_pair_ids.shape) != tuple(pair_features.shape[:3]):
                raise ValueError("view_pair_ids shape must match pair_features first three dimensions")
            pair_relation = (relation_type_ids == RELATION_PARTICLE_SAME_VIEW) | (
                relation_type_ids == RELATION_PARTICLE_DIFFERENT_VIEW
            )
            view_pair_bias = self.view_pair_embedding(view_pair_ids.clamp(min=0, max=self.view_pair_embedding.num_embeddings - 1))
            bias = bias + torch.where(pair_relation[:, :, :, None], view_pair_bias, torch.zeros_like(view_pair_bias))
        return bias.permute(0, 3, 1, 2).contiguous()


__all__ = [
    "NUM_FIVE_VIEW_RELATION_TYPES",
    "PAIRWISE_GEOMETRY_FEATURE_DIM",
    "PAIRWISE_GEOMETRY_FEATURE_NAMES",
    "RELATION_GLOBAL",
    "RELATION_PARTICLE_DIFFERENT_VIEW",
    "RELATION_PARTICLE_SAME_VIEW",
    "RELATION_SUMMARY_PARTICLE",
    "RELATION_VIEW_SUMMARY",
    "TOKEN_KIND_GLOBAL",
    "TOKEN_KIND_PARTICLE",
    "TOKEN_KIND_VIEW_SUMMARY",
    "PairwiseAttentionBias",
    "PairwiseAttentionBiasConfig",
    "build_pairwise_geometry_features",
    "build_pairwise_relation_type_ids",
    "build_pairwise_view_pair_ids",
    "wrapped_delta_phi",
]
