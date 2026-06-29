"""Subjet-subjet transformer with physics-aware pair bias.

Step 6 takes the learned multi-scale subjet tokens from Step 5 and lets them
reason with each other.  The pair bias is deliberately closer to a small
ParT-style subjet graph than a plain latent-token transformer: pair mass,
relative kT, energy sharing, assignment overlap, containment, and scale-pair
type all have a direct path into attention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .features import wrap_delta_phi
from .tokens import MultiScaleSubjetTokenBuilderOutput


try:  # Keep imports cheap on systems without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


MULTISCALE_SUBJET_TRANSFORMER_CONTRACT = "multiscale_subjet_transformer_v1"
MULTISCALE_SUBJET_TRANSFORMER_STEP = "multiscale_subjet_part_step6_subjet_transformer"

MULTISCALE_SUBJET_PAIR_FEATURE_NAMES = (
    "delta_eta",
    "sin_delta_phi",
    "cos_delta_phi",
    "delta_r",
    "log_delta_r",
    "log_pair_mass",
    "log_relative_kt",
    "z",
    "log_pt_ratio",
    "pt_balance",
    "assignment_overlap",
    "cluster_overlap",
    "containment_i_in_j",
    "containment_j_in_i",
    "same_scale",
    "same_subjet",
    "scale_radius_i",
    "scale_radius_j",
    "log_radius_ratio",
)
MULTISCALE_SUBJET_PAIR_FEATURE_DIM = len(MULTISCALE_SUBJET_PAIR_FEATURE_NAMES)


@dataclass(frozen=True)
class MultiScaleSubjetPairFeatureConfig:
    """Numerical conventions for Step 6 subjet-pair feature construction."""

    num_scales: int = 3
    eps: float = 1.0e-6
    delta_r_scale: float = 5.0
    radius_scale: float = 0.5
    max_log_value: float = 14.0
    max_log_ratio: float = 8.0

    def __post_init__(self) -> None:
        num_scales = int(self.num_scales)
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")
        object.__setattr__(self, "num_scales", num_scales)
        for name in ("eps", "delta_r_scale", "radius_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in ("max_log_value", "max_log_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class MultiScaleSubjetPairFeatureOutput:
    """Physics-aware pair features between soft subjet tokens."""

    pair_features: Any
    pair_mask: Any
    token_mask: Any
    scale_pair_type: Any
    feature_names: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_TRANSFORMER_CONTRACT,
            "step": MULTISCALE_SUBJET_TRANSFORMER_STEP,
            "pair_features_shape": list(self.pair_features.shape),
            "pair_mask_shape": list(self.pair_mask.shape),
            "token_mask_shape": list(self.token_mask.shape),
            "scale_pair_type_shape": list(self.scale_pair_type.shape),
            "feature_names": list(self.feature_names),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class MultiScaleSubjetTransformerConfig:
    """Configuration for the Step 6 subjet-subjet transformer."""

    token_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.05
    attention_dropout: float = 0.05
    use_pairwise_bias: bool = True
    use_scale_pair_embedding: bool = True
    pair_bias_hidden_dim: int = 64
    num_scales: int = 3
    mask_value: float = -1.0e4
    pair_feature_config: MultiScaleSubjetPairFeatureConfig | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("token_dim", "num_heads", "ffn_dim", "pair_bias_hidden_dim", "num_scales"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        num_layers = int(self.num_layers)
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        object.__setattr__(self, "num_layers", num_layers)
        if int(self.token_dim) % int(self.num_heads) != 0:
            raise ValueError("num_heads must divide token_dim")
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be finite and satisfy 0 <= {name} < 1")
            object.__setattr__(self, name, value)
        mask_value = float(self.mask_value)
        if not math.isfinite(mask_value) or mask_value >= 0.0:
            raise ValueError("mask_value must be finite and negative")
        pair_feature_config = normalize_subjet_pair_feature_config(self.pair_feature_config)
        if int(pair_feature_config.num_scales) != int(self.num_scales):
            pair_feature_config = MultiScaleSubjetPairFeatureConfig(
                num_scales=int(self.num_scales),
                eps=pair_feature_config.eps,
                delta_r_scale=pair_feature_config.delta_r_scale,
                radius_scale=pair_feature_config.radius_scale,
                max_log_value=pair_feature_config.max_log_value,
                max_log_ratio=pair_feature_config.max_log_ratio,
            )
        object.__setattr__(self, "mask_value", mask_value)
        object.__setattr__(self, "pair_feature_config", pair_feature_config)
        object.__setattr__(self, "use_pairwise_bias", bool(self.use_pairwise_bias))
        object.__setattr__(self, "use_scale_pair_embedding", bool(self.use_scale_pair_embedding))


@dataclass(frozen=True)
class MultiScaleSubjetTransformerOutput:
    """Output of the Step 6 subjet-subjet transformer."""

    subjet_tokens: Any
    subjet_mask: Any
    pair_features: Any
    pair_mask: Any
    pair_bias: Any | None
    attention_weights: Any | None
    pair_feature_output: MultiScaleSubjetPairFeatureOutput
    diagnostics: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_TRANSFORMER_CONTRACT,
            "step": MULTISCALE_SUBJET_TRANSFORMER_STEP,
            "subjet_tokens_shape": list(self.subjet_tokens.shape),
            "subjet_mask_shape": list(self.subjet_mask.shape),
            "pair_features_shape": list(self.pair_features.shape),
            "pair_mask_shape": list(self.pair_mask.shape),
            "pair_bias_shape": None if self.pair_bias is None else list(self.pair_bias.shape),
            "attention_weights_shape": None if self.attention_weights is None else list(self.attention_weights.shape),
            "diagnostics": dict(self.diagnostics),
        }


def normalize_subjet_pair_feature_config(
    config: MultiScaleSubjetPairFeatureConfig | Mapping[str, Any] | None = None,
) -> MultiScaleSubjetPairFeatureConfig:
    if config is None:
        return MultiScaleSubjetPairFeatureConfig()
    if isinstance(config, MultiScaleSubjetPairFeatureConfig):
        return config
    return MultiScaleSubjetPairFeatureConfig(**dict(config))


def normalize_subjet_transformer_config(
    config: MultiScaleSubjetTransformerConfig | Mapping[str, Any] | None = None,
) -> MultiScaleSubjetTransformerConfig:
    if config is None:
        return MultiScaleSubjetTransformerConfig()
    if isinstance(config, MultiScaleSubjetTransformerConfig):
        return config
    return MultiScaleSubjetTransformerConfig(**dict(config))


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _masked_mean(value: Any, mask: Any) -> float:
    torch = require_torch()
    weights = mask.to(dtype=value.dtype)
    denom = torch.clamp(weights.sum(), min=1.0)
    return float((value * weights).sum().detach().cpu().item() / float(denom.detach().cpu().item()))


def _validate_token_output(token_output: MultiScaleSubjetTokenBuilderOutput) -> None:
    tokens = token_output.subjet_tokens
    mask = token_output.subjet_mask
    if int(tokens.ndim) != 3:
        raise ValueError(f"subjet_tokens must have shape [batch, subjets, dim], got {tuple(tokens.shape)}")
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"subjet_mask shape {tuple(mask.shape)} does not match {tuple(tokens.shape[:2])}")
    batch_size, num_subjets = (int(value) for value in mask.shape)
    for name in (
        "assignment_weights",
        "cluster_weights",
    ):
        value = getattr(token_output, name)
        if int(value.ndim) != 3 or tuple(value.shape[:2]) != (batch_size, num_subjets):
            raise ValueError(f"{name} must have shape [batch, subjets, particles], got {tuple(value.shape)}")
    if tuple(token_output.estimated_centers.shape) != (batch_size, num_subjets, 2):
        raise ValueError("estimated_centers must have shape [batch, subjets, 2]")
    if tuple(token_output.soft_four_vectors.shape) != (batch_size, num_subjets, 4):
        raise ValueError("soft_four_vectors must have shape [batch, subjets, 4]")
    if tuple(token_output.cluster_pt_fraction.shape) != (batch_size, num_subjets):
        raise ValueError("cluster_pt_fraction must have shape [batch, subjets]")
    if int(token_output.scale_index.ndim) != 1 or int(token_output.scale_index.shape[0]) != num_subjets:
        raise ValueError("scale_index must have shape [subjets]")
    if int(token_output.scale_radius.ndim) != 1 or int(token_output.scale_radius.shape[0]) != num_subjets:
        raise ValueError("scale_radius must have shape [subjets]")


def build_multiscale_subjet_pair_features(
    token_output: MultiScaleSubjetTokenBuilderOutput,
    config: MultiScaleSubjetPairFeatureConfig | Mapping[str, Any] | None = None,
) -> MultiScaleSubjetPairFeatureOutput:
    """Build Step 6 pair features from Step 5 multi-scale subjet summaries."""

    torch = require_torch()
    pair_config = normalize_subjet_pair_feature_config(config)
    _validate_token_output(token_output)

    tokens = token_output.subjet_tokens
    device = tokens.device
    dtype = tokens.dtype
    mask = token_output.subjet_mask.to(device=device, dtype=torch.bool)
    centers = token_output.estimated_centers.to(device=device, dtype=dtype)
    four_vectors = token_output.soft_four_vectors.to(device=device, dtype=dtype)
    assignment_weights = token_output.assignment_weights.to(device=device, dtype=dtype)
    cluster_weights = token_output.cluster_weights.to(device=device, dtype=dtype)
    scale_index = token_output.scale_index.to(device=device, dtype=torch.long)
    scale_radius = token_output.scale_radius.to(device=device, dtype=dtype)
    if scale_index.numel() and int(scale_index.max().detach().cpu().item()) >= int(pair_config.num_scales):
        raise ValueError(
            f"scale_index contains {int(scale_index.max().detach().cpu().item())}, "
            f"but pair feature config only allows num_scales={int(pair_config.num_scales)}"
        )

    eps = float(pair_config.eps)
    eps2 = eps * eps
    delta_eta = centers[:, :, None, 0] - centers[:, None, :, 0]
    delta_phi = wrap_delta_phi(centers[:, :, None, 1] - centers[:, None, :, 1])
    delta_r = torch.sqrt(torch.clamp(delta_eta * delta_eta + delta_phi * delta_phi, min=eps2))

    px = four_vectors[:, :, 0]
    py = four_vectors[:, :, 1]
    pz = four_vectors[:, :, 2]
    energy = torch.clamp(four_vectors[:, :, 3], min=eps)
    pt = torch.sqrt(torch.clamp(px * px + py * py, min=eps2))

    pair_px = px[:, :, None] + px[:, None, :]
    pair_py = py[:, :, None] + py[:, None, :]
    pair_pz = pz[:, :, None] + pz[:, None, :]
    pair_energy = energy[:, :, None] + energy[:, None, :]
    pair_mass2 = torch.clamp(
        pair_energy * pair_energy - pair_px * pair_px - pair_py * pair_py - pair_pz * pair_pz,
        min=eps * eps,
    )
    pair_mass = torch.sqrt(pair_mass2)
    pt_i = torch.clamp(pt[:, :, None], min=eps)
    pt_j = torch.clamp(pt[:, None, :], min=eps)
    min_pt = torch.minimum(pt_i, pt_j)
    relative_kt = min_pt * delta_r
    z = min_pt / torch.clamp(pt_i + pt_j, min=eps)
    log_pt_ratio = torch.log(pt_i) - torch.log(pt_j)
    pt_balance = (pt_i - pt_j) / torch.clamp(pt_i + pt_j, min=eps)

    assignment_overlap = torch.minimum(assignment_weights[:, :, None, :], assignment_weights[:, None, :, :]).sum(dim=-1)
    cluster_overlap = torch.minimum(cluster_weights[:, :, None, :], cluster_weights[:, None, :, :]).sum(dim=-1)
    cluster_sum_i = torch.clamp(cluster_weights.sum(dim=-1)[:, :, None], min=eps)
    cluster_sum_j = torch.clamp(cluster_weights.sum(dim=-1)[:, None, :], min=eps)
    containment_i_in_j = cluster_overlap / cluster_sum_i
    containment_j_in_i = cluster_overlap / cluster_sum_j

    same_scale = (scale_index[:, None] == scale_index[None, :]).to(dtype=dtype)
    same_subjet = torch.eye(int(mask.shape[1]), dtype=dtype, device=device)
    radius_i = scale_radius[:, None]
    radius_j = scale_radius[None, :]
    radius_i_expanded = radius_i.expand(int(mask.shape[1]), int(mask.shape[1]))
    radius_j_expanded = radius_j.expand(int(mask.shape[1]), int(mask.shape[1]))
    log_radius_ratio = torch.log(torch.clamp(radius_i_expanded, min=eps)) - torch.log(torch.clamp(radius_j_expanded, min=eps))

    features = torch.stack(
        [
            torch.clamp(delta_eta / float(pair_config.delta_r_scale), -2.0, 2.0),
            torch.sin(delta_phi),
            torch.cos(delta_phi),
            torch.clamp(delta_r / float(pair_config.delta_r_scale), 0.0, 4.0),
            torch.clamp(torch.log(delta_r + eps), -float(pair_config.max_log_value), float(pair_config.max_log_value)),
            torch.clamp(torch.log(pair_mass + eps), -float(pair_config.max_log_value), float(pair_config.max_log_value)),
            torch.clamp(torch.log(relative_kt + eps), -float(pair_config.max_log_value), float(pair_config.max_log_value)),
            torch.clamp(z, 0.0, 0.5),
            torch.clamp(log_pt_ratio, -float(pair_config.max_log_ratio), float(pair_config.max_log_ratio)),
            torch.clamp(pt_balance, -1.0, 1.0),
            torch.clamp(assignment_overlap, 0.0, 1.0),
            torch.clamp(cluster_overlap, 0.0, float(cluster_weights.shape[-1])),
            torch.clamp(containment_i_in_j, 0.0, 1.0),
            torch.clamp(containment_j_in_i, 0.0, 1.0),
            same_scale[None, :, :].expand(int(mask.shape[0]), -1, -1),
            same_subjet[None, :, :].expand(int(mask.shape[0]), -1, -1),
            torch.clamp(radius_i_expanded[None, :, :] / float(pair_config.radius_scale), 0.0, 4.0).expand(int(mask.shape[0]), -1, -1),
            torch.clamp(radius_j_expanded[None, :, :] / float(pair_config.radius_scale), 0.0, 4.0).expand(int(mask.shape[0]), -1, -1),
            torch.clamp(log_radius_ratio[None, :, :], -float(pair_config.max_log_ratio), float(pair_config.max_log_ratio)).expand(
                int(mask.shape[0]), -1, -1
            ),
        ],
        dim=-1,
    )
    pair_mask = mask[:, :, None] & mask[:, None, :]
    features = torch.where(pair_mask[:, :, :, None], _nan_to_num_torch(features), torch.zeros_like(features))
    scale_pair_type = scale_index[:, None] * int(pair_config.num_scales) + scale_index[None, :]
    diagnostics = {
        "contract": MULTISCALE_SUBJET_TRANSFORMER_CONTRACT,
        "step": MULTISCALE_SUBJET_TRANSFORMER_STEP,
        "valid_pair_fraction": float(pair_mask.float().mean().detach().cpu().item()),
        "delta_r_mean": _masked_mean(delta_r, pair_mask),
        "log_pair_mass_mean": _masked_mean(torch.log(pair_mass + eps), pair_mask),
        "assignment_overlap_mean": _masked_mean(assignment_overlap, pair_mask),
        "cluster_overlap_mean": _masked_mean(cluster_overlap, pair_mask),
        "same_scale_pair_fraction": _masked_mean(same_scale[None, :, :].expand(int(mask.shape[0]), -1, -1), pair_mask),
    }
    return MultiScaleSubjetPairFeatureOutput(
        pair_features=features,
        pair_mask=pair_mask,
        token_mask=mask,
        scale_pair_type=scale_pair_type,
        feature_names=MULTISCALE_SUBJET_PAIR_FEATURE_NAMES,
        diagnostics=diagnostics,
    )


class MultiScaleSubjetPairBiasEncoder(_ModuleBase):
    """Map subjet-pair physics features into additive per-head attention bias."""

    def __init__(self, config: MultiScaleSubjetTransformerConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = normalize_subjet_transformer_config(config)
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(MULTISCALE_SUBJET_PAIR_FEATURE_DIM),
            torch.nn.Linear(MULTISCALE_SUBJET_PAIR_FEATURE_DIM, int(self.config.pair_bias_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.pair_bias_hidden_dim), int(self.config.num_heads)),
        )
        self.scale_pair_embedding = torch.nn.Embedding(
            int(self.config.num_scales) * int(self.config.num_scales),
            int(self.config.num_heads),
        )
        torch.nn.init.normal_(self.scale_pair_embedding.weight, mean=0.0, std=0.02)

    def forward(self, pair_output: MultiScaleSubjetPairFeatureOutput) -> Any:
        torch = require_torch()
        pair_features = _nan_to_num_torch(pair_output.pair_features.float())
        if int(pair_features.shape[-1]) != MULTISCALE_SUBJET_PAIR_FEATURE_DIM:
            raise ValueError(
                f"pair feature dim must be {MULTISCALE_SUBJET_PAIR_FEATURE_DIM}, got {int(pair_features.shape[-1])}"
            )
        pair_mask = pair_output.pair_mask.to(device=pair_features.device, dtype=torch.bool)
        bias = self.network(pair_features)
        if bool(self.config.use_scale_pair_embedding):
            scale_pair_type = pair_output.scale_pair_type.to(device=pair_features.device, dtype=torch.long)
            max_type = int(scale_pair_type.max().detach().cpu().item()) if scale_pair_type.numel() else -1
            if max_type >= int(self.config.num_scales) * int(self.config.num_scales):
                raise ValueError(
                    f"scale_pair_type contains {max_type}, but num_scales={int(self.config.num_scales)} "
                    "does not cover that scale pair"
                )
            bias = bias + self.scale_pair_embedding(scale_pair_type)[None, :, :, :]
        bias = bias.permute(0, 3, 1, 2).contiguous()
        return torch.where(pair_mask[:, None, :, :], _nan_to_num_torch(bias), torch.zeros_like(bias))


class MultiScaleSubjetTransformerBlock(_ModuleBase):
    """One pre-norm self-attention block over subjet tokens."""

    def __init__(self, config: MultiScaleSubjetTransformerConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = normalize_subjet_transformer_config(config)
        self.norm1 = torch.nn.LayerNorm(int(self.config.token_dim))
        self.attention = torch.nn.MultiheadAttention(
            int(self.config.token_dim),
            int(self.config.num_heads),
            dropout=float(self.config.attention_dropout),
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(float(self.config.dropout))
        self.norm2 = torch.nn.LayerNorm(int(self.config.token_dim))
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.token_dim), int(self.config.ffn_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.ffn_dim), int(self.config.token_dim)),
        )

    def _attention_mask(self, pair_bias: Any | None, token_mask: Any, tokens: Any) -> Any | None:
        torch = require_torch()
        if pair_bias is None:
            return None
        pair_bias = pair_bias.to(device=tokens.device, dtype=tokens.dtype)
        token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        batch_size, num_heads, num_tokens, other_tokens = (int(value) for value in pair_bias.shape)
        if num_heads != int(self.config.num_heads):
            raise ValueError(f"pair_bias has {num_heads} heads but config has {int(self.config.num_heads)}")
        if num_tokens != other_tokens:
            raise ValueError("pair_bias must be square in token dimensions")
        if tuple(token_mask.shape) != (batch_size, num_tokens):
            raise ValueError(f"token_mask shape {tuple(token_mask.shape)} does not match {(batch_size, num_tokens)}")
        pair_bias = pair_bias.masked_fill(~token_mask[:, None, None, :], float(self.config.mask_value))
        return pair_bias.reshape(batch_size * num_heads, num_tokens, num_tokens)

    def forward(
        self,
        tokens: Any,
        token_mask: Any,
        *,
        pair_bias: Any | None = None,
        need_weights: bool = False,
    ) -> tuple[Any, Any | None]:
        torch = require_torch()
        token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        x = torch.where(token_mask[:, :, None], _nan_to_num_torch(tokens.float()), torch.zeros_like(tokens.float()))
        normed = self.norm1(x)
        attn_mask = self._attention_mask(pair_bias, token_mask, normed)
        attended, weights = self.attention(
            normed,
            normed,
            normed,
            key_padding_mask=None if attn_mask is not None else ~token_mask,
            attn_mask=attn_mask,
            need_weights=bool(need_weights),
            average_attn_weights=False,
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        x = torch.where(token_mask[:, :, None], _nan_to_num_torch(x), torch.zeros_like(x))
        if weights is not None:
            weights = torch.where(token_mask[:, None, :, None], _nan_to_num_torch(weights), torch.zeros_like(weights))
            weights = torch.where(token_mask[:, None, None, :], weights, torch.zeros_like(weights))
        return x, weights


class MultiScaleSubjetTransformer(_ModuleBase):
    """Small subjet-subjet transformer used before particle readback."""

    def __init__(self, config: MultiScaleSubjetTransformerConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = normalize_subjet_transformer_config(config)
        self.pair_bias_encoder = MultiScaleSubjetPairBiasEncoder(self.config)
        self.layers = torch.nn.ModuleList(
            [MultiScaleSubjetTransformerBlock(self.config) for _ in range(int(self.config.num_layers))]
        )
        self.output_norm = torch.nn.LayerNorm(int(self.config.token_dim))

    def forward(
        self,
        token_output: MultiScaleSubjetTokenBuilderOutput,
        *,
        need_weights: bool = True,
    ) -> MultiScaleSubjetTransformerOutput:
        torch = require_torch()
        _validate_token_output(token_output)
        tokens = _nan_to_num_torch(token_output.subjet_tokens.float())
        mask = token_output.subjet_mask.to(device=tokens.device, dtype=torch.bool)
        if int(tokens.shape[-1]) != int(self.config.token_dim):
            raise ValueError(
                f"token_output has token_dim={int(tokens.shape[-1])}, but transformer expects {int(self.config.token_dim)}"
            )
        safe_mask = mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if bool(empty_rows.any()):
            safe_mask = safe_mask.clone()
            safe_mask[empty_rows, 0] = True
            tokens = tokens.clone()
            tokens[empty_rows, 0, :] = 0.0

        pair_output = build_multiscale_subjet_pair_features(token_output, self.config.pair_feature_config)
        pair_bias = self.pair_bias_encoder(pair_output) if bool(self.config.use_pairwise_bias) else None

        x = torch.where(safe_mask[:, :, None], tokens, torch.zeros_like(tokens))
        layer_weights = []
        for layer in self.layers:
            x, weights = layer(x, safe_mask, pair_bias=pair_bias, need_weights=need_weights)
            x = torch.where(mask[:, :, None], x, torch.zeros_like(x))
            if weights is not None:
                weights = torch.where(mask[:, None, :, None], weights, torch.zeros_like(weights))
                weights = torch.where(mask[:, None, None, :], weights, torch.zeros_like(weights))
                layer_weights.append(weights)
        x = self.output_norm(x)
        x = torch.where(mask[:, :, None], _nan_to_num_torch(x), torch.zeros_like(x))
        attention_weights = torch.stack(layer_weights, dim=0) if layer_weights else None
        pair_mask = pair_output.pair_mask.to(device=tokens.device, dtype=torch.bool)
        token_norm = x.norm(dim=-1)
        diagnostics = {
            "contract": MULTISCALE_SUBJET_TRANSFORMER_CONTRACT,
            "step": MULTISCALE_SUBJET_TRANSFORMER_STEP,
            "num_layers": int(self.config.num_layers),
            "num_heads": int(self.config.num_heads),
            "use_pairwise_bias": bool(self.config.use_pairwise_bias),
            "use_scale_pair_embedding": bool(self.config.use_scale_pair_embedding),
            "valid_subjet_fraction": float(mask.float().mean().detach().cpu().item()),
            "valid_pair_fraction": float(pair_mask.float().mean().detach().cpu().item()),
            "subjet_token_norm_mean": _masked_mean(token_norm, mask),
            "pair_feature_abs_mean": _masked_mean(pair_output.pair_features.abs().mean(dim=-1), pair_mask),
            "pair_delta_r_mean": pair_output.diagnostics.get("delta_r_mean"),
            "pair_log_mass_mean": pair_output.diagnostics.get("log_pair_mass_mean"),
            "pair_assignment_overlap_mean": pair_output.diagnostics.get("assignment_overlap_mean"),
            "pair_cluster_overlap_mean": pair_output.diagnostics.get("cluster_overlap_mean"),
        }
        if pair_bias is not None:
            diagnostics["pair_bias_abs_mean"] = _masked_mean(pair_bias.abs().mean(dim=1), pair_mask)
        else:
            diagnostics["pair_bias_abs_mean"] = None
        if attention_weights is not None:
            valid_weights = attention_weights.clamp_min(1.0e-12)
            entropy = -(attention_weights * valid_weights.log()).sum(dim=-1)
            attention_mask = mask[None, :, None, :]
            diagnostics["attention_entropy_mean"] = _masked_mean(entropy, attention_mask.expand_as(entropy))
            diag = torch.diagonal(attention_weights, dim1=-2, dim2=-1)
            diagnostics["attention_self_weight_mean"] = _masked_mean(diag, mask[None, :, None, :].expand_as(diag))
        return MultiScaleSubjetTransformerOutput(
            subjet_tokens=x,
            subjet_mask=mask,
            pair_features=pair_output.pair_features,
            pair_mask=pair_output.pair_mask,
            pair_bias=pair_bias,
            attention_weights=attention_weights,
            pair_feature_output=pair_output,
            diagnostics=diagnostics,
        )


MultiscaleSubjetPairFeatureConfig = MultiScaleSubjetPairFeatureConfig
MultiscaleSubjetPairFeatureOutput = MultiScaleSubjetPairFeatureOutput
MultiscaleSubjetTransformerConfig = MultiScaleSubjetTransformerConfig
MultiscaleSubjetTransformerOutput = MultiScaleSubjetTransformerOutput
MultiscaleSubjetPairBiasEncoder = MultiScaleSubjetPairBiasEncoder
MultiscaleSubjetTransformerBlock = MultiScaleSubjetTransformerBlock
MultiscaleSubjetTransformer = MultiScaleSubjetTransformer

SubjetPairFeatureConfig = MultiScaleSubjetPairFeatureConfig
SubjetPairFeatureOutput = MultiScaleSubjetPairFeatureOutput
SubjetPairBiasConfig = MultiScaleSubjetTransformerConfig
SubjetPairBiasEncoder = MultiScaleSubjetPairBiasEncoder
SubjetTransformerConfig = MultiScaleSubjetTransformerConfig
SubjetTransformerOutput = MultiScaleSubjetTransformerOutput
SubjetSubjetTransformer = MultiScaleSubjetTransformer
PairwiseBiasedSubjetAttentionBlock = MultiScaleSubjetTransformerBlock
MultiscaleSubjetPairBiasConfig = MultiScaleSubjetTransformerConfig
MultiscaleSubjetSubjetTransformer = MultiScaleSubjetTransformer
build_subjet_pair_features = build_multiscale_subjet_pair_features
