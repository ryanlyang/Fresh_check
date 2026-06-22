"""ParT-style pairwise geometry bias for subtoken particle transformers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_PAIRWISE_STEP = "subtoken_part_step10_pairwise_bias"
SUBTOKEN_PART_PAIRWISE_CONTRACT = "part_style_pairwise_attention_bias_v1"

SUBTOKEN_PART_PAIRWISE_FEATURE_NAMES = (
    "delta_eta",
    "sin_delta_phi",
    "cos_delta_phi",
    "delta_r",
    "log_delta_r",
    "delta_log_pt",
    "log_pair_mass",
    "log_relative_kt",
    "relative_kt",
    "z",
    "abs_delta_eta",
    "abs_delta_phi",
    "same_particle",
)
SUBTOKEN_PART_PAIRWISE_FEATURE_DIM = len(SUBTOKEN_PART_PAIRWISE_FEATURE_NAMES)


@dataclass(frozen=True)
class PairwiseFeatureConfig:
    """Raw-token geometry indices used by the pairwise feature builder."""

    raw_feature_dim: int = RAW_TOKEN_DIM
    pt_index: int = 0
    eta_index: int = 1
    phi_index: int = 2
    energy_index: int = 3
    include_cls_token: bool = True
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        raw_feature_dim = int(self.raw_feature_dim)
        if raw_feature_dim <= 0:
            raise ValueError("raw_feature_dim must be positive")
        for name in ("pt_index", "eta_index", "phi_index", "energy_index"):
            index = int(getattr(self, name))
            if index < 0 or index >= raw_feature_dim:
                raise ValueError(f"{name}={index} is outside raw_feature_dim={raw_feature_dim}")
            object.__setattr__(self, name, index)
        eps = float(self.eps)
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        object.__setattr__(self, "raw_feature_dim", raw_feature_dim)
        object.__setattr__(self, "include_cls_token", bool(self.include_cls_token))
        object.__setattr__(self, "eps", eps)


@dataclass(frozen=True)
class PairwiseFeatureOutput:
    """Pairwise features and masks, optionally with a leading CLS token."""

    pair_features: Any
    pair_mask: Any
    token_mask: Any
    raw_particle_mask: Any
    feature_names: tuple[str, ...]
    include_cls_token: bool

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_PAIRWISE_CONTRACT,
            "pair_features_shape": list(self.pair_features.shape),
            "pair_mask_shape": list(self.pair_mask.shape),
            "token_mask_shape": list(self.token_mask.shape),
            "raw_particle_mask_shape": list(self.raw_particle_mask.shape),
            "feature_names": list(self.feature_names),
            "include_cls_token": bool(self.include_cls_token),
        }


@dataclass(frozen=True)
class PairwiseBiasConfig:
    """MLP shape for converting pairwise geometry into per-head bias."""

    num_heads: int
    pair_feature_dim: int = SUBTOKEN_PART_PAIRWISE_FEATURE_DIM
    hidden_dim: int = 64
    dropout: float = 0.0
    mask_value: float = -1.0e4

    def __post_init__(self) -> None:
        for name in ("num_heads", "pair_feature_dim", "hidden_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        dropout = float(self.dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "mask_value", float(self.mask_value))


@dataclass(frozen=True)
class PairwiseBiasedAttentionOutput:
    """Output of one pairwise-biased global attention block."""

    tokens: Any
    attention_weights: Any | None
    attention_bias: Any
    token_mask: Any

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_PAIRWISE_CONTRACT,
            "tokens_shape": list(self.tokens.shape),
            "attention_bias_shape": list(self.attention_bias.shape),
            "token_mask_shape": list(self.token_mask.shape),
            "has_attention_weights": self.attention_weights is not None,
        }


def _normalize_feature_config(config: PairwiseFeatureConfig | Mapping[str, Any] | None = None) -> PairwiseFeatureConfig:
    if config is None:
        return PairwiseFeatureConfig()
    if isinstance(config, PairwiseFeatureConfig):
        return config
    return PairwiseFeatureConfig(**dict(config))


def _normalize_bias_config(config: PairwiseBiasConfig | Mapping[str, Any]) -> PairwiseBiasConfig:
    if isinstance(config, PairwiseBiasConfig):
        return config
    return PairwiseBiasConfig(**dict(config))


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def wrap_pairwise_delta_phi(delta_phi: Any) -> Any:
    """Wrap a delta-phi tensor to ``[-pi, pi]``."""

    torch = require_torch()
    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def _coerce_tokens_and_mask(tokens: Any, mask: Any) -> tuple[Any, Any]:
    torch = require_torch()
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
    else:
        tokens = tokens.float()
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=tokens.device)
    else:
        mask = mask.to(device=tokens.device, dtype=torch.bool)
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    return tokens, mask


class PairwiseFeatureBuilder(_ModuleBase):
    """Build wraparound-aware particle-pair geometry features from raw tokens."""

    def __init__(self, config: PairwiseFeatureConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = _normalize_feature_config(config)

    def forward(self, raw_tokens: Any, mask: Any) -> PairwiseFeatureOutput:
        torch = require_torch()
        tokens, mask = _coerce_tokens_and_mask(raw_tokens, mask)
        feature_dim = int(tokens.shape[-1])
        for name in ("pt_index", "eta_index", "phi_index", "energy_index"):
            index = int(getattr(self.config, name))
            if index >= feature_dim:
                raise ValueError(f"{name}={index} is outside input feature dimension {feature_dim}")

        pt = torch.clamp(tokens[:, :, int(self.config.pt_index)].abs(), min=float(self.config.eps))
        eta = tokens[:, :, int(self.config.eta_index)]
        phi = tokens[:, :, int(self.config.phi_index)]
        energy = torch.clamp(tokens[:, :, int(self.config.energy_index)].abs(), min=float(self.config.eps))

        delta_eta = eta[:, :, None] - eta[:, None, :]
        delta_phi = wrap_pairwise_delta_phi(phi[:, :, None] - phi[:, None, :])
        delta_r = torch.sqrt(torch.clamp(delta_eta * delta_eta + delta_phi * delta_phi, min=0.0))
        delta_log_pt = torch.log(pt[:, :, None]) - torch.log(pt[:, None, :])
        relative_kt = torch.minimum(pt[:, :, None], pt[:, None, :]) * delta_r
        z = torch.minimum(pt[:, :, None], pt[:, None, :]) / torch.clamp(pt[:, :, None] + pt[:, None, :], min=float(self.config.eps))

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(torch.clamp(eta, min=-20.0, max=20.0))
        pair_energy = energy[:, :, None] + energy[:, None, :]
        pair_px = px[:, :, None] + px[:, None, :]
        pair_py = py[:, :, None] + py[:, None, :]
        pair_pz = pz[:, :, None] + pz[:, None, :]
        pair_mass2 = torch.clamp(
            pair_energy * pair_energy - pair_px * pair_px - pair_py * pair_py - pair_pz * pair_pz,
            min=float(self.config.eps) * float(self.config.eps),
        )
        pair_mass = torch.sqrt(pair_mass2)
        num_particles = int(tokens.shape[1])
        same_particle = torch.eye(num_particles, dtype=tokens.dtype, device=tokens.device)[None, :, :]
        features = torch.stack(
            [
                torch.clamp(delta_eta / 5.0, -2.0, 2.0),
                torch.sin(delta_phi),
                torch.cos(delta_phi),
                torch.clamp(delta_r / 5.0, 0.0, 4.0),
                torch.clamp(torch.log(delta_r + float(self.config.eps)), -14.0, 4.0),
                torch.clamp(delta_log_pt, -8.0, 8.0),
                torch.clamp(torch.log(pair_mass + float(self.config.eps)), -14.0, 14.0),
                torch.clamp(torch.log(relative_kt + float(self.config.eps)), -14.0, 14.0),
                torch.clamp(relative_kt / 1000.0, 0.0, 10.0),
                torch.clamp(z, 0.0, 0.5),
                torch.clamp(delta_eta.abs() / 5.0, 0.0, 4.0),
                torch.clamp(delta_phi.abs() / math.pi, 0.0, 1.0),
                same_particle.expand(int(tokens.shape[0]), num_particles, num_particles),
            ],
            dim=-1,
        )
        pair_mask = mask[:, :, None] & mask[:, None, :]
        features = torch.where(pair_mask[:, :, :, None], _nan_to_num_torch(features), torch.zeros_like(features))
        token_mask = mask

        if bool(self.config.include_cls_token):
            batch_size = int(tokens.shape[0])
            zeros_row = torch.zeros(
                (batch_size, 1, num_particles, SUBTOKEN_PART_PAIRWISE_FEATURE_DIM),
                dtype=features.dtype,
                device=features.device,
            )
            features = torch.cat([zeros_row, features], dim=1)
            zeros_col = torch.zeros(
                (batch_size, num_particles + 1, 1, SUBTOKEN_PART_PAIRWISE_FEATURE_DIM),
                dtype=features.dtype,
                device=features.device,
            )
            features = torch.cat([zeros_col, features], dim=2)
            cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device)
            token_mask = torch.cat([cls_mask, mask], dim=1)
            pair_mask = token_mask[:, :, None] & token_mask[:, None, :]

        return PairwiseFeatureOutput(
            pair_features=features,
            pair_mask=pair_mask,
            token_mask=token_mask,
            raw_particle_mask=mask,
            feature_names=SUBTOKEN_PART_PAIRWISE_FEATURE_NAMES,
            include_cls_token=bool(self.config.include_cls_token),
        )


class PairwiseBiasEncoder(_ModuleBase):
    """Convert pairwise geometry features into additive per-head attention bias."""

    def __init__(self, config: PairwiseBiasConfig | Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        self.config = _normalize_bias_config(config)
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.pair_feature_dim)),
            torch.nn.Linear(int(self.config.pair_feature_dim), int(self.config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.hidden_dim), int(self.config.num_heads)),
        )

    def forward(self, pair_features_or_output: Any, pair_mask: Any | None = None) -> Any:
        torch = require_torch()
        if isinstance(pair_features_or_output, PairwiseFeatureOutput):
            pair_features = pair_features_or_output.pair_features
            pair_mask = pair_features_or_output.pair_mask
        else:
            pair_features = pair_features_or_output
        pair_features = _nan_to_num_torch(pair_features.float())
        if int(pair_features.ndim) != 4:
            raise ValueError(f"pair_features must have shape [batch, tokens, tokens, features], got {tuple(pair_features.shape)}")
        if int(pair_features.shape[-1]) != int(self.config.pair_feature_dim):
            raise ValueError(
                f"pair_features last dimension must be {int(self.config.pair_feature_dim)}, got {int(pair_features.shape[-1])}"
            )
        if pair_mask is None:
            pair_mask = torch.ones(pair_features.shape[:3], dtype=torch.bool, device=pair_features.device)
        else:
            pair_mask = pair_mask.to(device=pair_features.device, dtype=torch.bool)
        if tuple(pair_mask.shape) != tuple(pair_features.shape[:3]):
            raise ValueError(f"pair_mask shape {tuple(pair_mask.shape)} does not match {tuple(pair_features.shape[:3])}")
        bias = self.network(pair_features).permute(0, 3, 1, 2).contiguous()
        bias = torch.where(pair_mask[:, None, :, :], _nan_to_num_torch(bias), torch.zeros_like(bias))
        return bias


class PairwiseBiasedAttentionBlock(_ModuleBase):
    """One transformer-style global block with additive pairwise attention bias."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        mask_value: float = -1.0e4,
    ) -> None:
        torch = require_torch()
        super().__init__()
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide embed_dim")
        mlp_ratio = float(mlp_ratio)
        if mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        for name, value in (("dropout", dropout), ("attention_dropout", attention_dropout)):
            value = float(value)
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_value = float(mask_value)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=float(attention_dropout),
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(float(dropout))
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        hidden_dim = int(round(embed_dim * mlp_ratio))
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden_dim, embed_dim),
        )

    def _attention_mask(self, attention_bias: Any, token_mask: Any, *, dtype: Any, device: Any) -> Any:
        attention_bias = attention_bias.to(device=device, dtype=dtype)
        token_mask = token_mask.bool()
        batch_size, num_heads, num_tokens, other_tokens = (int(value) for value in attention_bias.shape)
        if num_heads != self.num_heads:
            raise ValueError(f"attention_bias num_heads={num_heads} does not match block num_heads={self.num_heads}")
        if num_tokens != other_tokens:
            raise ValueError("attention_bias must be square in token dimensions")
        if tuple(token_mask.shape) != (batch_size, num_tokens):
            raise ValueError(f"token_mask shape {tuple(token_mask.shape)} does not match {(batch_size, num_tokens)}")
        masked_bias = attention_bias.masked_fill(~token_mask[:, None, None, :], self.mask_value)
        return masked_bias.reshape(batch_size * num_heads, num_tokens, num_tokens)

    def forward(
        self,
        tokens: Any,
        attention_bias: Any,
        token_mask: Any,
        *,
        need_weights: bool = False,
    ) -> PairwiseBiasedAttentionOutput:
        torch = require_torch()
        tokens = _nan_to_num_torch(tokens.float())
        token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        if int(tokens.ndim) != 3:
            raise ValueError(f"tokens must have shape [batch, tokens, embed_dim], got {tuple(tokens.shape)}")
        if int(tokens.shape[-1]) != self.embed_dim:
            raise ValueError(f"tokens last dimension must be embed_dim={self.embed_dim}, got {int(tokens.shape[-1])}")
        if tuple(token_mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError(f"token_mask shape {tuple(token_mask.shape)} does not match {tuple(tokens.shape[:2])}")
        if tuple(attention_bias.shape[:1]) != tuple(tokens.shape[:1]) or tuple(attention_bias.shape[2:]) != tuple(
            (int(tokens.shape[1]), int(tokens.shape[1]))
        ):
            raise ValueError("attention_bias shape does not match tokens")
        x = torch.where(token_mask[:, :, None], tokens, torch.zeros_like(tokens))
        normed = self.norm1(x)
        attn_mask = self._attention_mask(attention_bias, token_mask, dtype=normed.dtype, device=normed.device)
        attended, weights = self.attention(
            normed,
            normed,
            normed,
            key_padding_mask=~token_mask,
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
        return PairwiseBiasedAttentionOutput(
            tokens=x,
            attention_weights=weights,
            attention_bias=attention_bias,
            token_mask=token_mask,
        )


__all__ = [
    "SUBTOKEN_PART_PAIRWISE_CONTRACT",
    "SUBTOKEN_PART_PAIRWISE_FEATURE_DIM",
    "SUBTOKEN_PART_PAIRWISE_FEATURE_NAMES",
    "SUBTOKEN_PART_PAIRWISE_STEP",
    "PairwiseBiasConfig",
    "PairwiseBiasEncoder",
    "PairwiseBiasedAttentionBlock",
    "PairwiseBiasedAttentionOutput",
    "PairwiseFeatureBuilder",
    "PairwiseFeatureConfig",
    "PairwiseFeatureOutput",
    "wrap_pairwise_delta_phi",
]
