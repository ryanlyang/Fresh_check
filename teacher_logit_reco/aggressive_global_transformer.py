"""Aggressive global-transformer reconstructor.

This is the first full aggressive reconstructor family.  It keeps the
ParT-like/global transformer encoder style from the conservative branch, but
delegates all reconstructed-view mechanics to ``AggressiveSoftViewHead``:
parent edits, parent weights, 64 extra slots, global calibration, and budget
diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM

from .aggressive_soft_view import (
    AGGRESSION_LEVEL,
    AggressiveSoftViewConfig,
    AggressiveSoftViewHead,
)
from .global_transformer import (
    TOKEN_EMBED_FEATURE_DIM,
    placeholder_jet_ids,
    sanitize_hlt_tokens,
    token_embedding_features,
)
from .views import SoftReconstructedView

try:  # Keep imports lightweight on machines without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR = "aggressive_global_transformer"


@dataclass
class AggressiveGlobalTransformerReconstructorConfig:
    """Configuration for the aggressive ParT-ish/global-transformer encoder."""

    input_dim: int = RAW_TOKEN_DIM
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.05
    aggressive_head_config: Dict[str, Any] = field(default_factory=dict)
    reconstructor_architecture: str = AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR
    aggression_level: str = AGGRESSION_LEVEL

    def __post_init__(self) -> None:
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if int(self.num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(self.hidden_dim) % int(self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        head_config = dict(self.aggressive_head_config or {})
        if "embedding_dim" in head_config and int(head_config["embedding_dim"]) != int(self.hidden_dim):
            raise ValueError("aggressive_head_config.embedding_dim must match hidden_dim")
        if "input_dim" in head_config and int(head_config["input_dim"]) != RAW_TOKEN_DIM:
            raise ValueError(f"aggressive_head_config.input_dim must be {RAW_TOKEN_DIM}")
        head_config["input_dim"] = RAW_TOKEN_DIM
        head_config["embedding_dim"] = int(self.hidden_dim)
        head_config.setdefault("dropout", float(self.dropout))
        head_config.setdefault("aggression_level", str(self.aggression_level))
        self.aggressive_head_config = AggressiveSoftViewConfig.from_mapping(head_config).to_dict()
        self.reconstructor_architecture = AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR
        self.aggression_level = str(self.aggressive_head_config["aggression_level"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "AggressiveGlobalTransformerReconstructorConfig" | None,
    ) -> "AggressiveGlobalTransformerReconstructorConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value

        payload = dict(value)
        payload.pop("architecture", None)
        payload.pop("reconstructor_architecture", None)
        payload.pop("aggression_level", None)
        nested_head = dict(payload.pop("aggressive_head_config", payload.pop("head_config", {})) or {})

        config_field_names = {item.name for item in fields(cls)}
        head_field_names = set(AggressiveSoftViewConfig().to_dict())
        config_kwargs: Dict[str, Any] = {}
        unknown: Dict[str, Any] = {}
        for key, item in payload.items():
            if key in config_field_names and key != "aggressive_head_config":
                config_kwargs[key] = item
            elif key in head_field_names:
                nested_head[key] = item
            else:
                unknown[key] = item
        if unknown:
            keys = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown aggressive global transformer config keys: {keys}")
        config_kwargs["aggressive_head_config"] = nested_head
        return cls(**config_kwargs)


def _masked_mean(values, mask):
    torch = require_torch()
    weights = mask.float()
    denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
    return (values * weights[:, :, None]).sum(dim=1) / denom


class AggressiveGlobalTransformerReconstructor(_ModuleBase):
    """ParT-like encoder with the shared aggressive soft-view head."""

    def __init__(
        self,
        config: Mapping[str, Any] | AggressiveGlobalTransformerReconstructorConfig | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.config = AggressiveGlobalTransformerReconstructorConfig.from_mapping(config)
        dim = int(self.config.hidden_dim)

        self.input_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(TOKEN_EMBED_FEATURE_DIM),
            torch.nn.Linear(TOKEN_EMBED_FEATURE_DIM, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(self.config.num_heads),
            dim_feedforward=dim * 4,
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(self.config.num_layers),
            norm=torch.nn.LayerNorm(dim),
        )
        self.context_norm = torch.nn.LayerNorm(dim)
        self.soft_view_head = AggressiveSoftViewHead(self.config.aggressive_head_config)

    def encode(self, hlt_tokens, hlt_mask):
        """Return sanitized inputs, particle embeddings, and global context."""

        torch = require_torch()
        hlt_tokens, hlt_mask, diagnostics = sanitize_hlt_tokens(
            hlt_tokens,
            hlt_mask,
            config=self.soft_view_head.config,
        )
        features = token_embedding_features(hlt_tokens, hlt_mask)
        encoded = self.input_proj(features)
        encoded = self.encoder(encoded, src_key_padding_mask=~hlt_mask.bool())
        encoded = torch.where(hlt_mask[:, :, None], encoded, torch.zeros_like(encoded))
        global_context = self.context_norm(_masked_mean(encoded, hlt_mask))
        return hlt_tokens, hlt_mask, encoded, global_context, diagnostics

    def forward(
        self,
        hlt_tokens,
        hlt_mask,
        *,
        labels=None,
        jet_ids: list[JetIdentity] | None = None,
        split: str = "in_memory",
    ) -> SoftReconstructedView:
        torch = require_torch()
        hlt_tokens, hlt_mask, particle_embeddings, global_context, encoder_diagnostics = self.encode(hlt_tokens, hlt_mask)
        batch_size = int(hlt_tokens.shape[0])
        if labels is None:
            labels = torch.full((batch_size,), -1, dtype=torch.long, device=hlt_tokens.device)
        elif isinstance(labels, torch.Tensor):
            labels = labels.to(device=hlt_tokens.device, dtype=torch.long)
        if jet_ids is None:
            jet_ids = placeholder_jet_ids(batch_size, labels=labels)

        return self.soft_view_head(
            hlt_tokens,
            hlt_mask,
            particle_embeddings,
            global_context,
            labels=labels,
            jet_ids=jet_ids,
            split=split,
            metadata={
                "construction": "aggressive_global_transformer_soft_view",
                "model_family": "teacher_logit_aggressive_global_transformer",
                "reconstructor_architecture": AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR,
                "encoder_architecture": "global_transformer",
                "aggression_level": self.config.aggression_level,
                "transformer_config": self.config.to_dict(),
                "encoder_diagnostics": encoder_diagnostics,
            },
        )


def build_aggressive_global_transformer_reconstructor(
    config: Mapping[str, Any] | AggressiveGlobalTransformerReconstructorConfig | None = None,
) -> AggressiveGlobalTransformerReconstructor:
    return AggressiveGlobalTransformerReconstructor(config)


__all__ = [
    "AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR",
    "AggressiveGlobalTransformerReconstructor",
    "AggressiveGlobalTransformerReconstructorConfig",
    "build_aggressive_global_transformer_reconstructor",
]
