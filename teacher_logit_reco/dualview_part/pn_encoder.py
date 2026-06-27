"""Step 4 PN reconstructed-view memory encoder.

The dual-view branch uses the PN reconstructed view as auxiliary evidence for
the strong HLT ParT anchor.  This module keeps that branch intentionally light:
raw reconstructed particle tokens and existence/confidence scores are embedded,
passed through a mask-safe Transformer encoder, and returned as PN memory tokens
plus a pooled context vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .config import DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE, DUALVIEW_PART_VIEW_PN_RECO


DUALVIEW_PART_STEP4 = "reliability_gated_dualview_part_step4_pn_memory_encoder"
DUALVIEW_PART_PN_ENCODER_CONTRACT = "pn_reco_memory_tokens_plus_context_v1"


if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    import torch as _torch

    _ModuleBase = _torch.nn.Module


def _positive_int(value: int, *, field_name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _dropout_value(value: float, *, field_name: str) -> float:
    value = float(value)
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1)")
    return value


@dataclass(frozen=True)
class PNMemoryEncoderConfig:
    """Configuration for the PN reconstructed-view memory encoder."""

    raw_token_dim: int = RAW_TOKEN_DIM
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    use_confidence: bool = True
    confidence_hidden_dim: int | None = None
    view_name: str = DUALVIEW_PART_VIEW_PN_RECO
    source_architecture: str = DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE
    output_contract: str = DUALVIEW_PART_PN_ENCODER_CONTRACT
    experiment_step: str = DUALVIEW_PART_STEP4

    def __post_init__(self) -> None:
        raw_token_dim = _positive_int(self.raw_token_dim, field_name="raw_token_dim")
        embed_dim = _positive_int(self.embed_dim, field_name="embed_dim")
        num_layers = _positive_int(self.num_layers, field_name="num_layers")
        num_heads = _positive_int(self.num_heads, field_name="num_heads")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        mlp_ratio = float(self.mlp_ratio)
        if mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        confidence_hidden_dim = self.confidence_hidden_dim
        if confidence_hidden_dim is None:
            confidence_hidden_dim = embed_dim
        confidence_hidden_dim = _positive_int(confidence_hidden_dim, field_name="confidence_hidden_dim")
        if not isinstance(self.use_confidence, bool):
            raise TypeError("use_confidence must be a bool")
        view_name = str(self.view_name).strip().lower().replace("-", "_").replace(" ", "_")
        if not view_name:
            raise ValueError("view_name must be non-empty")
        source_architecture = str(self.source_architecture).strip().lower()
        if not source_architecture:
            raise ValueError("source_architecture must be non-empty")
        object.__setattr__(self, "raw_token_dim", raw_token_dim)
        object.__setattr__(self, "embed_dim", embed_dim)
        object.__setattr__(self, "num_layers", num_layers)
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "mlp_ratio", mlp_ratio)
        object.__setattr__(self, "dropout", _dropout_value(self.dropout, field_name="dropout"))
        object.__setattr__(
            self,
            "attention_dropout",
            _dropout_value(self.attention_dropout, field_name="attention_dropout"),
        )
        object.__setattr__(self, "confidence_hidden_dim", confidence_hidden_dim)
        object.__setattr__(self, "view_name", view_name)
        object.__setattr__(self, "source_architecture", source_architecture)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "PNMemoryEncoderConfig" | None) -> "PNMemoryEncoderConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = dict(value)
        payload.pop("output_contract", None)
        payload.pop("experiment_step", None)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PNMemoryEncoderOutput:
    """Encoded PN reconstructed-view memory and pooled context."""

    memory: Any
    memory_mask: Any
    context: Any
    confidence: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory,
            "memory_mask": self.memory_mask,
            "context": self.context,
            "confidence": self.confidence,
            "diagnostics": dict(self.diagnostics),
        }


def _make_transformer_encoder(
    *,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    attention_dropout: float,
):
    torch = require_torch()
    layer = torch.nn.TransformerEncoderLayer(
        d_model=int(embed_dim),
        nhead=int(num_heads),
        dim_feedforward=int(round(float(embed_dim) * float(mlp_ratio))),
        dropout=float(dropout),
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    layer.self_attn.dropout = float(attention_dropout)
    return torch.nn.TransformerEncoder(
        layer,
        num_layers=int(num_layers),
        norm=torch.nn.LayerNorm(int(embed_dim)),
    )


class PNMemoryEncoder(_ModuleBase):
    """Mask-safe encoder for the PN reconstructed particle view."""

    def __init__(self, config: Mapping[str, Any] | PNMemoryEncoderConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = PNMemoryEncoderConfig.from_mapping(config)
        dim = int(self.config.embed_dim)
        self.feature_norm = torch.nn.LayerNorm(int(self.config.raw_token_dim))
        self.feature_projection = torch.nn.Linear(int(self.config.raw_token_dim), dim)
        self.confidence_projection = torch.nn.Sequential(
            torch.nn.Linear(1, int(self.config.confidence_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.confidence_hidden_dim), dim),
        )
        self.view_type_embedding = torch.nn.Parameter(torch.empty(1, 1, dim))
        self.cls_token = torch.nn.Parameter(torch.empty(1, 1, dim))
        self.input_dropout = torch.nn.Dropout(float(self.config.dropout))
        self.encoder = _make_transformer_encoder(
            embed_dim=dim,
            num_layers=int(self.config.num_layers),
            num_heads=int(self.config.num_heads),
            mlp_ratio=float(self.config.mlp_ratio),
            dropout=float(self.config.dropout),
            attention_dropout=float(self.config.attention_dropout),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch = require_torch()
        torch.nn.init.normal_(self.view_type_embedding, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @property
    def output_contract(self) -> str:
        return str(self.config.output_contract)

    @property
    def memory_dim(self) -> int:
        return int(self.config.embed_dim)

    @property
    def context_dim(self) -> int:
        return int(self.config.embed_dim)

    def to_config_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    def _validate_and_prepare_inputs(self, tokens: Any, mask: Any, confidence: Any | None):
        torch = require_torch()
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens, dtype=torch.float32)
        tokens = tokens.float()
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [batch, tokens, features], got {tuple(tokens.shape)}")
        if int(tokens.shape[-1]) != int(self.config.raw_token_dim):
            raise ValueError(f"expected token feature dim {self.config.raw_token_dim}, got {tokens.shape[-1]}")
        device = tokens.device
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 3 and int(mask.shape[1]) == 1:
            mask = mask[:, 0, :]
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape [batch, tokens] or [batch, 1, tokens], got {tuple(mask.shape)}")
        if tuple(mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError("mask shape must match token batch/token dimensions")
        if confidence is None:
            confidence = mask.float()
        elif not isinstance(confidence, torch.Tensor):
            confidence = torch.as_tensor(confidence, dtype=torch.float32, device=device)
        else:
            confidence = confidence.to(device=device, dtype=torch.float32)
        if confidence.ndim == 3 and int(confidence.shape[1]) == 1:
            confidence = confidence[:, 0, :]
        if tuple(confidence.shape) != tuple(tokens.shape[:2]):
            raise ValueError("confidence shape must match token batch/token dimensions")
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = torch.where(mask, confidence.clamp(0.0, 1.0), torch.zeros_like(confidence))
        return tokens, mask, confidence

    def forward(
        self,
        tokens: Any,
        mask: Any,
        confidence: Any | None = None,
        *,
        return_diagnostics: bool = False,
    ):
        """Encode PN reconstructed tokens.

        Args:
            tokens: Float tensor ``[B, N, F]``.
            mask: Bool tensor ``[B, N]`` where ``True`` means valid.
            confidence: Optional float tensor ``[B, N]`` in ``[0, 1]``.
            return_diagnostics: if true, return diagnostics in the output.
        """

        torch = require_torch()
        tokens, mask, confidence = self._validate_and_prepare_inputs(tokens, mask, confidence)
        batch_size, max_tokens, _ = tokens.shape
        token_embeddings = self.feature_projection(self.feature_norm(tokens))
        if bool(self.config.use_confidence):
            token_embeddings = token_embeddings + self.confidence_projection(confidence.unsqueeze(-1))
        token_embeddings = token_embeddings + self.view_type_embedding
        token_embeddings = torch.where(mask.unsqueeze(-1), token_embeddings, torch.zeros_like(token_embeddings))
        token_embeddings = self.input_dropout(token_embeddings)

        cls = self.cls_token.expand(int(batch_size), 1, int(self.config.embed_dim))
        encoded_input = torch.cat([cls, token_embeddings], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros((int(batch_size), 1), dtype=torch.bool, device=tokens.device),
                ~mask,
            ],
            dim=1,
        )
        encoded = self.encoder(encoded_input, src_key_padding_mask=padding_mask)
        context = encoded[:, 0]
        memory = encoded[:, 1:]
        memory = torch.where(mask.unsqueeze(-1), memory, torch.zeros_like(memory))

        if not torch.isfinite(memory).all() or not torch.isfinite(context).all():
            raise FloatingPointError("PN memory encoder produced non-finite outputs")

        diagnostics = {}
        if bool(return_diagnostics):
            valid_counts = mask.sum(dim=1)
            active_confidence = (confidence * mask.float()).sum(dim=1)
            diagnostics = {
                "experiment_step": DUALVIEW_PART_STEP4,
                "output_contract": self.output_contract,
                "batch_size": int(batch_size),
                "max_tokens": int(max_tokens),
                "memory_dim": int(self.memory_dim),
                "valid_token_count_mean": float(valid_counts.float().mean().detach().cpu().item()),
                "empty_jet_fraction": float((valid_counts == 0).float().mean().detach().cpu().item()),
                "confidence_sum_mean": float(active_confidence.float().mean().detach().cpu().item()),
                "confidence_enabled": bool(self.config.use_confidence),
            }
        return PNMemoryEncoderOutput(
            memory=memory,
            memory_mask=mask,
            context=context,
            confidence=confidence,
            diagnostics=diagnostics,
        )


def build_pn_memory_encoder(
    config: Mapping[str, Any] | PNMemoryEncoderConfig | None = None,
    **kwargs: Any,
) -> PNMemoryEncoder:
    """Build the Step 4 PN reconstructed-view encoder."""

    if kwargs:
        payload = PNMemoryEncoderConfig.from_mapping(config).to_dict()
        payload.update(kwargs)
        config = payload
    return PNMemoryEncoder(config)


__all__ = [
    "DUALVIEW_PART_PN_ENCODER_CONTRACT",
    "DUALVIEW_PART_STEP4",
    "PNMemoryEncoder",
    "PNMemoryEncoderConfig",
    "PNMemoryEncoderOutput",
    "build_pn_memory_encoder",
]
