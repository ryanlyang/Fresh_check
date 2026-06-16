"""Aggressive PN, PFN, and P-CNN teacher-logit reconstructors.

These classes keep architecture diversity in the encoder and keep the output
mechanism fixed through ``AggressiveSoftViewHead``.  Each encoder produces:

``particle_embeddings: [B, N, D]``
``global_context:      [B, D]``

Then the shared aggressive head handles parent edits, parent pruning,
extra-candidate generation, global calibration, and diagnostics.
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
from .global_transformer import placeholder_jet_ids, sanitize_hlt_tokens
from .particle_cnn_reconstructor import (
    PARTICLE_CNN_INPUT_FEATURE_DIM,
    PARTICLE_CNN_ORDERING_ASSUMPTION,
    ParticleCnnEncoder,
    audit_particle_cnn_cache_order,
    build_particle_cnn_features,
)
from .particle_flow_reconstructor import (
    PARTICLE_FLOW_INPUT_FEATURE_DIM,
    PARTICLE_FLOW_SUMMARY_FEATURE_DIM,
    ParticleFlowEncoder,
    build_particle_flow_features,
    build_particle_flow_summary_features,
)
from .particle_net_reconstructor import (
    PARTICLE_NET_INPUT_FEATURE_DIM,
    ParticleNetEncoder,
    particle_net_input_features,
    particle_net_knn_coordinates,
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


AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR = "aggressive_particle_net"
AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR = "aggressive_particle_flow"
AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR = "aggressive_particle_cnn"


def _as_positive_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        dims = (int(value),)
    else:
        dims = tuple(int(dim) for dim in value)
    if not dims:
        raise ValueError(f"{field_name} must contain at least one dimension")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"{field_name} must contain only positive dimensions")
    return dims


def _split_config_payload(cls, value: Mapping[str, Any], *, tuple_keys: tuple[str, ...]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    payload = dict(value)
    payload.pop("architecture", None)
    payload.pop("reconstructor_architecture", None)
    payload.pop("aggression_level", None)
    head_config = dict(payload.pop("aggressive_head_config", payload.pop("head_config", {})) or {})

    config_field_names = {item.name for item in fields(cls)}
    head_field_names = set(AggressiveSoftViewConfig().to_dict())
    config_kwargs: Dict[str, Any] = {}
    unknown: Dict[str, Any] = {}
    for key, item in payload.items():
        if key in config_field_names and key != "aggressive_head_config":
            config_kwargs[key] = item
        elif key in head_field_names:
            head_config[key] = item
        else:
            unknown[key] = item
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown aggressive reconstructor config keys: {keys}")
    for key in tuple_keys:
        if key in config_kwargs:
            config_kwargs[key] = tuple(config_kwargs[key])
    return config_kwargs, head_config


def _finalize_head_config(
    *,
    head_config: Mapping[str, Any] | None,
    embedding_dim: int,
    dropout: float,
    aggression_level: str,
) -> Dict[str, Any]:
    payload = dict(head_config or {})
    if "embedding_dim" in payload and int(payload["embedding_dim"]) != int(embedding_dim):
        raise ValueError("aggressive_head_config.embedding_dim must match embedding_dim")
    if "input_dim" in payload and int(payload["input_dim"]) != RAW_TOKEN_DIM:
        raise ValueError(f"aggressive_head_config.input_dim must be {RAW_TOKEN_DIM}")
    payload["input_dim"] = RAW_TOKEN_DIM
    payload["embedding_dim"] = int(embedding_dim)
    payload.setdefault("dropout", float(dropout))
    payload.setdefault("aggression_level", str(aggression_level))
    return AggressiveSoftViewConfig.from_mapping(payload).to_dict()


def _masked_mean(values, mask):
    torch = require_torch()
    weights = mask.float()
    denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
    return (values * weights[:, :, None]).sum(dim=1) / denom


def _masked_max(values, mask):
    torch = require_torch()
    if int(values.shape[1]) == 0:
        return values.new_zeros(values.shape[0], values.shape[-1])
    very_negative = torch.finfo(values.dtype).min / 8.0
    masked = values.masked_fill(~mask[:, :, None].bool(), very_negative)
    max_values = masked.max(dim=1).values
    return torch.where(mask.any(dim=1)[:, None], max_values, torch.zeros_like(max_values))


def _projection(input_dim: int, output_dim: int, *, dropout: float):
    torch = require_torch()
    return torch.nn.Sequential(
        torch.nn.LayerNorm(int(input_dim)),
        torch.nn.Linear(int(input_dim), int(output_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.LayerNorm(int(output_dim)),
    )


@dataclass
class AggressiveParticleNetReconstructorConfig:
    input_dim: int = RAW_TOKEN_DIM
    edgeconv_dims: tuple[int, ...] = (64, 128, 128)
    k: int = 16
    dropout: float = 0.05
    embedding_dim: int | None = None
    aggressive_head_config: Dict[str, Any] = field(default_factory=dict)
    reconstructor_architecture: str = AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR
    aggression_level: str = AGGRESSION_LEVEL

    def __post_init__(self) -> None:
        self.edgeconv_dims = _as_positive_int_tuple(self.edgeconv_dims, field_name="edgeconv_dims")
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        if int(self.k) <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.embedding_dim is None:
            self.embedding_dim = int(self.edgeconv_dims[-1])
        self.embedding_dim = int(self.embedding_dim)
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.aggressive_head_config = _finalize_head_config(
            head_config=self.aggressive_head_config,
            embedding_dim=self.embedding_dim,
            dropout=float(self.dropout),
            aggression_level=str(self.aggression_level),
        )
        self.reconstructor_architecture = AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR
        self.aggression_level = str(self.aggressive_head_config["aggression_level"])

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["edgeconv_dims"] = list(self.edgeconv_dims)
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "AggressiveParticleNetReconstructorConfig" | None,
    ) -> "AggressiveParticleNetReconstructorConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        config_kwargs, head_config = _split_config_payload(cls, value, tuple_keys=("edgeconv_dims",))
        config_kwargs["aggressive_head_config"] = head_config
        return cls(**config_kwargs)


@dataclass
class AggressiveParticleFlowReconstructorConfig:
    input_dim: int = RAW_TOKEN_DIM
    phi_dims: tuple[int, ...] = (128, 128, 128)
    context_dim: int = 256
    context_mlp_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.05
    embedding_dim: int = 128
    aggressive_head_config: Dict[str, Any] = field(default_factory=dict)
    reconstructor_architecture: str = AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR
    aggression_level: str = AGGRESSION_LEVEL

    def __post_init__(self) -> None:
        self.phi_dims = _as_positive_int_tuple(self.phi_dims, field_name="phi_dims")
        self.context_dim = int(self.context_dim)
        self.context_mlp_dims = _as_positive_int_tuple(self.context_mlp_dims, field_name="context_mlp_dims")
        self.embedding_dim = int(self.embedding_dim)
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.aggressive_head_config = _finalize_head_config(
            head_config=self.aggressive_head_config,
            embedding_dim=self.embedding_dim,
            dropout=float(self.dropout),
            aggression_level=str(self.aggression_level),
        )
        self.reconstructor_architecture = AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR
        self.aggression_level = str(self.aggressive_head_config["aggression_level"])

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["phi_dims"] = list(self.phi_dims)
        payload["context_mlp_dims"] = list(self.context_mlp_dims)
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "AggressiveParticleFlowReconstructorConfig" | None,
    ) -> "AggressiveParticleFlowReconstructorConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = dict(value)
        if "context_dims" in payload and "context_mlp_dims" not in payload:
            payload["context_mlp_dims"] = payload.pop("context_dims")
        config_kwargs, head_config = _split_config_payload(
            cls,
            payload,
            tuple_keys=("phi_dims", "context_mlp_dims"),
        )
        config_kwargs["aggressive_head_config"] = head_config
        return cls(**config_kwargs)


@dataclass
class AggressiveParticleCnnReconstructorConfig:
    input_dim: int = RAW_TOKEN_DIM
    hidden_channels: int = 128
    num_blocks: int = 6
    kernel_sizes: tuple[int, ...] = (5, 5, 3, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4)
    context_dim: int = 256
    context_mlp_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.05
    embedding_dim: int = 128
    aggressive_head_config: Dict[str, Any] = field(default_factory=dict)
    reconstructor_architecture: str = AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR
    aggression_level: str = AGGRESSION_LEVEL

    def __post_init__(self) -> None:
        self.hidden_channels = int(self.hidden_channels)
        self.num_blocks = int(self.num_blocks)
        self.kernel_sizes = _as_positive_int_tuple(self.kernel_sizes, field_name="kernel_sizes")
        self.dilations = _as_positive_int_tuple(self.dilations, field_name="dilations")
        self.context_dim = int(self.context_dim)
        self.context_mlp_dims = _as_positive_int_tuple(self.context_mlp_dims, field_name="context_mlp_dims")
        self.embedding_dim = int(self.embedding_dim)
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if len(self.kernel_sizes) != self.num_blocks:
            raise ValueError("kernel_sizes length must match num_blocks")
        if len(self.dilations) != self.num_blocks:
            raise ValueError("dilations length must match num_blocks")
        if any(kernel % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("kernel_sizes must be odd so Conv1d blocks preserve rank alignment")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.aggressive_head_config = _finalize_head_config(
            head_config=self.aggressive_head_config,
            embedding_dim=self.embedding_dim,
            dropout=float(self.dropout),
            aggression_level=str(self.aggression_level),
        )
        self.reconstructor_architecture = AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR
        self.aggression_level = str(self.aggressive_head_config["aggression_level"])

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["kernel_sizes"] = list(self.kernel_sizes)
        payload["dilations"] = list(self.dilations)
        payload["context_mlp_dims"] = list(self.context_mlp_dims)
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "AggressiveParticleCnnReconstructorConfig" | None,
    ) -> "AggressiveParticleCnnReconstructorConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = dict(value)
        if "context_dims" in payload and "context_mlp_dims" not in payload:
            payload["context_mlp_dims"] = payload.pop("context_dims")
        config_kwargs, head_config = _split_config_payload(
            cls,
            payload,
            tuple_keys=("kernel_sizes", "dilations", "context_mlp_dims"),
        )
        config_kwargs["aggressive_head_config"] = head_config
        return cls(**config_kwargs)


class AggressiveParticleNetReconstructor(_ModuleBase):
    """ParticleNet encoder feeding the shared aggressive soft-view head."""

    def __init__(self, config: Mapping[str, Any] | AggressiveParticleNetReconstructorConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = AggressiveParticleNetReconstructorConfig.from_mapping(config)
        self.encoder = ParticleNetEncoder(
            input_dim=PARTICLE_NET_INPUT_FEATURE_DIM,
            hidden_dims=self.config.edgeconv_dims,
            k=int(self.config.k),
            dropout=float(self.config.dropout),
        )
        encoder_dim = int(self.encoder.output_dim)
        embedding_dim = int(self.config.embedding_dim)
        self.particle_projection = _projection(encoder_dim, embedding_dim, dropout=float(self.config.dropout))
        self.context_projection = _projection(2 * encoder_dim, embedding_dim, dropout=float(self.config.dropout))
        self.soft_view_head = AggressiveSoftViewHead(self.config.aggressive_head_config)

    def encode(self, hlt_tokens, hlt_mask):
        torch = require_torch()
        hlt_tokens, hlt_mask, diagnostics = sanitize_hlt_tokens(
            hlt_tokens,
            hlt_mask,
            config=self.soft_view_head.config,
        )
        features = particle_net_input_features(hlt_tokens, hlt_mask)
        coords = particle_net_knn_coordinates(hlt_tokens, hlt_mask)
        encoded = self.encoder(features, coords, hlt_mask)
        particle_embeddings = self.particle_projection(encoded)
        particle_embeddings = torch.where(hlt_mask[:, :, None], particle_embeddings, torch.zeros_like(particle_embeddings))
        pooled = torch.cat([_masked_mean(encoded, hlt_mask), _masked_max(encoded, hlt_mask)], dim=-1)
        global_context = self.context_projection(pooled)
        encoder_aux = {
            "particle_net_features": features,
            "particle_net_coords": coords,
            "encoded_parent_features": encoded,
            "global_context": global_context,
        }
        return hlt_tokens, hlt_mask, particle_embeddings, global_context, diagnostics, encoder_aux

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
        hlt_tokens, hlt_mask, particle_embeddings, global_context, diagnostics, encoder_aux = self.encode(hlt_tokens, hlt_mask)
        labels, jet_ids = _labels_and_jet_ids(hlt_tokens, labels, jet_ids)
        view = self.soft_view_head(
            hlt_tokens,
            hlt_mask,
            particle_embeddings,
            global_context,
            labels=labels,
            jet_ids=jet_ids,
            split=split,
            metadata={
                "construction": "aggressive_particle_net_soft_view",
                "model_family": "teacher_logit_aggressive_particle_net",
                "reconstructor_architecture": AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR,
                "encoder_architecture": "particle_net",
                "aggression_level": self.config.aggression_level,
                "encoder_diagnostics": diagnostics,
                "particle_net_config": self.config.to_dict(),
            },
        )
        view.aux.update(encoder_aux)
        return view


class AggressiveParticleFlowReconstructor(_ModuleBase):
    """PFN/DeepSets encoder feeding the shared aggressive soft-view head."""

    def __init__(self, config: Mapping[str, Any] | AggressiveParticleFlowReconstructorConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = AggressiveParticleFlowReconstructorConfig.from_mapping(config)
        self.encoder = ParticleFlowEncoder(
            input_dim=PARTICLE_FLOW_INPUT_FEATURE_DIM,
            phi_dims=self.config.phi_dims,
            context_dim=int(self.config.context_dim),
            context_mlp_dims=self.config.context_mlp_dims,
            summary_dim=PARTICLE_FLOW_SUMMARY_FEATURE_DIM,
            dropout=float(self.config.dropout),
        )
        embedding_dim = int(self.config.embedding_dim)
        self.particle_projection = _projection(int(self.encoder.output_dim), embedding_dim, dropout=float(self.config.dropout))
        self.context_projection = _projection(int(self.config.context_dim), embedding_dim, dropout=float(self.config.dropout))
        self.soft_view_head = AggressiveSoftViewHead(self.config.aggressive_head_config)

    def encode(self, hlt_tokens, hlt_mask):
        torch = require_torch()
        hlt_tokens, hlt_mask, diagnostics = sanitize_hlt_tokens(
            hlt_tokens,
            hlt_mask,
            config=self.soft_view_head.config,
        )
        features = build_particle_flow_features(hlt_tokens, hlt_mask)
        summary_features = build_particle_flow_summary_features(hlt_tokens, hlt_mask)
        encoder_output = self.encoder(features, hlt_mask, summary_features=summary_features)
        particle_embeddings = self.particle_projection(encoder_output.particle_embeddings)
        particle_embeddings = torch.where(hlt_mask[:, :, None], particle_embeddings, torch.zeros_like(particle_embeddings))
        global_context = self.context_projection(encoder_output.jet_context)
        encoder_aux = {
            "particle_flow_features": features,
            "particle_flow_summary_features": summary_features,
            "particle_flow_embeddings": encoder_output.particle_embeddings,
            "pooling_report": encoder_output.pooling_report,
            "jet_context": encoder_output.jet_context,
            "global_context": global_context,
        }
        return hlt_tokens, hlt_mask, particle_embeddings, global_context, diagnostics, encoder_aux

    def forward(
        self,
        hlt_tokens,
        hlt_mask,
        *,
        labels=None,
        jet_ids: list[JetIdentity] | None = None,
        split: str = "in_memory",
    ) -> SoftReconstructedView:
        hlt_tokens, hlt_mask, particle_embeddings, global_context, diagnostics, encoder_aux = self.encode(hlt_tokens, hlt_mask)
        labels, jet_ids = _labels_and_jet_ids(hlt_tokens, labels, jet_ids)
        view = self.soft_view_head(
            hlt_tokens,
            hlt_mask,
            particle_embeddings,
            global_context,
            labels=labels,
            jet_ids=jet_ids,
            split=split,
            metadata={
                "construction": "aggressive_particle_flow_soft_view",
                "model_family": "teacher_logit_aggressive_particle_flow",
                "reconstructor_architecture": AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR,
                "encoder_architecture": "particle_flow",
                "aggression_level": self.config.aggression_level,
                "encoder_diagnostics": diagnostics,
                "particle_flow_config": self.config.to_dict(),
            },
        )
        view.aux.update(encoder_aux)
        return view


class AggressiveParticleCnnReconstructor(_ModuleBase):
    """Rank-convolution encoder feeding the shared aggressive soft-view head."""

    def __init__(self, config: Mapping[str, Any] | AggressiveParticleCnnReconstructorConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = AggressiveParticleCnnReconstructorConfig.from_mapping(config)
        self.encoder = ParticleCnnEncoder(
            input_dim=PARTICLE_CNN_INPUT_FEATURE_DIM,
            hidden_channels=int(self.config.hidden_channels),
            kernel_sizes=self.config.kernel_sizes,
            dilations=self.config.dilations,
            context_dim=int(self.config.context_dim),
            context_mlp_dims=self.config.context_mlp_dims,
            summary_dim=0,
            dropout=float(self.config.dropout),
        )
        embedding_dim = int(self.config.embedding_dim)
        self.particle_projection = _projection(int(self.encoder.output_dim), embedding_dim, dropout=float(self.config.dropout))
        self.context_projection = _projection(int(self.config.context_dim), embedding_dim, dropout=float(self.config.dropout))
        self.soft_view_head = AggressiveSoftViewHead(self.config.aggressive_head_config)

    def encode(self, hlt_tokens, hlt_mask):
        torch = require_torch()
        hlt_tokens, hlt_mask, diagnostics = sanitize_hlt_tokens(
            hlt_tokens,
            hlt_mask,
            config=self.soft_view_head.config,
        )
        cache_order_audit = audit_particle_cnn_cache_order(hlt_tokens, hlt_mask)
        features = build_particle_cnn_features(hlt_tokens, hlt_mask)
        encoder_output = self.encoder(features, hlt_mask)
        particle_embeddings = self.particle_projection(encoder_output.particle_embeddings)
        particle_embeddings = torch.where(hlt_mask[:, :, None], particle_embeddings, torch.zeros_like(particle_embeddings))
        global_context = self.context_projection(encoder_output.jet_context)
        diagnostics = {
            **diagnostics,
            "cache_order_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
            "cache_order_audit": cache_order_audit,
        }
        encoder_aux = {
            "particle_cnn_features": features,
            "rank_features": encoder_output.rank_features,
            "particle_cnn_embeddings": encoder_output.particle_embeddings,
            "pooling_report": encoder_output.pooling_report,
            "jet_context": encoder_output.jet_context,
            "global_context": global_context,
            "cache_order_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
            "cache_order_audit": cache_order_audit,
        }
        return hlt_tokens, hlt_mask, particle_embeddings, global_context, diagnostics, encoder_aux

    def forward(
        self,
        hlt_tokens,
        hlt_mask,
        *,
        labels=None,
        jet_ids: list[JetIdentity] | None = None,
        split: str = "in_memory",
    ) -> SoftReconstructedView:
        hlt_tokens, hlt_mask, particle_embeddings, global_context, diagnostics, encoder_aux = self.encode(hlt_tokens, hlt_mask)
        labels, jet_ids = _labels_and_jet_ids(hlt_tokens, labels, jet_ids)
        view = self.soft_view_head(
            hlt_tokens,
            hlt_mask,
            particle_embeddings,
            global_context,
            labels=labels,
            jet_ids=jet_ids,
            split=split,
            metadata={
                "construction": "aggressive_particle_cnn_soft_view",
                "model_family": "teacher_logit_aggressive_particle_cnn",
                "reconstructor_architecture": AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR,
                "encoder_architecture": "particle_cnn",
                "ordering_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
                "cache_order_audit": encoder_aux["cache_order_audit"],
                "aggression_level": self.config.aggression_level,
                "encoder_diagnostics": diagnostics,
                "particle_cnn_config": self.config.to_dict(),
            },
        )
        view.aux.update(encoder_aux)
        return view


def _labels_and_jet_ids(hlt_tokens, labels, jet_ids):
    torch = require_torch()
    batch_size = int(hlt_tokens.shape[0])
    if labels is None:
        labels = torch.full((batch_size,), -1, dtype=torch.long, device=hlt_tokens.device)
    elif isinstance(labels, torch.Tensor):
        labels = labels.to(device=hlt_tokens.device, dtype=torch.long)
    if jet_ids is None:
        jet_ids = placeholder_jet_ids(batch_size, labels=labels)
    return labels, jet_ids


def build_aggressive_particle_net_reconstructor(
    config: Mapping[str, Any] | AggressiveParticleNetReconstructorConfig | None = None,
) -> AggressiveParticleNetReconstructor:
    return AggressiveParticleNetReconstructor(config)


def build_aggressive_particle_flow_reconstructor(
    config: Mapping[str, Any] | AggressiveParticleFlowReconstructorConfig | None = None,
) -> AggressiveParticleFlowReconstructor:
    return AggressiveParticleFlowReconstructor(config)


def build_aggressive_particle_cnn_reconstructor(
    config: Mapping[str, Any] | AggressiveParticleCnnReconstructorConfig | None = None,
) -> AggressiveParticleCnnReconstructor:
    return AggressiveParticleCnnReconstructor(config)


__all__ = [
    "AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR",
    "AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR",
    "AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR",
    "AggressiveParticleCnnReconstructor",
    "AggressiveParticleCnnReconstructorConfig",
    "AggressiveParticleFlowReconstructor",
    "AggressiveParticleFlowReconstructorConfig",
    "AggressiveParticleNetReconstructor",
    "AggressiveParticleNetReconstructorConfig",
    "build_aggressive_particle_cnn_reconstructor",
    "build_aggressive_particle_flow_reconstructor",
    "build_aggressive_particle_net_reconstructor",
]
