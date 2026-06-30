"""Modality subtoken embedders for local-compression particles.

Step 4 turns the semantic modality features from ``features.py`` into bound
subtokens:

``[B, N, modality_fields] -> [B, N, modalities, embed_dim]``

Each subtoken gets its own modality value embedding, a modality type embedding,
a particle anchor derived from canonical PF features, and an optional pT-rank
embedding.  This is the binding step that lets fine-grained modality tokens
keep their particle identity before Step 5's local compressor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import (
    LOCAL_COMPRESSION_PART_CONTRACT,
    LocalCompressionPartConfig,
)
from .features import (
    LocalCompressionCanonicalInputs,
    LocalCompressionModalities,
    build_local_compression_canonical_inputs,
    build_local_compression_modalities,
)

try:  # Keep package import cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_COMPRESSION_SUBTOKENS_STEP = "local_compression_part_step4_subtokens"
LOCAL_COMPRESSION_SUBTOKENS_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_bound_modality_subtokens_v1"


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _make_mlp(input_dim: int, output_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0) -> Any:
    torch = require_torch()
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    hidden_dim = int(hidden_dim or max(input_dim, output_dim))
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class LocalCompressionSubtokenOutput:
    """Bound modality subtokens ready for the local compressor."""

    subtokens: Any
    anchor: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    modality_embeddings: Any
    modality_type_embeddings: Any
    modality_values: Mapping[str, Any]
    feature_names_by_modality: Mapping[str, tuple[str, ...]]
    source_names_by_modality: Mapping[str, tuple[str, ...]]
    pt_rank_embeddings: Any | None = None

    def __post_init__(self) -> None:
        torch = require_torch()
        if int(self.subtokens.ndim) != 4:
            raise ValueError("subtokens must have shape [batch, particles, modalities, embed_dim]")
        batch_size, num_particles, num_modalities, embed_dim = tuple(self.subtokens.shape)
        expected_anchor = (batch_size, num_particles, embed_dim)
        expected_mask = (batch_size, num_particles)
        expected_modality_mask = (batch_size, num_particles, num_modalities)
        expected_embeddings = (batch_size, num_particles, num_modalities, embed_dim)
        if tuple(self.anchor.shape) != expected_anchor:
            raise ValueError(f"anchor has shape {tuple(self.anchor.shape)}, expected {expected_anchor}")
        if tuple(self.mask.shape) != expected_mask:
            raise ValueError(f"mask has shape {tuple(self.mask.shape)}, expected {expected_mask}")
        if tuple(self.modality_mask.shape) != expected_modality_mask:
            raise ValueError(
                f"modality_mask has shape {tuple(self.modality_mask.shape)}, expected {expected_modality_mask}"
            )
        if tuple(self.modality_embeddings.shape) != expected_embeddings:
            raise ValueError("modality_embeddings shape must match subtokens")
        if tuple(self.modality_type_embeddings.shape) != expected_embeddings:
            raise ValueError("modality_type_embeddings shape must match subtokens")
        if len(tuple(self.modality_names)) != num_modalities:
            raise ValueError("modality_names length must match subtoken modality dimension")
        if self.pt_rank_embeddings is not None and tuple(self.pt_rank_embeddings.shape) != expected_anchor:
            raise ValueError(f"pt_rank_embeddings has shape {tuple(self.pt_rank_embeddings.shape)}, expected {expected_anchor}")
        if not bool(torch.isfinite(self.subtokens).all()):
            raise ValueError("subtokens contain non-finite values")
        object.__setattr__(self, "modality_names", tuple(self.modality_names))
        object.__setattr__(self, "feature_names_by_modality", dict(self.feature_names_by_modality))
        object.__setattr__(self, "source_names_by_modality", dict(self.source_names_by_modality))

    @property
    def batch_size(self) -> int:
        return int(self.subtokens.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.subtokens.shape[1])

    @property
    def num_modalities(self) -> int:
        return int(self.subtokens.shape[2])

    @property
    def embed_dim(self) -> int:
        return int(self.subtokens.shape[3])

    def component_without_anchor(self) -> Any:
        """Return modality/type/rank components with masked subtokens zeroed."""

        torch = require_torch()
        component = self.modality_embeddings + self.modality_type_embeddings
        if self.pt_rank_embeddings is not None:
            component = component + self.pt_rank_embeddings[:, :, None, :]
        return torch.where(self.modality_mask[:, :, :, None], component, torch.zeros_like(component))

    def recovered_anchor(self) -> Any:
        """Recover the anchor contribution from bound subtokens."""

        return self.subtokens - self.component_without_anchor()

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_SUBTOKENS_CONTRACT,
            "subtokens_shape": list(self.subtokens.shape),
            "anchor_shape": list(self.anchor.shape),
            "mask_shape": list(self.mask.shape),
            "modality_mask_shape": list(self.modality_mask.shape),
            "modality_names": list(self.modality_names),
            "has_pt_rank_embeddings": self.pt_rank_embeddings is not None,
            "active_modality_count": int(self.modality_mask.detach().cpu().sum().item()),
        }


class LocalCompressionModalityValueEncoder(_ModuleBase):
    """Small per-modality MLP."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        *,
        modality_name: str,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.modality_name = str(modality_name)
        self.network = _make_mlp(self.input_dim, self.embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, values: Any, mask: Any | None = None) -> Any:
        torch = require_torch()
        values = _nan_to_num_torch(values.float())
        if int(values.ndim) != 3:
            raise ValueError(f"{self.modality_name} values must have shape [batch, particles, features]")
        if int(values.shape[-1]) != self.input_dim:
            raise ValueError(
                f"{self.modality_name} values last dimension must be {self.input_dim}, got {int(values.shape[-1])}"
            )
        embeddings = self.network(values)
        if mask is not None:
            mask = mask.bool()
            if tuple(mask.shape) != tuple(values.shape[:2]):
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match values shape {tuple(values.shape[:2])}")
            embeddings = torch.where(mask[:, :, None], embeddings, torch.zeros_like(embeddings))
        return embeddings


class LocalCompressionParticleAnchorEncoder(_ModuleBase):
    """Particle anchor derived from canonical PF features F_i."""

    def __init__(self, input_dim: int, embed_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.network = _make_mlp(self.input_dim, self.embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, feature_rows: Any, mask: Any | None = None) -> Any:
        torch = require_torch()
        feature_rows = _nan_to_num_torch(feature_rows.float())
        if int(feature_rows.ndim) != 3:
            raise ValueError("feature_rows must have shape [batch, particles, PF_FEATURE_NAMES]")
        if int(feature_rows.shape[-1]) != self.input_dim:
            raise ValueError(f"feature_rows last dimension must be {self.input_dim}, got {int(feature_rows.shape[-1])}")
        anchor = self.network(feature_rows)
        if mask is not None:
            mask = mask.bool()
            if tuple(mask.shape) != tuple(feature_rows.shape[:2]):
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match feature_rows shape {tuple(feature_rows.shape[:2])}")
            anchor = torch.where(mask[:, :, None], anchor, torch.zeros_like(anchor))
        return anchor


class LocalCompressionPtRankEncoder(_ModuleBase):
    """Embed continuous pT-rank features already present in the geometry modality."""

    def __init__(self, embed_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.network = _make_mlp(2, self.embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, rank_features: Any, mask: Any | None = None) -> Any:
        torch = require_torch()
        rank_features = _nan_to_num_torch(rank_features.float())
        if int(rank_features.ndim) != 3 or int(rank_features.shape[-1]) != 2:
            raise ValueError("rank_features must have shape [batch, particles, 2]")
        embeddings = self.network(rank_features)
        if mask is not None:
            mask = mask.bool()
            if tuple(mask.shape) != tuple(rank_features.shape[:2]):
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match rank_features shape {tuple(rank_features.shape[:2])}")
            embeddings = torch.where(mask[:, :, None], embeddings, torch.zeros_like(embeddings))
        return embeddings


def _normalize_model_config(config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> LocalCompressionPartConfig:
    if config is None:
        return LocalCompressionPartConfig()
    if isinstance(config, LocalCompressionPartConfig):
        return config
    return LocalCompressionPartConfig(**dict(config))


def modality_input_dims(modalities: LocalCompressionModalities) -> dict[str, int]:
    """Return input width for each configured modality."""

    return {
        name: int(modalities.values_by_modality[name].shape[-1])
        for name in modalities.modality_names
    }


def build_pt_rank_features_from_modalities(modalities: LocalCompressionModalities) -> Any:
    """Extract ``pt_rank`` and ``log_pt_rank`` from the geometry modality."""

    names = modalities.feature_names_by_modality.get("geometry")
    if names is None:
        raise ValueError("geometry modality is required for pT-rank features")
    if "pt_rank" not in names or "log_pt_rank" not in names:
        raise ValueError("geometry modality must include pt_rank and log_pt_rank")
    values = modalities.values_by_modality["geometry"]
    rank = values[:, :, names.index("pt_rank")]
    log_rank = values[:, :, names.index("log_pt_rank")]
    return require_torch().stack([rank, log_rank], dim=-1)


class LocalCompressionSubtokenEncoder(_ModuleBase):
    """Encode semantic modality features into particle-bound subtokens."""

    def __init__(self, config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.feature_config = self.config.feature_config
        self.embed_dim = int(self.config.embed_dim)
        self.modality_names = self.feature_config.modality_names
        self._input_dims = {
            spec.name: spec.field_count
            for spec in self.feature_config.modalities
        }
        hidden_dim = int(max(self.embed_dim, round(self.embed_dim * float(self.config.mlp_ratio))))
        self.modality_encoders = torch.nn.ModuleDict(
            {
                name: LocalCompressionModalityValueEncoder(
                    input_dim,
                    self.embed_dim,
                    modality_name=name,
                    hidden_dim=hidden_dim,
                    dropout=float(self.config.dropout),
                )
                for name, input_dim in self._input_dims.items()
            }
        )
        self.anchor_encoder = (
            LocalCompressionParticleAnchorEncoder(
                len(self.feature_config.canonical_feature_names),
                self.embed_dim,
                hidden_dim=hidden_dim,
                dropout=float(self.config.dropout),
            )
            if self.config.use_particle_anchor
            else None
        )
        self.modality_type_embedding = (
            torch.nn.Embedding(len(self.modality_names), self.embed_dim)
            if self.config.use_modality_type_embeddings
            else None
        )
        self.pt_rank_encoder = (
            LocalCompressionPtRankEncoder(self.embed_dim, hidden_dim=hidden_dim, dropout=float(self.config.dropout))
            if self.config.use_pt_rank_embedding
            else None
        )

    @property
    def input_dims(self) -> dict[str, int]:
        return dict(self._input_dims)

    def _type_embeddings(self, modality_embeddings: Any) -> Any:
        torch = require_torch()
        if self.modality_type_embedding is None:
            return torch.zeros_like(modality_embeddings)
        type_ids = torch.arange(len(self.modality_names), dtype=torch.long, device=modality_embeddings.device)
        base = self.modality_type_embedding(type_ids)
        return base.view(1, 1, len(self.modality_names), self.embed_dim).expand_as(modality_embeddings)

    def forward(
        self,
        canonical: LocalCompressionCanonicalInputs,
        modalities: LocalCompressionModalities | None = None,
        *,
        modality_mask_override: Any | None = None,
    ) -> LocalCompressionSubtokenOutput:
        torch = require_torch()
        if modalities is None:
            modalities = build_local_compression_modalities(canonical, config=self.feature_config)
        if tuple(modalities.modality_names) != tuple(self.modality_names):
            raise ValueError("modalities are not in the encoder's configured modality order")
        if tuple(modalities.particle_mask.shape) != tuple(canonical.particle_mask.shape):
            raise ValueError("modalities and canonical inputs have incompatible particle masks")
        if not bool(torch.equal(modalities.particle_mask, canonical.particle_mask)):
            raise ValueError("modalities particle mask does not match canonical particle mask")

        modality_embeddings = torch.stack(
            [
                self.modality_encoders[name](modalities.values_by_modality[name], modalities.particle_mask)
                for name in self.modality_names
            ],
            dim=2,
        )
        modality_mask = modalities.modality_mask.to(device=modality_embeddings.device, dtype=torch.bool)
        if modality_mask_override is not None:
            override = torch.as_tensor(modality_mask_override, dtype=torch.bool, device=modality_embeddings.device)
            if tuple(override.shape) != tuple(modality_mask.shape):
                raise ValueError(
                    f"modality_mask_override shape {tuple(override.shape)} does not match {tuple(modality_mask.shape)}"
                )
            modality_mask = modality_mask & override
        modality_mask = modality_mask & modalities.particle_mask[:, :, None].to(device=modality_embeddings.device)
        modality_embeddings = torch.where(
            modality_mask[:, :, :, None],
            modality_embeddings,
            torch.zeros_like(modality_embeddings),
        )

        type_embeddings = self._type_embeddings(modality_embeddings)
        type_embeddings = torch.where(modality_mask[:, :, :, None], type_embeddings, torch.zeros_like(type_embeddings))

        if self.anchor_encoder is None:
            anchor = modality_embeddings.new_zeros((*modality_embeddings.shape[:2], self.embed_dim))
        else:
            anchor = self.anchor_encoder(canonical.feature_rows(), canonical.particle_mask)

        if self.pt_rank_encoder is None:
            pt_rank_embeddings = None
        else:
            pt_rank_embeddings = self.pt_rank_encoder(build_pt_rank_features_from_modalities(modalities), modalities.particle_mask)

        subtokens = modality_embeddings + type_embeddings + anchor[:, :, None, :]
        if pt_rank_embeddings is not None:
            subtokens = subtokens + pt_rank_embeddings[:, :, None, :]
        subtokens = torch.where(modality_mask[:, :, :, None], subtokens, torch.zeros_like(subtokens))
        return LocalCompressionSubtokenOutput(
            subtokens=subtokens,
            anchor=anchor,
            mask=canonical.particle_mask,
            modality_mask=modality_mask,
            modality_names=tuple(self.modality_names),
            modality_embeddings=modality_embeddings,
            modality_type_embeddings=type_embeddings,
            modality_values=dict(modalities.values_by_modality),
            feature_names_by_modality=dict(modalities.feature_names_by_modality),
            source_names_by_modality=dict(modalities.source_names_by_modality),
            pt_rank_embeddings=pt_rank_embeddings,
        )

    def forward_from_tokens(
        self,
        tokens: Any,
        mask: Any,
        *,
        weights: Any | None = None,
        max_constits: int | None = 128,
        weight_threshold: float = 0.0,
        modality_mask_override: Any | None = None,
    ) -> tuple[LocalCompressionCanonicalInputs, LocalCompressionModalities, LocalCompressionSubtokenOutput]:
        """Convenience wrapper for tests and later smoke scripts."""

        canonical = build_local_compression_canonical_inputs(
            tokens,
            mask,
            weights=weights,
            max_constits=max_constits,
            weight_threshold=weight_threshold,
            config=self.feature_config,
        )
        modalities = build_local_compression_modalities(canonical, config=self.feature_config)
        output = self.forward(canonical, modalities, modality_mask_override=modality_mask_override)
        return canonical, modalities, output


__all__ = [
    "LOCAL_COMPRESSION_SUBTOKENS_CONTRACT",
    "LOCAL_COMPRESSION_SUBTOKENS_STEP",
    "LocalCompressionModalityValueEncoder",
    "LocalCompressionParticleAnchorEncoder",
    "LocalCompressionPtRankEncoder",
    "LocalCompressionSubtokenEncoder",
    "LocalCompressionSubtokenOutput",
    "build_pt_rank_features_from_modalities",
    "modality_input_dims",
]
