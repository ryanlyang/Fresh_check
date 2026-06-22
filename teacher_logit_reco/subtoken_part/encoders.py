"""Modality encoders and particle-anchor binding for subtoken ParT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import (
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_TRACK,
    SubtokenFeatureConfig,
    SubtokenPartConfig,
)
from .features import (
    SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES,
    SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES,
    SubtokenInputs,
    build_subtoken_inputs,
)

try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_ENCODER_STEP = "subtoken_part_step3_modality_encoders"
SUBTOKEN_PART_ENCODER_CONTRACT = "modality_subtokens_anchor_mask_v1"


@dataclass(frozen=True)
class SubtokenEncoderOutput:
    """Bound modality subtokens ready for the within-particle mixer."""

    subtokens: Any
    anchor: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    modality_embeddings: Any
    modality_type_embeddings: Any
    modality_values: Mapping[str, Any]
    pt_rank_embeddings: Any | None = None

    def component_without_anchor(self) -> Any:
        component = self.modality_embeddings + self.modality_type_embeddings
        if self.pt_rank_embeddings is not None:
            component = component + self.pt_rank_embeddings[:, :, None, :]
        component = require_torch().where(
            self.modality_mask[:, :, :, None],
            component,
            component.new_zeros(component.shape),
        )
        return component

    def recovered_anchor(self) -> Any:
        return self.subtokens - self.component_without_anchor()

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_ENCODER_CONTRACT,
            "subtokens_shape": list(self.subtokens.shape),
            "anchor_shape": list(self.anchor.shape),
            "mask_shape": list(self.mask.shape),
            "modality_mask_shape": list(self.modality_mask.shape),
            "modality_names": list(self.modality_names),
        }


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _zero_masked(values: Any, mask: Any) -> Any:
    torch = require_torch()
    return torch.where(mask[:, :, None], values, torch.zeros_like(values))


def _make_mlp(input_dim: int, embed_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0) -> Any:
    torch = require_torch()
    input_dim = int(input_dim)
    embed_dim = int(embed_dim)
    hidden_dim = int(hidden_dim or max(embed_dim, input_dim))
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(hidden_dim, embed_dim),
    )


class ModalityValueEncoder(_ModuleBase):
    """Shared MLP used by the named modality encoders."""

    modality_name: str = "modality"

    def __init__(self, input_dim: int, embed_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
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


class KinematicsEncoder(ModalityValueEncoder):
    modality_name = SUBTOKEN_MODALITY_KINEMATICS


class IdentityEncoder(ModalityValueEncoder):
    modality_name = SUBTOKEN_MODALITY_IDENTITY


class TrackEncoder(ModalityValueEncoder):
    modality_name = SUBTOKEN_MODALITY_TRACK


class ParticleAnchorEncoder(_ModuleBase):
    """MLP that binds all modality subtokens from the same particle."""

    def __init__(self, input_dim: int, embed_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.network = _make_mlp(self.input_dim, self.embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, values: Any, mask: Any | None = None) -> Any:
        torch = require_torch()
        values = _nan_to_num_torch(values.float())
        if int(values.ndim) != 3:
            raise ValueError("anchor values must have shape [batch, particles, features]")
        if int(values.shape[-1]) != self.input_dim:
            raise ValueError(f"anchor values last dimension must be {self.input_dim}, got {int(values.shape[-1])}")
        anchor = self.network(values)
        if mask is not None:
            mask = mask.bool()
            if tuple(mask.shape) != tuple(values.shape[:2]):
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match values shape {tuple(values.shape[:2])}")
            anchor = torch.where(mask[:, :, None], anchor, torch.zeros_like(anchor))
        return anchor


def subtoken_modality_input_dims(feature_config: SubtokenFeatureConfig) -> dict[str, int]:
    dims = {modality.name: len(modality.raw_indices) for modality in feature_config.modalities}
    if feature_config.include_part_style_derived_features and SUBTOKEN_MODALITY_KINEMATICS in dims:
        dims[SUBTOKEN_MODALITY_KINEMATICS] += len(SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES)
    if feature_config.include_part_style_derived_features and SUBTOKEN_MODALITY_TRACK in dims:
        dims[SUBTOKEN_MODALITY_TRACK] += len(SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES)
    return dims


def subtoken_anchor_input_dim(feature_config: SubtokenFeatureConfig) -> int:
    if feature_config.anchor_source == "raw":
        return int(feature_config.raw_token_dim)
    if feature_config.anchor_source == "part_features":
        return int(feature_config.particle_feature_dim)
    if feature_config.anchor_source == "raw_and_part_features":
        return int(feature_config.raw_token_dim) + int(feature_config.particle_feature_dim)
    raise ValueError(f"unknown anchor_source {feature_config.anchor_source!r}")


def build_particle_anchor_features(inputs: SubtokenInputs) -> Any:
    torch = require_torch()
    source = inputs.feature_config.anchor_source
    if source == "raw":
        return inputs.raw_tokens
    if inputs.derived_kinematics is None:
        raise ValueError(f"anchor_source={source!r} requires derived particle features")
    if source == "part_features":
        return inputs.derived_kinematics.part_features
    if source == "raw_and_part_features":
        return torch.cat([inputs.raw_tokens, inputs.derived_kinematics.part_features], dim=-1)
    raise ValueError(f"unknown anchor_source {source!r}")


def build_pt_rank_features(tokens: Any, mask: Any) -> Any:
    """Build continuous per-particle pT-rank features.

    The leading valid particle receives rank fraction 0.0.  Masked particles
    receive zeros.  A continuous encoding avoids tying this branch to a fixed
    maximum particle count while still exposing the useful leading/subleading
    ordering signal.
    """

    torch = require_torch()
    tokens = _nan_to_num_torch(tokens.float())
    mask = mask.bool()
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(mask.ndim) != 2 or tuple(tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    if int(tokens.shape[-1]) <= 0:
        raise ValueError("tokens must contain a pT feature at index 0")

    batch_size, num_particles = mask.shape
    pt = tokens[:, :, 0]
    masked_pt = torch.where(mask, pt, torch.full_like(pt, torch.finfo(pt.dtype).min))
    order = torch.argsort(masked_pt, dim=1, descending=True)
    positions = torch.arange(num_particles, dtype=torch.long, device=tokens.device).view(1, num_particles)
    positions = positions.expand(batch_size, num_particles)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, positions)

    ranks_float = ranks.to(dtype=tokens.dtype)
    valid_counts = mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype)
    rank_fraction = ranks_float / torch.clamp(valid_counts - 1.0, min=1.0)
    log_rank = torch.log1p(ranks_float) / torch.log1p(torch.clamp(valid_counts, min=1.0))
    rank_fraction = torch.where(mask, rank_fraction, torch.zeros_like(rank_fraction))
    log_rank = torch.where(mask, log_rank, torch.zeros_like(log_rank))
    return torch.stack([rank_fraction, log_rank], dim=-1)


def _normalize_model_config(config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> SubtokenPartConfig:
    if config is None:
        return SubtokenPartConfig(num_classes=2)
    if isinstance(config, SubtokenPartConfig):
        return config
    return SubtokenPartConfig(**dict(config))


class SubtokenParticleEncoder(_ModuleBase):
    """Encode raw HLT particles into bound modality subtokens."""

    def __init__(self, config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.feature_config = self.config.feature_config
        self.embed_dim = int(self.config.embed_dim)
        self.modality_names = self.feature_config.modality_names
        dims = subtoken_modality_input_dims(self.feature_config)
        missing = [
            name
            for name in (SUBTOKEN_MODALITY_KINEMATICS, SUBTOKEN_MODALITY_IDENTITY, SUBTOKEN_MODALITY_TRACK)
            if name not in dims
        ]
        extra = [
            name
            for name in self.modality_names
            if name not in {SUBTOKEN_MODALITY_KINEMATICS, SUBTOKEN_MODALITY_IDENTITY, SUBTOKEN_MODALITY_TRACK}
        ]
        if missing or extra:
            raise ValueError(f"SubtokenParticleEncoder requires default modalities; missing={missing}, extra={extra}")
        self.modality_encoders = torch.nn.ModuleDict(
            {
                SUBTOKEN_MODALITY_KINEMATICS: KinematicsEncoder(
                    dims[SUBTOKEN_MODALITY_KINEMATICS],
                    self.embed_dim,
                    dropout=float(self.config.dropout),
                ),
                SUBTOKEN_MODALITY_IDENTITY: IdentityEncoder(
                    dims[SUBTOKEN_MODALITY_IDENTITY],
                    self.embed_dim,
                    dropout=float(self.config.dropout),
                ),
                SUBTOKEN_MODALITY_TRACK: TrackEncoder(
                    dims[SUBTOKEN_MODALITY_TRACK],
                    self.embed_dim,
                    dropout=float(self.config.dropout),
                ),
            }
        )
        self.anchor_encoder = ParticleAnchorEncoder(
            subtoken_anchor_input_dim(self.feature_config),
            self.embed_dim,
            dropout=float(self.config.dropout),
        )
        self.modality_type_embedding = (
            torch.nn.Embedding(len(self.modality_names), self.embed_dim)
            if self.config.use_modality_type_embeddings
            else None
        )
        self.pt_rank_encoder = (
            _make_mlp(2, self.embed_dim, dropout=float(self.config.dropout))
            if self.config.use_pt_rank_embedding
            else None
        )

    def _build_modality_mask(self, mask: Any, *, num_modalities: int, device: Any) -> Any:
        torch = require_torch()
        mask = mask.bool()
        base_mask = mask[:, :, None].expand(*mask.shape, int(num_modalities))
        probability = float(self.config.modality_dropout)
        if not self.training or probability <= 0.0:
            return base_mask
        keep_probability = 1.0 - probability
        keep = torch.rand(
            (*mask.shape, int(num_modalities)),
            dtype=torch.float32,
            device=device,
        ) < keep_probability
        keep = keep & base_mask

        missing_valid_particle = mask & ~keep.any(dim=2)
        fallback_index = torch.randint(int(num_modalities), mask.shape, dtype=torch.long, device=device)
        fallback = torch.nn.functional.one_hot(fallback_index, num_classes=int(num_modalities)).bool()
        keep = keep | (missing_valid_particle[:, :, None] & fallback)
        return keep & base_mask

    def forward(self, tokens_or_inputs: Any, mask: Any | None = None) -> SubtokenEncoderOutput:
        torch = require_torch()
        if isinstance(tokens_or_inputs, SubtokenInputs):
            inputs = tokens_or_inputs
        else:
            if mask is None:
                raise ValueError("mask is required when passing raw tokens")
            inputs = build_subtoken_inputs(tokens_or_inputs, mask, config=self.feature_config)

        modality_embeddings = torch.stack(
            [self.modality_encoders[name](inputs.modality_values[name], inputs.mask) for name in self.modality_names],
            dim=2,
        )
        modality_mask = self._build_modality_mask(
            inputs.mask,
            num_modalities=len(self.modality_names),
            device=modality_embeddings.device,
        )
        modality_embeddings = torch.where(
            modality_mask[:, :, :, None],
            modality_embeddings,
            torch.zeros_like(modality_embeddings),
        )

        if self.modality_type_embedding is None:
            type_embeddings = torch.zeros_like(modality_embeddings)
        else:
            type_ids = torch.arange(len(self.modality_names), dtype=torch.long, device=modality_embeddings.device)
            type_base = self.modality_type_embedding(type_ids)
            type_embeddings = type_base.view(1, 1, len(self.modality_names), self.embed_dim).expand_as(modality_embeddings)

        if self.config.use_particle_anchor:
            anchor = self.anchor_encoder(build_particle_anchor_features(inputs), inputs.mask)
        else:
            anchor = modality_embeddings.new_zeros((*modality_embeddings.shape[:2], self.embed_dim))

        if self.pt_rank_encoder is None:
            pt_rank_embeddings = None
        else:
            pt_rank_embeddings = self.pt_rank_encoder(build_pt_rank_features(inputs.raw_tokens, inputs.mask))
            pt_rank_embeddings = torch.where(inputs.mask[:, :, None], pt_rank_embeddings, torch.zeros_like(pt_rank_embeddings))

        subtokens = modality_embeddings + type_embeddings + anchor[:, :, None, :]
        if pt_rank_embeddings is not None:
            subtokens = subtokens + pt_rank_embeddings[:, :, None, :]
        subtokens = torch.where(modality_mask[:, :, :, None], subtokens, torch.zeros_like(subtokens))
        type_embeddings = torch.where(modality_mask[:, :, :, None], type_embeddings, torch.zeros_like(type_embeddings))

        return SubtokenEncoderOutput(
            subtokens=subtokens,
            anchor=anchor,
            mask=inputs.mask,
            modality_mask=modality_mask,
            modality_names=tuple(self.modality_names),
            modality_embeddings=modality_embeddings,
            modality_type_embeddings=type_embeddings,
            modality_values=dict(inputs.modality_values),
            pt_rank_embeddings=pt_rank_embeddings,
        )


__all__ = [
    "SUBTOKEN_PART_ENCODER_CONTRACT",
    "SUBTOKEN_PART_ENCODER_STEP",
    "IdentityEncoder",
    "KinematicsEncoder",
    "ParticleAnchorEncoder",
    "SubtokenEncoderOutput",
    "SubtokenParticleEncoder",
    "TrackEncoder",
    "build_particle_anchor_features",
    "build_pt_rank_features",
    "subtoken_anchor_input_dim",
    "subtoken_modality_input_dims",
]
