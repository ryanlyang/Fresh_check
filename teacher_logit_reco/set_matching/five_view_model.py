"""Two-stage five-view particle transformer tagger.

This is the first Step 8 model for the set-matching multi-view branch.  It
defaults to ordinary Transformer attention, with optional Step 9
geometry-aware additive attention bias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM

from teacher_logit_reco.reconstructor_builders import strip_compile_prefix_from_state_dict

from .experiment import VIEW_NAMES
from .five_view_attention import (
    PairwiseAttentionBias,
    PairwiseAttentionBiasConfig,
    TOKEN_KIND_GLOBAL,
    TOKEN_KIND_PARTICLE,
    TOKEN_KIND_VIEW_SUMMARY,
    build_pairwise_geometry_features,
    build_pairwise_relation_type_ids,
    build_pairwise_view_pair_ids,
)
from .five_view_data import FIVE_VIEW_SOURCE_TYPE_IDS


SET_MATCHING_FIVE_VIEW_TAGGER_STEP = "set_matching_multiview_step8_five_view_tagger_model"
FIVE_VIEW_TAGGER_CONTRACT = "plain_two_stage_multiview_transformer"
GEOMETRY_AWARE_FIVE_VIEW_TAGGER_CONTRACT = "geometry_aware_two_stage_multiview_transformer"

if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    import torch as _torch

    _ModuleBase = _torch.nn.Module


def _require_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a bool")


def _positive_int(value: int | None, *, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is required")
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
class FiveViewParticleTransformerConfig:
    """Configuration for the plain five-view tagger.

    The defaults are intentionally moderate.  They are large enough to express
    view-level disagreement, but small enough that smoke tests and first Slurm
    runs should not immediately hit memory limits.
    """

    particle_feature_dim: int = RAW_TOKEN_DIM
    num_classes: int = len(LABEL_NAMES)
    num_views: int = len(VIEW_NAMES)
    num_source_types: int = len(FIVE_VIEW_SOURCE_TYPE_IDS)
    max_view_embeddings: int = 16
    max_source_type_embeddings: int = 4
    embed_dim: int = 128
    stage1_layers: int = 2
    stage1_heads: int = 4
    stage2_layers: int = 4
    stage2_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    classifier_hidden_dim: int | None = None
    use_confidence: bool = True
    use_view_embedding: bool = True
    use_source_embedding: bool = True
    use_view_summaries: bool = True
    use_geometry_attention: bool = False
    geometry_hidden_dim: int = 64
    geometry_dropout: float = 0.0
    geometry_pt_index: int = 0
    geometry_eta_index: int = 1
    geometry_phi_index: int = 2
    output_contract: str = FIVE_VIEW_TAGGER_CONTRACT
    experiment_step: str = SET_MATCHING_FIVE_VIEW_TAGGER_STEP

    def __post_init__(self) -> None:
        particle_feature_dim = _positive_int(self.particle_feature_dim, field_name="particle_feature_dim")
        num_classes = _positive_int(self.num_classes, field_name="num_classes")
        num_views = _positive_int(self.num_views, field_name="num_views")
        num_source_types = _positive_int(self.num_source_types, field_name="num_source_types")
        max_view_embeddings = max(
            _positive_int(self.max_view_embeddings, field_name="max_view_embeddings"),
            num_views,
        )
        max_source_type_embeddings = max(
            _positive_int(self.max_source_type_embeddings, field_name="max_source_type_embeddings"),
            num_source_types,
        )
        embed_dim = _positive_int(self.embed_dim, field_name="embed_dim")
        stage1_layers = _positive_int(self.stage1_layers, field_name="stage1_layers")
        stage1_heads = _positive_int(self.stage1_heads, field_name="stage1_heads")
        stage2_layers = _positive_int(self.stage2_layers, field_name="stage2_layers")
        stage2_heads = _positive_int(self.stage2_heads, field_name="stage2_heads")
        if embed_dim % stage1_heads != 0:
            raise ValueError("embed_dim must be divisible by stage1_heads")
        if embed_dim % stage2_heads != 0:
            raise ValueError("embed_dim must be divisible by stage2_heads")
        mlp_ratio = float(self.mlp_ratio)
        if mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        classifier_hidden_dim = self.classifier_hidden_dim
        if classifier_hidden_dim is not None:
            classifier_hidden_dim = _positive_int(classifier_hidden_dim, field_name="classifier_hidden_dim")
        object.__setattr__(self, "particle_feature_dim", particle_feature_dim)
        object.__setattr__(self, "num_classes", num_classes)
        object.__setattr__(self, "num_views", num_views)
        object.__setattr__(self, "num_source_types", num_source_types)
        object.__setattr__(self, "max_view_embeddings", max_view_embeddings)
        object.__setattr__(self, "max_source_type_embeddings", max_source_type_embeddings)
        object.__setattr__(self, "embed_dim", embed_dim)
        object.__setattr__(self, "stage1_layers", stage1_layers)
        object.__setattr__(self, "stage1_heads", stage1_heads)
        object.__setattr__(self, "stage2_layers", stage2_layers)
        object.__setattr__(self, "stage2_heads", stage2_heads)
        object.__setattr__(self, "mlp_ratio", mlp_ratio)
        object.__setattr__(self, "dropout", _dropout_value(self.dropout, field_name="dropout"))
        object.__setattr__(self, "attention_dropout", _dropout_value(self.attention_dropout, field_name="attention_dropout"))
        object.__setattr__(self, "classifier_hidden_dim", classifier_hidden_dim)
        object.__setattr__(self, "use_confidence", _require_bool(self.use_confidence, field_name="use_confidence"))
        object.__setattr__(self, "use_view_embedding", _require_bool(self.use_view_embedding, field_name="use_view_embedding"))
        object.__setattr__(
            self,
            "use_source_embedding",
            _require_bool(self.use_source_embedding, field_name="use_source_embedding"),
        )
        object.__setattr__(self, "use_view_summaries", _require_bool(self.use_view_summaries, field_name="use_view_summaries"))
        use_geometry_attention = _require_bool(self.use_geometry_attention, field_name="use_geometry_attention")
        object.__setattr__(self, "use_geometry_attention", use_geometry_attention)
        object.__setattr__(self, "geometry_hidden_dim", _positive_int(self.geometry_hidden_dim, field_name="geometry_hidden_dim"))
        object.__setattr__(self, "geometry_dropout", _dropout_value(self.geometry_dropout, field_name="geometry_dropout"))
        for field_name in ("geometry_pt_index", "geometry_eta_index", "geometry_phi_index"):
            index = int(getattr(self, field_name))
            if index < 0 or index >= particle_feature_dim:
                raise ValueError(f"{field_name}={index} is outside particle_feature_dim={particle_feature_dim}")
            object.__setattr__(self, field_name, index)
        object.__setattr__(
            self,
            "output_contract",
            GEOMETRY_AWARE_FIVE_VIEW_TAGGER_CONTRACT if use_geometry_attention else FIVE_VIEW_TAGGER_CONTRACT,
        )

    @property
    def classifier_hidden(self) -> int:
        return int(self.classifier_hidden_dim or (2 * int(self.embed_dim)))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "FiveViewParticleTransformerConfig" | None,
    ) -> "FiveViewParticleTransformerConfig":
        if isinstance(value, cls):
            return value
        payload = dict(value or {})
        payload.pop("output_contract", None)
        payload.pop("experiment_step", None)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_source_type_ids(num_views: int):
    torch = require_torch()
    ids = torch.ones((int(num_views),), dtype=torch.long)
    ids[0] = 0
    return ids


def _make_transformer_encoder(*, embed_dim: int, num_layers: int, num_heads: int, mlp_ratio: float, dropout: float, attention_dropout: float):
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


def _encode_with_optional_attention_bias(encoder, tokens, padding_mask, attention_bias=None):
    """Run a TransformerEncoder with optional per-head additive attention bias."""

    if attention_bias is None:
        return encoder(tokens, src_key_padding_mask=padding_mask)
    batch_size, num_heads, length, other_length = (int(value) for value in attention_bias.shape)
    if length != other_length:
        raise ValueError("attention_bias must be square in the token dimensions")
    if tuple(tokens.shape[:2]) != (batch_size, length):
        raise ValueError("attention_bias shape does not match encoder tokens")
    if tuple(padding_mask.shape) != (batch_size, length):
        raise ValueError("padding_mask shape does not match encoder tokens")
    masked_bias = attention_bias.to(device=tokens.device, dtype=tokens.dtype).masked_fill(
        padding_mask[:, None, None, :],
        -1.0e4,
    )
    flattened_bias = masked_bias.reshape(batch_size * num_heads, length, length)
    return encoder(tokens, mask=flattened_bias)


class FiveViewParticleTransformerTagger(_ModuleBase):
    """Two-stage tagger for HLT plus four reconstructed particle views."""

    def __init__(self, config: Mapping[str, Any] | FiveViewParticleTransformerConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = FiveViewParticleTransformerConfig.from_mapping(config)
        dim = int(self.config.embed_dim)

        self.feature_norm = torch.nn.LayerNorm(int(self.config.particle_feature_dim))
        self.feature_projection = torch.nn.Linear(int(self.config.particle_feature_dim), dim)
        self.confidence_projection = torch.nn.Sequential(
            torch.nn.Linear(1, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
        )
        self.view_embedding = torch.nn.Embedding(int(self.config.max_view_embeddings), dim)
        self.source_embedding = torch.nn.Embedding(int(self.config.max_source_type_embeddings), dim)
        self.view_cls_token = torch.nn.Parameter(torch.empty(1, 1, dim))
        self.global_cls_token = torch.nn.Parameter(torch.empty(1, 1, dim))
        self.view_summary_type_embedding = torch.nn.Parameter(torch.empty(1, 1, dim))
        self.input_dropout = torch.nn.Dropout(float(self.config.dropout))
        self.stage1_encoder = _make_transformer_encoder(
            embed_dim=dim,
            num_layers=int(self.config.stage1_layers),
            num_heads=int(self.config.stage1_heads),
            mlp_ratio=float(self.config.mlp_ratio),
            dropout=float(self.config.dropout),
            attention_dropout=float(self.config.attention_dropout),
        )
        self.stage2_encoder = _make_transformer_encoder(
            embed_dim=dim,
            num_layers=int(self.config.stage2_layers),
            num_heads=int(self.config.stage2_heads),
            mlp_ratio=float(self.config.mlp_ratio),
            dropout=float(self.config.dropout),
            attention_dropout=float(self.config.attention_dropout),
        )
        if bool(self.config.use_geometry_attention):
            self.stage1_attention_bias = PairwiseAttentionBias(
                PairwiseAttentionBiasConfig(
                    num_heads=int(self.config.stage1_heads),
                    hidden_dim=int(self.config.geometry_hidden_dim),
                    max_view_embeddings=int(self.config.max_view_embeddings),
                    dropout=float(self.config.geometry_dropout),
                )
            )
            self.stage2_attention_bias = PairwiseAttentionBias(
                PairwiseAttentionBiasConfig(
                    num_heads=int(self.config.stage2_heads),
                    hidden_dim=int(self.config.geometry_hidden_dim),
                    max_view_embeddings=int(self.config.max_view_embeddings),
                    dropout=float(self.config.geometry_dropout),
                )
            )
        else:
            self.stage1_attention_bias = None
            self.stage2_attention_bias = None
        hidden = int(self.config.classifier_hidden)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(hidden, int(self.config.num_classes)),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch = require_torch()
        torch.nn.init.normal_(self.view_cls_token, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.global_cls_token, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.view_summary_type_embedding, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.view_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.source_embedding.weight, mean=0.0, std=0.02)

    def to_config_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    @property
    def output_contract(self) -> str:
        return self.config.output_contract

    def no_weight_decay(self) -> set[str]:
        return {"view_cls_token", "global_cls_token", "view_summary_type_embedding"}

    def _expand_metadata_ids(self, ids, *, batch_size: int, num_views: int, device, name: str, max_value: int):
        torch = require_torch()
        if ids is None:
            if name == "source_type_ids":
                ids = _default_source_type_ids(num_views).to(device=device)
            else:
                ids = torch.arange(int(num_views), dtype=torch.long, device=device)
        elif not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        else:
            ids = ids.to(device=device, dtype=torch.long)
        if ids.ndim == 1:
            if int(ids.shape[0]) != int(num_views):
                raise ValueError(f"{name} length {ids.shape[0]} does not match num_views {num_views}")
            ids = ids[None, :].expand(int(batch_size), int(num_views))
        elif ids.ndim == 2:
            if tuple(ids.shape) != (int(batch_size), int(num_views)):
                raise ValueError(f"{name} shape {tuple(ids.shape)} does not match {(batch_size, num_views)}")
        else:
            raise ValueError(f"{name} must be 1D [views] or 2D [batch, views], got {tuple(ids.shape)}")
        if int(ids.numel()) and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(max_value)):
            raise ValueError(f"{name} values must be in [0, {int(max_value)})")
        return ids

    def _validate_forward_inputs(self, view_features, view_masks, view_confidence) -> tuple[int, int, int, int]:
        if view_features.ndim != 4:
            raise ValueError(f"view_features must be 4D [batch, views, tokens, features], got {tuple(view_features.shape)}")
        batch_size, num_views, max_tokens, feature_dim = (int(value) for value in view_features.shape)
        if feature_dim != int(self.config.particle_feature_dim):
            raise ValueError(f"feature dim {feature_dim} does not match config {self.config.particle_feature_dim}")
        if num_views != int(self.config.num_views):
            raise ValueError(f"num_views {num_views} does not match config {self.config.num_views}")
        if tuple(view_masks.shape) != (batch_size, num_views, max_tokens):
            raise ValueError("view_masks shape must match view_features first three dimensions")
        if tuple(view_confidence.shape) != (batch_size, num_views, max_tokens):
            raise ValueError("view_confidence shape must match view_features first three dimensions")
        return batch_size, num_views, max_tokens, feature_dim

    def _build_stage1_attention_bias(self, *, view_features, view_confidence, view_masks, view_ids_tensor):
        torch = require_torch()
        if self.stage1_attention_bias is None:
            return None
        batch_size, num_views, max_tokens, feature_dim = (int(value) for value in view_features.shape)
        flat_features = view_features.reshape(batch_size * num_views, max_tokens, feature_dim)
        flat_confidence = view_confidence.reshape(batch_size * num_views, max_tokens)
        flat_masks = view_masks.reshape(batch_size * num_views, max_tokens)
        flat_view_ids = view_ids_tensor.reshape(batch_size * num_views)
        special_features = torch.zeros(
            (batch_size * num_views, 1, feature_dim),
            dtype=flat_features.dtype,
            device=flat_features.device,
        )
        special_confidence = torch.ones(
            (batch_size * num_views, 1),
            dtype=flat_confidence.dtype,
            device=flat_confidence.device,
        )
        token_features = torch.cat([special_features, flat_features], dim=1)
        token_confidence = torch.cat([special_confidence, flat_confidence], dim=1)
        token_is_particle = torch.cat(
            [
                torch.zeros((batch_size * num_views, 1), dtype=torch.bool, device=flat_masks.device),
                flat_masks,
            ],
            dim=1,
        )
        token_kinds = torch.where(
            token_is_particle,
            torch.full_like(token_is_particle, TOKEN_KIND_PARTICLE, dtype=torch.long),
            torch.full_like(token_is_particle, TOKEN_KIND_VIEW_SUMMARY, dtype=torch.long),
        )
        token_view_ids = flat_view_ids[:, None].expand(batch_size * num_views, max_tokens + 1)
        pair_features = build_pairwise_geometry_features(
            token_features,
            confidence=token_confidence,
            token_is_particle=token_is_particle,
            token_view_ids=token_view_ids,
            pt_index=int(self.config.geometry_pt_index),
            eta_index=int(self.config.geometry_eta_index),
            phi_index=int(self.config.geometry_phi_index),
        )
        relation_ids = build_pairwise_relation_type_ids(token_kinds, token_view_ids)
        view_pair_ids = build_pairwise_view_pair_ids(
            token_view_ids,
            max_view_embeddings=int(self.config.max_view_embeddings),
        )
        return self.stage1_attention_bias(pair_features, relation_ids, view_pair_ids)

    def _build_stage2_attention_bias(
        self,
        *,
        view_features,
        view_confidence,
        view_masks,
        view_ids_tensor,
        include_view_summaries: bool,
    ):
        torch = require_torch()
        if self.stage2_attention_bias is None:
            return None
        batch_size, num_views, max_tokens, feature_dim = (int(value) for value in view_features.shape)
        zero_global_features = torch.zeros((batch_size, 1, feature_dim), dtype=view_features.dtype, device=view_features.device)
        zero_global_confidence = torch.ones((batch_size, 1), dtype=view_confidence.dtype, device=view_confidence.device)
        global_kind = torch.full((batch_size, 1), TOKEN_KIND_GLOBAL, dtype=torch.long, device=view_features.device)
        global_view_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=view_features.device)

        feature_parts = [zero_global_features]
        confidence_parts = [zero_global_confidence]
        kind_parts = [global_kind]
        view_id_parts = [global_view_ids]
        if include_view_summaries:
            feature_parts.append(torch.zeros((batch_size, num_views, feature_dim), dtype=view_features.dtype, device=view_features.device))
            confidence_parts.append(view_masks.any(dim=-1).float())
            kind_parts.append(torch.full((batch_size, num_views), TOKEN_KIND_VIEW_SUMMARY, dtype=torch.long, device=view_features.device))
            view_id_parts.append(view_ids_tensor)

        feature_parts.append(view_features.reshape(batch_size, num_views * max_tokens, feature_dim))
        confidence_parts.append(view_confidence.reshape(batch_size, num_views * max_tokens))
        kind_parts.append(
            torch.full(
                (batch_size, num_views * max_tokens),
                TOKEN_KIND_PARTICLE,
                dtype=torch.long,
                device=view_features.device,
            )
        )
        view_id_parts.append(view_ids_tensor[:, :, None].expand(batch_size, num_views, max_tokens).reshape(batch_size, num_views * max_tokens))

        token_features = torch.cat(feature_parts, dim=1)
        token_confidence = torch.cat(confidence_parts, dim=1)
        token_kinds = torch.cat(kind_parts, dim=1)
        token_view_ids = torch.cat(view_id_parts, dim=1)
        token_is_particle = token_kinds == TOKEN_KIND_PARTICLE
        pair_features = build_pairwise_geometry_features(
            token_features,
            confidence=token_confidence,
            token_is_particle=token_is_particle,
            token_view_ids=token_view_ids,
            pt_index=int(self.config.geometry_pt_index),
            eta_index=int(self.config.geometry_eta_index),
            phi_index=int(self.config.geometry_phi_index),
        )
        relation_ids = build_pairwise_relation_type_ids(token_kinds, token_view_ids)
        view_pair_ids = build_pairwise_view_pair_ids(
            token_view_ids,
            max_view_embeddings=int(self.config.max_view_embeddings),
        )
        return self.stage2_attention_bias(pair_features, relation_ids, view_pair_ids)

    def forward(
        self,
        view_features,
        view_masks,
        view_confidence=None,
        view_ids=None,
        source_type_ids=None,
        *,
        return_diagnostics: bool = False,
    ):
        """Run the tagger.

        Args:
            view_features: Float tensor ``[B, V, N, F]``.
            view_masks: Bool tensor ``[B, V, N]`` where ``True`` means valid.
            view_confidence: Optional float tensor ``[B, V, N]``.
            view_ids: Optional semantic view ids, either ``[V]`` or ``[B, V]``.
            source_type_ids: Optional source ids, either ``[V]`` or ``[B, V]``.
            return_diagnostics: if true, return ``(logits, diagnostics)``.
        """

        torch = require_torch()
        if not isinstance(view_features, torch.Tensor):
            view_features = torch.as_tensor(view_features, dtype=torch.float32)
        view_features = view_features.float()
        device = view_features.device
        if not isinstance(view_masks, torch.Tensor):
            view_masks = torch.as_tensor(view_masks, dtype=torch.bool, device=device)
        else:
            view_masks = view_masks.to(device=device, dtype=torch.bool)
        if view_confidence is None:
            view_confidence = view_masks.float()
        elif not isinstance(view_confidence, torch.Tensor):
            view_confidence = torch.as_tensor(view_confidence, dtype=torch.float32, device=device)
        else:
            view_confidence = view_confidence.to(device=device, dtype=torch.float32)
        view_confidence = torch.where(view_masks, view_confidence.clamp(0.0, 1.0), torch.zeros_like(view_confidence))

        batch_size, num_views, max_tokens, _ = self._validate_forward_inputs(view_features, view_masks, view_confidence)
        view_ids_tensor = self._expand_metadata_ids(
            view_ids,
            batch_size=batch_size,
            num_views=num_views,
            device=device,
            name="view_ids",
            max_value=int(self.config.max_view_embeddings),
        )
        source_type_ids_tensor = self._expand_metadata_ids(
            source_type_ids,
            batch_size=batch_size,
            num_views=num_views,
            device=device,
            name="source_type_ids",
            max_value=int(self.config.max_source_type_embeddings),
        )

        token_embeddings = self.feature_projection(self.feature_norm(view_features))
        if bool(self.config.use_confidence):
            token_embeddings = token_embeddings + self.confidence_projection(view_confidence.unsqueeze(-1))
        if bool(self.config.use_view_embedding):
            token_embeddings = token_embeddings + self.view_embedding(view_ids_tensor)[:, :, None, :]
        if bool(self.config.use_source_embedding):
            token_embeddings = token_embeddings + self.source_embedding(source_type_ids_tensor)[:, :, None, :]
        token_embeddings = self.input_dropout(token_embeddings)

        flat_tokens = token_embeddings.reshape(batch_size * num_views, max_tokens, int(self.config.embed_dim))
        flat_masks = view_masks.reshape(batch_size * num_views, max_tokens)
        flat_view_ids = view_ids_tensor.reshape(batch_size * num_views)
        flat_source_ids = source_type_ids_tensor.reshape(batch_size * num_views)

        cls = self.view_cls_token.expand(batch_size * num_views, 1, int(self.config.embed_dim))
        if bool(self.config.use_view_embedding):
            cls = cls + self.view_embedding(flat_view_ids)[:, None, :]
        if bool(self.config.use_source_embedding):
            cls = cls + self.source_embedding(flat_source_ids)[:, None, :]
        stage1_input = torch.cat([cls, flat_tokens], dim=1)
        stage1_padding = torch.cat(
            [
                torch.zeros((batch_size * num_views, 1), dtype=torch.bool, device=device),
                ~flat_masks,
            ],
            dim=1,
        )
        stage1_attention_bias = self._build_stage1_attention_bias(
            view_features=view_features,
            view_confidence=view_confidence,
            view_masks=view_masks,
            view_ids_tensor=view_ids_tensor,
        )
        stage1_output = _encode_with_optional_attention_bias(
            self.stage1_encoder,
            stage1_input,
            stage1_padding,
            stage1_attention_bias,
        )
        view_summaries = stage1_output[:, 0].reshape(batch_size, num_views, int(self.config.embed_dim))
        encoded_particles = stage1_output[:, 1:].reshape(batch_size, num_views, max_tokens, int(self.config.embed_dim))

        global_cls = self.global_cls_token.expand(batch_size, 1, int(self.config.embed_dim))
        stage2_parts = [global_cls]
        stage2_padding_parts = [torch.zeros((batch_size, 1), dtype=torch.bool, device=device)]
        if bool(self.config.use_view_summaries):
            stage2_parts.append(view_summaries + self.view_summary_type_embedding)
            view_present = view_masks.any(dim=-1)
            stage2_padding_parts.append(~view_present)
        stage2_parts.append(encoded_particles.reshape(batch_size, num_views * max_tokens, int(self.config.embed_dim)))
        stage2_padding_parts.append(~view_masks.reshape(batch_size, num_views * max_tokens))
        stage2_input = torch.cat(stage2_parts, dim=1)
        stage2_padding = torch.cat(stage2_padding_parts, dim=1)
        stage2_attention_bias = self._build_stage2_attention_bias(
            view_features=view_features,
            view_confidence=view_confidence,
            view_masks=view_masks,
            view_ids_tensor=view_ids_tensor,
            include_view_summaries=bool(self.config.use_view_summaries),
        )
        stage2_output = _encode_with_optional_attention_bias(
            self.stage2_encoder,
            stage2_input,
            stage2_padding,
            stage2_attention_bias,
        )
        logits = self.classifier(stage2_output[:, 0])

        if not bool(return_diagnostics):
            return logits
        diagnostics = {
            "contract": self.output_contract,
            "batch_size": int(batch_size),
            "num_views": int(num_views),
            "max_tokens_per_view": int(max_tokens),
            "stage1_sequence_length": int(max_tokens + 1),
            "stage2_sequence_length": int(stage2_input.shape[1]),
            "geometry_attention_enabled": bool(self.config.use_geometry_attention),
            "valid_particle_count_mean": float(view_masks.sum(dim=(1, 2)).float().mean().detach().cpu().item()),
            "view_present_fraction": float(view_masks.any(dim=-1).float().mean().detach().cpu().item()),
        }
        return logits, diagnostics


def build_five_view_tagger(
    config: Mapping[str, Any] | FiveViewParticleTransformerConfig | None = None,
    **kwargs,
) -> FiveViewParticleTransformerTagger:
    """Build a Step 8 five-view tagger."""

    if kwargs:
        payload = FiveViewParticleTransformerConfig.from_mapping(config).to_dict()
        payload.update(kwargs)
        config = payload
    return FiveViewParticleTransformerTagger(config)


def five_view_tagger_checkpoint_payload(
    model: FiveViewParticleTransformerTagger,
    *,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a checkpoint payload for a five-view tagger."""

    if not isinstance(model, FiveViewParticleTransformerTagger):
        raise TypeError("model must be a FiveViewParticleTransformerTagger")
    payload = {
        "experiment_step": SET_MATCHING_FIVE_VIEW_TAGGER_STEP,
        "output_contract": model.output_contract,
        "model_config": model.to_config_dict(),
        "model_state_dict": model.state_dict(),
        "label_names": list(LABEL_NAMES),
        "view_names": list(VIEW_NAMES),
    }
    payload.update(dict(extra_payload or {}))
    return payload


def save_five_view_tagger_checkpoint(
    path: str | Path,
    model: FiveViewParticleTransformerTagger,
    *,
    extra_payload: Mapping[str, Any] | None = None,
) -> Path:
    """Save a five-view tagger checkpoint."""

    torch = require_torch()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(five_view_tagger_checkpoint_payload(model, extra_payload=extra_payload), checkpoint_path)
    return checkpoint_path


def load_five_view_tagger_checkpoint(
    path: str | Path,
    *,
    device="cpu",
    strict: bool = True,
) -> tuple[FiveViewParticleTransformerTagger, dict[str, Any]]:
    """Load a five-view tagger checkpoint."""

    torch = require_torch()
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Five-view tagger checkpoint must be a mapping: {checkpoint_path}")
    if "model_state_dict" not in payload:
        raise KeyError(f"Five-view tagger checkpoint is missing model_state_dict: {checkpoint_path}")
    config = FiveViewParticleTransformerConfig.from_mapping(payload.get("model_config") or payload)
    model = FiveViewParticleTransformerTagger(config)
    model.load_state_dict(strip_compile_prefix_from_state_dict(payload["model_state_dict"]), strict=bool(strict))
    model = model.to(device)
    model.eval()
    return model, dict(payload)


__all__ = [
    "FIVE_VIEW_TAGGER_CONTRACT",
    "GEOMETRY_AWARE_FIVE_VIEW_TAGGER_CONTRACT",
    "SET_MATCHING_FIVE_VIEW_TAGGER_STEP",
    "FiveViewParticleTransformerConfig",
    "FiveViewParticleTransformerTagger",
    "build_five_view_tagger",
    "five_view_tagger_checkpoint_payload",
    "load_five_view_tagger_checkpoint",
    "save_five_view_tagger_checkpoint",
]
