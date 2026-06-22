"""Dual-view particle/subtoken fusion for Step 19.

The standard branch intentionally uses the existing Particle Transformer feature
convention (`PF_FEATURE_NAMES`) as a particle-token view, while the subtoken
branch keeps the reliability-gated hierarchy.  Fusion can be late-logit
averaging, pooled embedding concatenation, or CrossViT-style class-token
cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .config import (
    SUBTOKEN_PART_DUAL_FUSION_CONCAT,
    SUBTOKEN_PART_DUAL_FUSION_CROSS_ATTENTION,
    SUBTOKEN_PART_DUAL_FUSION_LATE_LOGITS,
    SubtokenPartConfig,
)
from .features import build_derived_kinematics
from .pairwise import (
    PairwiseBiasConfig,
    PairwiseBiasEncoder,
    PairwiseBiasedAttentionBlock,
    PairwiseFeatureBuilder,
    PairwiseFeatureConfig,
    PairwiseFeatureOutput,
)

try:  # Keep module importable on machines without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_DUAL_VIEW_STEP = "subtoken_part_step19_dual_view_cross_attention"
SUBTOKEN_PART_DUAL_VIEW_CONTRACT = "standard_particle_subtoken_dual_view_fusion_v1"


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _make_transformer_encoder(
    *,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    attention_dropout: float,
) -> Any:
    torch = require_torch()
    layer = torch.nn.TransformerEncoderLayer(
        d_model=int(embed_dim),
        nhead=int(num_heads),
        dim_feedforward=int(4 * embed_dim),
        dropout=float(dropout),
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    layer.self_attn.dropout = float(attention_dropout)
    return torch.nn.TransformerEncoder(layer, num_layers=int(num_layers))


@dataclass(frozen=True)
class StandardParticleBranchOutput:
    """Output from the standard particle-token branch."""

    logits: Any
    cls_embedding: Any
    sequence_tokens: Any
    sequence_mask: Any
    part_features: Any
    part_feature_names: tuple[str, ...]
    pairwise_features: PairwiseFeatureOutput | None
    attention_bias: Any | None

    def summary(self) -> dict[str, Any]:
        payload = {
            "contract": SUBTOKEN_PART_DUAL_VIEW_CONTRACT,
            "branch": "standard_particle_token",
            "logits_shape": list(self.logits.shape),
            "cls_embedding_shape": list(self.cls_embedding.shape),
            "sequence_tokens_shape": list(self.sequence_tokens.shape),
            "sequence_mask_shape": list(self.sequence_mask.shape),
            "part_features_shape": list(self.part_features.shape),
            "part_feature_names": list(self.part_feature_names),
            "use_pairwise_bias": self.attention_bias is not None,
        }
        if self.attention_bias is not None:
            payload["attention_bias_shape"] = list(self.attention_bias.shape)
        if self.pairwise_features is not None:
            payload["pairwise_feature_names"] = list(self.pairwise_features.feature_names)
        return payload

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        valid_counts = self.sequence_mask[:, 1:].sum(dim=1).to(dtype=self.logits.dtype)
        diagnostics: dict[str, Any] = {
            "standard_valid_particle_count_mean": valid_counts.mean(),
            "standard_logit_abs_mean": self.logits.detach().abs().mean(),
            "standard_cls_abs_mean": self.cls_embedding.detach().abs().mean(),
        }
        if self.attention_bias is not None:
            diagnostics["standard_pairwise_attention_bias_abs_mean"] = self.attention_bias.detach().abs().mean()
        else:
            diagnostics["standard_pairwise_attention_bias_abs_mean"] = torch.zeros(
                (), dtype=self.logits.dtype, device=self.logits.device
            )
        return diagnostics


@dataclass(frozen=True)
class DualViewFusionOutput:
    """Fused Step-19 output plus branch diagnostics."""

    logits: Any
    fusion_mode: str
    fused_embedding: Any
    standard: StandardParticleBranchOutput
    subtoken_logits: Any
    subtoken_cls_embedding: Any
    subtoken_sequence_tokens: Any
    subtoken_sequence_mask: Any
    subtoken_to_standard_attention: Any | None = None
    standard_to_subtoken_attention: Any | None = None

    def summary(self) -> dict[str, Any]:
        payload = {
            "step": SUBTOKEN_PART_DUAL_VIEW_STEP,
            "contract": SUBTOKEN_PART_DUAL_VIEW_CONTRACT,
            "fusion_mode": self.fusion_mode,
            "logits_shape": list(self.logits.shape),
            "fused_embedding_shape": list(self.fused_embedding.shape),
            "subtoken_logits_shape": list(self.subtoken_logits.shape),
            "standard": self.standard.summary(),
        }
        if self.subtoken_to_standard_attention is not None:
            payload["subtoken_to_standard_attention_shape"] = list(self.subtoken_to_standard_attention.shape)
        if self.standard_to_subtoken_attention is not None:
            payload["standard_to_subtoken_attention_shape"] = list(self.standard_to_subtoken_attention.shape)
        return payload

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        logit_delta = self.subtoken_logits.detach() - self.standard.logits.detach()
        diagnostics = {
            "dual_fusion_logit_abs_mean": self.logits.detach().abs().mean(),
            "dual_subtoken_logit_abs_mean": self.subtoken_logits.detach().abs().mean(),
            "dual_standard_logit_abs_mean": self.standard.logits.detach().abs().mean(),
            "dual_branch_logit_delta_abs_mean": logit_delta.abs().mean(),
            "dual_fused_embedding_abs_mean": self.fused_embedding.detach().abs().mean(),
        }
        diagnostics.update(self.standard.diagnostics())
        if self.subtoken_to_standard_attention is not None:
            diagnostics["dual_subtoken_to_standard_attention_mean"] = (
                _nan_to_num_torch(self.subtoken_to_standard_attention.detach()).mean()
            )
        else:
            diagnostics["dual_subtoken_to_standard_attention_mean"] = torch.zeros(
                (), dtype=self.logits.dtype, device=self.logits.device
            )
        if self.standard_to_subtoken_attention is not None:
            diagnostics["dual_standard_to_subtoken_attention_mean"] = (
                _nan_to_num_torch(self.standard_to_subtoken_attention.detach()).mean()
            )
        else:
            diagnostics["dual_standard_to_subtoken_attention_mean"] = torch.zeros(
                (), dtype=self.logits.dtype, device=self.logits.device
            )
        return diagnostics


class StandardParticleTokenBranch(_ModuleBase):
    """ParT-style particle-token branch using canonical PF features."""

    def __init__(self, config: SubtokenPartConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.embed_dim = int(config.embed_dim)
        self.input_norm = torch.nn.LayerNorm(len(PF_FEATURE_NAMES))
        self.input_projection = torch.nn.Linear(len(PF_FEATURE_NAMES), self.embed_dim)
        self.cls_token = torch.nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.input_dropout = torch.nn.Dropout(float(config.dropout))
        if bool(config.standard_branch_use_pairwise_bias):
            self.pairwise_feature_builder = PairwiseFeatureBuilder(
                PairwiseFeatureConfig(
                    raw_feature_dim=int(config.feature_config.raw_token_dim),
                    include_cls_token=True,
                )
            )
            self.pairwise_bias_encoder = PairwiseBiasEncoder(
                PairwiseBiasConfig(
                    num_heads=int(config.global_heads),
                    hidden_dim=max(32, int(self.embed_dim // 2)),
                    dropout=float(config.dropout),
                )
            )
            self.blocks = torch.nn.ModuleList(
                [
                    PairwiseBiasedAttentionBlock(
                        embed_dim=self.embed_dim,
                        num_heads=int(config.global_heads),
                        mlp_ratio=4.0,
                        dropout=float(config.dropout),
                        attention_dropout=float(config.attention_dropout),
                    )
                    for _ in range(int(config.standard_branch_layers))
                ]
            )
            self.encoder = None
        else:
            self.pairwise_feature_builder = None
            self.pairwise_bias_encoder = None
            self.blocks = torch.nn.ModuleList()
            self.encoder = _make_transformer_encoder(
                embed_dim=self.embed_dim,
                num_layers=int(config.standard_branch_layers),
                num_heads=int(config.global_heads),
                dropout=float(config.dropout),
                attention_dropout=float(config.attention_dropout),
            )
        self.output_norm = torch.nn.LayerNorm(self.embed_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.Linear(self.embed_dim, 2 * self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(2 * self.embed_dim, int(config.num_classes)),
        )
        torch.nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def no_weight_decay(self) -> set[str]:
        return {"cls_token"}

    def forward(self, raw_tokens: Any, mask: Any) -> StandardParticleBranchOutput:
        torch = require_torch()
        raw_tokens = raw_tokens.float()
        mask = mask.to(device=raw_tokens.device, dtype=torch.bool)
        derived = build_derived_kinematics(raw_tokens, mask)
        part_features = _nan_to_num_torch(derived.part_features.float())
        particle_tokens = self.input_projection(self.input_norm(part_features))
        particle_tokens = torch.where(mask[:, :, None], _nan_to_num_torch(particle_tokens), torch.zeros_like(particle_tokens))
        batch_size = int(particle_tokens.shape[0])
        cls = self.cls_token.expand(batch_size, 1, self.embed_dim)
        sequence = torch.cat([cls, self.input_dropout(particle_tokens)], dim=1)
        sequence_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device),
                mask,
            ],
            dim=1,
        )
        pairwise_features = None
        attention_bias = None
        if bool(self.config.standard_branch_use_pairwise_bias):
            if self.pairwise_feature_builder is None or self.pairwise_bias_encoder is None:
                raise RuntimeError("standard branch pairwise modules are not initialized")
            pairwise_features = self.pairwise_feature_builder(raw_tokens, mask)
            attention_bias = self.pairwise_bias_encoder(pairwise_features)
            encoded = sequence
            for block in self.blocks:
                encoded = block(encoded, attention_bias, sequence_mask).tokens
        else:
            if self.encoder is None:
                raise RuntimeError("standard branch vanilla encoder is not initialized")
            encoded = self.encoder(sequence, src_key_padding_mask=~sequence_mask)
        cls_embedding = self.output_norm(encoded[:, 0])
        logits = _nan_to_num_torch(self.classifier(cls_embedding))
        return StandardParticleBranchOutput(
            logits=logits,
            cls_embedding=cls_embedding,
            sequence_tokens=encoded,
            sequence_mask=sequence_mask,
            part_features=part_features,
            part_feature_names=tuple(PF_FEATURE_NAMES),
            pairwise_features=pairwise_features,
            attention_bias=attention_bias,
        )


class DualViewFusionModule(_ModuleBase):
    """Fuse standard particle tokens with reliability-gated subtoken tokens."""

    def __init__(self, config: SubtokenPartConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.embed_dim = int(config.embed_dim)
        self.standard_branch = StandardParticleTokenBranch(config)
        self.fusion_classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * self.embed_dim),
            torch.nn.Linear(2 * self.embed_dim, 2 * self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(2 * self.embed_dim, int(config.num_classes)),
        )
        self.subtoken_to_standard = torch.nn.MultiheadAttention(
            self.embed_dim,
            int(config.global_heads),
            dropout=float(config.attention_dropout),
            batch_first=True,
        )
        self.standard_to_subtoken = torch.nn.MultiheadAttention(
            self.embed_dim,
            int(config.global_heads),
            dropout=float(config.attention_dropout),
            batch_first=True,
        )
        self.subtoken_cross_norm = torch.nn.LayerNorm(self.embed_dim)
        self.standard_cross_norm = torch.nn.LayerNorm(self.embed_dim)
        self.cross_dropout = torch.nn.Dropout(float(config.dropout))

    def no_weight_decay(self) -> set[str]:
        return {f"standard_branch.{name}" for name in self.standard_branch.no_weight_decay()}

    def _concat_fusion(self, subtoken_cls_embedding: Any, standard_cls_embedding: Any) -> tuple[Any, Any]:
        fused_embedding = torch_cat([subtoken_cls_embedding, standard_cls_embedding], dim=-1)
        logits = _nan_to_num_torch(self.fusion_classifier(fused_embedding))
        return logits, fused_embedding

    def _cross_attention_fusion(
        self,
        *,
        subtoken_cls_embedding: Any,
        subtoken_sequence_tokens: Any,
        subtoken_sequence_mask: Any,
        standard: StandardParticleBranchOutput,
    ) -> tuple[Any, Any, Any, Any]:
        torch = require_torch()
        subtoken_sequence_mask = subtoken_sequence_mask.bool()
        standard_mask = standard.sequence_mask.bool()
        sub_query = subtoken_cls_embedding[:, None, :]
        standard_query = standard.cls_embedding[:, None, :]
        sub_context, sub_to_standard_attn = self.subtoken_to_standard(
            sub_query,
            standard.sequence_tokens,
            standard.sequence_tokens,
            key_padding_mask=~standard_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        standard_context, standard_to_subtoken_attn = self.standard_to_subtoken(
            standard_query,
            subtoken_sequence_tokens,
            subtoken_sequence_tokens,
            key_padding_mask=~subtoken_sequence_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        fused_subtoken = self.subtoken_cross_norm(
            subtoken_cls_embedding + self.cross_dropout(_nan_to_num_torch(sub_context.squeeze(1)))
        )
        fused_standard = self.standard_cross_norm(
            standard.cls_embedding + self.cross_dropout(_nan_to_num_torch(standard_context.squeeze(1)))
        )
        fused_embedding = torch.cat([fused_subtoken, fused_standard], dim=-1)
        logits = _nan_to_num_torch(self.fusion_classifier(fused_embedding))
        return logits, fused_embedding, sub_to_standard_attn, standard_to_subtoken_attn

    def forward(
        self,
        *,
        raw_tokens: Any,
        raw_mask: Any,
        subtoken_logits: Any,
        subtoken_cls_embedding: Any,
        subtoken_sequence_tokens: Any,
        subtoken_sequence_mask: Any,
    ) -> DualViewFusionOutput:
        torch = require_torch()
        standard = self.standard_branch(raw_tokens, raw_mask)
        fusion_mode = str(self.config.dual_fusion_mode)
        subtoken_to_standard_attention = None
        standard_to_subtoken_attention = None
        if fusion_mode == SUBTOKEN_PART_DUAL_FUSION_LATE_LOGITS:
            logits = _nan_to_num_torch(0.5 * (subtoken_logits + standard.logits))
            fused_embedding = torch.cat([subtoken_cls_embedding, standard.cls_embedding], dim=-1)
        elif fusion_mode == SUBTOKEN_PART_DUAL_FUSION_CONCAT:
            logits, fused_embedding = self._concat_fusion(subtoken_cls_embedding, standard.cls_embedding)
        elif fusion_mode == SUBTOKEN_PART_DUAL_FUSION_CROSS_ATTENTION:
            (
                logits,
                fused_embedding,
                subtoken_to_standard_attention,
                standard_to_subtoken_attention,
            ) = self._cross_attention_fusion(
                subtoken_cls_embedding=subtoken_cls_embedding,
                subtoken_sequence_tokens=subtoken_sequence_tokens,
                subtoken_sequence_mask=subtoken_sequence_mask,
                standard=standard,
            )
        else:  # pragma: no cover - config validation should prevent this.
            raise ValueError(f"Unsupported dual fusion mode {fusion_mode!r}")
        return DualViewFusionOutput(
            logits=logits,
            fusion_mode=fusion_mode,
            fused_embedding=_nan_to_num_torch(fused_embedding),
            standard=standard,
            subtoken_logits=subtoken_logits,
            subtoken_cls_embedding=subtoken_cls_embedding,
            subtoken_sequence_tokens=subtoken_sequence_tokens,
            subtoken_sequence_mask=subtoken_sequence_mask,
            subtoken_to_standard_attention=subtoken_to_standard_attention,
            standard_to_subtoken_attention=standard_to_subtoken_attention,
        )


def torch_cat(values: list[Any], *, dim: int) -> Any:
    """Small indirection to keep torch import lazy for type-checking environments."""

    return require_torch().cat(values, dim=int(dim))


__all__ = [
    "SUBTOKEN_PART_DUAL_VIEW_CONTRACT",
    "SUBTOKEN_PART_DUAL_VIEW_STEP",
    "DualViewFusionModule",
    "DualViewFusionOutput",
    "StandardParticleBranchOutput",
    "StandardParticleTokenBranch",
]
