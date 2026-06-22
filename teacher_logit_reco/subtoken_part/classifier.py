"""Global classifier for reliability-gated subtoken particle tokens."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import (
    SUBTOKEN_PART_GATE_CONTEXT_SIGMOID,
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_NONE,
    SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION,
    SubtokenPartConfig,
)
from .context import ParticleContextOutput, ParticleContextTransformer
from .encoders import SubtokenEncoderOutput, SubtokenParticleEncoder
from .gates import ReliabilityGateHead, ReliabilityGateOutput
from .mixer import SubtokenMixerOutput, WithinParticleSubtokenTransformer
from .particle_tokens import ReliabilityAwareParticleOutput, ReliabilityAwareParticleTokenBuilder
from .pairwise import (
    PairwiseBiasConfig,
    PairwiseBiasEncoder,
    PairwiseBiasedAttentionBlock,
    PairwiseFeatureBuilder,
    PairwiseFeatureConfig,
    PairwiseFeatureOutput,
)
from .pooling import SubtokenAttentionPool, SubtokenPoolOutput
from .dual_view import DualViewFusionModule, DualViewFusionOutput

try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_CLASSIFIER_STEP = "subtoken_part_step9_classifier"
SUBTOKEN_PART_CLASSIFIER_CONTRACT = "subtoken_particle_transformer_classifier_v1"
SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP = "subtoken_part_step11_pairwise_classifier"
SUBTOKEN_PART_PAIRWISE_CLASSIFIER_CONTRACT = "pairwise_biased_subtoken_particle_transformer_classifier_v1"
SUBTOKEN_PART_DUAL_VIEW_CLASSIFIER_STEP = "subtoken_part_step19_dual_view_classifier"
SUBTOKEN_PART_DUAL_VIEW_CLASSIFIER_CONTRACT = "dual_view_standard_particle_subtoken_classifier_v1"


def _classifier_step(config: SubtokenPartConfig) -> str:
    if config.variant == SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION:
        return SUBTOKEN_PART_DUAL_VIEW_CLASSIFIER_STEP
    if bool(config.use_pairwise_bias):
        return SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP
    return SUBTOKEN_PART_CLASSIFIER_STEP


def _classifier_contract(config: SubtokenPartConfig) -> str:
    if config.variant == SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION:
        return SUBTOKEN_PART_DUAL_VIEW_CLASSIFIER_CONTRACT
    if bool(config.use_pairwise_bias):
        return SUBTOKEN_PART_PAIRWISE_CLASSIFIER_CONTRACT
    return SUBTOKEN_PART_CLASSIFIER_CONTRACT


@dataclass(frozen=True)
class SubtokenClassifierOutput:
    """Debug-rich output for the subtoken classifier."""

    logits: Any
    cls_embedding: Any
    global_tokens: Any
    global_mask: Any
    pairwise_features: PairwiseFeatureOutput | None
    attention_bias: Any | None
    encoded: SubtokenEncoderOutput
    mixed: SubtokenMixerOutput
    pooled: SubtokenPoolOutput
    context: ParticleContextOutput | None
    gates: ReliabilityGateOutput | None
    particles: ReliabilityAwareParticleOutput
    dual_view: DualViewFusionOutput | None
    config: SubtokenPartConfig

    def summary(self) -> dict[str, Any]:
        payload = {
            "step": _classifier_step(self.config),
            "contract": _classifier_contract(self.config),
            "variant": self.config.variant,
            "gate_mode": self.config.gate_mode,
            "use_pairwise_bias": bool(self.config.use_pairwise_bias),
            "use_dual_view_fusion": self.dual_view is not None,
            "smoke_only_without_pairwise_bias": not bool(self.config.use_pairwise_bias),
            "serious_comparison_ready": bool(self.config.use_pairwise_bias),
            "logits_shape": list(self.logits.shape),
            "cls_embedding_shape": list(self.cls_embedding.shape),
            "global_tokens_shape": list(self.global_tokens.shape),
            "global_mask_shape": list(self.global_mask.shape),
            "particle_tokens_shape": list(self.particles.particle_tokens.shape),
        }
        if self.pairwise_features is not None:
            payload.update(
                {
                    "pairwise_features_shape": list(self.pairwise_features.pair_features.shape),
                    "pairwise_feature_names": list(self.pairwise_features.feature_names),
                }
            )
        if self.attention_bias is not None:
            payload["attention_bias_shape"] = list(self.attention_bias.shape)
        if self.dual_view is not None:
            payload["dual_view"] = self.dual_view.summary()
        return payload

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        particle_mask = self.particles.mask
        batch_size = int(particle_mask.shape[0])
        valid_counts = particle_mask.sum(dim=1).to(dtype=self.logits.dtype)
        diagnostics: dict[str, Any] = {
            "step": _classifier_step(self.config),
            "contract": _classifier_contract(self.config),
            "batch_size": batch_size,
            "max_particles": int(particle_mask.shape[1]),
            "valid_particle_count_mean": valid_counts.mean(),
            "valid_particle_count_min": valid_counts.min(),
            "valid_particle_count_max": valid_counts.max(),
            "logit_abs_mean": self.logits.detach().abs().mean(),
            "use_dual_view_fusion": self.dual_view is not None,
            "smoke_only_without_pairwise_bias": not bool(self.config.use_pairwise_bias),
            "serious_comparison_ready": bool(self.config.use_pairwise_bias),
        }
        if self.attention_bias is not None:
            diagnostics["pairwise_attention_bias_abs_mean"] = self.attention_bias.detach().abs().mean()
            diagnostics["pairwise_attention_bias_abs_max"] = self.attention_bias.detach().abs().amax()
        else:
            diagnostics["pairwise_attention_bias_abs_mean"] = torch.zeros(
                (), dtype=self.logits.dtype, device=self.logits.device
            )
            diagnostics["pairwise_attention_bias_abs_max"] = torch.zeros(
                (), dtype=self.logits.dtype, device=self.logits.device
            )
        diagnostics.update({f"particle_{key}": value for key, value in self.particles.diagnostics().items()})
        if self.gates is not None:
            diagnostics.update({f"gate_{key}": value for key, value in self.gates.diagnostics().items()})
        else:
            diagnostics["gate_mean_gate_entropy"] = torch.zeros((), dtype=self.logits.dtype, device=self.logits.device)
        if self.dual_view is not None:
            diagnostics.update(self.dual_view.diagnostics())
        return diagnostics


def _normalize_model_config(config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> SubtokenPartConfig:
    if config is None:
        return SubtokenPartConfig(num_classes=2)
    if isinstance(config, SubtokenPartConfig):
        return config
    return SubtokenPartConfig(**dict(config))


def _config_payload(config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, SubtokenPartConfig):
        return {field.name: getattr(config, field.name) for field in fields(SubtokenPartConfig)}
    return dict(config)


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


def _coerce_tokens_and_mask(tokens_or_batch: Any, mask: Any | None = None) -> tuple[Any, Any]:
    torch = require_torch()
    if isinstance(tokens_or_batch, Mapping):
        tokens = (
            tokens_or_batch.get("tokens")
            if "tokens" in tokens_or_batch
            else tokens_or_batch.get("hlt_tokens", tokens_or_batch.get("raw_tokens"))
        )
        if tokens is None:
            raise ValueError("input mapping must contain 'tokens', 'hlt_tokens', or 'raw_tokens'")
        if mask is None:
            mask = tokens_or_batch.get("mask", tokens_or_batch.get("hlt_mask"))
    else:
        tokens = tokens_or_batch
    if mask is None:
        raise ValueError("mask is required")
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
    else:
        tokens = tokens.float()
    device = tokens.device
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    return tokens, mask


class SubtokenParticleTransformerClassifier(_ModuleBase):
    """HLT-only classifier using reliability-gated subtokens and global attention."""

    def __init__(self, config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.encoder = SubtokenParticleEncoder(self.config)
        self.local_mixer = WithinParticleSubtokenTransformer(self.config)
        self.local_pool = SubtokenAttentionPool(self.config)
        self.context_transformer = ParticleContextTransformer(self.config)
        self.gate_head = ReliabilityGateHead(self.config)
        self.particle_builder = ReliabilityAwareParticleTokenBuilder(self.config)
        self.global_cls_token = torch.nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.input_dropout = torch.nn.Dropout(float(self.config.dropout))
        if bool(self.config.use_pairwise_bias):
            self.pairwise_feature_builder = PairwiseFeatureBuilder(
                PairwiseFeatureConfig(
                    raw_feature_dim=int(self.config.feature_config.raw_token_dim),
                    include_cls_token=True,
                )
            )
            self.pairwise_bias_encoder = PairwiseBiasEncoder(
                PairwiseBiasConfig(
                    num_heads=int(self.config.global_heads),
                    hidden_dim=max(32, int(self.embed_dim // 2)),
                    dropout=float(self.config.dropout),
                )
            )
            self.global_pairwise_blocks = torch.nn.ModuleList(
                [
                    PairwiseBiasedAttentionBlock(
                        embed_dim=self.embed_dim,
                        num_heads=int(self.config.global_heads),
                        mlp_ratio=4.0,
                        dropout=float(self.config.dropout),
                        attention_dropout=float(self.config.attention_dropout),
                    )
                    for _ in range(int(self.config.global_layers))
                ]
            )
            self.global_encoder = None
        else:
            self.pairwise_feature_builder = None
            self.pairwise_bias_encoder = None
            self.global_pairwise_blocks = torch.nn.ModuleList()
            self.global_encoder = _make_transformer_encoder(
                embed_dim=self.embed_dim,
                num_layers=int(self.config.global_layers),
                num_heads=int(self.config.global_heads),
                dropout=float(self.config.dropout),
                attention_dropout=float(self.config.attention_dropout),
            )
        self.global_norm = torch.nn.LayerNorm(self.embed_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.Linear(self.embed_dim, 2 * self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(2 * self.embed_dim, int(self.config.num_classes)),
        )
        self.dual_view_fusion = (
            DualViewFusionModule(self.config)
            if self.config.variant == SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch = require_torch()
        torch.nn.init.normal_(self.global_cls_token, mean=0.0, std=0.02)

    def no_weight_decay(self) -> set[str]:
        names = {"global_cls_token"}
        if self.dual_view_fusion is not None:
            names.update({f"dual_view_fusion.{name}" for name in self.dual_view_fusion.no_weight_decay()})
        return names

    def to_config_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    @property
    def output_contract(self) -> str:
        return _classifier_contract(self.config)

    def _needs_context(self) -> bool:
        return self.config.gate_mode in {SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX, SUBTOKEN_PART_GATE_CONTEXT_SIGMOID}

    def _build_global_logits(
        self,
        particle_tokens: Any,
        mask: Any,
        raw_tokens: Any,
    ) -> tuple[Any, Any, Any, Any, PairwiseFeatureOutput | None, Any | None]:
        torch = require_torch()
        particle_tokens = _nan_to_num_torch(particle_tokens.float())
        mask = mask.bool()
        batch_size = int(particle_tokens.shape[0])
        cls = self.global_cls_token.expand(batch_size, 1, self.embed_dim)
        sequence = torch.cat([cls, self.input_dropout(particle_tokens)], dim=1)
        global_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=particle_tokens.device),
                mask.to(device=particle_tokens.device, dtype=torch.bool),
            ],
            dim=1,
        )
        pairwise_features = None
        attention_bias = None
        if bool(self.config.use_pairwise_bias):
            if self.pairwise_feature_builder is None or self.pairwise_bias_encoder is None:
                raise RuntimeError("Pairwise modules are not initialized despite use_pairwise_bias=True")
            pairwise_features = self.pairwise_feature_builder(raw_tokens, mask)
            if tuple(pairwise_features.token_mask.shape) != tuple(global_mask.shape):
                raise ValueError(
                    f"pairwise token mask shape {tuple(pairwise_features.token_mask.shape)} "
                    f"does not match global mask shape {tuple(global_mask.shape)}"
                )
            if not bool(torch.equal(pairwise_features.token_mask.to(device=global_mask.device), global_mask)):
                raise ValueError("pairwise token mask does not match global token mask")
            attention_bias = self.pairwise_bias_encoder(pairwise_features)
            encoded = sequence
            for block in self.global_pairwise_blocks:
                encoded = block(encoded, attention_bias, global_mask).tokens
        else:
            if self.global_encoder is None:
                raise RuntimeError("Vanilla global encoder is not initialized despite use_pairwise_bias=False")
            encoded = self.global_encoder(sequence, src_key_padding_mask=~global_mask)
        cls_embedding = self.global_norm(encoded[:, 0])
        logits = _nan_to_num_torch(self.classifier(cls_embedding))
        return logits, cls_embedding, encoded, global_mask, pairwise_features, attention_bias

    def forward_outputs(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        modality_mask_override: Any | None = None,
    ) -> SubtokenClassifierOutput:
        tokens, mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        encoded = self.encoder(tokens, mask, modality_mask_override=modality_mask_override)
        mixed = self.local_mixer(encoded)
        pooled = self.local_pool(mixed)
        contexted = self.context_transformer(pooled) if self._needs_context() else None
        if self.config.gate_mode == SUBTOKEN_PART_GATE_NONE:
            gate_output = None
        else:
            gate_output = self.gate_head(mixed, pooled, contexted, encoded)
        particles = self.particle_builder(mixed, pooled, gate_output, encoded)
        logits, cls_embedding, global_tokens, global_mask, pairwise_features, attention_bias = self._build_global_logits(
            particles.particle_tokens,
            particles.mask,
            tokens,
        )
        dual_view = None
        if self.dual_view_fusion is not None:
            dual_view = self.dual_view_fusion(
                raw_tokens=tokens,
                raw_mask=mask,
                subtoken_logits=logits,
                subtoken_cls_embedding=cls_embedding,
                subtoken_sequence_tokens=global_tokens,
                subtoken_sequence_mask=global_mask,
            )
            logits = dual_view.logits
            cls_embedding = dual_view.fused_embedding
        return SubtokenClassifierOutput(
            logits=logits,
            cls_embedding=cls_embedding,
            global_tokens=global_tokens,
            global_mask=global_mask,
            pairwise_features=pairwise_features,
            attention_bias=attention_bias,
            encoded=encoded,
            mixed=mixed,
            pooled=pooled,
            context=contexted,
            gates=gate_output,
            particles=particles,
            dual_view=dual_view,
            config=self.config,
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
        modality_mask_override: Any | None = None,
    ):
        output = self.forward_outputs(tokens_or_batch, mask, modality_mask_override=modality_mask_override)
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.logits, output.diagnostics()
        return output.logits


def build_subtoken_particle_transformer_classifier(
    config: SubtokenPartConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SubtokenParticleTransformerClassifier:
    """Build the HLT-only subtoken classifier."""

    if kwargs:
        payload = _config_payload(config)
        payload.update(kwargs)
        config = payload
    return SubtokenParticleTransformerClassifier(config)


__all__ = [
    "SUBTOKEN_PART_CLASSIFIER_CONTRACT",
    "SUBTOKEN_PART_PAIRWISE_CLASSIFIER_CONTRACT",
    "SUBTOKEN_PART_DUAL_VIEW_CLASSIFIER_CONTRACT",
    "SUBTOKEN_PART_CLASSIFIER_STEP",
    "SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP",
    "SUBTOKEN_PART_DUAL_VIEW_CLASSIFIER_STEP",
    "SubtokenClassifierOutput",
    "SubtokenParticleTransformerClassifier",
    "build_subtoken_particle_transformer_classifier",
]
