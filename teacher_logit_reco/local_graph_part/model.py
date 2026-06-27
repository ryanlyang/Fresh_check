"""Classifier wrapper for local-graph Particle Transformer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, build_hlt_classifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from teacher_logit_reco.subtoken_part.pairwise import (
    PairwiseBiasConfig,
    PairwiseBiasEncoder,
    PairwiseBiasedAttentionBlock,
    PairwiseFeatureBuilder,
    PairwiseFeatureConfig,
    PairwiseFeatureOutput,
)

from .knn import LocalKnnConfig, LocalKnnOutput, build_local_knn_graph
from .local_blocks import (
    EdgeConvLocalAdapter,
    EdgeConvLocalAdapterConfig,
    EdgeConvLocalAdapterOutput,
    PointAttentionLocalAdapter,
    PointAttentionLocalAdapterConfig,
    PointAttentionLocalAdapterOutput,
)

try:  # Keep imports cheap when PyTorch is not available.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_GRAPH_PART_CLASSIFIER_STEP = "local_graph_part_step5_pairwise_transformer_prototype"
LOCAL_GRAPH_PART_CLASSIFIER_CONTRACT = "local_graph_pairwise_transformer_prototype_v1"
LOCAL_GRAPH_AUGMENTED_PART_STEP = "local_graph_part_step5_augmented_weaver_part"
LOCAL_GRAPH_AUGMENTED_PART_CONTRACT = "local_graph_augmented_weaver_part_v1"
LOCAL_GRAPH_HLT_PART_BASELINE_STEP = "local_graph_part_step6_hlt_part_baseline"
LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT = "local_graph_hlt_part_baseline_raw_token_v1"

LOCAL_GRAPH_ADAPTER_NONE = "none"
LOCAL_GRAPH_ADAPTER_EDGECONV = "edgeconv"
LOCAL_GRAPH_ADAPTER_POINT_ATTENTION = "point_attention"
LOCAL_GRAPH_ADAPTERS = (
    LOCAL_GRAPH_ADAPTER_NONE,
    LOCAL_GRAPH_ADAPTER_EDGECONV,
    LOCAL_GRAPH_ADAPTER_POINT_ATTENTION,
)

LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL = "local_graph_adapter_disabled_control"
LOCAL_GRAPH_MODEL_VARIANT_EDGECONV = "local_edgeconv_adapter"
LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION = "local_point_attention_adapter"
LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE = "hlt_part_baseline"
LOCAL_GRAPH_MODEL_VARIANTS = (
    LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
    LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
    LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION,
)
LOCAL_GRAPH_COMPARISON_VARIANTS = (
    LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
    LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
    LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION,
)


def normalize_local_graph_adapter(value: str) -> str:
    clean = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": LOCAL_GRAPH_ADAPTER_NONE,
        "no_adapter": LOCAL_GRAPH_ADAPTER_NONE,
        "none": LOCAL_GRAPH_ADAPTER_NONE,
        "adapter_disabled": LOCAL_GRAPH_ADAPTER_NONE,
        "adapter_disabled_control": LOCAL_GRAPH_ADAPTER_NONE,
        "local_graph_no_adapter": LOCAL_GRAPH_ADAPTER_NONE,
        "local_graph_adapter_disabled_control": LOCAL_GRAPH_ADAPTER_NONE,
        "edge": LOCAL_GRAPH_ADAPTER_EDGECONV,
        "edge_conv": LOCAL_GRAPH_ADAPTER_EDGECONV,
        "edgeconv": LOCAL_GRAPH_ADAPTER_EDGECONV,
        "local_edgeconv_adapter": LOCAL_GRAPH_ADAPTER_EDGECONV,
        "point": LOCAL_GRAPH_ADAPTER_POINT_ATTENTION,
        "point_attention": LOCAL_GRAPH_ADAPTER_POINT_ATTENTION,
        "pointattention": LOCAL_GRAPH_ADAPTER_POINT_ATTENTION,
        "local_point_attention_adapter": LOCAL_GRAPH_ADAPTER_POINT_ATTENTION,
    }
    if clean not in aliases:
        if clean in {"baseline", "hlt_baseline", "hlt_part_baseline", "part_baseline"}:
            raise ValueError(
                f"{value!r} is the real HLT ParT baseline, not a local adapter. "
                "Use the baseline runner/build_hlt_classifier for hlt_part_baseline."
            )
        raise ValueError(f"local_adapter must be one of {LOCAL_GRAPH_ADAPTERS}, got {value!r}")
    return aliases[clean]


def local_graph_variant_for_adapter(adapter: str) -> str:
    adapter = normalize_local_graph_adapter(adapter)
    if adapter == LOCAL_GRAPH_ADAPTER_EDGECONV:
        return LOCAL_GRAPH_MODEL_VARIANT_EDGECONV
    if adapter == LOCAL_GRAPH_ADAPTER_POINT_ATTENTION:
        return LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION
    return LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL


def normalize_local_graph_comparison_variant(value: str) -> str:
    """Resolve Step 6 comparison variant names without conflating the baseline."""

    clean = str(value).strip().lower().replace("-", "_")
    aliases = {
        "baseline": LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
        "hlt_baseline": LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
        "hlt_part": LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
        "hlt_part_baseline": LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
        "part_baseline": LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
        LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE: LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
        "edge": LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
        "edgeconv": LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
        "edge_conv": LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
        LOCAL_GRAPH_MODEL_VARIANT_EDGECONV: LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
        "point": LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION,
        "point_attention": LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION,
        "pointattention": LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION,
        LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION: LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION,
        "none": LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
        "no_adapter": LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
        "adapter_disabled": LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
        "adapter_disabled_control": LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
        "local_graph_no_adapter": LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
        LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL: LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
    }
    if clean not in aliases:
        valid = (LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,) + LOCAL_GRAPH_MODEL_VARIANTS
        raise ValueError(f"local graph comparison variant must be one of {valid}, got {value!r}")
    return aliases[clean]


@dataclass(frozen=True)
class LocalGraphParticleTransformerConfig:
    """Configuration for the Step 5 prototype pairwise-transformer wrapper.

    This does not instantiate the reference Weaver ParticleTransformer and is
    kept only as a fast local-graph sandbox.
    """

    num_classes: int = 2
    raw_feature_dim: int = RAW_TOKEN_DIM
    embed_dim: int = 128
    global_layers: int = 4
    global_heads: int = 8
    local_adapter: str = LOCAL_GRAPH_ADAPTER_POINT_ATTENTION
    k: int = 16
    local_heads: int = 4
    local_hidden_dim: int | None = None
    use_pairwise_bias: bool = True
    dropout: float = 0.05
    attention_dropout: float = 0.05
    mlp_ratio: float = 4.0
    residual_gamma_init: float = 0.0
    pairwise_hidden_dim: int = 64
    mask_value: float = -1.0e4

    def __post_init__(self) -> None:
        for field_name in ("num_classes", "raw_feature_dim", "embed_dim", "global_layers", "global_heads", "k"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        local_heads = int(self.local_heads)
        if local_heads <= 0:
            raise ValueError("local_heads must be positive")
        object.__setattr__(self, "local_heads", local_heads)
        if int(self.embed_dim) % int(self.global_heads) != 0:
            raise ValueError("embed_dim must be divisible by global_heads")
        if normalize_local_graph_adapter(self.local_adapter) == LOCAL_GRAPH_ADAPTER_POINT_ATTENTION:
            if int(self.embed_dim) % int(self.local_heads) != 0:
                raise ValueError("embed_dim must be divisible by local_heads for point attention")
        hidden_dim = self.local_hidden_dim
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("local_hidden_dim must be positive when provided")
        object.__setattr__(self, "local_hidden_dim", hidden_dim)
        for field_name in ("dropout", "attention_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            object.__setattr__(self, field_name, value)
        mlp_ratio = float(self.mlp_ratio)
        if mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        pairwise_hidden_dim = int(self.pairwise_hidden_dim)
        if pairwise_hidden_dim <= 0:
            raise ValueError("pairwise_hidden_dim must be positive")
        object.__setattr__(self, "local_adapter", normalize_local_graph_adapter(self.local_adapter))
        object.__setattr__(self, "use_pairwise_bias", bool(self.use_pairwise_bias))
        object.__setattr__(self, "mlp_ratio", mlp_ratio)
        object.__setattr__(self, "pairwise_hidden_dim", pairwise_hidden_dim)
        object.__setattr__(self, "residual_gamma_init", float(self.residual_gamma_init))
        object.__setattr__(self, "mask_value", float(self.mask_value))

    @property
    def variant(self) -> str:
        return local_graph_variant_for_adapter(self.local_adapter)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = self.variant
        return payload


@dataclass(frozen=True)
class LocalGraphClassifierOutput:
    """Debug-rich output for the local graph classifier."""

    logits: Any
    cls_embedding: Any
    sequence_tokens: Any
    sequence_mask: Any
    particle_embeddings: Any
    adapted_particles: Any
    knn: LocalKnnOutput | None
    local_adapter_output: EdgeConvLocalAdapterOutput | PointAttentionLocalAdapterOutput | None
    pairwise_features: PairwiseFeatureOutput | None
    attention_bias: Any | None
    config: LocalGraphParticleTransformerConfig

    def summary(self) -> dict[str, Any]:
        payload = {
            "step": LOCAL_GRAPH_PART_CLASSIFIER_STEP,
            "contract": LOCAL_GRAPH_PART_CLASSIFIER_CONTRACT,
            "variant": self.config.variant,
            "local_adapter": self.config.local_adapter,
            "use_pairwise_bias": bool(self.config.use_pairwise_bias),
            "prototype_only": True,
            "uses_reference_part_backbone": False,
            "serious_comparison_ready": False,
            "logits_shape": list(self.logits.shape),
            "cls_embedding_shape": list(self.cls_embedding.shape),
            "sequence_tokens_shape": list(self.sequence_tokens.shape),
            "sequence_mask_shape": list(self.sequence_mask.shape),
            "particle_embeddings_shape": list(self.particle_embeddings.shape),
            "adapted_particles_shape": list(self.adapted_particles.shape),
        }
        if self.knn is not None:
            payload["knn"] = self.knn.summary()
        if self.local_adapter_output is not None:
            payload["local_adapter_output"] = self.local_adapter_output.summary()
        if self.pairwise_features is not None:
            payload.update(
                {
                    "pairwise_features_shape": list(self.pairwise_features.pair_features.shape),
                    "pairwise_feature_names": list(self.pairwise_features.feature_names),
                }
            )
        if self.attention_bias is not None:
            payload["attention_bias_shape"] = list(self.attention_bias.shape)
        return payload

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        particle_mask = self.sequence_mask[:, 1:]
        valid_counts = particle_mask.sum(dim=1).to(dtype=self.logits.dtype)
        diagnostics: dict[str, Any] = {
            "step": LOCAL_GRAPH_PART_CLASSIFIER_STEP,
            "contract": LOCAL_GRAPH_PART_CLASSIFIER_CONTRACT,
            "variant": self.config.variant,
            "batch_size": int(self.logits.shape[0]),
            "valid_particle_count_mean": valid_counts.mean(),
            "valid_particle_count_min": valid_counts.min(),
            "valid_particle_count_max": valid_counts.max(),
            "logit_abs_mean": self.logits.detach().abs().mean(),
            "prototype_only": torch.ones((), dtype=self.logits.dtype, device=self.logits.device),
            "uses_reference_part_backbone": torch.zeros((), dtype=self.logits.dtype, device=self.logits.device),
            "serious_comparison_ready": torch.zeros((), dtype=self.logits.dtype, device=self.logits.device),
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
        if self.local_adapter_output is not None:
            for key, value in self.local_adapter_output.diagnostics.items():
                diagnostics[f"local_{key}"] = value
        else:
            diagnostics["local_gamma"] = torch.zeros((), dtype=self.logits.dtype, device=self.logits.device)
        return diagnostics


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


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
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=tokens.device)
    else:
        mask = mask.to(device=tokens.device, dtype=torch.bool)
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    return tokens, mask


def _make_vanilla_global_encoder(config: LocalGraphParticleTransformerConfig) -> Any:
    torch = require_torch()
    layer = torch.nn.TransformerEncoderLayer(
        d_model=int(config.embed_dim),
        nhead=int(config.global_heads),
        dim_feedforward=int(round(float(config.mlp_ratio) * int(config.embed_dim))),
        dropout=float(config.dropout),
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    layer.self_attn.dropout = float(config.attention_dropout)
    return torch.nn.TransformerEncoder(layer, num_layers=int(config.global_layers))


class LocalGraphParticleTransformerClassifier(_ModuleBase):
    """Prototype HLT-only local graph classifier with custom global attention.

    This is not the real HLT ParT baseline/backbone. Serious comparisons should
    use :class:`LocalGraphAugmentedParticleTransformerClassifier`.
    """

    def __init__(self, config: LocalGraphParticleTransformerConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        if config is None:
            config = LocalGraphParticleTransformerConfig()
        elif isinstance(config, Mapping):
            config = LocalGraphParticleTransformerConfig(**dict(config))
        self.config = config
        self.embed_dim = int(config.embed_dim)
        self.input_embed = torch.nn.Sequential(
            torch.nn.LayerNorm(int(config.raw_feature_dim)),
            torch.nn.Linear(int(config.raw_feature_dim), self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.knn_config = LocalKnnConfig(k=int(config.k), raw_feature_dim=int(config.raw_feature_dim))
        self.local_adapter = self._build_local_adapter(config)
        self.cls_token = torch.nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.input_dropout = torch.nn.Dropout(float(config.dropout))
        if bool(config.use_pairwise_bias):
            self.pairwise_feature_builder = PairwiseFeatureBuilder(
                PairwiseFeatureConfig(raw_feature_dim=int(config.raw_feature_dim), include_cls_token=True)
            )
            self.pairwise_bias_encoder = PairwiseBiasEncoder(
                PairwiseBiasConfig(
                    num_heads=int(config.global_heads),
                    hidden_dim=int(config.pairwise_hidden_dim),
                    dropout=float(config.dropout),
                )
            )
            self.global_blocks = torch.nn.ModuleList(
                [
                    PairwiseBiasedAttentionBlock(
                        embed_dim=self.embed_dim,
                        num_heads=int(config.global_heads),
                        mlp_ratio=float(config.mlp_ratio),
                        dropout=float(config.dropout),
                        attention_dropout=float(config.attention_dropout),
                        mask_value=float(config.mask_value),
                    )
                    for _ in range(int(config.global_layers))
                ]
            )
            self.global_encoder = None
        else:
            self.pairwise_feature_builder = None
            self.pairwise_bias_encoder = None
            self.global_blocks = torch.nn.ModuleList()
            self.global_encoder = _make_vanilla_global_encoder(config)
        self.global_norm = torch.nn.LayerNorm(self.embed_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.Linear(self.embed_dim, 2 * self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(2 * self.embed_dim, int(config.num_classes)),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch = require_torch()
        torch.nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _build_local_adapter(self, config: LocalGraphParticleTransformerConfig) -> Any | None:
        adapter = str(config.local_adapter)
        if adapter == LOCAL_GRAPH_ADAPTER_NONE:
            return None
        if adapter == LOCAL_GRAPH_ADAPTER_EDGECONV:
            return EdgeConvLocalAdapter(
                EdgeConvLocalAdapterConfig(
                    input_dim=int(config.embed_dim),
                    k=int(config.k),
                    hidden_dim=config.local_hidden_dim,
                    dropout=float(config.dropout),
                    residual_gamma_init=float(config.residual_gamma_init),
                )
            )
        if adapter == LOCAL_GRAPH_ADAPTER_POINT_ATTENTION:
            return PointAttentionLocalAdapter(
                PointAttentionLocalAdapterConfig(
                    input_dim=int(config.embed_dim),
                    k=int(config.k),
                    num_heads=int(config.local_heads),
                    hidden_dim=config.local_hidden_dim,
                    dropout=float(config.dropout),
                    attention_dropout=float(config.attention_dropout),
                    residual_gamma_init=float(config.residual_gamma_init),
                )
            )
        raise ValueError(f"unsupported local_adapter: {adapter}")

    def to_config_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    @property
    def output_contract(self) -> str:
        return LOCAL_GRAPH_PART_CLASSIFIER_CONTRACT

    def no_weight_decay(self) -> set[str]:
        return {"cls_token"}

    def _build_global_sequence(self, particle_tokens: Any, mask: Any) -> tuple[Any, Any]:
        torch = require_torch()
        batch_size = int(particle_tokens.shape[0])
        cls = self.cls_token.expand(batch_size, 1, self.embed_dim)
        sequence = torch.cat([cls, self.input_dropout(particle_tokens)], dim=1)
        sequence_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=particle_tokens.device),
                mask.to(device=particle_tokens.device, dtype=torch.bool),
            ],
            dim=1,
        )
        return sequence, sequence_mask

    def forward_outputs(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        knn: LocalKnnOutput | None = None,
    ) -> LocalGraphClassifierOutput:
        torch = require_torch()
        raw_tokens, mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        if int(raw_tokens.shape[-1]) != int(self.config.raw_feature_dim):
            raise ValueError(
                f"raw token dimension must be {int(self.config.raw_feature_dim)}, got {int(raw_tokens.shape[-1])}"
            )
        raw_tokens = _nan_to_num_torch(raw_tokens)
        mask = mask.to(device=raw_tokens.device, dtype=torch.bool)
        particle_embeddings = self.input_embed(raw_tokens)
        particle_embeddings = torch.where(mask[:, :, None], _nan_to_num_torch(particle_embeddings), torch.zeros_like(particle_embeddings))

        local_output = None
        local_knn = knn
        adapted_particles = particle_embeddings
        if self.local_adapter is not None:
            if local_knn is None:
                local_knn = build_local_knn_graph(raw_tokens, mask, self.knn_config)
            local_output = self.local_adapter(particle_embeddings, raw_tokens, mask, knn=local_knn)
            adapted_particles = local_output.tokens
        adapted_particles = torch.where(mask[:, :, None], _nan_to_num_torch(adapted_particles), torch.zeros_like(adapted_particles))

        sequence, sequence_mask = self._build_global_sequence(adapted_particles, mask)
        pairwise_features = None
        attention_bias = None
        if bool(self.config.use_pairwise_bias):
            if self.pairwise_feature_builder is None or self.pairwise_bias_encoder is None:
                raise RuntimeError("pairwise modules are not initialized")
            pairwise_features = self.pairwise_feature_builder(raw_tokens, mask)
            if tuple(pairwise_features.token_mask.shape) != tuple(sequence_mask.shape):
                raise ValueError("pairwise token mask shape does not match global sequence mask")
            if not bool(torch.equal(pairwise_features.token_mask.to(device=sequence_mask.device), sequence_mask)):
                raise ValueError("pairwise token mask values do not match global sequence mask")
            attention_bias = self.pairwise_bias_encoder(pairwise_features)
            encoded = sequence
            for block in self.global_blocks:
                encoded = block(encoded, attention_bias, sequence_mask).tokens
        else:
            if self.global_encoder is None:
                raise RuntimeError("vanilla global encoder is not initialized")
            encoded = self.global_encoder(sequence, src_key_padding_mask=~sequence_mask)
            encoded = torch.where(sequence_mask[:, :, None], _nan_to_num_torch(encoded), torch.zeros_like(encoded))
        cls_embedding = self.global_norm(encoded[:, 0])
        logits = _nan_to_num_torch(self.classifier(cls_embedding))
        return LocalGraphClassifierOutput(
            logits=logits,
            cls_embedding=cls_embedding,
            sequence_tokens=encoded,
            sequence_mask=sequence_mask,
            particle_embeddings=particle_embeddings,
            adapted_particles=adapted_particles,
            knn=local_knn,
            local_adapter_output=local_output,
            pairwise_features=pairwise_features,
            attention_bias=attention_bias,
            config=self.config,
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
        knn: LocalKnnOutput | None = None,
    ):
        output = self.forward_outputs(tokens_or_batch, mask, knn=knn)
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.logits, output.diagnostics()
        return output.logits


def build_local_graph_particle_transformer_classifier(
    config: LocalGraphParticleTransformerConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> LocalGraphParticleTransformerClassifier:
    """Build the Step 5 prototype local graph pairwise-transformer classifier."""

    if kwargs:
        payload = {} if config is None else (config.to_dict() if isinstance(config, LocalGraphParticleTransformerConfig) else dict(config))
        payload.pop("variant", None)
        payload.update(kwargs)
        config = payload
    return LocalGraphParticleTransformerClassifier(config)


@dataclass(frozen=True)
class HLTPartBaselineRawTokenConfig:
    """Configuration for the true HLT ParT baseline in the Step 6 runner."""

    num_classes: int = 2
    model_size: str = "base"
    max_constits: int = 128
    weight_threshold: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("num_classes", "max_constits"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if str(self.model_size) not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")
        object.__setattr__(self, "weight_threshold", float(self.weight_threshold))

    @property
    def variant(self) -> str:
        return LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = self.variant
        payload["pf_feature_names"] = list(PF_FEATURE_NAMES)
        return payload


@dataclass(frozen=True)
class HLTPartBaselineRawTokenOutput:
    """Output wrapper for the true HLT ParT baseline."""

    logits: Any
    part_inputs: Mapping[str, Any]
    config: HLTPartBaselineRawTokenConfig
    uses_reference_part_backbone: bool

    def summary(self) -> dict[str, Any]:
        return {
            "step": LOCAL_GRAPH_HLT_PART_BASELINE_STEP,
            "contract": LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT,
            "variant": self.config.variant,
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "serious_comparison_ready": bool(self.uses_reference_part_backbone),
            "logits_shape": list(self.logits.shape),
            "part_feature_names": list(PF_FEATURE_NAMES),
            "part_inputs_shapes": {key: list(value.shape) for key, value in self.part_inputs.items()},
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        part_mask = self.part_inputs["mask"].squeeze(1).to(dtype=self.logits.dtype)
        valid_counts = part_mask.sum(dim=1)
        return {
            "step": LOCAL_GRAPH_HLT_PART_BASELINE_STEP,
            "contract": LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT,
            "variant": self.config.variant,
            "batch_size": int(self.logits.shape[0]),
            "valid_particle_count_mean": valid_counts.mean(),
            "valid_particle_count_min": valid_counts.min(),
            "valid_particle_count_max": valid_counts.max(),
            "logit_abs_mean": self.logits.detach().abs().mean(),
            "uses_reference_part_backbone": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
            "serious_comparison_ready": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
        }


class HLTPartBaselineRawTokenClassifier(_ModuleBase):
    """True HLT ParT baseline with the same raw-token forward API as local graph models."""

    def __init__(
        self,
        config: HLTPartBaselineRawTokenConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = HLTPartBaselineRawTokenConfig()
        elif isinstance(config, Mapping):
            config = HLTPartBaselineRawTokenConfig(**dict(config))
        self.config = config
        self.part_model = part_model or build_hlt_classifier(
            num_classes=int(config.num_classes),
            model_size=str(config.model_size),
        )

    @property
    def uses_reference_part_backbone(self) -> bool:
        return isinstance(self.part_model, ParticleTransformerHLTClassifier)

    @property
    def output_contract(self) -> str:
        return LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT

    def no_weight_decay(self) -> set[str]:
        names: set[str] = set()
        if hasattr(self.part_model, "no_weight_decay"):
            names.update({f"part_model.{name}" for name in self.part_model.no_weight_decay()})
        return names

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        payload["part_model_config"] = dict(getattr(self.part_model, "config", {}) or {})
        payload["uses_reference_part_backbone"] = bool(self.uses_reference_part_backbone)
        return payload

    def _build_part_inputs(self, raw_tokens: Any, raw_mask: Any) -> Mapping[str, Any]:
        return build_part_inputs_torch(
            raw_tokens,
            raw_mask,
            max_constits=int(self.config.max_constits),
            weight_threshold=float(self.config.weight_threshold),
        )

    def forward_outputs(self, tokens_or_batch: Any, mask: Any | None = None) -> HLTPartBaselineRawTokenOutput:
        torch = require_torch()
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        raw_tokens = _nan_to_num_torch(raw_tokens)
        raw_mask = raw_mask.to(device=raw_tokens.device, dtype=torch.bool)
        part_inputs = dict(self._build_part_inputs(raw_tokens, raw_mask))
        logits = self.part_model(
            part_inputs["points"],
            part_inputs["features"],
            part_inputs["lorentz_vectors"],
            part_inputs["mask"],
        )
        logits = _nan_to_num_torch(logits)
        return HLTPartBaselineRawTokenOutput(
            logits=logits,
            part_inputs=part_inputs,
            config=self.config,
            uses_reference_part_backbone=bool(self.uses_reference_part_backbone),
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
    ):
        output = self.forward_outputs(tokens_or_batch, mask)
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.logits, output.diagnostics()
        return output.logits


def build_hlt_part_baseline_raw_token_classifier(
    config: HLTPartBaselineRawTokenConfig | Mapping[str, Any] | None = None,
    *,
    part_model: Any | None = None,
    **kwargs: Any,
) -> HLTPartBaselineRawTokenClassifier:
    """Build the true HLT ParT baseline for the Step 6 local-graph runner."""

    if kwargs:
        payload = {} if config is None else (config.to_dict() if isinstance(config, HLTPartBaselineRawTokenConfig) else dict(config))
        payload.pop("variant", None)
        payload.pop("pf_feature_names", None)
        payload.update(kwargs)
        config = payload
    return HLTPartBaselineRawTokenClassifier(config, part_model=part_model)


@dataclass(frozen=True)
class LocalGraphAugmentedPartConfig:
    """Configuration for local graph augmentation in front of real Weaver ParT."""

    num_classes: int = 2
    model_size: str = "base"
    max_constits: int = 128
    local_adapter: str = LOCAL_GRAPH_ADAPTER_POINT_ATTENTION
    k: int = 16
    local_embed_dim: int = 128
    local_heads: int = 8
    local_hidden_dim: int | None = None
    dropout: float = 0.05
    attention_dropout: float = 0.05
    residual_gamma_init: float = 0.0
    weight_threshold: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("num_classes", "max_constits", "k", "local_embed_dim"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if str(self.model_size) not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")
        local_heads = int(self.local_heads)
        if local_heads <= 0:
            raise ValueError("local_heads must be positive")
        if normalize_local_graph_adapter(self.local_adapter) == LOCAL_GRAPH_ADAPTER_POINT_ATTENTION:
            if int(self.local_embed_dim) % local_heads != 0:
                raise ValueError("local_embed_dim must be divisible by local_heads for point attention")
        object.__setattr__(self, "local_heads", local_heads)
        hidden_dim = self.local_hidden_dim
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("local_hidden_dim must be positive when provided")
        object.__setattr__(self, "local_hidden_dim", hidden_dim)
        for field_name in ("dropout", "attention_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "local_adapter", normalize_local_graph_adapter(self.local_adapter))
        object.__setattr__(self, "residual_gamma_init", float(self.residual_gamma_init))
        object.__setattr__(self, "weight_threshold", float(self.weight_threshold))

    @property
    def variant(self) -> str:
        return local_graph_variant_for_adapter(self.local_adapter)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = self.variant
        payload["pf_feature_names"] = list(PF_FEATURE_NAMES)
        return payload


@dataclass(frozen=True)
class LocalGraphAugmentedPartOutput:
    """Output for the serious local graph + reference-ParT wrapper."""

    logits: Any
    part_inputs: Mapping[str, Any]
    canonical_features: Any
    adapted_features: Any
    knn: LocalKnnOutput | None
    local_adapter_output: EdgeConvLocalAdapterOutput | PointAttentionLocalAdapterOutput | None
    config: LocalGraphAugmentedPartConfig
    uses_reference_part_backbone: bool

    def summary(self) -> dict[str, Any]:
        return {
            "step": LOCAL_GRAPH_AUGMENTED_PART_STEP,
            "contract": LOCAL_GRAPH_AUGMENTED_PART_CONTRACT,
            "variant": self.config.variant,
            "local_adapter": self.config.local_adapter,
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "serious_comparison_ready": bool(self.uses_reference_part_backbone),
            "logits_shape": list(self.logits.shape),
            "canonical_features_shape": list(self.canonical_features.shape),
            "adapted_features_shape": list(self.adapted_features.shape),
            "part_feature_names": list(PF_FEATURE_NAMES),
            "part_inputs_shapes": {key: list(value.shape) for key, value in self.part_inputs.items()},
            "knn": None if self.knn is None else self.knn.summary(),
            "local_adapter_output": None if self.local_adapter_output is None else self.local_adapter_output.summary(),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        part_mask = self.part_inputs["mask"].squeeze(1).to(dtype=self.logits.dtype)
        valid_counts = part_mask.sum(dim=1)
        diagnostics: dict[str, Any] = {
            "step": LOCAL_GRAPH_AUGMENTED_PART_STEP,
            "contract": LOCAL_GRAPH_AUGMENTED_PART_CONTRACT,
            "variant": self.config.variant,
            "batch_size": int(self.logits.shape[0]),
            "valid_particle_count_mean": valid_counts.mean(),
            "valid_particle_count_min": valid_counts.min(),
            "valid_particle_count_max": valid_counts.max(),
            "logit_abs_mean": self.logits.detach().abs().mean(),
            "uses_reference_part_backbone": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
            "serious_comparison_ready": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
        }
        if self.local_adapter_output is not None:
            for key, value in self.local_adapter_output.diagnostics.items():
                diagnostics[f"local_{key}"] = value
        else:
            diagnostics["local_gamma"] = torch.zeros((), dtype=self.logits.dtype, device=self.logits.device)
        return diagnostics


class LocalGraphAugmentedParticleTransformerClassifier(_ModuleBase):
    """Local graph residual adapter feeding the real canonical ParT backbone.

    The local adapter operates on canonical ParT PF feature rows, then the
    adapted PF features are transposed back into the exact ``features`` tensor
    consumed by ``ParticleTransformerHLTClassifier``.
    """

    def __init__(
        self,
        config: LocalGraphAugmentedPartConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        if config is None:
            config = LocalGraphAugmentedPartConfig()
        elif isinstance(config, Mapping):
            config = LocalGraphAugmentedPartConfig(**dict(config))
        self.config = config
        self.part_model = part_model or build_hlt_classifier(
            num_classes=int(config.num_classes),
            model_size=str(config.model_size),
        )
        self.knn_config = LocalKnnConfig(k=int(config.k), raw_feature_dim=RAW_TOKEN_DIM)
        torch = require_torch()
        if config.local_adapter == LOCAL_GRAPH_ADAPTER_NONE:
            self.local_feature_embed = None
            self.local_feature_delta = None
        else:
            self.local_feature_embed = torch.nn.Sequential(
                torch.nn.LayerNorm(len(PF_FEATURE_NAMES)),
                torch.nn.Linear(len(PF_FEATURE_NAMES), int(config.local_embed_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(config.dropout)),
                torch.nn.Linear(int(config.local_embed_dim), int(config.local_embed_dim)),
            )
            self.local_feature_delta = torch.nn.Linear(int(config.local_embed_dim), len(PF_FEATURE_NAMES), bias=False)
        self.local_adapter = self._build_local_adapter(config)

    @property
    def uses_reference_part_backbone(self) -> bool:
        return isinstance(self.part_model, ParticleTransformerHLTClassifier)

    @property
    def output_contract(self) -> str:
        return LOCAL_GRAPH_AUGMENTED_PART_CONTRACT

    def no_weight_decay(self) -> set[str]:
        names: set[str] = set()
        if hasattr(self.part_model, "no_weight_decay"):
            names.update({f"part_model.{name}" for name in self.part_model.no_weight_decay()})
        return names

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        payload["part_model_config"] = dict(getattr(self.part_model, "config", {}) or {})
        payload["uses_reference_part_backbone"] = bool(self.uses_reference_part_backbone)
        return payload

    def _build_local_adapter(self, config: LocalGraphAugmentedPartConfig) -> Any | None:
        input_dim = int(config.local_embed_dim)
        if config.local_adapter == LOCAL_GRAPH_ADAPTER_NONE:
            return None
        if config.local_adapter == LOCAL_GRAPH_ADAPTER_EDGECONV:
            return EdgeConvLocalAdapter(
                EdgeConvLocalAdapterConfig(
                    input_dim=input_dim,
                    k=int(config.k),
                    hidden_dim=config.local_hidden_dim,
                    dropout=float(config.dropout),
                    residual_gamma_init=float(config.residual_gamma_init),
                )
            )
        if config.local_adapter == LOCAL_GRAPH_ADAPTER_POINT_ATTENTION:
            return PointAttentionLocalAdapter(
                PointAttentionLocalAdapterConfig(
                    input_dim=input_dim,
                    k=int(config.k),
                    num_heads=int(config.local_heads),
                    hidden_dim=config.local_hidden_dim,
                    dropout=float(config.dropout),
                    attention_dropout=float(config.attention_dropout),
                    residual_gamma_init=float(config.residual_gamma_init),
                )
            )
        raise ValueError(f"unsupported local_adapter: {config.local_adapter}")

    def _build_part_inputs(self, raw_tokens: Any, raw_mask: Any) -> Mapping[str, Any]:
        return build_part_inputs_torch(
            raw_tokens,
            raw_mask,
            max_constits=int(self.config.max_constits),
            weight_threshold=float(self.config.weight_threshold),
        )

    def forward_outputs(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        knn: LocalKnnOutput | None = None,
    ) -> LocalGraphAugmentedPartOutput:
        torch = require_torch()
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        raw_tokens = _nan_to_num_torch(raw_tokens)
        raw_mask = raw_mask.to(device=raw_tokens.device, dtype=torch.bool)
        part_inputs = dict(self._build_part_inputs(raw_tokens, raw_mask))
        part_mask = part_inputs["mask"].squeeze(1).to(device=raw_tokens.device, dtype=torch.bool)
        if int(part_mask.shape[1]) != int(raw_tokens.shape[1]):
            raise ValueError(
                "LocalGraphAugmentedParticleTransformerClassifier currently requires max_constits >= input slots "
                "so canonical ParT features remain aligned with raw HLT tokens."
            )
        canonical_features = part_inputs["features"].transpose(1, 2).contiguous()
        adapted_features = canonical_features
        local_output = None
        local_knn = knn
        if self.local_adapter is not None:
            if self.local_feature_embed is None or self.local_feature_delta is None:
                raise RuntimeError("local feature projection layers are not initialized")
            if local_knn is None:
                local_knn = build_local_knn_graph(raw_tokens, raw_mask, self.knn_config)
            local_embeddings = self.local_feature_embed(canonical_features)
            local_embeddings = local_embeddings * part_mask[:, :, None].to(dtype=local_embeddings.dtype)
            local_output = self.local_adapter(local_embeddings, raw_tokens, raw_mask, knn=local_knn)
            hidden_delta = local_output.tokens - local_embeddings
            adapted_features = canonical_features + self.local_feature_delta(hidden_delta)
        adapted_features = _nan_to_num_torch(adapted_features)
        adapted_features = adapted_features * part_mask[:, :, None].to(dtype=adapted_features.dtype)
        part_inputs["features"] = adapted_features.transpose(1, 2).contiguous()
        logits = self.part_model(
            part_inputs["points"],
            part_inputs["features"],
            part_inputs["lorentz_vectors"],
            part_inputs["mask"],
        )
        logits = _nan_to_num_torch(logits)
        return LocalGraphAugmentedPartOutput(
            logits=logits,
            part_inputs=part_inputs,
            canonical_features=canonical_features,
            adapted_features=adapted_features,
            knn=local_knn,
            local_adapter_output=local_output,
            config=self.config,
            uses_reference_part_backbone=bool(self.uses_reference_part_backbone),
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
        knn: LocalKnnOutput | None = None,
    ):
        output = self.forward_outputs(tokens_or_batch, mask, knn=knn)
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.logits, output.diagnostics()
        return output.logits


def build_local_graph_augmented_part_classifier(
    config: LocalGraphAugmentedPartConfig | Mapping[str, Any] | None = None,
    *,
    part_model: Any | None = None,
    **kwargs: Any,
) -> LocalGraphAugmentedParticleTransformerClassifier:
    """Build the serious local graph + reference-ParT classifier."""

    if kwargs:
        payload = {} if config is None else (config.to_dict() if isinstance(config, LocalGraphAugmentedPartConfig) else dict(config))
        payload.pop("variant", None)
        payload.pop("pf_feature_names", None)
        payload.update(kwargs)
        config = payload
    return LocalGraphAugmentedParticleTransformerClassifier(config, part_model=part_model)


def build_local_graph_comparison_classifier(
    variant: str,
    *,
    num_classes: int = 2,
    model_size: str = "base",
    max_constits: int = 128,
    k: int = 16,
    local_embed_dim: int = 128,
    local_heads: int = 8,
    local_hidden_dim: int | None = None,
    dropout: float = 0.05,
    attention_dropout: float = 0.05,
    residual_gamma_init: float = 0.0,
    weight_threshold: float = 0.0,
    part_model: Any | None = None,
):
    """Build one Step 6 comparison model by variant name."""

    variant = normalize_local_graph_comparison_variant(variant)
    if variant == LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE:
        return build_hlt_part_baseline_raw_token_classifier(
            HLTPartBaselineRawTokenConfig(
                num_classes=int(num_classes),
                model_size=str(model_size),
                max_constits=int(max_constits),
                weight_threshold=float(weight_threshold),
            ),
            part_model=part_model,
        )
    if variant == LOCAL_GRAPH_MODEL_VARIANT_EDGECONV:
        adapter = LOCAL_GRAPH_ADAPTER_EDGECONV
    elif variant == LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION:
        adapter = LOCAL_GRAPH_ADAPTER_POINT_ATTENTION
    elif variant == LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL:
        adapter = LOCAL_GRAPH_ADAPTER_NONE
    else:
        raise ValueError(f"unsupported local graph comparison variant: {variant}")
    return build_local_graph_augmented_part_classifier(
        LocalGraphAugmentedPartConfig(
            num_classes=int(num_classes),
            model_size=str(model_size),
            max_constits=int(max_constits),
            local_adapter=adapter,
            k=int(k),
            local_embed_dim=int(local_embed_dim),
            local_heads=int(local_heads),
            local_hidden_dim=local_hidden_dim,
            dropout=float(dropout),
            attention_dropout=float(attention_dropout),
            residual_gamma_init=float(residual_gamma_init),
            weight_threshold=float(weight_threshold),
        ),
        part_model=part_model,
    )


__all__ = [
    "LOCAL_GRAPH_ADAPTER_EDGECONV",
    "LOCAL_GRAPH_ADAPTER_NONE",
    "LOCAL_GRAPH_ADAPTER_POINT_ATTENTION",
    "LOCAL_GRAPH_ADAPTERS",
    "LOCAL_GRAPH_COMPARISON_VARIANTS",
    "LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT",
    "LOCAL_GRAPH_HLT_PART_BASELINE_STEP",
    "LOCAL_GRAPH_MODEL_VARIANT_EDGECONV",
    "LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE",
    "LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL",
    "LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION",
    "LOCAL_GRAPH_MODEL_VARIANTS",
    "LOCAL_GRAPH_PART_CLASSIFIER_CONTRACT",
    "LOCAL_GRAPH_PART_CLASSIFIER_STEP",
    "LOCAL_GRAPH_AUGMENTED_PART_CONTRACT",
    "LOCAL_GRAPH_AUGMENTED_PART_STEP",
    "HLTPartBaselineRawTokenClassifier",
    "HLTPartBaselineRawTokenConfig",
    "HLTPartBaselineRawTokenOutput",
    "LocalGraphAugmentedPartConfig",
    "LocalGraphAugmentedPartOutput",
    "LocalGraphAugmentedParticleTransformerClassifier",
    "LocalGraphClassifierOutput",
    "LocalGraphParticleTransformerClassifier",
    "LocalGraphParticleTransformerConfig",
    "build_hlt_part_baseline_raw_token_classifier",
    "build_local_graph_comparison_classifier",
    "build_local_graph_augmented_part_classifier",
    "build_local_graph_particle_transformer_classifier",
    "local_graph_variant_for_adapter",
    "normalize_local_graph_comparison_variant",
    "normalize_local_graph_adapter",
]
