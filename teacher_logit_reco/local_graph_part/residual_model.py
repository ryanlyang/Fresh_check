"""Residual-expert model for local-graph HLT ParT corrections.

The residual expert keeps the serious local-graph + reference-ParT path, but
uses the ParT output as a jet embedding and predicts an additive correction to
a frozen baseline HLT ParT logit margin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, build_hlt_classifier, require_torch
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .knn import LocalKnnOutput
from .local_blocks import EdgeConvLocalAdapterOutput, PointAttentionLocalAdapterOutput
from .model import (
    LOCAL_GRAPH_ADAPTER_NONE,
    LOCAL_GRAPH_ADAPTER_POINT_ATTENTION,
    LOCAL_GRAPH_AUGMENTED_PART_CONTRACT,
    LOCAL_GRAPH_AUGMENTED_PART_STEP,
    LocalGraphAugmentedPartConfig,
    LocalGraphAugmentedPartOutput,
    LocalGraphAugmentedParticleTransformerClassifier,
    normalize_local_graph_adapter,
)
from .residual_cache import LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES


LOCAL_GRAPH_RESIDUAL_EXPERT_STEP = "local_graph_residual_expert_step2_model"
LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT = "local_graph_residual_expert_model_v1"
LOCAL_GRAPH_RESIDUAL_EXPERT_VARIANT_SUFFIX = "_residual_expert"


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1.0e-8)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def _torch_binary_logits_from_log_odds(log_odds: Any) -> Any:
    torch = require_torch()
    log_odds = log_odds.reshape(-1)
    return torch.stack((-0.5 * log_odds, 0.5 * log_odds), dim=1)


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


@dataclass(frozen=True)
class LocalGraphResidualExpertConfig:
    """Configuration for the Step 2 additive residual expert wrapper."""

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

    backbone_output_dim: int = 128
    condition_dim: int = len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)
    condition_embed_dim: int = 64
    residual_hidden_dim: int = 128
    residual_dropout: float = 0.05
    residual_output_scale: float = 1.0
    gate_bias_init: float = -1.0
    raw_residual_init_std: float = 1.0e-3
    alpha_initial: float = 0.1
    alpha_learnable: bool = True
    alpha_max: float | None = 2.0

    def __post_init__(self) -> None:
        for field_name in (
            "max_constits",
            "k",
            "local_embed_dim",
            "local_heads",
            "backbone_output_dim",
            "condition_dim",
            "condition_embed_dim",
            "residual_hidden_dim",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if str(self.model_size) not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")
        if int(self.condition_dim) != len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES):
            raise ValueError(
                "condition_dim must match LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES for Step 2"
            )
        if int(self.local_embed_dim) % int(self.local_heads) != 0:
            raise ValueError("local_embed_dim must be divisible by local_heads")
        hidden_dim = self.local_hidden_dim
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("local_hidden_dim must be positive when provided")
        object.__setattr__(self, "local_hidden_dim", hidden_dim)
        for field_name in ("dropout", "attention_dropout", "residual_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            object.__setattr__(self, field_name, value)
        for field_name in ("residual_output_scale", "alpha_initial"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
            object.__setattr__(self, field_name, value)
        alpha_max = self.alpha_max
        if alpha_max is not None:
            alpha_max = float(alpha_max)
            if not math.isfinite(alpha_max) or alpha_max <= 0.0:
                raise ValueError("alpha_max must be positive when provided")
            object.__setattr__(self, "alpha_max", alpha_max)
        object.__setattr__(self, "local_adapter", normalize_local_graph_adapter(self.local_adapter))
        object.__setattr__(self, "residual_gamma_init", float(self.residual_gamma_init))
        object.__setattr__(self, "weight_threshold", float(self.weight_threshold))
        object.__setattr__(self, "gate_bias_init", float(self.gate_bias_init))
        object.__setattr__(self, "raw_residual_init_std", float(self.raw_residual_init_std))
        object.__setattr__(self, "alpha_learnable", bool(self.alpha_learnable))

    @property
    def base_variant(self) -> str:
        if self.local_adapter == LOCAL_GRAPH_ADAPTER_NONE:
            return "local_graph_adapter_disabled_control"
        if self.local_adapter == "edgeconv":
            return "local_edgeconv_adapter"
        if self.local_adapter == LOCAL_GRAPH_ADAPTER_POINT_ATTENTION:
            return "local_point_attention_adapter"
        raise ValueError(f"unsupported local_adapter: {self.local_adapter}")

    @property
    def variant(self) -> str:
        return f"{self.base_variant}{LOCAL_GRAPH_RESIDUAL_EXPERT_VARIANT_SUFFIX}"

    def to_augmented_part_config(self) -> LocalGraphAugmentedPartConfig:
        return LocalGraphAugmentedPartConfig(
            num_classes=int(self.backbone_output_dim),
            model_size=str(self.model_size),
            max_constits=int(self.max_constits),
            local_adapter=str(self.local_adapter),
            k=int(self.k),
            local_embed_dim=int(self.local_embed_dim),
            local_heads=int(self.local_heads),
            local_hidden_dim=self.local_hidden_dim,
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            residual_gamma_init=float(self.residual_gamma_init),
            weight_threshold=float(self.weight_threshold),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = self.variant
        payload["base_variant"] = self.base_variant
        payload["condition_feature_names"] = list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)
        payload["pf_feature_names"] = list(PF_FEATURE_NAMES)
        return payload


@dataclass(frozen=True)
class LocalGraphResidualExpertOutput:
    """Debug-rich residual expert output."""

    fused_logit: Any
    residual_logit: Any
    raw_residual_logit: Any
    baseline_logit: Any
    alpha: Any
    gate: Any
    fused_logits: Any
    residual_logits: Any
    backbone_embedding: Any
    condition_features: Any
    condition_embedding: Any
    backbone_output: LocalGraphAugmentedPartOutput
    config: LocalGraphResidualExpertConfig
    uses_reference_part_backbone: bool

    @property
    def logits(self) -> Any:
        """Binary fused logits, for trainer compatibility."""

        return self.fused_logits

    def summary(self) -> dict[str, Any]:
        return {
            "step": LOCAL_GRAPH_RESIDUAL_EXPERT_STEP,
            "contract": LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT,
            "variant": self.config.variant,
            "base_variant": self.config.base_variant,
            "local_adapter": self.config.local_adapter,
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "serious_comparison_ready": bool(self.uses_reference_part_backbone),
            "fused_logits_shape": list(self.fused_logits.shape),
            "residual_logit_shape": list(self.residual_logit.shape),
            "baseline_logit_shape": list(self.baseline_logit.shape),
            "backbone_embedding_shape": list(self.backbone_embedding.shape),
            "condition_features_shape": list(self.condition_features.shape),
            "condition_embedding_shape": list(self.condition_embedding.shape),
            "condition_feature_names": list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
            "backbone_summary": self.backbone_output.summary(),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        diagnostics = {
            "step": LOCAL_GRAPH_RESIDUAL_EXPERT_STEP,
            "contract": LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT,
            "variant": self.config.variant,
            "batch_size": int(self.fused_logit.shape[0]),
            "alpha": self.alpha.detach().reshape(()),
            "baseline_logit_mean": self.baseline_logit.detach().mean(),
            "baseline_logit_std": self.baseline_logit.detach().std(unbiased=False),
            "fused_logit_mean": self.fused_logit.detach().mean(),
            "fused_logit_std": self.fused_logit.detach().std(unbiased=False),
            "residual_mean": self.residual_logit.detach().mean(),
            "residual_std": self.residual_logit.detach().std(unbiased=False),
            "residual_abs_mean": self.residual_logit.detach().abs().mean(),
            "raw_residual_abs_mean": self.raw_residual_logit.detach().abs().mean(),
            "gate_mean": self.gate.detach().mean(),
            "gate_min": self.gate.detach().amin(),
            "gate_max": self.gate.detach().amax(),
            "uses_reference_part_backbone": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.fused_logit.dtype,
                device=self.fused_logit.device,
            ),
        }
        for key, value in self.backbone_output.diagnostics().items():
            if hasattr(value, "detach"):
                diagnostics[f"backbone_{key}"] = value
        return diagnostics


class LocalGraphResidualExpert(LocalGraphAugmentedParticleTransformerClassifier):
    """Local graph expert that predicts an additive correction to HLT ParT."""

    def __init__(
        self,
        config: LocalGraphResidualExpertConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
    ) -> None:
        torch = require_torch()
        if config is None:
            config = LocalGraphResidualExpertConfig()
        elif isinstance(config, Mapping):
            config = LocalGraphResidualExpertConfig(**dict(config))
        self.residual_config = config
        super().__init__(config.to_augmented_part_config(), part_model=part_model)
        self.config = config
        condition_dim = int(config.condition_dim)
        condition_embed_dim = int(config.condition_embed_dim)
        backbone_dim = int(config.backbone_output_dim)
        hidden_dim = int(config.residual_hidden_dim)
        self.condition_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(condition_dim),
            torch.nn.Linear(condition_dim, condition_embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.residual_dropout)),
            torch.nn.Linear(condition_embed_dim, condition_embed_dim),
            torch.nn.GELU(),
        )
        self.residual_body = torch.nn.Sequential(
            torch.nn.LayerNorm(backbone_dim + condition_embed_dim),
            torch.nn.Linear(backbone_dim + condition_embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.residual_dropout)),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.raw_residual_head = torch.nn.Linear(hidden_dim, 1)
        self.gate_head = torch.nn.Linear(hidden_dim, 1)
        if bool(config.alpha_learnable):
            self.alpha_unconstrained = torch.nn.Parameter(
                torch.tensor(_inverse_softplus(float(config.alpha_initial)), dtype=torch.float32)
            )
            self.register_buffer("alpha_fixed", torch.tensor(float(config.alpha_initial), dtype=torch.float32))
        else:
            self.register_parameter("alpha_unconstrained", None)
            self.register_buffer("alpha_fixed", torch.tensor(float(config.alpha_initial), dtype=torch.float32))
        self.reset_residual_parameters()

    @property
    def output_contract(self) -> str:
        return LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT

    @property
    def uses_reference_part_backbone(self) -> bool:
        return isinstance(self.part_model, ParticleTransformerHLTClassifier)

    def reset_residual_parameters(self) -> None:
        torch = require_torch()
        torch.nn.init.normal_(
            self.raw_residual_head.weight,
            mean=0.0,
            std=float(self.config.raw_residual_init_std),
        )
        torch.nn.init.zeros_(self.raw_residual_head.bias)
        torch.nn.init.zeros_(self.gate_head.weight)
        torch.nn.init.constant_(self.gate_head.bias, float(self.config.gate_bias_init))

    def alpha_value(self) -> Any:
        torch = require_torch()
        if self.alpha_unconstrained is None:
            alpha = self.alpha_fixed.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        else:
            alpha = torch.nn.functional.softplus(self.alpha_unconstrained)
        if self.config.alpha_max is not None:
            alpha = torch.clamp(alpha, max=float(self.config.alpha_max))
        return alpha

    def no_weight_decay(self) -> set[str]:
        names = super().no_weight_decay()
        names.add("alpha_unconstrained")
        return names

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        payload["backbone_config"] = self.config.to_augmented_part_config().to_dict()
        payload["part_model_config"] = dict(getattr(self.part_model, "config", {}) or {})
        payload["uses_reference_part_backbone"] = bool(self.uses_reference_part_backbone)
        return payload

    def _extract_baseline_inputs(
        self,
        tokens_or_batch: Any,
        *,
        baseline_logit: Any | None,
        baseline_condition_features: Any | None,
        batch_size: int,
        device: Any,
        dtype: Any,
    ) -> tuple[Any, Any]:
        torch = require_torch()
        if isinstance(tokens_or_batch, Mapping):
            if baseline_logit is None:
                baseline_logit = tokens_or_batch.get(
                    "z_base",
                    tokens_or_batch.get("baseline_logit", tokens_or_batch.get("baseline_margin")),
                )
            if baseline_condition_features is None:
                baseline_condition_features = tokens_or_batch.get(
                    "baseline_condition_features",
                    tokens_or_batch.get("condition_features", tokens_or_batch.get("baseline_features")),
                )
        if baseline_logit is None:
            raise ValueError("baseline_logit/z_base is required for residual expert forward")
        if not isinstance(baseline_logit, torch.Tensor):
            baseline_logit = torch.as_tensor(baseline_logit, dtype=dtype, device=device)
        else:
            baseline_logit = baseline_logit.to(device=device, dtype=dtype)
        baseline_logit = baseline_logit.reshape(-1)
        if int(baseline_logit.shape[0]) != int(batch_size):
            raise ValueError(
                f"baseline_logit length {int(baseline_logit.shape[0])} does not match batch size {int(batch_size)}"
            )
        if baseline_condition_features is None:
            zeros = torch.zeros_like(baseline_logit)
            ones = torch.ones_like(baseline_logit)
            baseline_condition_features = torch.stack(
                (baseline_logit, torch.sigmoid(baseline_logit), zeros, zeros, zeros, ones),
                dim=1,
            )
        elif not isinstance(baseline_condition_features, torch.Tensor):
            baseline_condition_features = torch.as_tensor(baseline_condition_features, dtype=dtype, device=device)
        else:
            baseline_condition_features = baseline_condition_features.to(device=device, dtype=dtype)
        if int(baseline_condition_features.ndim) != 2:
            raise ValueError("baseline_condition_features must have shape [batch, condition_dim]")
        if int(baseline_condition_features.shape[0]) != int(batch_size):
            raise ValueError("baseline_condition_features batch dimension does not match tokens")
        if int(baseline_condition_features.shape[1]) != int(self.config.condition_dim):
            raise ValueError(
                f"baseline_condition_features dim must be {int(self.config.condition_dim)}, "
                f"got {int(baseline_condition_features.shape[1])}"
            )
        return _nan_to_num_torch(baseline_logit), _nan_to_num_torch(baseline_condition_features)

    def forward_outputs(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        baseline_logit: Any | None = None,
        baseline_condition_features: Any | None = None,
        knn: LocalKnnOutput | None = None,
    ) -> LocalGraphResidualExpertOutput:
        torch = require_torch()
        backbone_output = super().forward_outputs(tokens_or_batch, mask, knn=knn)
        backbone_embedding = _nan_to_num_torch(backbone_output.logits)
        if int(backbone_embedding.ndim) != 2 or int(backbone_embedding.shape[1]) != int(self.config.backbone_output_dim):
            raise ValueError(
                f"backbone embedding must have shape [batch, {int(self.config.backbone_output_dim)}], "
                f"got {tuple(backbone_embedding.shape)}"
            )
        z_base, condition_features = self._extract_baseline_inputs(
            tokens_or_batch,
            baseline_logit=baseline_logit,
            baseline_condition_features=baseline_condition_features,
            batch_size=int(backbone_embedding.shape[0]),
            device=backbone_embedding.device,
            dtype=backbone_embedding.dtype,
        )
        condition_embedding = _nan_to_num_torch(self.condition_mlp(condition_features))
        residual_input = torch.cat((backbone_embedding, condition_embedding), dim=1)
        hidden = _nan_to_num_torch(self.residual_body(residual_input))
        raw_residual = self.raw_residual_head(hidden).reshape(-1)
        gate = torch.sigmoid(self.gate_head(hidden).reshape(-1))
        residual = float(self.config.residual_output_scale) * gate * raw_residual
        residual = _nan_to_num_torch(residual)
        alpha = self.alpha_value().to(device=residual.device, dtype=residual.dtype)
        fused = _nan_to_num_torch(z_base + alpha * residual)
        return LocalGraphResidualExpertOutput(
            fused_logit=fused,
            residual_logit=residual,
            raw_residual_logit=_nan_to_num_torch(raw_residual),
            baseline_logit=z_base,
            alpha=alpha,
            gate=gate,
            fused_logits=_torch_binary_logits_from_log_odds(fused),
            residual_logits=_torch_binary_logits_from_log_odds(residual),
            backbone_embedding=backbone_embedding,
            condition_features=condition_features,
            condition_embedding=condition_embedding,
            backbone_output=backbone_output,
            config=self.config,
            uses_reference_part_backbone=bool(self.uses_reference_part_backbone),
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        baseline_logit: Any | None = None,
        baseline_condition_features: Any | None = None,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
        knn: LocalKnnOutput | None = None,
    ):
        output = self.forward_outputs(
            tokens_or_batch,
            mask,
            baseline_logit=baseline_logit,
            baseline_condition_features=baseline_condition_features,
            knn=knn,
        )
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.fused_logits, output.diagnostics()
        return output.fused_logits


def build_local_graph_residual_expert(
    config: LocalGraphResidualExpertConfig | Mapping[str, Any] | None = None,
    *,
    part_model: Any | None = None,
    **kwargs: Any,
) -> LocalGraphResidualExpert:
    """Build a Step 2 residual expert."""

    if kwargs:
        payload = {} if config is None else (config.to_dict() if isinstance(config, LocalGraphResidualExpertConfig) else dict(config))
        payload.pop("variant", None)
        payload.pop("base_variant", None)
        payload.pop("condition_feature_names", None)
        payload.pop("pf_feature_names", None)
        payload.update(kwargs)
        config = payload
    return LocalGraphResidualExpert(config, part_model=part_model)


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_EXPERT_STEP",
    "LOCAL_GRAPH_RESIDUAL_EXPERT_VARIANT_SUFFIX",
    "LocalGraphResidualExpert",
    "LocalGraphResidualExpertConfig",
    "LocalGraphResidualExpertOutput",
    "build_local_graph_residual_expert",
]
