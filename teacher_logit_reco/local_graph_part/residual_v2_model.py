"""Trainable local-graph residual expert for V2.

This is the first trainable V2 component.  It does not rebuild or widen the
HLT ParT baseline.  Instead it consumes the exact frozen baseline margin and
true penultimate embedding cached by Step 4, builds an HLT-only local graph
context from raw HLT particles, and predicts a small additive correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .knn import LocalKnnOutput
from .local_blocks import PointAttentionLocalAdapter, PointAttentionLocalAdapterConfig, PointAttentionLocalAdapterOutput
from .residual_cache import LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES
from .residual_v2_protocol import (
    LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
)

try:  # Keep import-time behavior friendly when torch is not installed.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN_MAX = "mean_max"
LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN = "mean"
LOCAL_GRAPH_RESIDUAL_V2_POOL_MODES = (LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN_MAX, LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN)
LOCAL_GRAPH_RESIDUAL_V2_INPUT_FULL = "full"
LOCAL_GRAPH_RESIDUAL_V2_INPUT_EMBEDDING_ONLY = "embedding_only"
LOCAL_GRAPH_RESIDUAL_V2_INPUT_LOCAL_ONLY = "local_only"
LOCAL_GRAPH_RESIDUAL_V2_INPUT_MODES = (
    LOCAL_GRAPH_RESIDUAL_V2_INPUT_FULL,
    LOCAL_GRAPH_RESIDUAL_V2_INPUT_EMBEDDING_ONLY,
    LOCAL_GRAPH_RESIDUAL_V2_INPUT_LOCAL_ONLY,
)


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


def _coerce_tokens_and_mask(tokens_or_batch: Any, mask: Any | None) -> tuple[Any, Any]:
    torch = require_torch()
    if isinstance(tokens_or_batch, Mapping):
        tokens = tokens_or_batch.get("tokens", tokens_or_batch.get("hlt_tokens"))
        if mask is None:
            mask = tokens_or_batch.get("mask", tokens_or_batch.get("hlt_mask"))
    else:
        tokens = tokens_or_batch
    if tokens is None or mask is None:
        raise ValueError("tokens and mask are required for LocalGraphResidualExpertV2")
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
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens last dim must be RAW_TOKEN_DIM={RAW_TOKEN_DIM}, got {int(tokens.shape[-1])}")
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    return _nan_to_num_torch(tokens), mask


def _masked_mean(values: Any, mask: Any, dim: int) -> Any:
    weights = mask.to(dtype=values.dtype)
    numerator = torch_sum(values * weights.unsqueeze(-1), dim=dim)
    denominator = torch_clamp(torch_sum(weights, dim=dim, keepdim=True), min=1.0)
    return numerator / denominator


def _masked_max(values: Any, mask: Any, dim: int) -> Any:
    torch = require_torch()
    very_negative = torch.finfo(values.dtype).min / 16.0
    masked = values.masked_fill(~mask.unsqueeze(-1), very_negative)
    pooled = torch.max(masked, dim=dim).values
    has_any = mask.any(dim=dim)
    return torch.where(has_any.unsqueeze(-1), pooled, torch.zeros_like(pooled))


def torch_sum(value: Any, *, dim: int, keepdim: bool = False) -> Any:
    return require_torch().sum(value, dim=dim, keepdim=keepdim)


def torch_clamp(value: Any, *, min: float) -> Any:
    return require_torch().clamp(value, min=float(min))


@dataclass(frozen=True)
class LocalGraphResidualExpertV2Config:
    """Configuration for the V2 exact-baseline residual expert."""

    baseline_embedding_dim: int
    raw_token_dim: int = RAW_TOKEN_DIM
    max_constits: int = 128
    k: int = 16
    local_embed_dim: int = 128
    local_heads: int = 8
    local_hidden_dim: int | None = None
    local_adapter_gamma_init: float = 1.0
    local_pool_mode: str = LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN_MAX
    dropout: float = 0.05
    attention_dropout: float = 0.05
    weight_threshold: float = 0.0

    condition_dim: int = len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)
    condition_embed_dim: int = 64
    local_context_dim: int = 128
    residual_hidden_dim: int = 256
    residual_dropout: float = 0.05
    residual_output_scale: float = 1.0
    gate_bias_init: float = -1.0
    delta_init_std: float = 1.0e-3
    gamma_initial: float = 0.1
    gamma_learnable: bool = True
    gamma_max: float | None = 2.0
    residual_input_mode: str = LOCAL_GRAPH_RESIDUAL_V2_INPUT_FULL

    def __post_init__(self) -> None:
        for field_name in (
            "baseline_embedding_dim",
            "raw_token_dim",
            "max_constits",
            "k",
            "local_embed_dim",
            "local_heads",
            "condition_dim",
            "condition_embed_dim",
            "local_context_dim",
            "residual_hidden_dim",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if int(self.raw_token_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"V2 residual expert expects RAW_TOKEN_DIM={RAW_TOKEN_DIM}")
        if int(self.condition_dim) != len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES):
            raise ValueError("condition_dim must match V2 baseline condition feature names")
        if int(self.local_embed_dim) % int(self.local_heads) != 0:
            raise ValueError("local_embed_dim must be divisible by local_heads")
        hidden_dim = self.local_hidden_dim
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("local_hidden_dim must be positive when provided")
        object.__setattr__(self, "local_hidden_dim", hidden_dim)
        pool_mode = str(self.local_pool_mode)
        if pool_mode not in LOCAL_GRAPH_RESIDUAL_V2_POOL_MODES:
            raise ValueError(f"local_pool_mode must be one of {LOCAL_GRAPH_RESIDUAL_V2_POOL_MODES}")
        object.__setattr__(self, "local_pool_mode", pool_mode)
        input_mode = str(self.residual_input_mode)
        if input_mode not in LOCAL_GRAPH_RESIDUAL_V2_INPUT_MODES:
            raise ValueError(f"residual_input_mode must be one of {LOCAL_GRAPH_RESIDUAL_V2_INPUT_MODES}")
        object.__setattr__(self, "residual_input_mode", input_mode)
        for field_name in ("dropout", "attention_dropout", "residual_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "local_adapter_gamma_init",
            "weight_threshold",
            "residual_output_scale",
            "delta_init_std",
            "gamma_initial",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if float(self.weight_threshold) < 0.0:
            raise ValueError("weight_threshold must be nonnegative")
        if float(self.residual_output_scale) < 0.0:
            raise ValueError("residual_output_scale must be nonnegative")
        if float(self.delta_init_std) < 0.0:
            raise ValueError("delta_init_std must be nonnegative")
        if float(self.gamma_initial) < 0.0:
            raise ValueError("gamma_initial must be nonnegative")
        gamma_max = self.gamma_max
        if gamma_max is not None:
            gamma_max = float(gamma_max)
            if not math.isfinite(gamma_max) or gamma_max <= 0.0:
                raise ValueError("gamma_max must be positive when provided")
            object.__setattr__(self, "gamma_max", gamma_max)
        object.__setattr__(self, "gate_bias_init", float(self.gate_bias_init))
        object.__setattr__(self, "gamma_learnable", bool(self.gamma_learnable))

    @property
    def variant(self) -> str:
        return "local_graph_residual_expert_v2_point_attention"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = self.variant
        payload["condition_feature_names"] = list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)
        payload["required_embedding_role"] = LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE
        payload["control_mode"] = self.residual_input_mode
        return payload


@dataclass(frozen=True)
class LocalGraphResidualExpertV2Output:
    """Debug-rich V2 residual expert output."""

    fused_logit: Any
    baseline_logit: Any
    residual_logit: Any
    correction_logit: Any
    raw_delta: Any
    delta: Any
    gamma: Any
    gate: Any
    fused_logits: Any
    residual_logits: Any
    correction_logits: Any
    baseline_embedding: Any
    local_context: Any
    condition_features: Any
    condition_embedding: Any
    local_adapter_output: PointAttentionLocalAdapterOutput
    config: LocalGraphResidualExpertV2Config

    @property
    def logits(self) -> Any:
        return self.fused_logits

    def summary(self) -> dict[str, Any]:
        return {
            "step": LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP,
            "contract": LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT,
            "variant": self.config.variant,
            "required_embedding_role": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
            "fused_logits_shape": list(self.fused_logits.shape),
            "residual_logits_shape": list(self.residual_logits.shape),
            "correction_logits_shape": list(self.correction_logits.shape),
            "baseline_embedding_shape": list(self.baseline_embedding.shape),
            "local_context_shape": list(self.local_context.shape),
            "condition_features_shape": list(self.condition_features.shape),
            "condition_embedding_shape": list(self.condition_embedding.shape),
            "condition_feature_names": list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
            "local_adapter_output": self.local_adapter_output.summary(),
            "residual_input_mode": self.config.residual_input_mode,
        }

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = {
            "step": LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP,
            "contract": LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT,
            "variant": self.config.variant,
            "batch_size": int(self.fused_logit.shape[0]),
            "gamma": self.gamma.detach().reshape(()),
            "baseline_logit_mean": self.baseline_logit.detach().mean(),
            "baseline_logit_std": self.baseline_logit.detach().std(unbiased=False),
            "fused_logit_mean": self.fused_logit.detach().mean(),
            "fused_logit_std": self.fused_logit.detach().std(unbiased=False),
            "delta_mean": self.delta.detach().mean(),
            "delta_std": self.delta.detach().std(unbiased=False),
            "delta_abs_mean": self.delta.detach().abs().mean(),
            "correction_abs_mean": self.correction_logit.detach().abs().mean(),
            "raw_delta_abs_mean": self.raw_delta.detach().abs().mean(),
            "gate_mean": self.gate.detach().mean(),
            "gate_min": self.gate.detach().amin(),
            "gate_max": self.gate.detach().amax(),
            "baseline_embedding_norm_mean": self.baseline_embedding.detach().norm(dim=1).mean(),
            "local_context_norm_mean": self.local_context.detach().norm(dim=1).mean(),
        }
        for key, value in self.local_adapter_output.diagnostics.items():
            if hasattr(value, "detach"):
                diagnostics[f"local_{key}"] = value
        return diagnostics


class LocalGraphResidualExpertV2(_ModuleBase):
    """Exact-baseline residual model with an HLT-only local point-attention branch."""

    def __init__(self, config: LocalGraphResidualExpertV2Config | Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        if isinstance(config, Mapping):
            config = LocalGraphResidualExpertV2Config(**dict(config))
        self.config = config
        self.raw_embed = torch.nn.Sequential(
            torch.nn.LayerNorm(int(config.raw_token_dim)),
            torch.nn.Linear(int(config.raw_token_dim), int(config.local_embed_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(int(config.local_embed_dim), int(config.local_embed_dim)),
        )
        self.local_adapter = PointAttentionLocalAdapter(
            PointAttentionLocalAdapterConfig(
                input_dim=int(config.local_embed_dim),
                k=int(config.k),
                num_heads=int(config.local_heads),
                hidden_dim=config.local_hidden_dim,
                dropout=float(config.dropout),
                attention_dropout=float(config.attention_dropout),
                residual_gamma_init=float(config.local_adapter_gamma_init),
            )
        )
        pooled_dim = int(config.local_embed_dim) * (2 if config.local_pool_mode == LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN_MAX else 1)
        self.local_context_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(pooled_dim),
            torch.nn.Linear(pooled_dim, int(config.local_context_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(int(config.local_context_dim), int(config.local_context_dim)),
        )
        self.embedding_norm = torch.nn.LayerNorm(int(config.baseline_embedding_dim))
        self.local_norm = torch.nn.LayerNorm(int(config.local_context_dim))
        self.condition_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(int(config.condition_dim)),
            torch.nn.Linear(int(config.condition_dim), int(config.condition_embed_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.residual_dropout)),
            torch.nn.Linear(int(config.condition_embed_dim), int(config.condition_embed_dim)),
            torch.nn.GELU(),
        )
        head_input_dim = int(config.baseline_embedding_dim) + int(config.local_context_dim) + int(config.condition_embed_dim)
        self.residual_body = torch.nn.Sequential(
            torch.nn.LayerNorm(head_input_dim),
            torch.nn.Linear(head_input_dim, int(config.residual_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.residual_dropout)),
            torch.nn.Linear(int(config.residual_hidden_dim), int(config.residual_hidden_dim)),
            torch.nn.GELU(),
        )
        self.delta_head = torch.nn.Linear(int(config.residual_hidden_dim), 1)
        self.gate_head = torch.nn.Linear(int(config.residual_hidden_dim), 1)
        if bool(config.gamma_learnable):
            self.gamma_unconstrained = torch.nn.Parameter(
                torch.tensor(_inverse_softplus(float(config.gamma_initial)), dtype=torch.float32)
            )
            self.register_buffer("gamma_fixed", torch.tensor(float(config.gamma_initial), dtype=torch.float32))
        else:
            self.register_parameter("gamma_unconstrained", None)
            self.register_buffer("gamma_fixed", torch.tensor(float(config.gamma_initial), dtype=torch.float32))
        self.reset_residual_parameters()

    @property
    def output_contract(self) -> str:
        return LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT

    def reset_residual_parameters(self) -> None:
        torch = require_torch()
        torch.nn.init.normal_(self.delta_head.weight, mean=0.0, std=float(self.config.delta_init_std))
        torch.nn.init.zeros_(self.delta_head.bias)
        torch.nn.init.zeros_(self.gate_head.weight)
        torch.nn.init.constant_(self.gate_head.bias, float(self.config.gate_bias_init))

    def gamma_value(self) -> Any:
        torch = require_torch()
        if self.gamma_unconstrained is None:
            gamma = self.gamma_fixed.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        else:
            gamma = torch.nn.functional.softplus(self.gamma_unconstrained)
        if self.config.gamma_max is not None:
            gamma = torch.clamp(gamma, max=float(self.config.gamma_max))
        return gamma

    def no_weight_decay(self) -> set[str]:
        return {"gamma_unconstrained"}

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        payload["output_contract"] = LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT
        return payload

    def _extract_baseline_inputs(
        self,
        tokens_or_batch: Any,
        *,
        baseline_logit: Any | None,
        baseline_embedding: Any | None,
        baseline_condition_features: Any | None,
        batch_size: int,
        device: Any,
        dtype: Any,
    ) -> tuple[Any, Any, Any]:
        torch = require_torch()
        if isinstance(tokens_or_batch, Mapping):
            if baseline_logit is None:
                baseline_logit = tokens_or_batch.get(
                    "z_base",
                    tokens_or_batch.get("baseline_logit", tokens_or_batch.get("baseline_margin")),
                )
            if baseline_embedding is None:
                baseline_embedding = tokens_or_batch.get(
                    "baseline_embedding",
                    tokens_or_batch.get("embedding", tokens_or_batch.get("part_embedding")),
                )
            if baseline_condition_features is None:
                baseline_condition_features = tokens_or_batch.get(
                    "baseline_condition_features",
                    tokens_or_batch.get("condition_features", tokens_or_batch.get("baseline_features")),
                )
        if baseline_logit is None:
            raise ValueError("baseline_logit/z_base is required for LocalGraphResidualExpertV2")
        if baseline_embedding is None:
            raise ValueError("baseline_embedding is required for LocalGraphResidualExpertV2")
        if baseline_condition_features is None:
            raise ValueError("baseline_condition_features is required for LocalGraphResidualExpertV2")

        if not isinstance(baseline_logit, torch.Tensor):
            baseline_logit = torch.as_tensor(baseline_logit, dtype=dtype, device=device)
        else:
            baseline_logit = baseline_logit.to(device=device, dtype=dtype)
        baseline_logit = baseline_logit.reshape(-1).detach()
        if int(baseline_logit.shape[0]) != int(batch_size):
            raise ValueError("baseline_logit length does not match batch size")

        if not isinstance(baseline_embedding, torch.Tensor):
            baseline_embedding = torch.as_tensor(baseline_embedding, dtype=dtype, device=device)
        else:
            baseline_embedding = baseline_embedding.to(device=device, dtype=dtype)
        baseline_embedding = baseline_embedding.detach()
        if int(baseline_embedding.ndim) != 2:
            raise ValueError("baseline_embedding must have shape [batch, embedding_dim]")
        if int(baseline_embedding.shape[0]) != int(batch_size):
            raise ValueError("baseline_embedding batch dimension does not match tokens")
        if int(baseline_embedding.shape[1]) != int(self.config.baseline_embedding_dim):
            raise ValueError(
                f"baseline_embedding dim must be {int(self.config.baseline_embedding_dim)}, "
                f"got {int(baseline_embedding.shape[1])}"
            )

        if not isinstance(baseline_condition_features, torch.Tensor):
            baseline_condition_features = torch.as_tensor(baseline_condition_features, dtype=dtype, device=device)
        else:
            baseline_condition_features = baseline_condition_features.to(device=device, dtype=dtype)
        baseline_condition_features = baseline_condition_features.detach()
        if int(baseline_condition_features.ndim) != 2:
            raise ValueError("baseline_condition_features must have shape [batch, condition_dim]")
        if int(baseline_condition_features.shape[0]) != int(batch_size):
            raise ValueError("baseline_condition_features batch dimension does not match tokens")
        if int(baseline_condition_features.shape[1]) != int(self.config.condition_dim):
            raise ValueError(
                f"baseline_condition_features dim must be {int(self.config.condition_dim)}, "
                f"got {int(baseline_condition_features.shape[1])}"
            )
        return (
            _nan_to_num_torch(baseline_logit),
            _nan_to_num_torch(baseline_embedding),
            _nan_to_num_torch(baseline_condition_features),
        )

    def _local_context(self, raw_tokens: Any, mask: Any, knn: LocalKnnOutput | None = None) -> tuple[Any, PointAttentionLocalAdapterOutput]:
        embeddings = self.raw_embed(raw_tokens)
        embeddings = torch_where_mask(mask, embeddings)
        local_output = self.local_adapter(embeddings, raw_tokens, mask, knn=knn)
        local_tokens = local_output.tokens
        if self.config.local_pool_mode == LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN_MAX:
            pooled = require_torch().cat(
                (
                    _masked_mean(local_tokens, mask, dim=1),
                    _masked_max(local_tokens, mask, dim=1),
                ),
                dim=1,
            )
        else:
            pooled = _masked_mean(local_tokens, mask, dim=1)
        return _nan_to_num_torch(self.local_context_proj(pooled)), local_output

    def forward_outputs(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        baseline_logit: Any | None = None,
        baseline_embedding: Any | None = None,
        baseline_condition_features: Any | None = None,
        knn: LocalKnnOutput | None = None,
    ) -> LocalGraphResidualExpertV2Output:
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        local_context, local_output = self._local_context(raw_tokens, raw_mask, knn=knn)
        z_base, base_embedding, condition_features = self._extract_baseline_inputs(
            tokens_or_batch,
            baseline_logit=baseline_logit,
            baseline_embedding=baseline_embedding,
            baseline_condition_features=baseline_condition_features,
            batch_size=int(raw_tokens.shape[0]),
            device=raw_tokens.device,
            dtype=raw_tokens.dtype,
        )
        condition_embedding = _nan_to_num_torch(self.condition_mlp(condition_features))
        residual_embedding = self.embedding_norm(base_embedding)
        residual_local_context = self.local_norm(local_context)
        if self.config.residual_input_mode == LOCAL_GRAPH_RESIDUAL_V2_INPUT_EMBEDDING_ONLY:
            residual_local_context = require_torch().zeros_like(residual_local_context)
        elif self.config.residual_input_mode == LOCAL_GRAPH_RESIDUAL_V2_INPUT_LOCAL_ONLY:
            residual_embedding = require_torch().zeros_like(residual_embedding)
        residual_input = require_torch().cat(
            (
                residual_embedding,
                residual_local_context,
                condition_embedding,
            ),
            dim=1,
        )
        hidden = _nan_to_num_torch(self.residual_body(residual_input))
        raw_delta = self.delta_head(hidden).reshape(-1)
        gate = require_torch().sigmoid(self.gate_head(hidden).reshape(-1))
        delta = float(self.config.residual_output_scale) * gate * raw_delta
        delta = _nan_to_num_torch(delta)
        gamma = self.gamma_value().to(device=delta.device, dtype=delta.dtype)
        correction = _nan_to_num_torch(gamma * delta)
        fused = _nan_to_num_torch(z_base + correction)
        return LocalGraphResidualExpertV2Output(
            fused_logit=fused,
            baseline_logit=z_base,
            residual_logit=delta,
            correction_logit=correction,
            raw_delta=_nan_to_num_torch(raw_delta),
            delta=delta,
            gamma=gamma,
            gate=gate,
            fused_logits=_torch_binary_logits_from_log_odds(fused),
            residual_logits=_torch_binary_logits_from_log_odds(delta),
            correction_logits=_torch_binary_logits_from_log_odds(correction),
            baseline_embedding=base_embedding,
            local_context=local_context,
            condition_features=condition_features,
            condition_embedding=condition_embedding,
            local_adapter_output=local_output,
            config=self.config,
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        baseline_logit: Any | None = None,
        baseline_embedding: Any | None = None,
        baseline_condition_features: Any | None = None,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
        knn: LocalKnnOutput | None = None,
    ):
        output = self.forward_outputs(
            tokens_or_batch,
            mask,
            baseline_logit=baseline_logit,
            baseline_embedding=baseline_embedding,
            baseline_condition_features=baseline_condition_features,
            knn=knn,
        )
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.fused_logits, output.diagnostics()
        return output.fused_logits


def torch_where_mask(mask: Any, embeddings: Any) -> Any:
    return require_torch().where(mask[:, :, None], _nan_to_num_torch(embeddings), require_torch().zeros_like(embeddings))


def build_local_graph_residual_expert_v2(
    config: LocalGraphResidualExpertV2Config | Mapping[str, Any],
    **kwargs: Any,
) -> LocalGraphResidualExpertV2:
    if kwargs:
        payload = config.to_dict() if isinstance(config, LocalGraphResidualExpertV2Config) else dict(config)
        payload.pop("variant", None)
        payload.pop("condition_feature_names", None)
        payload.pop("required_embedding_role", None)
        payload.update(kwargs)
        config = payload
    return LocalGraphResidualExpertV2(config)


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN",
    "LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN_MAX",
    "LOCAL_GRAPH_RESIDUAL_V2_POOL_MODES",
    "LOCAL_GRAPH_RESIDUAL_V2_INPUT_EMBEDDING_ONLY",
    "LOCAL_GRAPH_RESIDUAL_V2_INPUT_FULL",
    "LOCAL_GRAPH_RESIDUAL_V2_INPUT_LOCAL_ONLY",
    "LOCAL_GRAPH_RESIDUAL_V2_INPUT_MODES",
    "LocalGraphResidualExpertV2",
    "LocalGraphResidualExpertV2Config",
    "LocalGraphResidualExpertV2Output",
    "build_local_graph_residual_expert_v2",
]
