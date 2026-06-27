"""Step 6 reliability-gated residual dual-view ParT model.

The model preserves the trained HLT ParT anchor at initialization:

    final_logits = hlt_logits + gate * delta_logits

The residual head is zero-initialized and the gate starts closed, so the first
forward pass is numerically the HLT baseline while the new PN/reliability path
remains trainable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .anchor import HLTPartAnchor, HLTPartAnchorOutput
from .config import DUALVIEW_PART_NUM_CLASSES
from .pn_encoder import PNMemoryEncoder, PNMemoryEncoderOutput, build_pn_memory_encoder
from .reliability import (
    ReliabilityFeatureConfig,
    ReliabilityFeatureOutput,
    build_reliability_features,
    reliability_feature_dim,
)


DUALVIEW_PART_STEP6 = "reliability_gated_dualview_part_step6_residual_logit_model"
DUALVIEW_PART_RESIDUAL_CONTRACT = "hlt_anchor_plus_pn_reco_gated_logit_residual_v1"
DUALVIEW_PART_GATE_SCALAR = "scalar"
DUALVIEW_PART_GATE_PER_CLASS = "per_class"


if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    import torch as _torch

    _ModuleBase = _torch.nn.Module


def _nonnegative_int(value: int | None, *, field_name: str) -> int:
    if value is None:
        return 0
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


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
class DualViewResidualParTConfig:
    """Configuration for the first dual-view residual model."""

    num_classes: int = DUALVIEW_PART_NUM_CLASSES
    hlt_context_dim: int = 128
    pn_context_dim: int = 128
    reliability_dim: int = field(default_factory=reliability_feature_dim)
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.05
    gate_mode: str = DUALVIEW_PART_GATE_SCALAR
    gate_bias_init: float = -5.0
    zero_initialize_delta: bool = True
    zero_initialize_gate_head: bool = True
    output_contract: str = DUALVIEW_PART_RESIDUAL_CONTRACT
    experiment_step: str = DUALVIEW_PART_STEP6

    def __post_init__(self) -> None:
        num_classes = _positive_int(self.num_classes, field_name="num_classes")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        hlt_context_dim = _nonnegative_int(self.hlt_context_dim, field_name="hlt_context_dim")
        pn_context_dim = _nonnegative_int(self.pn_context_dim, field_name="pn_context_dim")
        reliability_dim = _nonnegative_int(self.reliability_dim, field_name="reliability_dim")
        hidden_dim = _positive_int(self.hidden_dim, field_name="hidden_dim")
        num_hidden_layers = _nonnegative_int(self.num_hidden_layers, field_name="num_hidden_layers")
        gate_mode = str(self.gate_mode).strip().lower().replace("-", "_")
        if gate_mode not in {DUALVIEW_PART_GATE_SCALAR, DUALVIEW_PART_GATE_PER_CLASS}:
            raise ValueError(f"gate_mode must be {DUALVIEW_PART_GATE_SCALAR!r} or {DUALVIEW_PART_GATE_PER_CLASS!r}")
        if hlt_context_dim + pn_context_dim + reliability_dim <= 0:
            raise ValueError("at least one context/reliability input dimension must be positive")
        if not isinstance(self.zero_initialize_delta, bool):
            raise TypeError("zero_initialize_delta must be a bool")
        if not isinstance(self.zero_initialize_gate_head, bool):
            raise TypeError("zero_initialize_gate_head must be a bool")
        object.__setattr__(self, "num_classes", num_classes)
        object.__setattr__(self, "hlt_context_dim", hlt_context_dim)
        object.__setattr__(self, "pn_context_dim", pn_context_dim)
        object.__setattr__(self, "reliability_dim", reliability_dim)
        object.__setattr__(self, "hidden_dim", hidden_dim)
        object.__setattr__(self, "num_hidden_layers", num_hidden_layers)
        object.__setattr__(self, "dropout", _dropout_value(self.dropout, field_name="dropout"))
        object.__setattr__(self, "gate_mode", gate_mode)
        object.__setattr__(self, "gate_bias_init", float(self.gate_bias_init))

    @property
    def fusion_input_dim(self) -> int:
        return int(self.hlt_context_dim + self.pn_context_dim + self.reliability_dim)

    @property
    def gate_dim(self) -> int:
        return 1 if self.gate_mode == DUALVIEW_PART_GATE_SCALAR else int(self.num_classes)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "DualViewResidualParTConfig" | None,
    ) -> "DualViewResidualParTConfig":
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
class DualViewResidualParTOutput:
    """Forward output for the reliability-gated residual model."""

    logits: Any
    hlt_logits: Any
    delta_logits: Any
    residual_logits: Any
    gate: Any
    gate_logits: Any
    hlt_context: Any
    pn_context: Any
    reliability_features: Any
    hlt_output: HLTPartAnchorOutput | None = None
    pn_output: PNMemoryEncoderOutput | None = None
    reliability_output: ReliabilityFeatureOutput | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def gated_delta_logits(self) -> Any:
        return self.residual_logits

    @property
    def anchor_output(self) -> HLTPartAnchorOutput | None:
        return self.hlt_output

    def to_dict(self) -> dict[str, Any]:
        return {
            "logits": self.logits,
            "hlt_logits": self.hlt_logits,
            "delta_logits": self.delta_logits,
            "residual_logits": self.residual_logits,
            "gated_delta_logits": self.gated_delta_logits,
            "gate": self.gate,
            "gate_logits": self.gate_logits,
            "hlt_context": self.hlt_context,
            "pn_context": self.pn_context,
            "reliability_features": self.reliability_features,
            "diagnostics": dict(self.diagnostics),
        }


def _make_mlp(*, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int, dropout: float):
    torch = require_torch()
    layers: list[Any] = [torch.nn.LayerNorm(int(input_dim))]
    current_dim = int(input_dim)
    for _ in range(int(num_hidden_layers)):
        layers.extend(
            [
                torch.nn.Linear(current_dim, int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
            ]
        )
        current_dim = int(hidden_dim)
    layers.append(torch.nn.Linear(current_dim, int(output_dim)))
    return torch.nn.Sequential(*layers)


def _last_linear(module: Any):
    torch = require_torch()
    for child in reversed(list(module.modules())):
        if isinstance(child, torch.nn.Linear):
            return child
    raise RuntimeError("MLP has no Linear layer")


def _tensor_to_float(value: Any) -> float:
    return float(value.detach().float().cpu().item())


class DualViewResidualParT(_ModuleBase):
    """HLT ParT anchor plus PN-view gated residual logit correction."""

    def __init__(
        self,
        hlt_anchor: HLTPartAnchor | None = None,
        pn_encoder: PNMemoryEncoder | None = None,
        *,
        anchor: HLTPartAnchor | None = None,
        config: Mapping[str, Any] | DualViewResidualParTConfig | None = None,
        reliability_config: Mapping[str, Any] | ReliabilityFeatureConfig | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        if hlt_anchor is None:
            hlt_anchor = anchor
        if hlt_anchor is None:
            raise ValueError("DualViewResidualParT requires an HLTPartAnchor")
        self.hlt_anchor = hlt_anchor
        self.pn_encoder = pn_encoder or build_pn_memory_encoder()
        self.config = DualViewResidualParTConfig.from_mapping(config)
        self.reliability_config = ReliabilityFeatureConfig.from_mapping(reliability_config)
        if int(self.config.pn_context_dim) != int(self.pn_encoder.context_dim):
            raise ValueError(
                f"config pn_context_dim={self.config.pn_context_dim} does not match "
                f"pn_encoder.context_dim={self.pn_encoder.context_dim}"
            )
        if int(self.config.reliability_dim) != int(self.reliability_config.feature_dim):
            raise ValueError(
                f"config reliability_dim={self.config.reliability_dim} does not match "
                f"ReliabilityFeatureConfig.feature_dim={self.reliability_config.feature_dim}"
            )
        if int(self.config.num_classes) != int(self.hlt_anchor.config.num_classes):
            raise ValueError(
                f"config num_classes={self.config.num_classes} does not match "
                f"hlt_anchor num_classes={self.hlt_anchor.config.num_classes}"
            )
        self.delta_mlp = _make_mlp(
            input_dim=int(self.config.fusion_input_dim),
            output_dim=int(self.config.num_classes),
            hidden_dim=int(self.config.hidden_dim),
            num_hidden_layers=int(self.config.num_hidden_layers),
            dropout=float(self.config.dropout),
        )
        self.gate_mlp = _make_mlp(
            input_dim=int(self.config.fusion_input_dim),
            output_dim=int(self.config.gate_dim),
            hidden_dim=int(self.config.hidden_dim),
            num_hidden_layers=int(self.config.num_hidden_layers),
            dropout=float(self.config.dropout),
        )
        self.gate_bias = torch.nn.Parameter(torch.full((int(self.config.gate_dim),), float(self.config.gate_bias_init)))
        self.reset_residual_heads()

    @property
    def output_contract(self) -> str:
        return str(self.config.output_contract)

    @property
    def anchor(self) -> HLTPartAnchor:
        return self.hlt_anchor

    @property
    def fusion_input_dim(self) -> int:
        return int(self.config.fusion_input_dim)

    def reset_residual_heads(self) -> None:
        """Initialize residual path so the model starts as the HLT anchor."""

        torch = require_torch()
        if bool(self.config.zero_initialize_delta):
            final_delta = _last_linear(self.delta_mlp)
            torch.nn.init.zeros_(final_delta.weight)
            torch.nn.init.zeros_(final_delta.bias)
        if bool(self.config.zero_initialize_gate_head):
            final_gate = _last_linear(self.gate_mlp)
            torch.nn.init.zeros_(final_gate.weight)
            torch.nn.init.zeros_(final_gate.bias)
        with torch.no_grad():
            self.gate_bias.fill_(float(self.config.gate_bias_init))

    def train(self, mode: bool = True):
        super().train(mode)
        if all(not bool(param.requires_grad) for param in self.hlt_anchor.parameters()):
            self.hlt_anchor.eval()
        elif self.hlt_anchor.anchor_parameters_frozen():
            self.hlt_anchor.model.eval()
        return self

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def new_module_parameter_count(self) -> int:
        anchor_params = {id(param) for param in self.hlt_anchor.parameters()}
        return int(sum(param.numel() for param in self.parameters() if id(param) not in anchor_params))

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "experiment_step": DUALVIEW_PART_STEP6,
            "output_contract": self.output_contract,
            "config": self.config.to_dict(),
            "reliability_config": self.reliability_config.to_dict(),
            "hlt_anchor": self.hlt_anchor.metadata(),
            "pn_encoder": self.pn_encoder.to_config_dict(),
        }

    def _coerce_context(self, context: Any | None, *, name: str, batch_size: int, dim: int, like: Any):
        torch = require_torch()
        if int(dim) == 0:
            return torch.zeros((int(batch_size), 0), dtype=like.dtype, device=like.device)
        if context is None:
            return torch.zeros((int(batch_size), int(dim)), dtype=like.dtype, device=like.device)
        if not isinstance(context, torch.Tensor):
            context = torch.as_tensor(context, dtype=like.dtype, device=like.device)
        else:
            context = context.to(device=like.device, dtype=like.dtype)
        if context.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, dim], got {tuple(context.shape)}")
        if int(context.shape[0]) != int(batch_size) or int(context.shape[1]) != int(dim):
            raise ValueError(f"{name} shape {tuple(context.shape)} does not match expected {(batch_size, dim)}")
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(context).all():
            raise FloatingPointError(f"{name} contains non-finite values")
        return context

    def _anchor_forward(
        self,
        *,
        hlt_inputs: Mapping[str, Any] | None,
        hlt_tokens: Any | None,
        hlt_mask: Any | None,
        hlt_weights: Any | None,
    ) -> HLTPartAnchorOutput:
        if hlt_inputs is not None:
            return self.hlt_anchor.forward_inputs(hlt_inputs, return_context=True)
        if hlt_tokens is None or hlt_mask is None:
            raise ValueError("forward requires either hlt_inputs or hlt_tokens + hlt_mask")
        return self.hlt_anchor.forward_tokens(hlt_tokens, hlt_mask, weights=hlt_weights, return_context=True)

    def forward_batch(self, batch: Mapping[str, Any], *, return_diagnostics: bool = False) -> DualViewResidualParTOutput:
        """Forward from a ``collate_dualview_part_samples`` batch."""

        return self.forward(batch=batch, return_diagnostics=return_diagnostics)

    def forward(
        self,
        batch: Mapping[str, Any] | None = None,
        *,
        hlt_inputs: Mapping[str, Any] | None = None,
        hlt_tokens: Any | None = None,
        hlt_mask: Any | None = None,
        hlt_weights: Any | None = None,
        pn_reco_tokens: Any | None = None,
        pn_reco_mask: Any | None = None,
        pn_reco_confidence: Any | None = None,
        return_diagnostics: bool = False,
    ) -> DualViewResidualParTOutput:
        torch = require_torch()
        if batch is not None:
            hlt_inputs = hlt_inputs if hlt_inputs is not None else batch.get("hlt_inputs")
            hlt_tokens = hlt_tokens if hlt_tokens is not None else batch.get("hlt_tokens")
            hlt_mask = hlt_mask if hlt_mask is not None else batch.get("hlt_mask")
            hlt_weights = hlt_weights if hlt_weights is not None else batch.get("hlt_weights")
            pn_reco_tokens = pn_reco_tokens if pn_reco_tokens is not None else batch.get("pn_reco_tokens")
            pn_reco_mask = pn_reco_mask if pn_reco_mask is not None else batch.get("pn_reco_mask")
            pn_reco_confidence = (
                pn_reco_confidence
                if pn_reco_confidence is not None
                else batch.get("pn_reco_confidence")
            )
        if pn_reco_tokens is None or pn_reco_mask is None:
            raise ValueError("forward requires pn_reco_tokens and pn_reco_mask")

        hlt_output = self._anchor_forward(
            hlt_inputs=hlt_inputs,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            hlt_weights=hlt_weights,
        )
        hlt_logits = hlt_output.logits.float()
        if hlt_logits.ndim != 2 or int(hlt_logits.shape[1]) != int(self.config.num_classes):
            raise ValueError(f"HLT logits must have shape [batch, {self.config.num_classes}], got {tuple(hlt_logits.shape)}")
        batch_size = int(hlt_logits.shape[0])
        hlt_context = self._coerce_context(
            hlt_output.context,
            name="hlt_context",
            batch_size=batch_size,
            dim=int(self.config.hlt_context_dim),
            like=hlt_logits,
        )

        pn_output = self.pn_encoder(
            pn_reco_tokens,
            pn_reco_mask,
            pn_reco_confidence,
            return_diagnostics=return_diagnostics,
        )
        pn_context = self._coerce_context(
            pn_output.context,
            name="pn_context",
            batch_size=batch_size,
            dim=int(self.config.pn_context_dim),
            like=hlt_logits,
        )

        reliability_output = build_reliability_features(
            hlt_logits=hlt_logits,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            pn_tokens=pn_reco_tokens,
            pn_mask=pn_reco_mask,
            pn_confidence=pn_output.confidence,
            config=self.reliability_config,
            return_diagnostics=return_diagnostics,
        )
        reliability_features = reliability_output.features.to(device=hlt_logits.device, dtype=hlt_logits.dtype)
        if tuple(reliability_features.shape) != (batch_size, int(self.config.reliability_dim)):
            raise ValueError(
                "reliability feature shape must match "
                f"{(batch_size, int(self.config.reliability_dim))}, got {tuple(reliability_features.shape)}"
            )

        fusion_input = torch.cat([hlt_context, pn_context, reliability_features], dim=-1)
        if tuple(fusion_input.shape) != (batch_size, int(self.config.fusion_input_dim)):
            raise ValueError("fusion input shape does not match config")
        delta_logits = self.delta_mlp(fusion_input)
        gate_logits = self.gate_mlp(fusion_input) + self.gate_bias
        gate = torch.sigmoid(gate_logits)
        residual_logits = gate * delta_logits
        logits = hlt_logits + residual_logits
        for name, tensor in (
            ("delta_logits", delta_logits),
            ("gate_logits", gate_logits),
            ("gate", gate),
            ("residual_logits", residual_logits),
            ("logits", logits),
        ):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"DualViewResidualParT produced non-finite {name}")

        diagnostics: dict[str, Any] = {}
        if bool(return_diagnostics):
            changed = (logits.argmax(dim=-1) != hlt_logits.argmax(dim=-1)).float().mean()
            diagnostics = {
                "experiment_step": DUALVIEW_PART_STEP6,
                "output_contract": self.output_contract,
                "batch_size": int(batch_size),
                "fusion_input_dim": int(self.config.fusion_input_dim),
                "gate_mode": str(self.config.gate_mode),
                "gate_shape": list(gate.shape),
                "gate_mean": _tensor_to_float(gate.mean()),
                "gate_min": _tensor_to_float(gate.min()),
                "gate_max": _tensor_to_float(gate.max()),
                "gate_std": _tensor_to_float(gate.float().std(unbiased=False)),
                "delta_abs_mean": _tensor_to_float(delta_logits.detach().abs().mean()),
                "delta_l2_mean": _tensor_to_float(delta_logits.detach().square().sum(dim=-1).sqrt().mean()),
                "residual_abs_mean": _tensor_to_float(residual_logits.detach().abs().mean()),
                "residual_l2_mean": _tensor_to_float(residual_logits.detach().square().sum(dim=-1).sqrt().mean()),
                "prediction_changed_fraction": _tensor_to_float(changed),
                "anchor_frozen": bool(self.hlt_anchor.anchor_parameters_frozen()),
                "hlt_anchor": dict(hlt_output.diagnostics),
                "pn_encoder": dict(pn_output.diagnostics),
                "reliability": dict(reliability_output.diagnostics),
            }
        return DualViewResidualParTOutput(
            logits=logits,
            hlt_logits=hlt_logits,
            delta_logits=delta_logits,
            residual_logits=residual_logits,
            gate=gate,
            gate_logits=gate_logits,
            hlt_context=hlt_context,
            pn_context=pn_context,
            reliability_features=reliability_features,
            hlt_output=hlt_output,
            pn_output=pn_output,
            reliability_output=reliability_output,
            diagnostics=diagnostics,
        )


def build_dualview_residual_part(
    hlt_anchor: HLTPartAnchor | None = None,
    pn_encoder: PNMemoryEncoder | None = None,
    *,
    anchor: HLTPartAnchor | None = None,
    config: Mapping[str, Any] | DualViewResidualParTConfig | None = None,
    reliability_config: Mapping[str, Any] | ReliabilityFeatureConfig | None = None,
    infer_dims_from_modules: bool = True,
    **kwargs: Any,
) -> DualViewResidualParT:
    """Build the Step 6 residual model from an HLT anchor and PN encoder."""

    if hlt_anchor is None:
        hlt_anchor = anchor
    if hlt_anchor is None:
        raise ValueError("build_dualview_residual_part requires an HLTPartAnchor")
    encoder = pn_encoder or build_pn_memory_encoder()
    if kwargs:
        payload = DualViewResidualParTConfig.from_mapping(config).to_dict()
        payload.update(kwargs)
        config = payload
    if bool(infer_dims_from_modules):
        payload = DualViewResidualParTConfig.from_mapping(config).to_dict()
        payload["hlt_context_dim"] = int(hlt_anchor.context_dim)
        payload["pn_context_dim"] = int(encoder.context_dim)
        if reliability_config is not None:
            payload["reliability_dim"] = int(ReliabilityFeatureConfig.from_mapping(reliability_config).feature_dim)
        config = payload
    return DualViewResidualParT(
        hlt_anchor=hlt_anchor,
        pn_encoder=encoder,
        config=config,
        reliability_config=reliability_config,
    )


__all__ = [
    "DUALVIEW_PART_GATE_PER_CLASS",
    "DUALVIEW_PART_GATE_SCALAR",
    "DUALVIEW_PART_RESIDUAL_CONTRACT",
    "DUALVIEW_PART_STEP6",
    "DualViewResidualParT",
    "DualViewResidualParTConfig",
    "DualViewResidualParTOutput",
    "build_dualview_residual_part",
]
