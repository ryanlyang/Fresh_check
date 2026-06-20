"""Set-matching reconstructor adapters.

The set-matching experiment reuses the aggressive reconstructor families for
their architecture-diverse encoders and generation heads, but exposes a new
loss-facing contract:

``predicted_features, existence_logits, candidate_mask, diagnostics``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch

from teacher_logit_reco.reconstructor_builders import (
    AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR,
    build_teacher_logit_reconstructor,
    normalize_reconstructor_architecture,
    strip_compile_prefix_from_state_dict,
)
from teacher_logit_reco.views import SoftReconstructedView

from .experiment import (
    SET_RECONSTRUCTOR_ARCHITECTURES,
    normalize_set_reconstructor_architecture,
)


SET_MATCHING_RECONSTRUCTOR_STEP = "set_matching_multiview_step4_reconstructor"
SET_MATCHING_RECONSTRUCTOR_CONTRACT = (
    "predicted_features_existence_logits_candidate_mask_diagnostics"
)
DEFAULT_WEIGHT_LOGIT_EPSILON = 1.0e-4

if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    import torch as _torch

    _ModuleBase = _torch.nn.Module

SET_MATCHING_TO_AGGRESSIVE_ARCHITECTURE = {
    "gt": AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    "pn": AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR,
    "pfn": AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR,
    "pcnn": AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR,
}

AGGRESSIVE_TO_SET_MATCHING_ARCHITECTURE = {
    value: key for key, value in SET_MATCHING_TO_AGGRESSIVE_ARCHITECTURE.items()
}


def _maybe_torch_tensor(value: Any) -> bool:
    torch = require_torch()
    return isinstance(value, torch.Tensor)


def _as_tensor(value: Any, *, device=None, dtype=None):
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(value, device=device, dtype=dtype)


def _finite_stat(value, *, mask=None, reducer: str = "mean") -> float:
    torch = require_torch()
    tensor = value.detach()
    if mask is not None:
        tensor = tensor[mask.detach().bool()]
    if int(tensor.numel()) == 0:
        return 0.0
    tensor = tensor.float()
    if reducer == "mean":
        return float(tensor.mean().cpu().item())
    if reducer == "min":
        return float(tensor.min().cpu().item())
    if reducer == "max":
        return float(tensor.max().cpu().item())
    raise ValueError(f"Unknown reducer {reducer!r}")


def _jsonable_scalar_dict(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    payload: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[str(key)] = value
        elif isinstance(value, np.generic):
            payload[str(key)] = value.item()
        elif _maybe_torch_tensor(value) and int(value.numel()) == 1:
            payload[str(key)] = float(value.detach().cpu().item())
    return payload


def _logit_from_probability(probability, *, eps: float):
    torch = require_torch()
    clipped = torch.clamp(probability.float(), min=float(eps), max=1.0 - float(eps))
    return torch.log(clipped) - torch.log1p(-clipped)


@dataclass(frozen=True)
class SetMatchingReconstructorConfig:
    """Wrapper config for an aggressive reconstructor used with set matching."""

    architecture: str = "gt"
    model_config: dict[str, Any] = field(default_factory=dict)
    weight_logit_epsilon: float = DEFAULT_WEIGHT_LOGIT_EPSILON
    output_weight_threshold: float = 0.0
    output_contract: str = SET_MATCHING_RECONSTRUCTOR_CONTRACT
    experiment_step: str = SET_MATCHING_RECONSTRUCTOR_STEP

    def __post_init__(self) -> None:
        arch = normalize_set_matching_reconstructor_architecture(self.architecture)
        if float(self.weight_logit_epsilon) <= 0.0 or float(self.weight_logit_epsilon) >= 0.5:
            raise ValueError("weight_logit_epsilon must be in (0, 0.5)")
        if float(self.output_weight_threshold) < 0.0 or float(self.output_weight_threshold) > 1.0:
            raise ValueError("output_weight_threshold must be in [0, 1]")
        object.__setattr__(self, "architecture", arch)
        object.__setattr__(self, "model_config", dict(self.model_config or {}))

    @property
    def aggressive_architecture(self) -> str:
        return SET_MATCHING_TO_AGGRESSIVE_ARCHITECTURE[self.architecture]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["aggressive_architecture"] = self.aggressive_architecture
        payload["set_matching_architecture"] = self.architecture
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "SetMatchingReconstructorConfig" | None,
        *,
        architecture: str | None = None,
    ) -> "SetMatchingReconstructorConfig":
        if isinstance(value, cls):
            if architecture is None:
                return value
            payload = value.to_dict()
        else:
            payload = dict(value or {})

        arch = architecture
        if arch is None:
            arch = payload.pop("set_matching_architecture", None)
        if arch is None:
            arch = payload.pop("architecture", None)
        if arch is None:
            aggressive = payload.pop("aggressive_architecture", None)
            if aggressive is None:
                aggressive = payload.get("reconstructor_architecture")
            if aggressive is not None:
                try:
                    normalized_aggressive = normalize_reconstructor_architecture(str(aggressive))
                except ValueError:
                    normalized_aggressive = str(aggressive)
                arch = AGGRESSIVE_TO_SET_MATCHING_ARCHITECTURE.get(normalized_aggressive)
        if arch is None:
            arch = "gt"
        payload.pop("set_matching_architecture", None)
        payload.pop("architecture", None)
        payload.pop("aggressive_architecture", None)
        payload.pop("reconstructor_architecture", None)

        model_config = payload.pop("model_config", None)
        wrapper_keys = {
            "weight_logit_epsilon",
            "output_weight_threshold",
            "output_contract",
            "experiment_step",
        }
        wrapper_values = {key: payload.pop(key) for key in list(payload) if key in wrapper_keys}
        if model_config is None:
            model_config = dict(payload)
        elif payload:
            nested = dict(model_config)
            nested.update(payload)
            model_config = nested
        return cls(architecture=str(arch), model_config=dict(model_config or {}), **wrapper_values)


@dataclass
class SetMatchingReconstructorOutput:
    """Loss-facing set-matching reconstructor output."""

    predicted_features: Any
    existence_logits: Any
    candidate_mask: Any
    diagnostics: dict[str, Any]
    soft_view: SoftReconstructedView | None = None
    candidate_weights: Any | None = None

    def to_loss_kwargs(self, *, offline_features, offline_mask, hlt_features=None, hlt_mask=None) -> dict[str, Any]:
        payload = {
            "predicted_features": self.predicted_features,
            "existence_logits": self.existence_logits,
            "candidate_mask": self.candidate_mask,
            "offline_features": offline_features,
            "offline_mask": offline_mask,
        }
        if hlt_features is not None:
            payload["hlt_features"] = hlt_features
        if hlt_mask is not None:
            payload["hlt_mask"] = hlt_mask
        return payload

    def shape_report(self) -> dict[str, Any]:
        return {
            "predicted_features_shape": list(self.predicted_features.shape),
            "existence_logits_shape": list(self.existence_logits.shape),
            "candidate_mask_shape": list(self.candidate_mask.shape),
        }


def normalize_set_matching_reconstructor_architecture(architecture: str | None) -> str:
    """Normalize set-matching reconstructor architecture aliases."""

    if architecture is None:
        return "gt"
    key = str(architecture).strip().lower().replace("-", "_").replace(" ", "_")
    if key in SET_RECONSTRUCTOR_ARCHITECTURES:
        return normalize_set_reconstructor_architecture(key)
    if key in AGGRESSIVE_TO_SET_MATCHING_ARCHITECTURE:
        return AGGRESSIVE_TO_SET_MATCHING_ARCHITECTURE[key]
    aliases = {
        "global_transformer": "gt",
        "transformer": "gt",
        "part": "gt",
        "particle_transformer": "gt",
        "particle_net": "pn",
        "particle_flow": "pfn",
        "pf": "pfn",
        "deep_sets": "pfn",
        "deepsets": "pfn",
        "particle_cnn": "pcnn",
        "p_cnn": "pcnn",
        "particle_conv": "pcnn",
    }
    if key in aliases:
        return aliases[key]
    compact = key.replace("_", "")
    compact_aliases = {alias.replace("_", ""): value for alias, value in aliases.items()}
    compact_aliases.update({arch.replace("_", ""): short for arch, short in AGGRESSIVE_TO_SET_MATCHING_ARCHITECTURE.items()})
    if compact in compact_aliases:
        return compact_aliases[compact]
    expected = ", ".join(SET_RECONSTRUCTOR_ARCHITECTURES)
    raise ValueError(f"Unknown set-matching reconstructor architecture {architecture!r}; expected one of: {expected}")


def set_matching_output_from_soft_view(
    view: SoftReconstructedView,
    *,
    architecture: str,
    weight_logit_epsilon: float = DEFAULT_WEIGHT_LOGIT_EPSILON,
    output_weight_threshold: float = 0.0,
) -> SetMatchingReconstructorOutput:
    """Convert an aggressive soft view to the set-matching loss contract."""

    torch = require_torch()
    predicted_features = _as_tensor(view.tokens).float()
    candidate_weights = _as_tensor(view.weights, device=predicted_features.device, dtype=predicted_features.dtype)
    candidate_mask = _as_tensor(view.mask, device=predicted_features.device, dtype=torch.bool)
    if float(output_weight_threshold) > 0.0:
        candidate_mask = candidate_mask & (candidate_weights >= float(output_weight_threshold))
    existence_logits = _logit_from_probability(candidate_weights, eps=float(weight_logit_epsilon))
    existence_logits = torch.where(candidate_mask, existence_logits, torch.zeros_like(existence_logits))
    predicted_features = torch.where(candidate_mask[:, :, None], predicted_features, torch.zeros_like(predicted_features))

    metadata_diagnostics = {}
    if isinstance(view.metadata, Mapping):
        metadata_diagnostics.update(_jsonable_scalar_dict(view.metadata.get("diagnostics")))
    if isinstance(view.aux, Mapping):
        metadata_diagnostics.update(_jsonable_scalar_dict(view.aux.get("diagnostics")))

    diagnostics = {
        "architecture": normalize_set_matching_reconstructor_architecture(architecture),
        "output_contract": SET_MATCHING_RECONSTRUCTOR_CONTRACT,
        "n_jets": int(predicted_features.shape[0]),
        "n_candidates": int(predicted_features.shape[1]),
        "feature_dim": int(predicted_features.shape[2]),
        "candidate_count_mean": _finite_stat(candidate_mask.float(), reducer="mean") * float(candidate_mask.shape[1]),
        "candidate_weight_mean": _finite_stat(candidate_weights, mask=candidate_mask, reducer="mean"),
        "candidate_weight_min": _finite_stat(candidate_weights, mask=candidate_mask, reducer="min"),
        "candidate_weight_max": _finite_stat(candidate_weights, mask=candidate_mask, reducer="max"),
        "existence_logit_mean": _finite_stat(existence_logits, mask=candidate_mask, reducer="mean"),
        "predicted_pt_mean": _finite_stat(predicted_features[:, :, 0], mask=candidate_mask, reducer="mean"),
        "predicted_energy_mean": _finite_stat(predicted_features[:, :, 3], mask=candidate_mask, reducer="mean"),
        **metadata_diagnostics,
    }
    return SetMatchingReconstructorOutput(
        predicted_features=predicted_features,
        existence_logits=existence_logits,
        candidate_mask=candidate_mask,
        diagnostics=diagnostics,
        soft_view=view,
        candidate_weights=candidate_weights,
    )


class SetMatchingReconstructorAdapter(_ModuleBase):
    """Thin wrapper adapting aggressive reconstructors to set-matching losses."""

    def __init__(self, config: Mapping[str, Any] | SetMatchingReconstructorConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = SetMatchingReconstructorConfig.from_mapping(config)
        self.base_model = build_teacher_logit_reconstructor(
            self.config.aggressive_architecture,
            self.config.model_config,
        )
        self._torch = torch

    @property
    def architecture(self) -> str:
        return self.config.architecture

    @property
    def aggressive_architecture(self) -> str:
        return self.config.aggressive_architecture

    def forward(self, hlt_tokens, hlt_mask, *, labels=None, jet_ids=None, split: str = "in_memory"):
        view = self.base_model(hlt_tokens, hlt_mask, labels=labels, jet_ids=jet_ids, split=split)
        return set_matching_output_from_soft_view(
            view,
            architecture=self.architecture,
            weight_logit_epsilon=float(self.config.weight_logit_epsilon),
            output_weight_threshold=float(self.config.output_weight_threshold),
        )

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        base_config = getattr(self.base_model, "config", None)
        if hasattr(base_config, "to_dict"):
            payload["model_config"] = base_config.to_dict()
        return payload


def build_set_matching_reconstructor(
    architecture: str | None = None,
    config: Mapping[str, Any] | SetMatchingReconstructorConfig | None = None,
) -> SetMatchingReconstructorAdapter:
    """Build a set-matching reconstructor adapter for one of gt/pn/pfn/pcnn."""

    wrapper_config = SetMatchingReconstructorConfig.from_mapping(config, architecture=architecture)
    return SetMatchingReconstructorAdapter(wrapper_config)


def set_matching_reconstructor_checkpoint_payload(
    model: SetMatchingReconstructorAdapter,
    *,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a checkpoint payload for a set-matching reconstructor."""

    if not isinstance(model, SetMatchingReconstructorAdapter):
        raise TypeError("model must be a SetMatchingReconstructorAdapter")
    payload = {
        "experiment_step": SET_MATCHING_RECONSTRUCTOR_STEP,
        "output_contract": SET_MATCHING_RECONSTRUCTOR_CONTRACT,
        "set_matching_architecture": model.architecture,
        "aggressive_architecture": model.aggressive_architecture,
        "model_config": model.to_config_dict(),
        "model_state_dict": model.state_dict(),
    }
    payload.update(dict(extra_payload or {}))
    return payload


def save_set_matching_reconstructor_checkpoint(
    path: str | Path,
    model: SetMatchingReconstructorAdapter,
    *,
    extra_payload: Mapping[str, Any] | None = None,
) -> Path:
    """Save a set-matching reconstructor checkpoint."""

    torch = require_torch()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(set_matching_reconstructor_checkpoint_payload(model, extra_payload=extra_payload), checkpoint_path)
    return checkpoint_path


def load_set_matching_reconstructor_checkpoint(
    path: str | Path,
    *,
    device="cpu",
    strict: bool = True,
    expected_architecture: str | None = None,
) -> tuple[SetMatchingReconstructorAdapter, dict[str, Any]]:
    """Load a set-matching reconstructor checkpoint."""

    torch = require_torch()
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Set-matching reconstructor checkpoint must be a mapping: {checkpoint_path}")
    if "model_state_dict" not in payload:
        raise KeyError(f"Set-matching reconstructor checkpoint is missing model_state_dict: {checkpoint_path}")

    config_payload = payload.get("model_config") or payload
    architecture = payload.get("set_matching_architecture") or payload.get("architecture")
    config = SetMatchingReconstructorConfig.from_mapping(config_payload, architecture=architecture)
    if expected_architecture is not None:
        expected = normalize_set_matching_reconstructor_architecture(expected_architecture)
        if config.architecture != expected:
            raise ValueError(
                f"Checkpoint architecture mismatch for {checkpoint_path}: expected {expected}, found {config.architecture}"
            )
    model = build_set_matching_reconstructor(config=config)
    model.load_state_dict(strip_compile_prefix_from_state_dict(payload["model_state_dict"]), strict=bool(strict))
    model = model.to(device)
    model.eval()
    return model, dict(payload)


__all__ = [
    "AGGRESSIVE_TO_SET_MATCHING_ARCHITECTURE",
    "DEFAULT_WEIGHT_LOGIT_EPSILON",
    "SET_MATCHING_RECONSTRUCTOR_CONTRACT",
    "SET_MATCHING_RECONSTRUCTOR_STEP",
    "SET_MATCHING_TO_AGGRESSIVE_ARCHITECTURE",
    "SetMatchingReconstructorAdapter",
    "SetMatchingReconstructorConfig",
    "SetMatchingReconstructorOutput",
    "build_set_matching_reconstructor",
    "load_set_matching_reconstructor_checkpoint",
    "normalize_set_matching_reconstructor_architecture",
    "save_set_matching_reconstructor_checkpoint",
    "set_matching_output_from_soft_view",
    "set_matching_reconstructor_checkpoint_payload",
]
