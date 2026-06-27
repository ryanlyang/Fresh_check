"""Step 5 reliability features for the dual-view ParT gate.

These features are deliberately small and explicit.  They are not a replacement
for learned HLT/PN contexts; they give the residual gate easy-to-learn signals
about HLT certainty, PN reconstructed-view confidence, and HLT-vs-PN geometric
agreement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import MAX_CONSTITUENTS, RAW_TOKEN_DIM


DUALVIEW_PART_STEP5 = "reliability_gated_dualview_part_step5_reliability_features"
DUALVIEW_PART_RELIABILITY_CONTRACT = "hlt_pn_reliability_feature_vector_v1"

DUALVIEW_PART_RELIABILITY_FEATURE_NAMES: tuple[str, ...] = (
    "hlt_top1_prob",
    "hlt_top2_prob",
    "hlt_margin",
    "hlt_entropy_norm",
    "hlt_count_norm",
    "pn_count_norm",
    "pn_conf_count_norm",
    "pn_count_minus_hlt_norm",
    "pn_count_ratio",
    "pn_conf_mean",
    "pn_conf_max",
    "pn_conf_std",
    "pn_low_conf_frac",
    "hlt_log_pt_norm",
    "pn_log_pt_norm",
    "jet_log_pt_abs_diff",
    "jet_eta_abs_diff_norm",
    "jet_phi_abs_diff_norm",
    "jet_log_energy_abs_diff",
    "pn_to_hlt_delta_r_mean_norm",
    "pn_to_hlt_delta_r_max_norm",
    "hlt_to_pn_delta_r_mean_norm",
)


if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True


def _positive_int(value: int, *, field_name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


@dataclass(frozen=True)
class ReliabilityFeatureConfig:
    """Configuration for the compact reliability feature builder."""

    raw_token_dim: int = RAW_TOKEN_DIM
    max_constituents: int = MAX_CONSTITUENTS
    pt_index: int = 0
    eta_index: int = 1
    phi_index: int = 2
    energy_index: int = 3
    low_confidence_threshold: float = 0.25
    eta_scale: float = 5.0
    log_kinematic_scale: float = 10.0
    nearest_neighbor_delta_r_scale: float = 1.0
    output_contract: str = DUALVIEW_PART_RELIABILITY_CONTRACT
    experiment_step: str = DUALVIEW_PART_STEP5

    def __post_init__(self) -> None:
        raw_token_dim = _positive_int(self.raw_token_dim, field_name="raw_token_dim")
        max_constituents = _positive_int(self.max_constituents, field_name="max_constituents")
        for field_name in ("pt_index", "eta_index", "phi_index", "energy_index"):
            index = int(getattr(self, field_name))
            if index < 0 or index >= raw_token_dim:
                raise ValueError(f"{field_name}={index} is outside raw_token_dim={raw_token_dim}")
            object.__setattr__(self, field_name, index)
        low_confidence_threshold = float(self.low_confidence_threshold)
        if low_confidence_threshold < 0.0 or low_confidence_threshold > 1.0:
            raise ValueError("low_confidence_threshold must be in [0, 1]")
        eta_scale = float(self.eta_scale)
        log_kinematic_scale = float(self.log_kinematic_scale)
        nearest_neighbor_delta_r_scale = float(self.nearest_neighbor_delta_r_scale)
        if eta_scale <= 0.0:
            raise ValueError("eta_scale must be positive")
        if log_kinematic_scale <= 0.0:
            raise ValueError("log_kinematic_scale must be positive")
        if nearest_neighbor_delta_r_scale <= 0.0:
            raise ValueError("nearest_neighbor_delta_r_scale must be positive")
        object.__setattr__(self, "raw_token_dim", raw_token_dim)
        object.__setattr__(self, "max_constituents", max_constituents)
        object.__setattr__(self, "low_confidence_threshold", low_confidence_threshold)
        object.__setattr__(self, "eta_scale", eta_scale)
        object.__setattr__(self, "log_kinematic_scale", log_kinematic_scale)
        object.__setattr__(self, "nearest_neighbor_delta_r_scale", nearest_neighbor_delta_r_scale)

    @property
    def feature_dim(self) -> int:
        return len(DUALVIEW_PART_RELIABILITY_FEATURE_NAMES)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "ReliabilityFeatureConfig" | None,
    ) -> "ReliabilityFeatureConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = dict(value)
        payload.pop("output_contract", None)
        payload.pop("experiment_step", None)
        payload.pop("feature_names", None)
        payload.pop("feature_dim", None)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(DUALVIEW_PART_RELIABILITY_FEATURE_NAMES)
        payload["feature_dim"] = self.feature_dim
        return payload


@dataclass
class ReliabilityFeatureOutput:
    """Feature tensor and lightweight metadata for reliability-gated fusion."""

    features: Any
    feature_names: tuple[str, ...] = DUALVIEW_PART_RELIABILITY_FEATURE_NAMES
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "feature_names": list(self.feature_names),
            "feature_dim": self.feature_dim,
            "diagnostics": dict(self.diagnostics),
        }


def _as_float_tensor(value: Any, *, name: str, device=None):
    torch = require_torch()
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device) if device is not None else value
        return tensor.float()
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_bool_mask(value: Any, *, name: str, device=None):
    torch = require_torch()
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        mask = value.to(device=device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(value, dtype=torch.bool, device=device)
    if mask.ndim == 3 and int(mask.shape[1]) == 1:
        mask = mask[:, 0, :]
    if mask.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, tokens] or [batch, 1, tokens], got {tuple(mask.shape)}")
    return mask


def _wrap_phi_torch(value: Any):
    torch = require_torch()
    return torch.atan2(torch.sin(value), torch.cos(value))


def _empty_jet_summary(batch_size: int, *, device, dtype):
    torch = require_torch()
    return {
        "pt": torch.zeros((batch_size,), dtype=dtype, device=device),
        "eta": torch.zeros((batch_size,), dtype=dtype, device=device),
        "phi": torch.zeros((batch_size,), dtype=dtype, device=device),
        "energy": torch.zeros((batch_size,), dtype=dtype, device=device),
    }


def _default_mask_from_tokens(tokens, *, config: ReliabilityFeatureConfig):
    pt = tokens[:, :, int(config.pt_index)]
    energy = tokens[:, :, int(config.energy_index)]
    return (pt > 0.0) | (energy > 0.0)


def _prepare_tokens_mask_confidence(tokens, mask, confidence, *, batch_size: int, config: ReliabilityFeatureConfig, device, dtype):
    torch = require_torch()
    tokens = _as_float_tensor(tokens, name="tokens", device=device)
    if tokens is None:
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
        empty_tokens = torch.zeros((batch_size, 0, int(config.raw_token_dim)), dtype=dtype, device=device)
        empty_confidence = torch.zeros((batch_size, 0), dtype=dtype, device=device)
        return empty_tokens, empty_mask, empty_confidence
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch, tokens, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[0]) != int(batch_size):
        raise ValueError("tokens batch dimension must match hlt_logits")
    if int(tokens.shape[-1]) != int(config.raw_token_dim):
        raise ValueError(f"expected raw_token_dim={config.raw_token_dim}, got {tokens.shape[-1]}")
    tokens = torch.nan_to_num(tokens.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
    mask = _as_bool_mask(mask, name="mask", device=device)
    if mask is None:
        mask = _default_mask_from_tokens(tokens, config=config)
    if tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError("mask shape must match tokens batch/token dimensions")
    confidence = _as_float_tensor(confidence, name="confidence", device=device)
    if confidence is None:
        confidence = mask.float()
    if confidence.ndim == 3 and int(confidence.shape[1]) == 1:
        confidence = confidence[:, 0, :]
    if tuple(confidence.shape) != tuple(tokens.shape[:2]):
        raise ValueError("confidence shape must match tokens batch/token dimensions")
    confidence = torch.where(mask, confidence.to(dtype=dtype).clamp(0.0, 1.0), torch.zeros_like(confidence, dtype=dtype))
    return tokens, mask, confidence


def _jet_summary(tokens, mask, confidence, *, config: ReliabilityFeatureConfig):
    torch = require_torch()
    batch_size = int(tokens.shape[0])
    if int(tokens.shape[1]) == 0:
        return _empty_jet_summary(batch_size, device=tokens.device, dtype=tokens.dtype)
    weights = (mask.float() * confidence.float()).to(dtype=tokens.dtype)
    pt = tokens[:, :, int(config.pt_index)].clamp_min(0.0) * weights
    eta = tokens[:, :, int(config.eta_index)]
    phi = tokens[:, :, int(config.phi_index)]
    energy = tokens[:, :, int(config.energy_index)].clamp_min(0.0) * weights
    px = (pt * torch.cos(phi)).sum(dim=1)
    py = (pt * torch.sin(phi)).sum(dim=1)
    pz = (pt * torch.sinh(eta.clamp(-10.0, 10.0))).sum(dim=1)
    jet_pt = torch.hypot(px, py)
    jet_phi = torch.atan2(py, px)
    jet_eta = torch.asinh(pz / jet_pt.clamp_min(1.0e-6))
    jet_eta = torch.where(jet_pt > 1.0e-6, jet_eta, torch.zeros_like(jet_eta))
    jet_phi = torch.where(jet_pt > 1.0e-6, jet_phi, torch.zeros_like(jet_phi))
    return {
        "pt": jet_pt,
        "eta": jet_eta,
        "phi": jet_phi,
        "energy": energy.sum(dim=1),
    }


def _normalized_log1p(value, *, scale: float):
    return value.clamp_min(0.0).log1p() / float(scale)


def _masked_mean(value, mask, weights=None):
    torch = require_torch()
    if value.shape[1] == 0:
        return torch.zeros((value.shape[0],), dtype=value.dtype, device=value.device)
    mask_float = mask.float().to(dtype=value.dtype)
    if weights is not None:
        mask_float = mask_float * weights.to(dtype=value.dtype)
    denom = mask_float.sum(dim=1).clamp_min(1.0)
    return (value * mask_float).sum(dim=1) / denom


def _masked_max(value, mask):
    torch = require_torch()
    if value.shape[1] == 0:
        return torch.zeros((value.shape[0],), dtype=value.dtype, device=value.device)
    masked = torch.where(mask, value, torch.zeros_like(value))
    return masked.max(dim=1).values


def _nearest_neighbor_features(hlt_tokens, hlt_mask, pn_tokens, pn_mask, pn_confidence, *, config: ReliabilityFeatureConfig):
    torch = require_torch()
    batch_size = int(hlt_mask.shape[0])
    dtype = hlt_tokens.dtype
    device = hlt_tokens.device
    zeros = torch.zeros((batch_size,), dtype=dtype, device=device)
    if int(hlt_tokens.shape[1]) == 0 or int(pn_tokens.shape[1]) == 0:
        return zeros, zeros, zeros
    hlt_any = hlt_mask.any(dim=1)
    pn_any = pn_mask.any(dim=1)
    has_pairs = hlt_any & pn_any
    if not bool(has_pairs.any()):
        return zeros, zeros, zeros

    hlt_eta = hlt_tokens[:, :, int(config.eta_index)]
    hlt_phi = hlt_tokens[:, :, int(config.phi_index)]
    pn_eta = pn_tokens[:, :, int(config.eta_index)]
    pn_phi = pn_tokens[:, :, int(config.phi_index)]
    deta = pn_eta[:, :, None] - hlt_eta[:, None, :]
    dphi = _wrap_phi_torch(pn_phi[:, :, None] - hlt_phi[:, None, :])
    delta_r = torch.sqrt((deta * deta + dphi * dphi).clamp_min(0.0))
    pair_mask = pn_mask[:, :, None] & hlt_mask[:, None, :]
    large = torch.full_like(delta_r, 1.0e6)
    masked_delta_r = torch.where(pair_mask, delta_r, large)
    pn_min = masked_delta_r.min(dim=2).values
    hlt_min = masked_delta_r.min(dim=1).values

    pn_valid = pn_mask & hlt_any[:, None]
    hlt_valid = hlt_mask & pn_any[:, None]
    pn_min = torch.where(pn_valid, pn_min, torch.zeros_like(pn_min))
    hlt_min = torch.where(hlt_valid, hlt_min, torch.zeros_like(hlt_min))
    pn_mean = _masked_mean(pn_min, pn_valid, weights=pn_confidence)
    pn_max = _masked_max(pn_min, pn_valid)
    hlt_mean = _masked_mean(hlt_min, hlt_valid)
    scale = float(config.nearest_neighbor_delta_r_scale)
    return (
        (pn_mean / scale).clamp(0.0, 5.0),
        (pn_max / scale).clamp(0.0, 5.0),
        (hlt_mean / scale).clamp(0.0, 5.0),
    )


def build_reliability_features(
    *,
    hlt_logits: Any,
    hlt_tokens: Any | None = None,
    hlt_mask: Any | None = None,
    pn_tokens: Any | None = None,
    pn_mask: Any | None = None,
    pn_confidence: Any | None = None,
    config: Mapping[str, Any] | ReliabilityFeatureConfig | None = None,
    return_diagnostics: bool = False,
) -> ReliabilityFeatureOutput:
    """Build compact reliability features for the Step 6 gate MLP.

    All returned rows have the fixed order in
    ``DUALVIEW_PART_RELIABILITY_FEATURE_NAMES``.
    """

    torch = require_torch()
    cfg = ReliabilityFeatureConfig.from_mapping(config)
    logits = _as_float_tensor(hlt_logits, name="hlt_logits")
    if logits is None or logits.ndim != 2:
        raise ValueError(f"hlt_logits must have shape [batch, classes], got {None if logits is None else tuple(logits.shape)}")
    batch_size = int(logits.shape[0])
    device = logits.device
    dtype = logits.dtype
    num_classes = int(logits.shape[1])
    if num_classes <= 1:
        raise ValueError("hlt_logits must contain at least two classes")

    probs = torch.softmax(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    hlt_top1 = top2[:, 0]
    hlt_top2 = top2[:, 1]
    hlt_margin = hlt_top1 - hlt_top2
    hlt_entropy = -(probs.clamp_min(1.0e-8) * probs.clamp_min(1.0e-8).log()).sum(dim=-1)
    hlt_entropy_norm = hlt_entropy / math.log(float(num_classes))

    hlt_tokens, hlt_mask, hlt_confidence = _prepare_tokens_mask_confidence(
        hlt_tokens,
        hlt_mask,
        None,
        batch_size=batch_size,
        config=cfg,
        device=device,
        dtype=dtype,
    )
    pn_tokens, pn_mask, pn_confidence = _prepare_tokens_mask_confidence(
        pn_tokens,
        pn_mask,
        pn_confidence,
        batch_size=batch_size,
        config=cfg,
        device=device,
        dtype=dtype,
    )

    hlt_count = hlt_mask.sum(dim=1).to(dtype=dtype)
    pn_count = pn_mask.sum(dim=1).to(dtype=dtype)
    max_count = float(cfg.max_constituents)
    log_count_scale = math.log1p(max_count)
    hlt_count_norm = torch.log1p(hlt_count) / log_count_scale
    pn_count_norm = torch.log1p(pn_count) / log_count_scale
    pn_conf_count = pn_confidence.sum(dim=1)
    pn_conf_count_norm = torch.log1p(pn_conf_count) / log_count_scale
    pn_count_minus_hlt_norm = (pn_count - hlt_count) / max_count
    pn_count_ratio = pn_count / (hlt_count + pn_count).clamp_min(1.0)

    pn_count_denom = pn_count.clamp_min(1.0)
    pn_conf_mean = pn_confidence.sum(dim=1) / pn_count_denom
    pn_conf_max = _masked_max(pn_confidence, pn_mask)
    pn_conf_centered = torch.where(pn_mask, pn_confidence - pn_conf_mean[:, None], torch.zeros_like(pn_confidence))
    pn_conf_std = torch.sqrt((pn_conf_centered.square().sum(dim=1) / pn_count_denom).clamp_min(0.0))
    pn_low_conf = ((pn_confidence < float(cfg.low_confidence_threshold)) & pn_mask).sum(dim=1).to(dtype=dtype) / pn_count_denom

    hlt_summary = _jet_summary(hlt_tokens, hlt_mask, hlt_confidence, config=cfg)
    pn_summary = _jet_summary(pn_tokens, pn_mask, pn_confidence, config=cfg)
    hlt_log_pt_norm = _normalized_log1p(hlt_summary["pt"], scale=float(cfg.log_kinematic_scale))
    pn_log_pt_norm = _normalized_log1p(pn_summary["pt"], scale=float(cfg.log_kinematic_scale))
    jet_log_pt_abs_diff = (torch.log1p(hlt_summary["pt"].clamp_min(0.0)) - torch.log1p(pn_summary["pt"].clamp_min(0.0))).abs()
    jet_log_pt_abs_diff = jet_log_pt_abs_diff / float(cfg.log_kinematic_scale)
    jet_eta_abs_diff_norm = (hlt_summary["eta"] - pn_summary["eta"]).abs().clamp(max=float(cfg.eta_scale)) / float(cfg.eta_scale)
    jet_phi_abs_diff_norm = _wrap_phi_torch(hlt_summary["phi"] - pn_summary["phi"]).abs() / math.pi
    jet_log_energy_abs_diff = (
        torch.log1p(hlt_summary["energy"].clamp_min(0.0))
        - torch.log1p(pn_summary["energy"].clamp_min(0.0))
    ).abs() / float(cfg.log_kinematic_scale)

    pn_to_hlt_mean, pn_to_hlt_max, hlt_to_pn_mean = _nearest_neighbor_features(
        hlt_tokens,
        hlt_mask,
        pn_tokens,
        pn_mask,
        pn_confidence,
        config=cfg,
    )

    features = torch.stack(
        [
            hlt_top1,
            hlt_top2,
            hlt_margin,
            hlt_entropy_norm,
            hlt_count_norm,
            pn_count_norm,
            pn_conf_count_norm,
            pn_count_minus_hlt_norm,
            pn_count_ratio,
            pn_conf_mean,
            pn_conf_max,
            pn_conf_std,
            pn_low_conf,
            hlt_log_pt_norm,
            pn_log_pt_norm,
            jet_log_pt_abs_diff,
            jet_eta_abs_diff_norm,
            jet_phi_abs_diff_norm,
            jet_log_energy_abs_diff,
            pn_to_hlt_mean,
            pn_to_hlt_max,
            hlt_to_pn_mean,
        ],
        dim=1,
    )
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(features).all():
        raise FloatingPointError("Reliability feature builder produced non-finite features")

    diagnostics = {}
    if bool(return_diagnostics):
        diagnostics = {
            "experiment_step": DUALVIEW_PART_STEP5,
            "output_contract": DUALVIEW_PART_RELIABILITY_CONTRACT,
            "feature_dim": int(features.shape[1]),
            "batch_size": int(batch_size),
            "hlt_confidence_mean": float(hlt_top1.detach().mean().cpu().item()),
            "hlt_entropy_mean": float(hlt_entropy_norm.detach().mean().cpu().item()),
            "hlt_count_mean": float(hlt_count.detach().mean().cpu().item()),
            "pn_count_mean": float(pn_count.detach().mean().cpu().item()),
            "pn_confidence_mean": float(pn_conf_mean.detach().mean().cpu().item()),
            "jet_log_pt_abs_diff_mean": float(jet_log_pt_abs_diff.detach().mean().cpu().item()),
            "pn_to_hlt_delta_r_mean": float(pn_to_hlt_mean.detach().mean().cpu().item()),
        }

    return ReliabilityFeatureOutput(
        features=features,
        feature_names=DUALVIEW_PART_RELIABILITY_FEATURE_NAMES,
        diagnostics=diagnostics,
    )


def reliability_feature_dim() -> int:
    return len(DUALVIEW_PART_RELIABILITY_FEATURE_NAMES)


__all__ = [
    "DUALVIEW_PART_RELIABILITY_CONTRACT",
    "DUALVIEW_PART_RELIABILITY_FEATURE_NAMES",
    "DUALVIEW_PART_STEP5",
    "ReliabilityFeatureConfig",
    "ReliabilityFeatureOutput",
    "build_reliability_features",
    "reliability_feature_dim",
]
