"""Hungarian losses for DETR/free-slot particle reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from .features import (
    DetrSlotFeatureConfig,
    raw_to_aux_features,
    raw_to_core_features,
    wrapped_phi_difference,
)


DETR_SLOT_HUNGARIAN_LOSS_STEP = "detr_free_slot_step10_hungarian_loss"
DETR_SLOT_DEFAULT_CORE_WEIGHTS: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5)
DETR_SLOT_CORE_FEATURE_NAMES: tuple[str, str, str, str] = ("log_pt", "eta", "phi", "log_energy")


def _maybe_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def require_torch():
    torch = _maybe_torch()
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("DETR slot Hungarian losses require PyTorch")
    return torch


def _functional():
    return require_torch().nn.functional


def _as_float_tensor(value, *, name: str, device=None, dtype=None):
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype or torch.float32)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return tensor


def _as_bool_mask(value, *, name: str, expected_shape: tuple[int, int], device=None):
    torch = require_torch()
    if value is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=device)
    tensor = value.to(device=device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    tensor = tensor.bool()
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor


def _validate_3d(name: str, value) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, particles, features], got {tuple(value.shape)}")
    if int(value.shape[0]) <= 0 or int(value.shape[1]) <= 0 or int(value.shape[2]) <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {tuple(value.shape)}")


def _smooth_l1_from_diff(diff, *, beta: float):
    functional = _functional()
    return functional.smooth_l1_loss(diff, require_torch().zeros_like(diff), reduction="none", beta=float(beta))


def _zero_like(reference):
    return reference.sum() * 0.0


def _safe_mean(values: Sequence[Any], reference):
    torch = require_torch()
    if not values:
        return _zero_like(reference)
    return torch.stack(list(values)).mean()


def _scalar_float(value) -> float:
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        if int(value.numel()) != 1:
            raise ValueError("expected scalar tensor")
        return float(value.detach().cpu().item())
    return float(value)


@dataclass(frozen=True)
class DetrSlotHungarianLossConfig:
    """Weights and feature conventions for DETR/free-slot set reconstruction."""

    feature_config: DetrSlotFeatureConfig = field(default_factory=DetrSlotFeatureConfig)
    core_weights: tuple[float, float, float, float] = DETR_SLOT_DEFAULT_CORE_WEIGHTS
    assignment_aux_weight: float = 0.05
    matched_core_weight: float = 1.0
    matched_aux_weight: float = 0.10
    existence_weight: float = 1.0
    existence_positive_weight: float = 1.0
    existence_negative_weight: float = 0.20
    count_weight: float = 0.10
    jet_summary_weight: float = 0.05
    duplicate_weight: float = 0.0
    hlt_support_weight: float = 0.0
    max_nearest_hlt_delta_r: float = 0.8
    duplicate_delta_r_scale: float = 0.10
    duplicate_probability_threshold: float = 0.25
    max_count_for_summary: float = 128.0
    huber_beta: float = 1.0
    eps: float = 1.0e-6
    brute_force_fallback_limit: int = 8
    allow_bruteforce_fallback: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.feature_config, Mapping):
            object.__setattr__(self, "feature_config", DetrSlotFeatureConfig.from_mapping(self.feature_config))
        if len(self.core_weights) != 4:
            raise ValueError("core_weights must contain four values: log_pt, eta, phi, log_energy")
        if any(float(value) < 0.0 for value in self.core_weights):
            raise ValueError("core_weights must be non-negative")
        nonnegative_fields = (
            "assignment_aux_weight",
            "matched_core_weight",
            "matched_aux_weight",
            "existence_weight",
            "existence_positive_weight",
            "existence_negative_weight",
            "count_weight",
            "jet_summary_weight",
            "duplicate_weight",
            "hlt_support_weight",
        )
        for field_name in nonnegative_fields:
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if float(self.max_nearest_hlt_delta_r) <= 0.0:
            raise ValueError("max_nearest_hlt_delta_r must be positive")
        if float(self.duplicate_delta_r_scale) <= 0.0:
            raise ValueError("duplicate_delta_r_scale must be positive")
        if not 0.0 <= float(self.duplicate_probability_threshold) <= 1.0:
            raise ValueError("duplicate_probability_threshold must be in [0, 1]")
        if float(self.max_count_for_summary) <= 0.0:
            raise ValueError("max_count_for_summary must be positive")
        if float(self.huber_beta) <= 0.0:
            raise ValueError("huber_beta must be positive")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if int(self.brute_force_fallback_limit) < 1:
            raise ValueError("brute_force_fallback_limit must be at least 1")
        if not isinstance(self.allow_bruteforce_fallback, bool):
            raise ValueError("allow_bruteforce_fallback must be a bool")
        object.__setattr__(self, "core_weights", tuple(float(value) for value in self.core_weights))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_config"] = self.feature_config.to_dict()
        payload["core_feature_names"] = list(DETR_SLOT_CORE_FEATURE_NAMES)
        payload["step"] = DETR_SLOT_HUNGARIAN_LOSS_STEP
        return payload


@dataclass(frozen=True)
class DetrSlotHungarianAssignment:
    """Hungarian assignment record for one jet."""

    pred_indices: np.ndarray
    target_indices: np.ndarray
    existence_targets: np.ndarray
    method: str
    matched_count: int
    candidate_count: int
    target_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pred_indices": self.pred_indices.astype(int).tolist(),
            "target_indices": self.target_indices.astype(int).tolist(),
            "existence_targets": self.existence_targets.astype(float).tolist(),
            "method": str(self.method),
            "matched_count": int(self.matched_count),
            "candidate_count": int(self.candidate_count),
            "target_count": int(self.target_count),
        }


@dataclass
class DetrSlotHungarianLossOutput:
    """Loss tensor plus inspectable components, diagnostics, and assignments."""

    total_loss: Any
    components: dict[str, Any]
    diagnostics: dict[str, Any]
    assignments: list[DetrSlotHungarianAssignment] = field(default_factory=list)

    def detached_float_dict(self, *, prefix: str = "loss_") -> dict[str, float]:
        payload = {f"{prefix}total": float(self.total_loss.detach().cpu().item())}
        for name, value in self.components.items():
            payload[f"{prefix}{name}"] = float(value.detach().cpu().item())
        for name, value in self.diagnostics.items():
            if hasattr(value, "detach"):
                payload[f"metric_{name}"] = float(value.detach().cpu().item())
            else:
                payload[f"metric_{name}"] = float(value)
        return payload


def _linear_sum_assignment_numpy(
    cost: np.ndarray,
    *,
    brute_force_limit: int,
    allow_bruteforce_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2D, got {cost.shape}")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), "none"
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment(cost)
        return rows.astype(np.int64), cols.astype(np.int64), "scipy"
    except ImportError:
        pass

    if not bool(allow_bruteforce_fallback):
        raise ImportError(
            "scipy.optimize.linear_sum_assignment is required for DETR-slot Hungarian matching. "
            "Set allow_bruteforce_fallback=True only for small smoke/unit tests."
        )

    if max(n_rows, n_cols) > int(brute_force_limit):
        raise ImportError(
            "scipy.optimize.linear_sum_assignment is required for DETR-slot Hungarian matching "
            f"when matrix shape {cost.shape} exceeds brute_force_fallback_limit={brute_force_limit}"
        )

    if n_rows <= n_cols:
        best_cols: tuple[int, ...] | None = None
        best_cost = float("inf")
        for cols in itertools.permutations(range(n_cols), n_rows):
            value = float(sum(cost[row, col] for row, col in enumerate(cols)))
            if value < best_cost:
                best_cost = value
                best_cols = cols
        return np.arange(n_rows, dtype=np.int64), np.asarray(best_cols, dtype=np.int64), "bruteforce"

    best_rows: tuple[int, ...] | None = None
    best_cost = float("inf")
    for rows in itertools.permutations(range(n_rows), n_cols):
        value = float(sum(cost[row, col] for col, row in enumerate(rows)))
        if value < best_cost:
            best_cost = value
            best_rows = rows
    return np.asarray(best_rows, dtype=np.int64), np.arange(n_cols, dtype=np.int64), "bruteforce"


def pairwise_detr_slot_core_cost(pred_core, target_core, config: DetrSlotHungarianLossConfig | None = None):
    """Return ``[n_pred, n_target]`` assignment cost with wrapped phi."""

    config = config or DetrSlotHungarianLossConfig()
    pred_core = _as_float_tensor(pred_core, name="pred_core")
    target_core = _as_float_tensor(target_core, name="target_core", device=pred_core.device, dtype=pred_core.dtype)
    if pred_core.ndim != 2 or int(pred_core.shape[-1]) != 4:
        raise ValueError(f"pred_core must have shape [n_pred, 4], got {tuple(pred_core.shape)}")
    if target_core.ndim != 2 or int(target_core.shape[-1]) != 4:
        raise ValueError(f"target_core must have shape [n_target, 4], got {tuple(target_core.shape)}")
    cost = None
    for index, weight in enumerate(config.core_weights):
        if index == 2:
            diff = wrapped_phi_difference(pred_core[:, None, 2], target_core[None, :, 2])
        else:
            diff = pred_core[:, None, index] - target_core[None, :, index]
        term = float(weight) * _smooth_l1_from_diff(diff, beta=float(config.huber_beta))
        cost = term if cost is None else cost + term
    return cost


def pairwise_detr_slot_aux_cost(pred_features, target_features, config: DetrSlotHungarianLossConfig | None = None):
    """Return a weak auxiliary-feature matching cost in raw-token aux order."""

    config = config or DetrSlotHungarianLossConfig()
    pred_features = _as_float_tensor(pred_features, name="pred_features")
    target_features = _as_float_tensor(
        target_features,
        name="target_features",
        device=pred_features.device,
        dtype=pred_features.dtype,
    )
    if pred_features.ndim != 2:
        raise ValueError(f"pred_features must have shape [n_pred, features], got {tuple(pred_features.shape)}")
    if target_features.ndim != 2:
        raise ValueError(f"target_features must have shape [n_target, features], got {tuple(target_features.shape)}")
    if int(pred_features.shape[-1]) != int(target_features.shape[-1]):
        raise ValueError("pred_features and target_features feature dimensions must match")
    pred_aux = raw_to_aux_features(pred_features, config.feature_config)
    target_aux = raw_to_aux_features(target_features, config.feature_config)
    if int(pred_aux.shape[-1]) == 0:
        return pred_features.new_zeros((int(pred_features.shape[0]), int(target_features.shape[0])))
    diff = pred_aux[:, None, :] - target_aux[None, :, :]
    return _smooth_l1_from_diff(diff, beta=float(config.huber_beta)).mean(dim=-1)


def pairwise_detr_slot_assignment_cost(pred_features, target_features, config: DetrSlotHungarianLossConfig | None = None):
    """Return Hungarian cost using core kinematics plus weak aux evidence."""

    config = config or DetrSlotHungarianLossConfig()
    pred_features = _as_float_tensor(pred_features, name="pred_features")
    target_features = _as_float_tensor(
        target_features,
        name="target_features",
        device=pred_features.device,
        dtype=pred_features.dtype,
    )
    core_cost = pairwise_detr_slot_core_cost(
        raw_to_core_features(pred_features, config.feature_config),
        raw_to_core_features(target_features, config.feature_config),
        config,
    )
    if float(config.assignment_aux_weight) <= 0.0:
        return core_cost
    return core_cost + float(config.assignment_aux_weight) * pairwise_detr_slot_aux_cost(
        pred_features,
        target_features,
        config,
    )


def detr_slot_matched_core_loss(pred_core, target_core, config: DetrSlotHungarianLossConfig):
    if int(pred_core.shape[0]) == 0:
        return _zero_like(pred_core)
    terms = []
    for index, weight in enumerate(config.core_weights):
        if index == 2:
            diff = wrapped_phi_difference(pred_core[:, 2], target_core[:, 2])
        else:
            diff = pred_core[:, index] - target_core[:, index]
        terms.append(float(weight) * _smooth_l1_from_diff(diff, beta=float(config.huber_beta)))
    return sum(terms).mean()


def _aux_positions_for_feature_indices(config: DetrSlotHungarianLossConfig) -> tuple[list[int], list[int]]:
    indices = config.feature_config.aux_feature_indices()
    binary = set(int(index) for index in config.feature_config.binary_aux_indices)
    binary_positions: list[int] = []
    continuous_positions: list[int] = []
    for position, feature_index in enumerate(indices):
        if int(feature_index) in binary:
            binary_positions.append(int(position))
        else:
            continuous_positions.append(int(position))
    return binary_positions, continuous_positions


def detr_slot_matched_aux_loss(
    pred_features,
    target_features,
    config: DetrSlotHungarianLossConfig,
    *,
    pred_aux_logits=None,
):
    if int(pred_features.shape[0]) == 0:
        return _zero_like(pred_features)
    torch = require_torch()
    pred_aux = raw_to_aux_features(pred_features, config.feature_config)
    target_aux = raw_to_aux_features(target_features, config.feature_config)
    if int(pred_aux.shape[-1]) == 0:
        return _zero_like(pred_features)
    binary_positions, continuous_positions = _aux_positions_for_feature_indices(config)
    terms = []
    if continuous_positions:
        position_tensor = torch.as_tensor(continuous_positions, dtype=torch.long, device=pred_aux.device)
        diff = pred_aux.index_select(-1, position_tensor) - target_aux.index_select(-1, position_tensor)
        terms.append(_smooth_l1_from_diff(diff, beta=float(config.huber_beta)).mean())
    if binary_positions:
        position_tensor = torch.as_tensor(binary_positions, dtype=torch.long, device=pred_aux.device)
        target_binary = target_aux.index_select(-1, position_tensor).clamp(min=0.0, max=1.0)
        if pred_aux_logits is None:
            pred_binary = pred_aux.index_select(-1, position_tensor)
            diff = pred_binary - target_binary
            terms.append(_smooth_l1_from_diff(diff, beta=float(config.huber_beta)).mean())
        else:
            logits = _as_float_tensor(
                pred_aux_logits,
                name="pred_aux_logits",
                device=pred_aux.device,
                dtype=pred_aux.dtype,
            )
            if tuple(logits.shape) != tuple(pred_aux.shape):
                raise ValueError(f"pred_aux_logits must have shape {tuple(pred_aux.shape)}, got {tuple(logits.shape)}")
            binary_logits = logits.index_select(-1, position_tensor)
            terms.append(_functional().binary_cross_entropy_with_logits(binary_logits, target_binary, reduction="mean"))
    return _safe_mean(terms, pred_features)


def _existence_loss(logits, targets, config: DetrSlotHungarianLossConfig):
    if int(logits.numel()) == 0:
        return _zero_like(logits)
    functional = _functional()
    raw = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = require_torch().where(
        targets > 0.5,
        logits.new_full(targets.shape, float(config.existence_positive_weight)),
        logits.new_full(targets.shape, float(config.existence_negative_weight)),
    )
    denom = require_torch().clamp(weights.sum(), min=float(config.eps))
    return (raw * weights).sum() / denom


def _count_loss(existence_logits, candidate_mask, target_count: int, config: DetrSlotHungarianLossConfig):
    pred_count = (existence_logits.sigmoid() * candidate_mask.to(dtype=existence_logits.dtype)).sum()
    target = pred_count.new_tensor(float(target_count))
    return _smooth_l1_from_diff(pred_count - target, beta=float(config.huber_beta))


def _jet_summary(features, weights, config: DetrSlotHungarianLossConfig):
    torch = require_torch()
    core = raw_to_core_features(features, config.feature_config)
    pt = torch.clamp(features[:, int(config.feature_config.pt_index)], min=0.0)
    energy = torch.clamp(features[:, int(config.feature_config.energy_index)], min=0.0)
    eta = core[:, 1]
    phi = core[:, 2]
    weights = weights.to(dtype=features.dtype)
    pt_weights = weights * pt
    sum_pt = pt_weights.sum()
    sum_energy = (weights * energy).sum()
    denom = torch.clamp(sum_pt, min=float(config.eps))
    mean_eta = (pt_weights * eta).sum() / denom
    mean_sin_phi = (pt_weights * torch.sin(phi)).sum() / denom
    mean_cos_phi = (pt_weights * torch.cos(phi)).sum() / denom
    count = weights.sum() / float(config.max_count_for_summary)
    return torch.stack(
        [
            torch.log1p(sum_pt),
            torch.log1p(sum_energy),
            mean_eta,
            mean_sin_phi,
            mean_cos_phi,
            count,
        ]
    )


def detr_slot_jet_summary_loss(pred_features, existence_logits, candidate_mask, target_features, target_mask, config):
    pred_weights = existence_logits.sigmoid() * candidate_mask.to(dtype=existence_logits.dtype)
    target_weights = target_mask.to(dtype=pred_features.dtype)
    pred_summary = _jet_summary(pred_features, pred_weights, config)
    target_summary = _jet_summary(target_features, target_weights, config)
    return _smooth_l1_from_diff(pred_summary - target_summary, beta=float(config.huber_beta)).mean()


def detr_slot_support_loss(pred_features, existence_logits, candidate_mask, hlt_features, hlt_mask, config):
    torch = require_torch()
    if hlt_features is None or hlt_mask is None or float(config.hlt_support_weight) <= 0.0:
        return _zero_like(pred_features)
    if not bool(candidate_mask.any()) or not bool(hlt_mask.any()):
        return _zero_like(pred_features)
    pred_core = raw_to_core_features(pred_features, config.feature_config)
    hlt_core = raw_to_core_features(hlt_features, config.feature_config)
    active = candidate_mask
    valid_hlt = hlt_mask.bool()
    pred = pred_core[active]
    hlt = hlt_core[valid_hlt]
    if int(pred.shape[0]) == 0 or int(hlt.shape[0]) == 0:
        return _zero_like(pred_features)
    deta = pred[:, None, 1] - hlt[None, :, 1]
    dphi = wrapped_phi_difference(pred[:, None, 2], hlt[None, :, 2])
    min_dr = torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=0.0)).min(dim=1).values
    weights = existence_logits[active].sigmoid()
    denom = torch.clamp(weights.sum(), min=1.0)
    return ((torch.relu(min_dr - float(config.max_nearest_hlt_delta_r)) ** 2) * weights).sum() / denom


def detr_slot_duplicate_penalty(pred_features, existence_logits, candidate_mask, config):
    torch = require_torch()
    if float(config.duplicate_weight) <= 0.0 or not bool(candidate_mask.any()):
        return _zero_like(pred_features)
    probs = existence_logits.sigmoid()
    active = candidate_mask & (probs >= float(config.duplicate_probability_threshold))
    if int(active.sum().detach().cpu().item()) < 2:
        return _zero_like(pred_features)
    core = raw_to_core_features(pred_features, config.feature_config)[active]
    weights = probs[active]
    deta = core[:, None, 1] - core[None, :, 1]
    dphi = wrapped_phi_difference(core[:, None, 2], core[None, :, 2])
    dr2 = deta * deta + dphi * dphi
    eye = torch.eye(int(core.shape[0]), dtype=torch.bool, device=core.device)
    pair_weights = weights[:, None] * weights[None, :]
    penalty = torch.exp(-dr2 / max(float(config.duplicate_delta_r_scale) ** 2, float(config.eps))) * pair_weights
    penalty = penalty.masked_fill(eye, 0.0)
    denom = torch.clamp((~eye).to(dtype=penalty.dtype).sum(), min=1.0)
    return penalty.sum() / denom


def compute_detr_slot_hungarian_loss(
    *,
    predicted_features,
    existence_logits,
    predicted_aux_logits=None,
    offline_features,
    offline_mask,
    candidate_mask=None,
    hlt_features=None,
    hlt_mask=None,
    config: DetrSlotHungarianLossConfig | Mapping[str, Any] | None = None,
) -> DetrSlotHungarianLossOutput:
    """Compute the full DETR/free-slot Hungarian reconstruction objective."""

    torch = require_torch()
    config = (
        config
        if isinstance(config, DetrSlotHungarianLossConfig)
        else DetrSlotHungarianLossConfig(**dict(config or {}))
    )
    predicted_features = _as_float_tensor(predicted_features, name="predicted_features")
    existence_logits = _as_float_tensor(
        existence_logits,
        name="existence_logits",
        device=predicted_features.device,
        dtype=predicted_features.dtype,
    )
    offline_features = _as_float_tensor(
        offline_features,
        name="offline_features",
        device=predicted_features.device,
        dtype=predicted_features.dtype,
    )
    _validate_3d("predicted_features", predicted_features)
    _validate_3d("offline_features", offline_features)
    if int(predicted_features.shape[0]) != int(offline_features.shape[0]):
        raise ValueError("predicted_features and offline_features batch sizes must match")
    if int(predicted_features.shape[-1]) != int(offline_features.shape[-1]):
        raise ValueError("predicted_features and offline_features feature dimensions must match")
    if tuple(existence_logits.shape) != tuple(predicted_features.shape[:2]):
        raise ValueError(
            f"existence_logits must have shape {tuple(predicted_features.shape[:2])}, got {tuple(existence_logits.shape)}"
        )
    if predicted_aux_logits is not None:
        predicted_aux_logits = _as_float_tensor(
            predicted_aux_logits,
            name="predicted_aux_logits",
            device=predicted_features.device,
            dtype=predicted_features.dtype,
        )
        expected_aux_shape = tuple(predicted_features.shape[:2]) + (int(config.feature_config.aux_dim),)
        if tuple(predicted_aux_logits.shape) != expected_aux_shape:
            raise ValueError(
                f"predicted_aux_logits must have shape {expected_aux_shape}, got {tuple(predicted_aux_logits.shape)}"
            )
    candidate_mask = _as_bool_mask(
        candidate_mask,
        name="candidate_mask",
        expected_shape=tuple(predicted_features.shape[:2]),
        device=predicted_features.device,
    )
    offline_mask = _as_bool_mask(
        offline_mask,
        name="offline_mask",
        expected_shape=tuple(offline_features.shape[:2]),
        device=predicted_features.device,
    )
    if hlt_features is not None:
        hlt_features = _as_float_tensor(
            hlt_features,
            name="hlt_features",
            device=predicted_features.device,
            dtype=predicted_features.dtype,
        )
        _validate_3d("hlt_features", hlt_features)
        if int(hlt_features.shape[0]) != int(predicted_features.shape[0]):
            raise ValueError("hlt_features batch size must match predicted_features")
        hlt_mask = _as_bool_mask(
            hlt_mask,
            name="hlt_mask",
            expected_shape=tuple(hlt_features.shape[:2]),
            device=predicted_features.device,
        )

    pred_core_all = raw_to_core_features(predicted_features, config.feature_config)
    target_core_all = raw_to_core_features(offline_features, config.feature_config)

    row_core_losses = []
    row_aux_losses = []
    row_existence_losses = []
    row_count_losses = []
    row_jet_losses = []
    row_support_losses = []
    row_duplicate_losses = []
    assignments: list[DetrSlotHungarianAssignment] = []
    matched_counts: list[float] = []
    candidate_counts: list[float] = []
    target_counts: list[float] = []
    predicted_counts = []
    count_abs_errors = []
    existence_target_means: list[float] = []
    matched_delta_r_means = []
    matched_delta_r_p90s = []
    matched_logpt_maes = []
    matched_eta_maes = []
    matched_phi_maes = []
    jet_sum_pt_relative_errors = []
    jet_sum_energy_relative_errors = []
    existence_tp = predicted_features.new_tensor(0.0)
    existence_fp = predicted_features.new_tensor(0.0)
    existence_fn = predicted_features.new_tensor(0.0)
    method_counts = {"scipy": 0, "bruteforce": 0, "none": 0}

    batch_size = int(predicted_features.shape[0])
    for batch_index in range(batch_size):
        row_pred = predicted_features[batch_index]
        row_target = offline_features[batch_index]
        row_pred_core = pred_core_all[batch_index]
        row_target_core = target_core_all[batch_index]
        row_logits = existence_logits[batch_index]
        row_aux_logits = None if predicted_aux_logits is None else predicted_aux_logits[batch_index]
        row_candidate_mask = candidate_mask[batch_index]
        row_target_mask = offline_mask[batch_index]
        pred_valid_indices = torch.nonzero(row_candidate_mask, as_tuple=False).flatten()
        target_valid_indices = torch.nonzero(row_target_mask, as_tuple=False).flatten()
        candidate_count = int(pred_valid_indices.numel())
        target_count = int(target_valid_indices.numel())
        candidate_counts.append(float(candidate_count))
        target_counts.append(float(target_count))
        predicted_counts.append((row_logits.sigmoid() * row_candidate_mask.to(dtype=row_logits.dtype)).sum())

        if candidate_count > 0 and target_count > 0:
            valid_pred = row_pred.index_select(0, pred_valid_indices)
            valid_target = row_target.index_select(0, target_valid_indices)
            cost = pairwise_detr_slot_assignment_cost(valid_pred, valid_target, config)
            rows_np, cols_np, method = _linear_sum_assignment_numpy(
                cost.detach().cpu().numpy(),
                brute_force_limit=int(config.brute_force_fallback_limit),
                allow_bruteforce_fallback=bool(config.allow_bruteforce_fallback),
            )
            pred_match_indices = pred_valid_indices[torch.as_tensor(rows_np, dtype=torch.long, device=row_pred.device)]
            target_match_indices = target_valid_indices[torch.as_tensor(cols_np, dtype=torch.long, device=row_pred.device)]
        else:
            method = "none"
            pred_match_indices = torch.zeros((0,), dtype=torch.long, device=row_pred.device)
            target_match_indices = torch.zeros((0,), dtype=torch.long, device=row_pred.device)

        method_counts[method] = method_counts.get(method, 0) + 1
        existence_targets = torch.zeros_like(row_logits)
        if int(pred_match_indices.numel()) > 0:
            existence_targets[pred_match_indices] = 1.0

        matched_pred = row_pred.index_select(0, pred_match_indices)
        matched_target = row_target.index_select(0, target_match_indices)
        matched_aux_logits = None if row_aux_logits is None else row_aux_logits.index_select(0, pred_match_indices)
        matched_pred_core = row_pred_core.index_select(0, pred_match_indices)
        matched_target_core = row_target_core.index_select(0, target_match_indices)
        row_core_losses.append(detr_slot_matched_core_loss(matched_pred_core, matched_target_core, config))
        row_aux_losses.append(
            detr_slot_matched_aux_loss(
                matched_pred,
                matched_target,
                config,
                pred_aux_logits=matched_aux_logits,
            )
        )
        row_existence_losses.append(_existence_loss(row_logits[row_candidate_mask], existence_targets[row_candidate_mask], config))
        row_count_losses.append(_count_loss(row_logits, row_candidate_mask, target_count, config))
        row_jet_losses.append(
            detr_slot_jet_summary_loss(row_pred, row_logits, row_candidate_mask, row_target, row_target_mask, config)
        )
        row_support_losses.append(
            detr_slot_support_loss(
                row_pred,
                row_logits,
                row_candidate_mask,
                None if hlt_features is None else hlt_features[batch_index],
                None if hlt_mask is None else hlt_mask[batch_index],
                config,
            )
        )
        row_duplicate_losses.append(detr_slot_duplicate_penalty(row_pred, row_logits, row_candidate_mask, config))

        matched_count = int(pred_match_indices.numel())
        matched_counts.append(float(matched_count))
        pred_count = predicted_counts[-1]
        count_abs_errors.append((pred_count - pred_count.new_tensor(float(target_count))).abs())
        pred_active = (row_logits.sigmoid() >= 0.5) & row_candidate_mask
        target_active = existence_targets > 0.5
        existence_tp = existence_tp + (pred_active & target_active).to(dtype=predicted_features.dtype).sum()
        existence_fp = existence_fp + (pred_active & ~target_active).to(dtype=predicted_features.dtype).sum()
        existence_fn = existence_fn + (~pred_active & target_active).to(dtype=predicted_features.dtype).sum()
        pred_summary = _jet_summary(row_pred, row_logits.sigmoid() * row_candidate_mask.to(dtype=row_logits.dtype), config)
        target_summary = _jet_summary(row_target, row_target_mask.to(dtype=row_pred.dtype), config)
        jet_sum_pt_relative_errors.append(
            (torch.expm1(pred_summary[0]) - torch.expm1(target_summary[0])).abs()
            / torch.clamp(torch.expm1(target_summary[0]).abs(), min=float(config.eps))
        )
        jet_sum_energy_relative_errors.append(
            (torch.expm1(pred_summary[1]) - torch.expm1(target_summary[1])).abs()
            / torch.clamp(torch.expm1(target_summary[1]).abs(), min=float(config.eps))
        )
        existence_target_means.append(float(existence_targets[row_candidate_mask].detach().float().mean().cpu().item()) if candidate_count else 0.0)
        if matched_count > 0:
            d_eta = matched_pred_core[:, 1] - matched_target_core[:, 1]
            d_phi = wrapped_phi_difference(matched_pred_core[:, 2], matched_target_core[:, 2])
            delta_r = torch.sqrt(torch.clamp(d_eta * d_eta + d_phi * d_phi, min=0.0))
            matched_delta_r_means.append(delta_r.mean())
            matched_delta_r_p90s.append(torch.quantile(delta_r, 0.9))
            matched_logpt_maes.append((matched_pred_core[:, 0] - matched_target_core[:, 0]).abs().mean())
            matched_eta_maes.append(d_eta.abs().mean())
            matched_phi_maes.append(d_phi.abs().mean())

        assignments.append(
            DetrSlotHungarianAssignment(
                pred_indices=pred_match_indices.detach().cpu().numpy().astype(np.int64),
                target_indices=target_match_indices.detach().cpu().numpy().astype(np.int64),
                existence_targets=existence_targets.detach().cpu().numpy().astype(np.float32),
                method=method,
                matched_count=matched_count,
                candidate_count=candidate_count,
                target_count=target_count,
            )
        )

    components = {
        "matched_core_loss": _safe_mean(row_core_losses, predicted_features),
        "matched_aux_loss": _safe_mean(row_aux_losses, predicted_features),
        "existence_loss": _safe_mean(row_existence_losses, predicted_features),
        "count_loss": _safe_mean(row_count_losses, predicted_features),
        "jet_summary_loss": _safe_mean(row_jet_losses, predicted_features),
        "support_loss": _safe_mean(row_support_losses, predicted_features),
        "duplicate_loss": _safe_mean(row_duplicate_losses, predicted_features),
    }
    total = (
        float(config.matched_core_weight) * components["matched_core_loss"]
        + float(config.matched_aux_weight) * components["matched_aux_loss"]
        + float(config.existence_weight) * components["existence_loss"]
        + float(config.count_weight) * components["count_loss"]
        + float(config.jet_summary_weight) * components["jet_summary_loss"]
        + float(config.hlt_support_weight) * components["support_loss"]
        + float(config.duplicate_weight) * components["duplicate_loss"]
    )

    denom = max(float(batch_size), 1.0)
    diagnostics = {
        "matched_count_mean": predicted_features.new_tensor(float(np.mean(matched_counts)) if matched_counts else 0.0),
        "candidate_count_mean": predicted_features.new_tensor(float(np.mean(candidate_counts)) if candidate_counts else 0.0),
        "target_count_mean": predicted_features.new_tensor(float(np.mean(target_counts)) if target_counts else 0.0),
        "predicted_count_mean": _safe_mean(predicted_counts, predicted_features),
        "count_mae": _safe_mean(count_abs_errors, predicted_features),
        "existence_target_mean": predicted_features.new_tensor(
            float(np.mean(existence_target_means)) if existence_target_means else 0.0
        ),
        "existence_precision": existence_tp / torch.clamp(existence_tp + existence_fp, min=float(config.eps)),
        "existence_recall": existence_tp / torch.clamp(existence_tp + existence_fn, min=float(config.eps)),
        "matching_used_scipy_fraction": predicted_features.new_tensor(float(method_counts.get("scipy", 0)) / denom),
        "matching_used_bruteforce_fraction": predicted_features.new_tensor(float(method_counts.get("bruteforce", 0)) / denom),
        "matching_empty_fraction": predicted_features.new_tensor(float(method_counts.get("none", 0)) / denom),
        "matched_delta_r_mean": _safe_mean(matched_delta_r_means, predicted_features),
        "matched_delta_r_p90": _safe_mean(matched_delta_r_p90s, predicted_features),
        "matched_logpt_mae": _safe_mean(matched_logpt_maes, predicted_features),
        "matched_eta_mae": _safe_mean(matched_eta_maes, predicted_features),
        "matched_phi_mae": _safe_mean(matched_phi_maes, predicted_features),
        "jet_sum_pt_relative_error_mean": _safe_mean(jet_sum_pt_relative_errors, predicted_features),
        "jet_sum_energy_relative_error_mean": _safe_mean(jet_sum_energy_relative_errors, predicted_features),
    }
    return DetrSlotHungarianLossOutput(
        total_loss=total,
        components=components,
        diagnostics=diagnostics,
        assignments=assignments,
    )


def default_detr_slot_hungarian_loss_config() -> DetrSlotHungarianLossConfig:
    return DetrSlotHungarianLossConfig()


__all__ = [
    "DETR_SLOT_CORE_FEATURE_NAMES",
    "DETR_SLOT_DEFAULT_CORE_WEIGHTS",
    "DETR_SLOT_HUNGARIAN_LOSS_STEP",
    "DetrSlotHungarianAssignment",
    "DetrSlotHungarianLossConfig",
    "DetrSlotHungarianLossOutput",
    "compute_detr_slot_hungarian_loss",
    "default_detr_slot_hungarian_loss_config",
    "detr_slot_duplicate_penalty",
    "detr_slot_jet_summary_loss",
    "detr_slot_matched_aux_loss",
    "detr_slot_matched_core_loss",
    "detr_slot_support_loss",
    "pairwise_detr_slot_assignment_cost",
    "pairwise_detr_slot_aux_cost",
    "pairwise_detr_slot_core_cost",
    "require_torch",
]
