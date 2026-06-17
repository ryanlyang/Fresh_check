"""Hungarian set-matching losses for offline-particle reconstruction.

This module is intentionally independent from the teacher-logit loss stack.
It supervises a predicted unordered particle set against the offline target
set using a hard Hungarian assignment followed by differentiable matched
regression and slot-existence losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import itertools
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1.0e-6
DEFAULT_CORE_WEIGHTS: tuple[float, float, float, float] = (2.0, 1.0, 1.0, 1.0)
DEFAULT_CORE_FEATURE_NAMES: tuple[str, str, str, str] = ("log_pt", "eta", "phi", "log_energy")


def _maybe_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def require_torch():
    torch = _maybe_torch()
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("Set-matching losses require PyTorch")
    return torch


def _functional():
    torch = require_torch()
    return torch.nn.functional


def wrapped_delta_phi(delta_phi):
    """Wrap angular differences into [-pi, pi] with torch operations."""

    torch = require_torch()
    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def _smooth_l1_from_diff(diff, *, beta: float):
    torch = require_torch()
    functional = _functional()
    return functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=float(beta))


def _zero_like(reference):
    return reference.sum() * 0.0


def _validate_3d(name: str, value):
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, slots, features], got {tuple(value.shape)}")


def _validate_2d_mask(name: str, value, expected_shape: tuple[int, int]):
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(value.shape)}")


def _bool_tensor(value, *, device, shape: tuple[int, int], name: str):
    torch = require_torch()
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
    _validate_2d_mask(name, tensor, shape)
    return tensor


def _core_mean_std(config: "SetMatchingLossConfig", *, device, dtype):
    torch = require_torch()
    mean = torch.zeros((4,), dtype=dtype, device=device)
    std = torch.ones((4,), dtype=dtype, device=device)
    if config.core_mean is not None:
        mean = torch.as_tensor(config.core_mean, dtype=dtype, device=device)
    if config.core_std is not None:
        std = torch.as_tensor(config.core_std, dtype=dtype, device=device)
        std = torch.clamp(std, min=float(config.eps))
    return mean, std


@dataclass(frozen=True)
class SetMatchingLossConfig:
    """Weights and feature conventions for set-matching reconstruction."""

    pt_index: int = 0
    eta_index: int = 1
    phi_index: int = 2
    energy_index: int = 3
    core_weights: tuple[float, float, float, float] = DEFAULT_CORE_WEIGHTS
    core_mean: tuple[float, float, float, float] | None = None
    core_std: tuple[float, float, float, float] | None = None
    aux_indices: tuple[int, ...] | None = None
    matched_core_weight: float = 1.0
    matched_aux_weight: float = 0.10
    existence_weight: float = 1.0
    existence_positive_weight: float = 1.0
    count_weight: float = 0.10
    jet_summary_weight: float = 0.05
    correction_budget_weight: float = 0.02
    chamfer_weight: float = 0.0
    huber_beta: float = 1.0
    eps: float = EPS
    max_abs_eta: float = 5.0
    max_active_slots: int | None = None
    max_count_for_summary: float = 128.0
    negative_kinematics_budget_weight: float = 1.0
    eta_budget_weight: float = 0.25
    count_budget_weight: float = 0.10
    hlt_support_budget_weight: float = 0.0
    max_nearest_hlt_delta_r: float = 0.8
    brute_force_fallback_limit: int = 8

    def __post_init__(self) -> None:
        if len(self.core_weights) != 4:
            raise ValueError("core_weights must contain four values: log_pt, eta, phi, log_energy")
        if self.core_mean is not None and len(self.core_mean) != 4:
            raise ValueError("core_mean must contain four values when provided")
        if self.core_std is not None and len(self.core_std) != 4:
            raise ValueError("core_std must contain four values when provided")
        if int(self.pt_index) < 0 or int(self.eta_index) < 0 or int(self.phi_index) < 0 or int(self.energy_index) < 0:
            raise ValueError("feature indices must be non-negative")
        if float(self.huber_beta) <= 0.0:
            raise ValueError("huber_beta must be positive")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if self.max_active_slots is not None and int(self.max_active_slots) <= 0:
            raise ValueError("max_active_slots must be positive when provided")
        if float(self.max_count_for_summary) <= 0.0:
            raise ValueError("max_count_for_summary must be positive")
        if int(self.brute_force_fallback_limit) < 1:
            raise ValueError("brute_force_fallback_limit must be at least 1")
        object.__setattr__(self, "core_weights", tuple(float(value) for value in self.core_weights))
        if self.core_mean is not None:
            object.__setattr__(self, "core_mean", tuple(float(value) for value in self.core_mean))
        if self.core_std is not None:
            object.__setattr__(self, "core_std", tuple(max(float(value), float(self.eps)) for value in self.core_std))
        if self.aux_indices is not None:
            object.__setattr__(self, "aux_indices", tuple(int(value) for value in self.aux_indices))

    @property
    def core_indices(self) -> tuple[int, int, int, int]:
        return (int(self.pt_index), int(self.eta_index), int(self.phi_index), int(self.energy_index))

    def aux_feature_indices(self, feature_dim: int) -> tuple[int, ...]:
        if self.aux_indices is not None:
            return tuple(index for index in self.aux_indices if 0 <= int(index) < int(feature_dim))
        core = set(self.core_indices)
        return tuple(index for index in range(int(feature_dim)) if index not in core)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pt_index": int(self.pt_index),
            "eta_index": int(self.eta_index),
            "phi_index": int(self.phi_index),
            "energy_index": int(self.energy_index),
            "core_feature_names": list(DEFAULT_CORE_FEATURE_NAMES),
            "core_weights": list(self.core_weights),
            "core_mean": None if self.core_mean is None else list(self.core_mean),
            "core_std": None if self.core_std is None else list(self.core_std),
            "aux_indices": None if self.aux_indices is None else list(self.aux_indices),
            "matched_core_weight": float(self.matched_core_weight),
            "matched_aux_weight": float(self.matched_aux_weight),
            "existence_weight": float(self.existence_weight),
            "existence_positive_weight": float(self.existence_positive_weight),
            "count_weight": float(self.count_weight),
            "jet_summary_weight": float(self.jet_summary_weight),
            "correction_budget_weight": float(self.correction_budget_weight),
            "chamfer_weight": float(self.chamfer_weight),
            "huber_beta": float(self.huber_beta),
            "max_abs_eta": float(self.max_abs_eta),
            "max_active_slots": self.max_active_slots,
            "max_count_for_summary": float(self.max_count_for_summary),
            "negative_kinematics_budget_weight": float(self.negative_kinematics_budget_weight),
            "eta_budget_weight": float(self.eta_budget_weight),
            "count_budget_weight": float(self.count_budget_weight),
            "hlt_support_budget_weight": float(self.hlt_support_budget_weight),
            "max_nearest_hlt_delta_r": float(self.max_nearest_hlt_delta_r),
            "brute_force_fallback_limit": int(self.brute_force_fallback_limit),
        }


@dataclass(frozen=True)
class SetMatchingAssignment:
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
            "method": self.method,
            "matched_count": int(self.matched_count),
            "candidate_count": int(self.candidate_count),
            "target_count": int(self.target_count),
        }


@dataclass
class SetMatchingLossOutput:
    """Loss tensor plus inspectable components and assignments."""

    total_loss: Any
    components: dict[str, Any]
    diagnostics: dict[str, Any]
    assignments: list[SetMatchingAssignment] = field(default_factory=list)

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


def _component_values(tokens, config: SetMatchingLossConfig):
    torch = require_torch()
    mean, std = _core_mean_std(config, device=tokens.device, dtype=tokens.dtype)
    pt = torch.log(torch.clamp(tokens[..., int(config.pt_index)], min=float(config.eps)))
    eta = tokens[..., int(config.eta_index)]
    phi = wrapped_delta_phi(tokens[..., int(config.phi_index)])
    energy = torch.log(torch.clamp(tokens[..., int(config.energy_index)], min=float(config.eps)))
    values = (pt, eta, phi, energy)
    normalized = []
    for index, value in enumerate(values):
        normalized.append((value - mean[index]) / std[index])
    return tuple(normalized)


def pairwise_core_cost(pred_features, target_features, config: SetMatchingLossConfig | None = None):
    """Return [n_pred, n_target] assignment cost with phi wraparound."""

    config = config or SetMatchingLossConfig()
    pred_components = _component_values(pred_features, config)
    target_components = _component_values(target_features, config)
    _, std = _core_mean_std(config, device=pred_features.device, dtype=pred_features.dtype)
    weights = config.core_weights
    cost = None
    for index, (pred_value, target_value) in enumerate(zip(pred_components, target_components)):
        if index == 2:
            raw_delta = wrapped_delta_phi(
                pred_features[:, None, int(config.phi_index)] - target_features[None, :, int(config.phi_index)]
            )
            diff = raw_delta / std[index]
        else:
            diff = pred_value[:, None] - target_value[None, :]
        term = float(weights[index]) * _smooth_l1_from_diff(diff, beta=float(config.huber_beta))
        cost = term if cost is None else cost + term
    return cost


def matched_core_loss(pred_features, target_features, config: SetMatchingLossConfig):
    if int(pred_features.shape[0]) == 0:
        return _zero_like(pred_features)
    pred_components = _component_values(pred_features, config)
    target_components = _component_values(target_features, config)
    _, std = _core_mean_std(config, device=pred_features.device, dtype=pred_features.dtype)
    terms = []
    for index, (pred_value, target_value) in enumerate(zip(pred_components, target_components)):
        if index == 2:
            diff = wrapped_delta_phi(
                pred_features[:, int(config.phi_index)] - target_features[:, int(config.phi_index)]
            ) / std[index]
        else:
            diff = pred_value - target_value
        terms.append(float(config.core_weights[index]) * _smooth_l1_from_diff(diff, beta=float(config.huber_beta)))
    return sum(terms).mean()


def matched_aux_loss(pred_features, target_features, config: SetMatchingLossConfig):
    if int(pred_features.shape[0]) == 0:
        return _zero_like(pred_features)
    feature_dim = int(pred_features.shape[-1])
    indices = config.aux_feature_indices(feature_dim)
    if not indices:
        return _zero_like(pred_features)
    torch = require_torch()
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=pred_features.device)
    diff = pred_features.index_select(-1, index_tensor) - target_features.index_select(-1, index_tensor)
    return _smooth_l1_from_diff(diff, beta=float(config.huber_beta)).mean()


def _linear_sum_assignment_numpy(cost: np.ndarray, *, brute_force_limit: int) -> tuple[np.ndarray, np.ndarray, str]:
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

    if max(n_rows, n_cols) > int(brute_force_limit):
        raise ImportError(
            "scipy.optimize.linear_sum_assignment is required for Hungarian matching "
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

    best_rows = None
    best_cost = float("inf")
    for rows in itertools.permutations(range(n_rows), n_cols):
        value = float(sum(cost[row, col] for col, row in enumerate(rows)))
        if value < best_cost:
            best_cost = value
            best_rows = rows
    return np.asarray(best_rows, dtype=np.int64), np.arange(n_cols, dtype=np.int64), "bruteforce"


def _safe_mean(values: Sequence[Any], reference):
    torch = require_torch()
    if not values:
        return _zero_like(reference)
    return torch.stack(list(values)).mean()


def _existence_loss(logits, targets, config: SetMatchingLossConfig):
    if int(logits.numel()) == 0:
        return _zero_like(logits)
    functional = _functional()
    pos_weight = logits.new_tensor(float(config.existence_positive_weight))
    return functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="mean")


def _count_loss(existence_logits, candidate_mask, target_count, config: SetMatchingLossConfig):
    pred_count = (existence_logits.sigmoid() * candidate_mask.float()).sum()
    target = pred_count.new_tensor(float(target_count))
    return _smooth_l1_from_diff(pred_count - target, beta=float(config.huber_beta))


def _jet_summary(features, weights, config: SetMatchingLossConfig):
    torch = require_torch()
    weights = weights.float()
    pt = torch.clamp(features[:, int(config.pt_index)], min=0.0)
    energy = torch.clamp(features[:, int(config.energy_index)], min=0.0)
    eta = features[:, int(config.eta_index)]
    phi = features[:, int(config.phi_index)]
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


def jet_summary_loss(pred_features, existence_logits, candidate_mask, target_features, target_mask, config: SetMatchingLossConfig):
    pred_weights = existence_logits.sigmoid() * candidate_mask.float()
    target_weights = target_mask.float()
    pred_summary = _jet_summary(pred_features, pred_weights, config)
    target_summary = _jet_summary(target_features, target_weights, config)
    return _smooth_l1_from_diff(pred_summary - target_summary, beta=float(config.huber_beta)).mean()


def correction_budget_loss(
    pred_features,
    existence_logits,
    candidate_mask,
    config: SetMatchingLossConfig,
    *,
    hlt_features=None,
    hlt_mask=None,
):
    torch = require_torch()
    weights = existence_logits.sigmoid() * candidate_mask.float()
    denom = torch.clamp(weights.sum(), min=1.0)
    pt = pred_features[:, int(config.pt_index)]
    energy = pred_features[:, int(config.energy_index)]
    eta = pred_features[:, int(config.eta_index)]
    negative = ((torch.relu(-pt) ** 2 + torch.relu(-energy) ** 2) * weights).sum() / denom
    eta_excess = ((torch.relu(torch.abs(eta) - float(config.max_abs_eta)) ** 2) * weights).sum() / denom
    max_slots = int(config.max_active_slots or int(pred_features.shape[0]))
    count_excess = torch.relu(weights.sum() - float(max_slots)) ** 2 / float(max(max_slots, 1) ** 2)

    support = _zero_like(pred_features)
    if hlt_features is not None and hlt_mask is not None and float(config.hlt_support_budget_weight) > 0.0:
        valid_hlt = hlt_features[hlt_mask]
        if int(valid_hlt.shape[0]) > 0 and int(pred_features.shape[0]) > 0:
            deta = pred_features[:, None, int(config.eta_index)] - valid_hlt[None, :, int(config.eta_index)]
            dphi = wrapped_delta_phi(
                pred_features[:, None, int(config.phi_index)] - valid_hlt[None, :, int(config.phi_index)]
            )
            min_dr = torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=0.0)).min(dim=1).values
            support = (torch.relu(min_dr - float(config.max_nearest_hlt_delta_r)) ** 2 * weights).sum() / denom

    return (
        float(config.negative_kinematics_budget_weight) * negative
        + float(config.eta_budget_weight) * eta_excess
        + float(config.count_budget_weight) * count_excess
        + float(config.hlt_support_budget_weight) * support
    )


def chamfer_loss(pred_features, existence_logits, candidate_mask, target_features, target_mask, config: SetMatchingLossConfig):
    pred_valid = candidate_mask
    target_valid = target_mask
    if not bool(pred_valid.any()) or not bool(target_valid.any()):
        return _zero_like(pred_features)
    pred = pred_features[pred_valid]
    target = target_features[target_valid]
    costs = pairwise_core_cost(pred, target, config)
    pred_weights = existence_logits[pred_valid].sigmoid()
    pred_to_target = (costs.min(dim=1).values * pred_weights).sum() / torch_clamp_min(pred_weights.sum(), 1.0)
    target_to_pred = costs.min(dim=0).values.mean()
    return pred_to_target + target_to_pred


def torch_clamp_min(value, minimum: float):
    torch = require_torch()
    return torch.clamp(value, min=float(minimum))


def compute_set_matching_loss(
    *,
    predicted_features,
    existence_logits,
    offline_features,
    offline_mask,
    candidate_mask=None,
    hlt_features=None,
    hlt_mask=None,
    config: SetMatchingLossConfig | None = None,
) -> SetMatchingLossOutput:
    """Compute the full set-matching reconstruction objective."""

    torch = require_torch()
    config = config or SetMatchingLossConfig()
    predicted_features = torch.as_tensor(predicted_features)
    existence_logits = torch.as_tensor(existence_logits, dtype=predicted_features.dtype, device=predicted_features.device)
    offline_features = torch.as_tensor(offline_features, dtype=predicted_features.dtype, device=predicted_features.device)
    offline_mask = torch.as_tensor(offline_mask, dtype=torch.bool, device=predicted_features.device)
    _validate_3d("predicted_features", predicted_features)
    _validate_3d("offline_features", offline_features)
    if predicted_features.shape[0] != offline_features.shape[0]:
        raise ValueError("predicted and offline batch sizes must match")
    if predicted_features.shape[-1] != offline_features.shape[-1]:
        raise ValueError("predicted and offline feature dimensions must match")
    if existence_logits.shape != predicted_features.shape[:2]:
        raise ValueError(
            f"existence_logits must have shape {tuple(predicted_features.shape[:2])}, "
            f"got {tuple(existence_logits.shape)}"
        )
    _validate_2d_mask("offline_mask", offline_mask, tuple(offline_features.shape[:2]))
    candidate_mask = _bool_tensor(
        candidate_mask,
        device=predicted_features.device,
        shape=tuple(predicted_features.shape[:2]),
        name="candidate_mask",
    )

    if hlt_features is not None:
        hlt_features = torch.as_tensor(hlt_features, dtype=predicted_features.dtype, device=predicted_features.device)
        _validate_3d("hlt_features", hlt_features)
        if hlt_features.shape[0] != predicted_features.shape[0]:
            raise ValueError("hlt_features batch size must match predicted_features")
        hlt_mask = _bool_tensor(
            hlt_mask,
            device=predicted_features.device,
            shape=tuple(hlt_features.shape[:2]),
            name="hlt_mask",
        )

    row_core_losses = []
    row_aux_losses = []
    row_existence_losses = []
    row_count_losses = []
    row_jet_losses = []
    row_budget_losses = []
    row_chamfer_losses = []
    assignments: list[SetMatchingAssignment] = []
    matched_counts: list[float] = []
    candidate_counts: list[float] = []
    target_counts: list[float] = []
    pred_counts: list[Any] = []
    existence_target_means: list[float] = []
    method_counts = {"scipy": 0, "bruteforce": 0, "none": 0}

    batch_size = int(predicted_features.shape[0])
    for batch_index in range(batch_size):
        row_pred = predicted_features[batch_index]
        row_logits = existence_logits[batch_index]
        row_candidate_mask = candidate_mask[batch_index]
        row_target = offline_features[batch_index]
        row_target_mask = offline_mask[batch_index]
        pred_valid_indices = torch.nonzero(row_candidate_mask, as_tuple=False).flatten()
        target_valid_indices = torch.nonzero(row_target_mask, as_tuple=False).flatten()
        candidate_count = int(pred_valid_indices.numel())
        target_count = int(target_valid_indices.numel())
        candidate_counts.append(float(candidate_count))
        target_counts.append(float(target_count))
        pred_counts.append((row_logits.sigmoid() * row_candidate_mask.float()).sum())

        if candidate_count > 0 and target_count > 0:
            valid_pred = row_pred.index_select(0, pred_valid_indices)
            valid_target = row_target.index_select(0, target_valid_indices)
            cost = pairwise_core_cost(valid_pred, valid_target, config)
            rows_np, cols_np, method = _linear_sum_assignment_numpy(
                cost.detach().cpu().numpy(),
                brute_force_limit=int(config.brute_force_fallback_limit),
            )
            pred_match_indices = pred_valid_indices[torch.as_tensor(rows_np, dtype=torch.long, device=row_pred.device)]
            target_match_indices = target_valid_indices[torch.as_tensor(cols_np, dtype=torch.long, device=row_pred.device)]
        else:
            rows_np = np.zeros((0,), dtype=np.int64)
            cols_np = np.zeros((0,), dtype=np.int64)
            method = "none"
            pred_match_indices = torch.zeros((0,), dtype=torch.long, device=row_pred.device)
            target_match_indices = torch.zeros((0,), dtype=torch.long, device=row_pred.device)

        method_counts[method] = method_counts.get(method, 0) + 1
        existence_targets = torch.zeros_like(row_logits)
        if int(pred_match_indices.numel()) > 0:
            existence_targets[pred_match_indices] = 1.0

        matched_pred = row_pred.index_select(0, pred_match_indices)
        matched_target = row_target.index_select(0, target_match_indices)
        row_core_losses.append(matched_core_loss(matched_pred, matched_target, config))
        row_aux_losses.append(matched_aux_loss(matched_pred, matched_target, config))
        row_existence_losses.append(_existence_loss(row_logits[row_candidate_mask], existence_targets[row_candidate_mask], config))
        row_count_losses.append(_count_loss(row_logits, row_candidate_mask, target_count, config))
        row_jet_losses.append(jet_summary_loss(row_pred, row_logits, row_candidate_mask, row_target, row_target_mask, config))
        row_budget_losses.append(
            correction_budget_loss(
                row_pred,
                row_logits,
                row_candidate_mask,
                config,
                hlt_features=None if hlt_features is None else hlt_features[batch_index],
                hlt_mask=None if hlt_mask is None else hlt_mask[batch_index],
            )
        )
        row_chamfer_losses.append(chamfer_loss(row_pred, row_logits, row_candidate_mask, row_target, row_target_mask, config))

        matched_count = int(pred_match_indices.numel())
        matched_counts.append(float(matched_count))
        existence_target_means.append(float(existence_targets[row_candidate_mask].detach().float().mean().cpu().item()) if candidate_count else 0.0)
        assignments.append(
            SetMatchingAssignment(
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
        "correction_budget_loss": _safe_mean(row_budget_losses, predicted_features),
        "chamfer_loss": _safe_mean(row_chamfer_losses, predicted_features),
    }
    total = (
        float(config.matched_core_weight) * components["matched_core_loss"]
        + float(config.matched_aux_weight) * components["matched_aux_loss"]
        + float(config.existence_weight) * components["existence_loss"]
        + float(config.count_weight) * components["count_loss"]
        + float(config.jet_summary_weight) * components["jet_summary_loss"]
        + float(config.correction_budget_weight) * components["correction_budget_loss"]
        + float(config.chamfer_weight) * components["chamfer_loss"]
    )

    denom = max(float(batch_size), 1.0)
    diagnostics = {
        "matched_count_mean": predicted_features.new_tensor(float(np.mean(matched_counts)) if matched_counts else 0.0),
        "candidate_count_mean": predicted_features.new_tensor(float(np.mean(candidate_counts)) if candidate_counts else 0.0),
        "target_count_mean": predicted_features.new_tensor(float(np.mean(target_counts)) if target_counts else 0.0),
        "predicted_count_mean": _safe_mean(pred_counts, predicted_features),
        "existence_target_mean": predicted_features.new_tensor(
            float(np.mean(existence_target_means)) if existence_target_means else 0.0
        ),
        "matching_used_scipy_fraction": predicted_features.new_tensor(float(method_counts.get("scipy", 0)) / denom),
        "matching_used_bruteforce_fraction": predicted_features.new_tensor(float(method_counts.get("bruteforce", 0)) / denom),
        "matching_empty_fraction": predicted_features.new_tensor(float(method_counts.get("none", 0)) / denom),
    }
    return SetMatchingLossOutput(total_loss=total, components=components, diagnostics=diagnostics, assignments=assignments)


def default_set_matching_loss_config() -> SetMatchingLossConfig:
    return SetMatchingLossConfig()


__all__ = [
    "DEFAULT_CORE_FEATURE_NAMES",
    "DEFAULT_CORE_WEIGHTS",
    "EPS",
    "SetMatchingAssignment",
    "SetMatchingLossConfig",
    "SetMatchingLossOutput",
    "chamfer_loss",
    "compute_set_matching_loss",
    "correction_budget_loss",
    "default_set_matching_loss_config",
    "jet_summary_loss",
    "matched_aux_loss",
    "matched_core_loss",
    "pairwise_core_cost",
    "require_torch",
    "wrapped_delta_phi",
]
