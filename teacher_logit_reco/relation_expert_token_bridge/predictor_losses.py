"""Normalization, objectives, GradNorm, and uncertainty calibration for RETB."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, with_content_hash
from .predictors import (
    NORMALIZATION_MODES,
    UNCERTAINTY_HEADS,
    uncertainty_width,
)
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PREDICTOR_OBJECTIVE_CONTRACT = "retb_predictor_objectives_v1"
UNCERTAINTY_CALIBRATION_CONTRACT = "retb_uncertainty_calibration_v1"
LOSS_COLUMNS = ("token", "cosine", "relation", "expertKD", "swapKD", "CE")
FIXED_WEIGHTS = {
    "W_TOKEN_ONLY": (1.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    "W_TOKEN_HEAVY": (1.00, 0.25, 0.10, 0.25, 0.25, 0.10),
    "W_CANONICAL": (1.00, 0.25, 0.10, 0.50, 0.50, 0.25),
    "W_TASK_HEAVY": (0.50, 0.10, 0.05, 1.00, 1.00, 0.50),
    "W_LOGIT_ONLY": (0.00, 0.00, 0.00, 1.00, 1.00, 0.50),
}


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB predictor losses")
    return torch


def build_predictor_objective_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": PREDICTOR_OBJECTIVE_CONTRACT,
            "schema_version": 1,
            "column_order": list(LOSS_COLUMNS),
            "fixed_weights": {
                name: list(values) for name, values in FIXED_WEIGHTS.items()
            },
            "token_loss": {
                "coordinate": "normalized_offline_target",
                "huber_delta": 0.5,
                "heteroscedastic": True,
                "mean_denominator": "B_times_K_times_D",
            },
            "cosine": {
                "per_slot": True,
                "zero_norm_epsilon": 1.0e-8,
            },
            "relation": {
                "row_normalized_Gram": True,
                "mean_denominator": "B_times_K_squared",
            },
            "distillation_temperature": 2.0,
            "gradnorm": {
                "id": "W_GRADNORM",
                "initial_weights": "W_CANONICAL",
                "population": "model_train_only",
                "alpha": 1.5,
                "update": "deterministic_closed_form_inverse_norm_rate",
                "clip_relative_to_initial": [0.1, 10.0],
                "renormalize_to_initial_nonzero_sum": True,
                "validation_metrics_consumed": False,
            },
            "normalization": {
                "modes": list(NORMALIZATION_MODES),
                "standard_deviation_floor": 1.0e-4,
                "nonfinite": "execution_failure",
            },
            "logit_only_faithful_token_claim_permitted": False,
            "performance_based_termination": False,
        }
    )


def validate_predictor_objective_contract(
    payload: Mapping[str, Any],
) -> str:
    from .contracts import validate_content_hash

    digest = validate_content_hash(
        payload, expected_contract=PREDICTOR_OBJECTIVE_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_predictor_objective_contract()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("predictor objective contract semantics differ")
    return digest


def normalize_tokens(
    tokens: Any,
    *,
    mean: Any,
    standard_deviation: Any,
    mode: str,
) -> Any:
    module = _require_torch()
    if mode not in NORMALIZATION_MODES:
        raise ValueError("token normalization mode is unregistered")
    values = module.as_tensor(tokens)
    center = module.as_tensor(mean, device=values.device, dtype=values.dtype)
    scale = module.as_tensor(
        standard_deviation, device=values.device, dtype=values.dtype
    )
    if (
        values.ndim != 3
        or tuple(center.shape) != tuple(values.shape[1:])
        or tuple(scale.shape) != tuple(values.shape[1:])
        or not bool(
            module.isfinite(values).all()
            and module.isfinite(center).all()
            and module.isfinite(scale).all()
        )
        or bool((scale < 0).any())
    ):
        raise ValueError("token normalizer population differs")
    normalized = (values - center) / scale.clamp_min(1.0e-4)
    if mode == "N_CLIP16":
        normalized = normalized.clamp(-16.0, 16.0)
    elif mode == "N_CLIP8":
        normalized = normalized.clamp(-8.0, 8.0)
    if not bool(module.isfinite(normalized).all()):
        raise FloatingPointError("normalized tokens are nonfinite")
    return normalized


def inverse_normalize_tokens(
    normalized: Any, *, mean: Any, standard_deviation: Any
) -> Any:
    module = _require_torch()
    values = module.as_tensor(normalized)
    center = module.as_tensor(mean, device=values.device, dtype=values.dtype)
    scale = module.as_tensor(
        standard_deviation, device=values.device, dtype=values.dtype
    )
    if (
        values.ndim != 3
        or tuple(center.shape) != tuple(values.shape[1:])
        or tuple(scale.shape) != tuple(values.shape[1:])
        or not bool(
            module.isfinite(values).all()
            and module.isfinite(center).all()
            and module.isfinite(scale).all()
        )
    ):
        raise ValueError("inverse token normalizer population differs")
    output = values * scale.clamp_min(1.0e-4) + center
    if not bool(module.isfinite(output).all()):
        raise FloatingPointError("inverse-normalized tokens are nonfinite")
    return output


def token_tail_diagnostics(
    normalized: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    qcd_discriminant: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(normalized, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("token-tail diagnostic population differs")
    truth = None if labels is None else np.asarray(labels, dtype=np.int64)
    discriminant = (
        None
        if qcd_discriminant is None
        else np.asarray(qcd_discriminant, dtype=np.float32)
    )
    if truth is not None and (
        truth.shape != (len(values),)
        or bool(((truth < 0) | (truth >= 10)).any())
    ):
        raise ValueError("token-tail labels differ")
    if discriminant is not None and (
        discriminant.shape != (len(values),)
        or not np.isfinite(discriminant).all()
    ):
        raise ValueError("token-tail discriminant differs")
    rows = {}
    for threshold in (8, 16, 32):
        element_mask = np.abs(values) > threshold
        event_mask = element_mask.reshape(len(values), -1).any(axis=1)
        row = {
            "element_count": int(element_mask.sum()),
            "event_count": int(event_mask.sum()),
            "event_fraction": float(event_mask.mean()),
        }
        if truth is not None:
            row["event_count_by_class"] = [
                int(np.sum(event_mask & (truth == class_id)))
                for class_id in range(10)
            ]
        if discriminant is not None:
            row["mean_qcd_discriminant_for_tail_events"] = (
                None
                if not event_mask.any()
                else float(discriminant[event_mask].mean(dtype=np.float64))
            )
        rows[str(threshold)] = row
    return rows


def expand_log_variance(
    log_variance: Any, *, uncertainty_head: str, token_dimension: int
) -> Any:
    module = _require_torch()
    values = module.as_tensor(log_variance)
    d = int(token_dimension)
    expected = uncertainty_width(uncertainty_head, d)
    if values.ndim != 3 or int(values.shape[-1]) != expected:
        raise ValueError("uncertainty-head output shape differs")
    if uncertainty_head == "U_SLOT":
        return values.expand(*values.shape[:-1], d)
    if uncertainty_head == "U_GROUP4":
        return values.repeat_interleave(d // 4, dim=-1)
    return values


def uncertainty_weighted_token_loss(
    prediction: Any,
    target: Any,
    log_variance: Any,
    *,
    uncertainty_head: str,
) -> Any:
    module = _require_torch()
    if tuple(prediction.shape) != tuple(target.shape) or prediction.ndim != 3:
        raise ValueError("predictor token-loss shapes differ")
    expanded = expand_log_variance(
        log_variance,
        uncertainty_head=uncertainty_head,
        token_dimension=int(prediction.shape[-1]),
    )
    error = module.nn.functional.huber_loss(
        prediction.float(),
        target.detach().float(),
        delta=0.5,
        reduction="none",
    )
    return (module.exp(-expanded.float()) * error + expanded.float()).mean()


def directional_agreement_loss(prediction: Any, target: Any) -> Any:
    module = _require_torch()
    if tuple(prediction.shape) != tuple(target.shape) or prediction.ndim != 3:
        raise ValueError("predictor directional-loss shapes differ")
    predicted = module.nn.functional.normalize(
        prediction.float(), dim=-1, eps=1.0e-8
    )
    truth = module.nn.functional.normalize(
        target.detach().float(), dim=-1, eps=1.0e-8
    )
    return (1.0 - (predicted * truth).sum(dim=-1)).mean()


def token_relation_loss(prediction: Any, target: Any) -> Any:
    module = _require_torch()
    if tuple(prediction.shape) != tuple(target.shape) or prediction.ndim != 3:
        raise ValueError("predictor relation-loss shapes differ")
    predicted = module.nn.functional.normalize(
        prediction.float(), dim=-1, eps=1.0e-8
    )
    truth = module.nn.functional.normalize(
        target.detach().float(), dim=-1, eps=1.0e-8
    )
    difference = (
        predicted @ predicted.transpose(-1, -2)
        - truth @ truth.transpose(-1, -2)
    )
    return difference.square().mean()


def temperature_two_kl(student_logits: Any, teacher_logits: Any) -> Any:
    module = _require_torch()
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError("predictor KD logit shapes differ")
    temperature = 2.0
    return (
        module.nn.functional.kl_div(
            module.log_softmax(student_logits.float() / temperature, dim=-1),
            module.softmax(
                teacher_logits.detach().float() / temperature, dim=-1
            ),
            reduction="batchmean",
        )
        * temperature**2
    )


def _weights(weight_id: str) -> dict[str, float]:
    source = (
        FIXED_WEIGHTS["W_CANONICAL"]
        if weight_id == "W_GRADNORM"
        else FIXED_WEIGHTS.get(weight_id)
    )
    if source is None:
        raise ValueError("predictor objective weight ID is unregistered")
    return dict(zip(LOSS_COLUMNS, map(float, source)))


def predictor_objective(
    *,
    weight_id: str,
    uncertainty_head: str,
    predicted_tokens: Any,
    target_tokens: Any | None,
    log_variance: Any,
    predicted_expert_logits: Any | None = None,
    target_expert_logits: Any | None = None,
    predicted_hybrid_logits: Any | None = None,
    target_hybrid_logits: Any | None = None,
    labels: Any | None = None,
    gradnorm_weights: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    module = _require_torch()
    weights = _weights(weight_id)
    if gradnorm_weights is not None:
        if weight_id != "W_GRADNORM" or set(gradnorm_weights) != set(
            LOSS_COLUMNS
        ):
            raise ValueError("GradNorm objective weights differ")
        weights = {name: float(gradnorm_weights[name]) for name in LOSS_COLUMNS}
    pieces: dict[str, Any] = {}
    token_required = any(
        weights[name] != 0.0 for name in ("token", "cosine", "relation")
    )
    if token_required and target_tokens is None:
        raise ValueError("predictor token objective lacks targets")
    if weights["token"]:
        pieces["token"] = uncertainty_weighted_token_loss(
            predicted_tokens,
            target_tokens,
            log_variance,
            uncertainty_head=uncertainty_head,
        )
    if weights["cosine"]:
        pieces["cosine"] = directional_agreement_loss(
            predicted_tokens, target_tokens
        )
    if weights["relation"]:
        pieces["relation"] = token_relation_loss(
            predicted_tokens, target_tokens
        )
    if weights["expertKD"]:
        if predicted_expert_logits is None or target_expert_logits is None:
            raise ValueError("predictor expert KD lacks frozen-head logits")
        pieces["expertKD"] = temperature_two_kl(
            predicted_expert_logits, target_expert_logits
        )
    if weights["swapKD"]:
        if predicted_hybrid_logits is None or target_hybrid_logits is None:
            raise ValueError("predictor swap KD lacks frozen-fusion logits")
        pieces["swapKD"] = temperature_two_kl(
            predicted_hybrid_logits, target_hybrid_logits
        )
    if weights["CE"]:
        if predicted_hybrid_logits is None or labels is None:
            raise ValueError("predictor CE lacks hybrid logits or labels")
        pieces["CE"] = module.nn.functional.cross_entropy(
            predicted_hybrid_logits.float(), labels.long()
        )
    if not pieces:
        raise ValueError("predictor objective has no active loss terms")
    total = sum(weights[name] * pieces[name] for name in pieces)
    if not bool(module.isfinite(total)):
        raise FloatingPointError("predictor objective is nonfinite")
    return total, {
        "terms": pieces,
        "weights": weights,
        "logit_only_semantic_control": weight_id == "W_LOGIT_ONLY",
        "faithful_token_recovery_claim_eligible": weight_id != "W_LOGIT_ONLY",
    }


class DeterministicGradNorm:
    """Closed-form deterministic balancing over model_train batches only."""

    def __init__(self, *, alpha: float = 1.5) -> None:
        if float(alpha) != 1.5:
            raise ValueError("RETB GradNorm alpha is frozen to 1.5")
        self.alpha = 1.5
        self.initial = _weights("W_GRADNORM")
        self.current = dict(self.initial)
        self.initial_losses: dict[str, float] | None = None
        self.update_count = 0

    def update(
        self,
        *,
        terms: Mapping[str, Any],
        shared_parameters: Sequence[Any],
        split: str,
    ) -> dict[str, float]:
        module = _require_torch()
        if split != "model_train":
            raise ValueError("GradNorm may adapt on model_train only")
        active = [
            name
            for name in LOSS_COLUMNS
            if self.initial[name] > 0 and name in terms
        ]
        if set(terms) != set(active) or not shared_parameters:
            raise ValueError("GradNorm active terms/shared parameters differ")
        losses = {
            name: float(terms[name].detach().float().cpu()) for name in active
        }
        if any(not math.isfinite(value) or value < 0 for value in losses.values()):
            raise FloatingPointError("GradNorm losses must be finite/nonnegative")
        losses = {name: max(value, 1.0e-12) for name, value in losses.items()}
        if self.initial_losses is None:
            self.initial_losses = dict(losses)
        norms = {}
        for name in active:
            gradients = module.autograd.grad(
                terms[name],
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared = module.zeros((), device=terms[name].device)
            for gradient in gradients:
                if gradient is not None:
                    squared = squared + gradient.float().square().sum()
            norms[name] = float(squared.sqrt().detach().cpu())
        if any(not math.isfinite(value) for value in norms.values()):
            raise FloatingPointError("GradNorm gradient norm is nonfinite")
        rates = {
            name: losses[name] / self.initial_losses[name] for name in active
        }
        rate_mean = sum(rates.values()) / len(rates)
        raw = {}
        for name in active:
            norm = max(norms[name], 1.0e-12)
            desired_rate = (rates[name] / max(rate_mean, 1.0e-12)) ** self.alpha
            raw[name] = self.initial[name] * desired_rate / norm
        initial_sum = sum(self.initial[name] for name in active)
        lower = {name: 0.1 * self.initial[name] for name in active}
        upper = {name: 10.0 * self.initial[name] for name in active}
        low_scale, high_scale = 0.0, 1.0
        while (
            sum(
                min(upper[name], max(lower[name], high_scale * raw[name]))
                for name in active
            )
            < initial_sum
        ):
            high_scale *= 2.0
        for _ in range(80):
            middle = 0.5 * (low_scale + high_scale)
            total = sum(
                min(upper[name], max(lower[name], middle * raw[name]))
                for name in active
            )
            if total < initial_sum:
                low_scale = middle
            else:
                high_scale = middle
        projected = {
            name: min(
                upper[name], max(lower[name], high_scale * raw[name])
            )
            for name in active
        }
        for name in LOSS_COLUMNS:
            self.current[name] = (
                projected[name] if name in projected else 0.0
            )
        self.update_count += 1
        return dict(self.current)

    def state_dict(self) -> dict[str, Any]:
        return {
            "contract": "retb_deterministic_gradnorm_state_v1",
            "alpha": self.alpha,
            "initial": dict(self.initial),
            "current": dict(self.current),
            "initial_losses": self.initial_losses,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("contract") != "retb_deterministic_gradnorm_state_v1"
            or float(state.get("alpha", -1)) != self.alpha
            or state.get("initial") != self.initial
            or int(state.get("update_count", -1)) < 0
        ):
            raise ValueError("GradNorm resume state differs")
        current = {name: float(state["current"][name]) for name in LOSS_COLUMNS}
        if any(value < 0 or not math.isfinite(value) for value in current.values()):
            raise ValueError("GradNorm resume weights differ")
        self.current = current
        self.initial_losses = (
            None
            if state.get("initial_losses") is None
            else {
                name: float(value)
                for name, value in state["initial_losses"].items()
            }
        )
        self.update_count = int(state["update_count"])


def _group_channel_slices(uncertainty_head: str, d: int):
    if uncertainty_head == "U_SLOT":
        return (slice(0, d),)
    if uncertainty_head == "U_GROUP4":
        width = d // 4
        return tuple(slice(index * width, (index + 1) * width) for index in range(4))
    if uncertainty_head == "U_DIAGONAL":
        return tuple(slice(index, index + 1) for index in range(d))
    raise ValueError("uncertainty calibration head differs")


def _clipped_log_variance_offset(
    source_log_variance: np.ndarray,
    mean_squared_error: np.ndarray,
) -> float:
    """Minimize mean(exp(-clip(s+a))*e + clip(s+a)) exactly by pieces."""

    s = np.asarray(source_log_variance, dtype=np.float64).reshape(-1)
    e = np.asarray(mean_squared_error, dtype=np.float64).reshape(-1)
    if (
        s.shape != e.shape
        or len(s) == 0
        or not np.isfinite(s).all()
        or not np.isfinite(e).all()
        or bool((e < 0).any())
    ):
        raise ValueError("uncertainty offset population differs")
    order = np.argsort(s, kind="stable")
    sorted_s = s[order]
    sorted_e = e[order]
    weighted = sorted_e * np.exp(-sorted_s)
    prefix_e = np.concatenate(([0.0], np.cumsum(sorted_e, dtype=np.float64)))
    prefix_weighted = np.concatenate(
        ([0.0], np.cumsum(weighted, dtype=np.float64))
    )
    prefix_s = np.concatenate(
        ([0.0], np.cumsum(sorted_s, dtype=np.float64))
    )
    count = len(sorted_s)

    def objective(offset: float) -> float:
        low_stop = int(np.searchsorted(sorted_s, -8.0 - offset, side="right"))
        high_start = int(
            np.searchsorted(sorted_s, 4.0 - offset, side="left")
        )
        low_e = prefix_e[low_stop]
        active_e_exp_minus_s = (
            prefix_weighted[high_start] - prefix_weighted[low_stop]
        )
        active_s = prefix_s[high_start] - prefix_s[low_stop]
        active_count = high_start - low_stop
        high_e = prefix_e[count] - prefix_e[high_start]
        total = (
            low_e * math.exp(8.0)
            - 8.0 * low_stop
            + math.exp(-offset) * active_e_exp_minus_s
            + active_s
            + offset * active_count
            + high_e * math.exp(-4.0)
            + 4.0 * (count - high_start)
        )
        return total / count

    # Since s is already clipped to [-8,4], offsets outside [-12,12] add
    # no new calibrated values. Breakpoints delimit regions with a fixed
    # active set, and each such region has at most one stationary point.
    breakpoints = np.unique(
        np.clip(
            np.concatenate(
                (
                    np.asarray([-12.0, 12.0]),
                    -8.0 - sorted_s,
                    4.0 - sorted_s,
                )
            ),
            -12.0,
            12.0,
        )
    )
    candidates = [float(value) for value in breakpoints]
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        middle = float((left + right) * 0.5)
        low_stop = int(
            np.searchsorted(sorted_s, -8.0 - middle, side="right")
        )
        high_start = int(
            np.searchsorted(sorted_s, 4.0 - middle, side="left")
        )
        active_count = high_start - low_stop
        if active_count == 0:
            continue
        active_weighted = (
            prefix_weighted[high_start] - prefix_weighted[low_stop]
        )
        if active_weighted <= 0:
            continue
        stationary = math.log(active_weighted / active_count)
        if float(left) <= stationary <= float(right):
            candidates.append(stationary)
    return min(
        candidates,
        key=lambda value: (objective(value), abs(value), value),
    )


def fit_uncertainty_calibration(
    *,
    expert_id: str,
    uncertainty_head: str,
    predicted_tokens: np.ndarray,
    target_tokens: np.ndarray,
    log_variance: np.ndarray,
    predictor_checkpoint_sha256: str,
    predictor_registration_sha256: str,
    predictor_inference_manifest_sha256: str,
    target_cache_manifest_sha256: str,
    identity_manifest_sha256: str,
) -> dict[str, Any]:
    prediction = np.asarray(predicted_tokens, dtype=np.float64)
    target = np.asarray(target_tokens, dtype=np.float64)
    source_log_variance = np.asarray(log_variance, dtype=np.float64)
    if (
        expert_id not in EXPERT_ORDER
        or uncertainty_head not in UNCERTAINTY_HEADS
        or prediction.shape != target.shape
        or prediction.ndim != 3
        or source_log_variance.shape
        != prediction.shape[:2]
        + (uncertainty_width(uncertainty_head, prediction.shape[-1]),)
        or len(prediction) == 0
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
        or not np.isfinite(source_log_variance).all()
    ):
        raise ValueError("uncertainty calibration population differs")
    error2 = np.square(prediction - target)
    offsets, group_rmse, group_nll = [], [], []
    for group, channel_slice in enumerate(
        _group_channel_slices(uncertainty_head, prediction.shape[-1])
    ):
        s = source_log_variance[:, :, group]
        group_error = error2[:, :, channel_slice]
        offset = _clipped_log_variance_offset(
            s,
            group_error.mean(axis=-1, dtype=np.float64),
        )
        offsets.append(offset)
        calibrated = np.clip(s + offset, -8.0, 4.0)
        group_rmse.append(
            math.sqrt(float(group_error.mean(dtype=np.float64)))
        )
        group_nll.append(
            float(
                (
                    np.exp(-calibrated[:, :, None]) * group_error
                    + calibrated[:, :, None]
                ).mean(dtype=np.float64)
            )
        )
    calibrated_log_variance = np.clip(
        source_log_variance + np.asarray(offsets)[None, None, :],
        -8.0,
        4.0,
    )
    expanded = np.empty_like(error2)
    for group, channel_slice in enumerate(
        _group_channel_slices(uncertainty_head, prediction.shape[-1])
    ):
        expanded[:, :, channel_slice] = calibrated_log_variance[
            :, :, group : group + 1
        ]
    uncertainty_score = np.exp(expanded).mean(axis=(1, 2))
    event_rmse = np.sqrt(error2.mean(axis=(1, 2)))
    order = np.argsort(uncertainty_score, kind="stable")
    quantile_rows = []
    for quantile_index, indices in enumerate(np.array_split(order, 10)):
        quantile_rows.append(
            {
                "quantile_index": quantile_index,
                "event_count": int(len(indices)),
                "mean_expected_rmse": (
                    None
                    if len(indices) == 0
                    else float(
                        np.sqrt(uncertainty_score[indices]).mean(
                            dtype=np.float64
                        )
                    )
                ),
                "mean_observed_rmse": (
                    None
                    if len(indices) == 0
                    else float(event_rmse[indices].mean(dtype=np.float64))
                ),
            }
        )
    coverage_rows = []
    for numerator in range(1, 11):
        retained = order[: math.ceil(len(order) * numerator / 10)]
        coverage_rows.append(
            {
                "coverage": numerator / 10.0,
                "event_count": int(len(retained)),
                "mean_expected_rmse": float(
                    np.sqrt(uncertainty_score[retained]).mean(
                        dtype=np.float64
                    )
                ),
                "observed_rmse": float(
                    np.sqrt(
                        error2[retained].mean(dtype=np.float64)
                    )
                ),
            }
        )
    return with_content_hash(
        {
            "contract": UNCERTAINTY_CALIBRATION_CONTRACT,
            "schema_version": 1,
            "expert_id": expert_id,
            "uncertainty_head": uncertainty_head,
            "fit_split": "val_design",
            "labels_consumed": False,
            "event_count": int(len(prediction)),
            "additive_offset_by_group": offsets,
            "calibrated_expected_rmse_by_group": [
                math.sqrt(
                    float(
                        np.exp(
                            np.clip(
                                source_log_variance[:, :, group]
                                + offsets[group],
                                -8.0,
                                4.0,
                            )
                        ).mean(dtype=np.float64)
                    )
                )
                for group in range(len(offsets))
            ],
            "observed_rmse_by_group": group_rmse,
            "gaussian_nll_by_group": group_nll,
            "error_by_uncertainty_decile": quantile_rows,
            "coverage_error_curve": coverage_rows,
            "offset_solution": "piecewise_exact_clipped_gaussian_NLL",
            "post_offset_clip": [-8.0, 4.0],
            "parents": {
                "predictor_checkpoint": require_sha256(
                    predictor_checkpoint_sha256,
                    name="predictor_checkpoint_sha256",
                ),
                "predictor_registration": require_sha256(
                    predictor_registration_sha256,
                    name="predictor_registration_sha256",
                ),
                "predictor_inference_manifest": require_sha256(
                    predictor_inference_manifest_sha256,
                    name="predictor_inference_manifest_sha256",
                ),
                "target_cache_manifest": require_sha256(
                    target_cache_manifest_sha256,
                    name="target_cache_manifest_sha256",
                ),
                "identity_manifest": require_sha256(
                    identity_manifest_sha256,
                    name="identity_manifest_sha256",
                ),
            },
            "frozen_before_stack_val": True,
        }
    )


def apply_uncertainty_calibration(
    log_variance: Any, calibration: Mapping[str, Any]
) -> Any:
    module = _require_torch()
    values = module.as_tensor(log_variance)
    if calibration.get("contract") != UNCERTAINTY_CALIBRATION_CONTRACT:
        raise ValueError("uncertainty calibration contract differs")
    offsets = module.as_tensor(
        calibration["additive_offset_by_group"],
        dtype=values.dtype,
        device=values.device,
    )
    if values.ndim != 3 or values.shape[-1] != len(offsets):
        raise ValueError("uncertainty calibration shape differs")
    return (values + offsets[None, None]).clamp(-8.0, 4.0)


def validate_uncertainty_calibration(payload: Mapping[str, Any]) -> str:
    from .contracts import validate_content_hash

    digest = validate_content_hash(
        payload, expected_contract=UNCERTAINTY_CALIBRATION_CONTRACT
    )
    head = payload["uncertainty_head"]
    offsets = payload["additive_offset_by_group"]
    expected_width = {
        "U_SLOT": 1,
        "U_GROUP4": 4,
        "U_DIAGONAL": (
            len(offsets) if len(offsets) in {64, 128} else -1
        ),
    }.get(head, -1)
    if (
        payload["expert_id"] not in EXPERT_ORDER
        or payload["fit_split"] != "val_design"
        or payload["labels_consumed"]
        or not payload["frozen_before_stack_val"]
        or len(offsets) != expected_width
        or any(not math.isfinite(float(value)) for value in offsets)
        or payload.get("offset_solution")
        != "piecewise_exact_clipped_gaussian_NLL"
        or payload.get("post_offset_clip") != [-8.0, 4.0]
        or [row.get("coverage") for row in payload.get(
            "coverage_error_curve", []
        )]
        != [index / 10.0 for index in range(1, 11)]
        or set(payload["parents"])
        != {
            "predictor_checkpoint",
            "predictor_registration",
            "predictor_inference_manifest",
            "target_cache_manifest",
            "identity_manifest",
        }
    ):
        raise ValueError("uncertainty calibration semantics differ")
    return digest


__all__ = [
    "DeterministicGradNorm",
    "FIXED_WEIGHTS",
    "LOSS_COLUMNS",
    "PREDICTOR_OBJECTIVE_CONTRACT",
    "UNCERTAINTY_CALIBRATION_CONTRACT",
    "apply_uncertainty_calibration",
    "build_predictor_objective_contract",
    "directional_agreement_loss",
    "expand_log_variance",
    "fit_uncertainty_calibration",
    "inverse_normalize_tokens",
    "normalize_tokens",
    "predictor_objective",
    "temperature_two_kl",
    "token_relation_loss",
    "token_tail_diagnostics",
    "uncertainty_weighted_token_loss",
    "validate_uncertainty_calibration",
    "validate_predictor_objective_contract",
]
