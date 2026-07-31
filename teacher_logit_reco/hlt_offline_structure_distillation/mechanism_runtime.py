"""Executable Stage-G parity and identity-paired mechanism diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .combinations import build_mechanism_result

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def evaluate_auxiliary_head_removal(
    model: Any, loader: Any, *, device: str | Any
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for mechanism evaluation")
    resolved = torch.device(device)
    model.to(resolved).eval()
    maximum = 0.0
    identities = []
    with torch.no_grad():
        for raw in loader:
            batch = {
                key: value.to(resolved) if hasattr(value, "to") else value
                for key, value in raw.items()
            }
            vectors = batch.get("lorentz_vectors", batch.get("vectors"))
            points = batch.get("points", batch["features"][:, 15:17])
            complete, _ = model.forward_with_auxiliaries(
                points, batch["features"], vectors, batch["mask"]
            )
            stripped = model.classifier(
                points, batch["features"], vectors, batch["mask"]
            )
            difference = float((complete - stripped).abs().max().cpu())
            maximum = max(maximum, difference)
            identities.extend(str(value) for value in raw["event_identities"])
    if len(identities) != len(set(identities)):
        raise ValueError("mechanism parity identities are duplicated")
    if maximum != 0.0:
        raise RuntimeError("auxiliary-head removal changed classifier logits")
    return {
        "identity_count": len(identities),
        "maximum_absolute_logit_difference": maximum,
        "logits_bitwise_equal": True,
        "auxiliary_heads_removed": True,
    }


def eventwise_error_gain_tracking(
    *,
    identities: Sequence[str],
    labels: np.ndarray,
    baseline_logits: np.ndarray,
    candidate_logits: np.ndarray,
    target_error: np.ndarray,
) -> dict[str, Any]:
    ids = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    baseline = np.asarray(baseline_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    error = np.asarray(target_error, dtype=np.float64)
    if (
        len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
        or baseline.shape != (len(ids), 10)
        or candidate.shape != baseline.shape
        or error.shape != (len(ids),)
        or not np.isfinite(error).all()
    ):
        raise ValueError("eventwise mechanism arrays differ")
    gain = (
        (candidate.argmax(axis=1) == truth).astype(np.float64)
        - (baseline.argmax(axis=1) == truth).astype(np.float64)
    )
    x, y = error - error.mean(), gain - gain.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    correlation = None if denominator == 0 else float((x @ y) / denominator)
    decile_edges = np.quantile(error, np.linspace(0, 1, 11), method="linear")
    deciles = []
    for index in range(10):
        selected = (error >= decile_edges[index]) & (
            error <= decile_edges[index + 1]
            if index == 9
            else error < decile_edges[index + 1]
        )
        deciles.append(
            {
                "decile": index,
                "count": int(selected.sum()),
                "mean_classification_gain": (
                    None if not bool(selected.any()) else float(gain[selected].mean())
                ),
                "mean_target_error": (
                    None if not bool(selected.any()) else float(error[selected].mean())
                ),
            }
        )
    return {
        "identity_count": len(ids),
        "target_error_classification_gain_pearson": correlation,
        "target_error_decile_edges": decile_edges.tolist(),
        "deciles": deciles,
        "classification_gain_definition": (
            "candidate_correct_indicator_minus_baseline_correct_indicator"
        ),
    }


def mechanism_result_from_measurement(
    *,
    plan: Mapping[str, Any],
    intervention_id: str,
    measurement: Mapping[str, Any],
    evidence_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    status: str = "completed",
) -> dict[str, Any]:
    return build_mechanism_result(
        plan=plan,
        intervention_id=intervention_id,
        status=status,
        measurements=measurement,
        evidence_hashes=evidence_hashes,
        source=source,
    )


__all__ = [
    "evaluate_auxiliary_head_removal",
    "eventwise_error_gain_tracking",
    "mechanism_result_from_measurement",
]
