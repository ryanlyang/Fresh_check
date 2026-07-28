"""Offline noninferiority and instance-content certification for Stage E."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, with_content_hash


BRIDGE_CERTIFICATION_CONTRACT = "retb_bridge_content_certification_v1"
BRIDGE_NONINFERIORITY_CONTRACT = "retb_bridge_offline_noninferiority_v1"


def effective_rank(bank: np.ndarray) -> dict[str, Any]:
    values = np.asarray(bank)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("effective-rank bank must be finite [N,K,D]")
    matrix = np.ascontiguousarray(
        values.reshape(values.shape[0], -1), dtype=np.float64
    )
    singular = np.linalg.svd(
        matrix, full_matrices=False, compute_uv=False
    )
    maximum = float(singular[0]) if singular.size else 0.0
    if maximum == 0.0:
        rank = 0.0
        retained = 0
    else:
        kept = singular[singular > 1.0e-12 * maximum]
        retained = int(len(kept))
        probabilities = kept / kept.sum(dtype=np.float64)
        rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    return {
        "effective_rank": rank,
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "matrix_c_contiguous": bool(matrix.flags.c_contiguous),
        "singular_values_retained": retained,
        "zero_threshold_relative_to_smax": 1.0e-12,
        "full_matrices": False,
    }


def _slot_covariances(values: np.ndarray) -> np.ndarray:
    bank = np.asarray(values, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[0] < 2:
        raise ValueError("certification covariance requires [N,K,D], N>=2")
    centered = bank - bank.mean(axis=0, keepdims=True, dtype=np.float64)
    return np.einsum("nkd,nke->kde", centered, centered) / bank.shape[0]


def _retrieval_metrics(
    query: np.ndarray,
    candidates: np.ndarray,
    candidate_identities: Sequence[Sequence[str]],
    labels: np.ndarray,
) -> dict[str, Any]:
    queries = np.asarray(query, dtype=np.float32).reshape(len(query), -1)
    banks = np.asarray(candidates, dtype=np.float32).reshape(
        len(query), 32, -1
    )
    if len(candidate_identities) != len(query):
        raise ValueError("certification retrieval identities differ")
    ranks, nearest, by_class = [], [], {index: [] for index in range(10)}
    for index, (q, rows, identities) in enumerate(
        zip(queries, banks, candidate_identities)
    ):
        if len(identities) != 32 or len(set(identities)) != 32:
            raise ValueError("certification retrieval candidate ring differs")
        qnorm = np.linalg.norm(q.astype(np.float32))
        rnorm = np.linalg.norm(rows.astype(np.float32), axis=1)
        denominator = qnorm * rnorm
        scores = np.divide(
            rows @ q,
            np.maximum(denominator, np.float32(1.0e-8)),
            out=np.zeros(32, dtype=np.float32),
            where=denominator != 0,
        )
        order = sorted(range(32), key=lambda j: (-float(scores[j]), identities[j]))
        rank = order.index(0) + 1
        ranks.append(rank)
        nearest.append(rank == 1)
        by_class[int(labels[index])].append(rank == 1)
    return {
        "same_event_top1_accuracy": float(np.mean(nearest)),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])),
        "nearest_target_accuracy_by_class": {
            str(label): (
                None if not rows else float(np.mean(rows))
            )
            for label, rows in by_class.items()
        },
        "tie_rule": "descending_similarity_then_ascending_canonical_identity",
    }


def _lapack_binding() -> dict[str, Any]:
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    return {
        "numpy_version": np.__version__,
        "numpy_show_config": stream.getvalue(),
        "svd_backend": "numpy.linalg.svd",
    }


def certify_bridge_content(
    *,
    target_mode: str,
    expert_id: str,
    pipeline_seed: int,
    moving_tokens: np.ndarray,
    t0_tokens: np.ndarray,
    frozen_moving_logits: np.ndarray,
    frozen_t0_logits: np.ndarray,
    retrieval_candidates: np.ndarray,
    retrieval_candidate_identities: Sequence[Sequence[str]],
    labels: np.ndarray,
    candidate_checkpoint_sha256: str,
    t0_checkpoint_sha256: str,
    decoded_tokens: np.ndarray | None = None,
) -> dict[str, Any]:
    moving = np.asarray(moving_tokens, dtype=np.float64)
    pure = np.asarray(t0_tokens, dtype=np.float64)
    coordinate = (
        np.asarray(decoded_tokens, dtype=np.float64)
        if decoded_tokens is not None
        else moving
    )
    if (
        tuple(coordinate.shape) != tuple(pure.shape)
        or coordinate.ndim != 3
        or len(labels) != len(pure)
        or not np.isfinite(coordinate).all()
        or not np.isfinite(pure).all()
    ):
        raise ValueError("bridge certification token population differs")
    moving_logits = np.asarray(frozen_moving_logits, dtype=np.float64)
    t0_logits = np.asarray(frozen_t0_logits, dtype=np.float64)
    if (
        moving_logits.shape != (len(pure), 10)
        or t0_logits.shape != (len(pure), 10)
        or not np.isfinite(moving_logits).all()
        or not np.isfinite(t0_logits).all()
    ):
        raise ValueError("bridge certification logits differ")
    moving_variance = coordinate.var(axis=0, dtype=np.float64)
    pure_variance = pure.var(axis=0, dtype=np.float64)
    ratio = np.divide(
        moving_variance,
        pure_variance,
        out=np.where(moving_variance == 0, 1.0, np.inf),
        where=pure_variance != 0,
    )
    moving_cov = _slot_covariances(coordinate)
    pure_cov = _slot_covariances(pure)
    covariance_errors = np.linalg.norm(
        moving_cov - pure_cov, axis=(1, 2)
    ) / np.maximum(np.linalg.norm(pure_cov, axis=(1, 2)), 1.0e-8)
    moving_rank = effective_rank(moving)
    pure_rank = effective_rank(pure)
    decoded_rank = effective_rank(coordinate)
    retrieval = _retrieval_metrics(
        moving,
        retrieval_candidates,
        retrieval_candidate_identities,
        np.asarray(labels, dtype=np.int64),
    )
    agreement = float(
        np.mean(moving_logits.argmax(axis=1) == t0_logits.argmax(axis=1))
    )
    rank_ratio = (
        1.0
        if pure_rank["effective_rank"] == 0.0
        and decoded_rank["effective_rank"] == 0.0
        else 0.0
        if pure_rank["effective_rank"] == 0.0
        else decoded_rank["effective_rank"] / pure_rank["effective_rank"]
    )
    checks = {
        "class_agreement": agreement >= 0.990,
        "median_variance_ratio": float(np.median(ratio)) >= 0.50,
        "low_variance_fraction": float(np.mean(ratio < 0.25)) <= 0.05,
        "effective_rank_ratio": rank_ratio >= 0.80,
        "relative_covariance_error": float(covariance_errors.max()) <= 0.25,
        "within_class_retrieval_accuracy": (
            retrieval["same_event_top1_accuracy"] >= 0.20
        ),
    }
    return with_content_hash(
        {
            "contract": BRIDGE_CERTIFICATION_CONTRACT,
            "schema_version": 1,
            "target_mode": target_mode,
            "expert_id": expert_id,
            "pipeline_seed": int(pipeline_seed),
            "split": "val_design",
            "event_count": int(len(pure)),
            "parents": {
                "candidate_checkpoint": require_sha256(
                    candidate_checkpoint_sha256,
                    name="candidate_checkpoint_sha256",
                ),
                "T0_checkpoint": require_sha256(
                    t0_checkpoint_sha256, name="t0_checkpoint_sha256"
                ),
            },
            "thresholds": {
                "class_agreement_min": 0.990,
                "median_variance_ratio_min": 0.50,
                "low_variance_channel_fraction_max": 0.05,
                "effective_rank_ratio_min": 0.80,
                "relative_per_slot_covariance_error_max": 0.25,
                "retrieval_accuracy_min": 0.20,
            },
            "metrics": {
                "class_agreement": agreement,
                "median_per_slot_channel_variance_ratio": float(
                    np.median(ratio)
                ),
                "low_variance_slot_channel_fraction": float(
                    np.mean(ratio < 0.25)
                ),
                "maximum_relative_per_slot_covariance_error": float(
                    covariance_errors.max()
                ),
                "effective_rank_ratio_in_T0_coordinates": rank_ratio,
                "bridge_effective_rank": moving_rank,
                "decoded_effective_rank": decoded_rank,
                "T0_effective_rank": pure_rank,
                "retrieval": retrieval,
            },
            "checks": checks,
            "bridge_content_certified": all(checks.values()),
            "dimension_changing_T2_decoded_for_T0_checks": (
                decoded_tokens is not None
            ),
            "numerical_backend": _lapack_binding(),
        }
    )


def certify_offline_noninferiority(
    *,
    target_mode: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    t0_rows: Sequence[Mapping[str, Any]],
    candidate_bundle_sha256: str,
    t0_bundle_sha256: str,
) -> dict[str, Any]:
    if (
        len(candidate_rows) != 3
        or len(t0_rows) != 3
        or {int(row["seed"]) for row in candidate_rows} != {101, 202, 303}
        or {int(row["seed"]) for row in t0_rows} != {101, 202, 303}
    ):
        raise ValueError("offline noninferiority requires exact three seeds")
    candidate = {int(row["seed"]): row for row in candidate_rows}
    pure = {int(row["seed"]): row for row in t0_rows}
    accuracy_deficits, ce_increases, class_deficits = [], [], []
    for seed in (101, 202, 303):
        accuracy_deficits.append(
            float(pure[seed]["accuracy"]) - float(candidate[seed]["accuracy"])
        )
        ce_increases.append(
            float(candidate[seed]["cross_entropy"])
            - float(pure[seed]["cross_entropy"])
        )
        candidate_eff = candidate[seed]["per_class_efficiency"]
        pure_eff = pure[seed]["per_class_efficiency"]
        if set(candidate_eff) != set(pure_eff):
            raise ValueError("offline noninferiority class coverage differs")
        class_deficits.extend(
            float(pure_eff[name]) - float(candidate_eff[name])
            for name in sorted(pure_eff)
        )
    metrics = {
        "mean_accuracy_deficit": float(np.mean(accuracy_deficits)),
        "mean_cross_entropy_increase": float(np.mean(ce_increases)),
        "worst_per_class_efficiency_deficit": float(max(class_deficits)),
    }
    checks = {
        "accuracy": metrics["mean_accuracy_deficit"] <= 0.0020,
        "cross_entropy": metrics["mean_cross_entropy_increase"] <= 0.0050,
        "per_class_efficiency": (
            metrics["worst_per_class_efficiency_deficit"] <= 0.0100
        ),
    }
    return with_content_hash(
        {
            "contract": BRIDGE_NONINFERIORITY_CONTRACT,
            "schema_version": 1,
            "target_mode": target_mode,
            "split": "val_design",
            "seeds": [101, 202, 303],
            "candidate_bundle_sha256": require_sha256(
                candidate_bundle_sha256, name="candidate_bundle_sha256"
            ),
            "T0_bundle_sha256": require_sha256(
                t0_bundle_sha256, name="t0_bundle_sha256"
            ),
            "thresholds": {
                "mean_accuracy_deficit_max": 0.0020,
                "mean_cross_entropy_increase_max": 0.0050,
                "worst_per_class_efficiency_deficit_max": 0.0100,
            },
            "metrics": metrics,
            "checks": checks,
            "offline_noninferior": all(checks.values()),
            "scientific_failure_stops_workflow": False,
        }
    )


__all__ = [
    "BRIDGE_CERTIFICATION_CONTRACT",
    "BRIDGE_NONINFERIORITY_CONTRACT",
    "certify_bridge_content",
    "certify_offline_noninferiority",
    "effective_rank",
]
