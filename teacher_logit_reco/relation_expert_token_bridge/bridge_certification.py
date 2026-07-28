"""Offline noninferiority and instance-content certification for Stage E."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .bridge_targets import deterministic_within_class_negatives
from .registry import EXPERT_ORDER


BRIDGE_CERTIFICATION_CONTRACT = "retb_bridge_content_certification_v2"
BRIDGE_NONINFERIORITY_CONTRACT = "retb_bridge_offline_noninferiority_v2"
BRIDGE_ELIGIBILITY_CONTRACT = "retb_bridge_candidate_eligibility_v1"


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


def _retrieval_metrics_from_ring(
    *,
    query: np.ndarray,
    moving_banks: np.ndarray,
    identities: Sequence[str],
    labels: np.ndarray,
    pipeline_seed: int,
) -> dict[str, Any]:
    queries = np.asarray(query, dtype=np.float32).reshape(len(query), -1)
    banks = np.asarray(moving_banks, dtype=np.float32).reshape(
        len(query), -1
    )
    canonical = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    index_by_identity = {
        identity: index for index, identity in enumerate(canonical)
    }
    class_rings = {
        label: [
            identity
            for identity, current in zip(canonical, truth)
            if int(current) == label
        ]
        for label in range(10)
    }
    ranks, nearest = [], []
    by_class = {index: [] for index in range(10)}
    for index, (query_row, identity, label) in enumerate(
        zip(queries, canonical, truth)
    ):
        negatives = deterministic_within_class_negatives(
            identity=identity,
            class_label=int(label),
            class_rings=class_rings,
            pipeline_seed=int(pipeline_seed),
            certification=True,
        )
        candidate_identities = (identity, *negatives)
        candidate_rows = banks[
            [index_by_identity[value] for value in candidate_identities]
        ]
        qnorm = np.linalg.norm(query_row.astype(np.float32))
        rnorm = np.linalg.norm(candidate_rows.astype(np.float32), axis=1)
        denominator = qnorm * rnorm
        scores = np.divide(
            candidate_rows @ query_row,
            np.maximum(denominator, np.float32(1.0e-8)),
            out=np.zeros(32, dtype=np.float32),
            where=denominator != 0,
        )
        order = sorted(
            range(32),
            key=lambda candidate: (
                -float(scores[candidate]),
                candidate_identities[candidate],
            ),
        )
        rank = order.index(0) + 1
        ranks.append(rank)
        nearest.append(rank == 1)
        by_class[int(label)].append(rank == 1)
    return {
        "same_event_top1_accuracy": float(np.mean(nearest)),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])),
        "nearest_target_accuracy_by_class": {
            str(label): None if not rows else float(np.mean(rows))
            for label, rows in by_class.items()
        },
        "tie_rule": "descending_similarity_then_ascending_canonical_identity",
        "candidate_materialization": "streaming_exact_ring_lookup",
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
    shape_id: str,
    pipeline_seed: int,
    moving_tokens: np.ndarray,
    t0_tokens: np.ndarray,
    predicted_hlt_tokens: np.ndarray,
    frozen_moving_logits: Mapping[str, np.ndarray],
    frozen_t0_logits: Mapping[str, np.ndarray],
    identities: Sequence[str],
    labels: np.ndarray,
    candidate_checkpoint_sha256: str,
    t0_checkpoint_sha256: str,
    identity_manifest_sha256: str,
    coordinate_normalizer_sha256: str,
    t0_normalizer_sha256: str,
    decoded_tokens: np.ndarray | None = None,
) -> dict[str, Any]:
    if (
        target_mode
        not in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE", "T2_PROJECT"}
        or expert_id not in EXPERT_ORDER
        or int(pipeline_seed) not in {101, 202, 303}
    ):
        raise ValueError("bridge certification identity is not registered")
    moving = np.asarray(moving_tokens, dtype=np.float64)
    pure = np.asarray(t0_tokens, dtype=np.float64)
    predicted = np.asarray(predicted_hlt_tokens, dtype=np.float32)
    canonical_identities = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    coordinate = (
        np.asarray(decoded_tokens, dtype=np.float64)
        if decoded_tokens is not None
        else moving
    )
    if (
        moving.ndim != 3
        or moving.shape[1] not in {1, 2, 4, 8, 16}
        or moving.shape[2] not in {64, 128}
        or predicted.shape != moving.shape
        or
        tuple(coordinate.shape) != tuple(pure.shape)
        or coordinate.ndim != 3
        or len(canonical_identities) != len(pure)
        or len(canonical_identities) != len(set(canonical_identities))
        or truth.shape != (len(pure),)
        or bool(((truth < 0) | (truth >= 10)).any())
        or not np.isfinite(moving).all()
        or not np.isfinite(predicted).all()
        or not np.isfinite(coordinate).all()
        or not np.isfinite(pure).all()
    ):
        raise ValueError("bridge certification token population differs")
    class_counts = np.bincount(truth, minlength=10)
    if (
        bool((class_counts < 32).any())
        or len(set(int(value) for value in class_counts)) != 1
    ):
        raise ValueError(
            "bridge certification requires balanced classes with 32+ events"
        )
    if set(frozen_moving_logits) != {"expert", "fusion"} or set(
        frozen_t0_logits
    ) != {"expert", "fusion"}:
        raise ValueError("bridge certification requires expert and fusion logits")
    agreements: dict[str, float] = {}
    for consumer in ("expert", "fusion"):
        moving_logits = np.asarray(
            frozen_moving_logits[consumer], dtype=np.float64
        )
        t0_logits = np.asarray(frozen_t0_logits[consumer], dtype=np.float64)
        if (
            moving_logits.shape != (len(pure), 10)
            or t0_logits.shape != (len(pure), 10)
            or not np.isfinite(moving_logits).all()
            or not np.isfinite(t0_logits).all()
        ):
            raise ValueError("bridge certification logits differ")
        agreements[consumer] = float(
            np.mean(moving_logits.argmax(axis=1) == t0_logits.argmax(axis=1))
        )
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
    retrieval = _retrieval_metrics_from_ring(
        query=predicted,
        moving_banks=moving,
        identities=canonical_identities,
        labels=truth,
        pipeline_seed=int(pipeline_seed),
    )
    agreement = min(agreements.values())
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
            "schema_version": 2,
            "target_mode": target_mode,
            "expert_id": expert_id,
            "shape_id": str(shape_id),
            "pipeline_seed": int(pipeline_seed),
            "split": "val_design",
            "event_count": int(len(pure)),
            "token_inputs": "train_normalized_by_bound_coordinate_contracts",
            "parents": {
                "candidate_checkpoint": require_sha256(
                    candidate_checkpoint_sha256,
                    name="candidate_checkpoint_sha256",
                ),
                "T0_checkpoint": require_sha256(
                    t0_checkpoint_sha256, name="t0_checkpoint_sha256"
                ),
                "identity_manifest": require_sha256(
                    identity_manifest_sha256,
                    name="identity_manifest_sha256",
                ),
                "coordinate_normalizer": require_sha256(
                    coordinate_normalizer_sha256,
                    name="coordinate_normalizer_sha256",
                ),
                "T0_normalizer": require_sha256(
                    t0_normalizer_sha256,
                    name="t0_normalizer_sha256",
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
                "class_agreement_by_consumer": agreements,
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
            "retrieval_population": {
                "query": "predicted_HLT_bank",
                "candidate_bank": "moving_offline_bank",
                "negative_namespace": "retb_t1_cert_negatives_v1",
                "candidate_count": 32,
                "candidate_identities_derived_internally": True,
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
        target_mode
        not in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE", "T2_PROJECT"}
        or
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
        scalars = [
            float(candidate[seed]["accuracy"]),
            float(pure[seed]["accuracy"]),
            float(candidate[seed]["cross_entropy"]),
            float(pure[seed]["cross_entropy"]),
            *[float(value) for value in candidate_eff.values()],
            *[float(value) for value in pure_eff.values()],
        ]
        if (
            set(candidate_eff) != set(pure_eff)
            or len(candidate_eff) != 10
            or not np.isfinite(scalars).all()
            or any(
                value < 0.0 or value > 1.0
                for value in (
                    float(candidate[seed]["accuracy"]),
                    float(pure[seed]["accuracy"]),
                    *[float(value) for value in candidate_eff.values()],
                    *[float(value) for value in pure_eff.values()],
                )
            )
            or float(candidate[seed]["cross_entropy"]) < 0.0
            or float(pure[seed]["cross_entropy"]) < 0.0
        ):
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
            "schema_version": 2,
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


def validate_bridge_content_certification(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=BRIDGE_CERTIFICATION_CONTRACT
    )
    metrics = payload["metrics"]
    retrieval = metrics["retrieval"]
    checks = {
        "class_agreement": float(metrics["class_agreement"]) >= 0.990,
        "median_variance_ratio": (
            float(metrics["median_per_slot_channel_variance_ratio"]) >= 0.50
        ),
        "low_variance_fraction": (
            float(metrics["low_variance_slot_channel_fraction"]) <= 0.05
        ),
        "effective_rank_ratio": (
            float(metrics["effective_rank_ratio_in_T0_coordinates"]) >= 0.80
        ),
        "relative_covariance_error": (
            float(metrics["maximum_relative_per_slot_covariance_error"])
            <= 0.25
        ),
        "within_class_retrieval_accuracy": (
            float(retrieval["same_event_top1_accuracy"]) >= 0.20
        ),
    }
    if (
        payload["checks"] != checks
        or bool(payload["bridge_content_certified"]) != all(checks.values())
        or payload["target_mode"]
        not in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE", "T2_PROJECT"}
        or payload["expert_id"] not in EXPERT_ORDER
        or int(payload["pipeline_seed"]) not in {101, 202, 303}
        or payload["retrieval_population"]["negative_namespace"]
        != "retb_t1_cert_negatives_v1"
    ):
        raise ValueError("bridge content certification semantics differ")
    return digest


def validate_bridge_noninferiority(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=BRIDGE_NONINFERIORITY_CONTRACT
    )
    metrics = payload["metrics"]
    checks = {
        "accuracy": float(metrics["mean_accuracy_deficit"]) <= 0.0020,
        "cross_entropy": (
            float(metrics["mean_cross_entropy_increase"]) <= 0.0050
        ),
        "per_class_efficiency": (
            float(metrics["worst_per_class_efficiency_deficit"]) <= 0.0100
        ),
    }
    if (
        payload["checks"] != checks
        or bool(payload["offline_noninferior"]) != all(checks.values())
        or payload["target_mode"]
        not in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE", "T2_PROJECT"}
        or payload["seeds"] != [101, 202, 303]
    ):
        raise ValueError("bridge noninferiority semantics differ")
    return digest


def build_bridge_candidate_eligibility(
    *,
    target_mode: str,
    expert_id: str,
    shape_id: str,
    checkpoint_hashes_by_seed: Mapping[int, str],
    noninferiority: Mapping[str, Any] | None = None,
    content_certifications: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if (
        target_mode
        not in {
            "T0_PURE",
            "T1_ANCHORED_BRIDGE",
            "T1_TASK_BRIDGE",
            "T2_PROJECT",
        }
        or expert_id not in EXPERT_ORDER
        or set(int(seed) for seed in checkpoint_hashes_by_seed)
        != {101, 202, 303}
    ):
        raise ValueError("bridge eligibility identity differs")
    checkpoints = {
        str(seed): require_sha256(
            checkpoint_hashes_by_seed[seed],
            name=f"checkpoint_hashes_by_seed.{seed}",
        )
        for seed in (101, 202, 303)
    }
    if target_mode == "T0_PURE":
        if noninferiority is not None or content_certifications:
            raise ValueError("T0 eligibility cannot consume bridge certificates")
        noninferior = True
        content_hashes: list[str] = []
        content_certified: bool | None = None
        noninferiority_hash: str | None = None
    else:
        if noninferiority is None or len(content_certifications) != 3:
            raise ValueError("bridge eligibility requires three-seed certificates")
        noninferiority_hash = validate_bridge_noninferiority(noninferiority)
        if noninferiority["target_mode"] != target_mode:
            raise ValueError("bridge noninferiority mode differs")
        noninferior = bool(noninferiority["offline_noninferior"])
        content_by_seed = {}
        for artifact in content_certifications:
            digest = validate_bridge_content_certification(artifact)
            seed = int(artifact["pipeline_seed"])
            if (
                seed in content_by_seed
                or seed not in {101, 202, 303}
                or artifact["target_mode"] != target_mode
                or artifact["expert_id"] != expert_id
                or artifact["shape_id"] != str(shape_id)
            ):
                raise ValueError("bridge content certification lineage differs")
            content_by_seed[seed] = (digest, artifact)
        if set(content_by_seed) != {101, 202, 303}:
            raise ValueError("bridge content certification seed coverage differs")
        content_hashes = [
            content_by_seed[seed][0] for seed in (101, 202, 303)
        ]
        content_certified = all(
            bool(content_by_seed[seed][1]["bridge_content_certified"])
            for seed in (101, 202, 303)
        )
    return with_content_hash(
        {
            "contract": BRIDGE_ELIGIBILITY_CONTRACT,
            "schema_version": 1,
            "target_mode": target_mode,
            "expert_id": expert_id,
            "shape_id": str(shape_id),
            "pipeline_seeds": [101, 202, 303],
            "checkpoint_hashes_by_seed": checkpoints,
            "noninferiority_sha256": noninferiority_hash,
            "content_certification_hashes": content_hashes,
            "offline_noninferior": noninferior,
            "bridge_content_certified": content_certified,
            "maximum_performance_eligible": noninferior,
            "representation_preserving_claim_eligible": (
                target_mode == "T0_PURE"
                or (
                    target_mode
                    in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE"}
                    and bool(content_certified)
                )
            ),
            "learned_bridge_coordinate_claim": target_mode == "T2_PROJECT",
            "scientific_failure_stops_workflow": False,
        }
    )


def validate_bridge_candidate_eligibility(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=BRIDGE_ELIGIBILITY_CONTRACT
    )
    mode = payload["target_mode"]
    content = payload["bridge_content_certified"]
    expected_representation = mode == "T0_PURE" or (
        mode in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE"}
        and bool(content)
    )
    if (
        mode
        not in {
            "T0_PURE",
            "T1_ANCHORED_BRIDGE",
            "T1_TASK_BRIDGE",
            "T2_PROJECT",
        }
        or payload["expert_id"] not in EXPERT_ORDER
        or payload["pipeline_seeds"] != [101, 202, 303]
        or set(payload["checkpoint_hashes_by_seed"])
        != {"101", "202", "303"}
        or bool(payload["maximum_performance_eligible"])
        != bool(payload["offline_noninferior"])
        or bool(payload["representation_preserving_claim_eligible"])
        != expected_representation
        or bool(payload["learned_bridge_coordinate_claim"])
        != (mode == "T2_PROJECT")
    ):
        raise ValueError("bridge eligibility semantics differ")
    return digest


__all__ = [
    "BRIDGE_CERTIFICATION_CONTRACT",
    "BRIDGE_ELIGIBILITY_CONTRACT",
    "BRIDGE_NONINFERIORITY_CONTRACT",
    "build_bridge_candidate_eligibility",
    "certify_bridge_content",
    "certify_offline_noninferiority",
    "effective_rank",
    "validate_bridge_candidate_eligibility",
    "validate_bridge_content_certification",
    "validate_bridge_noninferiority",
]
