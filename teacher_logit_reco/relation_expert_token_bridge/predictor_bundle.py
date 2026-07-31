"""Deterministic joint selection of complete seven-expert predictor bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any

import numpy as np

from .contracts import (
    bind_source,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .predictors import (
    ARCHITECTURES,
    CONTEXTS,
    NORMALIZATION_MODES,
    UNCERTAINTY_HEADS,
)
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PREDICTOR_CANDIDATE_CONTRACT = "retb_joint_predictor_candidate_v1"
TARGET_COORDINATE_CONTRACT = "retb_locked_predictor_coordinate_v1"
PREDICTOR_CACHE_INDEX_CONTRACT = "retb_predictor_cache_index_v1"
BUNDLE_SEARCH_POLICY_CONTRACT = "retb_predictor_bundle_search_policy_v1"
BUNDLE_SEARCH_CONTRACT = "retb_predictor_bundle_search_v1"
PREDICTOR_BUNDLE_LOCK_CONTRACT = "retb_predictor_bundle_lock_v1"
CARRIED_PREDICTOR_BUNDLE_INDEX_CONTRACT = (
    "retb_carried_predictor_bundle_index_v1"
)
PIPELINE_SEEDS = (101, 202, 303)
BEAM_WIDTH = 32
ACCURACY_WINDOW = 1.0e-4


def shared_predictor_configuration_id(
    *,
    architecture: str,
    context: str,
    objective_id: str,
    uncertainty_head: str,
    normalization_mode: str,
    learning_rate: float,
    dropout: float,
    hlt_evidence_mode: str,
) -> str:
    return canonical_sha256(
        {
            "architecture": architecture,
            "context": context,
            "objective_id": objective_id,
            "uncertainty_head": uncertainty_head,
            "normalization_mode": normalization_mode,
            "learning_rate": float(learning_rate),
            "dropout": float(dropout),
            "hlt_evidence_mode": hlt_evidence_mode,
        }
    )


def build_bundle_search_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": BUNDLE_SEARCH_POLICY_CONTRACT,
            "schema_version": 1,
            "expert_order": list(EXPERT_ORDER),
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "beam_width": BEAM_WIDTH,
            "accuracy_window": ACCURACY_WINDOW,
            "coordinate_search": (
                "separate_search_per_locked_target_coordinate_then_global_rank"
            ),
            "initial_tuple": (
                "individually_best_one_predicted_six_oracle_hybrid_per_expert"
            ),
            "ranking": [
                "within_0.0001_of_global_maximum_mean_accuracy",
                "lower_mean_cross_entropy",
                "lower_mean_normalized_token_error",
                "lower_inference_FLOPs",
                "lower_parameter_count",
                "lexicographic_seven_candidate_tuple",
            ],
            "controls": [
                "INDIVIDUAL_HYBRID_DEFAULT",
                "GLOBAL_SHARED_CONFIGURATION",
                "ALL_W_CANONICAL",
            ],
            "fusion_or_predictor_retraining_permitted": False,
            "per_seed_configuration_selection_permitted": False,
            "all_negative_campaign_still_selects": True,
            "step11_start_requires_immutable_bundle_lock": True,
            "performance_based_termination": False,
        }
    )


def validate_bundle_search_policy(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=BUNDLE_SEARCH_POLICY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_bundle_search_policy()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("predictor bundle-search policy semantics differ")
    return digest


def _seed_artifacts(
    artifacts: Mapping[int | str, Mapping[str, Any]],
) -> dict[str, Any]:
    if {int(seed) for seed in artifacts} != set(PIPELINE_SEEDS):
        raise ValueError("predictor candidate seed coverage differs")
    required = {
        "predictor_registration",
        "predictor_checkpoint",
        "inference_manifest",
        "uncertainty_calibration",
        "capacity_report",
        "identity_order_sha256",
    }
    output = {}
    for seed in PIPELINE_SEEDS:
        row = artifacts.get(seed, artifacts.get(str(seed)))
        if row is None or set(row) != required:
            raise ValueError("predictor candidate seed artifacts differ")
        output[str(seed)] = {
            name: require_sha256(
                row[name], name=f"seed_artifacts.{seed}.{name}"
            )
            for name in sorted(required)
        }
    return output


def build_predictor_candidate(
    *,
    candidate_id: str,
    expert_id: str,
    coordinate_id: str,
    target_mode: str,
    shape_id: str,
    token_count: int,
    token_dimension: int,
    architecture: str,
    context: str,
    objective_id: str,
    uncertainty_head: str,
    normalization_mode: str,
    learning_rate: float,
    dropout: float,
    hlt_evidence_mode: str,
    shared_configuration_id: str,
    normalized_token_error_by_seed: Mapping[int | str, float],
    hybrid_accuracy_by_seed: Mapping[int | str, float],
    hybrid_cross_entropy_by_seed: Mapping[int | str, float],
    inference_flops: int,
    parameter_count: int,
    seed_artifacts: Mapping[int | str, Mapping[str, Any]],
    materialized_run_hashes: Mapping[int | str, str],
) -> dict[str, Any]:
    if (
        not candidate_id
        or expert_id not in EXPERT_ORDER
        or not coordinate_id
        or target_mode
        not in {
            "T0_PURE",
            "T1_ANCHORED_BRIDGE",
            "T1_TASK_BRIDGE",
            "T2_PROJECT",
        }
        or not shape_id
        or int(token_count) not in {1, 2, 4, 8, 16}
        or int(token_dimension) not in {64, 128}
        or architecture not in ARCHITECTURES
        or context not in CONTEXTS
        or uncertainty_head not in UNCERTAINTY_HEADS
        or normalization_mode not in NORMALIZATION_MODES
        or float(learning_rate) not in {2.0e-4, 5.0e-4, 1.0e-3}
        or float(dropout) not in {0.0, 0.1}
        or not objective_id
        or not hlt_evidence_mode
        or shared_configuration_id
        != shared_predictor_configuration_id(
            architecture=architecture,
            context=context,
            objective_id=objective_id,
            uncertainty_head=uncertainty_head,
            normalization_mode=normalization_mode,
            learning_rate=learning_rate,
            dropout=dropout,
            hlt_evidence_mode=hlt_evidence_mode,
        )
        or int(inference_flops) <= 0
        or int(parameter_count) <= 0
    ):
        raise ValueError("joint predictor candidate semantics differ")
    metric_inputs = {
        "normalized_token_error": normalized_token_error_by_seed,
        "hybrid_accuracy": hybrid_accuracy_by_seed,
        "hybrid_cross_entropy": hybrid_cross_entropy_by_seed,
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, rows in metric_inputs.items():
        if {int(seed) for seed in rows} != set(PIPELINE_SEEDS):
            raise ValueError("predictor candidate metric seed coverage differs")
        values = {}
        for seed in PIPELINE_SEEDS:
            value = float(rows.get(seed, rows.get(str(seed))))
            if (
                not math.isfinite(value)
                or value < 0.0
                or (name == "hybrid_accuracy" and value > 1.0)
            ):
                raise ValueError("predictor candidate metric differs")
            values[str(seed)] = value
        metrics[name] = values
    if {int(seed) for seed in materialized_run_hashes} != set(PIPELINE_SEEDS):
        raise ValueError("predictor candidate run-hash coverage differs")
    configuration = {
        "architecture": architecture,
        "context": context,
        "objective_id": objective_id,
        "uncertainty_head": uncertainty_head,
        "normalization_mode": normalization_mode,
        "learning_rate": float(learning_rate),
        "dropout": float(dropout),
        "hlt_evidence_mode": hlt_evidence_mode,
        "target_mode": target_mode,
        "shape_id": shape_id,
        "token_count": int(token_count),
        "token_dimension": int(token_dimension),
    }
    return with_content_hash(
        {
            "contract": PREDICTOR_CANDIDATE_CONTRACT,
            "schema_version": 1,
            "candidate_id": candidate_id,
            "expert_id": expert_id,
            "coordinate_id": coordinate_id,
            "target_mode": target_mode,
            "shape_id": shape_id,
            "token_shape": [int(token_count), int(token_dimension)],
            "configuration": configuration,
            "configuration_sha256": canonical_sha256(configuration),
            "shared_configuration_id": shared_configuration_id,
            "metrics_by_seed": metrics,
            "mean_hybrid_accuracy": float(
                np.mean(list(metrics["hybrid_accuracy"].values()), dtype=np.float64)
            ),
            "mean_hybrid_cross_entropy": float(
                np.mean(
                    list(metrics["hybrid_cross_entropy"].values()),
                    dtype=np.float64,
                )
            ),
            "mean_normalized_token_error": float(
                np.mean(
                    list(metrics["normalized_token_error"].values()),
                    dtype=np.float64,
                )
            ),
            "inference_flops": int(inference_flops),
            "parameter_count": int(parameter_count),
            "seed_artifacts": _seed_artifacts(seed_artifacts),
            "materialized_run_hashes": {
                str(seed): require_sha256(
                    materialized_run_hashes.get(
                        seed, materialized_run_hashes.get(str(seed))
                    ),
                    name=f"materialized_run_hashes.{seed}",
                )
                for seed in PIPELINE_SEEDS
            },
            "faithful_token_recovery_claim_eligible": (
                objective_id != "W_LOGIT_ONLY"
                and target_mode != "T1_TASK_BRIDGE"
            ),
        }
    )


def validate_predictor_candidate(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=PREDICTOR_CANDIDATE_CONTRACT
    )
    expected = build_predictor_candidate(
        candidate_id=payload.get("candidate_id"),
        expert_id=payload.get("expert_id"),
        coordinate_id=payload.get("coordinate_id"),
        target_mode=payload.get("target_mode"),
        shape_id=payload.get("shape_id"),
        token_count=int(payload.get("token_shape", [-1, -1])[0]),
        token_dimension=int(payload.get("token_shape", [-1, -1])[1]),
        architecture=payload.get("configuration", {}).get("architecture"),
        context=payload.get("configuration", {}).get("context"),
        objective_id=payload.get("configuration", {}).get("objective_id"),
        uncertainty_head=payload.get("configuration", {}).get(
            "uncertainty_head"
        ),
        normalization_mode=payload.get("configuration", {}).get(
            "normalization_mode"
        ),
        learning_rate=float(
            payload.get("configuration", {}).get("learning_rate", -1.0)
        ),
        dropout=float(payload.get("configuration", {}).get("dropout", -1.0)),
        hlt_evidence_mode=payload.get("configuration", {}).get(
            "hlt_evidence_mode"
        ),
        shared_configuration_id=payload.get("shared_configuration_id"),
        normalized_token_error_by_seed=payload.get("metrics_by_seed", {}).get(
            "normalized_token_error", {}
        ),
        hybrid_accuracy_by_seed=payload.get("metrics_by_seed", {}).get(
            "hybrid_accuracy", {}
        ),
        hybrid_cross_entropy_by_seed=payload.get("metrics_by_seed", {}).get(
            "hybrid_cross_entropy", {}
        ),
        inference_flops=int(payload.get("inference_flops", -1)),
        parameter_count=int(payload.get("parameter_count", -1)),
        seed_artifacts=payload.get("seed_artifacts", {}),
        materialized_run_hashes=payload.get("materialized_run_hashes", {}),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("joint predictor candidate semantics differ")
    return digest


def build_locked_target_coordinate(
    *,
    coordinate_id: str,
    target_modes: Mapping[str, str],
    allocation: Mapping[str, Sequence[int]],
    fusion_checkpoint_hashes: Mapping[int | str, str],
    fusion_registration_hashes: Mapping[int | str, str],
    stage_e_coordinate_sha256: str,
) -> dict[str, Any]:
    allowed_modes = {
        "T0_PURE",
        "T1_ANCHORED_BRIDGE",
        "T1_TASK_BRIDGE",
        "T2_PROJECT",
    }
    if (
        not coordinate_id
        or set(target_modes) != set(EXPERT_ORDER)
        or set(allocation) != set(EXPERT_ORDER)
        or any(target_modes[name] not in allowed_modes for name in EXPERT_ORDER)
        or {int(seed) for seed in fusion_checkpoint_hashes}
        != set(PIPELINE_SEEDS)
        or {int(seed) for seed in fusion_registration_hashes}
        != set(PIPELINE_SEEDS)
    ):
        raise ValueError("locked predictor coordinate coverage differs")
    shapes = {}
    for expert in EXPERT_ORDER:
        shape = list(allocation[expert])
        if (
            len(shape) != 2
            or int(shape[0]) not in {1, 2, 4, 8, 16}
            or int(shape[1]) not in {64, 128}
        ):
            raise ValueError("locked predictor coordinate allocation differs")
        shapes[expert] = [int(shape[0]), int(shape[1])]
    return with_content_hash(
        {
            "contract": TARGET_COORDINATE_CONTRACT,
            "schema_version": 1,
            "coordinate_id": coordinate_id,
            "expert_order": list(EXPERT_ORDER),
            "target_modes": {
                name: target_modes[name] for name in EXPERT_ORDER
            },
            "allocation": shapes,
            "fusion_checkpoint_hashes": {
                str(seed): require_sha256(
                    fusion_checkpoint_hashes.get(
                        seed, fusion_checkpoint_hashes.get(str(seed))
                    ),
                    name=f"fusion_checkpoint_hashes.{seed}",
                )
                for seed in PIPELINE_SEEDS
            },
            "fusion_registration_hashes": {
                str(seed): require_sha256(
                    fusion_registration_hashes.get(
                        seed, fusion_registration_hashes.get(str(seed))
                    ),
                    name=f"fusion_registration_hashes.{seed}",
                )
                for seed in PIPELINE_SEEDS
            },
            "stage_e_coordinate_sha256": require_sha256(
                stage_e_coordinate_sha256, name="stage_e_coordinate_sha256"
            ),
            "cross_mode_fusion_permitted": False,
        }
    )


def validate_locked_target_coordinate(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=TARGET_COORDINATE_CONTRACT
    )
    expected = build_locked_target_coordinate(
        coordinate_id=payload.get("coordinate_id"),
        target_modes=payload.get("target_modes", {}),
        allocation=payload.get("allocation", {}),
        fusion_checkpoint_hashes=payload.get("fusion_checkpoint_hashes", {}),
        fusion_registration_hashes=payload.get(
            "fusion_registration_hashes", {}
        ),
        stage_e_coordinate_sha256=payload.get("stage_e_coordinate_sha256"),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("locked predictor coordinate semantics differ")
    return digest


def build_predictor_cache_index(
    *,
    candidates: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, Any]],
    step9_bundle_sha256: str,
) -> dict[str, Any]:
    candidate_hashes, coordinate_hashes = {}, {}
    seen_pairs = set()
    for candidate in candidates:
        digest = validate_predictor_candidate(candidate)
        identifier = candidate["candidate_id"]
        pair = (candidate["expert_id"], identifier)
        if identifier in candidate_hashes or pair in seen_pairs:
            raise ValueError("predictor cache index candidate is duplicated")
        seen_pairs.add(pair)
        candidate_hashes[identifier] = digest
    for coordinate in coordinates:
        digest = validate_locked_target_coordinate(coordinate)
        identifier = coordinate["coordinate_id"]
        if identifier in coordinate_hashes:
            raise ValueError("predictor cache index coordinate is duplicated")
        coordinate_hashes[identifier] = digest
    if not candidate_hashes or not coordinate_hashes:
        raise ValueError("predictor cache index cannot be empty")
    coordinate_ids = set(coordinate_hashes)
    if any(candidate["coordinate_id"] not in coordinate_ids for candidate in candidates):
        raise ValueError("predictor candidate references an unknown coordinate")
    identity_orders = {}
    for seed in PIPELINE_SEEDS:
        values = {
            candidate["seed_artifacts"][str(seed)][
                "identity_order_sha256"
            ]
            for candidate in candidates
        }
        if len(values) != 1:
            raise ValueError("predictor cache identity order differs")
        identity_orders[str(seed)] = next(iter(values))
    return with_content_hash(
        {
            "contract": PREDICTOR_CACHE_INDEX_CONTRACT,
            "schema_version": 1,
            "step9_bundle_sha256": require_sha256(
                step9_bundle_sha256, name="step9_bundle_sha256"
            ),
            "candidate_hashes": dict(sorted(candidate_hashes.items())),
            "coordinate_hashes": dict(sorted(coordinate_hashes.items())),
            "candidate_count": len(candidate_hashes),
            "identity_order_hashes_by_seed": identity_orders,
            "all_three_seed_inference_caches_bound": True,
            "eligible_candidate_universe_frozen": True,
            "complete_identity_coverage_required": True,
        }
    )


def validate_predictor_cache_index(
    payload: Mapping[str, Any],
    *,
    candidates: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, Any]],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=PREDICTOR_CACHE_INDEX_CONTRACT
    )
    expected = build_predictor_cache_index(
        candidates=candidates,
        coordinates=coordinates,
        step9_bundle_sha256=payload.get("step9_bundle_sha256"),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("predictor cache index semantics differ")
    return digest


def _simple_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if (
        values.shape != (len(truth), 10)
        or len(truth) == 0
        or not np.isfinite(values).all()
        or bool(((truth < 0) | (truth >= 10)).any())
    ):
        raise ValueError("bundle score logits/labels differ")
    shifted = values - values.max(axis=1, keepdims=True)
    return {
        "accuracy": float((values.argmax(axis=1) == truth).mean()),
        "cross_entropy": float(
            (
                np.log(np.exp(shifted).sum(axis=1))
                - shifted[np.arange(len(truth)), truth]
            ).mean(dtype=np.float64)
        ),
    }


def score_frozen_bundle(
    candidate_tuple: Sequence[str],
    *,
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    predicted_banks: Mapping[tuple[str, int], np.ndarray],
    labels_by_seed: Mapping[int, np.ndarray],
    fusion_by_seed: Mapping[int, Any],
    batch_size: int = 1024,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to score RETB predictor bundles")
    if len(candidate_tuple) != len(EXPERT_ORDER):
        raise ValueError("predictor tuple coverage differs")
    rows = [candidates_by_id[name] for name in candidate_tuple]
    coordinate_ids = {row["coordinate_id"] for row in rows}
    if len(coordinate_ids) != 1:
        raise ValueError("cross-coordinate predictor tuple is forbidden")
    if any(row["expert_id"] != expert for row, expert in zip(rows, EXPERT_ORDER)):
        raise ValueError("predictor tuple expert order differs")
    for seed in PIPELINE_SEEDS:
        if len(
            {
                row["seed_artifacts"][str(seed)]["identity_order_sha256"]
                for row in rows
            }
        ) != 1:
            raise ValueError("bundle predicted-bank identity order differs")
    if set(labels_by_seed) != set(PIPELINE_SEEDS) or set(fusion_by_seed) != set(
        PIPELINE_SEEDS
    ):
        raise ValueError("bundle score seed coverage differs")
    seed_metrics = {}
    for seed in PIPELINE_SEEDS:
        labels = np.asarray(labels_by_seed[seed], dtype=np.int64)
        arrays = {
            expert: np.asarray(
                predicted_banks[(candidate_tuple[index], seed)],
                dtype=np.float32,
            )
            for index, expert in enumerate(EXPERT_ORDER)
        }
        if any(len(values) != len(labels) for values in arrays.values()):
            raise ValueError("bundle predicted-bank identity coverage differs")
        fusion = fusion_by_seed[seed]
        try:
            device = next(fusion.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        fusion.eval()
        pieces = []
        with torch.no_grad():
            for start in range(0, len(labels), int(batch_size)):
                stop = min(start + int(batch_size), len(labels))
                banks = {
                    name: torch.from_numpy(arrays[name][start:stop]).to(device)
                    for name in EXPERT_ORDER
                }
                output = fusion(token_banks=banks)
                pieces.append(output.float().cpu().numpy())
        seed_metrics[str(seed)] = _simple_metrics(
            np.concatenate(pieces), labels
        )
    token_error = float(
        sum(row["mean_normalized_token_error"] for row in rows)
        / len(rows)
    )
    return {
        "candidate_tuple": list(candidate_tuple),
        "coordinate_id": next(iter(coordinate_ids)),
        "metrics_by_seed": seed_metrics,
        "mean_accuracy": float(
            np.mean(
                [seed_metrics[str(seed)]["accuracy"] for seed in PIPELINE_SEEDS],
                dtype=np.float64,
            )
        ),
        "mean_cross_entropy": float(
            np.mean(
                [
                    seed_metrics[str(seed)]["cross_entropy"]
                    for seed in PIPELINE_SEEDS
                ],
                dtype=np.float64,
            )
        ),
        "mean_normalized_token_error": token_error,
        "inference_flops": int(sum(row["inference_flops"] for row in rows)),
        "parameter_count": int(sum(row["parameter_count"] for row in rows)),
    }


def _rank(rows: Sequence[Mapping[str, Any]], *, width: int) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("bundle selector has no eligible tuples")
    maximum = max(float(row["mean_accuracy"]) for row in rows)
    retained = [
        dict(row)
        for row in rows
        if maximum - float(row["mean_accuracy"]) <= ACCURACY_WINDOW + 1.0e-15
    ]
    retained.sort(
        key=lambda row: (
            float(row["mean_cross_entropy"]),
            float(row["mean_normalized_token_error"]),
            int(row["inference_flops"]),
            int(row["parameter_count"]),
            tuple(row["candidate_tuple"]),
        )
    )
    return retained[: int(width)]


def _individual_default(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    maximum = max(float(row["mean_hybrid_accuracy"]) for row in candidates)
    eligible = [
        row
        for row in candidates
        if maximum - float(row["mean_hybrid_accuracy"])
        <= ACCURACY_WINDOW + 1.0e-15
    ]
    return min(
        eligible,
        key=lambda row: (
            float(row["mean_hybrid_cross_entropy"]),
            float(row["mean_normalized_token_error"]),
            int(row["inference_flops"]),
            int(row["parameter_count"]),
            row["candidate_id"],
        ),
    )


def select_joint_predictor_bundle(
    *,
    candidates: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, Any]],
    score_tuple: Callable[[tuple[str, ...]], Mapping[str, Any]],
    predictor_cache_index_sha256: str,
    label_manifest_hashes_by_seed: Mapping[int | str, str],
    label_payload_hashes_by_seed: Mapping[int | str, str],
    source_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = build_bundle_search_policy()
    if source_snapshot is not None:
        policy = bind_source(policy, source_snapshot=source_snapshot)
    candidate_map = {}
    for row in candidates:
        validate_predictor_candidate(row)
        if row["candidate_id"] in candidate_map:
            raise ValueError("joint selector candidate ID is duplicated")
        candidate_map[row["candidate_id"]] = row
    coordinate_map = {}
    for row in coordinates:
        validate_locked_target_coordinate(row)
        if row["coordinate_id"] in coordinate_map:
            raise ValueError("joint selector coordinate ID is duplicated")
        coordinate_map[row["coordinate_id"]] = row
    if {int(seed) for seed in label_manifest_hashes_by_seed} != set(
        PIPELINE_SEEDS
    ) or {int(seed) for seed in label_payload_hashes_by_seed} != set(
        PIPELINE_SEEDS
    ):
        raise ValueError("joint selector label-manifest coverage differs")
    scored: dict[tuple[str, ...], dict[str, Any]] = {}

    def score(names: tuple[str, ...]) -> dict[str, Any]:
        if names not in scored:
            row = dict(score_tuple(names))
            if (
                tuple(row.get("candidate_tuple", ())) != names
                or row.get("coordinate_id")
                != candidate_map[names[0]]["coordinate_id"]
                or not all(
                    math.isfinite(float(row.get(name, math.nan)))
                    for name in (
                        "mean_accuracy",
                        "mean_cross_entropy",
                        "mean_normalized_token_error",
                    )
                )
                or not 0.0 <= float(row["mean_accuracy"]) <= 1.0
                or int(row.get("inference_flops", 0)) <= 0
                or int(row.get("parameter_count", 0)) <= 0
            ):
                raise ValueError("joint selector score semantics differ")
            scored[names] = row
        return scored[names]

    traces = []
    coordinate_finals = []
    individual_controls = []
    for coordinate_id in sorted(coordinate_map):
        coordinate = coordinate_map[coordinate_id]
        by_expert = {}
        for expert in EXPERT_ORDER:
            rows = [
                row
                for row in candidate_map.values()
                if row["expert_id"] == expert
                and row["coordinate_id"] == coordinate_id
                and row["target_mode"] == coordinate["target_modes"][expert]
                and row["token_shape"] == coordinate["allocation"][expert]
            ]
            if not rows:
                raise ValueError(
                    f"coordinate {coordinate_id} lacks candidates for {expert}"
                )
            by_expert[expert] = sorted(rows, key=lambda row: row["candidate_id"])
        defaults = tuple(
            _individual_default(by_expert[expert])["candidate_id"]
            for expert in EXPERT_ORDER
        )
        individual_controls.append(score(defaults))
        beam = [score(defaults)]
        depth_rows = []
        for depth, expert in enumerate(EXPERT_ORDER):
            expanded = {}
            for retained in beam:
                for candidate in by_expert[expert]:
                    names = list(retained["candidate_tuple"])
                    names[depth] = candidate["candidate_id"]
                    key = tuple(names)
                    expanded[key] = score(key)
            beam = _rank(list(expanded.values()), width=BEAM_WIDTH)
            depth_rows.append(
                {
                    "expert": expert,
                    "expanded_tuple_count": len(expanded),
                    "retained_tuples": [
                        list(row["candidate_tuple"]) for row in beam
                    ],
                }
            )
        coordinate_finals.extend(beam)
        traces.append(
            {
                "coordinate_id": coordinate_id,
                "default_tuple": list(defaults),
                "depths": depth_rows,
            }
        )
    final_rows = _rank(coordinate_finals, width=BEAM_WIDTH)
    selected = final_rows[0]

    global_shared_rows = []
    all_canonical_rows = []
    for coordinate_id in sorted(coordinate_map):
        coordinate_candidates = [
            row
            for row in candidate_map.values()
            if row["coordinate_id"] == coordinate_id
        ]
        shared_ids = set.intersection(
            *[
                {
                    row["shared_configuration_id"]
                    for row in coordinate_candidates
                    if row["expert_id"] == expert
                }
                for expert in EXPERT_ORDER
            ]
        )
        for shared_id in sorted(shared_ids):
            names = []
            valid = True
            for expert in EXPERT_ORDER:
                rows = [
                    row
                    for row in coordinate_candidates
                    if row["expert_id"] == expert
                    and row["shared_configuration_id"] == shared_id
                ]
                if not rows:
                    valid = False
                    break
                names.append(_individual_default(rows)["candidate_id"])
            if valid:
                global_shared_rows.append(score(tuple(names)))
        canonical = []
        for expert in EXPERT_ORDER:
            rows = [
                row
                for row in coordinate_candidates
                if row["expert_id"] == expert
                and row["configuration"]["objective_id"] == "W_CANONICAL"
            ]
            if not rows:
                canonical = []
                break
            canonical.append(_individual_default(rows)["candidate_id"])
        if canonical:
            all_canonical_rows.append(score(tuple(canonical)))
    if not global_shared_rows:
        raise ValueError(
            "bundle selector lacks the GLOBAL_SHARED_CONFIGURATION control"
        )
    if not all_canonical_rows:
        raise ValueError("bundle selector lacks the ALL_W_CANONICAL control")
    controls = {
        "INDIVIDUAL_HYBRID_DEFAULT": _rank(
            individual_controls, width=1
        )[0],
        "GLOBAL_SHARED_CONFIGURATION": _rank(
            global_shared_rows, width=1
        )[0],
        "ALL_W_CANONICAL": _rank(all_canonical_rows, width=1)[0],
    }
    selected_rows = [candidate_map[name] for name in selected["candidate_tuple"]]
    selected_lineage = {
        "candidate_hashes": {
            expert: selected_rows[index]["content_hash"]
            for index, expert in enumerate(EXPERT_ORDER)
        },
        "seed_specific_artifacts": {
            str(seed): {
                expert: selected_rows[index]["seed_artifacts"][str(seed)]
                for index, expert in enumerate(EXPERT_ORDER)
            }
            for seed in PIPELINE_SEEDS
        },
        "selected_candidate_descriptors": {
            expert: {
                "candidate_id": selected_rows[index]["candidate_id"],
                "target_mode": selected_rows[index]["target_mode"],
                "configuration": dict(
                    selected_rows[index]["configuration"]
                ),
            }
            for index, expert in enumerate(EXPERT_ORDER)
        },
    }
    search = with_content_hash(
        {
            "contract": BUNDLE_SEARCH_CONTRACT,
            "schema_version": 1,
            "policy_sha256": policy["content_hash"],
            "predictor_cache_index_sha256": require_sha256(
                predictor_cache_index_sha256,
                name="predictor_cache_index_sha256",
            ),
            "label_manifest_hashes_by_seed": {
                str(seed): require_sha256(
                    label_manifest_hashes_by_seed.get(
                        seed, label_manifest_hashes_by_seed.get(str(seed))
                    ),
                    name=f"label_manifest_hashes_by_seed.{seed}",
                )
                for seed in PIPELINE_SEEDS
            },
            "label_payload_hashes_by_seed": {
                str(seed): require_sha256(
                    label_payload_hashes_by_seed.get(
                        seed, label_payload_hashes_by_seed.get(str(seed))
                    ),
                    name=f"label_payload_hashes_by_seed.{seed}",
                )
                for seed in PIPELINE_SEEDS
            },
            "candidate_hashes": {
                name: candidate_map[name]["content_hash"]
                for name in sorted(candidate_map)
            },
            "coordinate_hashes": {
                name: coordinate_map[name]["content_hash"]
                for name in sorted(coordinate_map)
            },
            "trace": traces,
            "scored_tuple_count": len(scored),
            "scored_tuples": [
                scored[names] for names in sorted(scored)
            ],
            "final_beam": final_rows,
            "selected_tuple": list(selected["candidate_tuple"]),
            "selected_score": selected,
            "selected_lineage": selected_lineage,
            "controls": controls,
            "all_negative_campaign_completed": True,
            "fusion_or_predictor_retrained": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
            "performance_based_termination": False,
        }
    )
    if source_snapshot is not None:
        search = bind_source(search, source_snapshot=source_snapshot)
    coordinate = coordinate_map[selected["coordinate_id"]]
    lock = with_content_hash(
        {
            "contract": PREDICTOR_BUNDLE_LOCK_CONTRACT,
            "schema_version": 1,
            "search_sha256": search["content_hash"],
            "predictor_cache_index_sha256": predictor_cache_index_sha256,
            "coordinate_id": selected["coordinate_id"],
            "coordinate_sha256": coordinate["content_hash"],
            "target_modes": coordinate["target_modes"],
            "allocation": coordinate["allocation"],
            "expert_order": list(EXPERT_ORDER),
            "candidate_tuple": list(selected["candidate_tuple"]),
            "candidate_hashes": {
                expert: selected_rows[index]["content_hash"]
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "seed_specific_artifacts": {
                str(seed): {
                    expert: selected_rows[index]["seed_artifacts"][str(seed)]
                    for index, expert in enumerate(EXPERT_ORDER)
                }
                for seed in PIPELINE_SEEDS
            },
            "selected_candidate_descriptors": selected_lineage[
                "selected_candidate_descriptors"
            ],
            "fusion_checkpoint_hashes": coordinate[
                "fusion_checkpoint_hashes"
            ],
            "selection_data_hashes": {
                "label_manifests": search[
                    "label_manifest_hashes_by_seed"
                ],
                "label_payloads": search[
                    "label_payload_hashes_by_seed"
                ],
            },
            "configuration_shared_across_pipeline_seeds": True,
            "per_seed_selection_permitted": False,
            "locked_before_joint_training": True,
            "performance_based_termination": False,
        }
    )
    if source_snapshot is not None:
        lock = bind_source(lock, source_snapshot=source_snapshot)
    return {
        "policy": policy,
        "search": search,
        "predictor_bundle_lock": lock,
    }


def build_predictor_bundle_lock_from_scored_tuple(
    *,
    search: Mapping[str, Any],
    scored_tuple: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, Any]],
    source_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize a non-primary, pre-scored coordinate lock.

    Stage L carries both selected uniform shapes and all three heterogeneous
    allocations.  Those locks must be selected from the already-complete
    Stage-H search; this helper deliberately cannot score a new tuple or
    inspect a later split.
    """

    validate_content_hash(search, expected_contract=BUNDLE_SEARCH_CONTRACT)
    candidate_map = {}
    for row in candidates:
        validate_predictor_candidate(row)
        candidate_map[str(row["candidate_id"])] = row
    coordinate_map = {}
    for row in coordinates:
        validate_locked_target_coordinate(row)
        coordinate_map[str(row["coordinate_id"])] = row
    names = tuple(str(value) for value in scored_tuple["candidate_tuple"])
    matching_scores = [
        row
        for row in search["scored_tuples"]
        if tuple(row["candidate_tuple"]) == names
        and row["coordinate_id"] == scored_tuple["coordinate_id"]
    ]
    if len(matching_scores) != 1 or dict(matching_scores[0]) != dict(
        scored_tuple
    ):
        raise ValueError("carried bundle tuple was not pre-scored")
    coordinate = coordinate_map.get(str(scored_tuple["coordinate_id"]))
    if coordinate is None or len(names) != len(EXPERT_ORDER):
        raise ValueError("carried bundle coordinate differs")
    selected_rows = []
    for expert, name in zip(EXPERT_ORDER, names, strict=True):
        candidate = candidate_map.get(name)
        if (
            candidate is None
            or candidate["expert_id"] != expert
            or candidate["coordinate_id"] != coordinate["coordinate_id"]
        ):
            raise ValueError("carried bundle candidate lineage differs")
        selected_rows.append(candidate)
    lock = with_content_hash(
        {
            "contract": PREDICTOR_BUNDLE_LOCK_CONTRACT,
            "schema_version": 1,
            "search_sha256": search["content_hash"],
            "predictor_cache_index_sha256": search[
                "predictor_cache_index_sha256"
            ],
            "coordinate_id": coordinate["coordinate_id"],
            "coordinate_sha256": coordinate["content_hash"],
            "target_modes": coordinate["target_modes"],
            "allocation": coordinate["allocation"],
            "expert_order": list(EXPERT_ORDER),
            "candidate_tuple": list(names),
            "candidate_hashes": {
                expert: selected_rows[index]["content_hash"]
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "seed_specific_artifacts": {
                str(seed): {
                    expert: selected_rows[index]["seed_artifacts"][str(seed)]
                    for index, expert in enumerate(EXPERT_ORDER)
                }
                for seed in PIPELINE_SEEDS
            },
            "selected_candidate_descriptors": {
                expert: {
                    "candidate_id": selected_rows[index]["candidate_id"],
                    "target_mode": selected_rows[index]["target_mode"],
                    "configuration": dict(
                        selected_rows[index]["configuration"]
                    ),
                }
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "fusion_checkpoint_hashes": coordinate[
                "fusion_checkpoint_hashes"
            ],
            "selection_data_hashes": {
                "label_manifests": search[
                    "label_manifest_hashes_by_seed"
                ],
                "label_payloads": search[
                    "label_payload_hashes_by_seed"
                ],
            },
            "configuration_shared_across_pipeline_seeds": True,
            "per_seed_selection_permitted": False,
            "locked_before_joint_training": True,
            "performance_based_termination": False,
        }
    )
    if source_snapshot is not None:
        lock = bind_source(lock, source_snapshot=source_snapshot)
    return lock


def select_carried_predictor_bundles(
    *,
    search: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, Any]],
    carried_shape_roles: Sequence[str],
    source_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Lock the best already-scored tuple independently for each carried shape."""

    validate_content_hash(search, expected_contract=BUNDLE_SEARCH_CONTRACT)
    roles = [str(value) for value in carried_shape_roles]
    if not roles or len(roles) != len(set(roles)):
        raise ValueError("carried predictor shape roles differ")
    locks = {}
    selection_rows = []
    for role in roles:
        eligible = [
            row
            for row in search["scored_tuples"]
            if str(row["coordinate_id"]).endswith(f":{role}")
        ]
        if not eligible:
            raise ValueError(f"carried predictor shape lacks scores: {role}")
        selected = _rank(eligible, width=1)[0]
        lock = build_predictor_bundle_lock_from_scored_tuple(
            search=search,
            scored_tuple=selected,
            candidates=candidates,
            coordinates=coordinates,
            source_snapshot=source_snapshot,
        )
        locks[role] = lock
        selection_rows.append(
            {
                "carried_shape_role": role,
                "coordinate_id": selected["coordinate_id"],
                "candidate_tuple": list(selected["candidate_tuple"]),
                "score": dict(selected),
                "predictor_bundle_lock_sha256": lock["content_hash"],
            }
        )
    artifact = with_content_hash(
        {
            "contract": CARRIED_PREDICTOR_BUNDLE_INDEX_CONTRACT,
            "schema_version": 1,
            "predictor_bundle_search_sha256": search["content_hash"],
            "carried_shape_roles": roles,
            "selection_rows": selection_rows,
            "selection_rule": list(build_bundle_search_policy()["ranking"]),
            "new_tuple_scoring_permitted": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
            "scientific_underperformance_blocks_continuation": False,
        }
    )
    if source_snapshot is not None:
        artifact = bind_source(artifact, source_snapshot=source_snapshot)
    return {"index": artifact, "locks": locks}


def validate_predictor_bundle_selection(
    result: Mapping[str, Mapping[str, Any]],
) -> str:
    if set(result) != {"policy", "search", "predictor_bundle_lock"}:
        raise ValueError("predictor bundle selection members differ")
    policy_sha = validate_bundle_search_policy(result["policy"])
    search_sha = validate_content_hash(
        result["search"], expected_contract=BUNDLE_SEARCH_CONTRACT
    )
    lock_sha = validate_content_hash(
        result["predictor_bundle_lock"],
        expected_contract=PREDICTOR_BUNDLE_LOCK_CONTRACT,
    )
    search = result["search"]
    lock = result["predictor_bundle_lock"]
    scored = search.get("scored_tuples", [])
    selected_lineage = search.get("selected_lineage", {})
    if (
        search["policy_sha256"] != policy_sha
        or lock["search_sha256"] != search_sha
        or lock["candidate_tuple"] != search["selected_tuple"]
        or lock["coordinate_id"] != search["selected_score"]["coordinate_id"]
        or search.get("scored_tuple_count") != len(scored)
        or len(
            {
                tuple(row.get("candidate_tuple", ()))
                for row in scored
            }
        )
        != len(scored)
        or search["selected_score"] not in scored
        or set(search.get("controls", {}))
        != {
            "INDIVIDUAL_HYBRID_DEFAULT",
            "GLOBAL_SHARED_CONFIGURATION",
            "ALL_W_CANONICAL",
        }
        or any(row is None for row in search["controls"].values())
        or lock["candidate_hashes"]
        != selected_lineage.get("candidate_hashes")
        or lock["seed_specific_artifacts"]
        != selected_lineage.get("seed_specific_artifacts")
        or lock.get("selected_candidate_descriptors")
        != selected_lineage.get("selected_candidate_descriptors")
        or lock.get("selection_data_hashes")
        != {
            "label_manifests": search["label_manifest_hashes_by_seed"],
            "label_payloads": search["label_payload_hashes_by_seed"],
        }
        or not lock["configuration_shared_across_pipeline_seeds"]
        or lock["per_seed_selection_permitted"]
        or not lock["locked_before_joint_training"]
        or lock["performance_based_termination"]
        or search["stack_val_consumed"]
        or search["final_test_consumed"]
        or search["fusion_or_predictor_retrained"]
    ):
        raise ValueError("predictor bundle selection lineage differs")
    return lock_sha


__all__ = [
    "ACCURACY_WINDOW",
    "BEAM_WIDTH",
    "BUNDLE_SEARCH_CONTRACT",
    "BUNDLE_SEARCH_POLICY_CONTRACT",
    "CARRIED_PREDICTOR_BUNDLE_INDEX_CONTRACT",
    "PIPELINE_SEEDS",
    "PREDICTOR_BUNDLE_LOCK_CONTRACT",
    "PREDICTOR_CACHE_INDEX_CONTRACT",
    "PREDICTOR_CANDIDATE_CONTRACT",
    "TARGET_COORDINATE_CONTRACT",
    "build_bundle_search_policy",
    "build_locked_target_coordinate",
    "build_predictor_cache_index",
    "build_predictor_candidate",
    "build_predictor_bundle_lock_from_scored_tuple",
    "score_frozen_bundle",
    "select_joint_predictor_bundle",
    "select_carried_predictor_bundles",
    "shared_predictor_configuration_id",
    "validate_locked_target_coordinate",
    "validate_bundle_search_policy",
    "validate_predictor_bundle_selection",
    "validate_predictor_cache_index",
    "validate_predictor_candidate",
]
