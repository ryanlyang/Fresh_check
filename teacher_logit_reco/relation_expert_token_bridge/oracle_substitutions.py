"""Stage-I oracle substitutions and deterministic negative-token controls."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .evaluation import evaluate_classification
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


STAGE_I_POLICY_CONTRACT = "retb_stage_i_oracle_substitution_policy_v2"
STAGE_I_EVALUATION_CONTRACT = "retb_stage_i_oracle_substitution_evaluation_v2"
NEGATIVE_CONTROL_SEED = 730_013


def build_stage_i_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": STAGE_I_POLICY_CONTRACT,
            "schema_version": 2,
            "split": "val_design",
            "selection_use_permitted": False,
            "required_substitutions": [
                "NO_RECONSTRUCTION",
                "IDENTITY_PROJECTED_HLT",
                "ONE_PREDICTED_SIX_ORACLE_by_expert",
                "ONE_ORACLE_SIX_PREDICTED_by_expert",
                "ALL_PREDICTED",
                "ALL_ORACLE",
            ],
            "identity_projection_definition": (
                "exact_same_shape_native_HLT_expert_token_bank_with_no_"
                "learned_or_statistical_transform_shape_mismatch_fails"
            ),
            "no_reconstruction_definition": (
                "authenticated_frozen_native_HLT_fusion_logits_for_the_same_"
                "identity_order_and_pipeline_seed"
            ),
            "required_negative_controls": {
                "WRONG_EVENT_OFFLINE_TARGETS": (
                    "canonical_identity_order_cyclic_left_shift_one"
                ),
                "WITHIN_CLASS_WRONG_EVENT_TARGETS": (
                    "per_class_canonical_order_cyclic_left_shift_one"
                ),
                "WRONG_EVENT_BANK_by_expert": (
                    "replace_exactly_one_bank_by_global_cyclic_wrong_event"
                ),
                "WITHIN_CLASS_WRONG_EVENT_BANK_by_expert": (
                    "replace_exactly_one_bank_by_same_class_wrong_event"
                ),
                "SLOT_PERMUTED_TARGETS": (
                    "cyclic_left_shift_one_slot_per_bank_K1_unchanged"
                ),
                "EXPERT_BANK_SWAPPED_TARGETS": (
                    "cyclic_left_shift_within_equal_shape_groups"
                ),
                "WITHIN_CLASS_MEAN_TARGETS_by_expert": (
                    "replace_exactly_one_bank_with_its_per_class_mean"
                ),
                "ZERO_ORACLE_BANK_by_expert": "exact_float32_zero",
                "MATCHED_GAUSSIAN_NOISE": {
                    "seed": NEGATIVE_CONTROL_SEED,
                    "definition": (
                        "per_bank_per_slot_channel_empirical_standard_deviation"
                    ),
                    "noise_scale": 1.0,
                },
            },
            "wrong_event_controls_enter_selection": False,
            "model_retraining_permitted": False,
            "stack_val_permitted": False,
            "final_test_permitted": False,
            "performance_based_termination": False,
        }
    )


def validate_stage_i_policy(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_I_POLICY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_i_policy()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-I policy semantics differ")
    return digest


def _identities(values: Sequence[str]) -> tuple[str, ...]:
    rows = tuple(str(value) for value in values)
    if not rows or len(rows) != len(set(rows)) or any(not row for row in rows):
        raise ValueError("Stage-I identities are empty or duplicated")
    return rows


def _validate_banks(
    banks: Mapping[str, np.ndarray],
    *,
    event_count: int,
    expected_shapes: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, np.ndarray]:
    if set(banks) != set(EXPERT_ORDER):
        raise ValueError("Stage-I bank coverage differs")
    output = {}
    for expert in EXPERT_ORDER:
        values = np.asarray(banks[expert], dtype=np.float32)
        if (
            values.ndim != 3
            or values.shape[0] != event_count
            or values.shape[1] not in {1, 2, 4, 8, 16}
            or values.shape[2] not in {64, 128}
            or not np.isfinite(values).all()
            or (
                expected_shapes is not None
                and list(values.shape[1:])
                != [int(value) for value in expected_shapes[expert]]
            )
        ):
            raise ValueError(f"Stage-I token bank {expert} differs")
        output[expert] = values
    return output


def wrong_event_indices(event_count: int) -> np.ndarray:
    count = int(event_count)
    if count < 2:
        raise ValueError("wrong-event control requires at least two events")
    return np.roll(np.arange(count, dtype=np.int64), -1)


def within_class_wrong_event_indices(labels: np.ndarray) -> np.ndarray:
    truth = np.asarray(labels, dtype=np.int64)
    if truth.ndim != 1 or len(truth) == 0:
        raise ValueError("within-class control labels differ")
    indices = np.empty(len(truth), dtype=np.int64)
    for class_id in range(10):
        selected = np.flatnonzero(truth == class_id)
        if len(selected) < 2:
            raise ValueError(
                "within-class wrong-event control requires two events per class"
            )
        indices[selected] = np.roll(selected, -1)
    if np.any(indices == np.arange(len(truth))):
        raise RuntimeError("within-class wrong-event control retained identity")
    return indices


def _permute_events(
    banks: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {expert: banks[expert][indices] for expert in EXPERT_ORDER}


def _permute_slots(
    banks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if all(banks[expert].shape[1] == 1 for expert in EXPERT_ORDER):
        raise ValueError("slot-permutation control would be a complete no-op")
    return {
        expert: (
            banks[expert]
            if banks[expert].shape[1] == 1
            else np.roll(banks[expert], -1, axis=1)
        )
        for expert in EXPERT_ORDER
    }


def _swap_equal_shape_banks(
    banks: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    groups: dict[tuple[int, int], list[str]] = {}
    for expert in EXPERT_ORDER:
        groups.setdefault(tuple(banks[expert].shape[1:]), []).append(expert)
    output, sources = {}, {}
    for experts in groups.values():
        shifted = experts[1:] + experts[:1]
        for destination, source in zip(experts, shifted):
            output[destination] = banks[source]
            sources[destination] = source
    if all(sources[expert] == expert for expert in EXPERT_ORDER):
        raise ValueError("expert-bank swap control would be a complete no-op")
    return output, sources


def _within_class_mean(
    banks: Mapping[str, np.ndarray], labels: np.ndarray
) -> dict[str, np.ndarray]:
    output = {}
    for expert in EXPERT_ORDER:
        values = np.empty_like(banks[expert])
        for class_id in range(10):
            selected = labels == class_id
            if not bool(selected.any()):
                raise ValueError("within-class mean lacks class support")
            values[selected] = banks[expert][selected].mean(
                axis=0, dtype=np.float64
            ).astype(np.float32)
        output[expert] = values
    return output


def _matched_gaussian_noise(
    banks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(NEGATIVE_CONTROL_SEED)
    output = {}
    for expert in EXPERT_ORDER:
        values = banks[expert]
        scale = values.std(axis=0, dtype=np.float64, ddof=0).astype(np.float32)
        noise = generator.normal(size=values.shape).astype(np.float32) * scale
        output[expert] = values + noise
    if all(np.array_equal(output[name], banks[name]) for name in EXPERT_ORDER):
        raise ValueError("matched-Gaussian control would be a complete no-op")
    return output


def _bank_diagnostics(banks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows = {}
    for expert in EXPERT_ORDER:
        flattened = banks[expert].reshape(len(banks[expert]), -1).astype(
            np.float64
        )
        centered = flattened - flattened.mean(axis=0, keepdims=True)
        covariance_trace = float(
            np.square(centered).sum(dtype=np.float64)
            / max(len(flattened) - 1, 1)
        )
        rows[expert] = {
            "mean_event_l2_norm": float(
                np.linalg.norm(flattened, axis=1).mean(dtype=np.float64)
            ),
            "covariance_trace": covariance_trace,
        }
    return rows


def _fusion_logits(
    fusion: Any,
    banks: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for Stage-I evaluation")
    try:
        device = next(fusion.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    fusion.eval()
    pieces = []
    count = len(next(iter(banks.values())))
    with torch.no_grad():
        for start in range(0, count, int(batch_size)):
            stop = min(start + int(batch_size), count)
            batch = {
                expert: torch.from_numpy(banks[expert][start:stop]).to(device)
                for expert in EXPERT_ORDER
            }
            pieces.append(fusion(token_banks=batch).float().cpu().numpy())
    return np.concatenate(pieces).astype(np.float32)


def _condition(
    *,
    condition_id: str,
    logits: np.ndarray,
    labels: np.ndarray,
    banks: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    metrics = evaluate_classification(logits, labels, split="val_design")
    return {
        "condition_id": condition_id,
        "metrics": metrics,
        "bank_diagnostics": None if banks is None else _bank_diagnostics(banks),
    }


def evaluate_stage_i_substitutions(
    *,
    identities: Sequence[str],
    labels: np.ndarray,
    predicted_banks: Mapping[str, np.ndarray],
    oracle_banks: Mapping[str, np.ndarray],
    identity_projected_hlt_banks: Mapping[str, np.ndarray],
    no_reconstruction_logits: np.ndarray,
    frozen_offline_fusion: Any,
    pipeline_seed: int,
    stage_i_policy_sha256: str,
    stage_i_input_payload_sha256: str,
    predictor_bundle_lock_sha256: str,
    frozen_fusion_checkpoint_sha256: str,
    identity_manifest_sha256: str,
    label_manifest_sha256: str,
    predicted_cache_hashes: Mapping[str, str],
    oracle_target_cache_sha256: str,
    hlt_cache_sha256: str,
    identity_projected_hlt_cache_sha256: str,
    no_reconstruction_prediction_sha256: str,
    batch_size: int = 1024,
) -> dict[str, Any]:
    ids = _identities(identities)
    truth = np.asarray(labels, dtype=np.int64)
    if (
        int(pipeline_seed) not in {101, 202, 303}
        or truth.shape != (len(ids),)
        or bool(((truth < 0) | (truth >= 10)).any())
        or set(predicted_cache_hashes) != set(EXPERT_ORDER)
    ):
        raise ValueError("Stage-I evaluation population differs")
    predicted = _validate_banks(predicted_banks, event_count=len(ids))
    shapes = {
        expert: list(predicted[expert].shape[1:]) for expert in EXPERT_ORDER
    }
    oracle = _validate_banks(
        oracle_banks, event_count=len(ids), expected_shapes=shapes
    )
    identity_hlt = _validate_banks(
        identity_projected_hlt_banks,
        event_count=len(ids),
        expected_shapes=shapes,
    )
    no_reco = np.asarray(no_reconstruction_logits, dtype=np.float32)
    if no_reco.shape != (len(ids), 10) or not np.isfinite(no_reco).all():
        raise ValueError("Stage-I no-reconstruction logits differ")
    conditions = {}

    def add(identifier: str, banks: Mapping[str, np.ndarray] | None, logits=None):
        if identifier in conditions:
            raise ValueError("Stage-I condition is duplicated")
        resolved_logits = (
            np.asarray(logits, dtype=np.float32)
            if logits is not None
            else _fusion_logits(
                frozen_offline_fusion, banks, batch_size=batch_size
            )
        )
        conditions[identifier] = _condition(
            condition_id=identifier,
            logits=resolved_logits,
            labels=truth,
            banks=banks,
        )

    add("NO_RECONSTRUCTION", None, no_reco)
    add("IDENTITY_PROJECTED_HLT", identity_hlt)
    add("ALL_PREDICTED", predicted)
    add("ALL_ORACLE", oracle)
    for expert in EXPERT_ORDER:
        one_predicted = dict(oracle)
        one_predicted[expert] = predicted[expert]
        add(f"ONE_PREDICTED_SIX_ORACLE__{expert}", one_predicted)
        one_oracle = dict(predicted)
        one_oracle[expert] = oracle[expert]
        add(f"ONE_ORACLE_SIX_PREDICTED__{expert}", one_oracle)
    add(
        "WRONG_EVENT_OFFLINE_TARGETS",
        _permute_events(oracle, wrong_event_indices(len(ids))),
    )
    add(
        "WITHIN_CLASS_WRONG_EVENT_TARGETS",
        _permute_events(oracle, within_class_wrong_event_indices(truth)),
    )
    wrong = wrong_event_indices(len(ids))
    class_wrong = within_class_wrong_event_indices(truth)
    for expert in EXPERT_ORDER:
        replaced = dict(oracle)
        replaced[expert] = oracle[expert][wrong]
        add(f"WRONG_EVENT_BANK__{expert}", replaced)
        replaced = dict(oracle)
        replaced[expert] = oracle[expert][class_wrong]
        add(f"WITHIN_CLASS_WRONG_EVENT_BANK__{expert}", replaced)
    add("SLOT_PERMUTED_TARGETS", _permute_slots(oracle))
    swapped, swap_sources = _swap_equal_shape_banks(oracle)
    add("EXPERT_BANK_SWAPPED_TARGETS", swapped)
    within_class_means = _within_class_mean(oracle, truth)
    for expert in EXPERT_ORDER:
        replaced = dict(oracle)
        replaced[expert] = within_class_means[expert]
        add(f"WITHIN_CLASS_MEAN_TARGETS__{expert}", replaced)
    for expert in EXPERT_ORDER:
        zeroed = dict(oracle)
        zeroed[expert] = np.zeros_like(oracle[expert])
        add(f"ZERO_ORACLE_BANK__{expert}", zeroed)
    add("MATCHED_GAUSSIAN_NOISE", _matched_gaussian_noise(oracle))
    return with_content_hash(
        {
            "contract": STAGE_I_EVALUATION_CONTRACT,
            "schema_version": 2,
            "policy_sha256": require_sha256(
                stage_i_policy_sha256, name="stage_i_policy_sha256"
            ),
            "split": "val_design",
            "pipeline_seed": int(pipeline_seed),
            "event_count": len(ids),
            "identity_order_sha256": hashlib.sha256(
                "\n".join(ids).encode("utf-8")
            ).hexdigest(),
            "parents": {
                "predictor_bundle_lock": require_sha256(
                    predictor_bundle_lock_sha256,
                    name="predictor_bundle_lock_sha256",
                ),
                "stage_i_input_payload": require_sha256(
                    stage_i_input_payload_sha256,
                    name="stage_i_input_payload_sha256",
                ),
                "frozen_offline_fusion_checkpoint": require_sha256(
                    frozen_fusion_checkpoint_sha256,
                    name="frozen_fusion_checkpoint_sha256",
                ),
                "identity_manifest": require_sha256(
                    identity_manifest_sha256, name="identity_manifest_sha256"
                ),
                "label_manifest": require_sha256(
                    label_manifest_sha256, name="label_manifest_sha256"
                ),
                "oracle_target_cache": require_sha256(
                    oracle_target_cache_sha256,
                    name="oracle_target_cache_sha256",
                ),
                "hlt_cache": require_sha256(
                    hlt_cache_sha256, name="hlt_cache_sha256"
                ),
                "identity_projected_hlt_cache": require_sha256(
                    identity_projected_hlt_cache_sha256,
                    name="identity_projected_hlt_cache_sha256",
                ),
                "no_reconstruction_prediction": require_sha256(
                    no_reconstruction_prediction_sha256,
                    name="no_reconstruction_prediction_sha256",
                ),
            },
            "predicted_cache_hashes": {
                expert: require_sha256(
                    predicted_cache_hashes[expert],
                    name=f"predicted_cache_hashes.{expert}",
                )
                for expert in EXPERT_ORDER
            },
            "conditions": conditions,
            "expert_bank_swap_sources": swap_sources,
            "negative_control_seed": NEGATIVE_CONTROL_SEED,
            "condition_count": len(conditions),
            "wrong_event_controls_entered_selection": False,
            "model_retrained": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
            "performance_based_termination": False,
        }
    )


def validate_stage_i_evaluation(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_I_EVALUATION_CONTRACT
    )
    required_conditions = {
        "NO_RECONSTRUCTION",
        "IDENTITY_PROJECTED_HLT",
        "ALL_PREDICTED",
        "ALL_ORACLE",
        "WRONG_EVENT_OFFLINE_TARGETS",
        "WITHIN_CLASS_WRONG_EVENT_TARGETS",
        "SLOT_PERMUTED_TARGETS",
        "EXPERT_BANK_SWAPPED_TARGETS",
        "MATCHED_GAUSSIAN_NOISE",
    } | {
        f"{prefix}__{expert}"
        for prefix in (
            "ONE_PREDICTED_SIX_ORACLE",
            "ONE_ORACLE_SIX_PREDICTED",
            "WITHIN_CLASS_MEAN_TARGETS",
            "ZERO_ORACLE_BANK",
            "WRONG_EVENT_BANK",
            "WITHIN_CLASS_WRONG_EVENT_BANK",
        )
        for expert in EXPERT_ORDER
    }
    if (
        require_sha256(payload.get("policy_sha256"), name="policy_sha256")
        != payload["policy_sha256"]
        or payload["split"] != "val_design"
        or int(payload.get("pipeline_seed", -1)) not in {101, 202, 303}
        or int(payload.get("event_count", 0)) <= 0
        or require_sha256(
            payload.get("identity_order_sha256"),
            name="identity_order_sha256",
        )
        != payload["identity_order_sha256"]
        or set(payload["conditions"]) != required_conditions
        or int(payload["condition_count"]) != len(required_conditions)
        or set(payload["predicted_cache_hashes"]) != set(EXPERT_ORDER)
        or set(payload.get("parents", {}))
        != {
            "predictor_bundle_lock",
            "stage_i_input_payload",
            "frozen_offline_fusion_checkpoint",
            "identity_manifest",
            "label_manifest",
            "oracle_target_cache",
            "hlt_cache",
            "identity_projected_hlt_cache",
            "no_reconstruction_prediction",
        }
        or payload["wrong_event_controls_entered_selection"]
        or payload["model_retrained"]
        or payload["stack_val_consumed"]
        or payload["final_test_consumed"]
        or payload["performance_based_termination"]
        or int(payload.get("negative_control_seed", -1))
        != NEGATIVE_CONTROL_SEED
        or set(payload.get("expert_bank_swap_sources", {}))
        != set(EXPERT_ORDER)
        or all(
            payload["expert_bank_swap_sources"][expert] == expert
            for expert in EXPERT_ORDER
        )
    ):
        raise ValueError("Stage-I evaluation semantics differ")
    for name, value in payload["parents"].items():
        require_sha256(value, name=f"parents.{name}")
    for identifier, row in payload["conditions"].items():
        if (
            row.get("condition_id") != identifier
            or row["metrics"].get("split") != "val_design"
            or int(row["metrics"].get("event_count", -1))
            != int(payload["event_count"])
            or (
                identifier == "NO_RECONSTRUCTION"
                and row.get("bank_diagnostics") is not None
            )
            or (
                identifier != "NO_RECONSTRUCTION"
                and set(row.get("bank_diagnostics", {}))
                != set(EXPERT_ORDER)
            )
        ):
            raise ValueError("Stage-I condition semantics differ")
        validate_content_hash(row["metrics"])
    return digest


__all__ = [
    "NEGATIVE_CONTROL_SEED",
    "STAGE_I_EVALUATION_CONTRACT",
    "STAGE_I_POLICY_CONTRACT",
    "build_stage_i_policy",
    "evaluate_stage_i_substitutions",
    "validate_stage_i_evaluation",
    "validate_stage_i_policy",
    "within_class_wrong_event_indices",
    "wrong_event_indices",
]
