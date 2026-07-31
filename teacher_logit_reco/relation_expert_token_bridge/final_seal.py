"""Post-finalist targets, controls, execution seal, and final evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    bind_source,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .evaluation import evaluate_classification, stable_probabilities
from .paired_statistics import build_paired_confirmation_statistics
from .predictor_bundle import PIPELINE_SEEDS
from .stage_n_selection import LOCKED_SCALE_FINALISTS_CONTRACT


POSTLOCK_ORACLE_TARGET_CONTRACT = "retb_postlock_oracle_target_v1"
FINAL_TEST_INPUT_PREPARATION_CONTRACT = (
    "retb_prelock_final_test_input_preparation_v2"
)
FINALIST_CONTROLS_CONTRACT = "retb_scale_finalist_controls_v1"
FINAL_TEST_EXECUTION_LOCK_CONTRACT = (
    "retb_final_test_execution_lock_v1"
)
FINAL_TEST_EXECUTION_CLAIM_CONTRACT = (
    "retb_final_test_execution_claim_v3"
)
FINAL_TEST_EVALUATION_CONTRACT = "retb_sealed_final_test_evaluation_v3"

POSTLOCK_TARGET_PARENT_KEYS = frozenset(
    {
        "locked_scale_finalists",
        "scale_graph_run",
        "scale_offline_experts",
        "scale_offline_fusion",
        "scale_target_normalizer",
        "split_identity_manifest",
        "input_manifest",
    }
)
EXECUTION_PARENT_KEYS = frozenset(
    {
        "campaign_spec",
        "step14_bundle",
        "locked_scale_finalists",
        "finalist_controls",
        "prelock_final_test_inputs",
    }
)
FINAL_INPUT_KEYS = frozenset(
    {
        "final_test_identity_manifest",
        "final_test_raw_inputs",
        "final_test_HLT_inputs",
        "final_test_relation_sidecars",
        "final_test_REGION_sidecars",
    }
)
CONTROL_KINDS = (
    "FINALIST",
    "NAMED_BASELINE",
    "H_MONO_PARAM",
    "H_MONO_FLOP",
    "H_BASE_LONG",
)


def build_prelock_final_test_inputs(
    *,
    campaign_spec_sha256: str,
    split_manifest_sha256: str,
    degradation_profile_sha256: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if set(input_hashes) != set(FINAL_INPUT_KEYS):
        raise ValueError("prelock final-test input coverage differs")
    return with_content_hash(
        {
            "contract": FINAL_TEST_INPUT_PREPARATION_CONTRACT,
            "schema_version": 3,
            "parents": {
                "campaign_spec": require_sha256(
                    campaign_spec_sha256, name="campaign_spec_sha256"
                ),
                "split_manifest": require_sha256(
                    split_manifest_sha256, name="split_manifest_sha256"
                ),
                "degradation_profile": require_sha256(
                    degradation_profile_sha256,
                    name="degradation_profile_sha256",
                ),
            },
            "input_hashes": {
                name: require_sha256(
                    value, name=f"input_hashes.{name}"
                )
                for name, value in sorted(input_hashes.items())
            },
            "allowed_outputs": [
                "raw_offline_input_arrays",
                "deterministically_degraded_HLT_input_arrays",
                "relation_and_REGION_input_sidecars",
                "identity_and_source_manifests",
                "shared_label_free_HLT_inference_payloads",
            ],
            "checkpoint_loading_allowed": False,
            "labels_joined_to_model_output": False,
            "tokens_emitted": False,
            "logits_emitted": False,
            "probabilities_emitted": False,
            "predictions_emitted": False,
            "metrics_emitted": False,
        }
    )


def validate_prelock_final_test_inputs(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINAL_TEST_INPUT_PREPARATION_CONTRACT
    )
    expected = build_prelock_final_test_inputs(
        campaign_spec_sha256=payload.get("parents", {}).get(
            "campaign_spec"
        ),
        split_manifest_sha256=payload.get("parents", {}).get(
            "split_manifest"
        ),
        degradation_profile_sha256=payload.get("parents", {}).get(
            "degradation_profile"
        ),
        input_hashes=payload.get("input_hashes", {}),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("prelock final-test input semantics differ")
    return digest


def build_postlock_oracle_target(
    *,
    locked_scale_finalists: Mapping[str, Any],
    graph_id: str,
    pipeline_seed: int,
    split: str,
    parent_hashes: Mapping[str, str],
    target_cache_manifest_sha256: str,
    target_identity_order_sha256: str,
    target_dtype: str,
    float16_audit_sha256: str,
    float16_audit_passed: bool,
) -> dict[str, Any]:
    finalist_sha = validate_content_hash(
        locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    if (
        graph_id not in locked_scale_finalists.get(
            "finalist_graph_ids", []
        )
        or int(pipeline_seed) not in PIPELINE_SEEDS
        or split not in {"stack_val", "final_test"}
        or set(parent_hashes) != set(POSTLOCK_TARGET_PARENT_KEYS)
        or parent_hashes.get("locked_scale_finalists") != finalist_sha
        or target_dtype not in {"float16_audited", "float32_fallback"}
        or bool(float16_audit_passed)
        != (target_dtype == "float16_audited")
    ):
        raise ValueError("postlock target identity/parent coverage differs")
    run_map = {
        (row["graph_id"], row["pipeline_seed"]): row
        for row in locked_scale_finalists["all_shortlisted_scale_runs"]
    }
    run = run_map[(graph_id, int(pipeline_seed))]
    if (
        parent_hashes["scale_graph_run"]
        != run["scale_graph_run_sha256"]
        or parent_hashes["scale_offline_experts"]
        != run["component_hashes"]["offline_experts"]
        or parent_hashes["scale_offline_fusion"]
        != run["component_hashes"]["offline_fusion"]
        or parent_hashes["scale_target_normalizer"]
        != run["component_hashes"]["scale_target_token_normalizer"]
    ):
        raise ValueError("postlock target uses a non-scale teacher lineage")
    return with_content_hash(
        {
            "contract": POSTLOCK_ORACLE_TARGET_CONTRACT,
            "schema_version": 2,
            "graph_id": graph_id,
            "pipeline_seed": int(pipeline_seed),
            "split": split,
            "selection_eligible": False,
            "created_after_finalist_lock": True,
            "parent_hashes": {
                name: require_sha256(
                    value, name=f"parent_hashes.{name}"
                )
                for name, value in sorted(parent_hashes.items())
            },
            "target_cache_manifest_sha256": require_sha256(
                target_cache_manifest_sha256,
                name="target_cache_manifest_sha256",
            ),
            "target_identity_order_sha256": require_sha256(
                target_identity_order_sha256,
                name="target_identity_order_sha256",
            ),
            "target_dtype": target_dtype,
            "float16_round_trip_audit": {
                "artifact_sha256": require_sha256(
                    float16_audit_sha256,
                    name="float16_audit_sha256",
                ),
                "passed": bool(float16_audit_passed),
                "maximum_token_absolute_error": 5.0e-4,
                "maximum_expert_logit_absolute_error": 2.0e-4,
                "required_predicted_class_identities_exact": True,
                "failure_publishes_float32_and_continues": True,
            },
            "teacher_training_population": "scale_train",
            "five_hundred_k_teacher_allowed": False,
            "contains_tokens": True,
            "contains_expert_logits": True,
            "contains_labels": True,
            "test_result_may_affect_selection": False,
        }
    )


def validate_postlock_oracle_target(
    payload: Mapping[str, Any],
    *,
    locked_scale_finalists: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=POSTLOCK_ORACLE_TARGET_CONTRACT
    )
    expected = build_postlock_oracle_target(
        locked_scale_finalists=locked_scale_finalists,
        graph_id=payload.get("graph_id", ""),
        pipeline_seed=payload.get("pipeline_seed", -1),
        split=payload.get("split", ""),
        parent_hashes=payload.get("parent_hashes", {}),
        target_cache_manifest_sha256=payload.get(
            "target_cache_manifest_sha256"
        ),
        target_identity_order_sha256=payload.get(
            "target_identity_order_sha256"
        ),
        target_dtype=payload.get("target_dtype", ""),
        float16_audit_sha256=payload.get(
            "float16_round_trip_audit", {}
        ).get("artifact_sha256"),
        float16_audit_passed=payload.get(
            "float16_round_trip_audit", {}
        ).get("passed", False),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("postlock oracle-target semantics differ")
    return digest


def build_finalist_controls(
    *,
    locked_scale_finalists: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    finalist_sha = validate_content_hash(
        locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    finalist_ids = set(locked_scale_finalists["finalist_graph_ids"])
    if len(rows) != len(finalist_ids):
        raise ValueError("finalist-control graph coverage differs")
    checked, seen, evaluation_rows = [], set(), {}
    required = {
        "finalist_graph_id",
        "named_baseline_graph_id",
        "complete_graph_capacity_sha256",
        "H_MONO_PARAM_control_sha256",
        "H_MONO_FLOP_control_sha256",
        "H_BASE_LONG_control_sha256",
        "evaluation_rows",
    }
    for raw in rows:
        row = dict(raw)
        graph_id = str(row.get("finalist_graph_id"))
        if (
            set(row) != required
            or graph_id not in finalist_ids
            or graph_id in seen
            or not str(row["named_baseline_graph_id"])
        ):
            raise ValueError("finalist-control row differs")
        seen.add(graph_id)
        expected_pairs = {
            (kind, seed)
            for kind in CONTROL_KINDS
            for seed in PIPELINE_SEEDS
        }
        actual_pairs = set()
        local_rows = []
        for item in row["evaluation_rows"]:
            if set(item) != {
                "row_id",
                "owner_finalist_graph_id",
                "kind",
                "graph_id",
                "pipeline_seed",
                "checkpoint_sha256",
            }:
                raise ValueError("finalist evaluation-row fields differ")
            pair = (str(item["kind"]), int(item["pipeline_seed"]))
            row_id = str(item["row_id"])
            if (
                pair not in expected_pairs
                or pair in actual_pairs
                or not row_id
                or row_id in evaluation_rows
                or item["owner_finalist_graph_id"] != graph_id
            ):
                raise ValueError("finalist evaluation-row coverage differs")
            expected_graph = (
                graph_id
                if pair[0] == "FINALIST"
                else (
                    str(row["named_baseline_graph_id"])
                    if pair[0] == "NAMED_BASELINE"
                    else f"{pair[0]}::{graph_id}"
                )
            )
            if str(item["graph_id"]) != expected_graph:
                raise ValueError("finalist control graph identity differs")
            actual_pairs.add(pair)
            normalized = {
                "row_id": row_id,
                "owner_finalist_graph_id": graph_id,
                "kind": pair[0],
                "graph_id": expected_graph,
                "pipeline_seed": pair[1],
                "checkpoint_sha256": require_sha256(
                    item["checkpoint_sha256"],
                    name=f"evaluation_rows.{row_id}.checkpoint_sha256",
                ),
            }
            evaluation_rows[row_id] = normalized
            local_rows.append(normalized)
        if actual_pairs != expected_pairs:
            raise ValueError("finalist controls lack a kind/seed row")
        checked.append(
            {
                "finalist_graph_id": graph_id,
                "named_baseline_graph_id": str(
                    row["named_baseline_graph_id"]
                ),
                "complete_graph_capacity_sha256": require_sha256(
                    row["complete_graph_capacity_sha256"],
                    name=f"{graph_id}.complete_graph_capacity_sha256",
                ),
                "H_MONO_PARAM_control_sha256": require_sha256(
                    row["H_MONO_PARAM_control_sha256"],
                    name=f"{graph_id}.H_MONO_PARAM_control_sha256",
                ),
                "H_MONO_FLOP_control_sha256": require_sha256(
                    row["H_MONO_FLOP_control_sha256"],
                    name=f"{graph_id}.H_MONO_FLOP_control_sha256",
                ),
                "H_BASE_LONG_control_sha256": require_sha256(
                    row["H_BASE_LONG_control_sha256"],
                    name=f"{graph_id}.H_BASE_LONG_control_sha256",
                ),
                "evaluation_rows": sorted(
                    local_rows, key=lambda item: item["row_id"]
                ),
            }
        )
    return with_content_hash(
        {
            "contract": FINALIST_CONTROLS_CONTRACT,
            "schema_version": 1,
            "locked_scale_finalists_sha256": finalist_sha,
            "rows": sorted(
                checked, key=lambda row: row["finalist_graph_id"]
            ),
            "eligible_evaluation_rows": {
                row_id: evaluation_rows[row_id]
                for row_id in sorted(evaluation_rows)
            },
            "resolved_after_finalist_lock": True,
            "controls_selection_eligible": False,
            "controls_may_replace_finalist": False,
            "label_presentation_counts_cover_complete_training_graph": True,
            "capacity_covers_complete_deployable_graph": True,
        }
    )


def validate_finalist_controls(
    payload: Mapping[str, Any],
    *,
    locked_scale_finalists: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINALIST_CONTROLS_CONTRACT
    )
    expected = build_finalist_controls(
        locked_scale_finalists=locked_scale_finalists,
        rows=payload.get("rows", []),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("finalist-control semantics differ")
    return digest


def build_final_test_execution_lock(
    *,
    locked_scale_finalists: Mapping[str, Any],
    finalist_controls: Mapping[str, Any],
    postlock_targets: Sequence[Mapping[str, Any]],
    parent_hashes: Mapping[str, str],
    final_input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    finalist_sha = validate_content_hash(
        locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    controls_sha = validate_finalist_controls(
        finalist_controls,
        locked_scale_finalists=locked_scale_finalists,
    )
    if (
        set(parent_hashes) != set(EXECUTION_PARENT_KEYS)
        or parent_hashes.get("locked_scale_finalists") != finalist_sha
        or parent_hashes.get("finalist_controls") != controls_sha
        or set(final_input_hashes) != set(FINAL_INPUT_KEYS)
    ):
        raise ValueError("final-test execution-lock parents differ")
    expected_targets = {
        (graph_id, seed, split)
        for graph_id in locked_scale_finalists["finalist_graph_ids"]
        for seed in PIPELINE_SEEDS
        for split in ("stack_val", "final_test")
    }
    if len(postlock_targets) != len(expected_targets):
        raise ValueError("postlock target coverage differs")
    target_rows, seen = [], set()
    source = locked_scale_finalists.get("source")
    for target in postlock_targets:
        key = (
            str(target.get("graph_id")),
            int(target.get("pipeline_seed", -1)),
            str(target.get("split")),
        )
        if key not in expected_targets or key in seen:
            raise ValueError("postlock target graph/seed/split differs")
        seen.add(key)
        digest = validate_postlock_oracle_target(
            target, locked_scale_finalists=locked_scale_finalists
        )
        if target.get("source") != source:
            raise ValueError("postlock target source differs")
        target_rows.append(
            {
                "graph_id": key[0],
                "pipeline_seed": key[1],
                "split": key[2],
                "postlock_target_sha256": digest,
                "target_cache_manifest_sha256": target[
                    "target_cache_manifest_sha256"
                ],
                "target_identity_order_sha256": target[
                    "target_identity_order_sha256"
                ],
            }
        )
    return with_content_hash(
        {
            "contract": FINAL_TEST_EXECUTION_LOCK_CONTRACT,
            "schema_version": 1,
            "parent_hashes": {
                name: require_sha256(
                    value, name=f"parent_hashes.{name}"
                )
                for name, value in sorted(parent_hashes.items())
            },
            "final_input_hashes": {
                name: require_sha256(
                    value, name=f"final_input_hashes.{name}"
                )
                for name, value in sorted(final_input_hashes.items())
            },
            "postlock_targets": sorted(
                target_rows,
                key=lambda row: (
                    row["graph_id"],
                    row["pipeline_seed"],
                    row["split"],
                ),
            ),
            "eligible_evaluation_rows": finalist_controls[
                "eligible_evaluation_rows"
            ],
            "finalist_control_rows": finalist_controls["rows"],
            "finalist_graph_ids": list(
                locked_scale_finalists["finalist_graph_ids"]
            ),
            "ACCURACY_FINALIST": locked_scale_finalists[
                "ACCURACY_FINALIST"
            ],
            "REJECTION_FINALIST": locked_scale_finalists[
                "REJECTION_FINALIST"
            ],
            "degradation_profile_sha256": locked_scale_finalists[
                "lineage_hashes"
            ]["degradation_profile"],
            "all_postlock_targets_complete": True,
            "all_final_inputs_and_sidecars_bound": True,
            "all_graph_specific_controls_bound": True,
            "scientific_final_test_inference_authorized": True,
            "new_checkpoint_allowed": False,
            "test_result_may_select_replacement": False,
        }
    )


def validate_final_test_execution_lock(
    payload: Mapping[str, Any],
    *,
    locked_scale_finalists: Mapping[str, Any],
    finalist_controls: Mapping[str, Any],
    postlock_targets: Sequence[Mapping[str, Any]],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT
    )
    expected = build_final_test_execution_lock(
        locked_scale_finalists=locked_scale_finalists,
        finalist_controls=finalist_controls,
        postlock_targets=postlock_targets,
        parent_hashes=payload.get("parent_hashes", {}),
        final_input_hashes=payload.get("final_input_hashes", {}),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("final-test execution lock semantics differ")
    return digest


def build_final_test_execution_claim(
    *,
    execution_lock: Mapping[str, Any],
    execution_plan_sha256: str,
) -> dict[str, Any]:
    lock_sha = validate_content_hash(
        execution_lock,
        expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    )
    return with_content_hash(
        {
            "contract": FINAL_TEST_EXECUTION_CLAIM_CONTRACT,
            "schema_version": 2,
            "final_test_execution_lock_sha256": lock_sha,
            "execution_plan_sha256": require_sha256(
                execution_plan_sha256, name="execution_plan_sha256"
            ),
            "claimed_before_final_test_model_access": True,
            "retry_after_incomplete_claim_allowed": True,
            "completed_rows_must_not_be_reexecuted": True,
            "incomplete_rows_may_resume_under_same_claim": True,
            "row_completion_receipts_required": True,
            "claim_plan_row_checkpoint_bound_inference_attestation_required": True,
            "test_result_may_select_replacement": False,
        }
    )


def validate_final_test_execution_claim(
    payload: Mapping[str, Any],
    *,
    execution_lock: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINAL_TEST_EXECUTION_CLAIM_CONTRACT
    )
    expected = build_final_test_execution_claim(
        execution_lock=execution_lock,
        execution_plan_sha256=payload.get("execution_plan_sha256"),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("final-test execution claim semantics differ")
    return digest


def _final_prediction(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    identities: Sequence[str],
    expected_degradation_profile_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    if set(row) != {
        "row_id",
        "graph_id",
        "pipeline_seed",
        "checkpoint_sha256",
        "degradation_profile_sha256",
        "identities",
        "logits",
        "probabilities",
        "prediction_artifact_sha256",
    }:
        raise ValueError("final-test prediction row fields differ")
    logits = np.asarray(row["logits"])
    probability = np.asarray(row["probabilities"])
    expected_probability = stable_probabilities(
        logits.astype(np.float64)
    ).astype(np.float32)
    if (
        str(row["row_id"]) != expected["row_id"]
        or str(row["graph_id"]) != expected["graph_id"]
        or int(row["pipeline_seed"]) != expected["pipeline_seed"]
        or row["checkpoint_sha256"] != expected["checkpoint_sha256"]
        or row["degradation_profile_sha256"]
        != expected_degradation_profile_sha256
        or list(row["identities"]) != list(identities)
        or logits.shape != (len(identities), 10)
        or logits.dtype != np.float32
        or probability.dtype != np.float32
        or probability.shape != logits.shape
        or not np.isfinite(logits).all()
        or not np.allclose(
            probability,
            expected_probability,
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        or not np.array_equal(
            probability.argmax(axis=1), logits.argmax(axis=1)
        )
    ):
        raise ValueError("sealed final-test prediction semantics differ")
    require_sha256(
        row["degradation_profile_sha256"],
        name="degradation_profile_sha256",
    )
    require_sha256(
        row["prediction_artifact_sha256"],
        name="prediction_artifact_sha256",
    )
    return logits, probability


def build_sealed_final_test_evaluation(
    *,
    execution_lock: Mapping[str, Any],
    execution_claim: Mapping[str, Any],
    identities: Sequence[str],
    labels: np.ndarray,
    final_labels_artifact_sha256: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    lock_sha = validate_content_hash(
        execution_lock,
        expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    )
    claim_sha = validate_final_test_execution_claim(
        execution_claim, execution_lock=execution_lock
    )
    if execution_claim.get("source") != execution_lock.get("source"):
        raise ValueError("final-test execution claim source differs")
    ids = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    expected_rows = execution_lock["eligible_evaluation_rows"]
    final_targets = [
        row
        for row in execution_lock["postlock_targets"]
        if row["split"] == "final_test"
    ]
    final_identity_hash = canonical_sha256(list(ids))
    label_artifact_sha = require_sha256(
        final_labels_artifact_sha256,
        name="final_labels_artifact_sha256",
    )
    if (
        ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
        or bool(((truth < 0) | (truth >= 10)).any())
        or len(set(np.bincount(truth, minlength=10).tolist())) != 1
        or len(prediction_rows) != len(expected_rows)
        or {row["target_identity_order_sha256"] for row in final_targets}
        != {final_identity_hash}
    ):
        raise ValueError("final-test identity/label population differs")
    checked, logits_by_row, metrics = {}, {}, {}
    for row in prediction_rows:
        row_id = str(row.get("row_id"))
        if row_id not in expected_rows or row_id in checked:
            raise ValueError("final-test row is unregistered or duplicated")
        logits, _ = _final_prediction(
            row,
            expected=expected_rows[row_id],
            identities=ids,
            expected_degradation_profile_sha256=execution_lock[
                "degradation_profile_sha256"
            ],
        )
        checked[row_id] = {
            "prediction_artifact_sha256": row[
                "prediction_artifact_sha256"
            ],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "degradation_profile_sha256": row[
                "degradation_profile_sha256"
            ],
        }
        logits_by_row[row_id] = logits
        metrics[row_id] = bind_source(
            evaluate_classification(
                logits.astype(np.float64),
                truth,
                split="final_test",
            ),
            source_snapshot=source_snapshot,
        )
    if set(checked) != set(expected_rows):
        raise ValueError("final-test eligible-row coverage differs")
    paired = {}
    by_identity = {
        (
            row["owner_finalist_graph_id"],
            row["kind"],
            row["pipeline_seed"],
        ): row_id
        for row_id, row in expected_rows.items()
    }
    for row_id, row in expected_rows.items():
        if row["kind"] != "FINALIST":
            continue
        finalist_graph = row["graph_id"]
        baseline_row_id = by_identity[
            (
                row["owner_finalist_graph_id"],
                "NAMED_BASELINE",
                row["pipeline_seed"],
            )
        ]
        baseline_graph = expected_rows[baseline_row_id]["graph_id"]
        paired[row_id] = bind_source(
            build_paired_confirmation_statistics(
                identities=ids,
                labels=truth,
                candidate_logits=logits_by_row[row_id],
                baseline_logits=logits_by_row[baseline_row_id],
                candidate_graph_id=finalist_graph,
                baseline_graph_id=baseline_graph,
                pipeline_seed=row["pipeline_seed"],
                candidate_prediction_sha256=checked[row_id][
                    "prediction_artifact_sha256"
                ],
                baseline_prediction_sha256=checked[baseline_row_id][
                    "prediction_artifact_sha256"
                ],
            ),
            source_snapshot=source_snapshot,
        )
    paired_between_finalists = {}
    finalist_graph_ids = execution_lock["finalist_graph_ids"]
    if len(finalist_graph_ids) == 2:
        accuracy_graph = execution_lock["ACCURACY_FINALIST"]
        rejection_graph = execution_lock["REJECTION_FINALIST"]
        control_rows = execution_lock["finalist_control_rows"]
        available = {row["finalist_graph_id"] for row in control_rows}
        if set(finalist_graph_ids) != available:
            raise ValueError("finalist control ownership differs")
        for seed in PIPELINE_SEEDS:
            accuracy_row_id = by_identity[
                (accuracy_graph, "FINALIST", seed)
            ]
            rejection_row_id = by_identity[
                (rejection_graph, "FINALIST", seed)
            ]
            key = f"{accuracy_graph}_minus_{rejection_graph}:seed{seed}"
            paired_between_finalists[key] = bind_source(
                build_paired_confirmation_statistics(
                    identities=ids,
                    labels=truth,
                    candidate_logits=logits_by_row[accuracy_row_id],
                    baseline_logits=logits_by_row[rejection_row_id],
                    candidate_graph_id=accuracy_graph,
                    baseline_graph_id=rejection_graph,
                    pipeline_seed=seed,
                    candidate_prediction_sha256=checked[
                        accuracy_row_id
                    ]["prediction_artifact_sha256"],
                    baseline_prediction_sha256=checked[
                        rejection_row_id
                    ]["prediction_artifact_sha256"],
                ),
                source_snapshot=source_snapshot,
            )
    return bind_source(
        with_content_hash(
            {
                "contract": FINAL_TEST_EVALUATION_CONTRACT,
                "schema_version": 3,
                "final_test_execution_lock_sha256": lock_sha,
                "final_test_execution_claim_sha256": claim_sha,
                "identity_count": len(ids),
                "identity_order_sha256": canonical_sha256(list(ids)),
                "final_labels_artifact_sha256": label_artifact_sha,
                "final_labels_joined_only_after_execution_claim": True,
                "balanced_count_per_class": int(
                    np.bincount(truth, minlength=10)[0]
                ),
                "prediction_artifacts": checked,
                "classification_metrics": metrics,
                "paired_finalist_statistics": paired,
                "paired_between_distinct_finalists": (
                    paired_between_finalists
                ),
                "evaluated_row_ids": sorted(checked),
                "all_and_only_locked_rows_evaluated": True,
                "new_checkpoint_loaded": False,
                "test_result_selected_replacement": False,
                "evaluation_once_by_immutable_publication": True,
            }
        ),
        source_snapshot=source_snapshot,
    )


def publish_final_test_evaluation(
    *,
    output_dir: str | Path,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        artifact, expected_contract=FINAL_TEST_EVALUATION_CONTRACT
    )
    return write_immutable_json(
        Path(output_dir) / "retb_sealed_final_test_evaluation.json",
        artifact,
    )


__all__ = [
    "CONTROL_KINDS",
    "EXECUTION_PARENT_KEYS",
    "FINALIST_CONTROLS_CONTRACT",
    "FINAL_TEST_INPUT_PREPARATION_CONTRACT",
    "FINAL_INPUT_KEYS",
    "FINAL_TEST_EVALUATION_CONTRACT",
    "FINAL_TEST_EXECUTION_CLAIM_CONTRACT",
    "FINAL_TEST_EXECUTION_LOCK_CONTRACT",
    "POSTLOCK_ORACLE_TARGET_CONTRACT",
    "POSTLOCK_TARGET_PARENT_KEYS",
    "build_final_test_execution_lock",
    "build_final_test_execution_claim",
    "build_prelock_final_test_inputs",
    "build_finalist_controls",
    "build_postlock_oracle_target",
    "build_sealed_final_test_evaluation",
    "publish_final_test_evaluation",
    "validate_final_test_execution_lock",
    "validate_final_test_execution_claim",
    "validate_finalist_controls",
    "validate_postlock_oracle_target",
    "validate_prelock_final_test_inputs",
]
