"""Leakage-safe stack selection and exactly-once final-test sealing."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (
    mean_log_selection_rejection,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)

from .contracts import (
    FINAL_EVALUATION_CONTRACT,
    FINAL_EXECUTION_CLAIM_CONTRACT,
    FINAL_EXECUTION_LOCK_CONTRACT,
    FINAL_ROW_RESULT_CONTRACT,
    FINAL_INPUT_PREPARATION_CONTRACT,
    FINALIST_LOCK_CONTRACT,
    POSTLOCK_ORACLE_CONTRACT,
    STACK_PREDICTION_MANIFEST_CONTRACT,
    STACK_SELECTOR_TRACE_CONTRACT,
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


FINALIST_SEEDS = (202, 303, 404)


def build_stack_prediction_manifest(
    *,
    graph_id: str,
    seed: int,
    identities: Sequence[str],
    logits: np.ndarray,
    checkpoint_sha256: str,
    export_sha256: str,
    lineage_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    ids = tuple(str(value) for value in identities)
    values = np.asarray(logits, dtype=np.float32)
    if (
        int(seed) not in FINALIST_SEEDS
        or not ids
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or values.shape != (len(ids), 10)
        or not np.isfinite(values).all()
    ):
        raise ValueError("stack prediction population differs")
    return with_content_hash(
        {
            "contract": STACK_PREDICTION_MANIFEST_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "graph_id": str(graph_id),
            "seed": int(seed),
            "split": "stack_val",
            "identities": list(ids),
            "identity_order_sha256": canonical_sha256(list(ids)),
            "logits": values.astype(float).tolist(),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "export_sha256": require_sha256(export_sha256, name="export_sha256"),
            "lineage_hashes": {
                key: require_sha256(value, name=f"lineage.{key}")
                for key, value in sorted(lineage_hashes.items())
            },
            "contains_labels": False,
            "contains_targets": False,
            "runtime_inputs": "hlt_only",
        }
    )


def select_stack_finalists(
    *,
    predictions: Sequence[Mapping[str, Any]],
    label_identities: Sequence[str],
    labels: np.ndarray,
    label_manifest_sha256: str,
    capacity_by_graph: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    ids = tuple(str(value) for value in label_identities)
    truth = np.asarray(labels, dtype=np.int64)
    if (
        ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
    ):
        raise ValueError("stack selector label population differs")
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for prediction in predictions:
        validate_content_hash(
            prediction, expected_contract=STACK_PREDICTION_MANIFEST_CONTRACT
        )
        if prediction.get("source") != dict(source):
            raise ValueError("stack prediction source differs")
        if prediction.get("contains_labels") or prediction.get("contains_targets"):
            raise ValueError("stack prediction shard leaks privileged evidence")
        if tuple(prediction["identities"]) != ids:
            raise ValueError("stack prediction/label identity join differs")
        grouped.setdefault(prediction["graph_id"], {})[
            int(prediction["seed"])
        ] = prediction
    if not grouped or set(capacity_by_graph) != set(grouped):
        raise ValueError("stack graph/capacity coverage differs")
    rows = []
    for graph, by_seed in sorted(grouped.items()):
        if set(by_seed) != set(FINALIST_SEEDS):
            raise ValueError("stack graph has incomplete three-seed coverage")
        capacity = capacity_by_graph[graph]
        if (
            not bool(capacity.get("deployable"))
            or not bool(capacity.get("export_parity_validated"))
        ):
            raise ValueError("stack graph is not eligible/deployable")
        metrics = []
        for seed in FINALIST_SEEDS:
            logits = np.asarray(by_seed[seed]["logits"], dtype=np.float64)
            metrics.append(
                evaluate_classification(logits, truth, split="stack_val")
            )
        rows.append(
            {
                "graph_id": graph,
                "mean_balanced_accuracy": float(
                    np.mean(
                        [
                            row["macro_per_class_accuracy"]
                            for row in metrics
                        ]
                    )
                ),
                "mean_cross_entropy": float(
                    np.mean([row["cross_entropy"] for row in metrics])
                ),
                "mean_log_rejection": float(
                    np.mean(
                        [mean_log_selection_rejection(row) for row in metrics]
                    )
                ),
                "inference_flops": float(capacity["inference_flops"]),
                "deployed_parameters": int(capacity["deployed_parameters"]),
                "prediction_hashes": [
                    by_seed[seed]["content_hash"] for seed in FINALIST_SEEDS
                ],
                "metrics_hashes": [row["content_hash"] for row in metrics],
                "checkpoint_hashes": [
                    by_seed[seed]["checkpoint_sha256"]
                    for seed in FINALIST_SEEDS
                ],
                "export_hashes": [
                    by_seed[seed]["export_sha256"] for seed in FINALIST_SEEDS
                ],
                "lineage_hashes": dict(capacity["lineage_hashes"]),
            }
        )
    accuracy = min(
        rows,
        key=lambda row: (
            -row["mean_balanced_accuracy"],
            row["mean_cross_entropy"],
            row["inference_flops"],
            row["deployed_parameters"],
            row["graph_id"],
        ),
    )
    rejection = min(
        rows,
        key=lambda row: (
            -row["mean_log_rejection"],
            -row["mean_balanced_accuracy"],
            row["inference_flops"],
            row["deployed_parameters"],
            row["graph_id"],
        ),
    )
    return with_content_hash(
        {
            "contract": STACK_SELECTOR_TRACE_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "split": "stack_val",
            "label_manifest_sha256": require_sha256(
                label_manifest_sha256, name="label_manifest_sha256"
            ),
            "identity_order_sha256": canonical_sha256(list(ids)),
            "eligible_graphs": rows,
            "accuracy_finalist_graph_id": accuracy["graph_id"],
            "rejection_finalist_graph_id": rejection["graph_id"],
            "finalists_may_be_identical": True,
            "minimum_improvement_required": False,
            "final_test_consumed": False,
        }
    )


def build_finalist_lock(
    *,
    selector_trace: Mapping[str, Any],
    campaign_spec_sha256: str,
    required_lineage_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        selector_trace, expected_contract=STACK_SELECTOR_TRACE_CONTRACT
    )
    if selector_trace.get("source") != dict(source):
        raise ValueError("selector trace source differs")
    if not required_lineage_hashes:
        raise ValueError("finalist lock lineage must not be empty")
    checked = {
        key: require_sha256(value, name=f"finalist_lineage.{key}")
        for key, value in sorted(required_lineage_hashes.items())
    }
    # The caller supplies the exhaustive campaign-level set while each graph
    # carries only lineage relevant to that graph.
    required_categories = {
        "split_manifest",
        "validation_partition",
        "scale_pool",
        "hlt_profile",
        "hlt_replica_manifest",
        "hlt_cache_audit",
        "target_capability_audit",
        "target_registry",
        "confirmation_summary",
        "scale_shortlist",
        "scale_export_audit",
        "label_manifest",
        "selector_metrics",
        "selector_trace",
    }
    if not required_categories.issubset(checked):
        raise ValueError("finalist lock lacks required lineage categories")
    finalists = sorted(
        {
            selector_trace["accuracy_finalist_graph_id"],
            selector_trace["rejection_finalist_graph_id"],
        }
    )
    return with_content_hash(
        {
            "contract": FINALIST_LOCK_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "selector_trace_sha256": selector_trace["content_hash"],
            "lineage_hashes": checked,
            "accuracy_finalist_graph_id": selector_trace[
                "accuracy_finalist_graph_id"
            ],
            "rejection_finalist_graph_id": selector_trace[
                "rejection_finalist_graph_id"
            ],
            "unique_finalist_graph_ids": finalists,
            "locked_selection_artifacts": {
                row["graph_id"]: {
                    "prediction_hashes": list(row["prediction_hashes"]),
                    "metrics_hashes": list(row["metrics_hashes"]),
                    "checkpoint_hashes": list(row["checkpoint_hashes"]),
                    "export_hashes": list(row["export_hashes"]),
                    "lineage_hashes": dict(row["lineage_hashes"]),
                }
                for row in selector_trace["eligible_graphs"]
            },
            "postlock_oracle_diagnostics_authorized": True,
            "final_test_inference_authorized": False,
            "gain_sign_does_not_affect_lock": True,
        }
    )


def build_postlock_oracle_manifest(
    *,
    finalist_lock: Mapping[str, Any],
    diagnostic_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(finalist_lock, expected_contract=FINALIST_LOCK_CONTRACT)
    if (
        finalist_lock.get("source") != dict(source)
        or set(diagnostic_hashes)
        != {"stack_val_offline_teacher_agreement"}
    ):
        raise ValueError("post-lock oracle lineage differs or is empty")
    return with_content_hash(
        {
            "contract": POSTLOCK_ORACLE_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "finalist_lock_sha256": finalist_lock["content_hash"],
            "diagnostic_hashes": {
                key: require_sha256(value, name=f"diagnostic.{key}")
                for key, value in sorted(diagnostic_hashes.items())
            },
            "split": "stack_val",
            "selection_eligible": False,
            "generated_after_finalist_lock": True,
            "offline_teacher_inference_executed": True,
            "labels_consumed": False,
            "oracle_outputs_persisted": False,
        }
    )


def build_final_input_preparation(
    *,
    finalist_lock: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(finalist_lock, expected_contract=FINALIST_LOCK_CONTRACT)
    if finalist_lock.get("source") != dict(source) or not input_hashes:
        raise ValueError("final input preparation lineage differs or is empty")
    return with_content_hash(
        {
            "contract": FINAL_INPUT_PREPARATION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "finalist_lock_sha256": finalist_lock["content_hash"],
            "input_hashes": {
                key: require_sha256(value, name=f"input.{key}")
                for key, value in sorted(input_hashes.items())
            },
            "split": "final_test",
            "model_inference_performed": False,
        }
    )


def build_final_execution_lock(
    *,
    finalist_lock: Mapping[str, Any],
    postlock_oracle: Mapping[str, Any],
    prepared_inputs: Mapping[str, Any],
    finalist_control_hashes: Mapping[str, str],
    finalist_control_rows: Sequence[Mapping[str, Any]] = (),
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(finalist_lock, expected_contract=FINALIST_LOCK_CONTRACT)
    validate_content_hash(postlock_oracle, expected_contract=POSTLOCK_ORACLE_CONTRACT)
    validate_content_hash(
        prepared_inputs, expected_contract=FINAL_INPUT_PREPARATION_CONTRACT
    )
    if (
        finalist_lock.get("source") != dict(source)
        or postlock_oracle.get("source") != dict(source)
        or prepared_inputs.get("source") != dict(source)
        or postlock_oracle.get("finalist_lock_sha256")
        != finalist_lock["content_hash"]
        or prepared_inputs.get("finalist_lock_sha256")
        != finalist_lock["content_hash"]
    ):
        raise ValueError("final execution lock parent lineage differs")
    final_rows = []
    locked = finalist_lock["locked_selection_artifacts"]
    final_graph_ids = list(finalist_lock["unique_finalist_graph_ids"])
    for required_baseline in ("H_BASE", "H_PARTICLENET"):
        if (
            required_baseline in locked
            and required_baseline not in final_graph_ids
        ):
            final_graph_ids.append(required_baseline)
    for graph_id in sorted(final_graph_ids):
        evidence = locked[graph_id]
        for index, seed in enumerate(FINALIST_SEEDS):
            final_rows.append(
                {
                    "row_id": (
                        "FINAL_"
                        + canonical_sha256([graph_id, int(seed)])[:16]
                    ),
                    "graph_id": graph_id,
                    "seed": int(seed),
                    "checkpoint_sha256": evidence["checkpoint_hashes"][index],
                    "export_sha256": evidence["export_hashes"][index],
                }
            )
    seen_control_rows = set()
    for raw in finalist_control_rows:
        row = {
            key: raw[key]
            for key in (
                "row_id",
                "graph_id",
                "seed",
                "checkpoint_sha256",
                "export_sha256",
                "export_path",
                "control_family",
            )
        }
        for optional in (
            "matched_finalist_graph_id",
            "comparison_finalist_graph_id",
            "semantic_control_row_id",
            "semantic_reference_scope",
        ):
            if optional in raw:
                row[optional] = str(raw[optional])
        row["row_id"] = str(row["row_id"])
        if row["row_id"] in seen_control_rows:
            raise ValueError("finalist control row is duplicated")
        seen_control_rows.add(row["row_id"])
        row["seed"] = int(row["seed"])
        row["checkpoint_sha256"] = require_sha256(
            row["checkpoint_sha256"],
            name=f"{row['row_id']}.checkpoint_sha256",
        )
        row["export_sha256"] = require_sha256(
            row["export_sha256"],
            name=f"{row['row_id']}.export_sha256",
        )
        final_rows.append(row)
    rows = tuple(row["row_id"] for row in final_rows)
    controls = {
        key: require_sha256(value, name=f"final_control.{key}")
        for key, value in sorted(finalist_control_hashes.items())
    }
    required_controls = {
        "capacity_controls",
        "semantic_controls",
        "export_parity",
        "latency_controls",
        "matched_baselines",
    }
    if not required_controls.issubset(controls):
        raise ValueError("final execution lock lacks control coverage")
    nonce = canonical_sha256(
        [
            finalist_lock["content_hash"],
            postlock_oracle["content_hash"],
            prepared_inputs["content_hash"],
            final_rows,
        ]
    )
    return with_content_hash(
        {
            "contract": FINAL_EXECUTION_LOCK_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "finalist_lock_sha256": finalist_lock["content_hash"],
            "postlock_oracle_sha256": postlock_oracle["content_hash"],
            "prepared_inputs_sha256": prepared_inputs["content_hash"],
            "finalist_control_hashes": controls,
            "final_row_ids": list(rows),
            "final_rows": final_rows,
            "deployable_final_graph_ids": sorted(final_graph_ids),
            "mandatory_scaled_baselines_included": all(
                graph_id in final_graph_ids
                for graph_id in ("H_BASE", "H_PARTICLENET")
                if graph_id in locked
            ),
            "finalist_control_row_count": len(finalist_control_rows),
            "execution_claim": {
                "nonce": nonce,
                "initial_state": "unused",
                "exactly_once": True,
            },
            "final_test_inference_authorized": True,
            "replacement_selection_after_final_metrics_allowed": False,
        }
    )


def consume_final_execution_claim(
    *,
    execution_lock: Mapping[str, Any],
    claim_path: str | Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        execution_lock, expected_contract=FINAL_EXECUTION_LOCK_CONTRACT
    )
    path = Path(claim_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = with_content_hash(
        {
            "contract": FINAL_EXECUTION_CLAIM_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "execution_lock_sha256": execution_lock["content_hash"],
            "nonce": execution_lock["execution_claim"]["nonce"],
            "state": "consumed",
        }
    )
    data = canonical_json_bytes(claim)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError("final-test execution claim was already consumed") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A created claim is intentionally never removed automatically. A
        # partial execution is an integrity incident requiring authorization.
        raise
    return claim


def load_final_execution_claim(
    *,
    execution_lock: Mapping[str, Any],
    claim_path: str | Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the one durable claim when resuming interrupted row execution."""

    validate_content_hash(
        execution_lock, expected_contract=FINAL_EXECUTION_LOCK_CONTRACT
    )
    from .contracts import load_hashed_json

    claim = load_hashed_json(
        claim_path, expected_contract=FINAL_EXECUTION_CLAIM_CONTRACT
    )
    if (
        claim.get("source") != dict(source)
        or claim.get("execution_lock_sha256") != execution_lock["content_hash"]
        or claim.get("nonce") != execution_lock["execution_claim"]["nonce"]
        or claim.get("state") != "consumed"
    ):
        raise ValueError("existing final execution claim lineage differs")
    return claim


def build_final_row_result(
    *,
    execution_lock: Mapping[str, Any],
    consumed_claim: Mapping[str, Any],
    row_id: str,
    graph_id: str,
    seed: int,
    export_sha256: str,
    checkpoint_sha256: str,
    classification_metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        execution_lock, expected_contract=FINAL_EXECUTION_LOCK_CONTRACT
    )
    validate_content_hash(
        consumed_claim, expected_contract=FINAL_EXECUTION_CLAIM_CONTRACT
    )
    if (
        row_id not in execution_lock["final_row_ids"]
        or consumed_claim.get("execution_lock_sha256")
        != execution_lock["content_hash"]
        or consumed_claim.get("source") != dict(source)
    ):
        raise ValueError("final row execution lineage differs")
    expected = {
        row["row_id"]: row for row in execution_lock["final_rows"]
    }[row_id]
    if (
        str(graph_id) != expected["graph_id"]
        or int(seed) != int(expected["seed"])
        or export_sha256 != expected["export_sha256"]
        or checkpoint_sha256 != expected["checkpoint_sha256"]
    ):
        raise ValueError("final row graph/export identity differs")
    return with_content_hash(
        {
            "contract": FINAL_ROW_RESULT_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "execution_lock_sha256": execution_lock["content_hash"],
            "execution_claim_sha256": consumed_claim["content_hash"],
            "row_id": str(row_id),
            "graph_id": str(graph_id),
            "seed": int(seed),
            "export_sha256": require_sha256(
                export_sha256, name="export_sha256"
            ),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "classification_metrics": dict(classification_metrics),
            "completed": True,
        }
    )


def build_final_evaluation(
    *,
    execution_lock: Mapping[str, Any],
    consumed_claim: Mapping[str, Any],
    row_results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        execution_lock, expected_contract=FINAL_EXECUTION_LOCK_CONTRACT
    )
    validate_content_hash(
        consumed_claim, expected_contract=FINAL_EXECUTION_CLAIM_CONTRACT
    )
    by_id = {row["row_id"]: row for row in row_results}
    if set(by_id) != set(execution_lock["final_row_ids"]):
        raise ValueError("final evaluation row coverage differs")
    if (
        consumed_claim.get("execution_lock_sha256")
        != execution_lock["content_hash"]
        or consumed_claim.get("source") != dict(source)
    ):
        raise ValueError("final evaluation claim lineage differs")
    for row in by_id.values():
        validate_content_hash(
            row, expected_contract=FINAL_ROW_RESULT_CONTRACT
        )
        if (
            row.get("source") != dict(source)
            or row.get("execution_lock_sha256")
            != execution_lock["content_hash"]
            or row.get("execution_claim_sha256")
            != consumed_claim["content_hash"]
        ):
            raise ValueError("final evaluation row lineage differs")
    return with_content_hash(
        {
            "contract": FINAL_EVALUATION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "execution_lock_sha256": execution_lock["content_hash"],
            "execution_claim_sha256": consumed_claim["content_hash"],
            "row_result_hashes": {
                key: by_id[key]["content_hash"] for key in sorted(by_id)
            },
            "rows": [by_id[key] for key in sorted(by_id)],
            "final_test_executions": 1,
            "final_metrics_used_for_selection": False,
            "negative_results_reported": True,
        }
    )


__all__ = [
    "FINALIST_SEEDS",
    "build_final_evaluation",
    "build_final_row_result",
    "build_final_execution_lock",
    "build_final_input_preparation",
    "build_finalist_lock",
    "build_postlock_oracle_manifest",
    "build_stack_prediction_manifest",
    "consume_final_execution_claim",
    "load_final_execution_claim",
    "select_stack_finalists",
]
