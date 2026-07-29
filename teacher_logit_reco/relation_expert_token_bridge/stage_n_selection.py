"""Label-free Stage-N prediction shards and deterministic dual finalists."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .confirmation import (
    SCALE_SHORTLIST_CONTRACT,
    mean_log_selection_rejection,
)
from .contracts import (
    bind_source,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .evaluation import evaluate_classification, stable_probabilities
from .predictor_bundle import PIPELINE_SEEDS
from .scale_up import SCALE_COMPLETION_CONTRACT
from .splits import FINAL_SELECT_LABEL_MANIFEST_CONTRACT


STACK_SELECTION_PREDICTION_CONTRACT = (
    "retb_stack_val_label_free_prediction_v1"
)
STACK_SELECTION_METRICS_CONTRACT = "retb_stack_val_selection_metrics_v1"
SCALE_FINALIST_TRACE_CONTRACT = "retb_scale_finalist_selection_trace_v1"
LOCKED_SCALE_FINALISTS_CONTRACT = "retb_locked_scale_finalists_v1"

PREDICTION_PARENT_KEYS = frozenset(
    {
        "campaign_spec",
        "locked_scale_shortlist",
        "scale_completion",
        "scale_graph_run",
        "deployable_export",
        "stack_val_HLT_input_manifest",
        "deployable_scale_normalizer_bundle",
        "degradation_profile",
    }
)
FINALIST_LINEAGE_KEYS = frozenset(
    {
        "campaign_spec",
        "step14_bundle",
        "locked_scale_shortlist",
        "scale_completion",
        "split_manifest",
        "validation_partition_manifest",
        "scale_train_manifest",
        "hlt_replica_manifest",
        "hlt_cache_manifest",
        "realization_policy",
        "degradation_profile",
        "degradation_parameters",
        "offline_scale_normalizer_bundle",
        "shared_hlt_scale_normalizer_bundle",
        "shape_registry",
        "target_coordinate_lock",
        "predictor_bundle_lock",
    }
)
FORBIDDEN_PREDICTION_TERMS = (
    "label",
    "offline",
    "oracle",
    "target",
)
REQUIRED_SHAPE_ASSIGNMENTS = frozenset(
    {
        "SHAPE_HIGH",
        "SHAPE_COMPACT",
        "SHAPE_BRIDGE",
        "HET_PHYSICS",
        "HET_SELECTED",
        "HET_BEAM",
    }
)


def _prediction_arrays(
    *,
    identities: Sequence[str],
    graph_id: str,
    pipeline_seed: int,
    logits: np.ndarray,
    probabilities: np.ndarray | None,
) -> dict[str, np.ndarray]:
    ids = tuple(str(value) for value in identities)
    values = np.asarray(logits)
    if (
        not ids
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or values.shape != (len(ids), 10)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or not str(graph_id)
        or int(pipeline_seed) not in PIPELINE_SEEDS
    ):
        raise ValueError("stack-val prediction identity/logit semantics differ")
    expected_probability = stable_probabilities(
        values.astype(np.float64)
    ).astype(np.float32)
    probability = (
        expected_probability
        if probabilities is None
        else np.asarray(probabilities)
    )
    if (
        probability.shape != values.shape
        or probability.dtype != np.float32
        or not np.isfinite(probability).all()
        or not np.allclose(
            probability,
            expected_probability,
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        or not np.array_equal(
            probability.argmax(axis=1), values.argmax(axis=1)
        )
    ):
        raise ValueError("stored stack-val probabilities differ from logits")
    return {
        "identities": np.asarray(ids, dtype=np.str_),
        "graph_ids": np.full(len(ids), str(graph_id), dtype=np.str_),
        "pipeline_seeds": np.full(
            len(ids), int(pipeline_seed), dtype=np.int64
        ),
        "logits": values,
        "probabilities": probability,
    }


def build_stack_selection_prediction_manifest(
    *,
    identities: Sequence[str],
    graph_id: str,
    pipeline_seed: int,
    logits: np.ndarray,
    probabilities: np.ndarray | None,
    npz_filename: str,
    npz_sha256: str,
    parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    arrays = _prediction_arrays(
        identities=identities,
        graph_id=graph_id,
        pipeline_seed=pipeline_seed,
        logits=logits,
        probabilities=probabilities,
    )
    if (
        set(parent_hashes) != set(PREDICTION_PARENT_KEYS)
        or Path(npz_filename).name != npz_filename
        or not npz_filename.endswith(".npz")
        or any(
            any(term in name.lower() for term in FORBIDDEN_PREDICTION_TERMS)
            for name in parent_hashes
        )
    ):
        raise ValueError("stack-val prediction parent/filename differs")
    return with_content_hash(
        {
            "contract": STACK_SELECTION_PREDICTION_CONTRACT,
            "schema_version": 1,
            "graph_id": str(graph_id),
            "pipeline_seed": int(pipeline_seed),
            "split": "stack_val",
            "selection_eligible": True,
            "identity_count": len(arrays["identities"]),
            "identity_order": "lexicographic_canonical_jet_identity",
            "identity_order_sha256": canonical_sha256(
                arrays["identities"].tolist()
            ),
            "npz_filename": npz_filename,
            "npz_sha256": require_sha256(
                npz_sha256, name="npz_sha256"
            ),
            "npz_fields": [
                "graph_ids",
                "identities",
                "logits",
                "pipeline_seeds",
                "probabilities",
            ],
            "logit_dtype": "float32",
            "probability_dtype": "float32",
            "probability_absolute_tolerance": 1.0e-6,
            "probability_relative_tolerance": 1.0e-6,
            "predicted_class_identities_exact": True,
            "parent_hashes": {
                name: require_sha256(
                    value, name=f"parent_hashes.{name}"
                )
                for name, value in sorted(parent_hashes.items())
            },
            "contains_labels": False,
            "contains_offline_targets": False,
            "contains_oracle_tokens": False,
            "contains_oracle_logits": False,
            "HLT_only_deployable_inference": True,
            "final_test_consumed": False,
        }
    )


def publish_stack_selection_prediction(
    *,
    output_dir: str | Path,
    identities: Sequence[str],
    graph_id: str,
    pipeline_seed: int,
    logits: np.ndarray,
    probabilities: np.ndarray | None,
    parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    arrays = _prediction_arrays(
        identities=identities,
        graph_id=graph_id,
        pipeline_seed=pipeline_seed,
        logits=logits,
        probabilities=probabilities,
    )
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    encoded = stream.getvalue()
    filename = (
        f"{graph_id}_seed{int(pipeline_seed)}_stack_val_predictions.npz"
    )
    root = Path(output_dir)
    publication = write_immutable_bytes(root / filename, encoded)
    manifest = bind_source(
        build_stack_selection_prediction_manifest(
            identities=arrays["identities"].tolist(),
            graph_id=graph_id,
            pipeline_seed=pipeline_seed,
            logits=arrays["logits"],
            probabilities=arrays["probabilities"],
            npz_filename=filename,
            npz_sha256=publication["file_sha256"],
            parent_hashes=parent_hashes,
        ),
        source_snapshot=source_snapshot,
    )
    manifest_publication = write_immutable_json(
        root
        / f"{graph_id}_seed{int(pipeline_seed)}_stack_val_predictions.json",
        manifest,
    )
    return {
        "manifest": manifest,
        "npz_publication": publication,
        "manifest_publication": manifest_publication,
    }


def load_stack_selection_prediction(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(manifest_path)
    from .contracts import load_hashed_json

    manifest = load_hashed_json(
        path, expected_contract=STACK_SELECTION_PREDICTION_CONTRACT
    )
    npz_path = path.parent / manifest["npz_filename"]
    if (
        not npz_path.is_file()
        or npz_path.is_symlink()
        or hashlib.sha256(npz_path.read_bytes()).hexdigest()
        != manifest["npz_sha256"]
    ):
        raise ValueError("stack-val prediction shard bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != set(manifest["npz_fields"]):
            raise ValueError("stack-val prediction shard fields differ")
        arrays = {
            name: np.asarray(payload[name]) for name in payload.files
        }
    checked = _prediction_arrays(
        identities=arrays["identities"].tolist(),
        graph_id=manifest["graph_id"],
        pipeline_seed=manifest["pipeline_seed"],
        logits=arrays["logits"],
        probabilities=arrays["probabilities"],
    )
    if (
        not np.array_equal(arrays["graph_ids"], checked["graph_ids"])
        or not np.array_equal(
            arrays["pipeline_seeds"], checked["pipeline_seeds"]
        )
        or manifest["identity_order_sha256"]
        != canonical_sha256(checked["identities"].tolist())
        or manifest.get("contains_labels")
        or manifest.get("contains_offline_targets")
        or manifest.get("contains_oracle_tokens")
        or manifest.get("contains_oracle_logits")
        or manifest.get("final_test_consumed")
    ):
        raise ValueError("stack-val prediction manifest semantics differ")
    return manifest, checked


def _window_ranking(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary: str,
    window: float,
    tie_key: Any,
) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in rows]
    output = []
    while remaining:
        maximum = max(float(row[primary]) for row in remaining)
        group = [
            row
            for row in remaining
            if maximum - float(row[primary]) <= window
        ]
        group.sort(key=tie_key)
        output.extend(group)
        selected = {row["graph_id"] for row in group}
        remaining = [
            row for row in remaining if row["graph_id"] not in selected
        ]
    return output


def build_scale_finalist_bundle(
    *,
    locked_scale_shortlist: Mapping[str, Any],
    scale_completion: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
    final_select_label_manifest: Mapping[str, Any],
    lineage_hashes: Mapping[str, str],
    shape_assignments: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    shortlist_sha = validate_content_hash(
        locked_scale_shortlist, expected_contract=SCALE_SHORTLIST_CONTRACT
    )
    completion_sha = validate_content_hash(
        scale_completion, expected_contract=SCALE_COMPLETION_CONTRACT
    )
    label_sha = validate_content_hash(
        final_select_label_manifest,
        expected_contract=FINAL_SELECT_LABEL_MANIFEST_CONTRACT,
    )
    if (
        scale_completion.get("parents", {}).get(
            "locked_scale_shortlist"
        )
        != shortlist_sha
        or set(lineage_hashes) != set(FINALIST_LINEAGE_KEYS)
        or set(shape_assignments) != set(REQUIRED_SHAPE_ASSIGNMENTS)
        or lineage_hashes.get("locked_scale_shortlist") != shortlist_sha
        or lineage_hashes.get("scale_completion") != completion_sha
        or final_select_label_manifest.get("role")
        != "stage_n_selector_only"
        or final_select_label_manifest.get("feature_access_allowed")
        or final_select_label_manifest.get(
            "selection_inference_access_allowed"
        )
    ):
        raise ValueError("scale-finalist parent/access lineage differs")
    if (
        shape_assignments["SHAPE_BRIDGE"]
        != locked_scale_shortlist["SHAPE_BRIDGE"]
        or any(value in (None, "", {}, []) for value in shape_assignments.values())
    ):
        raise ValueError("scale-finalist shape assignments differ")
    label_rows = final_select_label_manifest["rows"]
    identities = [str(row["identity"]) for row in label_rows]
    labels = np.asarray(
        [int(row["label"]) for row in label_rows], dtype=np.int64
    )
    if (
        identities != sorted(identities)
        or len(identities) != len(set(identities))
        or len(set(np.bincount(labels, minlength=10).tolist())) != 1
    ):
        raise ValueError("final-select label manifest is not balanced/canonical")
    expected = {
        (graph_id, seed)
        for graph_id in locked_scale_shortlist["SCALE_SHORTLIST"]
        for seed in PIPELINE_SEEDS
    }
    run_map = {
        (row["graph_id"], row["pipeline_seed"]): row
        for row in scale_completion["runs"]
    }
    if len(prediction_records) != len(expected):
        raise ValueError("stack-val prediction coverage differs")
    metrics_artifacts = []
    seen = set()
    for record in prediction_records:
        manifest = record.get("manifest", {})
        key = (
            str(manifest.get("graph_id")),
            int(manifest.get("pipeline_seed", -1)),
        )
        if key not in expected or key in seen:
            raise ValueError("stack-val prediction graph/seed differs")
        seen.add(key)
        validate_content_hash(
            manifest,
            expected_contract=STACK_SELECTION_PREDICTION_CONTRACT,
        )
        arrays = _prediction_arrays(
            identities=record.get("identities", []),
            graph_id=key[0],
            pipeline_seed=key[1],
            logits=np.asarray(record.get("logits")),
            probabilities=np.asarray(record.get("probabilities")),
        )
        run = run_map[key]
        parents = manifest["parent_hashes"]
        if (
            manifest.get("source")
            != locked_scale_shortlist.get("source")
            or arrays["identities"].tolist() != identities
            or parents["locked_scale_shortlist"] != shortlist_sha
            or parents["scale_completion"] != completion_sha
            or parents["scale_graph_run"] != run[
                "scale_graph_run_sha256"
            ]
            or parents["deployable_export"]
            != run["deployable_export_sha256"]
            or manifest.get("contains_labels")
        ):
            raise ValueError("stack-val prediction lineage/identity differs")
        classification = bind_source(
            evaluate_classification(
                arrays["logits"].astype(np.float64),
                labels,
                split="stack_val",
            ),
            source_snapshot=source_snapshot,
        )
        metrics_artifacts.append(
            bind_source(
                with_content_hash(
                    {
                        "contract": STACK_SELECTION_METRICS_CONTRACT,
                        "schema_version": 1,
                        "graph_id": key[0],
                        "pipeline_seed": key[1],
                        "parents": {
                            "prediction_manifest": manifest[
                                "content_hash"
                            ],
                            "prediction_shard": manifest["npz_sha256"],
                            "final_select_label_manifest": label_sha,
                        },
                        "classification_metrics": classification,
                        "mean_log_Jeffreys_selection_rejection": (
                            mean_log_selection_rejection(classification)
                        ),
                        "label_join_performed_only_in_selector": True,
                        "prediction_shard_contains_labels": False,
                        "selection_eligible": True,
                        "final_test_consumed": False,
                    }
                ),
                source_snapshot=source_snapshot,
            )
        )
    if seen != expected:
        raise ValueError("stack-val predictions are incomplete")
    by_graph: dict[str, list[Mapping[str, Any]]] = {
        graph_id: [] for graph_id in locked_scale_shortlist["SCALE_SHORTLIST"]
    }
    for artifact in metrics_artifacts:
        by_graph[artifact["graph_id"]].append(artifact)
    graph_rows = []
    for graph_id, rows in by_graph.items():
        ordered = sorted(rows, key=lambda row: row["pipeline_seed"])
        if [row["pipeline_seed"] for row in ordered] != list(
            PIPELINE_SEEDS
        ):
            raise ValueError("scale-finalist metric seed coverage differs")
        if any(
            abs(
                row["classification_metrics"]["accuracy"]
                - row["classification_metrics"][
                    "macro_per_class_accuracy"
                ]
            )
            > 1.0e-15
            for row in ordered
        ):
            raise ValueError(
                "balanced stack-val accuracy differs from overall accuracy"
            )
        run_rows = [run_map[(graph_id, seed)] for seed in PIPELINE_SEEDS]
        capacities = [row["capacity"] for row in run_rows]
        if any(row != capacities[0] for row in capacities[1:]):
            raise ValueError("scale graph capacity drifts across seeds")
        graph_rows.append(
            {
                "graph_id": graph_id,
                "mean_accuracy": math.fsum(
                    row["classification_metrics"]["accuracy"]
                    for row in ordered
                )
                / 3.0,
                "mean_balanced_accuracy": math.fsum(
                    row["classification_metrics"][
                        "macro_per_class_accuracy"
                    ]
                    for row in ordered
                )
                / 3.0,
                "mean_cross_entropy": math.fsum(
                    row["classification_metrics"]["cross_entropy"]
                    for row in ordered
                )
                / 3.0,
                "mean_log_Jeffreys_selection_rejection": math.fsum(
                    row["mean_log_Jeffreys_selection_rejection"]
                    for row in ordered
                )
                / 3.0,
                "analytical_flops_batch1": capacities[0][
                    "analytical_flops_batch1"
                ],
                "parameter_count": capacities[0]["parameter_count"],
                "metric_artifact_sha256_by_seed": {
                    str(row["pipeline_seed"]): row["content_hash"]
                    for row in ordered
                },
                "prediction_manifest_sha256_by_seed": {
                    str(row["pipeline_seed"]): row["parents"][
                        "prediction_manifest"
                    ]
                    for row in ordered
                },
            }
        )
    accuracy_ranking = _window_ranking(
        graph_rows,
        primary="mean_balanced_accuracy",
        window=0.0001,
        tie_key=lambda row: (
            row["mean_cross_entropy"],
            row["analytical_flops_batch1"],
            row["parameter_count"],
            row["graph_id"],
        ),
    )
    rejection_ranking = _window_ranking(
        graph_rows,
        primary="mean_log_Jeffreys_selection_rejection",
        window=0.005,
        tie_key=lambda row: (
            -row["mean_balanced_accuracy"],
            row["mean_cross_entropy"],
            row["analytical_flops_batch1"],
            row["parameter_count"],
            row["graph_id"],
        ),
    )
    traces = {}
    for name, ranking, primary, window in (
        (
            "accuracy_selection_trace",
            accuracy_ranking,
            "mean_balanced_accuracy",
            0.0001,
        ),
        (
            "rejection_selection_trace",
            rejection_ranking,
            "mean_log_Jeffreys_selection_rejection",
            0.005,
        ),
    ):
        traces[name] = bind_source(
            with_content_hash(
                {
                    "contract": SCALE_FINALIST_TRACE_CONTRACT,
                    "schema_version": 1,
                    "selector": name,
                    "primary_metric": primary,
                    "window": window,
                    "population": "stack_val",
                    "ranking": [
                        {"rank": index + 1, **row}
                        for index, row in enumerate(ranking)
                    ],
                    "selected_graph_id": ranking[0]["graph_id"],
                    "scientific_underperformance_blocks_selection": False,
                    "final_test_consumed": False,
                }
            ),
            source_snapshot=source_snapshot,
        )
    accuracy_id = accuracy_ranking[0]["graph_id"]
    rejection_id = rejection_ranking[0]["graph_id"]
    finalist_ids = sorted({accuracy_id, rejection_id})
    prediction_manifests = {
        f"{record['manifest']['graph_id']}:{record['manifest']['pipeline_seed']}": {
            "manifest_sha256": record["manifest"]["content_hash"],
            "shard_sha256": record["manifest"]["npz_sha256"],
        }
        for record in prediction_records
    }
    lock = bind_source(
        with_content_hash(
            {
                "contract": LOCKED_SCALE_FINALISTS_CONTRACT,
                "schema_version": 1,
                "lineage_hashes": {
                    name: require_sha256(
                        value, name=f"lineage_hashes.{name}"
                    )
                    for name, value in sorted(lineage_hashes.items())
                },
                "final_select_label_manifest_sha256": label_sha,
                "shortlisted_graph_ids": list(
                    locked_scale_shortlist["SCALE_SHORTLIST"]
                ),
                "all_shortlisted_scale_runs": scale_completion["runs"],
                "prediction_artifacts": prediction_manifests,
                "selection_metric_artifact_hashes": sorted(
                    row["content_hash"] for row in metrics_artifacts
                ),
                "selection_trace_hashes": {
                    name: artifact["content_hash"]
                    for name, artifact in sorted(traces.items())
                },
                "ACCURACY_FINALIST": accuracy_id,
                "REJECTION_FINALIST": rejection_id,
                "finalist_graph_ids": finalist_ids,
                "locked_graph_definitions": {
                    graph_id: locked_scale_shortlist[
                        "locked_graph_definitions"
                    ][graph_id]
                    for graph_id in finalist_ids
                },
                "SHAPE_BRIDGE": locked_scale_shortlist["SHAPE_BRIDGE"],
                "shape_assignments": {
                    name: shape_assignments[name]
                    for name in sorted(shape_assignments)
                },
                "deterministic_selection_reasons": {
                    "ACCURACY_FINALIST": (
                        "maximum_mean_balanced_accuracy_with_0.0001_window"
                    ),
                    "REJECTION_FINALIST": (
                        "maximum_mean_18_term_log_Jeffreys_with_0.005_window"
                    ),
                },
                "same_graph_won_both": accuracy_id == rejection_id,
                "scientific_gain_flags": {
                    graph_id: {
                        "gain_positive": None,
                        "reason": (
                            "named_baseline_control_is_resolved_postlock"
                        ),
                    }
                    for graph_id in finalist_ids
                },
                "postlock_oracle_target_generation_authorized": True,
                "contains_postlock_oracle_target": False,
                "contains_final_test_model_output": False,
                "test_result_may_replace_finalist": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {
        "selection_metrics": {
            f"{row['graph_id']}:{row['pipeline_seed']}": row
            for row in metrics_artifacts
        },
        **traces,
        "locked_scale_finalists": lock,
    }


def validate_scale_finalist_bundle(
    bundle: Mapping[str, Any],
    *,
    locked_scale_shortlist: Mapping[str, Any],
    scale_completion: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
    final_select_label_manifest: Mapping[str, Any],
    shape_assignments: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> str:
    lock = bundle.get("locked_scale_finalists", {})
    rebuilt = build_scale_finalist_bundle(
        locked_scale_shortlist=locked_scale_shortlist,
        scale_completion=scale_completion,
        prediction_records=prediction_records,
        final_select_label_manifest=final_select_label_manifest,
        lineage_hashes=lock.get("lineage_hashes", {}),
        shape_assignments=shape_assignments,
        source_snapshot=source_snapshot,
    )
    if dict(bundle) != rebuilt:
        raise ValueError("scale-finalist bundle semantics differ")
    return validate_content_hash(
        lock, expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT
    )


def publish_scale_finalist_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(campaign_root)
    lock = bundle["locked_scale_finalists"]
    validate_content_hash(
        lock, expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT
    )
    publications = {}
    metrics_root = root / "evaluations" / "stack_val_selection_metrics"
    for key, artifact in bundle["selection_metrics"].items():
        publications[f"metric:{key}"] = write_immutable_json(
            metrics_root / f"{key.replace(':', '_seed')}.json", artifact
        )
    for name in (
        "accuracy_selection_trace",
        "rejection_selection_trace",
    ):
        publications[name] = write_immutable_json(
            root / "selection" / f"{name}.json", bundle[name]
        )
    publications["locked_scale_finalists"] = write_immutable_json(
        root / "selection" / "locked_scale_finalists.json", lock
    )
    return {
        "locked_scale_finalists_sha256": lock["content_hash"],
        "publications": publications,
    }


__all__ = [
    "FINALIST_LINEAGE_KEYS",
    "LOCKED_SCALE_FINALISTS_CONTRACT",
    "PREDICTION_PARENT_KEYS",
    "REQUIRED_SHAPE_ASSIGNMENTS",
    "SCALE_FINALIST_TRACE_CONTRACT",
    "STACK_SELECTION_METRICS_CONTRACT",
    "STACK_SELECTION_PREDICTION_CONTRACT",
    "build_scale_finalist_bundle",
    "build_stack_selection_prediction_manifest",
    "load_stack_selection_prediction",
    "publish_scale_finalist_bundle",
    "publish_stack_selection_prediction",
    "validate_scale_finalist_bundle",
]
