from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (
    REQUIRED_CONFIRMATION_CATEGORIES,
    SEED_COMPONENT_KEYS,
    SHORTLIST_PARENT_KEYS,
    aggregate_500k_confirmation,
    build_seed_confirmation,
    build_shortlisted_500k_controls,
    build_stage_l_graph_registry,
    mean_log_selection_rejection,
    select_bridge_shape,
    select_scale_shortlist,
    validate_500k_confirmation,
    validate_bridge_shape_selection,
    validate_scale_shortlist,
    validate_seed_confirmation,
    validate_stage_l_graph_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_l_reporting import (
    build_stage_l_report,
    publish_stage_l_report,
)
from teacher_logit_reco.relation_expert_token_bridge import (
    paired_statistics as paired_statistics_module,
)
from teacher_logit_reco.relation_expert_token_bridge.paired_statistics import (
    BOOTSTRAP_REPLICATES,
    PAIRED_STATISTICS_CONTRACT,
    build_paired_confirmation_statistics,
    validate_paired_confirmation_statistics,
)
from teacher_logit_reco.relation_expert_token_bridge.step13 import (
    FAILURE_INTERPRETATIONS,
    build_step13_bundle,
    validate_step13_bundle,
)


SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metrics(
    *, accuracy: float, cross_entropy: float, rejection: float
) -> dict:
    labels = np.repeat(np.arange(10, dtype=np.int64), 4)
    logits = np.full((len(labels), 10), -1.0, dtype=np.float64)
    logits[np.arange(len(labels)), labels] = 1.0
    base = evaluate_classification(logits, labels, split="val_design")
    payload = dict(base)
    payload.pop("content_hash")
    payload["accuracy"] = float(accuracy)
    payload["macro_per_class_accuracy"] = float(accuracy)
    payload["cross_entropy"] = float(cross_entropy)
    payload["per_class_efficiency"] = {
        name: float(accuracy) for name in base["class_order"]
    }
    updated = {}
    for signal, targets in payload["qcd_signal_rejection"].items():
        updated[signal] = {}
        for target, row in targets.items():
            updated[signal][target] = {
                **row,
                "finite_selection_rejection": float(rejection),
            }
    payload["qcd_signal_rejection"] = updated
    return with_content_hash(payload)


def _paired_statistics(
    *,
    graph_id: str,
    baseline_graph_id: str,
    pipeline_seed: int,
    prediction_sha256: str,
    baseline_prediction_sha256: str,
    candidate_accuracy: float,
    baseline_accuracy: float,
    candidate_mean_log_rejection: float,
    baseline_mean_log_rejection: float,
) -> dict:
    per_signal = [
        {
            "signal_class": signal,
            "target_signal_efficiency": target,
            "candidate_log_rejection_interval_95": [1.0, 2.0],
            "candidate_rejection_interval_95": [
                float(np.exp(1.0)),
                float(np.exp(2.0)),
            ],
            "baseline_log_rejection_interval_95": [0.5, 1.5],
            "baseline_rejection_interval_95": [
                float(np.exp(0.5)),
                float(np.exp(1.5)),
            ],
        }
        for signal in (
            "Hbb",
            "Hcc",
            "Hgg",
            "H4q",
            "Hqql",
            "Zqq",
            "Wqq",
            "Tbqq",
            "Tbl",
        )
        for target in (0.30, 0.50)
    ]
    return bind_source(
        with_content_hash(
        {
            "contract": PAIRED_STATISTICS_CONTRACT,
            "schema_version": 1,
            "candidate_graph_id": graph_id,
            "baseline_graph_id": baseline_graph_id,
            "pipeline_seed": pipeline_seed,
            "identity_count": 40,
            "identity_order_sha256": _digest("identity-order"),
            "identity_order": "lexicographic_canonical_jet_identity",
            "class_order": [
                "QCD",
                "Hbb",
                "Hcc",
                "Hgg",
                "H4q",
                "Hqql",
                "Zqq",
                "Wqq",
                "Tbqq",
                "Tbl",
            ],
            "balanced_count_per_class": 4,
            "parents": {
                "candidate_prediction": prediction_sha256,
                "baseline_prediction": baseline_prediction_sha256,
            },
            "central": {
                "candidate_accuracy": candidate_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "paired_accuracy_difference": (
                    candidate_accuracy - baseline_accuracy
                ),
                "relative_error_reduction": (
                    None
                    if baseline_accuracy == 1.0
                    else (
                        candidate_accuracy - baseline_accuracy
                    )
                    / (1.0 - baseline_accuracy)
                ),
                "McNemar_candidate_correct_baseline_wrong": 5,
                "McNemar_candidate_wrong_baseline_correct": 1,
                "per_class_accuracy_difference": {
                    name: 0.1
                    for name in (
                        "QCD",
                        "Hbb",
                        "Hcc",
                        "Hgg",
                        "H4q",
                        "Hqql",
                        "Zqq",
                        "Wqq",
                        "Tbqq",
                        "Tbl",
                    )
                },
                "candidate_mean_log_rejection": (
                    candidate_mean_log_rejection
                ),
                "baseline_mean_log_rejection": (
                    baseline_mean_log_rejection
                ),
                "mean_log_rejection_difference": (
                    candidate_mean_log_rejection
                    - baseline_mean_log_rejection
                ),
            },
            "bootstrap": {
                "seed": 917301,
                "bit_generator": "numpy.PCG64",
                "replicates": BOOTSTRAP_REPLICATES,
                "paired_sampling_unit": "canonical_jet_identity",
                "stratification": "true_class",
                "class_counts_preserved": True,
                "sampling": "with_replacement",
                "quantile_percent": [2.5, 97.5],
                "quantile_method": "linear",
                "paired_accuracy_difference_interval_95": [0.05, 0.15],
                "candidate_mean_log_rejection_interval_95": [1.5, 2.5],
                "baseline_mean_log_rejection_interval_95": [0.5, 1.5],
                "paired_mean_log_rejection_difference_interval_95": [
                    0.5,
                    1.5,
                ],
                "thresholds_recomputed_in_every_resample": True,
                "all_18_terms_recomputed_in_every_resample": True,
            },
            "per_signal_rejection": per_signal,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
        ),
        source_snapshot=SOURCE,
    )


def _shape_rows() -> list[dict]:
    rows = []
    for shape_id, K, D, gain in (
        ("S8_128", 8, 128, 0.015),
        ("S16_64", 16, 64, 0.020),
    ):
        for seed_index, seed in enumerate((101, 202, 303)):
            native = 0.60 + 0.001 * seed_index
            rows.append(
                {
                    "shape_id": shape_id,
                    "pipeline_seed": seed,
                    "K": K,
                    "D": D,
                    "split": "val_design",
                    "pipeline_lineage_kind": "PRIMARY_MATCHED_SEED",
                    "all_predicted_accuracy": native + gain,
                    "shape_matched_HF_NATIVE_accuracy": native,
                    "frozen_fusion_cross_entropy": (
                        0.80 if shape_id == "S8_128" else 0.82
                    ),
                    "normalized_token_error": (
                        0.20 if shape_id == "S8_128" else 0.18
                    ),
                    "prediction_artifact_sha256": _digest(
                        f"prediction-{shape_id}-{seed}"
                    ),
                    "native_metrics_artifact_sha256": _digest(
                        f"native-{shape_id}-{seed}"
                    ),
                    "token_metrics_artifact_sha256": _digest(
                        f"token-{shape_id}-{seed}"
                    ),
                    "label_manifest_sha256": SHA_A,
                    "stack_val_consumed": False,
                    "final_test_consumed": False,
                }
            )
    return rows


def _definitions() -> list[dict]:
    categories = [
        ("g_base", "PRIMARY_BASELINE", "reference_baseline"),
        (
            "g_uniform_compact",
            "UNIFORM_FINALIST",
            "scientific_candidate",
        ),
        (
            "g_uniform_high",
            "UNIFORM_FINALIST",
            "scientific_candidate",
        ),
        (
            "g_hetero_physics",
            "HETEROGENEOUS_FINALIST",
            "scientific_candidate",
        ),
        (
            "g_hetero_selected",
            "HETEROGENEOUS_FINALIST",
            "scientific_candidate",
        ),
        (
            "g_hetero_beam",
            "HETEROGENEOUS_FINALIST",
            "scientific_candidate",
        ),
        ("g_native", "NATIVE_HLT_FUSION", "scientific_candidate"),
        ("g_frozen", "FROZEN_RECONSTRUCTION", "scientific_candidate"),
        ("g_refiner", "TOKEN_REFINER", "scientific_candidate"),
        ("g_adapter", "CONSTRAINED_ADAPTER", "scientific_candidate"),
        ("g_unrestricted", "UNRESTRICTED_FUSION", "scientific_candidate"),
    ]
    return [
        {
            "graph_id": graph_id,
            "role": role,
            "semantic_category": category,
            "shortlist_eligible": role == "scientific_candidate",
            "named_baseline_graph_id": "g_base",
            "shape_id": {
                "g_uniform_compact": "S8_128",
                "g_uniform_high": "S16_64",
                "g_hetero_physics": "HET_PHYSICS",
                "g_hetero_selected": "HET_SELECTED",
                "g_hetero_beam": "HET_BEAM",
            }.get(graph_id, "S16_64"),
            "complete_graph_definition_sha256": _digest(
                f"definition-{graph_id}"
            ),
            "training_recipe_sha256": _digest(f"training-{graph_id}"),
            "inference_recipe_sha256": _digest(f"inference-{graph_id}"),
            "deployable_without_offline_or_oracle": True,
            "predicts_tokens": graph_id != "g_base",
            "configuration": {
                "consumer": graph_id,
                **(
                    {
                        "carried_shape_role": {
                            "g_uniform_compact": "SHAPE_COMPACT",
                            "g_uniform_high": "SHAPE_HIGH",
                            "g_hetero_physics": "HET_PHYSICS",
                            "g_hetero_selected": "HET_SELECTED",
                            "g_hetero_beam": "HET_BEAM",
                        }[graph_id]
                    }
                    if graph_id
                    in {
                        "g_uniform_compact",
                        "g_uniform_high",
                        "g_hetero_physics",
                        "g_hetero_selected",
                        "g_hetero_beam",
                    }
                    else {}
                ),
            },
        }
        for graph_id, category, role in categories
    ]


def _bound_stage_l_inputs():
    shape = bind_source(
        select_bridge_shape(
            rows=_shape_rows(),
            compact_shape_id="S8_128",
            high_shape_id="S16_64",
            step12_bundle_sha256=SHA_B,
            val_design_label_manifest_sha256=SHA_A,
        ),
        source_snapshot=SOURCE,
    )
    registry = bind_source(
        build_stage_l_graph_registry(
            definitions=_definitions(),
            step12_bundle_sha256=SHA_B,
            candidate_shape_ids=shape["candidate_shape_ids"],
            robustness_controls_completion_sha256=_digest(
                "robustness-controls-completion"
            ),
            semantic_controls_completion_sha256=_digest(
                "semantic-controls-completion"
            ),
        ),
        source_snapshot=SOURCE,
    )
    return shape, registry


def _seed_rows(
    registry: dict,
    *,
    all_candidates_worse: bool = False,
    omit: tuple[str, int] | None = None,
) -> list[dict]:
    definitions = {
        row["graph_id"]: row for row in registry["definitions"]
    }
    accuracy_order = {
        "g_base": 0.90 if all_candidates_worse else 0.70,
        "g_uniform_compact": 0.86 if all_candidates_worse else 0.79,
        "g_uniform_high": 0.85 if all_candidates_worse else 0.80,
        "g_hetero_physics": 0.84 if all_candidates_worse else 0.72,
        "g_hetero_selected": 0.83 if all_candidates_worse else 0.78,
        "g_hetero_beam": 0.82 if all_candidates_worse else 0.71,
        "g_native": 0.84 if all_candidates_worse else 0.77,
        "g_frozen": 0.83 if all_candidates_worse else 0.73,
        "g_refiner": 0.82 if all_candidates_worse else 0.74,
        "g_adapter": 0.81 if all_candidates_worse else 0.75,
        "g_unrestricted": 0.80 if all_candidates_worse else 0.76,
    }
    rejection_order = {
        "g_base": 4.0,
        "g_uniform_compact": 5.0,
        "g_uniform_high": 5.5,
        "g_hetero_physics": 6.0,
        "g_hetero_selected": 6.5,
        "g_hetero_beam": 6.25,
        "g_native": 7.0,
        "g_frozen": 20.0,
        "g_refiner": 30.0,
        "g_adapter": 50.0,
        "g_unrestricted": 40.0,
    }
    rows = []
    for graph_id, definition in definitions.items():
        for seed_index, seed in enumerate((101, 202, 303)):
            if omit == (graph_id, seed):
                continue
            metrics = bind_source(
                _metrics(
                    accuracy=accuracy_order[graph_id]
                    + 0.0001 * seed_index,
                    cross_entropy=1.0
                    - 0.01 * accuracy_order[graph_id],
                    rejection=rejection_order[graph_id],
                ),
                source_snapshot=SOURCE,
            )
            component_hashes = {
                name: _digest(f"{graph_id}-{seed}-{name}")
                for name in SEED_COMPONENT_KEYS
            }
            paired = _paired_statistics(
                graph_id=graph_id,
                baseline_graph_id=definition[
                    "named_baseline_graph_id"
                ],
                pipeline_seed=seed,
                prediction_sha256=component_hashes[
                    "prediction_manifest"
                ],
                baseline_prediction_sha256=_digest(
                    f"{definition['named_baseline_graph_id']}-{seed}-"
                    "prediction_manifest"
                ),
                candidate_accuracy=(
                    accuracy_order[graph_id] + 0.0001 * seed_index
                ),
                baseline_accuracy=(
                    accuracy_order[definition["named_baseline_graph_id"]]
                    + 0.0001 * seed_index
                ),
                candidate_mean_log_rejection=float(
                    np.log(rejection_order[graph_id])
                ),
                baseline_mean_log_rejection=float(
                    np.log(
                        rejection_order[
                            definition["named_baseline_graph_id"]
                        ]
                    )
                ),
            )
            component_hashes["paired_statistics"] = paired[
                "content_hash"
            ]
            artifact = bind_source(
                build_seed_confirmation(
                    graph_definition=definition,
                    pipeline_seed=seed,
                    classification_metrics=metrics,
                    normalized_token_error=(
                        None
                        if not definition["predicts_tokens"]
                        else 0.2
                    ),
                    analytical_flops_batch1=1000,
                    parameter_count=2000,
                    component_hashes=component_hashes,
                    paired_statistics=paired,
                    val_design_label_manifest_sha256=SHA_A,
                ),
                source_snapshot=SOURCE,
            )
            validate_seed_confirmation(
                artifact, graph_definition=definition
            )
            rows.append(artifact)
    return rows


def _parents(
    *, registry: dict, confirmation: dict, shape: dict
) -> dict[str, str]:
    values = {name: _digest(name) for name in SHORTLIST_PARENT_KEYS}
    values.update(
        {
            "campaign_spec": SHA_C,
            "step12_bundle": registry["step12_bundle_sha256"],
            "step13_bundle": _digest("step13"),
            "graph_registry": registry["content_hash"],
            "confirmation_summary": confirmation["content_hash"],
            "bridge_shape_selection": shape["content_hash"],
            "val_design_label_manifest": SHA_A,
        }
    )
    return values


def test_step13_bundle_and_bridge_shape_are_deterministic() -> None:
    bundle = build_step13_bundle(
        campaign_spec_sha256=SHA_A,
        step12_bundle_sha256=SHA_B,
        global_determinism_sha256=SHA_C,
        source_snapshot=SOURCE,
    )
    assert validate_step13_bundle(bundle) == bundle["step13_bundle"][
        "content_hash"
    ]
    assert bundle["stage_l_policy"]["failure_interpretations"] == [
        dict(row) for row in FAILURE_INTERPRETATIONS
    ]
    shape, _ = _bound_stage_l_inputs()
    assert validate_bridge_shape_selection(shape) == shape["content_hash"]
    assert shape["SHAPE_BRIDGE"]["shape_id"] == "S16_64"


def test_bridge_shape_deduplicates_equal_compact_and_high() -> None:
    rows = [
        row for row in _shape_rows() if row["shape_id"] == "S8_128"
    ]
    shape = select_bridge_shape(
        rows=rows,
        compact_shape_id="S8_128",
        high_shape_id="S8_128",
        step12_bundle_sha256=SHA_B,
        val_design_label_manifest_sha256=SHA_A,
    )
    assert shape["candidate_shape_ids"] == ["S8_128"]
    assert len(shape["ranking"]) == 1
    assert validate_bridge_shape_selection(shape) == shape["content_hash"]
    definitions = [
        {
            **row,
            "configuration": dict(row["configuration"]),
        }
        for row in _definitions()
        if row["graph_id"] != "g_uniform_high"
    ]
    compact = next(
        row
        for row in definitions
        if row["graph_id"] == "g_uniform_compact"
    )
    compact["configuration"]["carried_shape_role"] = (
        "SHAPE_COMPACT_AND_HIGH"
    )
    registry = build_stage_l_graph_registry(
        definitions=definitions,
        step12_bundle_sha256=SHA_B,
        candidate_shape_ids=shape["candidate_shape_ids"],
        robustness_controls_completion_sha256=_digest(
            "robustness-controls-completion"
        ),
        semantic_controls_completion_sha256=_digest(
            "semantic-controls-completion"
        ),
    )
    assert validate_stage_l_graph_registry(registry) == registry[
        "content_hash"
    ]
    assert "bridge_shape_selection" not in registry
    assert "bridge_shape_selection_sha256" not in registry
    assert registry[
        "all_predeclared_candidate_shapes_registered_before_selection"
    ]
    assert registry["bridge_shape_selected_after_confirmation"]


def test_graph_registry_requires_every_confirmation_category() -> None:
    shape, registry = _bound_stage_l_inputs()
    assert validate_stage_l_graph_registry(registry) == registry[
        "content_hash"
    ]
    assert {
        row["semantic_category"] for row in registry["definitions"]
    } == set(REQUIRED_CONFIRMATION_CATEGORIES)
    assert {
        row["configuration"]["carried_shape_role"]
        for row in registry["definitions"]
        if row["semantic_category"] == "UNIFORM_FINALIST"
    } == {"SHAPE_COMPACT", "SHAPE_HIGH"}
    assert {
        row["configuration"]["carried_shape_role"]
        for row in registry["definitions"]
        if row["semantic_category"] == "HETEROGENEOUS_FINALIST"
    } == {"HET_PHYSICS", "HET_SELECTED", "HET_BEAM"}
    with pytest.raises(ValueError, match="coverage differs"):
        build_stage_l_graph_registry(
            definitions=_definitions()[:-1],
            step12_bundle_sha256=SHA_B,
            candidate_shape_ids=shape["candidate_shape_ids"],
            robustness_controls_completion_sha256=_digest(
                "robustness-controls-completion"
            ),
            semantic_controls_completion_sha256=_digest(
                "semantic-controls-completion"
            ),
        )


def test_confirmation_rejects_incomplete_matched_seed_coverage() -> None:
    _, registry = _bound_stage_l_inputs()
    rows = _seed_rows(registry, omit=("g_frozen", 303))
    with pytest.raises(
        ValueError, match="complete matched-seed coverage"
    ):
        aggregate_500k_confirmation(
            graph_registry=registry,
            seed_confirmations=rows,
            val_design_label_manifest_sha256=SHA_A,
        )


def test_confirmation_rejects_paired_metric_or_baseline_drift() -> None:
    _, registry = _bound_stage_l_inputs()
    rows = _seed_rows(registry)
    target_index = next(
        index
        for index, row in enumerate(rows)
        if row["graph_id"] == "g_adapter"
        and row["pipeline_seed"] == 101
    )
    target = dict(rows[target_index])
    paired = dict(target["paired_statistics"])
    paired.pop("content_hash")
    paired["central"] = {
        **paired["central"],
        "candidate_accuracy": (
            paired["central"]["candidate_accuracy"] + 0.01
        ),
    }
    paired = with_content_hash(paired)
    target.pop("content_hash")
    target["paired_statistics"] = paired
    target["component_hashes"] = {
        **target["component_hashes"],
        "paired_statistics": paired["content_hash"],
    }
    rows[target_index] = with_content_hash(target)
    with pytest.raises(ValueError, match="paired statistics differ"):
        aggregate_500k_confirmation(
            graph_registry=registry,
            seed_confirmations=rows,
            val_design_label_manifest_sha256=SHA_A,
        )


def test_scale_shortlist_is_duplicate_free_top3_union() -> None:
    shape, registry = _bound_stage_l_inputs()
    rows = _seed_rows(registry)
    confirmation = bind_source(
        aggregate_500k_confirmation(
            graph_registry=registry,
            seed_confirmations=rows,
            val_design_label_manifest_sha256=SHA_A,
        ),
        source_snapshot=SOURCE,
    )
    shortlist = bind_source(
        select_scale_shortlist(
            confirmation_summary=confirmation,
            graph_registry=registry,
            bridge_shape_selection=shape,
            parent_hashes=_parents(
                registry=registry,
                confirmation=confirmation,
                shape=shape,
            ),
        ),
        source_snapshot=SOURCE,
    )
    assert shortlist["ACC_SCALE_TOP3"] == [
        "g_uniform_high",
        "g_uniform_compact",
        "g_hetero_selected",
    ]
    assert shortlist["REJ_SCALE_TOP3"] == [
        "g_adapter",
        "g_unrestricted",
        "g_refiner",
    ]
    assert shortlist["SCALE_SHORTLIST"] == sorted(
        {
            *shortlist["ACC_SCALE_TOP3"],
            *shortlist["REJ_SCALE_TOP3"],
        }
    )
    assert shortlist["shortlist_size"] == 6
    assert validate_scale_shortlist(
        shortlist,
        confirmation_summary=confirmation,
        graph_registry=registry,
        bridge_shape_selection=shape,
    ) == shortlist["content_hash"]


def test_all_negative_campaign_still_locks_and_reports(tmp_path) -> None:
    shape, registry = _bound_stage_l_inputs()
    rows = _seed_rows(registry, all_candidates_worse=True)
    confirmation = bind_source(
        aggregate_500k_confirmation(
            graph_registry=registry,
            seed_confirmations=rows,
            val_design_label_manifest_sha256=SHA_A,
        ),
        source_snapshot=SOURCE,
    )
    assert confirmation["all_candidates_worse_than_baseline"]
    shortlist = bind_source(
        select_scale_shortlist(
            confirmation_summary=confirmation,
            graph_registry=registry,
            bridge_shape_selection=shape,
            parent_hashes=_parents(
                registry=registry,
                confirmation=confirmation,
                shape=shape,
            ),
        ),
        source_snapshot=SOURCE,
    )
    assert 1 <= shortlist["shortlist_size"] <= 6
    controls = bind_source(
        build_shortlisted_500k_controls(
            locked_scale_shortlist=shortlist,
            rows=[
                {
                    "graph_id": graph_id,
                    "complete_graph_capacity_sha256": _digest(
                        f"capacity-{graph_id}"
                    ),
                    "monolithic_parameter_control_sha256": _digest(
                        f"param-{graph_id}"
                    ),
                    "monolithic_flop_control_sha256": _digest(
                        f"flop-{graph_id}"
                    ),
                    "H_BASE_LONG_label_exposure_control_sha256": _digest(
                        f"exposure-{graph_id}"
                    ),
                    "control_metrics_artifact_sha256": _digest(
                        f"controls-{graph_id}"
                    ),
                    "capacity_control_reproduces_gain": False,
                }
                for graph_id in shortlist["SCALE_SHORTLIST"]
            ],
        ),
        source_snapshot=SOURCE,
    )
    report, markdown = build_stage_l_report(
        confirmation=confirmation,
        shortlist=shortlist,
        shortlisted_controls=controls,
        step13_bundle_sha256=_digest("step13"),
        source_snapshot=SOURCE,
    )
    assert "NO_MODEL_IMPROVES" in markdown
    publications = publish_stage_l_report(
        output_dir=tmp_path, artifact=report, markdown=markdown
    )
    assert Path(publications["json"]["path"]).is_file()
    assert Path(publications["markdown"]["path"]).is_file()


def test_mean_log_rejection_uses_all_18_jeffreys_terms() -> None:
    metrics = _metrics(accuracy=0.7, cross_entropy=1.0, rejection=25.0)
    assert mean_log_selection_rejection(metrics) == pytest.approx(
        np.log(25.0)
    )


def test_paired_bootstrap_is_identity_paired_and_recomputes_rejection(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        paired_statistics_module, "BOOTSTRAP_REPLICATES", 16
    )
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    identities = [f"jet-{index:03d}" for index in range(len(labels))]
    baseline = np.full((len(labels), 10), -1.0, dtype=np.float64)
    candidate = baseline.copy()
    baseline[np.arange(len(labels)), labels] = 1.0
    candidate[np.arange(len(labels)), labels] = 1.2
    kwargs = {
        "identities": identities,
        "labels": labels,
        "candidate_logits": candidate,
        "baseline_logits": baseline,
        "candidate_graph_id": "candidate",
        "baseline_graph_id": "baseline",
        "pipeline_seed": 101,
        "candidate_prediction_sha256": SHA_A,
        "baseline_prediction_sha256": SHA_B,
    }
    first = build_paired_confirmation_statistics(**kwargs)
    second = build_paired_confirmation_statistics(**kwargs)
    assert first == second
    assert validate_paired_confirmation_statistics(first) == first[
        "content_hash"
    ]
    assert first["bootstrap"]["paired_sampling_unit"] == (
        "canonical_jet_identity"
    )
    assert first["bootstrap"][
        "thresholds_recomputed_in_every_resample"
    ]
    assert first["bootstrap"]["all_18_terms_recomputed_in_every_resample"]
    assert len(first["per_signal_rejection"]) == 18


def test_step13_entrypoints_exist_and_do_not_name_stack_val() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "scripts/build_retb_step13_contracts.py",
        "scripts/select_retb_bridge_shape.py",
        "scripts/register_retb_stage_l_graphs.py",
        "scripts/register_retb_500k_seed_confirmation.py",
        "scripts/aggregate_retb_confirmation.py",
        "scripts/select_retb_scale_shortlist.py",
        "scripts/write_retb_report.py",
        "scripts/compute_retb_paired_statistics.py",
        "scripts/attest_retb_shortlisted_500k_controls.py",
        "sbatch/run_retb_build_step13_contracts.sh",
        "sbatch/run_retb_select_bridge_shape.sh",
        "sbatch/run_retb_register_stage_l_graphs.sh",
        "sbatch/run_retb_register_500k_seed_confirmation.sh",
        "sbatch/run_retb_confirm.sh",
        "sbatch/run_retb_scale_shortlist.sh",
        "sbatch/run_retb_report.sh",
        "sbatch/run_retb_paired_statistics.sh",
        "sbatch/run_retb_shortlisted_500k_controls.sh",
    )
    assert all((root / path).is_file() for path in paths)
    for path in paths:
        text = (root / path).read_text("utf-8")
        assert "final_test" not in text
