from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (
    SCALE_SHORTLIST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    canonical_sha256,
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (
    CONTROL_KINDS,
    EXECUTION_PARENT_KEYS,
    FINAL_INPUT_KEYS,
    POSTLOCK_TARGET_PARENT_KEYS,
    build_final_test_execution_claim,
    build_final_test_execution_lock,
    build_finalist_controls,
    build_postlock_oracle_target,
    build_prelock_final_test_inputs,
    build_sealed_final_test_evaluation,
    validate_final_test_execution_lock,
    validate_finalist_controls,
    validate_postlock_oracle_target,
    validate_prelock_final_test_inputs,
)
from teacher_logit_reco.relation_expert_token_bridge import (
    paired_statistics as paired_statistics_module,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (
    SCALE_COMPONENT_KEYS,
    SCALE_REFIT_KEYS,
    aggregate_scale_completion,
    build_scale_graph_run,
    build_scale_refit_bundle,
    validate_scale_completion,
    validate_scale_graph_run,
    validate_scale_refit_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.splits import (
    FINAL_SELECT_LABEL_MANIFEST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (
    FINALIST_LINEAGE_KEYS,
    PREDICTION_PARENT_KEYS,
    build_scale_finalist_bundle,
    build_stack_selection_prediction_manifest,
    validate_scale_finalist_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_reporting import (
    build_stage_mn_report,
    publish_stage_mn_report,
)
from teacher_logit_reco.relation_expert_token_bridge.step14 import (
    build_step14_bundle,
    validate_step14_bundle,
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


def _shortlist() -> dict:
    definitions = {
        graph_id: {
            "graph_id": graph_id,
            "role": "scientific_candidate",
            "semantic_category": (
                "CONSTRAINED_ADAPTER"
                if graph_id == "g_accuracy"
                else "UNRESTRICTED_FUSION"
            ),
            "shortlist_eligible": True,
            "named_baseline_graph_id": f"base_{graph_id}",
            "shape_id": "S16_64",
            "complete_graph_definition_sha256": _digest(
                f"definition-{graph_id}"
            ),
            "training_recipe_sha256": _digest(f"training-{graph_id}"),
            "inference_recipe_sha256": _digest(f"inference-{graph_id}"),
            "deployable_without_offline_or_oracle": True,
            "predicts_tokens": True,
            "configuration": {"consumer": graph_id},
        }
        for graph_id in ("g_accuracy", "g_rejection")
    }
    return bind_source(
        with_content_hash(
            {
                "contract": SCALE_SHORTLIST_CONTRACT,
                "schema_version": 1,
                "SCALE_SHORTLIST": sorted(definitions),
                "locked_graph_definitions": definitions,
                "SHAPE_BRIDGE": {
                    "shape_id": "S16_64",
                    "K": 16,
                    "D": 64,
                    "scalars": 1024,
                },
            }
        ),
        source_snapshot=SOURCE,
    )


def _classification(split: str = "val_design") -> dict:
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    logits = np.full((len(labels), 10), -2.0)
    logits[np.arange(len(labels)), labels] = 2.0
    return bind_source(
        evaluate_classification(logits, labels, split=split),
        source_snapshot=SOURCE,
    )


def _refits(shortlist: dict, graph_id: str, seed: int) -> dict:
    rows = {}
    for name in SCALE_REFIT_KEYS:
        if name.startswith("shared_HLT_"):
            population = "shared_hlt_scale"
            identity = SHA_A
            replicas = [0, 1, 2, 3]
        elif name == "target_token":
            population = "scale_train_offline_targets"
            identity = SHA_A
            replicas = []
        elif name == "uncertainty_calibrator":
            population = "val_design_label_free"
            identity = SHA_B
            replicas = []
        else:
            population = "offline_scale"
            identity = SHA_A
            replicas = []
        rows[name] = {
            "artifact_sha256": _digest(f"refit-{graph_id}-{seed}-{name}"),
            "population": population,
            "identity_manifest_sha256": identity,
            "recipe_sha256": _digest(f"recipe-{name}"),
            "fitted_values_sha256": _digest(
                f"values-{graph_id}-{seed}-{name}"
            ),
            "labels_consumed": False,
            "replica_ids": replicas,
        }
    return bind_source(
        build_scale_refit_bundle(
            graph_id=graph_id,
            pipeline_seed=seed,
            locked_scale_shortlist_sha256=shortlist["content_hash"],
            scale_train_manifest_sha256=SHA_A,
            val_design_identity_manifest_sha256=SHA_B,
            refits=rows,
            five_hundred_k_artifact_hashes=[SHA_C],
        ),
        source_snapshot=SOURCE,
    )


def _scale_runs(shortlist: dict) -> list[dict]:
    runs = []
    for graph_id in shortlist["SCALE_SHORTLIST"]:
        for seed in (101, 202, 303):
            refit = _refits(shortlist, graph_id, seed)
            metrics = _classification()
            components = {
                name: _digest(f"{graph_id}-{seed}-{name}")
                for name in SCALE_COMPONENT_KEYS
            }
            components["pre_stack_val_confirmation_metrics"] = metrics[
                "content_hash"
            ]
            components["scale_target_token_normalizer"] = refit["refits"][
                "target_token"
            ]["artifact_sha256"]
            run = bind_source(
                build_scale_graph_run(
                    locked_scale_shortlist=shortlist,
                    graph_id=graph_id,
                    pipeline_seed=seed,
                    scale_refit_bundle=refit,
                    component_hashes=components,
                    selected_epoch=40,
                    val_stop_accuracy=0.7,
                    val_stop_cross_entropy=1.0,
                    analytical_flops_batch1=1000,
                    analytical_flops_batch128=100000,
                    parameter_count=2000,
                    pre_stack_confirmation_metrics=metrics,
                ),
                source_snapshot=SOURCE,
            )
            validate_scale_refit_bundle(refit)
            validate_scale_graph_run(
                run, locked_scale_shortlist=shortlist
            )
            runs.append(run)
    return runs


def _stack_population() -> tuple[list[str], np.ndarray]:
    labels = np.repeat(np.arange(10, dtype=np.int64), 10)
    identities = [f"jet-{index:04d}" for index in range(len(labels))]
    return identities, labels


def _selector_logits(kind: str, labels: np.ndarray) -> np.ndarray:
    logits = np.full((len(labels), 10), -4.0, dtype=np.float32)
    for index, label in enumerate(labels.tolist()):
        within = index % 10
        if label == 0:
            logits[index, 0] = 2.0
            if kind == "accuracy":
                logits[index, 1:] = 1.9
            continue
        if kind == "accuracy":
            if within < 4:
                logits[index, label] = 2.0
                logits[index, 0] = 1.0
            else:
                logits[index, label] = (
                    1.8 if within == 4 else 1.0 - 0.05 * within
                )
                logits[index, 0] = 2.0
        else:
            logits[index, label] = 2.0
            logits[index, 0] = 0.0
            if within >= 3:
                wrong = 1 + (label % 9)
                if wrong == label:
                    wrong = 1 + (wrong % 9)
                logits[index, wrong] = 3.0
    return logits


def _finalist_inputs():
    shortlist = _shortlist()
    runs = _scale_runs(shortlist)
    completion = bind_source(
        aggregate_scale_completion(
            locked_scale_shortlist=shortlist,
            scale_graph_runs=runs,
            step14_bundle_sha256=SHA_B,
            scale_train_manifest_sha256=SHA_A,
        ),
        source_snapshot=SOURCE,
    )
    identities, labels = _stack_population()
    label_manifest = bind_source(
        with_content_hash(
            {
                "contract": FINAL_SELECT_LABEL_MANIFEST_CONTRACT,
                "schema_version": 1,
                "source_manifest_sha256": SHA_A,
                "role": "stage_n_selector_only",
                "feature_access_allowed": False,
                "selection_inference_access_allowed": False,
                "ordering": "canonical_identity_then_label",
                "count": len(identities),
                "identity_hash": _digest("stack-identities"),
                "rows": [
                    {"identity": identity, "label": int(label)}
                    for identity, label in zip(
                        identities, labels, strict=True
                    )
                ],
            }
        ),
        source_snapshot=SOURCE,
    )
    run_map = {
        (row["graph_id"], row["pipeline_seed"]): row
        for row in completion["runs"]
    }
    records = []
    for graph_id in shortlist["SCALE_SHORTLIST"]:
        kind = "accuracy" if graph_id == "g_accuracy" else "rejection"
        for seed in (101, 202, 303):
            logits = _selector_logits(kind, labels)
            probabilities = (
                np.exp(logits - logits.max(axis=1, keepdims=True))
            )
            probabilities = (
                probabilities
                / probabilities.sum(axis=1, keepdims=True)
            ).astype(np.float32)
            run = run_map[(graph_id, seed)]
            parents = {
                name: _digest(f"parent-{name}") for name in PREDICTION_PARENT_KEYS
            }
            parents.update(
                {
                    "locked_scale_shortlist": shortlist["content_hash"],
                    "scale_completion": completion["content_hash"],
                    "scale_graph_run": run["scale_graph_run_sha256"],
                    "deployable_export": run["deployable_export_sha256"],
                }
            )
            manifest = bind_source(
                build_stack_selection_prediction_manifest(
                    identities=identities,
                    graph_id=graph_id,
                    pipeline_seed=seed,
                    logits=logits,
                    probabilities=probabilities,
                    npz_filename=f"{graph_id}_{seed}.npz",
                    npz_sha256=_digest(f"npz-{graph_id}-{seed}"),
                    parent_hashes=parents,
                ),
                source_snapshot=SOURCE,
            )
            records.append(
                {
                    "manifest": manifest,
                    "identities": np.asarray(identities),
                    "graph_ids": np.full(len(identities), graph_id),
                    "pipeline_seeds": np.full(
                        len(identities), seed, dtype=np.int64
                    ),
                    "logits": logits,
                    "probabilities": probabilities,
                }
            )
    lineage = {
        name: _digest(f"lineage-{name}") for name in FINALIST_LINEAGE_KEYS
    }
    lineage.update(
        {
            "locked_scale_shortlist": shortlist["content_hash"],
            "scale_completion": completion["content_hash"],
        }
    )
    shape_assignments = {
        "SHAPE_HIGH": {"shape_id": "S16_64", "K": 16, "D": 64},
        "SHAPE_COMPACT": {"shape_id": "S8_128", "K": 8, "D": 128},
        "SHAPE_BRIDGE": shortlist["SHAPE_BRIDGE"],
        "HET_PHYSICS": {"assignment_id": "HET_PHYSICS"},
        "HET_SELECTED": {"assignment_id": "HET_SELECTED"},
        "HET_BEAM": {"assignment_id": "HET_BEAM"},
    }
    bundle = build_scale_finalist_bundle(
        locked_scale_shortlist=shortlist,
        scale_completion=completion,
        prediction_records=records,
        final_select_label_manifest=label_manifest,
        lineage_hashes=lineage,
        shape_assignments=shape_assignments,
        source_snapshot=SOURCE,
    )
    return (
        shortlist,
        runs,
        completion,
        identities,
        labels,
        records,
        label_manifest,
        shape_assignments,
        bundle,
    )


def test_step14_policy_and_scale_completion_are_exact() -> None:
    bundle = build_step14_bundle(
        campaign_spec_sha256=SHA_A,
        step13_bundle_sha256=SHA_B,
        locked_scale_shortlist_sha256=SHA_C,
        global_determinism_sha256=_digest("determinism"),
        source_snapshot=SOURCE,
    )
    assert validate_step14_bundle(bundle) == bundle["step14_bundle"][
        "content_hash"
    ]
    shortlist = _shortlist()
    runs = _scale_runs(shortlist)
    completion = bind_source(
        aggregate_scale_completion(
            locked_scale_shortlist=shortlist,
            scale_graph_runs=runs,
            step14_bundle_sha256=SHA_B,
            scale_train_manifest_sha256=SHA_A,
        ),
        source_snapshot=SOURCE,
    )
    assert completion["expected_run_count"] == 6
    assert validate_scale_completion(
        completion,
        locked_scale_shortlist=shortlist,
        scale_graph_runs=runs,
    ) == completion["content_hash"]
    with pytest.raises(ValueError, match="row count differs"):
        aggregate_scale_completion(
            locked_scale_shortlist=shortlist,
            scale_graph_runs=runs[:-1],
            step14_bundle_sha256=SHA_B,
            scale_train_manifest_sha256=SHA_A,
        )


def test_scale_refits_reject_500k_substitution() -> None:
    shortlist = _shortlist()
    artifact = _refits(shortlist, "g_accuracy", 101)
    bad = {
        name: dict(row) for name, row in artifact["refits"].items()
    }
    bad["offline_input"]["artifact_sha256"] = SHA_C
    with pytest.raises(ValueError, match="scale-refit lineage differs"):
        build_scale_refit_bundle(
            graph_id="g_accuracy",
            pipeline_seed=101,
            locked_scale_shortlist_sha256=shortlist["content_hash"],
            scale_train_manifest_sha256=SHA_A,
            val_design_identity_manifest_sha256=SHA_B,
            refits=bad,
            five_hundred_k_artifact_hashes=[SHA_C],
        )


def test_label_free_prediction_rejects_probability_or_parent_drift() -> None:
    identities, labels = _stack_population()
    logits = _selector_logits("accuracy", labels)
    probability = np.full(logits.shape, 0.1, dtype=np.float32)
    parents = {
        name: _digest(f"prediction-{name}")
        for name in PREDICTION_PARENT_KEYS
    }
    with pytest.raises(ValueError, match="probabilities differ"):
        build_stack_selection_prediction_manifest(
            identities=identities,
            graph_id="g_accuracy",
            pipeline_seed=101,
            logits=logits,
            probabilities=probability,
            npz_filename="predictions.npz",
            npz_sha256=SHA_A,
            parent_hashes=parents,
        )
    bad_parents = {**parents, "offline_target_cache": SHA_B}
    bad_parents.pop("degradation_profile")
    with pytest.raises(ValueError, match="parent/filename differs"):
        build_stack_selection_prediction_manifest(
            identities=identities,
            graph_id="g_accuracy",
            pipeline_seed=101,
            logits=logits,
            probabilities=None,
            npz_filename="predictions.npz",
            npz_sha256=SHA_A,
            parent_hashes=bad_parents,
        )


def test_dual_stack_val_selectors_can_choose_different_graphs() -> None:
    (
        shortlist,
        _,
        completion,
        _,
        _,
        records,
        labels,
        shape_assignments,
        bundle,
    ) = _finalist_inputs()
    lock = bundle["locked_scale_finalists"]
    assert lock["ACCURACY_FINALIST"] == "g_accuracy"
    assert lock["REJECTION_FINALIST"] == "g_rejection"
    assert not lock["same_graph_won_both"]
    assert all(
        not record["manifest"]["contains_labels"] for record in records
    )
    assert validate_scale_finalist_bundle(
        bundle,
        locked_scale_shortlist=shortlist,
        scale_completion=completion,
        prediction_records=records,
        final_select_label_manifest=labels,
        shape_assignments=shape_assignments,
        source_snapshot=SOURCE,
    ) == lock["content_hash"]


def _controls(finalists: dict) -> dict:
    rows = []
    for graph_id in finalists["finalist_graph_ids"]:
        evaluation_rows = []
        baseline = f"base_{graph_id}"
        for kind in CONTROL_KINDS:
            for seed in (101, 202, 303):
                graph = (
                    graph_id
                    if kind == "FINALIST"
                    else (
                        baseline
                        if kind == "NAMED_BASELINE"
                        else f"{kind}::{graph_id}"
                    )
                )
                evaluation_rows.append(
                    {
                        "row_id": f"{graph_id}:{kind}:{seed}",
                        "owner_finalist_graph_id": graph_id,
                        "kind": kind,
                        "graph_id": graph,
                        "pipeline_seed": seed,
                        "checkpoint_sha256": _digest(
                            f"checkpoint-{graph_id}-{kind}-{seed}"
                        ),
                    }
                )
        rows.append(
            {
                "finalist_graph_id": graph_id,
                "named_baseline_graph_id": baseline,
                "complete_graph_capacity_sha256": _digest(
                    f"capacity-{graph_id}"
                ),
                "H_MONO_PARAM_control_sha256": _digest(
                    f"param-{graph_id}"
                ),
                "H_MONO_FLOP_control_sha256": _digest(
                    f"flop-{graph_id}"
                ),
                "H_BASE_LONG_control_sha256": _digest(
                    f"long-{graph_id}"
                ),
                "evaluation_rows": evaluation_rows,
            }
        )
    return bind_source(
        build_finalist_controls(
            locked_scale_finalists=finalists, rows=rows
        ),
        source_snapshot=SOURCE,
    )


def _targets(finalists: dict) -> list[dict]:
    run_map = {
        (row["graph_id"], row["pipeline_seed"]): row
        for row in finalists["all_shortlisted_scale_runs"]
    }
    rows = []
    target_identity_sha = canonical_sha256(_stack_population()[0])
    for graph_id in finalists["finalist_graph_ids"]:
        for seed in (101, 202, 303):
            run = run_map[(graph_id, seed)]
            for split in ("stack_val", "final_test"):
                parents = {
                    name: _digest(f"target-{name}")
                    for name in POSTLOCK_TARGET_PARENT_KEYS
                }
                parents.update(
                    {
                        "locked_scale_finalists": finalists[
                            "content_hash"
                        ],
                        "scale_graph_run": run[
                            "scale_graph_run_sha256"
                        ],
                        "scale_offline_experts": run["component_hashes"][
                            "offline_experts"
                        ],
                        "scale_offline_fusion": run["component_hashes"][
                            "offline_fusion"
                        ],
                        "scale_target_normalizer": run[
                            "component_hashes"
                        ]["scale_target_token_normalizer"],
                    }
                )
                rows.append(
                    bind_source(
                        build_postlock_oracle_target(
                            locked_scale_finalists=finalists,
                            graph_id=graph_id,
                            pipeline_seed=seed,
                            split=split,
                            parent_hashes=parents,
                            target_cache_manifest_sha256=_digest(
                                f"cache-{graph_id}-{seed}-{split}"
                            ),
                            target_identity_order_sha256=_digest(
                                f"identity-{split}"
                            )
                            if split == "stack_val"
                            else target_identity_sha,
                            target_dtype="float32_fallback",
                            float16_audit_sha256=_digest(
                                f"float16-audit-{graph_id}-{seed}-{split}"
                            ),
                            float16_audit_passed=False,
                        ),
                        source_snapshot=SOURCE,
                    )
                )
    return rows


def test_two_lock_sequence_and_sealed_final_evaluation(
    monkeypatch, tmp_path
) -> None:
    _, _, completion, identities, labels, _, _, _, bundle = _finalist_inputs()
    finalists = bundle["locked_scale_finalists"]
    controls = _controls(finalists)
    targets = _targets(finalists)
    assert validate_finalist_controls(
        controls, locked_scale_finalists=finalists
    ) == controls["content_hash"]
    assert all(
        validate_postlock_oracle_target(
            row, locked_scale_finalists=finalists
        )
        == row["content_hash"]
        for row in targets
    )
    input_hashes = {
        name: _digest(f"input-{name}") for name in FINAL_INPUT_KEYS
    }
    prelock = build_prelock_final_test_inputs(
        campaign_spec_sha256=SHA_A,
        split_manifest_sha256=SHA_B,
        degradation_profile_sha256=finalists["lineage_hashes"][
            "degradation_profile"
        ],
        input_hashes=input_hashes,
    )
    assert validate_prelock_final_test_inputs(prelock) == prelock[
        "content_hash"
    ]
    parents = {
        name: _digest(f"execution-{name}")
        for name in EXECUTION_PARENT_KEYS
    }
    parents.update(
        {
            "locked_scale_finalists": finalists["content_hash"],
            "finalist_controls": controls["content_hash"],
            "prelock_final_test_inputs": prelock["content_hash"],
        }
    )
    execution = bind_source(
        build_final_test_execution_lock(
            locked_scale_finalists=finalists,
            finalist_controls=controls,
            postlock_targets=targets,
            parent_hashes=parents,
            final_input_hashes=input_hashes,
        ),
        source_snapshot=SOURCE,
    )
    assert validate_final_test_execution_lock(
        execution,
        locked_scale_finalists=finalists,
        finalist_controls=controls,
        postlock_targets=targets,
    ) == execution["content_hash"]
    with pytest.raises(ValueError, match="postlock target coverage"):
        build_final_test_execution_lock(
            locked_scale_finalists=finalists,
            finalist_controls=controls,
            postlock_targets=targets[:-1],
            parent_hashes=parents,
            final_input_hashes=input_hashes,
        )
    bad_target_parents = dict(targets[0]["parent_hashes"])
    bad_target_parents["scale_offline_experts"] = _digest(
        "five-hundred-k-teacher"
    )
    with pytest.raises(ValueError, match="non-scale teacher"):
        build_postlock_oracle_target(
            locked_scale_finalists=finalists,
            graph_id=targets[0]["graph_id"],
            pipeline_seed=targets[0]["pipeline_seed"],
            split=targets[0]["split"],
            parent_hashes=bad_target_parents,
            target_cache_manifest_sha256=targets[0][
                "target_cache_manifest_sha256"
            ],
            target_identity_order_sha256=targets[0][
                "target_identity_order_sha256"
            ],
            target_dtype="float32_fallback",
            float16_audit_sha256=targets[0][
                "float16_round_trip_audit"
            ]["artifact_sha256"],
            float16_audit_passed=False,
        )
    monkeypatch.setattr(
        paired_statistics_module, "BOOTSTRAP_REPLICATES", 8
    )
    prediction_rows = []
    for row_id, locked in execution["eligible_evaluation_rows"].items():
        logits = _selector_logits("accuracy", labels)
        probability = (
            np.exp(logits - logits.max(axis=1, keepdims=True))
        )
        probability = (
            probability / probability.sum(axis=1, keepdims=True)
        ).astype(np.float32)
        prediction_rows.append(
            {
                "row_id": row_id,
                "graph_id": locked["graph_id"],
                "pipeline_seed": locked["pipeline_seed"],
                "checkpoint_sha256": locked["checkpoint_sha256"],
                "degradation_profile_sha256": execution[
                    "degradation_profile_sha256"
                ],
                "identities": identities,
                "logits": logits,
                "probabilities": probability,
                "prediction_artifact_sha256": _digest(
                    f"final-prediction-{row_id}"
                ),
            }
        )
    final = build_sealed_final_test_evaluation(
        execution_lock=execution,
        execution_claim=bind_source(
            build_final_test_execution_claim(
                execution_lock=execution,
                execution_plan_sha256=_digest("final-execution-plan"),
            ),
            source_snapshot=SOURCE,
        ),
        identities=identities,
        labels=labels,
        final_labels_artifact_sha256=next(
            row["target_cache_manifest_sha256"]
            for row in execution["postlock_targets"]
            if row["split"] == "final_test"
        ),
        prediction_rows=prediction_rows,
        source_snapshot=SOURCE,
    )
    assert final["all_and_only_locked_rows_evaluated"]
    assert not final["test_result_selected_replacement"]
    assert set(final["evaluated_row_ids"]) == set(
        execution["eligible_evaluation_rows"]
    )
    assert len(final["paired_between_distinct_finalists"]) == 3
    report, markdown = build_stage_mn_report(
        scale_completion=completion,
        locked_scale_finalists=finalists,
        execution_lock=execution,
        final_evaluation=final,
        source_snapshot=SOURCE,
    )
    assert "Accuracy finalist" in markdown
    publications = publish_stage_mn_report(
        output_dir=tmp_path, artifact=report, markdown=markdown
    )
    assert Path(publications["json"]["path"]).is_file()
    drifted = [dict(row) for row in prediction_rows]
    drifted[0]["checkpoint_sha256"] = _digest("wrong-checkpoint")
    with pytest.raises(ValueError, match="prediction semantics differ"):
        build_sealed_final_test_evaluation(
            execution_lock=execution,
            execution_claim=bind_source(
                build_final_test_execution_claim(
                    execution_lock=execution,
                    execution_plan_sha256=_digest(
                        "final-execution-plan"
                    ),
                ),
                source_snapshot=SOURCE,
            ),
            identities=identities,
            labels=labels,
            final_labels_artifact_sha256=next(
                row["target_cache_manifest_sha256"]
                for row in execution["postlock_targets"]
                if row["split"] == "final_test"
            ),
            prediction_rows=drifted,
            source_snapshot=SOURCE,
        )


def test_step14_entrypoints_exist_and_inference_worker_has_no_label_argument() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "scripts/build_retb_step14_contracts.py",
        "scripts/register_retb_scale_refits.py",
        "scripts/train_retb_scale_shortlist.py",
        "scripts/aggregate_retb_scale_completion.py",
        "scripts/infer_retb_scale_stack_val.py",
        "scripts/select_retb_scale_finalists.py",
        "scripts/prepare_retb_final_test_inputs.py",
        "scripts/build_retb_postlock_oracle_targets.py",
        "scripts/attest_retb_finalist_controls.py",
        "scripts/write_retb_final_test_execution_lock.py",
        "scripts/evaluate_retb_final_test.py",
        "scripts/write_retb_step14_report.py",
        "sbatch/run_retb_build_step14_contracts.sh",
        "sbatch/run_retb_register_scale_refits.sh",
        "sbatch/run_retb_train_scale_shortlist.sh",
        "sbatch/run_retb_scale_completion.sh",
        "sbatch/run_retb_infer_scale_stack_val.sh",
        "sbatch/run_retb_select_scale_finalists.sh",
        "sbatch/run_retb_prepare_final_inputs.sh",
        "sbatch/run_retb_postlock_targets.sh",
        "sbatch/run_retb_finalist_controls.sh",
        "sbatch/run_retb_final_execution_lock.sh",
        "sbatch/run_retb_final_test.sh",
        "sbatch/run_retb_step14_report.sh",
    )
    assert all((root / path).is_file() for path in paths)
    inference = (root / "scripts/infer_retb_scale_stack_val.py").read_text(
        "utf-8"
    )
    assert "label-manifest" not in inference
    assert "final_test" not in inference
    prelock = (
        root / "scripts/prepare_retb_final_test_inputs.py"
    ).read_text("utf-8")
    assert "--checkpoint" not in prelock
