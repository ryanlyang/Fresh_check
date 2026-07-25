from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.select_prediction_anchored_bridge_consumer import (
    main as consumer_selection_main,
)
from teacher_logit_reco.local_particle_residual_field import (
    ALL50_TEACHER_NAMESPACE,
    BRIDGE_CONTROLS,
    N3_F0_TEACHER_NAMESPACE,
    PRIMARY_TEACHER_NAMESPACE,
    aggregate_consumer_evaluations,
    build_no_eligible_consumer_stop,
    build_campaign_registry,
    build_consumer_replica_evaluation,
    build_live_teacher_config,
    build_teacher_binding,
    cache_bound_teacher_logits,
    classification_metrics,
    evaluate_bound_consumer_conditions,
    finalize_consumer_confirmation,
    load_teacher_logit_cache,
    require_post_teacher_release,
    select_bridge_consumer_preconfirmation,
    validate_teacher_binding,
    verify_cached_direct_agreement,
    verify_equal_field_zero_kd,
    verify_teacher_identity_chain,
    write_teacher_logit_cache,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import with_content_hash
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import sha256_file
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    write_immutable_json,
)


CLASS_ORDER = tuple(f"class_{index}" for index in range(10))


def _predictions(labels: np.ndarray, correct_count: int) -> np.ndarray:
    prediction = (labels + 1) % len(CLASS_ORDER)
    prediction[:correct_count] = labels[:correct_count]
    logits = np.full((len(labels), len(CLASS_ORDER)), -3.0, dtype=np.float32)
    logits[np.arange(len(labels)), prediction] = 3.0
    return logits


def _conditions(labels: np.ndarray, endpoint_correct: int) -> dict[str, np.ndarray]:
    n = len(labels)
    conditions = {
        "f0": _predictions(labels, 140),
        "rho_0.000": _predictions(labels, 140),
        "rho_0.025": _predictions(labels, 145),
        "rho_0.050": _predictions(labels, 150),
        "rho_0.075": _predictions(labels, endpoint_correct - 5),
        "rho_0.100": _predictions(labels, endpoint_correct),
        "oracle_physical45": _predictions(labels, min(n, endpoint_correct + 15)),
        "oracle_all50": _predictions(labels, min(n, endpoint_correct + 20)),
        "reliability5_only": _predictions(labels, min(n, endpoint_correct + 3)),
        "zero_field_consumer_diagnostic": _predictions(labels, 100),
    }
    conditions.update({name: _predictions(labels, 140) for name in BRIDGE_CONTROLS})
    return conditions


def _evaluations(
    run_id: str,
    endpoint_counts=(158, 160, 162),
    *,
    median_checkpoint_sha256: str | None = None,
):
    labels = np.arange(200, dtype=np.int64) % 10
    event_ids = [f"event-{index}" for index in range(len(labels))]
    evaluations = []
    for seed, count in zip((101, 202, 303), endpoint_counts):
        evaluations.append(
            build_consumer_replica_evaluation(
                run_id=run_id,
                seed_id=seed,
                checkpoint_sha256=(
                    median_checkpoint_sha256
                    if seed == 202 and median_checkpoint_sha256 is not None
                    else f"{seed % 10}" * 64
                ),
                recipe_sha256="a" * 64,
                split_sha256="b" * 64,
                ram_audit_sha256="c" * 64,
                class_order=CLASS_ORDER,
                labels=labels,
                event_ids=event_ids,
                logits_by_condition=_conditions(labels, count),
                matched_compute_f0_accuracy=0.701,
                slices={"particle_multiplicity": ["low" if i < 100 else "high" for i in range(200)]},
                bootstrap_resamples=200,
            )
        )
    return evaluations


def _selection_and_aggregate(
    *,
    median_checkpoint_sha256: str | None = None,
    clean_checkpoint_path: str = "clean.pt",
):
    clean = aggregate_consumer_evaluations(
        _evaluations(
            "T10_clean", median_checkpoint_sha256=median_checkpoint_sha256
        ),
        bootstrap_resamples=300,
    )
    robust = aggregate_consumer_evaluations(
        _evaluations("T10_robust", (156, 158, 160)), bootstrap_resamples=300
    )
    selection = select_bridge_consumer_preconfirmation(
        [robust, clean],
        f0_checkpoint_sha256="d" * 64,
        bridge_recipe_sha256="a" * 64,
        selected_checkpoint_paths={"T10_clean": clean_checkpoint_path, "T10_robust": "robust.pt"},
    )
    return clean, robust, selection


def _confirmed_selection(
    *,
    median_checkpoint_sha256: str | None = None,
    clean_checkpoint_path: str = "clean.pt",
):
    clean, robust, pre = _selection_and_aggregate(
        median_checkpoint_sha256=median_checkpoint_sha256,
        clean_checkpoint_path=clean_checkpoint_path,
    )
    receipt = with_content_hash(
        {
            "contract": "prediction_anchored_split_access_receipt_v1",
            "status": "AUTHORIZED",
            "split_name": "stack_val_consumer",
            "purpose": "consumer_confirmation",
            "seal_kind": "consumer_preconfirmation",
            "one_shot": True,
            "selection_sha256": pre["content_hash"],
        }
    )
    final = finalize_consumer_confirmation(
        pre,
        {
            "checkpoint_sha256": pre["checkpoint_sha256"],
            "bridge_recipe_sha256": pre["bridge_recipe_sha256"],
            "f0_accuracy": 0.71,
            "bridge_0p10_accuracy": 0.79,
            "provenance_valid": True,
        },
        access_receipt=receipt,
    )
    return clean, pre, final, receipt


def test_ten_class_metrics_include_discrimination_calibration_and_confusion():
    labels = np.arange(100) % 10
    metrics = classification_metrics(
        _predictions(labels, 80), labels, class_order=CLASS_ORDER
    )
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert len(metrics["confusion_matrix"]) == 10
    assert set(metrics["one_vs_rest_auc"]) == set(CLASS_ORDER)
    assert metrics["expected_calibration_error"] >= 0
    assert metrics["brier_score"] >= 0


def test_same_checkpoint_runner_executes_every_condition_through_one_bound_forward(tmp_path):
    checkpoint = tmp_path / "consumer.pt"
    checkpoint.write_bytes(b"one exact consumer")
    checkpoint_hash = sha256_file(checkpoint)
    labels = np.arange(200, dtype=np.int64) % 10
    conditions = _conditions(labels, 160)

    class Forward:
        checkpoint_sha256 = checkpoint_hash

        def __init__(self):
            self.seen = []

        def __call__(self, batch, condition):
            self.seen.append(condition)
            return conditions[condition]

    forward = Forward()
    evaluation = evaluate_bound_consumer_conditions(
        run_id="T10_clean",
        seed_id=101,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        recipe_sha256="a" * 64,
        split_sha256="b" * 64,
        ram_audit_sha256="c" * 64,
        class_order=CLASS_ORDER,
        batches=[
            {
                "labels": labels,
                "event_ids": [f"event-{index}" for index in range(len(labels))],
            }
        ],
        forward_fn=forward,
        matched_compute_f0_accuracy=0.701,
        bootstrap_resamples=50,
    )
    assert evaluation.artifact["same_checkpoint_all_conditions"] is True
    assert set(forward.seen) == set(conditions)
    forward.checkpoint_sha256 = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        evaluate_bound_consumer_conditions(
            run_id="T10_clean",
            seed_id=101,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_hash,
            recipe_sha256="a" * 64,
            split_sha256="b" * 64,
            ram_audit_sha256="c" * 64,
            class_order=CLASS_ORDER,
            batches=[{"labels": labels, "event_ids": [f"event-{i}" for i in range(200)]}],
            forward_fn=forward,
            matched_compute_f0_accuracy=0.701,
            bootstrap_resamples=10,
        )


def test_section18_aggregate_is_paired_median_based_and_selector_is_order_independent():
    clean, robust, selection = _selection_and_aggregate()
    assert clean["eligible"] is True
    assert all(clean["validity_rules"].values())
    assert clean["median_seed_id"] == 202
    assert clean["best_seed_id"] == 303
    assert clean["best_seed_checkpoint_rejected"] is True
    reverse = select_bridge_consumer_preconfirmation(
        [clean, robust],
        f0_checkpoint_sha256="d" * 64,
        bridge_recipe_sha256="a" * 64,
        selected_checkpoint_paths={"T10_clean": "clean.pt", "T10_robust": "robust.pt"},
    )
    assert selection["selected_consumer_recipe"] == "T10_clean"
    assert reverse["selected_consumer_recipe"] == selection["selected_consumer_recipe"]
    assert selection["selected_median_seed_id"] == 202


def test_exact_tie_uses_fixed_secondary_order_and_lexicographic_recipe_id():
    clean = aggregate_consumer_evaluations(_evaluations("T10_clean"), bootstrap_resamples=100)
    robust = aggregate_consumer_evaluations(_evaluations("T10_robust"), bootstrap_resamples=100)
    selection = select_bridge_consumer_preconfirmation(
        [robust, clean],
        f0_checkpoint_sha256="d" * 64,
        bridge_recipe_sha256="a" * 64,
        selected_checkpoint_paths={"T10_clean": "clean.pt", "T10_robust": "robust.pt"},
    )
    assert selection["selection_reason"]["tie_pool"] == ["T10_clean", "T10_robust"]
    assert selection["selected_consumer_recipe"] == "T10_clean"


def test_gate_rejects_two_of_three_and_response_control_failures():
    mixed = aggregate_consumer_evaluations(
        _evaluations("T10_clean", (138, 141, 139)), bootstrap_resamples=100
    )
    assert mixed["eligible"] is False
    assert mixed["validity_rules"]["rule_4_replica_and_median_positive"] is False
    mixed_robust = aggregate_consumer_evaluations(
        _evaluations("T10_robust", (138, 141, 139)), bootstrap_resamples=100
    )
    with pytest.raises(RuntimeError, match="no bridge-consumer"):
        select_bridge_consumer_preconfirmation(
            [mixed, mixed_robust],
            f0_checkpoint_sha256="d" * 64,
            bridge_recipe_sha256="a" * 64,
        )
    stopped = build_no_eligible_consumer_stop([mixed, mixed_robust])
    assert stopped["status"] == "STOPPED_NO_ELIGIBLE_BRIDGE_CONSUMER"
    assert stopped["selected_consumer_id"] is None
    assert stopped["guessed_consumer_used"] is False
    assert stopped["stack_val_consumer_opened"] is False
    assert stopped["downstream_submission_allowed"] is False
    for recipe in ("T10_clean", "T10_robust"):
        summary = stopped["recipe_summaries"][recipe]
        assert summary["eligible"] is False
        assert summary["failed_rules"]
        assert len(summary["replica_effects"]) == 3


def test_selection_cli_publishes_diagnostic_stop_when_no_recipe_is_eligible(
    tmp_path, capsys
):
    clean = aggregate_consumer_evaluations(
        _evaluations("T10_clean", (138, 141, 139)), bootstrap_resamples=100
    )
    robust = aggregate_consumer_evaluations(
        _evaluations("T10_robust", (138, 141, 139)), bootstrap_resamples=100
    )
    clean_path = tmp_path / "clean.json"
    robust_path = tmp_path / "robust.json"
    write_immutable_json(clean_path, clean)
    write_immutable_json(robust_path, robust)
    selection_path = tmp_path / "selection" / "consumer_preconfirmation.json"

    status = consumer_selection_main(
        [
            "select",
            "--clean-aggregate",
            str(clean_path),
            "--robust-aggregate",
            str(robust_path),
            "--clean-checkpoint",
            str(tmp_path / "clean.pt"),
            "--robust-checkpoint",
            str(tmp_path / "robust.pt"),
            "--f0-checkpoint-sha256",
            "d" * 64,
            "--bridge-recipe-sha256",
            "a" * 64,
            "--output",
            str(selection_path),
        ]
    )

    assert status == 2
    assert not selection_path.exists()
    stopped_path = selection_path.parent / "stopped_campaign.json"
    stopped = json.loads(stopped_path.read_text(encoding="utf-8"))
    assert stopped["status"] == "STOPPED_NO_ELIGIBLE_BRIDGE_CONSUMER"
    assert stopped["downstream_submission_allowed"] is False
    printed = json.loads(capsys.readouterr().out)
    assert printed["selection"] is None
    assert printed["stopped_campaign"]["recipe_summaries"]["T10_clean"][
        "failed_rules"
    ]


def test_warn_and_continue_prefers_robust_and_records_every_failed_rule(
    tmp_path, capsys
):
    clean = aggregate_consumer_evaluations(
        _evaluations("T10_clean", (138, 141, 139)), bootstrap_resamples=100
    )
    robust = aggregate_consumer_evaluations(
        _evaluations("T10_robust", (138, 141, 139)), bootstrap_resamples=100
    )
    clean_path = tmp_path / "clean.json"
    robust_path = tmp_path / "robust.json"
    write_immutable_json(clean_path, clean)
    write_immutable_json(robust_path, robust)
    selection_path = tmp_path / "selection" / "consumer_preconfirmation.json"

    status = consumer_selection_main(
        [
            "select",
            "--clean-aggregate",
            str(clean_path),
            "--robust-aggregate",
            str(robust_path),
            "--clean-checkpoint",
            str(tmp_path / "clean.pt"),
            "--robust-checkpoint",
            str(tmp_path / "robust.pt"),
            "--f0-checkpoint-sha256",
            "d" * 64,
            "--bridge-recipe-sha256",
            "a" * 64,
            "--quality-gate-policy",
            "warn_and_continue",
            "--preferred-consumer",
            "T10_robust",
            "--output",
            str(selection_path),
        ]
    )

    assert status == 0
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["selected_consumer_recipe"] == "T10_robust"
    assert selection["aggregate_quality_eligible"] is False
    assert selection["quality_gate_override_applied"] is True
    assert selection["failed_quality_rules"]
    assert selection["downstream_submission_allowed"] is True
    assert not (selection_path.parent / "stopped_campaign.json").exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["selection"]["quality_gate_policy"] == "warn_and_continue"


def test_warn_and_continue_confirmation_keeps_quality_failure_as_warning(tmp_path):
    checkpoint = tmp_path / "robust.pt"
    checkpoint.write_bytes(b"quality-warning median")
    checkpoint_sha256 = sha256_file(checkpoint)
    clean = aggregate_consumer_evaluations(
        _evaluations("T10_clean", (138, 141, 139)), bootstrap_resamples=100
    )
    robust = aggregate_consumer_evaluations(
        _evaluations(
            "T10_robust",
            (138, 139, 141),
            median_checkpoint_sha256=checkpoint_sha256,
        ),
        bootstrap_resamples=100,
    )
    pre = select_bridge_consumer_preconfirmation(
        [clean, robust],
        f0_checkpoint_sha256="d" * 64,
        bridge_recipe_sha256="a" * 64,
        selected_checkpoint_paths={
            "T10_clean": "clean.pt",
            "T10_robust": str(checkpoint),
        },
        quality_gate_policy="warn_and_continue",
        preferred_consumer_recipe="T10_robust",
    )
    receipt = with_content_hash(
        {
            "contract": "prediction_anchored_split_access_receipt_v1",
            "status": "AUTHORIZED",
            "split_name": "stack_val_consumer",
            "purpose": "consumer_confirmation",
            "seal_kind": "consumer_preconfirmation",
            "one_shot": True,
            "selection_sha256": pre["content_hash"],
        }
    )
    selected = finalize_consumer_confirmation(
        pre,
        {
            "checkpoint_sha256": pre["checkpoint_sha256"],
            "bridge_recipe_sha256": pre["bridge_recipe_sha256"],
            "f0_accuracy": 0.75,
            "bridge_0p10_accuracy": 0.74,
            "provenance_valid": True,
        },
        access_receipt=receipt,
        output_dir=tmp_path,
    )
    assert selected["status"] == "CONFIRMED_LOCKED"
    confirmation = selected["stack_val_consumer_confirmation"]
    assert confirmation["quality_passed"] is False
    assert confirmation["integrity_passed"] is True
    assert confirmation["quality_gate_override_applied"] is True
    assert confirmation["execution_authorized"] is True
    assert selected["downstream_submission_allowed"] is True
    assert (tmp_path / "selected_bridge_consumer.json").is_file()
    assert not (tmp_path / "stopped_campaign.json").exists()
    binding = build_teacher_binding(
        binding_kind="primary",
        run_id="T10_robust",
        aggregate=robust,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        channel_policy="physical45",
        validation_manifest_hashes={
            "model_val_select": "e" * 64,
            "stack_val_consumer": "f" * 64,
        },
        target_cache_namespace=PRIMARY_TEACHER_NAMESPACE,
        bridge_recipe_sha256=selected["bridge_recipe_sha256"],
        primary_selection=selected,
    )
    assert binding["aggregate_quality_eligible"] is False
    assert binding["quality_gate_override_applied"] is True
    validate_teacher_binding(
        binding, expected_kind="primary", primary_selection=selected
    )
    release = require_post_teacher_release(
        build_campaign_registry(alternate_teacher_valid=True),
        selected_consumer=selected,
        primary_binding=binding,
    )
    assert release["teacher_gate_passed"] is True
    assert release["teacher_integrity_gate_passed"] is True
    assert release["teacher_quality_passed"] is False
    assert release["teacher_quality_warning_override_applied"] is True


def test_confirmation_is_one_shot_fail_closed_and_never_chooses_runner_up(tmp_path):
    _, _, pre = _selection_and_aggregate()
    receipt = with_content_hash(
        {
            "contract": "prediction_anchored_split_access_receipt_v1",
            "status": "AUTHORIZED",
            "split_name": "stack_val_consumer",
            "purpose": "consumer_confirmation",
            "seal_kind": "consumer_preconfirmation",
            "one_shot": True,
            "selection_sha256": pre["content_hash"],
        }
    )
    stopped = finalize_consumer_confirmation(
        pre,
        {
            "checkpoint_sha256": pre["checkpoint_sha256"],
            "bridge_recipe_sha256": pre["bridge_recipe_sha256"],
            "f0_accuracy": 0.75,
            "bridge_0p10_accuracy": 0.74,
            "provenance_valid": True,
        },
        access_receipt=receipt,
        output_dir=tmp_path,
    )
    assert stopped["status"] == "STOPPED_CONSUMER_CONFIRMATION_FAILED"
    assert stopped["runner_up_considered"] is False
    assert not (tmp_path / "selected_bridge_consumer.json").exists()
    assert json.loads((tmp_path / "stopped_campaign.json").read_text())["downstream_submission_allowed"] is False


def test_nonfinite_confirmation_writes_a_failure_artifact_not_a_selection(tmp_path):
    _, _, pre = _selection_and_aggregate()
    receipt = with_content_hash(
        {
            "contract": "prediction_anchored_split_access_receipt_v1",
            "status": "AUTHORIZED",
            "split_name": "stack_val_consumer",
            "purpose": "consumer_confirmation",
            "seal_kind": "consumer_preconfirmation",
            "one_shot": True,
            "selection_sha256": pre["content_hash"],
        }
    )
    stopped = finalize_consumer_confirmation(
        pre,
        {
            "checkpoint_sha256": pre["checkpoint_sha256"],
            "bridge_recipe_sha256": pre["bridge_recipe_sha256"],
            "f0_accuracy": float("nan"),
            "bridge_0p10_accuracy": 0.8,
            "provenance_valid": True,
        },
        access_receipt=receipt,
        output_dir=tmp_path,
    )
    assert stopped["stack_val_consumer_confirmation"]["all_metrics_finite"] is False
    assert stopped["stack_val_consumer_confirmation"]["delta_same"] is None
    assert (tmp_path / "stopped_campaign.json").is_file()


def test_primary_binding_cache_live_hashes_and_n3_namespace_are_exact(tmp_path):
    checkpoint_path = tmp_path / "selected.pt"
    checkpoint_path.write_bytes(b"exact ordered median checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    aggregate, _, selected, _ = _confirmed_selection(
        median_checkpoint_sha256=checkpoint_sha256,
        clean_checkpoint_path=str(checkpoint_path),
    )
    binding = build_teacher_binding(
        binding_kind="primary",
        run_id=selected["selected_consumer_recipe"],
        aggregate=aggregate,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=selected["checkpoint_sha256"],
        channel_policy="physical45",
        validation_manifest_hashes={"model_val_select": "e" * 64, "stack_val_consumer": "f" * 64},
        target_cache_namespace=PRIMARY_TEACHER_NAMESPACE,
        bridge_recipe_sha256=selected["bridge_recipe_sha256"],
        primary_selection=selected,
    )
    validate_teacher_binding(binding, expected_kind="primary", primary_selection=selected)
    live = build_live_teacher_config(binding, primary_selection=selected)
    logits = np.arange(300, dtype=np.float32).reshape(30, 10) / 100
    labels = np.arange(30, dtype=np.int64) % 10
    ids = [f"distill-{index}" for index in range(30)]
    manifest = write_teacher_logit_cache(
        binding=binding,
        logits=logits,
        labels=labels,
        event_ids=ids,
        stack_train_distill_manifest_sha256="1" * 64,
        class_order=CLASS_ORDER,
        temperature_convention="raw_logits_temperature_applied_in_kd_loss",
        output_root=tmp_path,
        live_teacher_config=live,
        primary_selection=selected,
    )
    assert manifest["checkpoint_sha256"] == manifest["live_checkpoint_sha256"] == selected["checkpoint_sha256"]
    assert manifest["teacher_binding_sha256"] == binding["content_hash"]
    loaded_manifest, arrays = load_teacher_logit_cache(
        tmp_path / PRIMARY_TEACHER_NAMESPACE,
        binding=binding,
        live_teacher_config=live,
        stack_train_distill_manifest_sha256="1" * 64,
        primary_selection=selected,
    )
    np.testing.assert_array_equal(arrays["logits"], logits)
    assert loaded_manifest["field_condition"] == "bridge_0.100"
    n3 = write_teacher_logit_cache(
        binding=binding,
        logits=logits,
        labels=labels,
        event_ids=ids,
        stack_train_distill_manifest_sha256="1" * 64,
        class_order=CLASS_ORDER,
        temperature_convention="raw_logits_temperature_applied_in_kd_loss",
        output_root=tmp_path,
        namespace=N3_F0_TEACHER_NAMESPACE,
        live_teacher_config=live,
        primary_selection=selected,
    )
    assert n3["field_condition"] == "f0"
    assert n3["cache_namespace"] != manifest["cache_namespace"]

    class BoundForward:
        def __init__(self, bound_hash):
            self.checkpoint_sha256 = bound_hash

        def __call__(self, batch, condition):
            assert condition == "bridge_0.100"
            return batch["direct_logits"]

    generated = cache_bound_teacher_logits(
        binding=binding,
        checkpoint_path=checkpoint_path,
        batches=[
            {
                "labels": labels[:15],
                "event_ids": ids[:15],
                "direct_logits": logits[:15],
            },
            {
                "labels": labels[15:],
                "event_ids": ids[15:],
                "direct_logits": logits[15:],
            },
        ],
        forward_fn=BoundForward(checkpoint_sha256),
        stack_train_distill_manifest_sha256="1" * 64,
        class_order=CLASS_ORDER,
        temperature_convention="raw_logits_temperature_applied_in_kd_loss",
        output_root=tmp_path / "bound_forward",
        live_teacher_config=live,
        primary_selection=selected,
    )
    assert generated["event_count"] == 30
    assert generated["same_checkpoint_target_and_live"] is True
    bundle = with_content_hash(
        {
            "contract": "prediction_anchored_test_bundle_config_v1",
            "teacher_checkpoint_sha256": checkpoint_sha256,
            "teacher_binding_sha256": binding["content_hash"],
            "privileged_paths_required_at_inference": False,
        }
    )
    chain = verify_teacher_identity_chain(
        binding=binding,
        cache_manifest=manifest,
        live_teacher_config=live,
        bundle_config=bundle,
        primary_selection=selected,
    )
    assert chain["same_checkpoint_selection_target_live_bundle"] is True
    wrong_bundle = with_content_hash(
        {
            "contract": "prediction_anchored_test_bundle_config_v1",
            "teacher_checkpoint_sha256": "0" * 64,
            "teacher_binding_sha256": binding["content_hash"],
        }
    )
    with pytest.raises(ValueError, match="bundle.teacher_checkpoint"):
        verify_teacher_identity_chain(
            binding=binding,
            cache_manifest=manifest,
            live_teacher_config=live,
            bundle_config=wrong_bundle,
            primary_selection=selected,
        )


def test_nonprimary_binding_cannot_substitute_primary_and_zero_kd_is_exact():
    aggregate, _, selected, _ = _confirmed_selection()
    with pytest.raises(ValueError, match="non-primary"):
        build_teacher_binding(
            binding_kind="alternate",
            run_id="T10_robust",
            aggregate=aggregate,
            checkpoint_path="alt.pt",
            checkpoint_sha256=aggregate["median_checkpoint_sha256"],
            channel_policy="physical45",
            validation_manifest_hashes={"model_val_select": "e" * 64},
            target_cache_namespace="physical45_alternate_bridge_teacher",
            bridge_recipe_sha256="a" * 64,
            primary_selection=selected,
        )
    logits = np.random.default_rng(5).normal(size=(8, 10)).astype(np.float32)
    assert verify_cached_direct_agreement(logits, logits.copy())["cached_direct_agreement"]
    zero = verify_equal_field_zero_kd(logits, logits.copy(), temperature=2.0)
    assert zero["kd_loss"] == 0.0
    with pytest.raises(AssertionError, match="not zero"):
        verify_equal_field_zero_kd(logits, logits + np.eye(8, 10, dtype=np.float32), tolerance=1e-12)


def test_all50_binding_has_distinct_lineage_and_rejects_primary_selection(tmp_path):
    checkpoint = tmp_path / "all50.pt"
    checkpoint.write_bytes(b"all50 median weights")
    checkpoint_hash = sha256_file(checkpoint)
    aggregate = aggregate_consumer_evaluations(
        _evaluations(
            "T10_all50_clean", median_checkpoint_sha256=checkpoint_hash
        ),
        bootstrap_resamples=100,
    )
    binding = build_teacher_binding(
        binding_kind="all50",
        run_id="T10_all50_clean",
        aggregate=aggregate,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        channel_policy="all50",
        validation_manifest_hashes={"model_val_select": "e" * 64},
        target_cache_namespace=ALL50_TEACHER_NAMESPACE,
        bridge_recipe_sha256="a" * 64,
    )
    assert binding["binding_kind"] == "all50"
    assert binding["target_cache_namespace"] == ALL50_TEACHER_NAMESPACE
    _, _, primary, _ = _confirmed_selection()
    with pytest.raises(ValueError, match="non-primary"):
        validate_teacher_binding(
            binding, expected_kind="all50", primary_selection=primary
        )
