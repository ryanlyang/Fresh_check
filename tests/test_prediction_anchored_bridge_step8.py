from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from tests.test_prediction_anchored_bridge_step6 import _fixture, _tensor_inputs
from tests.test_prediction_anchored_bridge_step7 import _absolute, _verified_reference
from tests.test_prediction_anchored_bridge_step4 import _confirmed_selection
from teacher_logit_reco.local_particle_residual_field import (
    A3_INTERACTION_RUN_IDS,
    A3_PRIMARY_ALIAS,
    ALL50_RUN_IDS,
    ARCH_A3_HLG_PRIMARY,
    NEGATIVE_CONTROL_RUN_IDS,
    PAIRED_SEED_IDS,
    PERTURBATION_AUDIT_SEEDS,
    STEP8_SPECIAL_CANONICAL_RUN_IDS,
    STEP3_RUN_IDS,
    PredictionAnchoredAll50HLG,
    Step8FixedStorage,
    Step8TrainConfig,
    all50_group_balanced_huber,
    apply_small_field_perturbation,
    bridge_distribution_distance,
    build_adversarial_channel_report,
    build_bridge_quantile_reference,
    build_campaign_registry,
    build_live_teacher_config,
    build_matched_wrong_event_map,
    build_teacher_binding,
    compute_step8_objective,
    compute_gain_and_recovery,
    correction_bridge_alignment,
    evaluate_reliability_only_response,
    fit_bridge_scalers,
    measure_step8_registry_states,
    prepare_step8_control_batch,
    record_step3_registry_measurements,
    require_production_ready,
    require_post_teacher_release,
    resolve_registry_run,
    resolve_step8_run_recipe,
    run_small_field_perturbation_audit,
    run_step8_paired_seed_miniature,
    step8_run_recipes,
    train_step8_replica,
    validate_step8_teacher_lineage,
    validate_four_control_matching,
    validate_step8_registry_semantics,
    verify_all50_equal_field_semantics,
    write_teacher_logit_cache,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import sha256_file, with_content_hash


CLASS_ORDER = tuple(f"class_{index}" for index in range(10))


def _all50_fixture(n=3, p=5):
    batch, physical = _fixture(n=n, p=p)
    rng = np.random.default_rng(808)
    true = batch["f0"] + rng.normal(scale=0.25, size=batch["f0"].shape).astype(np.float32)
    true[~batch["mask"]] = 0
    full_bridge = batch["f0"] + np.float32(0.1) * (true - batch["f0"])
    full_bridge[~batch["mask"]] = 0
    all50 = fit_bridge_scalers(
        [(batch["f0"], true, batch["mask"])],
        parent_hashes={"source": "a" * 64, "r0": "b" * 64},
        channel_policy="all50",
    ).to_artifact()
    batch["all50_bridge_fields"] = full_bridge
    return batch, physical, all50


def test_step8_recipe_matrix_coefficients_warmups_namespaces_and_alias_are_exact():
    recipes = step8_run_recipes()
    assert len(A3_INTERACTION_RUN_IDS) == 8
    assert len(STEP8_SPECIAL_CANONICAL_RUN_IDS) == 14
    assert recipes[A3_PRIMARY_ALIAS].canonical_run_id == ARCH_A3_HLG_PRIMARY
    assert recipes[A3_PRIMARY_ALIAS].to_artifact()["distillation_coefficients"] == {
        "kd": 1.0, "ce": 0.5, "bridge": 0.2, "true": 0.0,
        "anchor": 0.02, "smooth": 0.01, "gate": 0.0,
    }
    assert recipes["D10_XA3_bridge_only"].phase_coefficients("field_warmup")["bridge"] == 1.0
    assert not recipes["D10_XA3_ce_only"].field_warmup
    assert not recipes["D10_XA3_kd_only"].field_warmup
    assert not recipes["D10_XA3_kd_ce"].field_warmup
    assert not recipes["D10_XA3_full_no_warmup"].field_warmup
    assert recipes["D10_XA3_full_no_smooth"].smooth == 0
    assert recipes[ALL50_RUN_IDS[0]].binding_kind == "all50"
    assert recipes[ALL50_RUN_IDS[0]].channel_policy == "all50"
    assert recipes["D10_TALT_A3"].binding_kind == "alternate"
    assert recipes[NEGATIVE_CONTROL_RUN_IDS[3]].cache_namespace == "physical45_selected_teacher_on_f0_control"
    assert all(not recipes[value].selectable_for_primary_deployment for value in (*ALL50_RUN_IDS, "D10_TALT_A3", *NEGATIVE_CONTROL_RUN_IDS))
    with pytest.raises(ValueError, match="no field warm-up"):
        recipes["D10_XA3_kd_only"].phase_coefficients("field_warmup")
    semantics = validate_step8_registry_semantics(build_campaign_registry())
    assert semantics["registry_counts"] == {
        "configuration_count": 54,
        "reconstruction_breadth_count": 46,
        "post_teacher_configuration_count": 45,
    }
    assert semantics["canonical_semantic_rows"]["D10_TALT_A3"]["execution_status"] == "SKIPPED_INVALID_PARENT"


def test_all50_b1_wraps_exact_a3_adds_only_160_64_5_head_and_b2_cannot_change_reliability():
    batch, physical, all50 = _all50_fixture(n=2, p=4)
    f0 = torch.as_tensor(batch["f0"])
    b1 = PredictionAnchoredAll50HLG(
        ALL50_RUN_IDS[0], physical45_scaler_artifact=physical,
        all50_scaler_artifact=all50, dropout=0.0,
    )
    b2 = PredictionAnchoredAll50HLG(
        ALL50_RUN_IDS[1], physical45_scaler_artifact=physical,
        all50_scaler_artifact=all50, dropout=0.0,
    )
    assert b1.config_artifact()["head"] == [160, 64, 5]
    assert b1.config_artifact()["canonical_a3_model_config"] == b2.config_artifact()["canonical_a3_model_config"]
    first = b1(*_tensor_inputs(batch))
    second = b2(*_tensor_inputs(batch))
    assert torch.equal(first.f_hat, f0)
    assert torch.equal(second.f_hat[..., 45:], f0[..., 45:])
    final = b1.reliability_head[-1]
    torch.nn.init.constant_(final.bias, 0.2)
    changed = b1(*_tensor_inputs(batch))
    assert torch.count_nonzero(changed.reliability_correction[changed.mask]) > 0
    assert torch.equal(changed.base_output.f_hat[..., 45:], f0[..., 45:])
    assert changed.diagnostics["reliability_thresholding_applied"] is False


def test_all50_full_loss_has_thirteen_groups_and_b1_b2_equal_field_semantics_are_explicit():
    batch, physical, all50 = _all50_fixture(n=2, p=4)
    b1 = PredictionAnchoredAll50HLG(
        ALL50_RUN_IDS[0], physical45_scaler_artifact=physical,
        all50_scaler_artifact=all50, dropout=0.0,
    )
    output = b1(*_tensor_inputs(batch))
    target = torch.as_tensor(batch["all50_bridge_fields"])
    loss, groups = all50_group_balanced_huber(
        output.f_hat, target, output.mask, b1.base.scalers, b1.all50_scalers
    )
    assert torch.isfinite(loss)
    assert len(groups) == 13
    warmup_batch = {**batch, "bridge_fields": batch["all50_bridge_fields"]}
    objective, diagnostics = compute_step8_objective(
        output, warmup_batch, resolve_step8_run_recipe(ALL50_RUN_IDS[0]),
        phase="field_warmup", physical_scalers=b1.base.scalers,
        all50_scalers=b1.all50_scalers,
    )
    assert torch.isfinite(objective)
    assert "diagnostic.reliability5" in diagnostics["bridge_groups"]
    logits = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    b1_audit = verify_all50_equal_field_semantics(
        ALL50_RUN_IDS[0], target_logits=logits, live_logits=logits.clone(),
        target_fields=target, live_fields=target.clone(),
    )
    assert b1_audit["equal_field_zero_kd_applicable"] is True
    b2_live = target.clone()
    b2_live[..., 45:] = torch.as_tensor(batch["f0"])[..., 45:]
    b2_audit = verify_all50_equal_field_semantics(
        ALL50_RUN_IDS[1], target_logits=logits, live_logits=logits + 1,
        target_fields=target, live_fields=b2_live,
    )
    assert b2_audit["equal_field_zero_kd_applicable"] is False
    assert b2_audit["reliability_mismatch_max_abs"] > 0


def test_four_controls_use_exact_wrong_map_preserve_group_scale_and_match_positive_budgets():
    batch, _, _ = _all50_fixture(n=12, p=4)
    batch["labels"] = np.repeat(np.arange(3), 4).astype(np.int64)
    event_ids = [f"evt-{index}" for index in range(12)]
    wrong = build_matched_wrong_event_map(
        tokens=batch["hlt_tokens"], mask=batch["mask"], labels=batch["labels"],
        event_ids=event_ids, seed=71,
    )
    targets = np.arange(120, dtype=np.float32).reshape(12, 10)
    n0, a0 = prepare_step8_control_batch(
        NEGATIVE_CONTROL_RUN_IDS[0], batch, target_logits=targets,
        wrong_event_map=wrong, target_cache_namespace="physical45_selected_bridge_teacher",
    )
    permutation = np.asarray(wrong["permutation"])
    np.testing.assert_array_equal(n0["target_logits"], targets[permutation])
    n1, a1 = prepare_step8_control_batch(
        NEGATIVE_CONTROL_RUN_IDS[1], batch, target_logits=None,
        wrong_event_map=wrong, target_cache_namespace=None,
    )
    original = batch["bridge_fields"][..., :45] - batch["f0"][..., :45]
    shuffled = n1["bridge_fields"][..., :45] - batch["f0"][..., :45]
    for indices in __import__(
        "teacher_logit_reco.local_particle_residual_field", fromlist=["physical_loss_groups"]
    ).physical_loss_groups().values():
        assert np.linalg.norm(shuffled[..., indices][batch["mask"]]) == pytest.approx(
            np.linalg.norm(original[..., indices][batch["mask"]]), rel=2e-6
        )
    n2, a2 = prepare_step8_control_batch(
        NEGATIVE_CONTROL_RUN_IDS[2], batch, target_logits=targets,
        wrong_event_map=wrong, target_cache_namespace="physical45_selected_bridge_teacher",
    )
    assert a2["same_wrong_event_map_for_logits_and_bridge"] is True
    n3, a3 = prepare_step8_control_batch(
        NEGATIVE_CONTROL_RUN_IDS[3], batch, target_logits=targets,
        wrong_event_map=None,
        target_cache_namespace="physical45_selected_teacher_on_f0_control",
    )
    np.testing.assert_array_equal(n3["target_logits"], targets)
    assert all(not artifact["persistent_dense_fields_written"] for artifact in (a0, a1, a2, a3))
    ids = (*NEGATIVE_CONTROL_RUN_IDS, "D10_XA3_kd_only", "D10_XA3_bridge_only", ARCH_A3_HLG_PRIMARY)
    matching = validate_four_control_matching(
        model_config_sha256={name: "a" * 64 for name in ids},
        optimizer_steps={name: 100 for name in ids},
        paired_seed_ids={name: PAIRED_SEED_IDS for name in ids},
    )
    assert matching["all_four_matched"] is True


def test_step8_training_loop_executes_exact_phases_freezes_consumer_and_rejects_zero_kd_cache():
    batch, physical = _fixture(n=2, p=4)
    batch["labels"] = np.asarray([0, 1], dtype=np.int64)

    class TinyConsumer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(10))

        def forward(self, fields, mask):
            valid = mask.float().unsqueeze(-1)
            pooled = (fields[..., :10] * valid).sum(1) / valid.sum(1).clamp_min(1)
            return pooled @ self.weight

    def consumer_forward(consumer, typed, fields):
        return consumer(fields, typed["mask"])

    model = __import__(
        "teacher_logit_reco.local_particle_residual_field", fromlist=["build_step7_hlg_correction_model"]
    ).build_step7_hlg_correction_model(ARCH_A3_HLG_PRIMARY, scaler_artifact=physical, dropout=0.0)
    config = Step8TrainConfig(
        run_id=A3_PRIMARY_ALIAS, paired_seed_id=101,
        stack_train_distill_manifest_sha256="1" * 64,
        model_val_stop_manifest_sha256="2" * 64,
        distillation_steps=1, field_warmup_steps=1,
    )
    target_logits = np.ones((2, 10), dtype=np.float32)
    train_batch = {**batch, "target_logits": target_logits}
    consumer = TinyConsumer()
    result = train_step8_replica(
        model, [deepcopy(train_batch), deepcopy(train_batch)], config,
        live_consumer=consumer, consumer_forward_fn=consumer_forward,
    )
    assert [row["phase"] for row in result["phase_rows"]] == ["field_warmup", "distillation"]
    assert result["optimizer_steps_completed"] == 2
    assert result["teacher_parameters_frozen"] is True
    assert all(not parameter.requires_grad for parameter in consumer.parameters())
    ce_model = __import__(
        "teacher_logit_reco.local_particle_residual_field", fromlist=["build_step7_hlg_correction_model"]
    ).build_step7_hlg_correction_model(ARCH_A3_HLG_PRIMARY, scaler_artifact=physical, dropout=0.0)
    ce_config = Step8TrainConfig(
        run_id="D10_XA3_ce_only", paired_seed_id=202,
        stack_train_distill_manifest_sha256="1" * 64,
        model_val_stop_manifest_sha256="2" * 64,
        distillation_steps=1, field_warmup_steps=0,
    )
    with pytest.raises(ValueError, match="zero KD"):
        train_step8_replica(
            ce_model, [deepcopy(train_batch)], ce_config,
            live_consumer=TinyConsumer(), consumer_forward_fn=consumer_forward,
        )


def test_primary_n3_and_all50_lineage_are_immutable_namespace_isolated_and_release_ignores_c0(tmp_path):
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"selected exact median")
    checkpoint_hash = sha256_file(checkpoint)
    aggregate, _, selected, _ = _confirmed_selection(
        median_checkpoint_sha256=checkpoint_hash,
        clean_checkpoint_path=str(checkpoint),
    )
    primary = build_teacher_binding(
        binding_kind="primary", run_id=selected["selected_consumer_recipe"],
        aggregate=aggregate, checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_hash, channel_policy="physical45",
        validation_manifest_hashes={"model_val_select": "e" * 64, "stack_val_consumer": "f" * 64},
        target_cache_namespace="physical45_selected_bridge_teacher",
        bridge_recipe_sha256=selected["bridge_recipe_sha256"],
        primary_selection=selected,
    )
    live = build_live_teacher_config(primary, primary_selection=selected)
    logits = np.arange(40, dtype=np.float32).reshape(4, 10)
    labels = np.arange(4, dtype=np.int64)
    ids = [f"distill-{index}" for index in range(4)]
    n3 = write_teacher_logit_cache(
        binding=primary, logits=logits, labels=labels, event_ids=ids,
        stack_train_distill_manifest_sha256="1" * 64, class_order=CLASS_ORDER,
        temperature_convention="raw_logits_tau_in_loss", output_root=tmp_path / "primary",
        namespace="physical45_selected_teacher_on_f0_control",
        live_teacher_config=live, primary_selection=selected,
    )
    lineage = validate_step8_teacher_lineage(
        resolve_step8_run_recipe(NEGATIVE_CONTROL_RUN_IDS[3]),
        binding=primary, cache_manifest=n3, live_teacher_config=live,
        primary_selection=selected,
    )
    assert lineage["target_and_live_checkpoint_identical"] is True
    assert lineage["cache_namespace"] == "physical45_selected_teacher_on_f0_control"
    release = require_post_teacher_release(
        build_campaign_registry(), selected_consumer=selected, primary_binding=primary
    )
    assert release["teacher_gate_passed"] is True
    assert release["c0_success_consulted"] is False
    assert release["registered_post_teacher_configuration_count"] == 45

    batch, _, all50_scaler = _all50_fixture(n=2, p=4)
    all50_checkpoint = tmp_path / "all50.pt"
    all50_checkpoint.write_bytes(b"all50 exact median")
    all50_hash = sha256_file(all50_checkpoint)
    all50_aggregate = with_content_hash(
        {
            "contract": "test_all50_aggregate_v1",
            "median_seed_id": 202,
            "median_checkpoint_sha256": all50_hash,
        }
    )
    all50_binding = build_teacher_binding(
        binding_kind="all50", run_id="T10_all50_clean", aggregate=all50_aggregate,
        checkpoint_path=str(all50_checkpoint), checkpoint_sha256=all50_hash,
        channel_policy="all50", validation_manifest_hashes={"model_val_select": "e" * 64},
        target_cache_namespace="all50_selected_bridge_teacher",
        bridge_recipe_sha256="b" * 64,
        all50_scaler_artifact=all50_scaler,
    )
    assert all50_binding["all50_correction_scaler_sha256"] == all50_scaler["content_hash"]
    assert len(all50_binding["all50_extra_correction_statistics"]["trust_scale"]) == 5
    all50_live = build_live_teacher_config(all50_binding)
    all50_cache = write_teacher_logit_cache(
        binding=all50_binding, logits=logits, labels=labels, event_ids=ids,
        stack_train_distill_manifest_sha256="1" * 64, class_order=CLASS_ORDER,
        temperature_convention="raw_logits_tau_in_loss", output_root=tmp_path / "all50",
        live_teacher_config=all50_live,
    )
    all50_lineage = validate_step8_teacher_lineage(
        resolve_step8_run_recipe(ALL50_RUN_IDS[0]), binding=all50_binding,
        cache_manifest=all50_cache, live_teacher_config=all50_live,
        all50_scaler_artifact=all50_scaler,
    )
    assert all50_lineage["all50_correction_scaler_sha256"] == all50_scaler["content_hash"]
    with pytest.raises(ValueError, match="not the one embedded"):
        wrong = dict(all50_scaler)
        wrong.pop("content_hash")
        wrong["trust_scale"] = list(wrong["trust_scale"])
        wrong["trust_scale"][49] = float(wrong["trust_scale"][49]) + 1.0e-6
        wrong = with_content_hash(wrong)
        validate_step8_teacher_lineage(
            resolve_step8_run_recipe(ALL50_RUN_IDS[0]), binding=all50_binding,
            cache_manifest=all50_cache, live_teacher_config=all50_live,
            all50_scaler_artifact=wrong,
        )


def test_perturbation_is_exact_four_seed_keyed_order_invariant_and_signed_thresholded():
    n, p = 8, 3
    mask = np.ones((n, p), dtype=bool)
    fields = np.zeros((n, p, 50), dtype=np.float32)
    ids = [f"event-{index}" for index in range(n)]
    sigma = np.ones(50)
    trust = np.ones(50)
    first, artifact = apply_small_field_perturbation(
        fields, mask, ids, sigma, trust, audit_seed=PERTURBATION_AUDIT_SEEDS[0]
    )
    assert artifact["sigma_multiplier"] == 0.05
    assert artifact["clip_trust_multiplier"] == 0.10
    assert np.max(np.abs(first[..., :45])) <= 0.1000001
    assert np.count_nonzero(first[..., 45:]) == 0
    order = np.asarray([5, 2, 7, 1, 6, 0, 4, 3])
    reordered, _ = apply_small_field_perturbation(
        fields[order], mask[order], [ids[index] for index in order], sigma, trust,
        audit_seed=PERTURBATION_AUDIT_SEEDS[0],
    )
    np.testing.assert_array_equal(reordered[np.argsort(order)], first)
    labels = np.zeros(n, dtype=np.int64)

    def stable(value):
        return np.zeros((len(value), 10), dtype=np.float32)

    passed = run_small_field_perturbation_audit(
        f_hat=fields, mask=mask, event_ids=ids, labels=labels,
        sigma_delta=sigma, trust_scale=trust, consumer_logits_fn=stable,
        class_order=CLASS_ORDER, selectable_for_primary_deployment=True,
    )
    assert passed["audit_seeds"] == list(PERTURBATION_AUDIT_SEEDS)
    assert passed["threshold_passed"] is True
    assert passed["automatic_selectability_gate_passed"] is True

    def brittle(value):
        score = value[..., 0].mean(axis=1) * 1e7
        logits = np.full((len(value), 10), -100.0, dtype=np.float32)
        logits[:, 0] = -score
        logits[:, 1] = score
        return logits

    failed = run_small_field_perturbation_audit(
        f_hat=fields, mask=mask, event_ids=ids, labels=labels,
        sigma_delta=sigma, trust_scale=trust, consumer_logits_fn=brittle,
        class_order=CLASS_ORDER, selectable_for_primary_deployment=True,
    )
    assert failed["threshold_passed"] is False

    def improvable(value):
        score = value[..., 0].mean(axis=1) * 1e7
        logits = np.full((len(value), 10), -100.0, dtype=np.float32)
        logits[:, 0] = -score
        logits[:, 1] = score + 1.0e-3
        return logits

    improvement = run_small_field_perturbation_audit(
        f_hat=fields, mask=mask, event_ids=ids, labels=labels,
        sigma_delta=sigma, trust_scale=trust, consumer_logits_fn=improvable,
        class_order=CLASS_ORDER, selectable_for_primary_deployment=False,
    )
    assert any(row["accuracy_loss_base_minus_perturbed"] < 0 for row in improvement["per_seed"])


def test_alignment_and_compact_quantile_distance_are_finite_threshold_free_and_final_test_locked():
    batch, physical, _ = _all50_fixture(n=4, p=5)
    sigma = np.asarray(physical["sigma_delta"])
    alignment = correction_bridge_alignment(
        batch["bridge_fields"], batch["f0"], batch["bridge_fields"], batch["mask"]
    )
    assert alignment["overall_cosine"] == pytest.approx(1.0)
    assert alignment["automatic_selection_threshold"] is None
    reference = build_bridge_quantile_reference(
        batch["bridge_fields"], batch["f0"], batch["mask"], sigma,
        stack_train_distill_manifest_sha256="d" * 64,
    )
    assert len(reference["quantiles"]) == 1001
    assert len(reference["quantiles"][0]) == 45
    assert "quantile_levels" not in reference
    distance = bridge_distribution_distance(
        reference, batch["bridge_fields"], batch["f0"], batch["mask"], sigma,
        validation_split="model_val_select",
    )
    assert max(distance["per_channel_mean_absolute_quantile_distance"]) == pytest.approx(0.0)
    assert distance["automatic_selection_threshold"] is None
    labels = np.arange(4, dtype=np.int64) % 10
    perturb = run_small_field_perturbation_audit(
        f_hat=batch["bridge_fields"], mask=batch["mask"],
        event_ids=[f"q-{index}" for index in range(4)], labels=labels,
        sigma_delta=sigma, trust_scale=np.asarray(physical["trust_scale"]),
        consumer_logits_fn=lambda value: np.zeros((len(value), 10), dtype=np.float32),
        class_order=CLASS_ORDER, selectable_for_primary_deployment=True,
    )
    adversarial = build_adversarial_channel_report(
        perturbation_audit=perturb, alignment=alignment,
        distribution_distance=distance, saturation_fraction=0.005,
        reliability_pass_through_exact=True,
        selectable_for_primary_deployment=True,
    )
    assert adversarial["automatic_selectability_gates_passed"] is True
    assert adversarial["hidden_alignment_or_distribution_cutoff_present"] is False
    with pytest.raises(PermissionError, match="final_test"):
        bridge_distribution_distance(
            reference, batch["bridge_fields"], batch["f0"], batch["mask"], sigma,
            validation_split="final_test",
        )


def test_reliability_only_response_is_explicit_nonselectable_shortcut_diagnostic():
    batch, _, _ = _all50_fixture(n=10, p=3)
    labels = np.arange(10, dtype=np.int64)
    physical = batch["bridge_fields"]
    all50 = batch["all50_bridge_fields"]

    def consumer(fields):
        logits = np.full((len(fields), 10), -2.0, dtype=np.float32)
        logits[np.arange(len(fields)), labels] = 2.0 + fields[..., 45].mean(axis=1)
        return logits

    report = evaluate_reliability_only_response(
        consumer_logits_fn=consumer, f0=batch["f0"], physical45_bridge=physical,
        all50_bridge=all50, labels=labels, class_order=CLASS_ORDER,
    )
    assert set(report["metrics"]) == {"f0", "physical45_bridge", "reliability5_only", "all50_bridge"}
    assert report["shortcut_risk_diagnostic_only"] is True
    assert report["selectable_for_primary_deployment"] is False
    accuracy_gain = compute_gain_and_recovery(
        baseline_value=0.70, teacher_bridge_value=0.72, deployable_value=0.71,
        metric_name="accuracy", metric_direction="higher_is_better",
    )
    assert accuracy_gain["recovery_fraction"] == pytest.approx(0.5)
    loss_gain = compute_gain_and_recovery(
        baseline_value=1.0, teacher_bridge_value=0.8, deployable_value=0.9,
        metric_name="cross_entropy", metric_direction="lower_is_better",
    )
    assert loss_gain["teacher_bridge_gain"] == pytest.approx(0.2)
    no_ceiling = compute_gain_and_recovery(
        baseline_value=0.7, teacher_bridge_value=0.69, deployable_value=0.71,
        metric_name="accuracy", metric_direction="higher_is_better",
    )
    assert no_ceiling["recovery_fraction"] is None


def test_step8_measures_all_special_rows_deduplicates_alias_and_passes_then_refuses_budget():
    batch, physical, all50 = _all50_fixture(n=2, p=4)
    absolute = _absolute(batch, physical)
    reference = _verified_reference(physical, width=32)
    fixed = Step8FixedStorage(
        child_split_manifest_bytes=100,
        r0_weights_bytes=100,
        target_logit_namespace_bytes={
            "physical45_selected_bridge_teacher": 100,
            "all50_selected_bridge_teacher": 100,
            "physical45_selected_teacher_on_f0_control": 100,
        },
        recipes_bindings_reports_bytes=100,
        final_deployable_bundle_bytes=100,
    )
    upstream_registry, _ = record_step3_registry_measurements(
        build_campaign_registry(),
        {
            run_id: {"run_id": run_id, "model_state_dict": {"weight": torch.ones(2)}}
            for run_id in STEP3_RUN_IDS
        },
    )
    updated, artifact = measure_step8_registry_states(
        upstream_registry, physical45_scaler_artifact=physical,
        all50_scaler_artifact=all50, absolute_scaler_artifact=absolute,
        source_manifest_sha256="d" * 64, deployed_reference=reference,
        fixed_storage=fixed,
    )
    assert artifact["newly_measured_configuration_count"] == 14
    assert artifact["unmeasured_runnable_run_ids"] == []
    assert artifact["production_readiness"]["production_submission_allowed"] is True
    assert artifact["registry_configuration_count"] == 54
    assert artifact["reconstruction_breadth_count"] == 46
    assert artifact["post_teacher_configuration_count"] == 45
    assert resolve_registry_run(updated, A3_PRIMARY_ALIAS)["content_hash"] == resolve_registry_run(updated, ARCH_A3_HLG_PRIMARY)["content_hash"]
    assert all(row["measurement_status"] == "MEASURED" for row in updated["runs"] if row["execution_status"] == "RUNNABLE")
    with pytest.raises(PermissionError, match="exceeds budget"):
        require_production_ready(
            updated, fixed_persistent_bytes=6 * 1024**3,
            selected_budget_bytes=5 * 1024**3,
        )


def test_paired_miniature_compares_hlg_particle_raw_and_r0_direct_for_all_three_seeds():
    batch, physical = _fixture(n=2, p=4)
    batch["labels"] = np.asarray([0, 1], dtype=np.int64)
    report = run_step8_paired_seed_miniature(
        physical45_scaler_artifact=physical, batch=batch
    )
    assert report["paired_seed_ids"] == [101, 202, 303]
    assert report["complete_required_comparison"] is True
    assert len(report["family_ids"]) == 4
    assert len(report["rows"]) == 12
    assert report["scientific_results_allowed"] is False


def test_step8_operator_plan_cli_validates_all_semantic_rows_without_writing(tmp_path):
    batch, physical, all50 = _all50_fixture(n=2, p=4)
    absolute = _absolute(batch, physical)
    reference = _verified_reference(physical, width=4).to_artifact()
    paths = {}
    for name, value in (("physical", physical), ("all50", all50), ("absolute", absolute), ("reference", reference)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_prediction_anchored_bridge_step8.py"
    completed = subprocess.run(
        [
            sys.executable, str(script), "--mode", "plan",
            "--physical45-scaler", str(paths["physical"]),
            "--all50-scaler", str(paths["all50"]),
            "--absolute-scaler", str(paths["absolute"]),
            "--deployed-resource-reference", str(paths["reference"]),
            "--particle-width", "4", "--dry-run",
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert len(payload["plan"]["a3_interaction_run_ids"]) == 8
    assert len(payload["plan"]["canonical_special_run_ids"]) == 14
    assert payload["plan"]["dense_field_cache_persisted"] is False


def test_fixed_storage_cli_measures_real_files_and_rejects_dense_metadata(tmp_path):
    child = tmp_path / "children.json"
    r0 = tmp_path / "r0.pt"
    bundle = tmp_path / "bundle.pt"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    child.write_bytes(b"children")
    r0.write_bytes(b"r0-weights")
    bundle.write_bytes(b"representative-bundle")
    (metadata / "binding.json").write_bytes(b"binding")
    namespace_args = []
    for name in (
        "physical45_selected_bridge_teacher",
        "all50_selected_bridge_teacher",
        "physical45_selected_teacher_on_f0_control",
    ):
        root = tmp_path / name
        root.mkdir()
        (root / "teacher_logits.npz").write_bytes(b"logits")
        (root / "teacher_logits_manifest.json").write_bytes(b"manifest")
        namespace_args.extend(["--target-namespace", f"{name}={root}"])
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_prediction_anchored_bridge_fixed_storage.py"
    completed = subprocess.run(
        [
            sys.executable, str(script), "--child-split-manifest", str(child),
            "--r0-weights", str(r0), *namespace_args,
            "--metadata-path", str(metadata),
            "--final-deployable-bundle", str(bundle), "--dry-run",
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["measurement"]["filesystem_bytes_measured"] is True
    assert payload["fixed_storage"]["dense_field_cache_bytes"] == 0
    (metadata / "dense_field.npy").write_bytes(b"forbidden")
    failed = subprocess.run(
        [
            sys.executable, str(script), "--child-split-manifest", str(child),
            "--r0-weights", str(r0), *namespace_args,
            "--metadata-path", str(metadata),
            "--final-deployable-bundle", str(bundle), "--dry-run",
        ],
        capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "forbidden dense-field" in failed.stderr
