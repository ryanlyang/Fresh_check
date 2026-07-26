from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    A0_A0_PAIRS,
    DirectControlCandidate,
    aggregate_three_seed_configuration,
    apply_particle_view_control,
    assert_split_access,
    build_a0_a0_pair_recipes,
    build_fusion_recipe,
    build_paired_statistics_report,
    build_sealed_split_authorization,
    build_selected_path_fairness_ledger,
    build_stage_g_control_plan,
    build_stack_confirmation_report,
    build_stack_fusion_partition,
    build_step8_control_registry,
    evaluate_fusion_recipe,
    exact_two_sided_mcnemar_pvalue,
    expand_three_seed_confirmation_rows,
    fit_linear_logit_fusion,
    select_direct_resource_control,
    select_particle_view_winner_families,
    with_content_hash,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_step8_structural_controls_are_deterministic_mask_safe_and_deranged():
    view = torch.arange(4 * 5 * 2, dtype=torch.float32).reshape(4, 5, 2)
    mask = torch.ones(4, 1, 5, dtype=torch.bool)
    mask[0, 0, -1] = False
    view[0, -1] = 999
    labels = torch.tensor([0, 0, 1, 1])
    shuffled, audit = apply_particle_view_control(
        view,
        mask,
        control_id="EVENT_SHUFFLED_VIEW",
        seed=101,
        labels=labels,
    )
    repeated, repeated_audit = apply_particle_view_control(
        view,
        mask,
        control_id="EVENT_SHUFFLED_VIEW",
        seed=101,
        labels=labels,
    )
    assert torch.equal(shuffled, repeated)
    assert audit == repeated_audit
    assert audit["event_permutation_fixed_points"] == 0
    assert shuffled[0, -1].abs().max().item() == 0

    random_view, _ = apply_particle_view_control(
        view, mask, control_id="SAME_NORM_RANDOM_VIEW", seed=202
    )
    valid = mask[:, 0]
    clean = torch.where(valid[:, :, None], view, torch.zeros_like(view))
    assert torch.allclose(
        random_view.flatten(1).norm(dim=1),
        clean.flatten(1).norm(dim=1),
        rtol=1e-5,
    )
    centered, _ = apply_particle_view_control(
        view, mask, control_id="JET_MEAN_REMOVED"
    )
    for event in range(4):
        assert centered[event, valid[event]].mean(dim=0).abs().max() < 1e-5

    pt = torch.tensor([[1, 5, 4, 3, 2]] * 4, dtype=torch.float32)
    top, _ = apply_particle_view_control(
        view,
        mask,
        control_id="TRUE_VIEW_TOP_PT_25",
        particle_pt=pt,
    )
    assert int((top.abs().sum(dim=-1) > 0).sum()) <= 8
    registry = build_step8_control_registry()
    assert registry["performance_gates"] is False
    assert registry["structural_control_count"] == 16


def test_step8_direct_control_matching_uses_locked_relative_error_and_warns():
    candidates = [
        DirectControlCandidate("small", 90, 180, _sha("small")),
        DirectControlCandidate("large", 104, 240, _sha("large")),
    ]
    selected = select_direct_resource_control(
        candidates=candidates,
        target_parameters=100,
        target_flops=200,
        requested_quantity="parameters",
        selected_bundle_sha256=_sha("bundle"),
        flop_fixture_sha256=_sha("fixture"),
        flop_counter_sha256=_sha("counter"),
    )
    assert selected["selected"]["config_id"] == "large"
    assert selected["relative_parameter_error"] == pytest.approx(0.04)
    assert selected["match_within_requested_tolerance"]
    warned = select_direct_resource_control(
        candidates=[candidates[0]],
        target_parameters=200,
        target_flops=400,
        requested_quantity="flops",
        selected_bundle_sha256=_sha("bundle"),
        flop_fixture_sha256=_sha("fixture"),
        flop_counter_sha256=_sha("counter"),
    )
    assert warned["quality_warning"] == "WARN_CONTROL_MATCH_TOLERANCE"
    assert warned["warning_is_non_gating"]


def _replica(
    config: str,
    seed: int,
    accuracy: float,
    *,
    privileged: bool,
    deployable: bool = True,
    diagnostic: bool = False,
    recovery: float | None = 0.2,
):
    return {
        "configuration_id": config,
        "run_id": f"{config}_{seed}",
        "seed": seed,
        "split": "model_val_select",
        "deployable_accuracy": accuracy,
        "deployable_cross_entropy": 1.0 - accuracy,
        "recovery_status": "finite" if recovery is not None else "undefined",
        "recovery_fraction": recovery,
        "oracle_gain": 0.1 if recovery is not None else None,
        "deployed_parameters": 1000 if config == "priv" else 900,
        "bundle_sha256": _sha(f"{config}-{seed}"),
        "privileged_claim_eligible": privileged,
        "pre_stage_g_deployable_eligible": deployable,
        "diagnostic": diagnostic,
    }


def _selection():
    rows = []
    for seed, accuracy in zip((101, 202, 303), (0.80, 0.82, 0.81)):
        rows.append(_replica("priv", seed, accuracy, privileged=True))
    for seed, accuracy in zip((101, 202, 303), (0.83, 0.82, 0.84)):
        rows.append(
            _replica(
                "ce_only",
                seed,
                accuracy,
                privileged=False,
                recovery=None,
            )
        )
    for seed, accuracy in zip((101, 202, 303), (0.90, 0.90, 0.90)):
        rows.append(
            _replica(
                "shuffle",
                seed,
                accuracy,
                privileged=False,
                deployable=False,
                diagnostic=True,
                recovery=None,
            )
        )
    return select_particle_view_winner_families(rows)


def test_step8_seed_aggregation_and_winner_families_are_distinct():
    expansion = expand_three_seed_confirmation_rows(
        screen_rows=[
            {"configuration_id": "canonical", "architecture": "PVA3"},
            {"configuration_id": "best", "architecture": "basic"},
        ],
        confirmation_roles={
            "canonical_predeclared": "canonical",
            "best_architecture": "best",
            "best_small_ce": "best",
        },
    )
    assert expansion["row_count"] == 6
    best = [row for row in expansion["rows"] if row["configuration_id"] == "best"]
    assert all(
        row["confirmation_roles"] == ["best_architecture", "best_small_ce"]
        for row in best
    )
    aggregate = aggregate_three_seed_configuration(
        [
            _replica("priv", 101, 0.80, privileged=True),
            _replica("priv", 202, 0.82, privileged=True),
            _replica("priv", 303, 0.81, privileged=True),
        ]
    )
    assert aggregate["mean_accuracy"] == pytest.approx(0.81)
    assert aggregate["representative_seed"] == 303
    selection = _selection()
    assert (
        selection["selected_privileged_scientific_model"]["configuration_id"]
        == "priv"
    )
    assert (
        selection["selected_pre_stage_g_hlt_deployable_model"]["configuration_id"]
        == "ce_only"
    )
    assert selection["best_diagnostic_control"]["configuration_id"] == "shuffle"
    assert selection["stack_val_loaded"] is False
    assert selection["minimum_quality_gate"] is None


def _training_ledger(config: str, seed: int):
    return with_content_hash(
        {
            "contract": "test_training_ledger_v1",
            "train_identity_sha256": _sha("train"),
            "totals_retained_deployable_path": {
                "optimizer_steps": 30,
                "label_bearing_steps": 20,
                "labeled_examples_processed": 200,
                "ce_bearing_steps": 12,
                "teacher_kd_steps": 10,
                "view_supervision_steps": 10,
                "training_flops": 1234,
            },
            "totals_all_training": {
                "optimizer_steps": 50,
                "label_bearing_steps": 40,
                "labeled_examples_processed": 400,
                "ce_bearing_steps": 20,
                "teacher_kd_steps": 10,
                "view_supervision_steps": 10,
                "training_flops": 4321,
            },
            "config": config,
            "seed": seed,
        }
    )


def _resource(config: str, seed: int):
    return with_content_hash(
        {
            "contract": "test_resource_v1",
            "architecture_config_sha256": _sha(f"arch-{config}"),
            "total_parameters": 1000,
            "forward_flops": {"exact_integer_total": 2000},
            "seed": seed,
        }
    )


def _fairness_and_authorization():
    selection = _selection()
    ledgers = {
        config: {seed: _training_ledger(config, seed) for seed in (101, 202, 303)}
        for config in ("priv", "ce_only")
    }
    resources = {
        config: {seed: _resource(config, seed) for seed in (101, 202, 303)}
        for config in ("priv", "ce_only")
    }
    fairness = build_selected_path_fairness_ledger(
        selection=selection,
        replica_training_ledgers=ledgers,
        resource_profiles=resources,
        train_identity_sha256=_sha("train"),
        flop_fixture_sha256=_sha("fixture"),
        flop_counter_sha256=_sha("counter"),
    )
    controls = [
        {
            "bundle_sha256": _sha(f"control-{seed}"),
            "seed": seed,
            "role": "stage_g_control",
        }
        for seed in (101, 202, 303)
    ]
    authorization = build_sealed_split_authorization(
        selection=selection,
        fairness_ledger=fairness,
        stack_split_sha256=_sha("stack"),
        final_test_split_sha256=_sha("test"),
        stage_g_control_bundles=controls,
    )
    return selection, fairness, authorization


def test_step8_fairness_closure_and_sealed_split_permissions_fail_closed():
    selection, fairness, authorization = _fairness_and_authorization()
    assert fairness["stage_g_controls_may_start"]
    assert fairness["distinct_entry_count"] == 2
    first = fairness["entries"][0]["replicas"][0]
    assert first["a0_view_long_deploy_exact_ce_updates"] == 12
    assert first["a0_view_total_label_budget_exact_updates"] == 40
    stage_g = build_stage_g_control_plan(
        fairness_ledger=fairness,
        candidates=[
            DirectControlCandidate("matched", 1000, 2000, _sha("matched"))
        ],
        a0_checkpoint_by_seed={
            seed: _sha(f"a0-{seed}") for seed in (101, 202, 303)
        },
        a0_config_sha256=_sha("a0-config"),
    )
    assert stage_g["job_count"] == 24
    assert stage_g["performance_gates"] is False
    exact = next(
        row for row in stage_g["jobs"] if row["control_id"] == "A0_VIEW_LONG_DEPLOY"
    )
    assert exact["registered_checkpoints"] == [
        "exact_matched_update",
        "best_model_val_stop_within_budget",
    ]

    privileged = selection["selected_privileged_scientific_model"]
    assert_split_access(
        authorization,
        split="stack_val",
        artifact_sha256=privileged["replicas"][0]["bundle_sha256"],
    )
    assert_split_access(
        authorization,
        split="final_test",
        artifact_sha256=privileged["representative_bundle_sha256"],
    )
    with pytest.raises(PermissionError):
        assert_split_access(
            authorization,
            split="final_test",
            artifact_sha256=_sha("control-101"),
        )
    with pytest.raises(PermissionError, match="HLT-only"):
        assert_split_access(
            authorization,
            split="final_test",
            artifact_sha256=privileged["representative_bundle_sha256"],
            requires_offline=True,
        )


def test_step8_stack_controls_can_warn_but_never_replace_selection():
    selection, _, authorization = _fairness_and_authorization()
    rows = []
    for family in (
        "selected_privileged_scientific_model",
        "selected_pre_stage_g_hlt_deployable_model",
    ):
        for replica in selection[family]["replicas"]:
            rows.append(
                {
                    "bundle_sha256": replica["bundle_sha256"],
                    "configuration_id": selection[family]["configuration_id"],
                    "seed": replica["seed"],
                    "role": "preselected_winner_replica",
                    "split": "stack_val",
                    "accuracy": 0.80,
                }
            )
    for seed in (101, 202, 303):
        rows.append(
            {
                "bundle_sha256": _sha(f"control-{seed}"),
                "configuration_id": "selected_parameter_match",
                "seed": seed,
                "role": "stage_g_control",
                "split": "stack_val",
                "accuracy": 0.90,
            }
        )
    report = build_stack_confirmation_report(
        selection=selection, authorization=authorization, stack_rows=rows
    )
    assert report["selection_changed"] is False
    assert report["winner_summaries"][
        "selected_privileged_scientific_model"
    ]["post_stage_g_control_numerically_better"]


def test_step8_paired_bootstrap_mcnemar_and_fusion_are_deterministic():
    baseline_logits = np.array(
        [[3, 0], [0, 3], [3, 0], [3, 0], [0, 3], [3, 0]], dtype=float
    )
    candidate_logits = np.array(
        [[3, 0], [0, 3], [0, 3], [3, 0], [0, 3], [0, 3]], dtype=float
    )
    labels = np.array([0, 1, 1, 0, 1, 1])
    baseline_correct = baseline_logits.argmax(1) == labels
    candidate_correct = candidate_logits.argmax(1) == labels
    n01, n10, pvalue = exact_two_sided_mcnemar_pvalue(
        baseline_correct, candidate_correct
    )
    assert (n01, n10) == (0, 2)
    assert pvalue == pytest.approx(0.5)
    report = build_paired_statistics_report(
        baseline_logits=baseline_logits,
        candidate_logits=candidate_logits,
        labels=labels,
        split="stack_val",
        baseline_artifact_sha256=_sha("baseline"),
        candidate_artifact_sha256=_sha("candidate"),
        split_sha256=_sha("stack"),
        event_identity_sha256=_sha("identities"),
        replicates=200,
    )
    repeated = build_paired_statistics_report(
        baseline_logits=baseline_logits,
        candidate_logits=candidate_logits,
        labels=labels,
        split="stack_val",
        baseline_artifact_sha256=_sha("baseline"),
        candidate_artifact_sha256=_sha("candidate"),
        split_sha256=_sha("stack"),
        event_identity_sha256=_sha("identities"),
        replicates=200,
    )
    assert report == repeated

    partition = build_stack_fusion_partition(
        event_identities=list(range(6)),
        stack_split_sha256=_sha("stack"),
    )
    parameters = fit_linear_logit_fusion(
        source_logits=(baseline_logits, candidate_logits),
        labels=labels,
        fit_indices=partition["fit_indices"],
        steps=20,
    )
    recipe = build_fusion_recipe(
        fusion_id="A0_PLUS_DVIEW",
        source_bundle_sha256=(_sha("baseline"), _sha("candidate")),
        class_order=("a", "b"),
        stack_partition=partition,
        method="linear_logit",
        linear_parameters=parameters,
    )
    fusion_report = evaluate_fusion_recipe(
        recipe=recipe,
        stack_partition=partition,
        source_logits=(baseline_logits, candidate_logits),
        labels=labels,
        source_bundle_sha256=(_sha("baseline"), _sha("candidate")),
    )
    assert fusion_report["evaluation_only"]
    assert fusion_report["metrics"]["split"] == "stack_val_evaluation"

    a0_recipes = build_a0_a0_pair_recipes(
        checkpoints_by_seed={seed: _sha(f"a0-{seed}") for seed in (101, 202, 303)},
        class_order=("a", "b"),
        stack_partition=partition,
    )
    assert A0_A0_PAIRS == ((101, 202), (202, 303), (303, 101))
    assert len(a0_recipes) == 3
    assert all(
        len(set(recipe["source_bundle_sha256"])) == 2 for recipe in a0_recipes
    )


def test_step8_tigris_contracts_use_correct_account_environment_and_no_user_site():
    from pathlib import Path

    for name in (
        "run_particle_view_confirmation.sh",
        "run_particle_view_controls.sh",
        "run_particle_view_fusion.sh",
    ):
        text = (Path("sbatch") / name).read_text(encoding="utf-8")
        assert "#SBATCH --account=reu-aisocial" in text
        assert "export PYTHONNOUSERSITE=1" in text
        assert "CONDA_ENV:=atlas_kd_tigris" in text
        assert "miniforge3-aarch64" in text
        assert "reu-aisoc\n" not in text
