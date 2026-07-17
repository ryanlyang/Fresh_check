from __future__ import annotations

import pytest

from teacher_logit_reco.adaptive_binary_pseudooffline.convergence_schedule import (
    ABPH_ACCELERATED_SCHEDULE_CONTRACT,
    StageScheduleBudget,
    accelerated_stage_budget,
    budget_for_stage_role,
    build_extension_comparison_report,
    decide_stage_continuation,
    infer_campaign_schedule_profile,
    reconstruction_materially_improved,
    relative_reconstruction_improvement,
    tagging_conclusion_changed,
    tagging_signal_category,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.training import (
    CurriculumController,
    ReconstructorCurriculumConfig,
    _stage_lr_multiplier,
)


def _history(scores, root_losses=None):
    root_losses = scores if root_losses is None else root_losses
    return [
        {
            "global_update": 100 * (index + 1),
            "model_val_rollout": {
                "selection_score": score,
                "metrics": {"loss.raw.root": root_loss},
            },
        }
        for index, (score, root_loss) in enumerate(zip(scores, root_losses))
    ]


def _accelerated_root_config(budget: StageScheduleBudget) -> ReconstructorCurriculumConfig:
    return ReconstructorCurriculumConfig(
        root_updates=budget.nominal_updates,
        hierarchy_updates_per_depth=1,
        renderer_updates=1,
        distribution_updates=1,
        maximum_capacity=1,
        hierarchy_capacities=(),
        renderer_enabled=False,
        distribution_enabled=False,
        schedule_contract=ABPH_ACCELERATED_SCHEDULE_CONTRACT,
        campaign_schedule_profile="pilot",
        root_budget=budget,
        root_stage_role="trained",
        hierarchy_stage_role="disabled",
        renderer_stage_role="disabled",
        distribution_stage_role="disabled",
    )


def test_campaign_profiles_have_the_locked_nominal_extension_and_hard_budgets():
    assert accelerated_stage_budget("pilot", "root") == StageScheduleBudget(
        12_000, 4_000, 24_000
    )
    assert accelerated_stage_budget("pilot", "hierarchy") == StageScheduleBudget(
        8_000, 4_000, 16_000
    )
    assert accelerated_stage_budget("pilot", "renderer") == StageScheduleBudget(
        15_000, 5_000, 30_000
    )
    assert accelerated_stage_budget("pilot", "distribution") == StageScheduleBudget(
        10_000, 5_000, 20_000
    )
    assert accelerated_stage_budget("highdata", "root") == StageScheduleBudget(
        50_000, 10_000, 80_000
    )
    assert accelerated_stage_budget("highdata", "hierarchy") == StageScheduleBudget(
        40_000, 10_000, 60_000
    )
    assert accelerated_stage_budget("highdata", "renderer") == StageScheduleBudget(
        80_000, 20_000, 120_000
    )
    assert accelerated_stage_budget("highdata", "distribution") == StageScheduleBudget(
        50_000, 10_000, 80_000
    )
    assert budget_for_stage_role(
        accelerated_stage_budget("pilot", "root"), "warm_started_handoff"
    ) == StageScheduleBudget(1, 0, 1)


def test_schedule_profile_is_inferred_only_from_immutable_split_sizes():
    assert (
        infer_campaign_schedule_profile(
            model_train_jets=500_000, model_val_jets=150_000
        )
        == "pilot"
    )
    assert (
        infer_campaign_schedule_profile(
            model_train_jets=5_000_000, model_val_jets=1_000_000
        )
        == "highdata"
    )
    with pytest.raises(ValueError, match="immutable pilot or high-data split"):
        infer_campaign_schedule_profile(
            model_train_jets=1_250_000, model_val_jets=250_000
        )


def test_continuation_rule_extends_only_when_every_registered_check_passes():
    decision = decide_stage_continuation(
        _history((1.0, 0.996, 0.992), (1.0, 0.99, 0.98)),
        required_objectives=("root",),
        best_checkpoint_global_update=300,
        stage_update=12_000,
        nominal_updates=12_000,
        extension_blocks_completed=0,
        hard_max_updates=24_000,
        nonfinite_updates=0,
        compiler_failure_updates=0,
    )
    assert decision.continue_training is True
    assert decision.outcome == "extended_for_convergence"
    assert decision.schedule_truncated is False
    assert decision.robust_relative_slope < -0.002
    assert all(decision.checks.values())

    degraded = decide_stage_continuation(
        _history((1.0, 0.996, 0.992), (1.0, 0.94, 1.01)),
        required_objectives=("root",),
        best_checkpoint_global_update=300,
        stage_update=12_000,
        nominal_updates=12_000,
        extension_blocks_completed=0,
        hard_max_updates=24_000,
        nonfinite_updates=0,
        compiler_failure_updates=0,
    )
    assert degraded.continue_training is False
    assert degraded.outcome == "nominal_completed"
    assert degraded.checks["required_objectives_within_5_percent_of_best"] is False


def test_extension_plateau_and_improving_hard_cap_are_distinguished():
    plateau = decide_stage_continuation(
        _history((1.0, 0.999, 0.999)),
        required_objectives=("root",),
        best_checkpoint_global_update=200,
        stage_update=16_000,
        nominal_updates=12_000,
        extension_blocks_completed=1,
        hard_max_updates=24_000,
        nonfinite_updates=0,
        compiler_failure_updates=0,
    )
    assert plateau.continue_training is False
    assert plateau.outcome == "plateau_stopped"
    assert plateau.schedule_truncated is False

    hard_cap = decide_stage_continuation(
        _history((1.0, 0.996, 0.992), (1.0, 0.99, 0.98)),
        required_objectives=("root",),
        best_checkpoint_global_update=300,
        stage_update=24_000,
        nominal_updates=12_000,
        extension_blocks_completed=3,
        hard_max_updates=24_000,
        nonfinite_updates=0,
        compiler_failure_updates=0,
    )
    assert hard_cap.continue_training is False
    assert hard_cap.outcome == "hard_max_reached"
    assert hard_cap.schedule_truncated is True


def test_cosine_reaches_minimum_at_nominal_and_never_restarts_in_extension():
    controller = CurriculumController(
        _accelerated_root_config(StageScheduleBudget(3, 2, 5))
    )
    for _ in range(3):
        reached_boundary = controller.advance()
    assert reached_boundary is True
    nominal_state = controller.state()
    assert nominal_state.stage_update == 3
    nominal_lr = _stage_lr_multiplier(
        nominal_state, warmup_fraction=0.05, minimum_fraction=0.05
    )
    assert nominal_lr == pytest.approx(0.05)
    controller.approve_extension(
        {"stage_key": "phase1_root", "outcome": "extended_for_convergence"}
    )
    extension_state = controller.state()
    assert extension_state.stage_maximum_updates == 5
    assert extension_state.stage_extension_blocks == 1
    assert _stage_lr_multiplier(
        extension_state, warmup_fraction=0.05, minimum_fraction=0.05
    ) == pytest.approx(0.05)
    controller.advance()
    assert _stage_lr_multiplier(
        controller.state(), warmup_fraction=0.05, minimum_fraction=0.05
    ) == pytest.approx(0.05)


def test_old_curriculum_payload_is_explicitly_legacy_not_accelerated():
    old = {
        "root_updates": 10,
        "hierarchy_updates_per_depth": 8,
        "renderer_updates": 12,
        "distribution_updates": 12,
        "evaluation_interval": 2,
        "root_patience_evaluations": 2,
        "hierarchy_patience_evaluations": 2,
        "renderer_patience_evaluations": 2,
        "distribution_patience_evaluations": 2,
        "renderer_true_parent_fraction": 0.25,
        "renderer_transition_fraction": 0.5,
        "maximum_capacity": 1,
        "hierarchy_capacities": (),
        "renderer_enabled": False,
        "distribution_enabled": False,
    }
    loaded = ReconstructorCurriculumConfig.from_dict(old)
    assert loaded.accelerated is False
    assert loaded.campaign_schedule_profile == "legacy"
    assert loaded.stage_budget("root") == StageScheduleBudget(10, 0, 10)


def test_truncation_and_tagging_thresholds_use_registered_inclusive_boundaries():
    assert relative_reconstruction_improvement(1.0, 0.995) == pytest.approx(0.005)
    assert reconstruction_materially_improved(1.0, 0.995) is True
    assert reconstruction_materially_improved(1.0, 0.99501) is False
    assert tagging_signal_category(0.002) == "positive_signal"
    assert tagging_signal_category(-0.002) == "negative_signal"
    assert tagging_signal_category(0.001999) == "no_clear_signal"
    assert tagging_signal_category(-0.001999) == "no_clear_signal"
    assert tagging_conclusion_changed(0.001, 0.002) is True
    assert tagging_conclusion_changed(0.003, 0.004) is False


def test_extension_comparison_records_raw_thresholds_hashes_and_checkpoint_policy():
    report = build_extension_comparison_report(
        variant_name="D1_kt32_mh4_particles",
        nominal_checkpoint_hash="nominal-hash",
        extension_checkpoint_hash="extension-hash",
        matched_a0_artifact_hash="a0-hash",
        frozen_tagger_recipe_hash="tagger-recipe-hash",
        nominal_best_loss=1.0,
        extension_best_loss=0.995,
        nominal_tagging_gain=0.001,
        extension_tagging_gain=0.002,
        initialization_seed=24731,
        training_budget_hash="budget-hash",
    )
    assert report["final_test_loaded"] is False
    assert report["material_reconstruction_improvement"] is True
    assert report["categories"]["tagging_conclusion_changed"] is True
    assert report["schedule_truncated"] is True
    assert report["screening_checkpoint_policy"] == "use_extension_cap"
    assert report["thresholds"] == {
        "material_relative_reconstruction_improvement": 0.005,
        "positive_signal_accuracy_gain": 0.002,
        "negative_signal_accuracy_gain": -0.002,
    }
    assert report["matched_a0_artifact_hash"] == "a0-hash"
    with pytest.raises(ValueError, match="restricted to model_val"):
        build_extension_comparison_report(
            variant_name="D1_kt32_mh4_particles",
            nominal_checkpoint_hash="nominal-hash",
            extension_checkpoint_hash="extension-hash",
            matched_a0_artifact_hash="a0-hash",
            frozen_tagger_recipe_hash="tagger-recipe-hash",
            nominal_best_loss=1.0,
            extension_best_loss=0.999,
            nominal_tagging_gain=0.0,
            extension_tagging_gain=0.0,
            initialization_seed=24731,
            training_budget_hash="budget-hash",
            evaluation_split="final_test",
        )
