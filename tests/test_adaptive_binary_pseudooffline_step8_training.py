from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from teacher_logit_reco.adaptive_binary_pseudooffline.training import (
    ABPH_RECONSTRUCTOR_MODULE_GROUPS,
    CurriculumController,
    CurriculumState,
    CyclingSequenceBatchSource,
    OptimizerGroupPolicy,
    ReconstructorCurriculumConfig,
    ReconstructorStepContext,
    ReconstructorStepResult,
    ReconstructorTrainerConfig,
    active_reconstruction_loss_names,
    assemble_reconstruction_loss_terms,
    build_reconstructor_optimizer,
    compose_reconstruction_loss,
    configure_reconstructor_optimizer,
    load_reconstructor_curriculum_checkpoint,
    train_reconstructor_curriculum,
)


class _TinyCurriculumModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hlt_encoder = torch.nn.Linear(1, 1, bias=False)
        self.root = torch.nn.Linear(1, 1, bias=False)
        self.hierarchy_2 = torch.nn.Linear(1, 1, bias=False)
        self.hierarchy_4 = torch.nn.Linear(1, 1, bias=False)
        self.hierarchy_8 = torch.nn.Linear(1, 1, bias=False)
        self.hierarchy_16 = torch.nn.Linear(1, 1, bias=False)
        self.hierarchy_32 = torch.nn.Linear(1, 1, bias=False)
        self.renderer = torch.nn.Linear(1, 1, bias=False)
        self.distribution = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.4)

    def module_groups(self):
        return {name: getattr(self, name) for name in ABPH_RECONSTRUCTOR_MODULE_GROUPS}


def _step(model, batch, context):
    x = batch["x"]
    target = batch["target"]
    evidence = model.hlt_encoder(x)
    predictions = {
        "root": model.root(evidence),
        **{
            f"group_{capacity}": getattr(model, f"hierarchy_{capacity}")(evidence)
            for capacity in (2, 4, 8, 16, 32)
        },
        "particle": model.renderer(evidence),
        "distribution": model.distribution(evidence),
    }

    def squared(name):
        return (predictions[name] - target).square().mean()

    active_group = f"group_{max(2, context.curriculum.active_capacity)}"
    terms = {
        "root": squared("root"),
        **{f"group_{capacity}": squared(f"group_{capacity}") for capacity in (2, 4, 8, 16, 32)},
        "topology": squared(active_group),
        "frontier": squared(active_group),
        "particle": squared("particle"),
        "particle_feature": 0.5 * squared("particle"),
        "auxiliary": 0.25 * squared("particle"),
        "distribution": squared("distribution"),
        "calibration": 0.25 * squared("distribution"),
    }
    return ReconstructorStepResult(
        loss_terms=terms,
        metrics={
            "prediction_error": squared("root").detach(),
            "mode_is_rollout": float(context.mode == "rollout"),
        },
        batch_size=int(x.shape[0]),
        tensors_to_check=tuple(predictions.values()),
    )


def _curriculum(updates: int = 2) -> ReconstructorCurriculumConfig:
    return ReconstructorCurriculumConfig(
        root_updates=updates,
        hierarchy_updates_per_depth=updates,
        renderer_updates=updates,
        distribution_updates=updates,
        evaluation_interval=1,
        root_patience_evaluations=20,
        hierarchy_patience_evaluations=20,
        renderer_patience_evaluations=20,
        distribution_patience_evaluations=20,
    )


def _trainer_config(output_dir, updates: int = 2) -> ReconstructorTrainerConfig:
    return ReconstructorTrainerConfig(
        output_dir=str(output_dir),
        seed=9127,
        device="cpu",
        amp=False,
        gradient_accumulation_steps=1,
        root_hierarchy_effective_batch_size=8,
        renderer_distribution_effective_batch_size=8,
        ema_decay=0.0,
        warmup_fraction=0.25,
        curriculum=_curriculum(updates),
    )


def _policies():
    return {
        name: OptimizerGroupPolicy(peak_lr=0.05, weight_decay=0.0)
        for name in ABPH_RECONSTRUCTOR_MODULE_GROUPS
    }


def _source():
    return CyclingSequenceBatchSource(
        [
            {
                "x": torch.ones(8, 1),
                "target": torch.ones(8, 1),
            }
        ]
    )


def _validation_batches():
    return [
        {
            "x": torch.ones(8, 1),
            "target": torch.ones(8, 1),
        }
    ]


def test_curriculum_uses_locked_depth_and_teacher_forcing_schedule():
    controller = CurriculumController(_curriculum(updates=10))
    for _ in range(10):
        controller.advance()
    state = controller.state()
    assert state.stage_key == "phase2_hierarchy_2"
    assert state.teacher_forcing_probability == pytest.approx(1.0)
    for _ in range(5):
        controller.advance()
    state = controller.state()
    assert state.active_capacity == 2
    assert 0.25 < state.teacher_forcing_probability < 1.0
    for _ in range(3):
        controller.advance()
    assert controller.state().teacher_forcing_probability == pytest.approx(0.25)

    saved = controller.state_dict()
    restored = CurriculumController(_curriculum(updates=10))
    restored.load_state_dict(saved)
    assert restored.state() == controller.state()


def test_required_objectives_fail_closed_and_validation_requires_rollout():
    state = CurriculumState(
        stage_index=6,
        stage_key="phase3_renderer",
        phase=3,
        phase_name="deterministic_particle_rendering",
        global_update=1,
        stage_update=1,
        stage_maximum_updates=10,
        active_capacity=32,
        stage_progress=0.1,
        teacher_forcing_probability=0.5,
        distribution_weight=0.0,
    )
    context = ReconstructorStepContext(
        curriculum=state,
        split="model_val",
        mode="rollout",
        validation=True,
        teacher_forcing_probability=0.0,
    )
    required = active_reconstruction_loss_names(context)
    assert "frontier" in required
    assert "particle" in required
    losses = {name: torch.tensor(1.0, requires_grad=True) for name in required}
    losses.pop("frontier")
    with pytest.raises(KeyError, match="frontier"):
        compose_reconstruction_loss(
            ReconstructorStepResult(loss_terms=losses, metrics={}, batch_size=1),
            context,
        )
    with pytest.raises(ValueError, match="zero-teacher-forcing"):
        ReconstructorStepContext(
            curriculum=state,
            split="model_val",
            mode="teacher_forced",
            validation=True,
            teacher_forcing_probability=1.0,
        )


def test_typed_component_outputs_assemble_without_topology_or_calibration_double_count():
    root = SimpleNamespace(total=torch.tensor(1.0))
    accounting = SimpleNamespace(
        components={"topology_nll": torch.tensor(0.2)},
        weights=SimpleNamespace(topology_nll=1.0),
    )
    hierarchy = SimpleNamespace(
        accounting_loss=accounting,
        total_loss=torch.tensor(1.2),
    )
    frontier = SimpleNamespace(mode="rollout", total_frontier_loss=torch.tensor(0.3))
    particle = SimpleNamespace(
        total=torch.tensor(0.4),
        component_losses={"kinematic": torch.tensor(0.1), "pid": torch.tensor(0.3)},
    )
    auxiliary = SimpleNamespace(total=torch.tensor(0.5))
    distribution = SimpleNamespace(
        total=torch.tensor(0.6),
        calibration_loss=torch.tensor(0.1),
        metrics={"weights": {"calibration": 0.1}},
    )
    terms = assemble_reconstruction_loss_terms(
        root_loss=root,
        hierarchy_supervision=(hierarchy,),
        rollout_alignment=frontier,
        particle_matching=particle,
        particle_auxiliary=auxiliary,
        distribution_loss=distribution,
    )
    assert terms["root"].item() == pytest.approx(1.0)
    assert terms["group_2"].item() == pytest.approx(1.0)
    assert terms["topology"].item() == pytest.approx(0.2)
    assert terms["frontier"].item() == pytest.approx(0.3)
    assert terms["particle_feature"].item() == pytest.approx(0.2)
    assert terms["distribution"].item() == pytest.approx(0.59)
    assert terms["calibration"].item() == pytest.approx(0.1)


def test_optimizer_groups_record_every_learning_rate_and_trainability(tmp_path):
    model = _TinyCurriculumModel()
    optimizer = build_reconstructor_optimizer(
        model, model.module_groups(), policies=_policies()
    )
    controller = CurriculumController(_curriculum())
    controller.advance()
    controller.advance()
    state = controller.state()
    config = _trainer_config(tmp_path)
    rows = configure_reconstructor_optimizer(optimizer, state, config)
    assert {row["group_name"] for row in rows} == set(ABPH_RECONSTRUCTOR_MODULE_GROUPS)
    assert next(row for row in rows if row["group_name"] == "hierarchy_2")["trainable"]
    assert not next(row for row in rows if row["group_name"] == "hierarchy_4")["trainable"]
    assert not next(row for row in rows if row["group_name"] == "renderer")["trainable"]
    assert all("learning_rate" in row and "parameter_count" in row for row in rows)


def test_complete_short_overfit_visits_and_selects_every_depth(tmp_path):
    torch.manual_seed(44)
    model = _TinyCurriculumModel()
    config = _trainer_config(tmp_path / "complete")
    report = train_reconstructor_curriculum(
        model,
        model.module_groups(),
        _source(),
        _validation_batches,
        _step,
        config,
        provenance={"manifest_hash": "test-manifest"},
        optimizer_policies=_policies(),
    )
    assert report["ok"] is True
    assert report["selection_split"] == "model_val"
    assert report["selection_mode"] == "rollout"
    assert report["teacher_forced_validation_count"] == 0
    expected_stages = {
        "phase1_root",
        *(f"phase2_hierarchy_{capacity}" for capacity in (2, 4, 8, 16, 32)),
        "phase3_renderer",
        "phase4_distribution",
    }
    assert set(report["best_by_stage"]) == expected_stages
    curves = json.loads((tmp_path / "complete" / "training_curves.json").read_text())
    assert curves["rollout_validation_required"] is True
    assert all(row["model_val_rollout"]["mode"] == "rollout" for row in curves["evaluations"])
    for row in curves["evaluations"]:
        assert set(row["train"]["objective_gradient_norms"]) == set(
            row["train"]["required_losses"]
        )
        assert set(row["train"]["optimizer_group_gradient_norms"]) == set(
            ABPH_RECONSTRUCTOR_MODULE_GROUPS
        )
        assert row["train"]["effective_batch_size"] == row["train"][
            "expected_effective_batch_size"
        ]
    capacities = {
        row["curriculum"]["active_capacity"]
        for row in curves["evaluations"]
        if row["curriculum"]["phase"] == 2
    }
    assert capacities == {2, 4, 8, 16, 32}
    for stage in expected_stages:
        rows = [row for row in curves["evaluations"] if row["curriculum"]["stage_key"] == stage]
        assert rows[-1]["model_val_rollout"]["selection_score"] <= rows[0]["model_val_rollout"]["selection_score"]
    selected = load_reconstructor_curriculum_checkpoint(
        tmp_path / "complete" / "best_model_val.pt", require_selected=True
    )
    assert selected["selection_mode"] == "rollout"
    assert selected["final_test_loaded"] is False
    assert selected["model_metadata"]["model_metadata_hash"]
    assert set(selected["model_metadata"]["module_groups"]) == set(
        ABPH_RECONSTRUCTOR_MODULE_GROUPS
    )


def test_resume_restores_curriculum_optimizer_ema_rng_and_data_cursor(tmp_path):
    torch.manual_seed(77)
    model = _TinyCurriculumModel()
    config = _trainer_config(tmp_path / "resumed")
    interrupted = train_reconstructor_curriculum(
        model,
        model.module_groups(),
        _source(),
        _validation_batches,
        _step,
        config,
        provenance={"manifest_hash": "resume-manifest"},
        maximum_optimizer_updates=5,
        optimizer_policies=_policies(),
    )
    assert interrupted["ok"] is False
    last = load_reconstructor_curriculum_checkpoint(tmp_path / "resumed" / "last.pt")
    assert last["curriculum_state_dict"]["global_update"] == 5
    assert last["train_source_state_dict"]["cycles"] == 5

    torch.manual_seed(999)
    resumed_model = _TinyCurriculumModel()
    resumed = train_reconstructor_curriculum(
        resumed_model,
        resumed_model.module_groups(),
        _source(),
        _validation_batches,
        _step,
        config,
        provenance={"manifest_hash": "resume-manifest"},
        resume_from=tmp_path / "resumed" / "last.pt",
        optimizer_policies=_policies(),
    )
    assert resumed["ok"] is True
    assert resumed["curriculum"]["global_update"] == 16
    assert resumed["rollout_validation_count"] == 16


def test_stage_boundary_last_checkpoint_contains_selected_ema_handoff(tmp_path):
    torch.manual_seed(103)
    model = _TinyCurriculumModel()
    config = _trainer_config(tmp_path / "handoff")
    train_reconstructor_curriculum(
        model,
        model.module_groups(),
        _source(),
        _validation_batches,
        _step,
        config,
        provenance={"manifest_hash": "handoff-manifest"},
        maximum_optimizer_updates=2,
        optimizer_policies=_policies(),
    )
    last = load_reconstructor_curriculum_checkpoint(tmp_path / "handoff" / "last.pt")
    selected = load_reconstructor_curriculum_checkpoint(
        tmp_path / "handoff" / "best_phase1_root.pt", require_selected=True
    )
    assert last["curriculum_state_dict"]["stage_index"] == 1
    assert last["optimizer_state_dict"]["state"] == {}
    for name, value in selected["model_state_dict"].items():
        assert torch.equal(last["online_model_state_dict"][name], value)
