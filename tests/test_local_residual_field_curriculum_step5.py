from __future__ import annotations

import json
from pathlib import Path

import pytest

from teacher_logit_reco.local_particle_residual_field.curriculum import (
    LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
    ALPHA_SCHEDULE_PIECEWISE,
    ALPHA_SCHEDULE_SIGMOID,
    LocalResidualFieldCurriculumScheduler,
    LocalResidualFieldCurriculumSchedulerConfig,
    load_selected_consumer_record,
    paired_consumers_confirmed_from_env,
)


def _write_selected_consumer(path: Path, *, consumer: str = "Orobust_light", endpoint: float = 0.5) -> Path:
    endpoint_key = f"{float(endpoint):.2f}"
    path.write_text(
        json.dumps(
            {
                "contract": LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
                "selected_consumer_id": consumer,
                "selected_alpha_endpoint": endpoint,
                "selection_source": "D_alpha_eval_Ofull,D_alpha_eval_Orobust",
                "selection_reason": "smoother response",
                "model_val_alpha_curve": {"0.25": 0.78, endpoint_key: 0.79},
                "stack_val_alpha_curve": {"0.25": 0.779, endpoint_key: 0.788},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_step5_selected_consumer_file_drives_piecewise_alpha_and_report(tmp_path: Path):
    selected_path = _write_selected_consumer(tmp_path / "selected_consumer.json")

    config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {
            "alpha_schedule": ALPHA_SCHEDULE_PIECEWISE,
            "fixed_alpha": 0.25,
            "piecewise_alpha": [
                {"epoch": 0, "alpha": 0.25},
                {"epoch": 2, "alpha": "selected_endpoint"},
            ],
            "loss_weights": {"ce": 1.0, "oracle_path": 0.0, "field": 0.2},
            "loss_weight_schedule": {
                "oracle_path": {
                    "type": "linear",
                    "start": 0.0,
                    "end": 1.0,
                    "start_epoch": 1,
                    "end_epoch": 3,
                }
            },
        },
        selected_consumer_path=selected_path,
        require_selected_consumer=True,
    )
    scheduler = LocalResidualFieldCurriculumScheduler(config, total_epochs=4)

    assert scheduler.alpha_for_epoch(0) == pytest.approx(0.25)
    assert scheduler.alpha_for_epoch(2) == pytest.approx(0.5)
    epoch3 = scheduler.state_for_epoch(3)
    assert epoch3["active_consumer_id"] == "Orobust_light"
    assert epoch3["selected_consumer_id"] == "Orobust_light"
    assert epoch3["selected_alpha_endpoint"] == pytest.approx(0.5)
    assert epoch3["teacher"]["alpha"] == pytest.approx(0.5)
    assert epoch3["loss_weights"]["oracle_path"] == pytest.approx(1.0)

    report_path = tmp_path / "schedule_report.json"
    report = scheduler.write_report(report_path)
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == loaded
    assert loaded["selected_consumer"]["selected_consumer_id"] == "Orobust_light"
    assert loaded["epochs"][2]["alpha"] == pytest.approx(0.5)
    assert len(loaded["selected_consumer_hash"]) == 64
    assert loaded["selected_consumer_source_report"] == "D_alpha_eval_Ofull,D_alpha_eval_Orobust"
    assert len(loaded["config_hash"]) == 64
    assert loaded["epoch_count"] == 4
    assert all("alpha" in row and "loss_weights" in row for row in loaded["epochs"])
    assert all(row["selected_consumer_source_report"] for row in loaded["epochs"])


def test_step5_stage1b_requires_selected_consumer_unless_paired_mode():
    with pytest.raises(ValueError, match="selected_consumer.json"):
        LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
            {"alpha_schedule": ALPHA_SCHEDULE_PIECEWISE},
            require_selected_consumer=True,
            env={},
        )

    config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {
            "alpha_schedule": "fixed_alpha",
            "fixed_alpha": 0.25,
            "selected_consumer_id": "Ofull",
        },
        require_selected_consumer=True,
        env={"LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS": "1"},
    )

    assert config.paired_consumer_mode is True
    assert paired_consumers_confirmed_from_env(
        {"LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS": "confirmed"}
    )


def test_step5_teacher_sequence_is_sorted_and_inherits_alpha_endpoint(tmp_path: Path):
    config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {
            "alpha_schedule": ALPHA_SCHEDULE_PIECEWISE,
            "fixed_alpha": 0.25,
            "piecewise_alpha": [(0, 0.25), (3, "selected_endpoint")],
            "selected_consumer_id": "Ofull",
            "selected_alpha_endpoint": 0.75,
            "teacher_sequence": [
                {"epoch": 4, "consumer_id": "Orobust_light"},
                {"epoch": 0, "consumer_id": "Ofull"},
            ],
        },
        require_selected_consumer=True,
        env={"LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS": "1"},
    )
    scheduler = LocalResidualFieldCurriculumScheduler(config, total_epochs=6)

    assert scheduler.teacher_for_epoch(0)["consumer_id"] == "Ofull"
    assert scheduler.teacher_for_epoch(5)["consumer_id"] == "Orobust_light"
    assert scheduler.alpha_for_epoch(5) == pytest.approx(0.75)
    assert scheduler.state_for_epoch(5)["teacher"]["alpha"] == pytest.approx(0.75)
    assert scheduler.state_for_epoch(5)["paired_consumer_mode"] is True


def test_step5_sigmoid_alpha_uses_selected_endpoint(tmp_path: Path):
    selected_path = _write_selected_consumer(tmp_path / "selected_consumer.json", consumer="Ofull", endpoint=0.75)
    config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {
            "alpha_schedule": ALPHA_SCHEDULE_SIGMOID,
            "sigmoid_alpha_start": 0.25,
            "sigmoid_alpha_midpoint": 0.5,
            "sigmoid_alpha_sharpness": 16.0,
        },
        selected_consumer_path=selected_path,
        require_selected_consumer=True,
    )
    scheduler = LocalResidualFieldCurriculumScheduler(config, total_epochs=7)

    values = [scheduler.alpha_for_epoch(epoch) for epoch in range(7)]
    assert values[0] < values[3] < values[-1]
    assert values[-1] <= 0.75
    assert values[-1] > 0.70
    record = load_selected_consumer_record(selected_path)
    assert record.selected_consumer_id == "Ofull"
    assert record.selected_alpha_endpoint == pytest.approx(0.75)


def test_step5_selector_artifact_validation_is_strict_and_preserves_zero_endpoint(tmp_path: Path):
    selected_path = _write_selected_consumer(tmp_path / "selected_consumer.json", endpoint=0.0)
    record = load_selected_consumer_record(selected_path)
    assert record.selected_alpha_endpoint == pytest.approx(0.0)
    assert len(record.source_hash) == 64

    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    payload.pop("selection_reason")
    selected_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_selected_consumer_record(selected_path)

    payload["selection_reason"] = "restored"
    payload["contract"] = "wrong_contract"
    selected_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selected consumer contract"):
        load_selected_consumer_record(selected_path)


def test_step5_scheduler_rejects_guessed_consumer_or_alpha_and_changed_selector(tmp_path: Path):
    guessed = LocalResidualFieldCurriculumSchedulerConfig(
        alpha_schedule="fixed_alpha",
        fixed_alpha=0.25,
        selected_consumer_id="Ofull",
    )
    with pytest.raises(ValueError, match="must read selected_consumer.json"):
        LocalResidualFieldCurriculumScheduler(guessed, total_epochs=2)

    paired_without_alpha = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {
            "alpha_schedule": "fixed_alpha",
            "selected_consumer_id": "Ofull",
        },
        env={"LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS": "1"},
    )
    with pytest.raises(ValueError, match="requires an explicit fixed or selected alpha"):
        LocalResidualFieldCurriculumScheduler(paired_without_alpha, total_epochs=2)

    selected_path = _write_selected_consumer(tmp_path / "selected_consumer.json")
    selected = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {"alpha_schedule": "fixed_alpha"},
        selected_consumer_path=selected_path,
    )
    selected_path.write_text(selected_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after scheduler configuration"):
        LocalResidualFieldCurriculumScheduler(selected, total_epochs=2)


def test_step5_nonpaired_sequence_cannot_override_selected_consumer(tmp_path: Path):
    selected_path = _write_selected_consumer(tmp_path / "selected_consumer.json", consumer="Ofull")
    with pytest.raises(ValueError, match="may only use selected consumer Ofull"):
        LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
            {
                "alpha_schedule": "fixed_alpha",
                "teacher_sequence": [
                    {"epoch": 0, "consumer_id": "Ofull"},
                    {"epoch": 2, "consumer_id": "Orobust_light"},
                ],
            },
            selected_consumer_path=selected_path,
        )


def test_step5_loss_weight_schedules_and_epoch_report_are_complete(tmp_path: Path):
    selected_path = _write_selected_consumer(tmp_path / "selected_consumer.json", endpoint=0.75)
    config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {
            "alpha_schedule": "fixed_alpha",
            "loss_weights": {"ce": 1.0, "field": 0.2, "oracle_path": 0.0, "gate": 0.0},
            "loss_weight_schedule": {
                "field": {"type": "fixed", "value": 0.1},
                "oracle_path": {
                    "type": "piecewise",
                    "points": [(0, 0.0), (2, 0.5)],
                },
                "gate": {
                    "type": "sigmoid",
                    "start": 0.0,
                    "end": 0.2,
                    "midpoint": 0.5,
                    "sharpness": 10.0,
                },
                "ce": {
                    "type": "linear",
                    "start": 1.0,
                    "end": 0.8,
                    "start_epoch": 0,
                    "end_epoch": 3,
                },
            },
        },
        selected_consumer_path=selected_path,
    )
    scheduler = LocalResidualFieldCurriculumScheduler(config, total_epochs=4)
    rows = scheduler.epoch_report()

    assert len(rows) == 4
    assert all(set(row["loss_weights"]) == {"ce", "field", "gate", "oracle_path"} for row in rows)
    assert rows[0]["loss_weights"]["field"] == pytest.approx(0.1)
    assert rows[2]["loss_weights"]["oracle_path"] == pytest.approx(0.5)
    assert rows[0]["loss_weights"]["ce"] > rows[-1]["loss_weights"]["ce"]
    assert rows[0]["loss_weights"]["gate"] < rows[-1]["loss_weights"]["gate"]

    with pytest.raises(ValueError, match="epochs must be unique"):
        LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
            {
                "alpha_schedule": "piecewise_alpha",
                "piecewise_alpha": [(0, 0.25), (0, 0.5)],
            },
            selected_consumer_path=selected_path,
        )
