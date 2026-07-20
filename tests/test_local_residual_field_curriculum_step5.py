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
    path.write_text(
        json.dumps(
            {
                "contract": LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
                "selected_consumer_id": consumer,
                "selected_alpha_endpoint": endpoint,
                "selection_source": "D_alpha_eval_Ofull,D_alpha_eval_Orobust",
                "selection_reason": "smoother response",
                "model_val_alpha_curve": {"0.25": 0.78, "0.50": 0.79},
                "stack_val_alpha_curve": {"0.25": 0.779, "0.50": 0.788},
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


def test_step5_stage1b_requires_selected_consumer_unless_paired_mode():
    with pytest.raises(ValueError, match="selected_consumer.json"):
        LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
            {"alpha_schedule": ALPHA_SCHEDULE_PIECEWISE},
            require_selected_consumer=True,
            env={},
        )

    config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {"alpha_schedule": ALPHA_SCHEDULE_PIECEWISE},
        require_selected_consumer=True,
        env={"LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS": "1"},
    )

    assert config.paired_consumer_mode is True
    assert paired_consumers_confirmed_from_env(
        {"LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS": "confirmed"}
    )


def test_step5_teacher_sequence_is_sorted_and_inherits_alpha_endpoint(tmp_path: Path):
    selected_path = _write_selected_consumer(tmp_path / "selected_consumer.json", consumer="Ofull", endpoint=0.75)
    config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {
            "alpha_schedule": ALPHA_SCHEDULE_PIECEWISE,
            "fixed_alpha": 0.25,
            "piecewise_alpha": [(0, 0.25), (3, "selected_endpoint")],
            "teacher_sequence": [
                {"epoch": 4, "consumer_id": "Orobust_light"},
                {"epoch": 0, "consumer_id": "Ofull"},
            ],
        },
        selected_consumer_path=selected_path,
        require_selected_consumer=True,
    )
    scheduler = LocalResidualFieldCurriculumScheduler(config, total_epochs=6)

    assert scheduler.teacher_for_epoch(0)["consumer_id"] == "Ofull"
    assert scheduler.teacher_for_epoch(5)["consumer_id"] == "Orobust_light"
    assert scheduler.alpha_for_epoch(5) == pytest.approx(0.75)


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
