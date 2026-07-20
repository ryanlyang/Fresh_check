from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from teacher_logit_reco.local_particle_residual_field import (
    LocalResidualFieldReportConfig,
    build_local_residual_field_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metrics(accuracy: float) -> dict:
    return {
        "accuracy": accuracy,
        "cross_entropy": 0.7,
        "loss": 0.7,
        "n_jets": 20,
        "attempted_jets": 20,
    }


def _selected(path: Path, consumer: str = "Ofull", endpoint: float = 0.75) -> None:
    curve = {
        "0.0": {"accuracy": 0.70, "cross_entropy": 0.75},
        "0.25": {"accuracy": 0.72, "cross_entropy": 0.72},
        str(endpoint): {"accuracy": 0.74, "cross_entropy": 0.69},
    }
    _write_json(
        path,
        {
            "contract": "local_residual_field_selected_consumer_v1",
            "selected_consumer_id": consumer,
            "selected_alpha_endpoint": endpoint,
            "selection_source": "D_alpha_eval_Ofull,D_alpha_eval_Orobust",
            "selection_reason": "smoothest useful model-val and stack-val response",
            "model_val_alpha_curve": curve,
            "stack_val_alpha_curve": curve,
        },
    )


def _oracle_curves(root: Path) -> None:
    for run_id, consumer, boost in (
        ("D_alpha_eval_Ofull", "Ofull", 0.04),
        ("D_alpha_eval_Orobust", "Orobust_light", 0.03),
    ):
        curve = {
            "0.0": {"accuracy": 0.70, "cross_entropy": 0.75, "teacher_entropy": 1.2},
            "0.25": {"accuracy": 0.70 + boost / 2, "cross_entropy": 0.72, "teacher_entropy": 1.1},
            "0.75": {"accuracy": 0.70 + boost, "cross_entropy": 0.68, "teacher_entropy": 1.0},
        }
        _write_json(
            root / run_id / "run_report.json",
            {
                "ok": True,
                "run_id": run_id,
                "consumer_id": consumer,
                "model_val_alpha_curve": curve,
                "stack_val_alpha_curve": curve,
                "deployable": False,
                "uses_true_fields": True,
            },
        )


def _schedule(endpoint: float, *, fixed: float | None = None, epochs: int = 7, consumer: str = "Ofull") -> list[dict]:
    rows = []
    for epoch in range(epochs):
        alpha = fixed
        if alpha is None:
            alpha = 0.25 if epoch < 3 else (0.50 if epoch < 6 else endpoint)
        rows.append(
            {
                "epoch": epoch,
                "alpha": alpha,
                "active_consumer_id": consumer,
                "selected_consumer_id": consumer,
                "selected_alpha_endpoint": endpoint,
            }
        )
    return rows


def _curriculum_run(
    root: Path,
    run_id: str,
    *,
    consumer: str = "Ofull",
    endpoint: float = 0.75,
    paired: bool = False,
    selected_hash: str | None = None,
    schedule: list[dict] | None = None,
) -> None:
    recipe = run_id.split("_", 1)[0]
    weights = {
        "ce": 1.0,
        "student_kd": 0.0,
        "oracle_path": 0.0 if recipe == "Q0" else 1.0,
        "field": 0.2,
        "gate": 0.05,
        "reg": 0.01,
    }
    report = {
        "ok": True,
        "contract": "local_residual_field_curriculum_train_v1",
        "run_id": run_id,
        "best_epoch": 1,
        "best_model_val": _metrics(0.75),
        "stack_val": _metrics(0.74),
        "final_test": _metrics(0.73),
        "checkpoint": str(root / run_id / "best_model_val.pt"),
        "selected_consumer_id": consumer,
        "selected_alpha_endpoint": endpoint,
        "teacher_used_during_training": consumer if recipe != "Q0" else None,
        "scientific_recipe_equivalent": True,
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "deployable": True,
        "resolved_run": {
            "run_id": recipe,
            "selected_consumer_id": consumer,
            "selected_alpha_endpoint": endpoint,
            "paired_consumer_mode": paired,
            "gate_mode": "learned_sigmoid",
            "student_init_source": "A0",
            "loss_weights": weights,
        },
    }
    _write_json(root / run_id / "run_report.json", report)
    states = schedule
    if states is None:
        states = _schedule(endpoint, consumer=consumer)
    if selected_hash is not None:
        states = [{**state, "selected_consumer_hash": selected_hash} for state in states]
    _write_json(root / run_id / "curriculum_schedule.json", {"epochs": states})
    _write_json(
        root / run_id / "training_curves.json",
        {
            "epochs": [
                {
                    "epoch": 0,
                    "schedule": {**states[0], "loss_weights": weights},
                    "freeze": {"phase": "residual_path_warmup"},
                    "train": _metrics(0.70),
                    "model_val": _metrics(0.71),
                }
            ]
        },
    )


def _legacy_rows(root: Path) -> None:
    for run_id, offline in (("A0", False), ("A4", True)):
        _write_json(
            root / run_id / "run_report.json",
            {
                "ok": True,
                "run_id": run_id,
                "best_model_val": _metrics(0.70 if run_id == "A0" else 0.80),
                "stack_val": _metrics(0.69 if run_id == "A0" else 0.79),
                "field_source": "hlt_only" if run_id == "A0" else "offline_particles",
                "runtime_inputs": "HLT_only" if run_id == "A0" else "offline_particles",
                "uses_offline_particles": offline,
                "deployable": not offline,
            },
        )


def _config(tmp_path: Path, **overrides) -> LocalResidualFieldReportConfig:
    values = {
        "output_dir": str(tmp_path / "report"),
        "tagger_root": str(tmp_path / "taggers"),
        "reconstructor_root": str(tmp_path / "reconstructors"),
        "curriculum_root": str(tmp_path / "curriculum"),
        "oracle_diagnostics_root": str(tmp_path / "oracles"),
        "required_tagger_run_ids": (),
        "required_reconstructor_run_ids": (),
        "required_fusion_groups": (),
        "confirm_final_test": True,
    }
    values.update(overrides)
    return LocalResidualFieldReportConfig(**values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_step9_curriculum_report_separates_rows_and_enforces_runtime_contract(tmp_path: Path) -> None:
    _legacy_rows(tmp_path / "taggers")
    _oracle_curves(tmp_path / "oracles")
    selected_path = tmp_path / "selected_consumer.json"
    _selected(selected_path)
    selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    for run_id in ("P7a", "Q0"):
        _curriculum_run(tmp_path / "curriculum", run_id, selected_hash=selected_hash)
    _curriculum_run(
        tmp_path / "curriculum",
        "Q3",
        selected_hash=selected_hash,
        schedule=_schedule(0.75, fixed=0.75, epochs=3),
    )
    _write_json(
        tmp_path / "curriculum" / "P7a" / "alpha_mix_diagnostics.json",
        {"rows": [{"run_id": "P7a", "split": "model_val", "lambda": 0.5, "accuracy": 0.73}]},
    )

    report = build_local_residual_field_report(
        _config(tmp_path, selected_consumer_json=str(selected_path), require_curriculum=True)
    )

    assert report["ok"] is True, report["problems"]
    output = tmp_path / "report"
    for filename in (
        "oracle_teacher_curve.csv",
        "curriculum_student_metrics.csv",
        "alpha_mix_diagnostics.csv",
        "teacher_student_agreement.csv",
        "field_error_where_oracle_helps.csv",
        "gate_calibration.csv",
        "consumer_selection.csv",
        "deployable_leaderboard.csv",
        "oracle_diagnostics.csv",
        "offline_reference.csv",
        "curriculum_training_diagnostics.csv",
    ):
        rows = _read_csv(output / filename)
        header = next(csv.reader((output / filename).open("r", encoding="utf-8")))
        for column in (
            "runtime_inputs",
            "uses_true_fields",
            "uses_offline_particles",
            "uses_teacher_logits_at_runtime",
            "deployable",
            "split",
            "selection_allowed",
            "baseline_label",
        ):
            assert column in header
        assert isinstance(rows, list)

    leaderboard = _read_csv(output / "deployable_leaderboard.csv")
    assert {row["run_id"] for row in leaderboard} >= {"A0", "P7a", "Q0", "Q3"}
    assert all(row["runtime_inputs"] == "HLT_only" and row["deployable"] == "True" for row in leaderboard)
    assert all(row["uses_true_fields"] == "False" for row in leaderboard)
    assert {row["run_id"] for row in _read_csv(output / "offline_reference.csv")} == {"A4"}
    assert {row["run_id"] for row in _read_csv(output / "consumer_selection.csv")} == {
        "D_alpha_eval_Ofull",
        "D_alpha_eval_Orobust",
    }
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "`O0`: zero-field oracle-consumer diagnostic" in summary


def test_step9_rejects_stage1b_without_selector(tmp_path: Path) -> None:
    _oracle_curves(tmp_path / "oracles")
    _curriculum_run(
        tmp_path / "curriculum",
        "P2",
        schedule=_schedule(0.75, fixed=0.25, epochs=3),
    )

    report = build_local_residual_field_report(_config(tmp_path))

    assert report["ok"] is False
    assert any("require selected_consumer.json" in problem for problem in report["problems"])


def test_step9_rejects_stage1b_that_did_not_persist_selector_hash(tmp_path: Path) -> None:
    _oracle_curves(tmp_path / "oracles")
    selected_path = tmp_path / "selected_consumer.json"
    _selected(selected_path)
    _curriculum_run(
        tmp_path / "curriculum",
        "P2",
        selected_hash="guessed-not-the-selector-hash",
        schedule=_schedule(0.75, fixed=0.25, epochs=3),
    )

    report = build_local_residual_field_report(_config(tmp_path, selected_consumer_json=str(selected_path)))

    assert report["ok"] is False
    assert any("did not persist the selected_consumer.json hash" in problem for problem in report["problems"])


def test_step9_rejects_q3_ramp_and_q0_recipe_drift(tmp_path: Path) -> None:
    _oracle_curves(tmp_path / "oracles")
    selected_path = tmp_path / "selected_consumer.json"
    _selected(selected_path)
    selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    _curriculum_run(tmp_path / "curriculum", "P7a", selected_hash=selected_hash)
    _curriculum_run(tmp_path / "curriculum", "Q0", selected_hash=selected_hash)
    q0_path = tmp_path / "curriculum" / "Q0" / "run_report.json"
    q0 = json.loads(q0_path.read_text(encoding="utf-8"))
    q0["resolved_run"]["loss_weights"]["field"] = 0.1
    _write_json(q0_path, q0)
    _curriculum_run(tmp_path / "curriculum", "Q3", selected_hash=selected_hash)

    report = build_local_residual_field_report(_config(tmp_path, selected_consumer_json=str(selected_path)))

    assert report["ok"] is False
    assert any("Q0 does not match selected P7a loss recipe for field" in problem for problem in report["problems"])
    assert any("Q3 must use selected_alpha_endpoint" in problem for problem in report["problems"])


def test_step9_paired_mode_writes_consumer_specific_tables(tmp_path: Path) -> None:
    _oracle_curves(tmp_path / "oracles")
    _curriculum_run(tmp_path / "curriculum", "P4_Ofull", consumer="Ofull", paired=True)
    _curriculum_run(
        tmp_path / "curriculum",
        "P4_Orobust_light",
        consumer="Orobust_light",
        paired=True,
        schedule=_schedule(0.75, consumer="Orobust_light"),
    )

    report = build_local_residual_field_report(_config(tmp_path, paired_consumer_mode=True))

    assert report["ok"] is True, report["problems"]
    for consumer in ("Ofull", "Orobust_light"):
        rows = _read_csv(tmp_path / "report" / f"curriculum_student_metrics_{consumer}.csv")
        assert rows
        assert {row["selected_consumer_id"] for row in rows} == {consumer}
        assert (tmp_path / "report" / f"deployable_leaderboard_{consumer}.csv").exists()
    assert {row["selected_consumer_id"] for row in _read_csv(tmp_path / "report" / "paired_consumer_comparison.csv")} == {
        "Ofull",
        "Orobust_light",
    }
