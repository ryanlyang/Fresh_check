"""Isolated timing estimates must not authorize or modify production training."""
from __future__ import annotations

import copy
import importlib.util
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "part_timing_diagnostic", ROOT / "scripts/diagnose_part_fullscale_timing.py")
timing = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(timing)

RECIPE = {"training_population": 100000000, "fetch_step": 0.01,
          "global_batch_size": 512, "total_steps": 1000000,
          "epochs": 50, "validation_events_per_epoch": 1280000}


def measured():
    return {"training_windows": [{"steps": 1000, "seconds": 202.0,
                                   "checkpoint_seconds": 2.0} for _ in range(10)],
            "full_checkpoint_seconds": [2.0] * 12, "best_checkpoint_seconds": [1.0],
            "validation": {"events": 1280000, "seconds": 100.0},
            "setup_seconds": 10.0, "complete": True}


def test_projection_does_not_double_count_periodic_checkpoints():
    result = timing.project(measured(), RECIPE)
    assert result["usable"]
    # 200,000 compute/data seconds + 5000 validation + 2200 full saves + 50 best + 10 setup.
    assert result["observed_projection_hours"] == pytest.approx(207260 / 3600)
    assert result["projected_full_checkpoint_count"] == 1100
    assert result["projected_checkpoint_hours"] == pytest.approx(2250 / 3600)
    assert result["with_25_percent_slowdown_hours"] == pytest.approx(207260 / 3600 * 1.25)
    assert result["assessment"] == "72h_plausible_not_guaranteed"


@pytest.mark.parametrize("missing", ["short_train", "short_validation", "checkpoint", "error"])
def test_inadequate_or_failed_measurement_is_inconclusive(missing):
    row = measured()
    if missing == "short_train":
        row["training_windows"] = row["training_windows"][:3]
    elif missing == "short_validation":
        row["validation"]["events"] -= 512
    elif missing == "checkpoint":
        row["best_checkpoint_seconds"] = []
    else:
        row["error"] = "worker timeout"
    result = timing.project(row, RECIPE)
    assert not result["usable"] and result["reasons"]
    assert "observed_projection_hours" not in result


def test_cold_start_sensitivity_is_separate_and_72h_risk_is_reported():
    row = measured()
    row["training_windows"][0]["seconds"] += 100
    result = timing.project(row, RECIPE)
    assert result["excluding_first_training_window_hours"] < result["observed_projection_hours"]
    assert result["assessment"] == "72h_limited_headroom"
    for window in row["training_windows"]:
        window["seconds"] += 100
    assert timing.project(row, RECIPE)["assessment"] == "72h_at_risk"


def test_slow_later_windows_cannot_be_hidden_by_fast_startup():
    row = measured()
    row["training_windows"][0]["seconds"] = 2.1
    for window in row["training_windows"][1:]:
        window["seconds"] = 272
    result = timing.project(row, RECIPE)
    assert result["observed_projection_hours"] < 72
    assert result["excluding_first_training_window_hours"] > 72
    assert result["assessment"] == "72h_at_risk"
    assert result["planning_projection_hours"] == result["excluding_first_training_window_hours"]


def test_budget_limited_but_sufficient_sample_is_distinguished_from_completion():
    row = measured()
    row["training_windows"] = row["training_windows"][:8]
    row["complete"] = False
    assert timing.project(row, RECIPE)["usable"]


def test_partial_report_does_not_claim_both_models_are_measured(tmp_path):
    from part_ev_fullscale import protocol
    driver = SimpleNamespace(P=protocol)
    report = {"recipe": RECIPE, "models": {"part": measured()}}
    timing.write_report(driver, tmp_path, report)
    assert not report["usable"] and not report["complete"]
    assert (tmp_path / "report.json").is_file()
    assert "inconclusive" in (tmp_path / "report.md").read_text()
    report["models"]["part_ev"] = copy.deepcopy(measured())
    timing.write_report(driver, tmp_path, report)
    assert report["usable"] and report["complete"]


@pytest.mark.parametrize("args", [
    ["--budget-minutes", "101"], ["--train-steps", "0"], ["--validation-steps", "2501"]])
def test_cli_rejects_unbounded_diagnostics_before_touching_output(tmp_path, args):
    with pytest.raises(SystemExit) as exc:
        timing.main(["--campaign", str(tmp_path), "--output", str(tmp_path / "unused"), *args])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_production_output_is_not_allowed(tmp_path):
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "runs").mkdir()
    with pytest.raises(SystemExit):
        timing.main(["--campaign", str(tmp_path), "--output", str(tmp_path / "runs")])
    assert not (tmp_path / "runs/started.marker").exists()


@pytest.mark.parametrize("stop_requested", [False, True])
def test_probe_uses_isolated_checkpoints_and_cleans_them(tmp_path, monkeypatch, stop_requested):
    import torch
    from part_ev_fullscale import data, optimizer
    model = torch.nn.Linear(1, 10)
    batch = (torch.ones(512, 1), torch.zeros(512, dtype=torch.long))
    loaders = []
    class Loader:
        def __iter__(self):
            return repeat(batch)
    def build_loader(*args, **kwargs):
        loader = Loader()
        loaders.append(loader)
        return loader, None
    monkeypatch.setattr(data, "build_loader", build_loader)
    shutdown = []
    monkeypatch.setattr(data, "shutdown_loader", lambda loader: shutdown.append(loader))
    monkeypatch.setattr(optimizer, "Ranger", lambda params: SimpleNamespace(param_groups=[{}]))
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args: None)
    for method in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, method, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    checkpoints = []
    def save_checkpoint(path, *args):
        checkpoints.append(path)
        path.write_bytes(b"isolated measurement")
    driver = SimpleNamespace(
        make_model=lambda *args: (model, model),
        train_step=lambda *args: (1.0, False),
        inputs_and_labels=lambda batch, device: ([batch[0]], batch[1]),
        save_checkpoint=save_checkpoint, STOP_REQUESTED=stop_requested,
        P=SimpleNamespace(learning_rate=lambda step: 0.001), progress=lambda *args, **kwargs: None)
    args = SimpleNamespace(output=tmp_path, device=torch.device("cpu"), runtime={},
                           train_steps=1001, validation_steps=1)
    row = {"training_windows": [], "full_checkpoint_seconds": [], "best_checkpoint_seconds": []}
    timing.probe_model(driver, {"content_hash": "frozen", "inventory": {}}, args, "part", timing.time.monotonic() + 600,
                       row, lambda: None)
    if stop_requested:
        assert row["stop_reason"] == "signal_or_time_budget"
        assert not row["training_windows"] and not checkpoints
        assert len(shutdown) == 2
        return
    assert row["complete"]
    assert sum(w["steps"] for w in row["training_windows"]) == 1001
    assert row["validation"]["events"] == 512
    assert len(checkpoints) == 3  # Periodic, pre-validation, post-validation.
    assert all(path.parent == tmp_path for path in checkpoints)
    assert not list(tmp_path.glob("*.pt"))
    assert len(shutdown) == 2 and all(loader.timeout == 300 for loader in loaders)


@pytest.mark.parametrize("architecture", ["part", "part_ev"])
def test_adamw_probe_uses_frozen_optimizer_schedule_and_clipping(tmp_path, monkeypatch, architecture):
    import torch
    from part_ev_fullscale import data, optimizer, protocol

    model = torch.nn.Linear(1, 10)
    batch = (torch.ones(512, 1), torch.zeros(512, dtype=torch.long))
    recipe = protocol.recipe_for(training_recipe="adamw")
    spec = {"content_hash": "frozen_adamw", "inventory": {},
            "training_recipe": "adamw", "recipe": recipe}
    original_spec = copy.deepcopy(spec)
    selected_optimizer = SimpleNamespace(param_groups=[{}, {}])
    selected_scaler = object()
    factory_calls, schedule_steps, training_calls = [], [], []

    class Loader:
        def __iter__(self):
            return repeat(batch)

    monkeypatch.setattr(data, "build_loader", lambda *args, **kwargs: (Loader(), None))
    monkeypatch.setattr(data, "shutdown_loader", lambda loader: None)
    monkeypatch.setattr(optimizer, "Ranger", lambda *args: pytest.fail("AdamW must not use Ranger"))

    def build_optimizer(base, frozen_recipe):
        assert base is model and frozen_recipe is recipe
        factory_calls.append((base, frozen_recipe))
        return selected_optimizer

    monkeypatch.setattr(optimizer, "build_optimizer", build_optimizer)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args: selected_scaler)
    for method in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, method, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    def learning_rate(step, frozen_recipe):
        assert frozen_recipe is recipe
        schedule_steps.append(step)
        return protocol.learning_rate(step, frozen_recipe)

    def train_step(actual_model, actual_optimizer, scaler, actual_batch, device, *, gradient_clipping):
        assert actual_model is model and actual_optimizer is selected_optimizer
        assert scaler is selected_scaler and actual_batch is batch
        assert device.type == "cpu" and gradient_clipping == recipe["gradient_clipping"] == 1.0
        expected_lr = protocol.learning_rate(len(training_calls), recipe)
        assert [group["lr"] for group in actual_optimizer.param_groups] == [expected_lr, expected_lr]
        training_calls.append(expected_lr)
        return 1.0, False

    driver = SimpleNamespace(
        make_model=lambda *args: (model, model), train_step=train_step,
        inputs_and_labels=lambda actual_batch, device: ([actual_batch[0]], actual_batch[1]),
        save_checkpoint=lambda path, *args: path.write_bytes(b"diagnostic only"),
        STOP_REQUESTED=False, P=SimpleNamespace(learning_rate=learning_rate),
        progress=lambda *args, **kwargs: None)
    args = SimpleNamespace(output=tmp_path, device=torch.device("cpu"), runtime={},
                           train_steps=3, validation_steps=1)
    row = {"training_windows": [], "full_checkpoint_seconds": [], "best_checkpoint_seconds": []}

    timing.probe_model(driver, spec, args, architecture, timing.time.monotonic() + 600,
                       row, lambda: None)

    assert len(factory_calls) == 1
    assert schedule_steps == [step for step in range(3) for _ in selected_optimizer.param_groups]
    assert training_calls == [protocol.learning_rate(step, recipe) for step in range(3)]
    assert row["complete"] and row["validation"]["events"] == 512
    assert spec == original_spec
    assert not list(tmp_path.glob("*.pt"))


def test_preexisting_checkpoint_is_neither_overwritten_nor_deleted(tmp_path):
    checkpoint = tmp_path / "part_timing_checkpoint.pt"
    checkpoint.write_bytes(b"untouched")
    args = SimpleNamespace(output=tmp_path)
    with pytest.raises(FileExistsError):
        timing.probe_model(None, None, args, "part", 0, {}, lambda: None)
    assert checkpoint.read_bytes() == b"untouched"


def test_same_diagnostic_cannot_be_rerun_over_its_results(tmp_path, monkeypatch):
    output = tmp_path / "diagnostics/timing.unique"
    output.mkdir(parents=True)
    (output / "started.marker").write_text("existing")
    monkeypatch.setattr(timing, "load_driver", lambda *args: pytest.fail("Must not import before guard"))
    with pytest.raises(FileExistsError):
        timing.main(["--campaign", str(tmp_path), "--output", str(output)])
