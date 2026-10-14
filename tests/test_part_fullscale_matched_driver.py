"""Isolated three-seed capacity-control campaign routing and honest reporting."""
from __future__ import annotations

import importlib.util
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from part_ev_fullscale import protocol as P
from part_ev_fullscale.model import build_model, parameter_match_metadata


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_part_fullscale.py"
    imported = importlib.util.spec_from_file_location("matched_control_driver", path)
    module = importlib.util.module_from_spec(imported)
    imported.loader.exec_module(module)
    return module


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    environment = {"fixture": "pinned"}
    monkeypatch.setattr(P, "environment", lambda: environment)
    config = Path(P.__file__).resolve().parents[1] / "particle_transformer/data/JetClass/JetClass_full.yaml"
    spec = {
        "schema": P.SCHEMA, "comparison": "parameter_matched", "training_recipe": "original",
        "tasks": P.tasks_for("parameter_matched"), "recipe": P.recipe_for("parameter_matched"),
        "environment": environment, "inventory": {}, "data_config_sha256": P.file_sha256(config),
        "parameter_match": parameter_match_metadata(),
    }
    spec["content_hash"] = P.digest(spec)
    P.write_json(tmp_path / "campaign.json", spec)
    return spec


def test_prepare_freezes_control_only_and_preserves_existing_campaign(tmp_path, monkeypatch, driver):
    from part_ev_fullscale import data, model

    monkeypatch.setattr(P, "environment", lambda: {"fixture": "same environment"})
    monkeypatch.setattr(model, "reference_metadata", lambda: {})
    monkeypatch.setattr(data, "inspect_dataset", lambda *args, **kwargs: {
        "splits": {"train": {"event_count": 100000000}, "val": {"event_count": 5000000}}})
    args = SimpleNamespace(output=tmp_path, data_root=tmp_path / "data",
                           comparison="parameter_matched", training_recipe="original")
    driver.prepare(args)
    before = (tmp_path / "campaign.json").read_bytes()
    spec = P.load_campaign(tmp_path)
    assert spec["tasks"] == [{"architecture": "part_matched", "seed": s} for s in (101, 202, 303)]
    assert spec["parameter_match"] == parameter_match_metadata()
    assert spec["recipe"]["optimizer"] == "Weaver_RAdam_plus_Lookahead"
    assert spec["test_is_opt_in"]
    driver.prepare(args)
    assert (tmp_path / "campaign.json").read_bytes() == before
    args.comparison = "standard"
    with pytest.raises(ValueError, match="another comparison"):
        driver.prepare(args)
    assert (tmp_path / "campaign.json").read_bytes() == before


def test_real_model_count_is_checked_against_frozen_audit(driver, campaign):
    driver.validate_parameter_match(campaign, build_model("part_matched"))
    with pytest.raises(ValueError, match="parameter count"):
        driver.validate_parameter_match(campaign, build_model("part"))
    campaign["parameter_match"]["matched_parameters"] -= 1
    with pytest.raises(ValueError, match="architecture audit"):
        driver.validate_parameter_match(campaign)


@pytest.mark.parametrize("phase", ["benchmark", "train", "test", "report"])
def test_missing_architecture_audit_is_rejected_before_execution(tmp_path, monkeypatch, driver, campaign, phase):
    del campaign["parameter_match"]
    monkeypatch.setattr(P, "load_campaign", lambda *args: campaign)
    monkeypatch.setattr(driver, "distributed_setup", lambda: pytest.fail("Audit must precede GPU execution"))
    with pytest.raises(ValueError, match="architecture audit"):
        getattr(driver, phase)(SimpleNamespace(output=tmp_path, task_index=0))


@pytest.mark.parametrize("missing", [None, 1, 2])
def test_test_phase_locks_exactly_three_completed_checkpoints(tmp_path, monkeypatch, driver, campaign, missing):
    expected = []
    for index, task in enumerate(campaign["tasks"]):
        directory = P.task_directory(tmp_path, index, tasks=campaign["tasks"])
        P.write_json(directory / "best_model_val.pt", {"checkpoint_fixture": index})
        sha = P.file_sha256(directory / "best_model_val.pt")
        if missing != index:
            P.write_json(directory / "training_complete.json", {
                "campaign_sha256": campaign["content_hash"], "step": 1000000,
                "validated_epochs": 50, "best_step": 900000, "best_checkpoint_sha256": sha})
        P.write_json(directory / "test/metrics.json", {"already_complete_fixture": True})
        expected.append({"task": task, "sha256": sha, "step": 900000})
    monkeypatch.setattr(driver, "distributed_setup", lambda: (0, 1, torch.device("cpu")))
    inspected = []
    monkeypatch.setattr(driver, "inspect_test", lambda *args: inspected.append(True))
    args = SimpleNamespace(output=tmp_path, task_index=0, test_data=tmp_path / "test_data")
    if missing is not None:
        with pytest.raises(FileNotFoundError):
            driver.test(args)
        assert not inspected and not (tmp_path / "locked_finalists.json").exists()
    else:
        driver.test(args)
        assert P.read_json(tmp_path / "locked_finalists.json") == expected
        assert inspected == [True]


@pytest.mark.parametrize("phase", ["train", "test"])
@pytest.mark.parametrize("index", [-1, 3, 5])
def test_invalid_control_task_rejected_before_gpu(tmp_path, monkeypatch, driver, campaign, phase, index):
    monkeypatch.setattr(driver, "distributed_setup", lambda: pytest.fail("Must validate task before GPU setup"))
    with pytest.raises(ValueError):
        getattr(driver, phase)(SimpleNamespace(output=tmp_path, task_index=index))


def test_control_report_aggregates_three_seeds_without_inventing_comparisons(tmp_path, driver, campaign):
    for index, task in enumerate(campaign["tasks"]):
        metrics = {
            "events": 20000000, "accuracy": 0.86 + 0.001 * index,
            "macro_auc_ovo": 0.98 + 0.001 * index, "macro_auc_ovr": 0.99 + 0.001 * index,
            "qcd_rejection": {"Hbb": {"0.5": {"rejection": 10000 + 100 * index}}},
        }
        directory = P.task_directory(tmp_path, index, tasks=campaign["tasks"])
        P.write_json(directory / "test/metrics.json", {
            "task": task, "campaign_sha256": campaign["content_hash"],
            "inventory_sha256": "same-population", "metrics": metrics})
    driver.report(SimpleNamespace(output=tmp_path))
    report = P.read_json(tmp_path / "reports/fullscale_report.json")
    assert report["control_only"] and not report["exploratory_single_seed"]
    assert report["seeds"] == [101, 202, 303]
    assert report["paired_seeds"] == [] and report["paired_seed_accuracy_gains"] == []
    assert report["mean_accuracy_gain"] is None
    assert report["parameter_match"] == parameter_match_metadata()
    assert set(report["models"]) == {"part_matched"}
    row = report["models"]["part_matched"]
    assert row["seed_count"] == 3
    assert row["accuracy"]["mean"] == pytest.approx(0.861)
    assert row["accuracy"]["seed_std"] == pytest.approx(0.001)
    rejection = row["qcd_rejection"]["Hbb"]["0.5"]
    assert rejection["per_seed"] == {"101": 10000, "202": 10100, "303": 10200}
    assert rejection["mean"] == 10100 and rejection["seed_std"] == 100


def test_benchmark_builds_only_matched_control_and_uses_original_optimizer(tmp_path, monkeypatch, driver, campaign):
    from part_ev_fullscale import data
    from part_ev_fullscale.optimizer import Ranger

    calls, updates, audits = [], [], []
    batch = (torch.ones(2, 1), torch.zeros(2, dtype=torch.long))

    def make_model(architecture, seed, device, world):
        calls.append((architecture, seed, world))
        model = torch.nn.Linear(1, 10)
        return model, model

    def update(model, optimizer, scaler, batch, device):
        assert isinstance(optimizer, Ranger)
        assert optimizer.param_groups[0]["lr"] == 1e-3
        updates.append(True)  # No clipping/schedule keyword modifications.
        return 1.0, False

    # Real architecture audit has its own test; replace only for this tiny model.
    monkeypatch.setattr(driver, "validate_parameter_match", lambda spec, base=None: audits.append(base))
    monkeypatch.setattr(driver, "distributed_setup", lambda: (0, 1, torch.device("cpu")))
    monkeypatch.setattr(driver, "runtime_contract", lambda *args: {"world_size": 1})
    monkeypatch.setattr(driver, "make_model", make_model)
    monkeypatch.setattr(driver, "train_step", update)
    monkeypatch.setattr(driver, "inputs_and_labels", lambda batch, device: ([batch[0]], batch[1]))
    monkeypatch.setattr(data, "build_loader", lambda *args, **kwargs: (repeat(batch), None))
    monkeypatch.setattr(data, "shutdown_loader", lambda loader: None)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args: None)
    for method in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, method, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    driver.benchmark(SimpleNamespace(output=tmp_path, warmup=1, steps=2))
    assert calls == [("part_matched", 101, 1)] and len(updates) == 3
    assert audits[0] is None and isinstance(audits[1], torch.nn.Module)
    result = P.read_json(tmp_path / "benchmarks/gpus_1.json")
    assert result["passed"] and set(result["models"]) == {"part_matched"}
