"""Campaign isolation and reporting for the bias-free, matched-seed comparison."""
from __future__ import annotations

import copy
import importlib.util
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace

import pytest

from part_ev_fullscale import protocol as P


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_part_fullscale.py"
    spec = importlib.util.spec_from_file_location("part_fullscale_plain_protocol_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def campaign_factory(monkeypatch):
    environment = {"fixture": "same pinned environment"}
    monkeypatch.setattr(P, "environment", lambda: copy.deepcopy(environment))

    def make(output, comparison="standard", *, legacy=False):
        config = Path(P.__file__).resolve().parents[1] / "particle_transformer/data/JetClass/JetClass_full.yaml"
        payload = {
            "schema": P.SCHEMA,
            "recipe": P.recipe_for(comparison),
            "tasks": P.tasks_for(comparison),
            "environment": copy.deepcopy(environment),
            "inventory": {"splits": {
                "train": {"event_count": 100000000},
                "val": {"event_count": 5000000},
            }},
            "data_config_sha256": P.file_sha256(config),
        }
        if not legacy:
            payload["comparison"] = comparison
        payload["content_hash"] = P.digest(payload)
        P.write_json(output / "campaign.json", payload)
        return payload

    return make


def resign(output, payload):
    payload = copy.deepcopy(payload)
    payload.pop("content_hash", None)
    payload["content_hash"] = P.digest(payload)
    P.write_json(output / "campaign.json", payload)


def test_plain_protocol_changes_only_pair_bias_and_architecture_names():
    assert P.COMPARISONS["standard"] == ("part", "part_ev")
    assert P.COMPARISONS["plain"] == ("part_plain", "part_plain_ev")
    assert P.tasks_for() == P.TASKS
    assert P.recipe_for() == P.RECIPE
    assert P.recipe_for("plain") == {**P.RECIPE, "pair_bias": "none"}
    for comparison, architectures in P.COMPARISONS.items():
        assert P.tasks_for(comparison) == [
            {"architecture": architecture, "seed": seed}
            for architecture in architectures for seed in (101, 202, 303)
        ]
    assert P.RECIPE["pair_bias"] == "shared"


def test_recipe_and_task_helpers_return_independent_objects():
    first_recipe = P.recipe_for("plain")
    first_recipe["betas"][0] = 0
    first_recipe["lr_decay_epochs"].clear()
    first_tasks = P.tasks_for("plain")
    first_tasks[0]["seed"] = 999
    first_tasks.clear()
    assert P.recipe_for("plain")["betas"] == [0.95, 0.999]
    assert P.recipe_for("plain")["lr_decay_epochs"] == list(range(35, 50))
    assert P.tasks_for("plain")[0] == {"architecture": "part_plain", "seed": 101}
    assert P.RECIPE["betas"] == [0.95, 0.999]


@pytest.mark.parametrize("helper", ["recipe_for", "tasks_for"])
def test_unknown_comparison_is_rejected(helper):
    with pytest.raises(ValueError):
        getattr(P, helper)("unrecognized")


@pytest.mark.parametrize("comparison,legacy", [("standard", True), ("standard", False), ("plain", False)])
def test_campaign_loading_is_read_only_and_backward_compatible(tmp_path, campaign_factory, comparison, legacy):
    expected = campaign_factory(tmp_path, comparison, legacy=legacy)
    before = (tmp_path / "campaign.json").read_bytes()
    assert P.load_campaign(tmp_path) == expected
    assert (tmp_path / "campaign.json").read_bytes() == before


@pytest.mark.parametrize("alteration", ["bias", "tasks", "seed", "task_order", "comparison", "unknown_comparison"])
def test_resigned_inconsistent_plain_campaign_is_rejected(tmp_path, campaign_factory, alteration):
    payload = campaign_factory(tmp_path, "plain")
    if alteration == "bias":
        payload["recipe"]["pair_bias"] = "shared"
    elif alteration == "tasks":
        payload["tasks"] = P.tasks_for("standard")
    elif alteration == "seed":
        payload["tasks"][0]["seed"] = 999
    elif alteration == "task_order":
        payload["tasks"].reverse()
    elif alteration == "comparison":
        payload["comparison"] = "standard"
    else:
        payload["comparison"] = "unrecognized"
    resign(tmp_path, payload)
    with pytest.raises(ValueError):
        P.load_campaign(tmp_path)


def test_legacy_campaign_cannot_silently_become_plain(tmp_path, campaign_factory):
    payload = campaign_factory(tmp_path, "standard", legacy=True)
    payload["recipe"] = P.recipe_for("plain")
    payload["tasks"] = P.tasks_for("plain")
    resign(tmp_path, payload)
    with pytest.raises(ValueError):
        P.load_campaign(tmp_path)


def test_campaign_checksum_still_protects_comparison(tmp_path, campaign_factory):
    payload = campaign_factory(tmp_path, "plain")
    payload["comparison"] = "standard"
    P.write_json(tmp_path / "campaign.json", payload)
    with pytest.raises(ValueError, match="changed|incompatible"):
        P.load_campaign(tmp_path)


def test_plain_task_paths_do_not_alias_existing_standard_runs(tmp_path):
    plain = P.tasks_for("plain")
    for index, task in enumerate(plain):
        expected = tmp_path / "runs" / task["architecture"] / f"seed_{task['seed']}"
        assert P.task_directory(tmp_path, index, tasks=plain) == expected
        assert P.task_directory(tmp_path, index) != expected
    assert P.task_directory(tmp_path, 0) == tmp_path / "runs/part/seed_101"
    for invalid in (-1, 6):
        with pytest.raises(ValueError):
            P.task_directory(tmp_path, invalid, tasks=plain)


@pytest.mark.parametrize("comparison", ["standard", "plain"])
def test_prepare_is_idempotent_but_cannot_repurpose_campaign(
    tmp_path, monkeypatch, driver, comparison
):
    from part_ev_fullscale import data, model

    environment = {"fixture": "pinned environment"}
    inspected = []

    def inspect(directory, *, check_entries):
        inspected.append((directory, check_entries))
        return {"splits": {
            "train": {"event_count": 100000000}, "val": {"event_count": 5000000},
        }}

    monkeypatch.setattr(P, "environment", lambda: dict(environment))
    monkeypatch.setattr(data, "inspect_dataset", inspect)
    monkeypatch.setattr(model, "reference_metadata", lambda: {})
    args = SimpleNamespace(output=tmp_path, data_root=tmp_path / "data", comparison=comparison)
    driver.prepare(args)
    before = (tmp_path / "campaign.json").read_bytes()
    payload = P.load_campaign(tmp_path)
    assert payload.get("comparison", "standard") == comparison
    assert payload["tasks"] == P.tasks_for(comparison)
    assert payload["recipe"] == P.recipe_for(comparison)
    assert payload["test_is_opt_in"] is True
    driver.prepare(args)
    assert (tmp_path / "campaign.json").read_bytes() == before
    assert inspected == [(args.data_root, True)]

    other = "plain" if comparison == "standard" else "standard"
    with pytest.raises(ValueError):
        driver.prepare(SimpleNamespace(output=tmp_path, data_root=args.data_root, comparison=other))
    assert (tmp_path / "campaign.json").read_bytes() == before
    assert inspected == [(args.data_root, True)]


def test_legacy_prepare_call_keeps_standard_default(tmp_path, monkeypatch, driver, campaign_factory):
    campaign_factory(tmp_path, "standard", legacy=True)
    before = (tmp_path / "campaign.json").read_bytes()
    driver.prepare(SimpleNamespace(output=tmp_path, data_root=tmp_path / "data"))
    assert (tmp_path / "campaign.json").read_bytes() == before


def test_plain_training_requires_its_own_matching_benchmark(tmp_path, driver):
    runtime = {"world_size": 1, "gpu": "GH200", "campaign_sha256": "plain-campaign"}
    expected = P.COMPARISONS["plain"]
    path = tmp_path / "benchmarks/gpus_1.json"
    with pytest.raises(RuntimeError, match="benchmark"):
        driver.require_benchmark(tmp_path, runtime, architectures=expected)
    for models in (
        {"part": {}, "part_ev": {}},
        {"part_plain": {}},
        {"part_plain": {}, "part_plain_ev": {}, "part": {}},
    ):
        P.write_json(path, {"runtime": runtime, "passed": True, "models": models})
        with pytest.raises(ValueError):
            driver.require_benchmark(tmp_path, runtime, architectures=expected)
    P.write_json(path, {"runtime": runtime, "passed": True, "models": {name: {} for name in expected}})
    driver.require_benchmark(tmp_path, runtime, architectures=expected)
    with pytest.raises(ValueError):
        driver.require_benchmark(tmp_path, runtime)


@pytest.mark.parametrize("comparison", ["standard", "plain"])
def test_report_uses_campaign_models_and_matched_seed_gains(tmp_path, driver, campaign_factory, comparison):
    spec = campaign_factory(tmp_path, comparison)
    accuracies = [0.80, 0.81, 0.82, 0.82, 0.84, 0.83]
    for index, (task, accuracy) in enumerate(zip(spec["tasks"], accuracies)):
        record = {
            "task": task,
            "campaign_sha256": spec["content_hash"],
            "checkpoint_sha256": f"checkpoint-{index}",
            "inventory_sha256": "same-test-population",
            "metrics": {
                "events": 20000000, "accuracy": accuracy,
                "macro_auc_ovo": accuracy + 0.1, "macro_auc_ovr": accuracy + 0.12,
                "qcd_rejection": {"Hbb": {"0.5": {"rejection": 100.0 + index}}},
            },
        }
        directory = P.task_directory(tmp_path, index, tasks=spec["tasks"])
        P.write_json(directory / "test/metrics.json", record)
    driver.report(SimpleNamespace(output=tmp_path))
    summary = P.read_json(tmp_path / "reports/fullscale_report.json")
    baseline, ev = P.COMPARISONS[comparison]
    assert set(summary["models"]) == {baseline, ev}
    assert summary["models"][baseline]["accuracy"]["mean"] == pytest.approx(0.81)
    assert summary["models"][ev]["accuracy"]["mean"] == pytest.approx(0.83)
    assert summary["paired_seed_accuracy_gains"] == pytest.approx([0.02, 0.03, 0.01])
    assert summary["mean_accuracy_gain"] == pytest.approx(0.02)
    assert summary["models"][ev]["qcd_rejection"]["Hbb"]["0.5"]["per_seed"] == {
        "101": 103.0, "202": 104.0, "303": 105.0,
    }
    assert [row["task"] for row in summary["per_seed_results"]] == spec["tasks"]


def test_plain_report_never_reads_standard_result_paths(tmp_path, driver, campaign_factory):
    spec = campaign_factory(tmp_path, "plain")
    for index, task in enumerate(P.tasks_for("standard")):
        P.write_json(P.task_directory(tmp_path, index) / "test/metrics.json", {
            "task": task, "campaign_sha256": spec["content_hash"], "metrics": {"events": 20000000},
        })
    with pytest.raises(FileNotFoundError):
        driver.report(SimpleNamespace(output=tmp_path))
    assert not (tmp_path / "reports/fullscale_report.json").exists()


@pytest.mark.parametrize("comparison", ["standard", "plain"])
def test_benchmark_runs_only_the_frozen_campaign_architectures(
    tmp_path, monkeypatch, driver, campaign_factory, comparison
):
    import torch
    from part_ev_fullscale import data, optimizer

    spec = campaign_factory(tmp_path, comparison)
    runtime = {"world_size": 1, "gpu": "fixture", "campaign_sha256": spec["content_hash"]}
    made = []
    closed = []
    batch = (torch.ones(2, 1), torch.zeros(2, dtype=torch.long))

    def model(architecture, seed, device, world):
        made.append((architecture, seed, world))
        result = torch.nn.Linear(1, 10)
        return result, result

    monkeypatch.setattr(driver, "distributed_setup", lambda: (0, 1, torch.device("cpu")))
    monkeypatch.setattr(driver, "runtime_contract", lambda *args: runtime)
    monkeypatch.setattr(driver, "make_model", model)
    monkeypatch.setattr(driver, "train_step", lambda *args: (1.0, False))
    monkeypatch.setattr(driver, "inputs_and_labels", lambda batch, device: ([batch[0]], batch[1]))
    monkeypatch.setattr(data, "build_loader", lambda *args, **kwargs: (repeat(batch), None))
    monkeypatch.setattr(data, "shutdown_loader", lambda loader: closed.append(loader))
    monkeypatch.setattr(optimizer, "Ranger", lambda parameters: object())
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args: None)
    for method in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, method, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    driver.benchmark(SimpleNamespace(output=tmp_path, warmup=1, steps=2))
    expected = P.COMPARISONS[comparison]
    assert made == [(architecture, 101, 1) for architecture in expected]
    assert len(closed) == 2
    result = P.read_json(tmp_path / "benchmarks/gpus_1.json")
    assert set(result["models"]) == set(expected)
    assert result["passed"] is True
    driver.require_benchmark(tmp_path, runtime, architectures=expected)


@pytest.mark.parametrize("task_index", [0, 3])
def test_final_test_reuse_locks_all_six_plain_checkpoints(
    tmp_path, monkeypatch, driver, campaign_factory, task_index
):
    import torch

    spec = campaign_factory(tmp_path, "plain")
    expected = []
    for index, task in enumerate(spec["tasks"]):
        directory = P.task_directory(tmp_path, index, tasks=spec["tasks"])
        # Fast-path test: checkpoint bytes are hashed, never loaded as a model.
        checkpoint = directory / "best_model_val.pt"
        P.write_json(checkpoint, {"fixture_checkpoint": index})
        sha = P.file_sha256(checkpoint)
        P.write_json(directory / "training_complete.json", {
            "campaign_sha256": spec["content_hash"], "step": 1000000,
            "validated_epochs": 50, "best_checkpoint_sha256": sha, "best_step": 800000,
        })
        expected.append({"task": task, "sha256": sha, "step": 800000})
    chosen = P.task_directory(tmp_path, task_index, tasks=spec["tasks"])
    P.write_json(chosen / "test/metrics.json", {"fixture_already_complete": True})
    monkeypatch.setattr(driver, "distributed_setup", lambda: (0, 1, torch.device("cpu")))
    monkeypatch.setattr(driver, "inspect_test", lambda *args: {})

    driver.test(SimpleNamespace(output=tmp_path, task_index=task_index, test_data=tmp_path / "test_data"))
    assert P.read_json(tmp_path / "locked_finalists.json") == expected
    assert not (tmp_path / "runs/part").exists()
    assert not (tmp_path / "runs/part_ev").exists()


@pytest.mark.parametrize("comparison,legacy", [("standard", True), ("plain", False)])
def test_timing_routes_plain_models_and_preserves_legacy_benchmark_api(
    tmp_path, monkeypatch, driver, campaign_factory, comparison, legacy
):
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_part_fullscale_timing.py"
    imported = importlib.util.spec_from_file_location("part_plain_protocol_timing", path)
    timing = importlib.util.module_from_spec(imported)
    imported.loader.exec_module(timing)
    spec = campaign_factory(tmp_path, comparison, legacy=legacy)
    output = tmp_path / "diagnostics/timing.fixture"
    output.mkdir(parents=True)
    expected = P.COMPARISONS[comparison]
    measured = []
    benchmark_calls = []

    if legacy:
        def require_benchmark(campaign, runtime):
            benchmark_calls.append((campaign, runtime))
    else:
        def require_benchmark(campaign, runtime, architectures):
            benchmark_calls.append((campaign, runtime, architectures))

    runtime = {"fixture": "same GPU runtime"}
    fake_driver = SimpleNamespace(
        P=P, distributed_setup=lambda: (0, 1, "cpu"),
        runtime_contract=lambda *args: runtime,
        require_benchmark=require_benchmark, STOP_REQUESTED=False,
        progress=lambda *args, **kwargs: None,
    )

    def probe(driver, campaign, args, architecture, deadline, row, persist):
        measured.append(architecture)
        row.update({
            "training_windows": [{"steps": 1000, "seconds": 202.0,
                                  "checkpoint_seconds": 2.0} for _ in range(10)],
            "full_checkpoint_seconds": [2.0] * 12, "best_checkpoint_seconds": [1.0],
            "validation": {"events": 1280000, "seconds": 100.0},
            "setup_seconds": 10.0, "complete": True,
        })

    monkeypatch.setattr(timing, "load_driver", lambda *args: fake_driver)
    monkeypatch.setattr(timing, "probe_model", probe)
    assert timing.main(["--campaign", str(tmp_path), "--output", str(output)]) == 0
    assert measured == list(expected)
    assert len(benchmark_calls) == 1
    assert len(benchmark_calls[0]) == (2 if legacy else 3)
    if not legacy:
        assert benchmark_calls[0][2] == expected
    result = P.read_json(output / "report.json")
    assert set(result["models"]) == set(expected)
    assert result["complete"] and result["usable"]
    assert result["campaign_sha256"] == spec["content_hash"]
    assert result["changes_production"] is False
