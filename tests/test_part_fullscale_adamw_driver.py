"""CPU checks of the AdamW pilot's clipping, gating, preparation and reporting."""
from __future__ import annotations

import contextlib
import importlib.util
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from part_ev_fullscale import protocol as P


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_part_fullscale.py"
    imported = importlib.util.spec_from_file_location("adamw_pilot_driver", path)
    module = importlib.util.module_from_spec(imported)
    imported.loader.exec_module(module)
    return module


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    environment = {"fixture": "pinned"}
    monkeypatch.setattr(P, "environment", lambda: environment)
    config = Path(P.__file__).resolve().parents[1] / "particle_transformer/data/JetClass/JetClass_full.yaml"
    spec = {
        "schema": P.SCHEMA, "comparison": "standard", "training_recipe": "adamw",
        "tasks": P.tasks_for(training_recipe="adamw"),
        "recipe": P.recipe_for(training_recipe="adamw"), "environment": environment,
        "inventory": {}, "data_config_sha256": P.file_sha256(config),
    }
    spec["content_hash"] = P.digest(spec)
    P.write_json(tmp_path / "campaign.json", spec)
    return spec


class RecordingScaler:
    """Real gradient scaling on CPU, with observable AMP call order."""
    def __init__(self):
        self.calls = []
        self.scale_value = 8.0
        self.unscaled = False
        self.nonfinite = False

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        self.calls.append("scale")
        return loss * self.scale_value

    def unscale_(self, optimizer):
        assert not self.unscaled
        self.calls.append("unscale")
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)
                    self.nonfinite |= not bool(torch.isfinite(parameter.grad).all())
        self.unscaled = True

    def step(self, optimizer):
        self.calls.append("step")
        if not self.unscaled:
            self.unscale_(optimizer)
        if not self.nonfinite:
            optimizer.step()

    def update(self):
        self.calls.append("update")
        if self.nonfinite:
            self.scale_value /= 2


@pytest.mark.parametrize("nonfinite", [False, True])
def test_clipping_unscales_first_and_retains_amp_nonfinite_skip(
    monkeypatch, driver, nonfinite
):
    torch.manual_seed(12)
    model = torch.nn.Linear(3, 10)
    values, truth = torch.ones(4, 3) * 100, torch.zeros(4, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scaler = RecordingScaler()
    before = [p.detach().clone() for p in model.parameters()]
    if nonfinite:
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    monkeypatch.setattr(driver, "inputs_and_labels", lambda batch, device: ([batch[0]], batch[1]))
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    clip = torch.nn.utils.clip_grad_norm_

    def checked_clip(parameters, maximum):
        assert scaler.calls == ["scale", "unscale"]
        scaler.calls.append("clip")
        parameters = list(parameters)
        result = clip(parameters, maximum)
        if not nonfinite:
            norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in parameters]))
            assert norm <= 1.00001
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", checked_clip)
    diagnostics = {}
    loss, skipped = driver.train_step(model, optimizer, scaler, (values, truth), "cpu",
                                      gradient_clipping=1.0, diagnostics=diagnostics)
    assert loss > 0
    assert scaler.calls == ["scale", "unscale", "clip", "step", "update"]
    assert skipped is nonfinite
    assert diagnostics["nonfinite_grad_norm"] is nonfinite
    if nonfinite:
        assert diagnostics["grad_norm_before_clip"] is None
        assert not diagnostics["gradient_clipped"]
        for actual, expected in zip(model.parameters(), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert diagnostics["gradient_clipped"]
        assert diagnostics["grad_norm_before_clip"] > 1
        assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), before))
    P.digest(diagnostics)  # No NaN/Infinity can leak into JSON progress records.


def test_original_train_step_does_not_add_clipping(monkeypatch, driver):
    model = torch.nn.Linear(3, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scaler = RecordingScaler()
    batch = (torch.ones(4, 3), torch.zeros(4, dtype=torch.long))
    monkeypatch.setattr(driver, "inputs_and_labels", lambda batch, device: ([batch[0]], batch[1]))
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args: pytest.fail("Unexpected clipping"))
    driver.train_step(model, optimizer, scaler, batch, "cpu")
    assert scaler.calls == ["scale", "step", "unscale", "update"]


def test_prepare_freezes_two_tasks_and_never_changes_existing_profile(tmp_path, monkeypatch, driver):
    from part_ev_fullscale import data, model

    monkeypatch.setattr(P, "environment", lambda: {"fixture": "same environment"})
    monkeypatch.setattr(model, "reference_metadata", lambda: {})
    monkeypatch.setattr(data, "inspect_dataset", lambda *args, **kwargs: {
        "splits": {"train": {"event_count": 100000000}, "val": {"event_count": 5000000}}})
    args = SimpleNamespace(output=tmp_path, data_root=tmp_path / "data",
                           comparison="standard", training_recipe="adamw")
    driver.prepare(args)
    before = (tmp_path / "campaign.json").read_bytes()
    spec = P.load_campaign(tmp_path)
    assert spec["training_recipe"] == "adamw" and len(spec["tasks"]) == 2
    driver.prepare(args)
    assert (tmp_path / "campaign.json").read_bytes() == before
    args.training_recipe = "original"
    with pytest.raises(ValueError, match="another training recipe"):
        driver.prepare(args)
    assert (tmp_path / "campaign.json").read_bytes() == before


@pytest.mark.parametrize("phase", ["benchmark", "train", "test", "report"])
def test_recipe_flag_is_prepare_only(tmp_path, driver, phase):
    with pytest.raises(SystemExit) as failure:
        driver.main([phase, "--output", str(tmp_path), "--training-recipe", "adamw"])
    assert failure.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing", [False, True])
def test_test_phase_locks_exactly_two_completed_checkpoints(tmp_path, monkeypatch, driver, campaign, missing):
    expected = []
    for index, task in enumerate(campaign["tasks"]):
        directory = P.task_directory(tmp_path, index, tasks=campaign["tasks"])
        P.write_json(directory / "best_model_val.pt", {"checkpoint_fixture": index})
        sha = P.file_sha256(directory / "best_model_val.pt")
        if not (missing and index == 1):
            P.write_json(directory / "training_complete.json", {
                "campaign_sha256": campaign["content_hash"], "step": 1000000,
                "validated_epochs": 50, "best_step": 900000, "best_checkpoint_sha256": sha})
        P.write_json(directory / "test/metrics.json", {"already_complete_fixture": True})
        expected.append({"task": task, "sha256": sha, "step": 900000})
    monkeypatch.setattr(driver, "distributed_setup", lambda: (0, 1, torch.device("cpu")))
    inspected = []
    monkeypatch.setattr(driver, "inspect_test", lambda *args: inspected.append(True))
    args = SimpleNamespace(output=tmp_path, task_index=0, test_data=tmp_path / "test_data")
    if missing:
        with pytest.raises(FileNotFoundError):
            driver.test(args)
        assert not inspected and not (tmp_path / "locked_finalists.json").exists()
    else:
        driver.test(args)
        assert P.read_json(tmp_path / "locked_finalists.json") == expected
        assert inspected == [True]


@pytest.mark.parametrize("phase", ["train", "test"])
def test_out_of_range_pilot_task_rejected_before_gpu_setup(tmp_path, monkeypatch, driver, campaign, phase):
    monkeypatch.setattr(driver, "distributed_setup", lambda: pytest.fail("Must validate task before GPU setup"))
    with pytest.raises(ValueError):
        getattr(driver, phase)(SimpleNamespace(output=tmp_path, task_index=2))


def test_one_seed_report_does_not_invent_seed_variation(tmp_path, driver, campaign):
    for index, task in enumerate(campaign["tasks"]):
        metrics = {
            "events": 20000000, "accuracy": 0.86 + 0.002 * index,
            "macro_auc_ovo": 0.98 + 0.001 * index, "macro_auc_ovr": 0.99 + 0.001 * index,
            "qcd_rejection": {"Hbb": {"0.5": {"rejection": 10000 + index}}},
        }
        directory = P.task_directory(tmp_path, index, tasks=campaign["tasks"])
        P.write_json(directory / "test/metrics.json", {
            "task": task, "campaign_sha256": campaign["content_hash"],
            "inventory_sha256": "same-population", "metrics": metrics})
    driver.report(SimpleNamespace(output=tmp_path))
    report = P.read_json(tmp_path / "reports/fullscale_report.json")
    assert report["training_recipe"] == "adamw" and report["exploratory_single_seed"]
    assert report["paired_seeds"] == [101]
    assert report["paired_seed_accuracy_gains"] == pytest.approx([0.002])
    assert report["mean_accuracy_gain"] == pytest.approx(0.002)
    for index, architecture in enumerate(("part", "part_ev")):
        row = report["models"][architecture]
        assert row["seed_count"] == 1
        assert row["accuracy"]["seed_std"] is None
        assert row["macro_auc_ovo"]["seed_std"] is None
        rejection = row["qcd_rejection"]["Hbb"]["0.5"]
        assert rejection["per_seed"] == {"101": 10000 + index}
        assert rejection["seed_std"] is None


@pytest.mark.parametrize("skip_measured", [False, True])
def test_adamw_benchmark_uses_factory_clipping_and_production_warmup(
    tmp_path, monkeypatch, driver, campaign, skip_measured
):
    from part_ev_fullscale import data

    runtime = {"world_size": 1, "gpu": "fixture", "campaign_sha256": campaign["content_hash"]}
    batch = (torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    calls = []

    def model(*args):
        result = torch.nn.Linear(1, 10)
        return result, result

    def update(model, optimizer, scaler, batch, device, *, gradient_clipping, diagnostics):
        assert isinstance(optimizer, torch.optim.AdamW)
        assert gradient_clipping == 1.0
        expected_lr = (len(calls) % 3 + 1) * 2e-8
        assert all(group["lr"] == pytest.approx(expected_lr) for group in optimizer.param_groups)
        calls.append(optimizer)
        diagnostics.update(gradient_clipped=True, nonfinite_grad_norm=False, grad_norm_before_clip=2.0)
        return 1.0, skip_measured and len(calls) > 1

    monkeypatch.setattr(driver, "distributed_setup", lambda: (0, 1, torch.device("cpu")))
    monkeypatch.setattr(driver, "runtime_contract", lambda *args: runtime)
    monkeypatch.setattr(driver, "make_model", model)
    monkeypatch.setattr(driver, "train_step", update)
    monkeypatch.setattr(driver, "inputs_and_labels", lambda batch, device: ([batch[0]], batch[1]))
    monkeypatch.setattr(data, "build_loader", lambda *args, **kwargs: (repeat(batch), None))
    monkeypatch.setattr(data, "shutdown_loader", lambda loader: None)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args: None)
    for method in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, method, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    args = SimpleNamespace(output=tmp_path, warmup=1, steps=2)
    if skip_measured:
        with pytest.raises(FloatingPointError, match="5%"):
            driver.benchmark(args)
        assert not (tmp_path / "benchmarks/gpus_1.json").exists()
    else:
        driver.benchmark(args)
        assert len(calls) == 6
        report = P.read_json(tmp_path / "benchmarks/gpus_1.json")
        for row in report["models"].values():
            assert row["optimizer"] == "AdamW"
            assert row["gradient_clipped_steps_rank0"] == 3
            assert row["learning_rate_policy"] == "production_warmup_schedule"
            assert row["first_learning_rate"] == pytest.approx(2e-8)
            assert row["last_learning_rate"] == pytest.approx(6e-8)
