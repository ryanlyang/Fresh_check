"""CPU-only regression tests for the full-scale protocol and restart state."""
from __future__ import annotations

import copy
import importlib.util
from itertools import combinations
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from part_ev_fullscale import protocol as P
from part_ev_fullscale.checkpoint import (
    atomic_torch_save,
    capture_rank_state,
    restore_rank_state,
    seed_everything,
    trusted_load,
)
from part_ev_fullscale.metrics import calculate_metrics
from part_ev_fullscale.optimizer import Ranger


def reference_ranger(params):
    """Actual pinned Weaver updates, not a second handwritten implementation."""
    from weaver.utils.nn.optimizer.lookahead import Lookahead
    from weaver.utils.nn.optimizer.radam import RAdam
    return Lookahead(RAdam(params, lr=1e-3, betas=(0.95, 0.999), eps=1e-5,
                           weight_decay=0), alpha=0.5, k=6, pullback_momentum="none")


def parameters():
    return [torch.nn.Parameter(torch.tensor([0.7, -1.2, 0.3], dtype=torch.float32)),
            torch.nn.Parameter(torch.tensor([[1.2, -0.3], [2.4, 0.8]], dtype=torch.float32))]


def update(optimizer, params, step):
    for index, param in enumerate(params):
        values = torch.arange(param.numel(), dtype=param.dtype).reshape_as(param)
        param.grad = torch.sin(values + step * 0.37 + index) + 0.05 * param.detach()
    optimizer.step()


def test_ranger_matches_reference_through_first_slow_weight_update():
    actual_params, expected_params = parameters(), parameters()
    actual, expected = Ranger(actual_params), reference_ranger(expected_params)
    for step in range(1, 8):
        update(actual, actual_params, step)
        update(expected, expected_params, step)
        for actual_param, expected_param in zip(actual_params, expected_params):
            torch.testing.assert_close(actual_param, expected_param, rtol=0, atol=0)
        assert actual.step_counter == expected.step_counter == step % 6
        for slow, param in zip(actual.slow, expected_params):
            torch.testing.assert_close(slow, expected.state[param]["cached_params"], rtol=0, atol=0)
    assert actual.param_groups[0]["weight_decay"] == 0
    assert actual.param_groups[0]["betas"] == (0.95, 0.999)
    assert actual.param_groups[0]["eps"] == 1e-5


def test_ranger_checkpoint_at_step_four_resumes_across_step_six(tmp_path):
    uninterrupted_params = parameters()
    uninterrupted = Ranger(uninterrupted_params)
    for step in range(1, 5):
        update(uninterrupted, uninterrupted_params, step)
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"params": [p.detach().clone() for p in uninterrupted_params],
                       "optimizer": uninterrupted.state_dict()}, path)
    assert not list(tmp_path.glob("*.tmp"))
    saved = trusted_load(path)
    assert saved["optimizer"]["step_counter"] == 4
    resumed_params = [torch.nn.Parameter(p.clone()) for p in saved["params"]]
    resumed = Ranger(resumed_params)
    resumed.load_state_dict(saved["optimizer"])
    for step in range(5, 14):
        update(uninterrupted, uninterrupted_params, step)
        update(resumed, resumed_params, step)
        for expected, actual in zip(uninterrupted_params, resumed_params):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for expected, actual in zip(uninterrupted.slow, resumed.slow):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert resumed.step_counter == uninterrupted.step_counter


@pytest.mark.parametrize("alter", ["schema", "slow", "counter"])
def test_incomplete_ranger_restart_is_rejected(alter):
    optimizer = Ranger(parameters())
    state = copy.deepcopy(optimizer.state_dict())
    if alter == "schema":
        state["schema"] = "legacy_missing_lookahead"
    elif alter == "slow":
        state["slow"] = []
    else:
        state["step_counter"] = 6
    with pytest.raises(ValueError):
        optimizer.load_state_dict(state)


def test_learning_rate_milestones_match_reference_multistep_schedule():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(35, 50)), gamma=0.01 ** (1 / 15))
    for epoch in range(50):
        assert P.learning_rate(epoch * 20000) == pytest.approx(optimizer.param_groups[0]["lr"], rel=1e-14)
        assert P.learning_rate(epoch * 20000 + 19999) == P.learning_rate(epoch * 20000)
        optimizer.step()
        scheduler.step()
    assert P.learning_rate(699999) == 1e-3
    assert P.learning_rate(700000) < 1e-3
    assert P.learning_rate(999999) == pytest.approx(1e-5, rel=1e-14)
    for invalid in (-1, 1000000):
        with pytest.raises(ValueError):
            P.learning_rate(invalid)


def test_global_batch_and_run_count_are_fixed_across_gpu_choices():
    assert [P.local_batch_size(world) for world in (1, 2, 4)] == [512, 256, 128]
    assert all(P.local_batch_size(world) * world == 512 for world in (1, 2, 4))
    for invalid in (0, 3, 8):
        with pytest.raises(ValueError):
            P.local_batch_size(invalid)
    assert len(P.TASKS) == 6
    assert {row["architecture"] for row in P.TASKS} == {"part", "part_ev"}
    assert P.TASKS[:3] == [{"architecture": "part", "seed": seed} for seed in P.SEEDS]
    assert P.RECIPE["epochs"] * P.RECIPE["steps_per_epoch"] == 1000000
    assert P.RECIPE["total_steps"] * P.RECIPE["global_batch_size"] == 512000000


class TrimmerFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from weaver.nn.model.ParticleTransformer import SequenceTrimmer
        self.trimmer = SequenceTrimmer(enabled=True)

    def forward(self, x, vectors, mask, pairs):
        return self.trimmer(x, vectors, mask, pairs)


def trim_inputs():
    n, length = 6, 12
    x = torch.arange(n * 3 * length).float().reshape(n, 3, length)
    vectors = torch.arange(n * 4 * length).float().reshape(n, 4, length)
    mask = torch.arange(length)[None, None, :] < torch.tensor([4, 5, 6, 8, 10, 12])[:, None, None]
    pairs = torch.arange(n * 2 * length * length).float().reshape(n, 2, length, length)
    return x, vectors, mask, pairs


@pytest.mark.parametrize("warmup_calls", [4, 7])
def test_rng_and_trimmer_counter_restart_replays_same_permutation(tmp_path, warmup_calls):
    seed_everything(202)
    model = TrimmerFixture().train()
    inputs = trim_inputs()
    for _ in range(warmup_calls):
        model(*inputs)
    state = capture_rank_state(model)
    assert state["trimmers"]["trimmer"] == min(warmup_calls, 5)
    path = tmp_path / "rank_state.pt"
    atomic_torch_save(state, path)
    expected = [tuple(value.clone() for value in model(*inputs)) for _ in range(3)]
    expected_random = (random.random(), np.random.random(5), torch.rand(5))
    seed_everything(917301)
    restarted = TrimmerFixture().train()
    restarted.trimmer._counter = 999
    restore_rank_state(restarted, trusted_load(path))
    for expected_step in expected:
        actual_step = restarted(*inputs)
        for actual, expected_value in zip(actual_step, expected_step):
            torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
    assert random.random() == expected_random[0]
    np.testing.assert_array_equal(np.random.random(5), expected_random[1])
    torch.testing.assert_close(torch.rand(5), expected_random[2], rtol=0, atol=0)


def balanced_logits():
    truth = np.repeat(np.arange(10, dtype=np.int64), 200)
    logits = np.zeros((len(truth), 10), dtype=np.float32)
    # Each signal has scores 0..199, QCD has signal-vs-QCD scores 20..219.
    logits[:200, 1:] = np.arange(20, 220)[:, None]
    for label in range(1, 10):
        logits[truth == label, label] = np.arange(200)
    return logits, truth


def test_balanced_ten_class_accuracy_and_macro_auc_are_population_metrics():
    logits, truth = balanced_logits()
    metrics = calculate_metrics(logits, truth)
    assert metrics["events"] == 2000
    assert set(metrics["class_support"].values()) == {200}
    assert metrics["accuracy"] == pytest.approx(9 * 199 / 2000)
    probabilities = softmax(logits.astype(np.float64), axis=1)
    pairs = []
    for first, second in combinations(range(10), 2):
        selected = (truth == first) | (truth == second)
        auc_first = roc_auc_score(truth[selected] == first, probabilities[selected, first])
        auc_second = roc_auc_score(truth[selected] == second, probabilities[selected, second])
        pairs.append((auc_first + auc_second) / 2)
    assert metrics["macro_auc_ovo"] == pytest.approx(np.mean(pairs), abs=1e-14)
    expected_ovr = np.mean([roc_auc_score(truth == label, probabilities[:, label]) for label in range(10)])
    assert metrics["macro_auc_ovr"] == pytest.approx(expected_ovr, abs=1e-14)


def test_rejection_exact_counts_and_class_specific_operating_points():
    metrics = calculate_metrics(*balanced_logits())
    rejection = metrics["qcd_rejection"]
    assert set(rejection) == set(P.CLASS_NAMES) - {"QCD"}
    for name in ("Hbb", "Hcc", "Hgg", "H4q", "Zqq", "Wqq", "Tbqq"):
        assert set(rejection[name]) == {"0.3", "0.5"}
        low, middle = rejection[name]["0.3"], rejection[name]["0.5"]
        assert (low["threshold_log_odds"], low["signal_accepted"], low["qcd_accepted"]) == (140, 60, 80)
        assert low["rejection"] == 2.5 and low["achieved_signal_efficiency"] == 0.3
        assert (middle["threshold_log_odds"], middle["signal_accepted"], middle["qcd_accepted"]) == (100, 100, 120)
        assert middle["rejection"] == pytest.approx(200 / 120)
    assert set(rejection["Hqql"]) == {"0.99"}
    assert rejection["Hqql"]["0.99"]["signal_accepted"] == 198
    assert set(rejection["Tbl"]) == {"0.995"}
    assert rejection["Tbl"]["0.995"]["signal_accepted"] == 199


def test_zero_background_is_explicit_not_infinite_or_discarded():
    truth = np.repeat(np.arange(10, dtype=np.int64), 200)
    logits = np.eye(10, dtype=np.float32)[truth] * 10
    result = calculate_metrics(logits, truth)
    assert result["accuracy"] == 1.0 and result["macro_auc_ovo"] == 1.0
    for values in result["qcd_rejection"].values():
        for row in values.values():
            assert row["qcd_accepted"] == 0 and row["qcd_total"] == 200
            assert row["rejection"] is None
            assert row["achieved_signal_efficiency"] == 1.0  # Ties are retained.


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "missing_class"])
def test_metrics_reject_incomplete_or_invalid_results(failure):
    logits, truth = balanced_logits()
    if failure == "shape":
        logits = logits[:, :9]
    elif failure == "nonfinite":
        logits[0, 0] = np.nan
    else:
        logits, truth = logits[truth != 9], truth[truth != 9]
    with pytest.raises(ValueError):
        calculate_metrics(logits, truth)


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_part_fullscale.py"
    spec = importlib.util.spec_from_file_location("part_fullscale_driver_orchestration_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["standard", "plain", "adamw", "parameter_matched"])
def cpu_orchestration(monkeypatch, driver, request):
    """Exercise the real train/replay/checkpoint loop; replace only GPU/data work."""
    from part_ev_fullscale import data

    stats = {"loaders": [], "updates": 0, "validations": 0}
    comparison = "standard" if request.param == "adamw" else request.param
    training_recipe = "adamw" if request.param == "adamw" else "original"
    campaign = {"content_hash": "cpu-orchestration-campaign", "inventory": {},
                "comparison": comparison, "training_recipe": training_recipe,
                "recipe": P.recipe_for(comparison, training_recipe),
                "tasks": P.tasks_for(comparison, training_recipe)}
    if comparison == "parameter_matched":
        from part_ev_fullscale import model as model_module
        # This orchestration test uses TinyModel's eight parameters; the real
        # 2.2M-parameter architecture/count is checked in the model tests.
        campaign["parameter_match"] = {"matched_parameters": 8}
        monkeypatch.setattr(model_module, "parameter_match_metadata", lambda: {"matched_parameters": 8})
    stats["tasks"] = campaign["tasks"]
    runtime = {"world_size": 1, "local_batch_size": 512, "gpu": "CPU test fixture",
               "campaign_sha256": campaign["content_hash"]}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.dropout = torch.nn.Dropout(0.25)
            self._counter = 0

        def forward(self, values):
            self._counter += 1
            if self.training:
                # Deliberately consume all three saved CPU RNG streams.
                values = values + 0.01 * (random.random() + np.random.random())
            return self.linear(self.dropout(values))

    class SyntheticLoader:
        def __init__(self, phase, seed):
            self.phase = phase
            self.generator = torch.Generator().manual_seed(seed + (10000 if phase == "val" else 0))
            self.consumed = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            self.consumed += 1
            values = torch.randn(8, 3, generator=self.generator)
            targets = torch.randint(2, size=(8,), generator=self.generator)
            return values, targets

    def loader(phase, inventory, batch_size, *, seed, **kwargs):
        result = SyntheticLoader(phase, seed)
        stats["loaders"].append(result)
        return result, None

    def make_model(architecture, seed, device, world):
        seed_everything(seed)
        model = TinyModel()
        return model, model

    def train_step(model, optimizer, scaler, batch, device, *, gradient_clipping=None, diagnostics=None):
        stats["updates"] += 1
        values, targets = batch
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(values), targets)
        loss.backward()
        if gradient_clipping is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            diagnostics.update(grad_norm_before_clip=float(norm),
                               gradient_clipped=float(norm) > gradient_clipping, nonfinite_grad_norm=False)
        optimizer.step()
        return float(loss.detach()), False

    def validate(model, iterator, device):
        stats["validations"] += 1
        model.eval()
        values, targets = next(iterator)
        with torch.no_grad():
            logits = model(values)
            # Stable, continuous fixture criterion; not a scientific metric.
            accuracy = float((torch.tanh(logits.mean()) + 1) / 2)
        return {"accuracy": accuracy, "loss": float(logits.square().mean()), "events": 512}

    scaler_class = torch.amp.GradScaler
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: scaler_class("cpu", enabled=False))
    monkeypatch.setattr(P, "load_campaign", lambda output: copy.deepcopy(campaign))
    monkeypatch.setattr(P, "learning_rate", lambda step, recipe=None: 1e-3)
    for key, value in {"epochs": 2, "steps_per_epoch": 2, "total_steps": 4,
                       "validation_events_per_epoch": 512}.items():
        monkeypatch.setitem(P.RECIPE, key, value)
        campaign["recipe"][key] = value
    monkeypatch.setattr(driver, "distributed_setup", lambda: (0, 1, torch.device("cpu")))
    monkeypatch.setattr(driver, "runtime_contract", lambda *args: copy.deepcopy(runtime))
    monkeypatch.setattr(driver, "require_benchmark", lambda *args: None)
    monkeypatch.setattr(driver, "make_model", make_model)
    monkeypatch.setattr(driver, "train_step", train_step)
    monkeypatch.setattr(driver, "validate", validate)
    monkeypatch.setattr(data, "build_loader", loader)
    monkeypatch.setattr(data, "shutdown_loader", lambda value: setattr(value, "closed", True))
    return driver, stats


def assert_nested_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for observed, reference in zip(actual, expected):
            assert_nested_exact(observed, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("interrupt_checkpoint", [1, 2])
def test_train_orchestration_restarts_before_or_after_validation_exactly(
    tmp_path, monkeypatch, cpu_orchestration, interrupt_checkpoint
):
    driver, stats = cpu_orchestration
    baseline_output, resumed_output = tmp_path / "baseline", tmp_path / "resumed"
    driver.train(SimpleNamespace(output=baseline_output, task_index=0))
    assert stats["updates"] == 4 and stats["validations"] == 2
    assert [value.consumed for value in stats["loaders"]] == [4, 2]
    baseline_dir = P.task_directory(baseline_output, 0, tasks=stats["tasks"])
    expected_last = trusted_load(baseline_dir / "last.pt")
    expected_best = trusted_load(baseline_dir / "best_model_val.pt")
    assert expected_last["state"]["step"] == 4
    assert expected_last["state"]["validated_epochs"] == 2
    assert (baseline_dir / "training_complete.json").is_file()
    assert (baseline_output / "training_runtime.json").is_file()
    assert len(list((baseline_dir / "validation").glob("epoch_*.json"))) == 2

    original_save = driver.save_checkpoint
    checkpoints = 0

    class SimulatedInterruption(Exception):
        pass

    def save_and_interrupt(*args, **kwargs):
        nonlocal checkpoints
        original_save(*args, **kwargs)
        checkpoints += 1
        if checkpoints == interrupt_checkpoint:
            raise SimulatedInterruption("simulated job death immediately after a durable checkpoint")

    monkeypatch.setattr(driver, "save_checkpoint", save_and_interrupt)
    with pytest.raises(SimulatedInterruption):
        driver.train(SimpleNamespace(output=resumed_output, task_index=0))
    resumed_dir = P.task_directory(resumed_output, 0, tasks=stats["tasks"])
    interrupted = trusted_load(resumed_dir / "last.pt")
    assert interrupted["state"]["step"] == 2
    assert interrupted["state"]["validated_epochs"] == interrupt_checkpoint - 1
    assert not (resumed_dir / "training_complete.json").exists()
    assert all(loader.closed for loader in stats["loaders"])

    monkeypatch.setattr(driver, "save_checkpoint", original_save)
    driver.train(SimpleNamespace(output=resumed_output, task_index=0))
    assert_nested_exact(trusted_load(resumed_dir / "last.pt"), expected_last)
    assert_nested_exact(trusted_load(resumed_dir / "best_model_val.pt"), expected_best)
    assert stats["updates"] == 8  # Four baseline + four interrupted/resumed; replay performs no updates.
    assert stats["validations"] == 4
    assert [loader.consumed for loader in stats["loaders"][-2:]] == [4, 2]
    assert all(loader.closed for loader in stats["loaders"])
    assert len(list((resumed_dir / "validation").glob("epoch_*.json"))) == 2

    # A terminal training marker skips work on an explicitly resubmitted array.
    loader_count = len(stats["loaders"])
    driver.train(SimpleNamespace(output=resumed_output, task_index=0))
    assert stats["updates"] == 8 and len(stats["loaders"]) == loader_count


def test_production_requires_completed_matching_gpu_benchmark(tmp_path, driver):
    runtime = {"world_size": 2, "gpu": "GH200", "campaign_sha256": "fixture"}
    with pytest.raises(RuntimeError, match="benchmark"):
        driver.require_benchmark(tmp_path, runtime)
    path = tmp_path / "benchmarks/gpus_2.json"
    correct = {"runtime": runtime, "passed": True, "models": {"part": {}, "part_ev": {}}}
    mutations = [
        {"runtime": {**runtime, "world_size": 4}},
        {"runtime": {**runtime, "campaign_sha256": "another-campaign"}},
        {"passed": False},
        {"models": {"part": {}}},
        {"models": {"part": {}, "part_ev": {}, "unexpected_architecture": {}}},
    ]
    for changed in mutations:
        P.write_json(path, {**correct, **changed})
        with pytest.raises(ValueError, match="exact campaign/runtime"):
            driver.require_benchmark(tmp_path, runtime)
    P.write_json(path, correct)
    driver.require_benchmark(tmp_path, runtime)


def test_production_runtime_cannot_change_after_first_run(tmp_path, driver):
    runtime = {"world_size": 1, "local_batch_size": 512, "gpu": "GH200"}
    driver.freeze_runtime(tmp_path, runtime)
    driver.freeze_runtime(tmp_path, dict(runtime))
    with pytest.raises(ValueError, match="same runtime/GPU count"):
        driver.freeze_runtime(tmp_path, {**runtime, "world_size": 2, "local_batch_size": 256})
