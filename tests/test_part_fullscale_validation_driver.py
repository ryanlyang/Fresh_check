"""Read-only donor auditing and resumable finite fullscale-validation inference."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from part_ev_fullscale import protocol as P


@pytest.fixture
def evaluator():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_part_fullscale_validation.py"
    spec = importlib.util.spec_from_file_location("fullscale_validation_driver_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def donor_factory(tmp_path, evaluator):
    def make(*, file_events=100_000, legacy=False):
        campaign = tmp_path / "original"
        source = campaign / "source"
        source_names = (
            "scripts/run_part_fullscale.py", "part_ev_fullscale/__init__.py",
            "part_ev_fullscale/protocol.py", "part_ev_fullscale/model.py",
            "part_ev_fullscale/data.py", "part_ev_fullscale/metrics.py",
            "part_ev_fullscale/checkpoint.py",
            "particle_transformer/data/JetClass/JetClass_full.yaml",
        )
        manifest = []
        for name in source_names:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("# frozen fixture " + name + "\n").encode())
            manifest.append(f"{evaluator.file_sha256(path)}  ./{name}\n")
        (campaign / "source.sha256").write_text("".join(manifest))
        (campaign / "launch.env").write_text("PART_FULLSCALE_PYTHON=/pinned/python\n")
        records = []
        for name in evaluator.FILE_LABELS:
            for index in range(120, 125):
                path = tmp_path / "data" / "val_5M" / f"{name}_{index:03d}.root"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("ROOT fixture " + path.name).encode())
                stat = path.stat()
                records.append(dict(name=path.name, path=str(path), size_bytes=stat.st_size,
                                    mtime_ns=stat.st_mtime_ns, num_entries=file_events))
        spec = dict(schema=P.SCHEMA, tasks=copy.deepcopy(P.TASKS), recipe=copy.deepcopy(P.RECIPE),
                    environment={"fixture": "unchanged"}, inventory={"splits": {
                        "train": {"directory": str(tmp_path / "absent_train")},
                        "test": {"directory": str(tmp_path / "absent_test")},
                        "val": {"files": records, "event_count": 50 * file_events}}})
        if not legacy:
            spec.update(comparison="standard", training_recipe="original")
        spec["content_hash"] = P.digest(spec)
        P.write_json(campaign / "campaign.json", spec)
        runtime = dict(world_size=1, campaign_sha256=spec["content_hash"], tf32=False,
                       cudnn_benchmark=False, gpu="fixture GPU")
        P.write_json(campaign / "training_runtime.json", runtime)
        for task in spec["tasks"]:
            run = campaign / "runs" / task["architecture"] / f"seed_{task['seed']}"
            history = []
            for epoch in range(1, 51):
                row = dict(step=epoch * 20_000, accuracy=.9 if epoch in (42, 43) else .8,
                           loss=.5, events=1_280_000)
                history.append(row)
                P.write_json(run / "validation" / f"epoch_{epoch:02d}.json", row)
            checkpoint = run / "best_model_val.pt"
            torch.save(dict(task=task, campaign_sha256=spec["content_hash"], runtime=runtime,
                            step=840_000, validation=history[41], model={"weight": 2 * torch.eye(10)}),
                       checkpoint)
            P.write_json(run / "training_complete.json", dict(
                task=task, campaign_sha256=spec["content_hash"], runtime=runtime, step=1_000_000,
                validated_epochs=50, best_step=840_000, best_accuracy=.9,
                best_checkpoint_sha256=evaluator.file_sha256(checkpoint)))
        return SimpleNamespace(campaign=campaign, spec=spec, runtime=runtime,
                               output=tmp_path / "evaluation")
    return make


@pytest.mark.parametrize("legacy", [False, True])
def test_original_and_legacy_profiles_remain_read_only(evaluator, donor_factory, legacy):
    donor = donor_factory(legacy=legacy)
    before = snapshot(donor.campaign)
    evaluator.validate_profile(donor.spec)
    selected = evaluator.lock_checkpoints(donor.campaign, donor.spec, P)
    assert [row["task"] for row in selected] == P.TASKS
    assert {row["selected_check"] for row in selected} == {42}
    assert {row["step"] for row in selected} == {840_000}
    assert snapshot(donor.campaign) == before


@pytest.mark.parametrize("mutation", ["plain", "adamw", "task_order", "wrong_seed", "budget"])
def test_other_campaign_profiles_are_rejected(evaluator, mutation):
    spec = dict(tasks=copy.deepcopy(P.TASKS), recipe=copy.deepcopy(P.RECIPE))
    if mutation == "plain":
        spec["comparison"] = "plain"
    elif mutation == "adamw":
        spec["training_recipe"] = "adamw"
    elif mutation == "task_order":
        spec["tasks"].reverse()
    elif mutation == "wrong_seed":
        spec["tasks"][0]["seed"] = 999
    else:
        spec["recipe"]["total_steps"] = 200_000
    with pytest.raises(ValueError, match="original shared-bias Ranger"):
        evaluator.validate_profile(spec)


@pytest.mark.parametrize("mutation", [
    "incomplete", "wrong_task", "runtime", "late_tie", "wrong_accuracy", "checkpoint", "missing_epoch",
    "extra_epoch", "wrong_step", "wrong_events", "invalid_accuracy",
])
def test_all_six_completion_and_selection_contracts_are_checked(evaluator, donor_factory, mutation):
    donor = donor_factory()
    run = donor.campaign / "runs/part_ev/seed_303"
    done = P.read_json(run / "training_complete.json")
    if mutation == "incomplete":
        done["step"] -= 1
    elif mutation == "wrong_task":
        done["task"] = {"architecture": "part", "seed": 303}
    elif mutation == "runtime":
        done["runtime"]["gpu"] = "different GPU"
    elif mutation == "late_tie":
        done["best_step"] = 860_000
    elif mutation == "wrong_accuracy":
        done["best_accuracy"] = .89
    elif mutation == "checkpoint":
        (run / "best_model_val.pt").write_bytes(b"changed checkpoint")
    elif mutation == "missing_epoch":
        (run / "validation/epoch_17.json").unlink()
    elif mutation == "extra_epoch":
        P.write_json(run / "validation/epoch_51.json", {})
    else:
        path = run / "validation/epoch_20.json"
        row = P.read_json(path)
        if mutation == "wrong_step":
            row["step"] += 1
        elif mutation == "wrong_events":
            row["events"] -= 1
        else:
            row["accuracy"] = 2.0
        P.write_json(path, row)
    P.write_json(run / "training_complete.json", done)
    with pytest.raises(ValueError):
        evaluator.lock_checkpoints(donor.campaign, donor.spec, P)


@pytest.mark.parametrize("location", ["same", "inside", "above"])
def test_output_cannot_overlap_original_campaign(tmp_path, evaluator, location):
    campaign = tmp_path / "donor"
    campaign.mkdir()
    output = {"same": campaign, "inside": campaign / "validation", "above": tmp_path}[location]
    with pytest.raises(ValueError, match="separate"):
        evaluator.separate_output(output, campaign)


@pytest.mark.parametrize("mutation", ["changed", "missing_dependency", "duplicate", "outside"])
def test_frozen_source_manifest_is_verified(evaluator, donor_factory, mutation):
    donor = donor_factory()
    source = donor.campaign / "source"
    assert evaluator.verify_frozen_source(donor.campaign) == source
    manifest = donor.campaign / "source.sha256"
    lines = manifest.read_text().splitlines(keepends=True)
    if mutation == "changed":
        (source / "part_ev_fullscale/model.py").write_text("# changed\n")
    elif mutation == "missing_dependency":
        manifest.write_text("".join(line for line in lines if "model.py" not in line))
    elif mutation == "duplicate":
        manifest.write_text("".join(lines + lines[:1]))
    else:
        outside = donor.campaign / "outside.py"
        outside.write_text("# not frozen source\n")
        manifest.write_text("".join(lines) + f"{evaluator.file_sha256(outside)}  ../outside.py\n")
    with pytest.raises(ValueError):
        evaluator.verify_frozen_source(donor.campaign)


def test_loading_rejects_previously_imported_foreign_fullscale_modules(evaluator, donor_factory):
    donor = donor_factory()
    with pytest.raises(RuntimeError, match="different campaign"):
        evaluator.load_driver(donor.campaign)


def test_validation_inventory_never_requires_training_or_test_data(evaluator, donor_factory, monkeypatch):
    donor = donor_factory()
    opened = []
    monkeypatch.setattr(evaluator, "root_entries", lambda path: opened.append(path) or 100_000)
    before = snapshot(donor.campaign)
    inventory = evaluator.validation_inventory(donor.spec)
    assert set(inventory["splits"]) == {"val"}
    assert inventory["splits"]["val"]["event_count"] == 5_000_000
    assert len(opened) == 50 and all("val_5M" in path for path in opened)
    assert all(len(row["sha256"]) == 64 for row in inventory["splits"]["val"]["files"])
    assert snapshot(donor.campaign) == before


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_name", "header", "changed_data"])
def test_validation_inventory_requires_exact_official_coverage(evaluator, donor_factory, monkeypatch, mutation):
    donor = donor_factory()
    monkeypatch.setattr(evaluator, "root_entries", lambda path: 99_999 if mutation == "header" else 100_000)
    rows = donor.spec["inventory"]["splits"]["val"]["files"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "wrong_name":
        rows[0]["name"] = "HToBB_100.root"  # Test filenames are never accepted as validation.
    elif mutation == "changed_data":
        Path(rows[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        evaluator.validation_inventory(donor.spec)


@pytest.fixture
def toy_campaign(evaluator, donor_factory, monkeypatch):
    monkeypatch.setattr(evaluator, "FILE_EVENTS", 2)
    monkeypatch.setattr(evaluator, "VALIDATION_EVENTS", 100)
    monkeypatch.setattr(evaluator, "CLASS_EVENTS", 10)
    donor = donor_factory(file_events=2)
    calls = dict(models=[], loads=[], shutdowns=[])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(10, 10))
            self.use_amp = True  # The frozen driver owns the native AMP choice.
            self._counter = 0

        def forward(self, features):
            assert self.use_amp is True and self._counter == 5 and not self.training
            assert not torch.is_grad_enabled()
            return features @ self.weight

    def make_model(architecture, seed, device, world):
        assert world == 1 and device.type == "cpu"
        calls["models"].append((architecture, seed))
        model = Model()
        return model, model

    protocol = SimpleNamespace(load_campaign=lambda path: P.read_json(path / "campaign.json"),
                               file_lock=P.file_lock, write_json=P.write_json)
    frozen = SimpleNamespace(P=protocol, STOP_REQUESTED=False, make_model=make_model,
                             distributed_setup=lambda: (0, 1, torch.device("cpu")),
                             inputs_and_labels=lambda batch, device: ([batch[0]], batch[1]))
    monkeypatch.setattr(evaluator, "load_driver", lambda path: frozen)
    monkeypatch.setattr(evaluator, "root_entries", lambda path: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fixture GPU")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    from part_ev_fullscale import data
    monkeypatch.setattr(data, "shutdown_loader", lambda loader: calls["shutdowns"].append(loader))
    helper = evaluator.support()
    monkeypatch.setattr(helper, "EVENTS", 100)
    monkeypatch.setattr(helper, "CLASS_EVENTS", 10)

    def loader(inventory, row, batch_size):
        assert set(inventory["splits"]) == {"val"} and batch_size == 512
        calls["loads"].append(row["name"])
        label = evaluator.FILE_LABELS[row["name"].rsplit("_", 1)[0]]
        truth = torch.full((2,), label, dtype=torch.long)
        features = torch.nn.functional.one_hot(truth, 10).float()
        return [(features[:1], truth[:1]), (features[1:], truth[1:])]

    monkeypatch.setattr(helper, "finite_validation_loader", loader)
    monkeypatch.setattr(evaluator, "support", lambda: helper)
    donor.frozen, donor.calls, donor.helper = frozen, calls, helper
    return donor


def prepare_toy(evaluator, donor):
    evaluator.prepare(SimpleNamespace(output=donor.output, campaign=donor.campaign))
    return evaluator.read_json(donor.output / "evaluation.json")


def test_prepare_rejects_incomplete_training_before_data_or_output_access(evaluator, toy_campaign, monkeypatch):
    donor = toy_campaign
    path = donor.campaign / "runs/part_ev/seed_303/training_complete.json"
    done = evaluator.read_json(path)
    done["validated_epochs"] = 49
    P.write_json(path, done)
    monkeypatch.setattr(evaluator, "root_entries", lambda path: pytest.fail("Must check completion before data"))
    with pytest.raises(ValueError, match="completed"):
        prepare_toy(evaluator, donor)
    assert not donor.output.exists()


def test_prepare_rejects_output_inside_validation_data(evaluator, toy_campaign):
    donor = toy_campaign
    donor.output = Path(donor.spec["inventory"]["splits"]["val"]["files"][0]["path"]).parent / "new_output"
    with pytest.raises(ValueError, match="overlap the validation data"):
        prepare_toy(evaluator, donor)
    assert not donor.output.exists()


def test_prepare_and_six_model_evaluation_report_and_resume_leave_donor_untouched(evaluator, toy_campaign):
    donor = toy_campaign
    before = snapshot(donor.campaign)
    plan = prepare_toy(evaluator, donor)
    assert plan["test_accessed"] is False and plan["split"] == "validation"
    assert plan["policy"]["native_model_use_amp"] is True
    assert plan["policy"]["outer_autocast"] is False
    evaluator.prepare(SimpleNamespace(output=donor.output, campaign=donor.campaign))
    for index in range(6):
        evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=index))
    assert len(donor.calls["models"]) == 6
    assert len(donor.calls["loads"]) == len(donor.calls["shutdowns"]) == 300
    for task in P.TASKS:
        path = donor.output / "runs" / task["architecture"] / f"seed_{task['seed']}" / "metrics.json"
        row = evaluator.read_json(path)
        assert row["metrics"]["events"] == 100 and row["metrics"]["accuracy"] == 1
        assert set(row["metrics"]["class_support"].values()) == {10}
        assert len(row["prediction_sidecar_sha256"]) == 50
    evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=0))
    assert len(donor.calls["models"]) == 6 and len(donor.calls["loads"]) == 300
    evaluator.report(SimpleNamespace(output=donor.output))
    result = evaluator.read_json(donor.output / "reports/validation_report.json")
    assert result["test_accessed"] is False
    assert (donor.output / "reports/validation_report.md").is_file()
    assert snapshot(donor.campaign) == before


@pytest.mark.parametrize("mutation", ["metric_value", "task", "policy", "selected_check", "missing_prediction"])
def test_report_rejects_tampered_results_or_missing_prediction_coverage(evaluator, toy_campaign, mutation):
    donor = toy_campaign
    prepare_toy(evaluator, donor)
    evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=0))
    directory = donor.output / "runs/part/seed_101"
    path = directory / "metrics.json"
    record = evaluator.read_json(path)
    if mutation == "metric_value":
        record["metrics"]["accuracy"] = .5
    elif mutation == "task":
        record["task"] = {"architecture": "part_ev", "seed": 101}
    elif mutation == "policy":
        record["policy"]["native_model_use_amp"] = False
    elif mutation == "selected_check":
        record["selected_check"] = 43
    else:
        (directory / "predictions/file_017.npz").unlink()
    if mutation not in ("metric_value", "missing_prediction"):
        record.pop("content_hash", None)
        record["content_hash"] = evaluator.digest(record)
    P.write_json(path, record)
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluator.report(SimpleNamespace(output=donor.output))
    assert not (donor.output / "reports/validation_report.json").exists()


def test_interruption_preserves_complete_files_and_resume_does_not_repeat_them(evaluator, toy_campaign, monkeypatch):
    donor = toy_campaign
    prepare_toy(evaluator, donor)
    original = donor.helper.finite_validation_loader

    def stop_after_first(inventory, row, batch_size):
        if donor.calls["loads"]:
            donor.frozen.STOP_REQUESTED = True
        return original(inventory, row, batch_size)

    monkeypatch.setattr(donor.helper, "finite_validation_loader", stop_after_first)
    with pytest.raises(SystemExit) as stopped:
        evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=0))
    assert stopped.value.code == 75
    directory = donor.output / "runs/part/seed_101/predictions"
    assert (directory / "file_000.json").exists() and not (directory / "file_001.json").exists()
    committed = (directory / "file_000.npz").read_bytes()
    donor.frozen.STOP_REQUESTED = False
    monkeypatch.setattr(donor.helper, "finite_validation_loader", original)
    evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=0))
    assert (directory / "file_000.npz").read_bytes() == committed
    assert donor.calls["loads"].count(evaluator.read_json(donor.output / "evaluation.json")
                                      ["inventory"]["splits"]["val"]["files"][0]["name"]) == 1


@pytest.mark.parametrize("mutation", ["checkpoint_identity", "payload", "missing_payload", "data", "source_metadata"])
def test_resume_rejects_changed_or_mixed_inputs(evaluator, toy_campaign, mutation):
    donor = toy_campaign
    plan = prepare_toy(evaluator, donor)
    evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=0))
    directory = donor.output / "runs/part/seed_101/predictions"
    if mutation == "checkpoint_identity":
        sidecar = evaluator.read_json(directory / "file_000.json")
        sidecar["identity"]["checkpoint_sha256"] = "wrong checkpoint"
        P.write_json(directory / "file_000.json", sidecar)
    elif mutation == "payload":
        (directory / "file_000.npz").write_bytes(b"corrupt")
    elif mutation == "missing_payload":
        (directory / "file_000.npz").unlink()
    elif mutation == "data":
        Path(plan["inventory"]["splits"]["val"]["files"][0]["path"]).write_bytes(b"changed")
    else:
        (donor.campaign / "launch.env").write_text("changed\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=0))


@pytest.mark.parametrize("mutation", ["too_few", "too_many", "wrong_label", "nan"])
def test_inference_rejects_missing_repeated_mislabeled_or_nonfinite_events(evaluator, toy_campaign, monkeypatch, mutation):
    donor = toy_campaign
    prepare_toy(evaluator, donor)
    original = donor.helper.finite_validation_loader

    def invalid_loader(inventory, row, batch_size):
        batches = original(inventory, row, batch_size)
        if mutation == "too_few":
            return batches[:1]
        if mutation == "too_many":
            return batches + batches[:1]
        features, labels = batches[0]
        if mutation == "wrong_label":
            batches[0] = (features, (labels + 1) % 10)
        else:
            features[:] = float("nan")
        return batches

    monkeypatch.setattr(donor.helper, "finite_validation_loader", invalid_loader)
    with pytest.raises(ValueError):
        evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=0))
    assert not (donor.output / "runs/part/seed_101/metrics.json").exists()


@pytest.mark.parametrize("index", [-1, 6, True])
def test_invalid_task_is_rejected_before_gpu_setup(evaluator, toy_campaign, monkeypatch, index):
    donor = toy_campaign
    prepare_toy(evaluator, donor)
    monkeypatch.setattr(donor.frozen, "distributed_setup", lambda: pytest.fail("No GPU allocation for invalid task"))
    with pytest.raises(ValueError, match="Task index"):
        evaluator.evaluate(SimpleNamespace(output=donor.output, task_index=index))


@pytest.mark.parametrize("argv", [
    ["prepare", "--output", "new"], ["evaluate", "--output", "new"],
    ["report", "--output", "new", "--campaign", "old"],
    ["prepare", "--output", "new", "--campaign", "old", "--task-index", "0"],
])
def test_cli_rejects_inapplicable_or_missing_arguments(evaluator, argv):
    with pytest.raises(SystemExit) as error:
        evaluator.main(argv)
    assert error.value.code == 2
