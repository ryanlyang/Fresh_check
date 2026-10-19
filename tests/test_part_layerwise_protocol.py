import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from part_layerwise_screen import protocol as P
from part_layerwise_screen.reporting import contrasts, paired_accuracy
from part_ev_screen.protocol import recipes
from part_screen_common import engine as E
from scripts import run_part_layerwise_screen as D
from tests.test_part_ev_screen_training import batch
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_only_two_new_ranger_tasks_and_unchanged_recipe():
    assert P.CONTROLS == ["PART_RANGER", "EV_RANGER"]
    assert len(P.TASKS) == 2 and all(t["seed"] == 101 and t["recipe"] == "ranger_short_v1" for t in P.TASKS)
    assert P.recipes() == recipes()
    assert not P.ADMISSION["benchmark_performed"]
    assert all(t["run_id"] not in P.CONTROLS for t in P.TASKS)
    for index in (-1, 2, None, True):
        with pytest.raises(ValueError): P.task_directory("/unused", index)


def test_create_and_frozen_contract(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    env = {"fixture": "matched"}
    contract = dict(environment=env, inventory={"fixture": True}, controls=P.CONTROLS,
                    recipes=recipes(), runtime={"gpu": "A100"})
    calls = []
    monkeypatch.setattr(P, "donor_contract", lambda donor, controls, **kw: (calls.append((donor, controls, kw)) or contract))
    monkeypatch.setattr(P, "runtime_environment", lambda: env)
    monkeypatch.setattr(P, "verify_dataset_metadata", lambda inventory: calls.append(inventory))
    launch = dict(donor="/donor", submission_policy="direct_no_separate_preflight")
    spec = P.create_spec(source, launch)
    assert calls[0][1] == P.CONTROLS and calls[0][2]["local_source"] == source
    assert spec["parameters"] == P.COUNTS and spec["recipes"] == {"ranger_short_v1": recipes()["ranger_short_v1"]}
    P.write_json(tmp_path/"experiment_spec.json", spec)
    assert P.load_campaign(tmp_path) == spec
    assert P.require_approval(tmp_path, spec, dict(gpu="A100", campaign_sha256="new")) == P.ADMISSION
    with pytest.raises(ValueError, match="Runtime"): P.require_approval(tmp_path, spec, {"gpu": "GH200"})
    for key in ("tasks", "parameters", "model", "admission", "source", "recipes"):
        broken = {k: v for k, v in spec.items() if k != "content_hash"}
        broken[key] = {} if key != "tasks" else []
        P.write_json(tmp_path/"experiment_spec.json", P.hashed(broken))
        with pytest.raises(ValueError): P.load_campaign(tmp_path)


def test_interaction_is_difference_in_differences():
    values = dict(PART_RANGER=.8, EV_RANGER=.81, PART_LAYERWISE_RANGER=.82, EV_LAYERWISE_RANGER=.85)
    result = contrasts(values)
    assert result["ev_gain_shared"] == pytest.approx(.01)
    assert result["ev_gain_layerwise"] == pytest.approx(.03)
    assert result["layerwise_gain_without_ev"] == pytest.approx(.02)
    assert result["layerwise_gain_with_ev"] == pytest.approx(.04)
    assert result["ev_by_layerwise_interaction"] == pytest.approx(.02)


def test_paired_accuracy_checks_event_labels(tmp_path):
    labels = np.array([0, 1, 2])
    a, b = np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)[[0, 0, 1]]
    np.savez(tmp_path/"a.npz", logits=a, labels=labels)
    np.savez(tmp_path/"b.npz", logits=b, labels=labels)
    left = dict(predictions=[dict(name="file", path=str(tmp_path/"a.npz"))])
    right = dict(predictions=[dict(name="file", path=str(tmp_path/"b.npz"))])
    assert paired_accuracy(left, right) == dict(wins=2, losses=0, ties=1, events=3, accuracy_difference=2/3)
    np.savez(tmp_path/"b.npz", logits=b, labels=labels[::-1])
    with pytest.raises(ValueError, match="labels"): paired_accuracy(left, right)


def test_direct_startup_is_discarded_before_training(monkeypatch, tmp_path):
    runner = D.DirectEngine(P, D.build_model, D.initial_parity, D.diagnostics)
    calls = []
    monkeypatch.setattr(D, "prepare", lambda output: {"fixture": True})
    monkeypatch.setattr(runner, "startup_subprocess", lambda *a: calls.append("startup"))
    monkeypatch.setattr(E.ScreenEngine, "train", lambda *a: calls.append("fresh_common_engine"))
    runner.train(SimpleNamespace(output=tmp_path, task_index=0))
    assert calls == ["startup", "fresh_common_engine"]
    def fail(*args): raise RuntimeError("startup failed")
    monkeypatch.setattr(runner, "startup_subprocess", fail)
    with pytest.raises(RuntimeError, match="startup failed"): runner.train(SimpleNamespace(output=tmp_path, task_index=0))
    assert calls == ["startup", "fresh_common_engine"]


def test_startup_real_cpu_probe_and_bad_input_rejection(tmp_path, monkeypatch):
    task = P.TASKS[0]
    first = batch()
    spec = dict(tasks=P.TASKS, content_hash="fixture", inventory={}, parameters=P.COUNTS,
        recipes=recipes(), environment={"fixture": True},
        donor=dict(input_fingerprints={"0": D.batch_fingerprint(first)}))
    runner = D.DirectEngine(P, D.build_model, D.initial_parity, D.diagnostics)
    monkeypatch.setattr(runner, "gpu_setup", lambda spec: (torch.device("cpu"), {}))
    monkeypatch.setattr(D, "build_loader", lambda *a, **kw: ([first], None))
    monkeypatch.setattr(P, "runtime_environment", lambda: {"fixture": True})
    runner.startup(tmp_path, spec, 0)
    record = P.read_json(P.task_directory(tmp_path, 0)/"startup.json")
    assert record["probe_discarded"] and not record["sustained_benchmark"]
    assert record["initial_parity"]["passed"] and record["training"]["precision"]["passed"]
    assert record["validation_precision"]["passed"]
    assert record["environment_changes_after_probe"] == []
    assert not (P.task_directory(tmp_path, 0)/"last.pt").exists()
    spec["donor"]["input_fingerprints"]["0"] = "wrong"
    with pytest.raises(ValueError, match="batch differs"): runner.startup(tmp_path/"bad", spec, 0)


def test_environment_error_lists_actual_changed_fields(monkeypatch):
    expected = dict(packages={"torch": "2.5.1"}, installed_distributions={"example": "1"})
    actual = dict(packages={"torch": "2.5.1"}, installed_distributions={"example": "2", "new": "3"})
    differences = P.environment_differences(expected, actual)
    assert differences == [
        dict(field="installed_distributions.example", expected="1", actual="2"),
        dict(field="installed_distributions.new", expected="<missing>", actual="3")]
    monkeypatch.setattr(P, "runtime_environment", lambda: actual)
    with pytest.raises(ValueError, match="installed_distributions.example") as error:
        P.require_environment(expected)
    assert "2 fields" in str(error.value) and '"actual": "2"' in str(error.value)
    # Core versions and source hashes remain enforced, not just distributions.
    for field in ("packages", "weaver_sources", "python", "architecture", "torch_cuda"):
        monkeypatch.setattr(P, "runtime_environment", lambda: {field: "changed"})
        with pytest.raises(ValueError, match=field): P.require_environment({field: "original"})


def test_probe_runs_in_child_and_parent_environment_is_rechecked(tmp_path, monkeypatch):
    import json
    import os
    import sys
    import importlib.metadata
    # Model a lazy import exposing additional distribution metadata via sys.path.
    vendor = tmp_path / "vendor"
    info = vendor / "layerwise_probe_only-9.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: layerwise-probe-only\nVersion: 9.0\n")
    driver = tmp_path / "frozen driver.py"
    record = tmp_path / "child.json"
    driver.write_text('''import importlib.metadata, json, os, sys
from pathlib import Path
root = Path(__file__).parent
sys.path.insert(0, str(root / "vendor"))
os.environ["LAYERWISE_PROBE_TEST_ONLY"] = "child"
(root / "child.json").write_text(json.dumps(dict(pid=os.getpid(), executable=sys.executable,
    args=sys.argv[1:], version=importlib.metadata.version("layerwise-probe-only"))))
''')
    monkeypatch.setattr(D, "__file__", str(driver))
    def inventory():
        return {"installed_distributions": {d.metadata["Name"].lower(): d.version
                for d in importlib.metadata.distributions() if d.metadata["Name"] == "layerwise-probe-only"}}
    monkeypatch.setattr(P, "runtime_environment", inventory)
    expected = inventory()
    assert expected == {"installed_distributions": {}}
    monkeypatch.delenv("LAYERWISE_PROBE_TEST_ONLY", raising=False)
    paths = list(sys.path)
    calls = []
    def prepare(output):
        P.require_environment(expected)
        calls.append("prepare")
    def production(self, args):
        P.require_environment(expected)
        calls.append("production_rechecked")
        assert sys.path == paths and "LAYERWISE_PROBE_TEST_ONLY" not in os.environ
    monkeypatch.setattr(D, "prepare", prepare)
    monkeypatch.setattr(E.ScreenEngine, "train", production)
    runner = D.DirectEngine(P, D.build_model, D.initial_parity, D.diagnostics)
    runner.train(SimpleNamespace(output=tmp_path, task_index=1))
    child = json.loads(record.read_text())
    assert child["pid"] != os.getpid() and child["executable"] == sys.executable
    assert child["version"] == "9.0"
    assert child["args"] == ["--phase", "startup", "--output", str(tmp_path), "--task-index", "1"]
    assert calls == ["prepare", "production_rechecked"]


def test_failed_child_never_starts_production(tmp_path, monkeypatch):
    import subprocess
    driver = tmp_path / "failed_probe.py"
    driver.write_text("raise SystemExit(7)\n")
    monkeypatch.setattr(D, "__file__", str(driver))
    monkeypatch.setattr(D, "prepare", lambda _: None)
    production = []
    monkeypatch.setattr(E.ScreenEngine, "train", lambda *a: production.append(True))
    runner = D.DirectEngine(P, D.build_model, D.initial_parity, D.diagnostics)
    with pytest.raises(subprocess.CalledProcessError) as error:
        runner.train(SimpleNamespace(output=tmp_path, task_index=0))
    assert error.value.returncode == 7 and production == []


def test_internal_startup_phase_checks_campaign_and_does_not_recurse(tmp_path, monkeypatch):
    calls = []
    spec = {"fixture": True}
    def load(output):
        calls.append(("load", output))
        return spec
    monkeypatch.setattr(P, "load_campaign", load)
    monkeypatch.setattr(D.ENGINE, "startup", lambda output, loaded, index:
                        calls.append(("startup", loaded, index)))
    monkeypatch.setattr(D.ENGINE, "train", lambda *a: pytest.fail("Child must not call production train"))
    assert D.main(["--phase", "startup", "--output", str(tmp_path), "--task-index", "1"]) == 0
    assert calls == [("load", tmp_path.resolve()), ("startup", spec, 1)]


def test_actual_environment_change_after_child_is_not_cached_or_ignored(tmp_path, monkeypatch):
    current = {"packages": {"torch": "original"}}
    monkeypatch.setattr(P, "runtime_environment", lambda: current)
    expected = copy.deepcopy(current)
    monkeypatch.setattr(D, "prepare", lambda _: P.require_environment(expected))
    runner = D.DirectEngine(P, D.build_model, D.initial_parity, D.diagnostics)
    def child(args):
        current["packages"]["torch"] = "changed"
    monkeypatch.setattr(runner, "startup_subprocess", child)
    production = []
    def parent_train(*args):
        P.require_environment(expected)
        production.append(True)
    monkeypatch.setattr(E.ScreenEngine, "train", parent_train)
    with pytest.raises(ValueError, match="packages.torch"):
        runner.train(SimpleNamespace(output=tmp_path, task_index=0))
    assert not production


def test_pending_control_report_is_not_complete(tmp_path, monkeypatch):
    from part_layerwise_screen import reporting as R
    monkeypatch.setattr(P, "load_campaign", lambda _: {"donor": {}})
    monkeypatch.setattr(R, "refresh_controls", lambda *a: dict(ready=False,
        controls={"EV_RANGER": {"status": "pending_evaluation"}}))
    assert R.report(tmp_path) is False
    assert not (tmp_path/"reports/layerwise_screen_report.json").exists()


def test_complete_report_with_verified_predictions_and_tamper_rejection(tmp_path, monkeypatch):
    from part_layerwise_screen import reporting as R
    from part_screen_common.controls import selected_training, validate_predictions
    from tests.test_part_screen_controls import make_selected, prediction_fixture
    output = tmp_path/"layerwise"
    output.mkdir()
    names = ["PART_RANGER", "EV_RANGER", "PART_LAYERWISE_RANGER", "EV_LAYERWISE_RANGER"]
    records, paths, tasks, parameters = {}, [], [], {}
    for index, name in enumerate(names):
        parent = tmp_path/f"fixture_{index}"
        parent.mkdir()
        run, task, recipe, runtime, sha = make_selected(parent)
        task = dict(task, run_id=name, architecture=("part", "part_ev", "part_layerwise", "part_ev_layerwise")[index])
        recipe = dict(recipe, evaluation_population=100)
        files = prediction_fixture(run, task, sha)
        done = P.read_json(run/"training_complete.json")
        P.write_json(run/"training_complete.json", dict(done, task=task))
        training = selected_training(run, task, "campaign", recipe, runtime)
        evaluation = validate_predictions(run, task, "campaign", sha, files, population=100)
        records[name] = dict(status="verified", training=training, evaluation=evaluation)
        if index >= 2:
            tasks.append(task); paths.append(run); parameters[task["architecture"]] = 1
            P.write_json(run/"startup.json", {"probe_discarded": True})
    spec = dict(content_hash="campaign", tasks=tasks, parameters=parameters, admission=P.ADMISSION,
        donor=dict(runtime=runtime, parameters={"PART_RANGER": 1, "EV_RANGER": 1}),
        recipes={"ranger_short_v1": recipe}, inventory={"splits": {"val": {"files": files}}})
    registry = P.hashed(dict(ready=True, controls={name: records[name] for name in names[:2]}))
    locked = dict(campaign_sha256="campaign", checkpoints=[dict(task=t,
        checkpoint=records[t["run_id"]]["training"]["checkpoint"],
        sha256=records[t["run_id"]]["training"]["checkpoint_sha256"]) for t in tasks])
    P.write_json(output/"selection/locked_checkpoints.json", locked)
    monkeypatch.setattr(P, "load_campaign", lambda _: spec)
    monkeypatch.setattr(P, "task_directory", lambda _, i: paths[i])
    monkeypatch.setattr(R, "refresh_controls", lambda *a: registry)
    assert R.report(output)
    result = P.read_json(output/"reports/layerwise_screen_report.json")
    assert result["complete"] and len(result["models"]) == 4
    assert result["contrasts"]["accuracy"]["ev_by_layerwise_interaction"] == 0
    assert result["paired_accuracy"]["ev_layerwise"]["ties"] == 100
    assert (output/"reports/layerwise_screen_report.md").is_file()
    assert R.report(output)  # stable, repeatable report identity
    with (paths[0]/"full_validation/file_000.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"): R.report(output)
