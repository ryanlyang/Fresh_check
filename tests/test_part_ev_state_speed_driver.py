"""CPU-only regression for lazy-import metadata changes during benchmarks."""
from types import SimpleNamespace

import pytest

from scripts import run_part_ev_state_speed as R


@pytest.fixture
def flow(tmp_path, monkeypatch):
    task = R.P.TASKS[0]
    spec = dict(tasks=R.P.TASKS, inventory={}, content_hash="fixture")
    saved_environment = dict(installed_distributions={"torch": "frozen"})
    state = dict(expanded=False, startup_changed=False, loads=0, parity=0,
                 benchmark=0, speed=0, shutdown=0, validated_speed=0)

    def environment():
        packages = dict(saved_environment["installed_distributions"])
        if state["expanded"]:
            packages.update(autocommand="2.2.2", **{"jaraco.context": "6.1.0"})
        if state["startup_changed"]:
            packages["torch"] = "changed"
        return dict(installed_distributions=packages)

    def load_campaign(output, *, check_environment=True):
        assert output == tmp_path
        assert check_environment  # No bypass via check_environment=False.
        state["loads"] += 1
        if environment() != saved_environment:
            raise ValueError("Use the unchanged donor environment")
        return spec

    def parity(engine, received_spec, candidate, received_task, batch, device):
        assert received_spec is spec and received_task is task
        assert state["expanded"]
        state["parity"] += 1
        return dict(passed=True)

    def make_driver(engine, callback):
        def benchmark(args):
            # Match the historical driver's strict check before setup, followed
            # by optimizer imports expanding the visible distribution metadata.
            state["benchmark"] += 1
            assert R.P.load_campaign(args.output) is spec
            state["expanded"] = True
            callback("candidate", task, "batch", "device")
            R.P.write_json(tmp_path / "benchmarks" / (task["run_id"] + ".json"),
                           dict(input_fingerprints={"0": "fingerprint"}))

        return SimpleNamespace(benchmark=benchmark, gpu_setup=lambda _: ("device", {}))

    def speed_probe(engine, received_spec, received_task, batch, device, runtime):
        assert received_spec is spec and received_task is task
        assert state["expanded"] and state["parity"] == 1
        state["speed"] += 1
        return dict(models={"timing": "fixture"})

    def validate_speed_report(*args):
        state["validated_speed"] += 1

    monkeypatch.setattr(R.P, "runtime_environment", environment)
    monkeypatch.setattr(R.P, "load_campaign", load_campaign)
    monkeypatch.setattr(R, "benchmark_driver", make_driver)
    monkeypatch.setattr(R.B, "initial_parity", parity)
    monkeypatch.setattr(R.B, "speed_probe", speed_probe)
    monkeypatch.setattr(R.B, "validate_speed_report", validate_speed_report)
    monkeypatch.setattr(R.D, "seed_everything", lambda _: None)
    monkeypatch.setattr(R.D, "build_loader", lambda *a, **kw: (["batch"], None))
    monkeypatch.setattr(R.D, "batch_fingerprint", lambda _: "fingerprint")
    monkeypatch.setattr(R.D, "shutdown_loader", lambda _: state.update(shutdown=state["shutdown"] + 1))
    monkeypatch.setattr(R.D, "progress", lambda *a, **kw: None)
    return state, tmp_path / "benchmarks" / (task["run_id"] + "_speed.json")


@pytest.mark.parametrize("existing_speed", [False, True])
def test_checked_spec_survives_optimizer_metadata_expansion(tmp_path, flow, existing_speed):
    state, path = flow
    if existing_speed:
        payload = dict(models={"timing": "previous"})
        payload["content_hash"] = R.P.digest(payload)
        R.P.write_json(path, payload)
        original = path.read_bytes()
    assert R.main(["--phase", "benchmark", "--output", str(tmp_path), "--task-index", "0"]) == 0
    assert state["loads"] == 2  # CLI and historical probe, both before optimizer.
    assert state["parity"] == 1
    assert state["speed"] == state["shutdown"] == int(not existing_speed)
    assert state["validated_speed"] == int(existing_speed)
    if existing_speed:
        assert path.read_bytes() == original
    else:
        assert R.P.read_json(path)["models"] == {"timing": "fixture"}


def test_real_startup_environment_change_still_blocks_benchmark(tmp_path, flow):
    state, path = flow
    state["startup_changed"] = True
    with pytest.raises(ValueError, match="unchanged donor environment"):
        R.main(["--phase", "benchmark", "--output", str(tmp_path), "--task-index", "0"])
    assert state["loads"] == 1
    assert state["benchmark"] == state["parity"] == state["speed"] == 0
    assert not path.exists()


def test_numerical_failure_still_blocks_timing(tmp_path, flow, monkeypatch):
    state, path = flow

    def fail(*args):
        raise R.B.ParityError("nonzero message mismatch", dict(passed=False))

    monkeypatch.setattr(R.B, "initial_parity", fail)
    with pytest.raises(R.B.ParityError, match="nonzero message mismatch"):
        R.main(["--phase", "benchmark", "--output", str(tmp_path), "--task-index", "0"])
    assert state["loads"] == 2
    assert state["speed"] == 0
    assert not path.exists()
