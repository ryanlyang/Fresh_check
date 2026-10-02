"""Read-only lineage and fail-closed portability checks for SPORC inference."""
from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_relational_part_sporc_inference as D


def _tasks():
    return [
        {
            "run_id": f"model_{model}", "seed": seed,
            "checkpoint": f"runs/model_{model}/seed_{seed}/best_model_val.pt",
            "checkpoint_sha256": "a" * 64, "model_contract_sha256": "b" * 64,
            "relation_families": ["PT", "TRACK", "REGION"],
        }
        for model in range(4) for seed in (101, 202, 303)
    ]


def _original_plan():
    files = [
        {"name": f"class_{label}_{part}.root", "label": label,
         "entries": 1_000_000, "sha256": "c" * 64}
        for part in range(2) for label in reversed(range(10))
    ]
    return D.F.hashed({
        "contract": "original_test", "campaign_sha256": "d" * 64,
        "lock_sha256": "e" * 64, "parent": "/original/campaign",
        "data_dir": "/read-only/test", "tasks": _tasks(), "files": files,
        "event_count": 20_000_000, "chunk_size": 10_000, "batch_size": 64,
        "precision": "FP32", "class_order": [f"class_{i}" for i in range(10)],
        "metric_policy": {"selection": "fixed"}, "rejection_targets": [0.3, 0.5],
        "assignments": D.F.file_assignment(files, 20),
        "performance_gate": False, "used_for_model_selection": False,
    })


def _migration_spec(original):
    return D.F.hashed({
        "source_campaign_sha256": original["campaign_sha256"],
        "environment": {"torch": "2.5.1+cu118", "architecture": "x86_64"},
        "validation_policy": copy.deepcopy(D.VALIDATION_POLICY),
        "math_options": copy.deepcopy(D.MATH_OPTIONS), "jobs": 4,
        "probe_file_indices": D.choose_probe_files(original),
    })


@pytest.mark.parametrize("name", ["scripts/driver.py", "inputs/a.json", "model.pt"])
def test_safe_relative_accepts_artifact_names(name):
    assert D.safe_relative(name) == PurePosixPath(name)


@pytest.mark.parametrize("name", [
    "", ".", "..", "../model.pt", "models/../../model.pt", "/model.pt",
    "//server/share/model.pt", "C:/model.pt", "C:model.pt", "models\\model.pt",
    "models/../model.pt", "model.pt:stream",
])
def test_safe_relative_rejects_escape_or_empty_paths(name):
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        D.safe_relative(name)


@pytest.mark.parametrize("relation", ["same", "child", "ancestor"])
def test_output_cannot_overlap_read_only_inputs(tmp_path, relation):
    source = tmp_path / "input"
    output = {"same": source, "child": source / "migration", "ancestor": tmp_path}[relation]
    with pytest.raises(ValueError, match="separate"):
        D.require_separate_output(output, [source])


def test_output_may_be_a_sibling_of_inputs(tmp_path):
    D.require_separate_output(tmp_path / "migration", [tmp_path / "source", tmp_path / "data"])


def test_authenticate_files_detects_changed_bytes_and_missing_files(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("original", encoding="utf-8")
    expected = {path.name: D.F.digest(path)}
    D.authenticate_files(tmp_path, expected)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact/source changed"):
        D.authenticate_files(tmp_path, expected)
    with pytest.raises(FileNotFoundError):
        D.authenticate_files(tmp_path, {"missing.json": "a" * 64})


def test_authenticate_files_rejects_escape_before_reading(tmp_path):
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        D.authenticate_files(tmp_path, {"../outside.json": "a" * 64})


def test_probe_selection_is_deterministic_class_balanced_and_first_file():
    plan = _original_plan()
    before = copy.deepcopy(plan)
    selected = D.choose_probe_files(plan)
    assert selected == list(reversed(range(10)))
    assert [plan["files"][i]["label"] for i in selected] == list(range(10))
    assert len(set(selected)) == 10
    assert D.choose_probe_files(plan) == selected
    assert plan == before


def test_probe_selection_requires_every_class():
    plan = _original_plan()
    plan["files"] = [row for row in plan["files"] if row["label"] != 4]
    with pytest.raises(ValueError, match="lacks a class"):
        D.choose_probe_files(plan)


def test_probe_selection_does_not_skip_a_short_first_file():
    plan = _original_plan()
    plan["files"][0]["entries"] = D.VALIDATION_POLICY["golden_events_per_class"] - 1
    with pytest.raises(ValueError, match="too short"):
        D.choose_probe_files(plan)


def test_logits_accept_exact_and_within_absolute_or_relative_tolerance():
    expected = np.zeros((2, 10), dtype=np.float32)
    assert D.assert_logits(expected, expected) == 0
    actual = expected.copy()
    actual[0, 0] = D.VALIDATION_POLICY["logit_atol"] / 2
    assert D.assert_logits(actual, expected) == pytest.approx(actual[0, 0])
    # The receipt records maximum absolute error, but parity uses atol + rtol.
    expected.fill(100)
    actual = expected + np.float32(0.001)
    assert D.assert_logits(actual, expected) > D.VALIDATION_POLICY["logit_atol"]


@pytest.mark.parametrize("side", ["actual", "expected"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_logits_reject_nonfinite_values_on_either_side(side, bad_value):
    arrays = {"actual": np.zeros((2, 10)), "expected": np.zeros((2, 10))}
    arrays[side][0, 0] = bad_value
    with pytest.raises(ValueError, match="nonfinite"):
        D.assert_logits(**arrays)


def test_logits_reject_broadcastable_shape_difference():
    with pytest.raises(ValueError, match="shape"):
        D.assert_logits(np.zeros((1, 10)), np.zeros((2, 10)))


def test_logits_reject_outside_frozen_tolerance():
    with pytest.raises(ValueError, match="tolerances are frozen"):
        D.assert_logits(np.full((2, 10), 1e-3), np.zeros((2, 10)))


def _trimmer_model(counter, *, enabled):
    import torch
    model = torch.nn.Module()
    model.branch = torch.nn.Module()
    model.branch.trimmer = torch.nn.Module()
    model.branch.trimmer._counter = counter
    model.branch.trimmer.enabled = enabled
    # Neither a differently named counter nor a counterless trimmer belongs
    # to the transient-state inventory.
    model.other = torch.nn.Module()
    model.other._counter = 99
    model.counterless = torch.nn.Module()
    model.counterless.trimmer = torch.nn.Module()
    return model.eval()


@pytest.mark.parametrize("tensor_counter", [False, True])
def test_trimmer_state_snapshots_only_actual_counters_without_aliasing(tensor_counter):
    import torch
    counter = torch.tensor(10, dtype=torch.int64) if tensor_counter else 10
    model = _trimmer_model(counter, enabled=True)
    state = D.trimmer_state(model)
    assert set(state) == {"branch.trimmer"}
    assert int(state["branch.trimmer"]) == 10
    if tensor_counter:
        assert state["branch.trimmer"].data_ptr() != counter.data_ptr()
        counter.add_(7)
    else:
        model.branch.trimmer._counter += 7
    assert int(state["branch.trimmer"]) == 10


@pytest.mark.parametrize("tensor_counter", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_probe_trimmer_state_updates_reset_cache_and_preserves_flags(tensor_counter, enabled):
    import torch
    counter = torch.tensor(0, dtype=torch.int64) if tensor_counter else 0
    model = _trimmer_model(counter, enabled=enabled)
    trimmer = model.branch.trimmer
    D.F.reset_transient_trimmers([model])
    assert int(trimmer._rpt_full_test_initial_counter) == 0
    state = {"branch.trimmer": torch.tensor(10) if tensor_counter else 10}
    D.probe_trimmer_state(model, state)
    assert int(trimmer._counter) == 10
    assert int(trimmer._rpt_full_test_initial_counter) == 10
    if tensor_counter:
        assert trimmer._counter.device.type == "cpu"
        assert trimmer._counter.dtype == torch.int64
        assert trimmer._counter.data_ptr() != state["branch.trimmer"].data_ptr()
        assert trimmer._rpt_full_test_initial_counter.data_ptr() != trimmer._counter.data_ptr()
        state["branch.trimmer"].add_(100)
        trimmer._counter.add_(1)
    else:
        state["branch.trimmer"] = 110
        trimmer._counter += 1
    # This is the reset performed by infer_arrays: it must now restore the
    # captured warm state, not the fresh zero previously cached on this model.
    D.F.reset_transient_trimmers([model])
    assert int(trimmer._counter) == 10
    assert int(trimmer._rpt_full_test_initial_counter) == 10
    assert trimmer.enabled is enabled
    assert model.training is False and trimmer.training is False
    assert model.other._counter == 99


@pytest.mark.parametrize("state", [{}, {"wrong.trimmer": 10}, {"branch.trimmer": 10, "extra.trimmer": 10}])
def test_probe_trimmer_state_rejects_inventory_mismatch_without_mutation(state):
    model = _trimmer_model(2, enabled=False)
    with pytest.raises(ValueError, match="inventories differ"):
        D.probe_trimmer_state(model, state)
    assert model.branch.trimmer._counter == 2
    assert model.branch.trimmer.enabled is False
    assert not hasattr(model.branch.trimmer, "_rpt_full_test_initial_counter")


def test_new_plan_preserves_models_population_chunk_boundaries_and_original():
    original = _original_plan()
    before = copy.deepcopy(original)
    spec = _migration_spec(original)
    backend = {"content_hash": "f" * 64}
    plan = D.new_plan(spec, original, backend)
    for field in (
        "tasks", "files", "event_count", "chunk_size", "batch_size", "data_dir",
        "parent", "lock_sha256", "precision", "class_order", "metric_policy",
        "rejection_targets", "performance_gate", "used_for_model_selection",
    ):
        assert plan[field] == original[field]
    assert plan["assignments"] == D.F.file_assignment(original["files"], 4)
    assert plan["assignments"] != original["assignments"]
    assert [i for group in plan["assignments"] for i in group] == list(range(20))
    for row in original["files"]:
        assert D.F.chunks(row, plan) == D.F.chunks(row, original)
    assert plan["contract"] == D.VERSION + "_test"
    assert plan["campaign_sha256"] == spec["content_hash"]
    assert plan["source_campaign_sha256"] == original["campaign_sha256"]
    assert plan["source_test_plan_sha256"] == original["content_hash"]
    assert plan["backend_manifest_sha256"] == backend["content_hash"]
    assert plan["environment"] == spec["environment"]
    assert plan["validation_policy"] == D.VALIDATION_POLICY
    assert plan["math_options"] == D.MATH_OPTIONS
    assert plan == D.F.hashed({k: v for k, v in plan.items() if k != "content_hash"})
    assert plan["content_hash"] != original["content_hash"]
    assert original == before
    plan["tasks"][0]["relation_families"].append("mutated")
    plan["files"][0]["entries"] = 1
    assert original == before


def test_new_plan_binds_backend_identity():
    original = _original_plan()
    spec = _migration_spec(original)
    assert D.new_plan(spec, original, {"content_hash": "a" * 64})["content_hash"] != D.new_plan(
        spec, original, {"content_hash": "b" * 64})["content_hash"]


@pytest.fixture
def receipt_inputs(tmp_path):
    original = _original_plan()
    spec = _migration_spec(original)
    plan = D.new_plan(spec, original, {"content_hash": "f" * 64})
    gpu = {"name": "A100", "capability": [8, 0], "total_memory": 40 * 1024**3}
    receipt = {
        "contract": D.VERSION + "_validation", "passed": True,
        "campaign_sha256": spec["content_hash"], "plan_sha256": plan["content_hash"],
        "policy": copy.deepcopy(D.VALIDATION_POLICY), "gpu": gpu,
        "probe_file_indices": list(spec["probe_file_indices"]),
        "new_model_order": [[t["run_id"], t["seed"]] for t in plan["tasks"]],
        "tree_checks": [
            {"file_index": index, "entry": entry, "passed": True,
             "topology_and_categories_exact": True, "continuous_shapes_exact": True,
             "continuous_values_finite": True, "maximum_continuous_absolute_error": 1e-8,
             "continuous_absolute_tolerance": D.VALIDATION_POLICY["tree_continuous_atol"]}
            for index in spec["probe_file_indices"]
            for entry in range(D.VALIDATION_POLICY["tree_events_per_class"])
        ],
        "golden_checks": [
            {"file_index": index, "passed": True, "golden_chunk_sha256": "1" * 64,
             "per_model_max_absolute_error": {f"logits_{i:02d}": 1e-8 for i in range(12)}}
            for index in spec["probe_file_indices"]
        ],
        "new_cpu_gpu_checks": [
            {"run_id": t["run_id"], "seed": t["seed"], "passed": True,
             "maximum_absolute_error": 1e-8,
             "per_state_maximum_absolute_error": {"fresh": 1e-8, "after_ten_full_batches": 1e-8}}
            for t in plan["tasks"]
        ],
        "new_timings": [{"label": i, "events": 640, "seconds": 2.0} for i in range(10)],
    }
    return tmp_path, spec, plan, gpu, receipt


def _save_receipt(inputs):
    output, _, _, _, receipt = inputs
    frozen = D.F.hashed(receipt)
    D.F.write_json(output / "portability_validation.json", frozen)
    return frozen


def test_validation_receipt_accepts_bound_passed_probe(receipt_inputs):
    output, spec, plan, gpu, _ = receipt_inputs
    frozen = _save_receipt(receipt_inputs)
    assert D.validation_receipt(output, spec, plan, device=gpu) == frozen
    assert D.validation_receipt(output, spec, plan) == frozen


def test_validation_receipt_missing_fails_closed(receipt_inputs):
    output, spec, plan, _, _ = receipt_inputs
    with pytest.raises(FileNotFoundError):
        D.validation_receipt(output, spec, plan)


@pytest.mark.parametrize("field,value", [
    ("contract", "wrong"), ("passed", False), ("passed", 1),
    ("campaign_sha256", "0" * 64), ("plan_sha256", "0" * 64),
    ("policy", {}), ("new_model_order", []), ("probe_file_indices", []),
    ("golden_checks", []), ("new_cpu_gpu_checks", []), ("new_timings", []),
])
def test_validation_receipt_rejects_stale_outer_bindings(receipt_inputs, field, value):
    output, spec, plan, gpu, receipt = receipt_inputs
    receipt[field] = value
    _save_receipt(receipt_inputs)
    with pytest.raises(ValueError):
        D.validation_receipt(output, spec, plan, device=gpu)


def test_validation_receipt_rejects_gpu_identity_change(receipt_inputs):
    output, spec, plan, gpu, _ = receipt_inputs
    _save_receipt(receipt_inputs)
    with pytest.raises(ValueError):
        D.validation_receipt(output, spec, plan, device={**gpu, "total_memory": gpu["total_memory"] * 2})


@pytest.mark.parametrize("field,value", [
    ("passed", False), ("topology_and_categories_exact", False),
    ("continuous_shapes_exact", False), ("continuous_values_finite", False),
    ("maximum_continuous_absolute_error", -1.0),
    ("maximum_continuous_absolute_error", 3e-6),
    ("continuous_absolute_tolerance", 3e-6), ("entry", 99), ("file_index", 99),
])
def test_validation_receipt_rejects_invalid_tree_checks(receipt_inputs, field, value):
    output, spec, plan, _, receipt = receipt_inputs
    receipt["tree_checks"][0][field] = value
    _save_receipt(receipt_inputs)
    with pytest.raises(ValueError):
        D.validation_receipt(output, spec, plan)


@pytest.mark.parametrize("change", ["missing", "duplicate", "empty"])
def test_validation_receipt_requires_full_tree_coverage(receipt_inputs, change):
    output, spec, plan, _, receipt = receipt_inputs
    if change == "missing":
        del receipt["tree_checks"]
    elif change == "duplicate":
        receipt["tree_checks"][1] = copy.deepcopy(receipt["tree_checks"][0])
    else:
        receipt["tree_checks"] = []
    _save_receipt(receipt_inputs)
    with pytest.raises((ValueError, KeyError)):
        D.validation_receipt(output, spec, plan)


@pytest.mark.parametrize("change", ["file", "passed", "model_coverage", "negative"])
def test_validation_receipt_requires_valid_golden_checks(receipt_inputs, change):
    output, spec, plan, _, receipt = receipt_inputs
    check = receipt["golden_checks"][0]
    if change == "file":
        check["file_index"] = receipt["golden_checks"][1]["file_index"]
    elif change == "passed":
        check["passed"] = False
    elif change == "model_coverage":
        del check["per_model_max_absolute_error"]["logits_11"]
    else:
        check["per_model_max_absolute_error"]["logits_00"] = -1
    _save_receipt(receipt_inputs)
    with pytest.raises(ValueError):
        D.validation_receipt(output, spec, plan)


@pytest.mark.parametrize("field,value", [
    ("run_id", "other"), ("seed", 404), ("passed", False),
    ("maximum_absolute_error", -1),
])
def test_validation_receipt_requires_ordered_passed_cpu_checks(receipt_inputs, field, value):
    output, spec, plan, _, receipt = receipt_inputs
    receipt["new_cpu_gpu_checks"][0][field] = value
    _save_receipt(receipt_inputs)
    with pytest.raises(ValueError):
        D.validation_receipt(output, spec, plan)


@pytest.mark.parametrize("change", ["missing_warm", "missing_fresh", "missing_states", "extra_state", "negative_warm", "wrong_maximum"])
def test_validation_receipt_requires_both_trimmer_states(receipt_inputs, change):
    output, spec, plan, _, receipt = receipt_inputs
    row = receipt["new_cpu_gpu_checks"][0]
    states = row["per_state_maximum_absolute_error"]
    if change == "missing_warm":
        del states["after_ten_full_batches"]
    elif change == "missing_fresh":
        del states["fresh"]
    elif change == "missing_states":
        del row["per_state_maximum_absolute_error"]
    elif change == "extra_state":
        states["unapproved"] = 1e-8
    elif change == "negative_warm":
        states["after_ten_full_batches"] = -1
    else:
        states["after_ten_full_batches"] = 2e-8
    _save_receipt(receipt_inputs)
    with pytest.raises(ValueError, match="CPU/GPU parity coverage"):
        D.validation_receipt(output, spec, plan)


def test_validation_receipt_accepts_maximum_across_fresh_and_warm_states(receipt_inputs):
    output, spec, plan, _, receipt = receipt_inputs
    row = receipt["new_cpu_gpu_checks"][0]
    row["per_state_maximum_absolute_error"]["after_ten_full_batches"] = 2e-8
    row["maximum_absolute_error"] = 2e-8
    saved = _save_receipt(receipt_inputs)
    assert D.validation_receipt(output, spec, plan) == saved


def test_validation_receipt_rejects_nonfinite_warm_state_error(receipt_inputs, monkeypatch):
    output, spec, plan, _, receipt = receipt_inputs
    receipt["new_cpu_gpu_checks"][0]["per_state_maximum_absolute_error"]["after_ten_full_batches"] = float("nan")
    monkeypatch.setattr(D.F, "read_json", lambda path: copy.deepcopy(receipt))
    with pytest.raises(ValueError, match="CPU/GPU parity coverage"):
        D.validation_receipt(output, spec, plan)


@pytest.mark.parametrize("field,value", [
    ("label", 1), ("events", 639), ("seconds", 0),
    ("seconds", -1),
])
def test_validation_receipt_requires_class_balanced_finite_timings(receipt_inputs, field, value):
    output, spec, plan, _, receipt = receipt_inputs
    receipt["new_timings"][0][field] = value
    _save_receipt(receipt_inputs)
    with pytest.raises(ValueError):
        D.validation_receipt(output, spec, plan)


@pytest.mark.parametrize("section", ["tree_checks", "golden_checks", "new_cpu_gpu_checks", "new_timings"])
def test_validation_receipt_rejects_nonfinite_decoded_checks(receipt_inputs, monkeypatch, section):
    output, spec, plan, _, receipt = receipt_inputs
    row = receipt[section][0]
    if section == "golden_checks":
        row["per_model_max_absolute_error"]["logits_00"] = float("nan")
    else:
        field = {"tree_checks": "maximum_continuous_absolute_error",
                 "new_cpu_gpu_checks": "maximum_absolute_error", "new_timings": "seconds"}[section]
        row[field] = float("inf")
    # Normal JSON authentication rejects these too; exercise the receipt's
    # own numerical guard independently of that earlier serialization check.
    monkeypatch.setattr(D.F, "read_json", lambda path: copy.deepcopy(receipt))
    with pytest.raises(ValueError):
        D.validation_receipt(output, spec, plan)


@pytest.fixture
def load_spec_inputs(tmp_path, monkeypatch):
    from scripts import run_relational_part_offline_ablations as A
    output = tmp_path / "migration"
    frozen = output / "source"
    source_campaign = tmp_path / "source_campaign"
    environment = {"architecture": "x86_64", "torch": "2.5.1+cu118"}
    snapshot = {"teacher_logit_reco/model.py": "a" * 64, "jetclass_fixed_hlt.py": "b" * 64}
    source = {
        "content_hash": "c" * 64, "models": {"model": {}}, "seeds": [101, 202, 303],
        "parent": str(tmp_path / "parent"), "previous_evaluation": str(tmp_path / "previous"),
    }
    original = {"content_hash": "d" * 64}
    reference = _original_plan()
    spec = {
        "contract": D.VERSION, "schema_version": 1, "output": str(output),
        "environment": environment, "validation_policy": copy.deepcopy(D.VALIDATION_POLICY),
        "math_options": copy.deepcopy(D.MATH_OPTIONS), "jobs": 40, "inference_only": True,
        "performance_gate": False, "source_files": {**snapshot, "scripts/driver.py": "e" * 64},
        "source_campaign": str(source_campaign), "source_campaign_sha256": source["content_hash"],
        "source_test_plan_sha256": original["content_hash"], "models": source["models"],
        "seeds": source["seeds"], "original_parent": source["parent"],
        "previous_evaluation": source["previous_evaluation"],
        "reference_plan_sha256": reference["content_hash"], "reference_report_sha256": "1" * 64,
        "probe_file_indices": D.choose_probe_files(reference),
    }
    artifacts = {
        "sporc_campaign.json": spec,
        "reference_plan.json": reference,
        "reference_report.json": {"content_hash": spec["reference_report_sha256"]},
    }
    authenticated = []
    monkeypatch.setattr(D, "REPO", frozen)
    monkeypatch.setattr(D, "runtime_environment", lambda: copy.deepcopy(environment))
    monkeypatch.setattr(D, "authenticate_files", lambda root, expected: authenticated.append((root, expected)))
    monkeypatch.setattr(D.F, "read_json", lambda path: copy.deepcopy(artifacts[Path(path).name]))
    monkeypatch.setattr(D.F, "source_snapshot", lambda root: dict(snapshot))
    monkeypatch.setattr(A, "load_campaign", lambda root, *, runtime: copy.deepcopy(source) if runtime is False else None)
    monkeypatch.setattr(A, "test_plan", lambda root, campaign: copy.deepcopy(original))
    return output, spec, source, original, artifacts, authenticated


def test_load_spec_authenticates_frozen_source_and_original_lock(load_spec_inputs):
    output, spec, source, original, _, authenticated = load_spec_inputs
    assert D.load_spec(output) == (spec, source, original)
    assert authenticated == [(output / "source", spec["source_files"])]


def test_load_spec_rejects_changed_runtime(load_spec_inputs, monkeypatch):
    output, *_ = load_spec_inputs
    monkeypatch.setattr(D, "runtime_environment", lambda: {"architecture": "aarch64"})
    with pytest.raises(ValueError, match="source/runtime/validation"):
        D.load_spec(output)


def test_load_spec_rejects_execution_outside_frozen_source(load_spec_inputs, monkeypatch):
    output, *_ = load_spec_inputs
    monkeypatch.setattr(D, "REPO", output / "interactive_checkout")
    with pytest.raises(ValueError, match="source/runtime/validation"):
        D.load_spec(output)


def test_load_spec_rejects_added_model_source(load_spec_inputs, monkeypatch):
    output, spec, *_ = load_spec_inputs
    monkeypatch.setattr(D.F, "source_snapshot", lambda root: {
        **{k: v for k, v in spec["source_files"].items() if not k.startswith("scripts/")},
        "teacher_logit_reco/untracked.py": "9" * 64,
    })
    with pytest.raises(ValueError, match="source inventory"):
        D.load_spec(output)


@pytest.mark.parametrize("field", ["source_campaign_sha256", "source_test_plan_sha256", "original_parent"])
def test_load_spec_rejects_changed_original_lock_binding(load_spec_inputs, field):
    output, spec, *_ = load_spec_inputs
    spec[field] = "changed"
    with pytest.raises(ValueError, match="Original locked campaign"):
        D.load_spec(output)


def test_load_spec_rejects_changed_reference_artifact(load_spec_inputs):
    output, _, _, _, artifacts, _ = load_spec_inputs
    artifacts["reference_report.json"]["content_hash"] = "9" * 64
    with pytest.raises(ValueError, match="reference artifact"):
        D.load_spec(output)


def test_load_spec_rejects_changed_probe_selection(load_spec_inputs):
    output, spec, *_ = load_spec_inputs
    spec["probe_file_indices"] = list(reversed(spec["probe_file_indices"]))
    with pytest.raises(ValueError, match="probe"):
        D.load_spec(output)


def test_bootstrap_rejects_output_within_original_parent_before_authentication(tmp_path, monkeypatch):
    source_campaign = tmp_path / "source_campaign"
    parent = tmp_path / "parent"
    spec = {"contract": "relational_part_offline_factorial_followup_v1",
            "parent": str(parent), "previous_evaluation": str(tmp_path / "previous"),
            "test_data": str(tmp_path / "test_data")}
    monkeypatch.setattr(D.F, "read_json", lambda path: spec)
    def unexpected_authentication(*args):
        pytest.fail("Input/output overlap must be rejected before authentication/copying")
    monkeypatch.setattr(D, "authenticate_files", unexpected_authentication)
    output = parent / "migration"
    with pytest.raises(ValueError, match="separate"):
        D.bootstrap(SimpleNamespace(source_campaign=source_campaign, output=output, jobs=40))
    assert not output.exists()


def test_bootstrap_does_not_overwrite_copied_original_source_with_adapter(tmp_path, monkeypatch):
    source_campaign = tmp_path / "source_campaign"
    output = tmp_path / "migration"
    collision = D.ADAPTER_FILES[0]
    source_file = source_campaign / "source" / collision
    source_file.parent.mkdir(parents=True)
    source_file.write_text("authenticated original", encoding="utf-8")
    reference = _original_plan()
    report = D.F.hashed({"plan_sha256": reference["content_hash"]})
    environment = {"architecture": "x86_64", "torch_cuda_build": "11.8",
                   "weaver_particle_transformer_sha256": "a" * 64}
    spec = D.F.hashed({
        "contract": "relational_part_offline_factorial_followup_v1",
        "parent": str(tmp_path / "parent"), "previous_evaluation": str(tmp_path / "previous"),
        "test_data": str(tmp_path / "test_data"), "environment": environment,
        "source_files": {collision: D.F.digest(source_file)},
        "reference_plan_sha256": reference["content_hash"],
        "reference_report_sha256": report["content_hash"],
    })
    source_plan = {"campaign_sha256": spec["content_hash"], "event_count": 20_000_000}
    artifacts = {"campaign_spec.json": spec, "evaluation_plan.json": source_plan,
                 "reference_plan.json": reference, "reference_report.json": report}
    monkeypatch.setattr(D.F, "read_json", lambda path: copy.deepcopy(artifacts[Path(path).name]))
    monkeypatch.setattr(D, "runtime_environment", lambda: environment)
    monkeypatch.setattr(D, "check_storage", lambda *args: None)
    with pytest.raises(ValueError, match="overwrite original source"):
        D.bootstrap(SimpleNamespace(source_campaign=source_campaign, output=output, jobs=4))
    assert source_file.read_text(encoding="utf-8") == "authenticated original"
    if (output / "source" / collision).exists():
        assert (output / "source" / collision).read_bytes() == source_file.read_bytes()
    assert not (output / "sporc_campaign.json").exists()


def test_cli_help_runs_without_a_gpu():
    result = subprocess.run([sys.executable, str(Path(D.__file__)), "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--source-campaign" in result.stdout
    assert "validate" in result.stdout and "infer" in result.stdout


def test_cli_check_routes_to_read_only_authentication(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(sys, "argv", ["driver", "check", "--output", str(tmp_path)])
    monkeypatch.setattr(D, "load_spec", lambda output: checked.append(output))
    monkeypatch.setattr(D.F, "progress", lambda *args, **kwargs: None)
    assert D.main() == 0
    assert checked == [tmp_path.resolve()]
