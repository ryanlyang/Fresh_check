"""Same-runtime 24-checkpoint evaluation, lineage, and no score-based gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_relational_part_sporc_unified as U
from scripts import relational_part_sporc_unified_reporting as R
F = U.F


def seal(value):
    return F.hashed({k: v for k, v in value.items() if k != "content_hash"})


def overwrite(path, value):
    path.write_bytes(F.canonical(seal(value)))


def tasks(names, origin=None):
    return [dict(run_id=n, seed=s, checkpoint=f"runs/{n}/seed_{s}/best_model_val.pt",
                 checkpoint_sha256="a" * 64, model_contract_sha256="b" * 64, relation_families=[],
                 **({"origin": origin} if origin else {})) for n in names for s in F.SEEDS]


def plans():
    rows = [dict(label=i, entries=10, name=f"class_{i}.root", sha256=str(i) * 64) for i in range(10)]
    protocol = dict(files=rows, event_count=100, chunk_size=6, batch_size=64,
                    class_order=list(F.CLASS_NAMES), metric_policy={"ROC": "global"}, rejection_targets=list(F.TARGETS))
    old = F.hashed(dict(protocol, tasks=tasks(F.MODELS), contract=F.VERSION))
    follow = F.hashed(dict(protocol, tasks=tasks(U.FOLLOWUP_MODELS), contract="followup"))
    return old, follow


def comparison(events=640, delta=0):
    return U.compare_logits(np.full((events, 10), delta, np.float32), np.zeros((events, 10), np.float32))


def valid_receipt(spec, plan):
    # Synthetic ten-class coverage; test fixture files are smaller than real
    # production probes. Their indices, not small entry counts, define coverage.
    indices = list(range(10))
    return F.hashed(dict(contract=U.VERSION + "_validation", campaign_sha256=spec["content_hash"],
        plan_sha256=plan["content_hash"], passed=True, policy=U.POLICY,
        model_order=[[t["run_id"], t["seed"]] for t in plan["tasks"]],
        gpu=dict(name="synthetic A100", capability=[8, 0], total_memory=1), cross_platform_equivalence_claimed=False,
        gpu_checks=[dict(file_index=i, model_index=m, repeatability=comparison()) for i in indices for m in range(24)],
        cpu_gpu_diagnostics=[dict(file_index=i, model_index=m, state=s, comparison=comparison(8, delta=1.0))
                             for i in indices for m in range(24) for s in U.POLICY["states"]],
        tree_checks=[dict(file_index=i, entry=j, passed=True, topology_and_categories_exact=True,
                         continuous_shapes_exact=True, continuous_values_finite=True, maximum_continuous_absolute_error=0)
                     for i in indices for j in range(4)]))


@pytest.fixture
def campaign(tmp_path):
    old, followup = plans()
    all_tasks = U.combined_tasks(old, followup)
    # Reference scores deliberately differ from fresh SPORC metrics; there are
    # NO historical prediction files in this fixture.
    labels = np.repeat(np.arange(10), 10)
    reference_metrics = F.calculate_metrics(np.eye(10, dtype=np.float32)[labels] * 10, labels)
    old_rows = [F.hashed(dict(contract=F.VERSION + "_result", task=t, plan_sha256=old["content_hash"],
                            used_for_model_selection=False, file_completion_hashes=["c" * 64] * 10,
                            metrics=reference_metrics)) for t in old["tasks"]]
    old_report = F.hashed(dict(contract=F.VERSION + "_report", plan_sha256=old["content_hash"],
                             event_count_per_model=100, per_model_results=old_rows))
    root = tmp_path / "unified"
    F.write_json(root / "reference_plan.json", old)
    F.write_json(root / "reference_report.json", old_report)
    spec = F.hashed(dict(contract=U.VERSION, tasks=all_tasks, models=list(F.MODELS + U.FOLLOWUP_MODELS),
                        reference_plan_sha256=old["content_hash"], reference_report_sha256=old_report["content_hash"]))
    plan = U.new_plan({**spec, "jobs": 10, "environment": {}, "source_campaign_sha256": "d" * 64},
                     followup, {"content_hash": "e" * 64})
    # new_plan takes the already-authenticated manifest hash unchanged.
    plan = seal(dict(plan, performance_gate=False, used_for_model_selection=False))
    receipt = valid_receipt(spec, plan)
    F.write_json(root / "runtime_validation.json", receipt)
    rng, expected, hashes = np.random.default_rng(321), [], []
    for i, row in enumerate(plan["files"]):
        records = []
        for start, stop in F.chunks(row, plan):
            values = {f"logits_{m:02d}": rng.normal(size=(stop - start, 10)).astype(np.float32) for m in range(24)}
            expected.append(values)
            path = F.chunk_path(root / "test", i, start)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **values)
            record = F.hashed(dict(metadata=F.chunk_metadata(plan, i, start, stop), npz_sha256=F.digest(path),
                                   runtime_validation_sha256=receipt["content_hash"]))
            F.write_json(path.with_suffix(".json"), record)
            records.append(record["content_hash"])
        done = F.hashed(dict(contract=plan["contract"] + "_file_complete", plan_sha256=plan["content_hash"], file_index=i, chunks=records))
        F.write_json(root / f"test/predictions/file_{i:03d}/complete.json", done)
        hashes.append(done["content_hash"])
    return root, spec, plan, receipt, expected, hashes


def test_combined_task_roots_are_explicit_and_ids_unchanged():
    old, follow = plans()
    result = U.combined_tasks(old, follow)
    assert len(result) == 24
    assert {t["origin"] for t in result[:12]} == {"historical"}
    assert {t["origin"] for t in result[12:]} == {"followup"}
    assert [{k: v for k, v in t.items() if k != "origin"} for t in result] == old["tasks"] + follow["tasks"]


@pytest.mark.parametrize("change", ["duplicate", "seed", "files", "metric_policy", "event_count"])
def test_combined_tasks_reject_incomparable_or_incomplete_inputs(change):
    old, follow = plans()
    if change == "duplicate":
        follow["tasks"][0] = deepcopy(follow["tasks"][1])
    elif change == "seed":
        follow["tasks"][0]["seed"] = 999
    else:
        follow[change] = None
    with pytest.raises(ValueError):
        U.combined_tasks(old, follow)


def test_cross_device_drift_is_observational_but_repeatability_is_required():
    actual, reference = np.ones((8, 10), np.float32), np.zeros((8, 10), np.float32)
    assert U.compare_logits(actual, reference)["passed"] is False
    with pytest.raises(ValueError, match="Same-GPU"):
        U.repeatability(actual, reference)
    assert U.repeatability(actual, actual)["passed"] is True
    actual[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        U.compare_logits(actual, reference)


def test_model_loader_routes_each_task_to_its_original_checkpoint_root(monkeypatch):
    calls = []
    monkeypatch.setattr(U.D, "load_models", lambda spec, source, ts, dev, historical=False:
                        calls.append((ts[0]["run_id"], historical)) or [ts[0]["run_id"]])
    original, follow = plans()
    values = U.load_models({}, {}, U.combined_tasks(original, follow), "cpu")
    assert len(values) == 24 and [v[1] for v in calls] == [True] * 12 + [False] * 12


def test_receipt_does_not_gate_cpu_gpu_numerical_differences(campaign, monkeypatch):
    root, spec, plan, receipt, _, _ = campaign
    monkeypatch.setattr(U.D, "choose_probe_files", lambda plan: list(range(10)))
    assert all(not r["comparison"]["passed"] for r in receipt["cpu_gpu_diagnostics"])
    assert U.validation_receipt(root, spec, plan) == receipt


@pytest.mark.parametrize("mutation", ["old_contract", "missing_gpu", "duplicate_cpu", "tree", "gpu_shape", "repeat_failed", "cpu_nonfinite"])
def test_receipt_rejects_missing_or_inconsistent_required_checks(campaign, monkeypatch, mutation):
    root, spec, plan, receipt, _, _ = campaign
    monkeypatch.setattr(U.D, "choose_probe_files", lambda plan: list(range(10)))
    bad = deepcopy(receipt)
    if mutation == "old_contract": bad["contract"] = U.D.VERSION + "_validation"
    elif mutation == "missing_gpu": bad["gpu_checks"].pop()
    elif mutation == "duplicate_cpu": bad["cpu_gpu_diagnostics"][0] = bad["cpu_gpu_diagnostics"][1]
    elif mutation == "tree": bad["tree_checks"][0]["maximum_continuous_absolute_error"] = 1
    elif mutation == "gpu_shape": bad["gpu_checks"][0]["repeatability"]["elements"] = 80
    elif mutation == "repeat_failed": bad["gpu_checks"][0]["repeatability"]["passed"] = False
    else: bad["cpu_gpu_diagnostics"][0]["comparison"]["finite"] = False
    overwrite(root / "runtime_validation.json", bad)
    with pytest.raises(ValueError): U.validation_receipt(root, spec, plan)


def test_read_chunk_rejects_legacy_twelve_model_artifact(campaign):
    root, _, plan, _, _, _ = campaign
    path = F.chunk_path(root / "test", 0, 0)
    meta = F.chunk_metadata(plan, 0, 0, 6)
    arrays, record = U.read_chunk(path, meta)
    assert len(arrays) == 24
    np.savez_compressed(path, **{k: v for k, v in arrays.items() if int(k[-2:]) < 12})
    overwrite(path.with_suffix(".json"), dict(record, npz_sha256=F.digest(path)))
    with pytest.raises(ValueError, match="exactly 24"):
        U.read_chunk(path, meta)


def test_global_aggregation_uses_only_sporc_same_seed_peers(campaign):
    root, spec, plan, receipt, expected, hashes = campaign
    row = R.aggregate(root, spec, plan, receipt, 12)
    labels = np.repeat(np.arange(10), 10)
    candidate = np.concatenate([r["logits_12"] for r in expected])
    base = np.concatenate([r["logits_00"] for r in expected])
    assert row["metrics"] == F.calculate_metrics(candidate, labels)
    assert row["paired_accuracy"]["OFF_RPT_BASE"] == F.paired_accuracy(candidate.argmax(1), base.argmax(1), labels)
    assert len(row["paired_accuracy"]) == 7 and row["file_completion_hashes"] == hashes
    assert row["evaluation_runtime"] == "SPORC" and row["predictions_reused"] is False


def fill_metrics(campaign):
    root, spec, plan, receipt, _, hashes = campaign
    labels = np.repeat(np.arange(10), 10)
    metrics = F.calculate_metrics(np.zeros((100, 10), np.float32), labels)
    for i, task in enumerate(plan["tasks"]):
        F.write_json(root / f"test/metrics/model_{i:02d}.json", F.hashed(dict(
            contract=U.VERSION + "_result", campaign_sha256=spec["content_hash"], plan_sha256=plan["content_hash"],
            runtime_validation_sha256=receipt["content_hash"], task=task, metrics=metrics,
            file_completion_hashes=hashes, evaluation_runtime="SPORC", predictions_reused=False,
            used_for_model_selection=False, paired_accuracy={n: {} for n in spec["models"] if n != task["run_id"]})))


def test_report_uses_new_baseline_and_separates_historical_comparison(campaign):
    root, spec, plan, receipt, _, _ = campaign
    fill_metrics(campaign)
    result = R.report(root, spec, plan, receipt)
    assert len(result["models"]) == 8 and len(result["per_model_results"]) == 24
    assert result["models"]["OFF_RPT_BASE"]["mean_accuracy"] == pytest.approx(.1)
    assert all(r["difference_vs_baseline"] == 0 and r["evaluation_runtime"] == "SPORC" and not r["predictions_reused"]
               for r in result["models"].values())
    assert result["historical_comparison"]["models"]["OFF_RPT_BASE"]["mean_accuracy_difference"] == pytest.approx(-.9)
    assert set(result["contrasts"]) == {"standard", "selected", "selected_minus_standard"}
    markdown = (root / "reports/offline_unified_report.md").read_text()
    assert "30%" in markdown and "50%" in markdown and "99.5%" in markdown
    assert "NOT reused" in markdown


@pytest.mark.parametrize("key,value", [("evaluation_runtime", "historical_Tigris"), ("predictions_reused", True),
    ("runtime_validation_sha256", "bad"), ("plan_sha256", "bad"), ("file_completion_hashes", [])])
def test_report_rejects_mixed_runtime_or_coverage(campaign, key, value):
    root, spec, plan, receipt, _, _ = campaign
    fill_metrics(campaign)
    path = root / "test/metrics/model_00.json"
    overwrite(path, dict(F.read_json(path), **{key: value}))
    with pytest.raises(ValueError, match="mixes"):
        R.report(root, spec, plan, receipt)


def test_storage_counts_unwritten_events_not_compressed_bytes(tmp_path, monkeypatch):
    output, source = tmp_path / "new", tmp_path / "pending"
    path = output / "test/predictions/file_000/chunk_000000.json"
    F.write_json(path, F.hashed(dict(metadata=dict(entry_start=0, entry_stop=10))))
    path.with_suffix(".npz").write_bytes(b"compressed")
    monkeypatch.setattr(U.shutil, "disk_usage", lambda p: SimpleNamespace(free=100 * 1024**3))
    projection = U.storage_projection(output, source, 100)
    assert projection["unified_remaining_raw_logit_bytes"] == 90 * 24 * 10 * 4
    assert projection["tigris_remaining_raw_logit_bytes"] == 100 * 12 * 10 * 4
    monkeypatch.setattr(U.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="Nothing deleted"):
        U.storage_projection(output, source, 100)
    assert path.is_file()


def test_math_policy_disables_both_tf32_paths(monkeypatch):
    import torch
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()
    monkeypatch.setattr(U.D, "configure_math", lambda: None)
    try:
        U.configure_math()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(saved[2])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]


def test_infer_writes_all_24_models_resumes_and_rejects_corrupt_commits(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import uproot
    root, data_dir = tmp_path / "output", tmp_path / "data"
    data_dir.mkdir()
    data = data_dir / "qcd.root"
    data.write_bytes(b"fake ROOT for an isolated driver orchestration test")
    old, follow = plans()
    row = dict(name=data.name, label=0, entries=10, sha256=F.digest(data))
    plan = F.hashed(dict(tasks=U.combined_tasks(old, follow), data_dir=str(data_dir), files=[row],
                        assignments=[[0]], chunk_size=6, event_count=10, contract=U.VERSION + "_test"))
    spec, receipt = {"source_campaign": str(tmp_path / "source")}, {"content_hash": "v" * 64}
    monkeypatch.setattr(U, "load_ready", lambda output: (spec, {}, plan, None))
    monkeypatch.setattr(U, "configure_math", lambda: None)
    monkeypatch.setattr(U, "validation_receipt", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(U, "storage_projection", lambda *args: {})
    monkeypatch.setattr(U.D, "gpu_identity", lambda: {"name": "test", "capability": [8, 0]})
    monkeypatch.setattr(F, "exclusive", lambda path: nullcontext())
    models_loaded, forwards = [], []
    monkeypatch.setattr(U, "load_models", lambda spec, source, tasks, device: models_loaded.append(len(tasks)) or list(range(24)))

    class FakeRoot:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, key):
            assert key == "tree"
            return self
        def arrays(self, branches, entry_start, entry_stop, library):
            return list(range(entry_start, entry_stop))

    monkeypatch.setattr(uproot, "open", lambda path: FakeRoot())
    def forward(arrays, label, models, backend, device):
        forwards.append(list(arrays))
        return {f"logits_{i:02d}": np.full((len(arrays), 10), i, np.float32) for i in range(24)}
    monkeypatch.setattr(F, "infer_arrays", forward)
    args = SimpleNamespace(output=root, task_index=0)
    # An uncommitted orphan is recomputed, without replacing a committed result.
    path = F.chunk_path(root / "test", 0, 0)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"interrupted uncommitted write")
    U.infer(args)
    assert models_loaded == [24] and forwards == [list(range(6)), list(range(6, 10))]
    arrays, record = U.read_chunk(path, F.chunk_metadata(plan, 0, 0, 6))
    assert len(arrays) == 24 and arrays["logits_23"][0, 0] == 23
    assert record["runtime_validation_sha256"] == receipt["content_hash"]
    U.infer(args)
    assert models_loaded == [24] and len(forwards) == 2
    path.write_bytes(b"corrupted committed predictions")
    with pytest.raises(ValueError, match="bytes differ"):
        U.infer(args)
