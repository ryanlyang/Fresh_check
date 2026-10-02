"""Small-population checks of the inference-only report's scientific lineage."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import relational_part_sporc_reporting as R


NEW_MODELS = {
    "OFF_RPT_BASE_LAYERWISE": {"features": "standard", "bias": "layerwise", "edge_value": False},
    "OFF_RPT_BASE_SHARED_EDGEVALUE": {"features": "standard", "bias": "shared", "edge_value": True},
    "OFF_RPT_SELECTED_SHARED_EDGEVALUE": {"features": "selected", "bias": "shared", "edge_value": True},
    "OFF_RPT_SELECTED_SHARED": {"features": "selected", "bias": "shared", "edge_value": False},
}


def _seal(value):
    return R.F.hashed({k: v for k, v in value.items() if k != "content_hash"})


def _replace(path, value):
    path.write_text(json.dumps(_seal(value)), encoding="utf-8")


def _metric(accuracy, auroc, saturated=False):
    return R.F.hashed({"contract": R.F.VERSION + "_metrics", "event_count": 100,
        "class_order": list(R.F.CLASS_NAMES), "accuracy": accuracy,
        "one_vs_rest_auc": {name: auroc + i * .001 for i, name in enumerate(R.F.CLASS_NAMES)},
        "qcd_signal_rejection": {signal: {str(target): {"background_rejection": None if saturated else 100.}
            for target in R.F.TARGETS} for signal in R.F.CLASS_NAMES[1:]}})


def _predictions(root, plan, rng):
    completions, wanted = [], []
    for index, row in enumerate(plan["files"]):
        records = []
        for start, stop in R.F.chunks(row, plan):
            path = R.F.chunk_path(root, index, start)
            path.parent.mkdir(parents=True, exist_ok=True)
            values = {f"logits_{i:02d}": rng.normal(size=(stop - start, 10)).astype(np.float32) for i in range(12)}
            np.savez_compressed(path, **values)
            wanted.append(values["logits_00"])
            receipt = R.F.hashed({"metadata": R.F.chunk_metadata(plan, index, start, stop),
                                  "npz_sha256": R.F.digest(path)})
            R.F.write_json(path.with_suffix(".json"), receipt)
            records.append(receipt["content_hash"])
        done = R.F.hashed({"contract": plan["contract"] + "_file_complete",
            "plan_sha256": plan["content_hash"], "file_index": index, "chunks": records})
        R.F.write_json(root / f"predictions/file_{index:03d}/complete.json", done)
        completions.append(done["content_hash"])
    return completions, np.concatenate(wanted)


@pytest.fixture
def campaign(tmp_path):
    previous = tmp_path / "historical"
    output = tmp_path / "sporc"
    old_tasks = [{"run_id": name, "seed": seed} for name in R.F.MODELS for seed in R.F.SEEDS]
    tasks = [{"run_id": name, "seed": seed} for name in NEW_MODELS for seed in R.F.SEEDS]
    rows = [{"label": label, "name": f"class_{label}.root", "entries": 10, "sha256": str(label) * 64}
            for label in range(10)]
    protocol = {"files": rows, "chunk_size": 6, "batch_size": 64, "event_count": 100,
        "class_order": list(R.F.CLASS_NAMES), "precision": "original FP32",
        "metric_policy": {"ROC": "global"}, "rejection_targets": list(R.F.TARGETS)}
    old_plan = R.F.hashed({**protocol, "contract": R.F.VERSION, "tasks": old_tasks})
    R.F.write_json(previous / "evaluation_plan.json", old_plan)
    R.F.write_json(output / "reference_plan.json", old_plan)
    rng = np.random.default_rng(129)
    old_completions, _ = _predictions(previous, old_plan, rng)
    old_results = [R.F.hashed({"contract": R.F.VERSION + "_result", "plan_sha256": old_plan["content_hash"],
        "task": task, "file_completion_hashes": old_completions,
        "metrics": _metric(.8 + (i // 3) * .001, .9 + (i % 3) * .001),
        "paired_accuracy": {}, "used_for_model_selection": False}) for i, task in enumerate(old_tasks)]
    old_report = R.F.hashed({"contract": R.F.VERSION + "_report", "plan_sha256": old_plan["content_hash"],
        "event_count_per_model": 100, "per_model_results": old_results})
    R.F.write_json(previous / "reports/full_test_report.json", old_report)
    R.F.write_json(output / "reference_report.json", old_report)
    spec = R.F.hashed({"contract": R.VERSION, "models": NEW_MODELS, "seeds": list(R.F.SEEDS),
        "source_campaign": str(tmp_path / "trained"), "source_campaign_sha256": "f" * 64,
        "previous_evaluation": str(previous), "original_parent": str(tmp_path / "parent"),
        "reference_plan_sha256": old_plan["content_hash"], "reference_report_sha256": old_report["content_hash"]})
    plan = R.F.hashed({**protocol, "contract": R.VERSION + "_test", "campaign_sha256": spec["content_hash"],
        "reference_plan_sha256": old_plan["content_hash"], "tasks": tasks,
        "performance_gate": False, "used_for_model_selection": False})
    R.F.write_json(output / "portability_validation.json", R.F.hashed({
        "contract": R.VERSION + "_validation", "passed": True,
        "campaign_sha256": spec["content_hash"], "plan_sha256": plan["content_hash"],
        "gpu": {"name": "Synthetic test GPU", "capability": [8, 0], "total_memory": 8 * 1024**3},
        "new_timings": [{"label": label, "events": 10, "seconds": 1.} for label in range(10)],
        "peak_gpu_allocated_bytes": 1024**3, "peak_gpu_reserved_bytes": 2 * 1024**3,
        "estimated_largest_task_seconds_excluding_io": 100.,
        "timing_is_estimate_not_performance_gate": True}))
    new_completions, logits = _predictions(output / "test", plan, rng)
    return output, previous, spec, plan, old_plan, old_report, {"old": old_completions, "new": new_completions}, logits


def _results(campaign):
    output, _, spec, plan, _, _, completions, _ = campaign
    for i, task in enumerate(plan["tasks"]):
        result = R.F.hashed({"contract": R.VERSION + "_result", "campaign_sha256": spec["content_hash"],
            "plan_sha256": plan["content_hash"], "task": task, "file_completion_hashes": completions,
            "metrics": _metric(.81 + (i // 3) * .001, .91 + (i % 3) * .001, saturated=i == 3),
            "paired_accuracy": {name: {} for name in (*R.F.MODELS, *NEW_MODELS) if name != task["run_id"]},
            "used_for_model_selection": False})
        R.F.write_json(output / "test/metrics" / f"model_{i:02d}.json", result)


def test_aggregation_is_global_paired_authenticated_and_restartable(campaign):
    output, previous, spec, plan, _, _, completions, logits = campaign
    before = {path: R.F.digest(path) for path in previous.rglob("*") if path.is_file()}
    result = R.aggregate(output, spec, plan, 0)
    assert R.aggregate(output, spec, plan, 0) == result
    assert result["metrics"] == R.F.calculate_metrics(logits, np.repeat(np.arange(10), 10))
    assert result["campaign_sha256"] == spec["content_hash"]
    assert result["file_completion_hashes"] == completions
    assert len(result["paired_accuracy"]) == 7
    assert all(row["sampling_unit"] == "paired_event_identity" for row in result["paired_accuracy"].values())
    assert before == {path: R.F.digest(path) for path in before}
    with R.F.chunk_path(previous, 0, 0).open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="bytes"):
        R.aggregate(output, spec, plan, 0)


def test_complete_report_summarizes_macro_auc_and_separates_runtimes(campaign):
    output, previous, spec, plan, _, old, _, _ = campaign
    _results(campaign)
    before = {path: R.F.digest(path) for path in previous.rglob("*") if path.is_file()}
    result = R.report(output, spec, plan)
    assert R.report(output, spec, plan) == result
    assert len(result["models"]) == 8 and len(result["per_model_results"]) == 24
    assert result["per_model_results"][:12] == old["per_model_results"]
    assert result["new_training_runs"] == 0 and result["inference_only"] is True
    assert result["performance_gate"] is False and result["runtime_migration"] == spec
    validation = R.F.read_json(output / "portability_validation.json")
    assert result["portability_validation_sha256"] == validation["content_hash"]
    assert result["portability_validation"] == validation
    base = result["models"]["OFF_RPT_BASE"]
    assert base["mean_macro_auroc"] == pytest.approx(.9055)
    assert base["macro_auroc_seed_sample_standard_deviation"] == pytest.approx(.001)
    assert base["evaluation_runtime"] == "historical_Tigris"
    shared = result["models"]["OFF_RPT_BASE_SHARED_EDGEVALUE"]
    assert shared["evaluation_runtime"] == "SPORC"
    assert shared["qcd_signal_rejection"]["Hbb"]["0.5"]["mean"] is None
    assert result["contrasts"]["standard"]["EV_at_shared"]["mean_accuracy_difference"] == pytest.approx(.011)
    assert len(result["contrasts"]["selected_minus_standard"]) == 4
    markdown = (output / "reports/offline_factorial_report.md").read_text(encoding="utf-8")
    assert all(text in markdown for text in ("Macro AUROC", "SAT", "AFTER inspection", "Tigris", "SPORC"))
    assert before == {path: R.F.digest(path) for path in before}


@pytest.mark.parametrize("mutation", ("bytes", "contract", "campaign", "plan", "failed"))
def test_report_requires_current_authenticated_passed_portability_validation(campaign, mutation):
    output, _, spec, plan, _, _, _, _ = campaign
    _results(campaign)
    path = output / "portability_validation.json"
    receipt = R.F.read_json(path)
    if mutation == "bytes":
        receipt["peak_gpu_allocated_bytes"] += 1
        path.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        if mutation == "contract":
            receipt["contract"] = "another_validation"
        elif mutation == "campaign":
            receipt["campaign_sha256"] = "f" * 64
        elif mutation == "plan":
            receipt["plan_sha256"] = "f" * 64
        else:
            receipt["passed"] = False
        _replace(path, receipt)
    with pytest.raises(ValueError, match="authentication|attestation"):
        R.report(output, spec, plan)


@pytest.mark.parametrize("population", ("old", "new"))
@pytest.mark.parametrize("mutation", ("bytes", "completion", "metadata"))
def test_report_revalidates_both_populations(campaign, population, mutation):
    output, previous, spec, plan, _, _, _, _ = campaign
    _results(campaign)
    root = previous if population == "old" else output / "test"
    chunk = R.F.chunk_path(root, 0, 0)
    if mutation == "bytes":
        with chunk.open("ab") as stream:
            stream.write(b"drift")
    elif mutation == "completion":
        path = root / "predictions/file_000/complete.json"
        value = R.F.read_json(path)
        value["chunks"] = []
        _replace(path, value)
    else:
        path = chunk.with_suffix(".json")
        value = R.F.read_json(path)
        value["metadata"]["entry_stop"] += 1
        _replace(path, value)
    with pytest.raises(ValueError, match="bytes|coverage|lineage"):
        R.report(output, spec, plan)


@pytest.mark.parametrize("mutation", ("nested_metrics", "task", "paired", "coverage", "campaign"))
def test_report_rejects_resealed_result_with_invalid_nested_lineage(campaign, mutation):
    output, _, spec, plan, _, _, _, _ = campaign
    _results(campaign)
    path = output / "test/metrics/model_00.json"
    result = R.F.read_json(path)
    if mutation == "nested_metrics":
        result["metrics"]["accuracy"] += .01
    elif mutation == "task":
        result["task"] = deepcopy(plan["tasks"][1])
    elif mutation == "paired":
        result["paired_accuracy"].pop("OFF_RPT_BASE")
    elif mutation == "coverage":
        result["file_completion_hashes"]["new"] = []
    else:
        result["campaign_sha256"] = "f" * 64
    _replace(path, result)
    with pytest.raises(ValueError, match="authentication|mixes"):
        R.report(output, spec, plan)


@pytest.mark.parametrize("mutation", ("nested_result", "nested_metrics", "old_task", "old_plan", "coverage"))
def test_reference_authentication_checks_nested_published_records(campaign, mutation):
    output, previous, spec, plan, _, original, _, _ = campaign
    old = deepcopy(original)
    row = old["per_model_results"][0]
    if mutation == "nested_result":
        row["plan_sha256"] = "e" * 64
    elif mutation == "nested_metrics":
        row["metrics"]["accuracy"] += .01
        old["per_model_results"][0] = _seal(row)
    elif mutation == "old_task":
        row["task"]["seed"] = 0
        old["per_model_results"][0] = _seal(row)
    elif mutation == "old_plan":
        row["plan_sha256"] = "e" * 64
        old["per_model_results"][0] = _seal(row)
    else:
        row["file_completion_hashes"] = []
        old["per_model_results"][0] = _seal(row)
    old = _seal(old)
    _replace(output / "reference_report.json", old)
    _replace(previous / "reports/full_test_report.json", old)
    spec = _seal({**spec, "reference_report_sha256": old["content_hash"]})
    plan = _seal({**plan, "campaign_sha256": spec["content_hash"]})
    with pytest.raises(ValueError, match="authentication|task/plan/coverage"):
        R.authenticated_references(output, spec, plan)
