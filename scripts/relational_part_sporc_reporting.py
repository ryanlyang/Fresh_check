"""Authenticated, bounded-memory reporting for the SPORC inference-only port.

The driver validates the source campaign, runtime, checkpoint lock and validation
receipt before calling these functions. Historical predictions remain read-only;
only the new campaign's ``test/metrics`` and ``reports`` paths are written here.
"""
from __future__ import annotations

from pathlib import Path
import statistics

from scripts import evaluate_relational_part_offline_full_test as F


VERSION = "relational_part_sporc_inference_v1"


def _authenticated(value, description):
    if not isinstance(value, dict) or value != F.hashed({k: v for k, v in value.items() if k != "content_hash"}):
        raise ValueError(f"{description} authentication differs")
    return value


def _matrix(tasks, names, description):
    wanted = {(name, seed) for name in names for seed in F.SEEDS}
    if len(tasks) != len(wanted) or {(t["run_id"], t["seed"]) for t in tasks} != wanted:
        raise ValueError(f"{description} must contain exactly four models by three seeds")


def _coverage(plan):
    counts = [0] * len(F.CLASS_NAMES)
    for row in plan["files"]:
        label, entries = row["label"], row["entries"]
        if type(label) is not int or not 0 <= label < 10 or type(entries) is not int or entries <= 0:
            raise ValueError("Prediction plan has invalid class or entry coverage")
        counts[label] += entries
    if (sum(counts) != plan["event_count"] or not counts[0]
            or any(count != counts[0] for count in counts)):
        raise ValueError("Official test coverage is incomplete or unbalanced")


def _metric(metric, plan):
    _authenticated(metric, "Nested metrics")
    if (metric["contract"] != F.VERSION + "_metrics"
            or metric["event_count"] != plan["event_count"]
            or metric["class_order"] != plan["class_order"]
            or set(metric["one_vs_rest_auc"]) != set(F.CLASS_NAMES)):
        raise ValueError("Nested metrics population or class order differs")


def authenticated_references(output: Path, spec: dict, plan: dict):
    """Bind copied references and every nested old result to the published plan."""
    _authenticated(spec, "SPORC campaign")
    _authenticated(plan, "SPORC test plan")
    old_plan = F.read_json(output / "reference_plan.json")
    old_report = F.read_json(output / "reference_report.json")
    previous = Path(spec["previous_evaluation"])
    if (old_plan["content_hash"] != spec["reference_plan_sha256"]
            or old_report["content_hash"] != spec["reference_report_sha256"]
            or F.read_json(previous / "evaluation_plan.json") != old_plan
            or F.read_json(previous / "reports/full_test_report.json") != old_report
            or old_plan["contract"] != F.VERSION
            or old_report["contract"] != F.VERSION + "_report"
            or old_report["plan_sha256"] != old_plan["content_hash"]
            or old_report["event_count_per_model"] != old_plan["event_count"]
            or plan["contract"] != VERSION + "_test"
            or plan["campaign_sha256"] != spec["content_hash"]
            or plan["reference_plan_sha256"] != old_plan["content_hash"]
            or plan["performance_gate"] is not False
            or plan["used_for_model_selection"] is not False):
        raise ValueError("Reference report/plan or SPORC campaign binding differs")
    for key in ("files", "event_count", "chunk_size", "batch_size", "class_order",
                "precision", "metric_policy", "rejection_targets"):
        if plan[key] != old_plan[key]:
            raise ValueError(f"Historical and SPORC test protocols differ: {key}")
    if (len(spec["models"]) != 4 or set(spec["models"]) & set(F.MODELS)
            or spec["seeds"] != list(F.SEEDS)
            or plan["class_order"] != list(F.CLASS_NAMES)):
        raise ValueError("Eight-cell model/seed/class matrix differs")
    _matrix(old_plan["tasks"], F.MODELS, "Historical matrix")
    _matrix(plan["tasks"], spec["models"], "SPORC matrix")
    _coverage(plan)
    rows = old_report["per_model_results"]
    if len(rows) != len(old_plan["tasks"]):
        raise ValueError("Historical report lacks all twelve results")
    for task, row in zip(old_plan["tasks"], rows):
        _authenticated(row, "Historical nested result")
        _metric(row["metrics"], old_plan)
        if (row["contract"] != F.VERSION + "_result"
                or row["task"] != task or row["plan_sha256"] != old_plan["content_hash"]
                or row["used_for_model_selection"] is not False
                or len(row["file_completion_hashes"]) != len(old_plan["files"])
                or row["file_completion_hashes"] != rows[0]["file_completion_hashes"]):
            raise ValueError("Historical nested result task/plan/coverage differs")
    return old_plan, old_report


def _completed(root, plan, file_index, records):
    done = F.read_json(root / "predictions" / f"file_{file_index:03d}" / "complete.json")
    if done != F.hashed({"contract": plan["contract"] + "_file_complete",
                         "plan_sha256": plan["content_hash"], "file_index": file_index,
                         "chunks": records}):
        raise ValueError("Prediction file completion/coverage differs")
    return done["content_hash"]


def prediction_completions(root: Path, plan: dict):
    """Rehash every NPZ and authenticate exact coverage without loading logits.

    ``read_chunk(keys=set())`` still verifies all archive member names, the NPZ
    bytes and authenticated sidecar metadata; it does not retain model arrays.
    """
    _coverage(plan)
    completions = []
    for index, row in enumerate(plan["files"]):
        records = []
        for start, stop in F.chunks(row, plan):
            _, receipt = F.read_chunk(F.chunk_path(root, index, start),
                                      F.chunk_metadata(plan, index, start, stop), keys=set())
            records.append(receipt["content_hash"])
        completions.append(_completed(root, plan, index, records))
    return completions


def aggregate(output: Path, spec: dict, plan: dict, task_index: int):
    """Score one SPORC checkpoint against all seven same-seed matrix peers."""
    import numpy as np

    output = Path(output)
    previous_plan, previous_report = authenticated_references(output, spec, plan)
    if not 0 <= task_index < len(plan["tasks"]):
        raise IndexError("Metric task index outside twelve-model SPORC matrix")
    task = plan["tasks"][task_index]
    populations = (("old", previous_plan, Path(spec["previous_evaluation"])),
                   ("new", plan, output / "test"))
    indices = {tag: [i for i, t in enumerate(current["tasks"]) if t["seed"] == task["seed"]]
               for tag, current, _ in populations}
    references = {t["run_id"]: np.empty(plan["event_count"], dtype=np.int8)
                  for t in [*previous_plan["tasks"], *plan["tasks"]] if t["seed"] == task["seed"]}
    # Keep only one checkpoint's global logits, plus eight compact argmax arrays.
    values = np.empty((plan["event_count"], 10), dtype=np.float32)
    labels = np.empty(plan["event_count"], dtype=np.int8)
    offset = 0
    completions = {"old": [], "new": []}
    for index, row in enumerate(plan["files"]):
        records = {"old": [], "new": []}
        for start, stop in F.chunks(row, plan):
            end = offset + stop - start
            for tag, current, root in populations:
                arrays, receipt = F.read_chunk(F.chunk_path(root, index, start),
                    F.chunk_metadata(current, index, start, stop),
                    keys={f"logits_{i:02d}" for i in indices[tag]})
                records[tag].append(receipt["content_hash"])
                for i in indices[tag]:
                    references[current["tasks"][i]["run_id"]][offset:end] = arrays[f"logits_{i:02d}"].argmax(axis=1)
                if tag == "new":
                    values[offset:end] = arrays[f"logits_{task_index:02d}"]
            labels[offset:end] = row["label"]
            offset = end
        for tag, current, root in populations:
            completions[tag].append(_completed(root, current, index, records[tag]))
    if (offset != plan["event_count"] or not np.array_equal(
            np.bincount(labels, minlength=10), np.full(10, plan["event_count"] // 10))):
        raise ValueError("Official test coverage is incomplete or unbalanced")
    if any(row["file_completion_hashes"] != completions["old"] for row in previous_report["per_model_results"]):
        raise ValueError("Historical predictions differ from the published metrics")
    F.progress("calculate_global_metrics", run_id=task["run_id"], seed=task["seed"], event_count=offset)
    metrics = F.calculate_metrics(values, labels)
    _metric(metrics, plan)
    prediction = values.argmax(axis=1)
    paired = {name: F.paired_accuracy(prediction, reference, labels)
              for name, reference in references.items() if name != task["run_id"]}
    result = F.hashed({"contract": VERSION + "_result", "campaign_sha256": spec["content_hash"],
        "plan_sha256": plan["content_hash"], "task": task, "file_completion_hashes": completions,
        "metrics": metrics, "paired_accuracy": paired, "used_for_model_selection": False})
    F.write_json(output / "test/metrics" / f"model_{task_index:02d}.json", result)
    F.progress("metric_complete", run_id=task["run_id"], seed=task["seed"], accuracy=metrics["accuracy"])
    return result


def _summary(results, names, spec, plan, old_plan):
    models = {}
    for name in names:
        matching = {r["task"]["seed"]: r for r in results if r["task"]["run_id"] == name}
        if set(matching) != set(F.SEEDS):
            raise ValueError("Report is missing a seed")
        accuracy = {str(seed): matching[seed]["metrics"]["accuracy"] for seed in F.SEEDS}
        auroc = {str(seed): statistics.mean(matching[seed]["metrics"]["one_vs_rest_auc"].values())
                 for seed in F.SEEDS}
        historical = name in F.MODELS
        row = {"mean_accuracy": statistics.mean(accuracy.values()),
               "seed_sample_standard_deviation": statistics.stdev(accuracy.values()),
               "per_seed_accuracy": accuracy, "mean_macro_auroc": statistics.mean(auroc.values()),
               "macro_auroc_seed_sample_standard_deviation": statistics.stdev(auroc.values()),
               "per_seed_macro_auroc": auroc,
               "evaluation_runtime": "historical_Tigris" if historical else "SPORC",
               "prediction_plan_sha256": old_plan["content_hash"] if historical else plan["content_hash"],
               "predictions_reused": historical, "qcd_signal_rejection": {}}
        for signal in F.CLASS_NAMES[1:]:
            row["qcd_signal_rejection"][signal] = {}
            for target in F.TARGETS:
                values = [matching[s]["metrics"]["qcd_signal_rejection"][signal][str(target)]["background_rejection"]
                          for s in F.SEEDS]
                finite = all(value is not None for value in values)
                row["qcd_signal_rejection"][signal][str(target)] = {
                    "per_seed": dict(zip(map(str, F.SEEDS), values)),
                    "mean": statistics.mean(values) if finite else None,
                    "seed_sample_standard_deviation": statistics.stdev(values) if finite else None}
        models[name] = row
    for row in models.values():
        row["difference_vs_baseline"] = row["mean_accuracy"] - models["OFF_RPT_BASE"]["mean_accuracy"]
        row["macro_auroc_difference_vs_baseline"] = row["mean_macro_auroc"] - models["OFF_RPT_BASE"]["mean_macro_auroc"]
    return models


def _difference(models, positive, negative):
    row = {"positive": positive, "negative": negative}
    for metric, stem in (("accuracy", ""), ("macro_auroc", "macro_auroc_")):
        values = {str(s): models[positive]["per_seed_" + metric][str(s)]
                  - models[negative]["per_seed_" + metric][str(s)] for s in F.SEEDS}
        row["mean_" + metric + "_difference"] = statistics.mean(values.values())
        row[stem + "per_seed_difference"] = values
        row[stem + "seed_sample_standard_deviation"] = statistics.stdev(values.values())
    return row


def _contrasts(models):
    contrasts = {}
    for features, shared, layer, shared_ev, layer_ev in (
        ("standard", "OFF_RPT_BASE", "OFF_RPT_BASE_LAYERWISE", "OFF_RPT_BASE_SHARED_EDGEVALUE", "OFF_RPT_BASE_EDGEVALUE"),
        ("selected", "OFF_RPT_SELECTED_SHARED", "OFF_RPT_SELECTED_LAYERWISE", "OFF_RPT_SELECTED_SHARED_EDGEVALUE", "OFF_RPT_SELECTED_EDGEVALUE")):
        row = {}
        for label, positive, negative in (("EV_at_shared", shared_ev, shared), ("EV_at_layerwise", layer_ev, layer),
                                          ("layerwise_without_EV", layer, shared), ("layerwise_with_EV", layer_ev, shared_ev)):
            row[label] = _difference(models, positive, negative)
        interaction = {"definition": "(layerwise_EV-layerwise_no_EV)-(shared_EV-shared_no_EV)"}
        for metric, stem in (("accuracy", ""), ("macro_auroc", "macro_auroc_")):
            values = {str(s): row["EV_at_layerwise"][stem + "per_seed_difference"][str(s)]
                      - row["EV_at_shared"][stem + "per_seed_difference"][str(s)] for s in F.SEEDS}
            interaction["mean_" + metric + "_difference"] = statistics.mean(values.values())
            interaction[stem + "per_seed_difference"] = values
            interaction[stem + "seed_sample_standard_deviation"] = statistics.stdev(values.values())
        row["EV_by_layerwise_interaction"] = interaction
        contrasts[features] = row
    contrasts["selected_minus_standard"] = {
        label: _difference(models, positive, negative) for label, positive, negative in (
            ("shared_without_EV", "OFF_RPT_SELECTED_SHARED", "OFF_RPT_BASE"),
            ("shared_with_EV", "OFF_RPT_SELECTED_SHARED_EDGEVALUE", "OFF_RPT_BASE_SHARED_EDGEVALUE"),
            ("layerwise_without_EV", "OFF_RPT_SELECTED_LAYERWISE", "OFF_RPT_BASE_LAYERWISE"),
            ("layerwise_with_EV", "OFF_RPT_SELECTED_EDGEVALUE", "OFF_RPT_BASE_EDGEVALUE"))}
    return contrasts


def report(output: Path, spec: dict, plan: dict):
    """Reauthenticate both prediction populations and emit the eight-cell report."""
    output = Path(output)
    old_plan, old_report = authenticated_references(output, spec, plan)
    validation = F.read_json(output / "portability_validation.json")
    if (validation.get("contract") != VERSION + "_validation" or validation.get("passed") is not True
            or validation.get("campaign_sha256") != spec["content_hash"]
            or validation.get("plan_sha256") != plan["content_hash"]):
        raise ValueError("Missing, stale or incompatible SPORC portability attestation")
    completions = {"old": prediction_completions(Path(spec["previous_evaluation"]), old_plan),
                   "new": prediction_completions(output / "test", plan)}
    if any(row["file_completion_hashes"] != completions["old"] for row in old_report["per_model_results"]):
        raise ValueError("Historical prediction coverage differs from the published results")
    results = [F.read_json(output / "test/metrics" / f"model_{i:02d}.json") for i in range(len(plan["tasks"]))]
    names = (*F.MODELS, *spec["models"])
    for task, result in zip(plan["tasks"], results):
        _metric(result["metrics"], plan)
        if (result["contract"] != VERSION + "_result" or result["campaign_sha256"] != spec["content_hash"]
                or result["plan_sha256"] != plan["content_hash"] or result["task"] != task
                or result["file_completion_hashes"] != completions
                or result["used_for_model_selection"] is not False
                or set(result["paired_accuracy"]) != set(names) - {task["run_id"]}):
            raise ValueError("SPORC report mixes checkpoints, metrics or prediction coverage")
    all_results = old_report["per_model_results"] + results
    models = _summary(all_results, names, spec, plan, old_plan)
    contrasts = _contrasts(models)
    result = F.hashed({"contract": VERSION + "_report", "schema_version": 1,
        "campaign_sha256": spec["content_hash"], "plan_sha256": plan["content_hash"],
        "reference_plan_sha256": old_plan["content_hash"], "reference_report_sha256": old_report["content_hash"],
        "source_campaign_sha256": spec["source_campaign_sha256"], "runtime_migration": spec,
        "portability_validation_sha256": validation["content_hash"], "portability_validation": validation,
        "prediction_populations": {
            "historical_Tigris": {"root": spec["previous_evaluation"], "plan_sha256": old_plan["content_hash"],
                                   "models": list(F.MODELS), "file_completion_hashes": completions["old"]},
            "SPORC": {"root": str(output / "test"), "plan_sha256": plan["content_hash"],
                      "models": list(spec["models"]), "file_completion_hashes": completions["new"]}},
        "event_count_per_model": plan["event_count"], "seeds": list(F.SEEDS),
        "class_order": plan["class_order"], "metric_policy": plan["metric_policy"],
        "macro_auroc_definition": "Per seed: arithmetic mean of all ten one-vs-rest softmax-probability AUROCs; then mean and sample SD across three seeds",
        "models": models, "contrasts": contrasts, "per_model_results": all_results,
        "new_training_runs": 0, "existing_checkpoints_retrained": False, "inference_only": True,
        "performance_gate": False, "used_for_model_selection": False,
        "followup_after_test_inspection": True, "training_jets": 1_000_000,
        "interpretation": "Matched training protocol, not universally parameter/compute matched. Historical four-cell Tigris predictions and new four-cell SPORC predictions retain distinct runtime provenance; migration is not a bitwise-equivalence claim. Shared retains original packing/trimmer; layerwise retains historical confirmation path."})
    F.write_json(output / "reports/offline_factorial_report.json", result)
    _markdown(output, result)
    F.progress("complete", report=str(output / "reports/offline_factorial_report.md"))
    return result


def _markdown(output, result):
    models = result["models"]
    validation = result["portability_validation"]
    lines = ["# Offline RPT: SPORC inference-only eight-cell report", "",
        f"{result['event_count_per_model']:,} official test jets per model; 1M training jets; three seeds. No training or model selection occurs in this campaign.", "",
        "The historical four-cell predictions are reused from authenticated Tigris results; only the four already-trained follow-up cells are inferred on SPORC. These populations have distinct runtime provenance, not a claimed bitwise-equivalent runtime.", "",
        f"Portability validation passed on {validation['gpu']['name']}; authenticated receipt `{validation['content_hash']}`. Peak allocated/reserved GPU memory: {validation['peak_gpu_allocated_bytes'] / 1024**3:.3f}/{validation['peak_gpu_reserved_bytes'] / 1024**3:.3f} GiB. Estimated largest inference task excluding I/O: {validation['estimated_largest_task_seconds_excluding_io'] / 3600:.2f} hours; timing is an estimate, not a performance gate. JSON retains the complete validation receipt and per-class timings.", "",
        "The follow-up architecture choices were fixed AFTER inspection of earlier test results.", "",
        "| Model | Runtime | Accuracy | Delta base (pp) | Accuracy seed SD (pp) | Macro AUROC | AUROC seed SD |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name, row in models.items():
        lines.append(f"| {name} | {row['evaluation_runtime']} | {100*row['mean_accuracy']:.4f}% | {100*row['difference_vs_baseline']:+.4f} | {100*row['seed_sample_standard_deviation']:.4f} | {row['mean_macro_auroc']:.6f} | {row['macro_auroc_seed_sample_standard_deviation']:.6f} |")
    lines.extend(["", "Macro AUROC is the unweighted mean of ten one-vs-rest softmax-probability AUROCs for each seed, then summarized across seeds. Seed SD is a sample standard deviation, not a confidence interval.", "",
        "## Matched architectural contrasts", "",
        "| Features | Contrast | Accuracy delta (pp) | Macro AUROC delta |", "|---|---|---:|---:|"])
    for family, contrasts in result["contrasts"].items():
        for label, row in contrasts.items():
            lines.append(f"| {family} | {label} | {100*row['mean_accuracy_difference']:+.4f} | {row['mean_macro_auroc_difference']:+.6f} |")
    for target in F.TARGETS:
        lines.extend(["", f"## QCD rejection at {100*target:g}% signal efficiency", "",
            "| Model | " + " | ".join(F.CLASS_NAMES[1:]) + " |", "|---|" + "---:|" * 9])
        for name, row in models.items():
            cells = []
            for signal in F.CLASS_NAMES[1:]:
                value = row["qcd_signal_rejection"][signal][str(target)]["mean"]
                base = models["OFF_RPT_BASE"]["qcd_signal_rejection"][signal][str(target)]["mean"]
                cell = "SAT" if value is None else f"{value:.2f}"
                if name != "OFF_RPT_BASE" and value is not None and base is not None:
                    cell += f" ({100*(value/base-1):+.1f}%)"
                cells.append(cell)
            lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines.extend(["", "SAT means at least one zero-background seed, not infinite population rejection.",
        "Means summarize independently evaluated seeds, not ensemble predictions. JSON retains per-seed full metrics, thresholds, efficiencies, passing counts, conditional Wilson intervals and all seven same-seed paired accuracy comparisons for each SPORC checkpoint.",
        "Wilson intervals condition on the empirical signal threshold and exclude its sampling uncertainty. Paired accuracy intervals condition on the checkpoints and quantify event resampling, not training-seed variability.",
        "The family encoders and EV introduce extra parameters/compute; these ablations do not establish parameter-matched causality. Shared-bias models retain original Weaver trimming and pair-normalization packing; layerwise models retain the earlier confirmation implementation.", ""])
    text = "\n".join(lines)
    path = output / "reports/offline_factorial_report.md"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Existing SPORC Markdown report differs")
        return
    with F.atomic_file(path, binary=False) as stream:
        stream.write(text)
