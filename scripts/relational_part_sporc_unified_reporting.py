"""Global, same-runtime metrics for the 24-checkpoint SPORC evaluation."""
from __future__ import annotations

from pathlib import Path
import statistics

from scripts import run_relational_part_sporc_unified as U
F, R = U.F, U.R


def validate_plan(spec, plan):
    R._authenticated(spec, "Unified campaign")
    R._authenticated(plan, "Unified plan")
    R._coverage(plan)
    if (spec["contract"] != U.VERSION or plan["contract"] != U.VERSION + "_test"
            or plan["campaign_sha256"] != spec["content_hash"] or plan["tasks"] != spec["tasks"]
            or plan["historical_predictions_reused"] is not False or plan["primary_baseline_runtime"] != "SPORC"
            or plan["performance_gate"] is not False or plan["used_for_model_selection"] is not False
            or len(plan["tasks"]) != 24 or set(spec["models"]) != set(F.MODELS + U.FOLLOWUP_MODELS)
            or {(t["run_id"], t["seed"]) for t in plan["tasks"]} != {(n, s) for n in spec["models"] for s in F.SEEDS}):
        raise ValueError("Unified primary population must be all eight cells on SPORC")


def completions(output, plan, receipt):
    hashes = []
    for i, row in enumerate(plan["files"]):
        records = []
        for start, stop in F.chunks(row, plan):
            _, record = U.read_chunk(F.chunk_path(output / "test", i, start),
                F.chunk_metadata(plan, i, start, stop), keys=set())
            if record["runtime_validation_sha256"] != receipt["content_hash"]:
                raise ValueError("Prediction runtime validation differs")
            records.append(record["content_hash"])
        hashes.append(R._completed(output / "test", plan, i, records))
    return hashes


def aggregate(output, spec, plan, receipt, task_index):
    import numpy as np
    validate_plan(spec, plan)
    if not 0 <= task_index < 24:
        raise IndexError("Unified metric task index must be 0..23")
    task = plan["tasks"][task_index]
    indices = [i for i, t in enumerate(plan["tasks"]) if t["seed"] == task["seed"]]
    references = {plan["tasks"][i]["run_id"]: np.empty(plan["event_count"], np.int8) for i in indices}
    values = np.empty((plan["event_count"], 10), np.float32)
    labels = np.empty(plan["event_count"], np.int8)
    offset, hashes = 0, []
    for index, row in enumerate(plan["files"]):
        records = []
        for start, stop in F.chunks(row, plan):
            arrays, record = U.read_chunk(F.chunk_path(output / "test", index, start),
                F.chunk_metadata(plan, index, start, stop), keys={f"logits_{i:02d}" for i in indices})
            if record["runtime_validation_sha256"] != receipt["content_hash"]:
                raise ValueError("Prediction runtime validation differs")
            records.append(record["content_hash"])
            end = offset + stop - start
            values[offset:end] = arrays[f"logits_{task_index:02d}"]
            labels[offset:end] = row["label"]
            for i in indices:
                references[plan["tasks"][i]["run_id"]][offset:end] = arrays[f"logits_{i:02d}"].argmax(1)
            offset = end
        hashes.append(R._completed(output / "test", plan, index, records))
    if offset != plan["event_count"] or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, offset // 10)):
        raise ValueError("Incomplete or unbalanced global test population")
    metrics = F.calculate_metrics(values, labels)
    R._metric(metrics, plan)
    prediction = values.argmax(1)
    paired = {name: F.paired_accuracy(prediction, reference, labels)
              for name, reference in references.items() if name != task["run_id"]}
    result = F.hashed({"contract": U.VERSION + "_result", "campaign_sha256": spec["content_hash"],
        "plan_sha256": plan["content_hash"], "runtime_validation_sha256": receipt["content_hash"],
        "task": task, "metrics": metrics, "paired_accuracy": paired, "file_completion_hashes": hashes,
        "evaluation_runtime": "SPORC", "predictions_reused": False, "used_for_model_selection": False})
    F.write_json(output / "test/metrics" / f"model_{task_index:02d}.json", result)
    F.progress("unified_metric_complete", run_id=task["run_id"], seed=task["seed"], accuracy=metrics["accuracy"])
    return result


def summarize(results, names, runtime, plan_hash):
    models = {}
    for name in names:
        matching = [r for r in results if r["task"]["run_id"] == name]
        by_seed = {r["task"]["seed"]: r["metrics"] for r in matching}
        if len(matching) != 3 or set(by_seed) != set(F.SEEDS):
            raise ValueError("Summary requires exactly three seeds per configuration")
        accuracy = {str(s): by_seed[s]["accuracy"] for s in F.SEEDS}
        auroc = {str(s): statistics.mean(by_seed[s]["one_vs_rest_auc"].values()) for s in F.SEEDS}
        row = {"mean_accuracy": statistics.mean(accuracy.values()), "per_seed_accuracy": accuracy,
            "seed_sample_standard_deviation": statistics.stdev(accuracy.values()),
            "mean_macro_auroc": statistics.mean(auroc.values()), "per_seed_macro_auroc": auroc,
            "macro_auroc_seed_sample_standard_deviation": statistics.stdev(auroc.values()),
            "evaluation_runtime": runtime, "prediction_plan_sha256": plan_hash,
            "predictions_reused": runtime != "SPORC", "qcd_signal_rejection": {}}
        for signal in F.CLASS_NAMES[1:]:
            row["qcd_signal_rejection"][signal] = {}
            for target in F.TARGETS:
                values = {str(s): by_seed[s]["qcd_signal_rejection"][signal][str(target)]["background_rejection"] for s in F.SEEDS}
                finite = all(v is not None for v in values.values())
                row["qcd_signal_rejection"][signal][str(target)] = {
                    "per_seed": values, "mean": statistics.mean(values.values()) if finite else None,
                    "seed_sample_standard_deviation": statistics.stdev(values.values()) if finite else None,
                    "zero_background_seeds": sum(v is None for v in values.values())}
        models[name] = row
    for row in models.values():
        row["difference_vs_baseline"] = row["mean_accuracy"] - models["OFF_RPT_BASE"]["mean_accuracy"]
        row["macro_auroc_difference_vs_baseline"] = row["mean_macro_auroc"] - models["OFF_RPT_BASE"]["mean_macro_auroc"]
    return models


def platform_comparison(models, old_models):
    rows = {}
    for name in F.MODELS:
        new, old = models[name], old_models[name]
        row = {"per_seed_accuracy_difference": {s: new["per_seed_accuracy"][s] - old["per_seed_accuracy"][s] for s in map(str, F.SEEDS)},
            "mean_accuracy_difference": new["mean_accuracy"] - old["mean_accuracy"],
            "mean_macro_auroc_difference": new["mean_macro_auroc"] - old["mean_macro_auroc"],
            "per_seed_macro_auroc_difference": {s: new["per_seed_macro_auroc"][s] - old["per_seed_macro_auroc"][s] for s in map(str, F.SEEDS)},
            "qcd_signal_rejection": {}}
        for signal in F.CLASS_NAMES[1:]:
            row["qcd_signal_rejection"][signal] = {}
            for target in F.TARGETS:
                a = new["qcd_signal_rejection"][signal][str(target)]
                b = old["qcd_signal_rejection"][signal][str(target)]
                row["qcd_signal_rejection"][signal][str(target)] = {
                    "sporc_per_seed": a["per_seed"], "tigris_per_seed": b["per_seed"],
                    "mean_difference": None if a["mean"] is None or b["mean"] is None else a["mean"] - b["mean"]}
        rows[name] = row
    return {"definition": "SPORC minus historical Tigris; same checkpoint, not an architectural gain",
            "used_for_model_or_platform_selection": False, "models": rows}


def report(output, spec, plan, receipt):
    validate_plan(spec, plan)
    hashes = completions(output, plan, receipt)
    results = [F.read_json(output / "test/metrics" / f"model_{i:02d}.json") for i in range(24)]
    for task, row in zip(plan["tasks"], results):
        R._metric(row["metrics"], plan)
        if (row["contract"] != U.VERSION + "_result" or row["task"] != task
                or row["campaign_sha256"] != spec["content_hash"] or row["plan_sha256"] != plan["content_hash"]
                or row["runtime_validation_sha256"] != receipt["content_hash"]
                or row["file_completion_hashes"] != hashes or row["evaluation_runtime"] != "SPORC"
                or row["predictions_reused"] is not False or row["used_for_model_selection"] is not False
                or set(row["paired_accuracy"]) != set(spec["models"]) - {task["run_id"]}):
            raise ValueError("Report mixes runtimes, checkpoints, or populations")
    old = F.read_json(output / "reference_plan.json")
    old_report = F.read_json(output / "reference_report.json")
    U.validate_reference(old, old_report)
    if old["content_hash"] != spec["reference_plan_sha256"] or old_report["content_hash"] != spec["reference_report_sha256"]:
        raise ValueError("Historical comparison reference changed")
    models = summarize(results, spec["models"], "SPORC", plan["content_hash"])
    old_models = summarize(old_report["per_model_results"], F.MODELS, "historical_Tigris", old["content_hash"])
    result = F.hashed({"contract": U.VERSION + "_report", "schema_version": 1,
        "campaign_sha256": spec["content_hash"], "plan_sha256": plan["content_hash"],
        "runtime_validation": receipt, "runtime_validation_sha256": receipt["content_hash"],
        "event_count_per_model": plan["event_count"], "class_order": plan["class_order"], "seeds": list(F.SEEDS),
        "metric_policy": plan["metric_policy"], "math_options": U.MATH_OPTIONS,
        "primary_comparison_runtime": "SPORC", "historical_predictions_reused": False,
        "models": models, "contrasts": R._contrasts(models), "per_model_results": results,
        "historical_comparison": platform_comparison(models, old_models),
        "historical_comparison_reference_plan_sha256": old["content_hash"],
        "historical_comparison_reference_report_sha256": old_report["content_hash"],
        "inference_only": True, "existing_checkpoints_retrained": False, "performance_gate": False,
        "used_for_model_selection": False, "platform_selection_by_scores": False,
        "cross_platform_equivalence_claimed": False})
    F.write_json(output / "reports/offline_unified_report.json", result)
    markdown(output, result)
    F.progress("unified_report_complete", report=str(output / "reports/offline_unified_report.md"))
    return result


def markdown(output, result):
    lines = ["# Offline RPT: unified SPORC evaluation", "",
        f"Eight configurations, three seeds each; {result['event_count_per_model']:,} official test jets per checkpoint. No retraining.", "",
        "All primary scores and baseline contrasts below use fresh SPORC predictions. Historical Tigris scores are NOT reused in the primary table. Cross-platform numerical equivalence is not claimed.", "",
        "FP32 inference with matmul and cuDNN TF32 disabled globally. CPU/GPU numerical differences are retained in the runtime-validation receipt, not used as accuracy/performance gates.", "",
        "| Model | Accuracy | Delta SPORC base (pp) | Seed SD (pp) | Macro AUROC | Delta AUROC |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, row in result["models"].items():
        lines.append(f"| {name} | {100*row['mean_accuracy']:.4f}% | {100*row['difference_vs_baseline']:+.4f} | {100*row['seed_sample_standard_deviation']:.4f} | {row['mean_macro_auroc']:.6f} | {row['macro_auroc_difference_vs_baseline']:+.6f} |")
    lines += ["", "Macro AUROC is the mean of ten one-vs-rest AUROCs within each seed, then averaged over seeds. Seed SD is not a confidence interval.", "",
              "## Architectural contrasts (SPORC only)", "", "| Feature group | Contrast | Accuracy delta (pp) | AUROC delta |", "|---|---|---:|---:|"]
    for group, contrasts in result["contrasts"].items():
        for label, row in contrasts.items():
            lines.append(f"| {group} | {label} | {100*row['mean_accuracy_difference']:+.4f} | {row['mean_macro_auroc_difference']:+.6f} |")
    for target in F.TARGETS:
        lines += ["", f"## QCD rejection at {100*target:g}% signal efficiency", "",
                  "| Model | " + " | ".join(F.CLASS_NAMES[1:]) + " |", "|---|" + "---:|" * 9]
        for name, row in result["models"].items():
            cells = [row["qcd_signal_rejection"][s][str(target)]["mean"] for s in F.CLASS_NAMES[1:]]
            lines.append("| " + name + " | " + " | ".join("SAT" if v is None else f"{v:.2f}" for v in cells) + " |")
    lines += ["", "SAT means at least one zero-background seed; no finite mean is reported. JSON retains per-seed counts, thresholds, achieved efficiencies, conditional Wilson intervals and seven same-seed paired accuracy comparisons. These are independent seed evaluations, not an ensemble.", "",
              "## Separate cross-platform comparison", "",
              "Same checkpoint, SPORC minus historical Tigris. These differences are not architectural gains or a rule for choosing the preferred platform.", "",
              "| Original configuration | Accuracy difference (pp) | Macro AUROC difference |", "|---|---:|---:|"]
    for name, row in result["historical_comparison"]["models"].items():
        lines.append(f"| {name} | {100*row['mean_accuracy_difference']:+.4f} | {row['mean_macro_auroc_difference']:+.6f} |")
    lines += ["", "Historical comparisons currently cover the original four configurations. The pending Tigris follow-up evaluation is not needed to complete this SPORC report. All configurations are reported regardless of gains.", ""]
    text = "\n".join(lines)
    path = output / "reports/offline_unified_report.md"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Existing unified report differs")
        return
    with F.atomic_file(path, binary=False) as stream:
        stream.write(text)
