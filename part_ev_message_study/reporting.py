"""One-seed, same-population comparisons; no test-set or seed-significance claims."""
from pathlib import Path
import numpy as np
from . import protocol as P, controls as C


def contrasts(values):
    result = {}
    for task in P.TASKS:
        name = task["run_id"]
        for control in C.CONTROL_IDS:
            result[f"{name}_minus_{control}"] = values[name] - values[control]
    result["combined_minus_state"] = values["EV_STATE_SEPARATE_RANGER"] - values["EV_STATE_RANGER"]
    result["combined_minus_separate"] = values["EV_STATE_SEPARATE_RANGER"] - values["EV_SEPARATE_RANGER"]
    result["state_separate_interaction"] = (values["EV_STATE_SEPARATE_RANGER"] -
        values["EV_STATE_RANGER"] - values["EV_SEPARATE_RANGER"] + values["EV_RANGER"])
    return result


def paired_counts(left, right, files):
    result = dict(new_only_correct=0, control_only_correct=0, both_correct=0, both_wrong=0)
    for i in range(files):
        with np.load(left / f"file_{i:03d}.npz", allow_pickle=False) as a, np.load(
                right / f"file_{i:03d}.npz", allow_pickle=False) as b:
            if not np.array_equal(a["labels"], b["labels"]):
                raise ValueError("Paired event labels differ")
            ac = a["logits"].argmax(1) == a["labels"]
            bc = b["logits"].argmax(1) == b["labels"]
            for key, mask in (("new_only_correct",ac & ~bc), ("control_only_correct",~ac & bc),
                              ("both_correct",ac & bc), ("both_wrong",~ac & ~bc)):
                result[key] += int(mask.sum())
    return result


def report(output, spec, engine, refresh_controls):
    output = Path(output)
    with P.file_lock(output / ".report.lock", blocking=False):
        registry, labels = refresh_controls(output, spec, require_complete=True, recompute=True)
        models = dict(registry["controls"])
        donor = Path(spec["donor"]["path"])
        for task in P.TASKS:
            trained = C.validate_training(output, spec, task, spec["donor"]["runtime"])
            row, truth = C.validate_evaluation(output, spec, task, trained, recompute=engine.classification_metrics)
            models[task["run_id"]], labels[task["run_id"]] = row, truth
        if any(not np.array_equal(truth, labels["PART_RANGER"]) for truth in labels.values()):
            raise ValueError("Evaluation file/entry labels differ between models")
        metrics = ("accuracy", "macro_auc_ovo", "macro_auc_ovr", "loss")
        effects = {metric: contrasts({name: row["metrics"][metric] for name,row in models.items()})
                   for metric in metrics}
        curves = {metric: [contrasts({name: row["validation_checks"][i][metric]
                   for name,row in models.items()}) for i in range(10)] for metric in metrics}
        paired = {}
        files = len(spec["inventory"]["splits"]["val"]["files"])
        for i, task in enumerate(P.TASKS):
            new = P.task_directory(output, i) / "full_validation"
            for control in C.CONTROL_IDS:
                old = donor / "runs" / control / "seed_101/full_validation"
                paired[f"{task['run_id']}_minus_{control}"] = paired_counts(new, old, files)
        payload = dict(schema=P.SCHEMA, campaign_sha256=spec["content_hash"],
                       population="official_validation_5M", single_seed_exploratory=True,
                       controls_sha256=registry["content_hash"], models=models,
                       contrasts=effects, per_check_contrasts=curves, paired_correctness=paired)
        payload["content_hash"] = P.digest(payload)
        directory = output / "reports"
        P.write_json(directory / "message_study_report.json", payload)
        lines = ["# Five-run EdgeValue message study", "",
                 "Five fresh seed-101 variants; two reused controls. These are screening validation",
                 "results, not independent test results or across-seed uncertainty estimates.",
                 "All results use the selected checkpoint on the same 5M validation events.", "",
                 "| Model | Accuracy (%) | Delta EV (pp) | Delta ParT (pp) | OVO AUC | OVR AUC | Selected check |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in models.items():
            m = row["metrics"]
            ev, part = (models[c]["metrics"]["accuracy"] for c in ("EV_RANGER","PART_RANGER"))
            lines.append(f"| {name} | {100*m['accuracy']:.4f} | {100*(m['accuracy']-ev):+.4f} | "
                         f"{100*(m['accuracy']-part):+.4f} | {m['macro_auc_ovo']:.6f} | "
                         f"{m['macro_auc_ovr']:.6f} | {row['best_check']} |")
        lines += ["", "## Matched-check learning curves", "",
                  "| Check | " + " | ".join(models) + " |", "|---:|" + "---:|"*len(models)]
        for check in range(10):
            lines.append(f"| {check+1} | " + " | ".join(
                f"{100*row['validation_checks'][check]['accuracy']:.4f}%" for row in models.values()) + " |")
        lines += ["", "## QCD rejection", "",
                  "Zero accepted QCD events are marked ZB; passing counts remain in JSON.",
                  "", "| Signal / efficiency | " + " | ".join(models) + " |", "|---|" + "---:|"*len(models)]
        for signal, targets in models["PART_RANGER"]["metrics"]["qcd_rejection"].items():
            for target in targets:
                cells = []
                for row in models.values():
                    value = row["metrics"]["qcd_rejection"][signal][target]["rejection"]
                    cells.append("ZB" if value is None else f"{value:.2f}")
                lines.append(f"| {signal} / {100*float(target):g}% | " + " | ".join(cells) + " |")
        lines += ["", "## Cost of the new variants", "",
                  "| Model | Parameters | Benchmark jets/s | Full-length peak GiB | Training h | AMP skips |",
                  "|---|---:|---:|---:|---:|---:|"]
        for task in P.TASKS:
            row = models[task["run_id"]]
            bench = P.read_json(output / "benchmarks" / (task["run_id"] + ".json"))
            lines.append(f"| {task['run_id']} | {P.COUNTS[task['architecture']]:,} | {bench['jets_per_second']:.1f} | "
                         f"{bench['full_length_peak_bytes']/1024**3:.2f} | "
                         f"{row['training']['train_seconds']/3600:.2f} | {row['training']['amp_skipped_steps']} |")
        lines += ["", "Parameter-matched controls are deferred. A first single-seed gain does not",
                  "separate the proposed mechanism from added capacity. Timing excludes queue and replay time.", ""]
        (directory / "message_study_report.md").write_text("\n".join(lines), encoding="utf-8")
        print(directory / "message_study_report.md", flush=True)
        return payload
