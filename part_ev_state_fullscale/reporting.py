"""Audit all fifty checks and report matched seeds, never inflate the single EV seed."""
from pathlib import Path
import statistics
import math
import numpy as np
from part_ev_fullscale.protocol import read_json, file_sha256
from part_ev_fullscale.checkpoint import trusted_load
from part_ev_message_study.controls import runtime_identity, validate_evaluation
from . import protocol as P

def validate_training(root, spec, task, runtime):
    run = root / "runs" / task["run_id"] / f"seed_{task['seed']}"
    complete = read_json(run / "training_complete.json")
    recipe = spec["recipes"][task["recipe"]]
    if (complete["task"] != task or complete["campaign_sha256"] != spec["content_hash"]
            or complete["step"] != recipe["total_steps"]
            or complete["validated_checks"] != recipe["epochs"]
            or runtime_identity(complete["runtime"]) != runtime
            or complete["runtime"]["campaign_sha256"] != spec["content_hash"]):
        raise ValueError("Training completion/recipe/runtime mismatch")
    checks = [read_json(run / "validation" / f"check_{i:02d}.json") for i in range(1, recipe["epochs"] + 1)]
    if len(list((run / "validation").glob("check_*.json"))) != recipe["epochs"]:
        raise ValueError("Unexpected validation checks beyond the frozen schedule")
    for i, row in enumerate(checks, 1):
        if (row["check"] != i or row["step"] != i * recipe["steps_per_epoch"]
                or row["events"] != recipe["validation_events_per_epoch"]
                or not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1):
            raise ValueError("Invalid/incomplete validation check sequence")
    best = max(checks, key=lambda row: row["accuracy"])  # Python retains first tie.
    if complete["best_check"] != best["check"] or complete["best_accuracy"] != best["accuracy"]:
        raise ValueError("Selected check is not the first highest-accuracy check")
    checkpoint = run / "best_model_val.pt"
    checksum = file_sha256(checkpoint)
    if checksum != complete["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    saved = trusted_load(checkpoint)
    if (saved["task"] != task or saved["campaign_sha256"] != spec["content_hash"]
            or saved["runtime"] != complete["runtime"] or saved["validation"] != best):
        raise ValueError("Selected checkpoint metadata differs from validated training")
    optimizer = read_json(run / "optimizer.json")
    if (optimizer["optimizer"] != "Ranger"
            or optimizer["parameters"] != spec["parameters"][task["architecture"]]):
        raise ValueError("Control optimizer/parameter count differs")
    for group in optimizer["groups"]:
        for field in ("weight_decay", "betas", "eps"):
            if group[field] != recipe[field]:
                raise ValueError("Control optimizer settings differ")
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in
              [root / "campaign.json", run / "training_complete.json",
               run / "optimizer.json", checkpoint, *sorted((run / "validation").glob("check_*.json"))]}
    return dict(task=task, checkpoint_sha256=checksum, best_check=best["check"],
                training=complete, validation_checks=checks, artifact_sha256=hashes)



def summarize(models):
    metrics = ("accuracy", "macro_auc_ovo", "macro_auc_ovr", "loss")
    result = {}
    for metric in metrics:
        paired = [models[P.task_key(P.TASKS[i+3])]["metrics"][metric] -
                  models[P.task_key(P.TASKS[i])]["metrics"][metric] for i in range(3)]
        result[metric] = dict(state_minus_part=dict(seeds=list(P.SEEDS), values=paired,
            mean=statistics.mean(paired), sample_std=statistics.stdev(paired)),
            ev_minus_part_seed101=(models[P.task_key(P.TASKS[6])]["metrics"][metric] -
                                  models[P.task_key(P.TASKS[0])]["metrics"][metric]),
            state_minus_ev_seed101=(models[P.task_key(P.TASKS[3])]["metrics"][metric] -
                                   models[P.task_key(P.TASKS[6])]["metrics"][metric]))
    return result


def report(output, spec, engine):
    output = Path(output)
    with P.file_lock(output / ".report.lock", blocking=False):
        summary = P.require_approval(output, spec)
        engine.lock_finalists(output, spec)
        models, common_labels = {}, None
        for task in spec["tasks"]:
            trained = validate_training(output, spec, task, runtime_identity(summary["runtime"]))
            row, labels = validate_evaluation(output, spec, task, trained,
                                             recompute=engine.classification_metrics)
            if common_labels is not None and not np.array_equal(common_labels, labels):
                raise ValueError("Evaluation event labels differ between runs")
            common_labels = labels
            models[P.task_key(task)] = row
        effects = summarize(models)
        payload = dict(schema=P.SCHEMA, campaign_sha256=spec["content_hash"],
            population="official_validation_5M", independent_test=False, models=models,
            contrasts=effects, main_comparison_seeds=list(P.SEEDS), ordinary_ev_seeds=[101])
        payload["content_hash"] = P.digest(payload)
        directory = output / "reports"
        P.write_json(directory / "state_fullscale_report.json", payload)
        lines = ["# Full-schedule ParT versus last-four State-EV", "",
            "Fresh 1M-step runs; 50 checkpoint-validation checks. These are full-5M",
            "validation results, overlapping checkpoint selection, not independent test results.",
            "The main contrast has three matched seeds. Ordinary EV has seed 101 only.",
            "Sample standard deviations describe seed variation, not confidence intervals.", "",
            "| Model | Seed | Accuracy (%) | OVO AUC | OVR AUC | Selected check |",
            "|---|---:|---:|---:|---:|---:|"]
        for task in spec["tasks"]:
            row = models[P.task_key(task)]
            m = row["metrics"]
            lines.append(f"| {task['run_id']} | {task['seed']} | {100*m['accuracy']:.4f} | "
                         f"{m['macro_auc_ovo']:.6f} | {m['macro_auc_ovr']:.6f} | {row['best_check']} |")
        lines += ["", "## Main three-seed comparison", "",
                  "| Quantity | Accuracy (percentage points) | OVO AUC |", "|---|---:|---:|"]
        for arch in ("part", "state_late4"):
            rows = [models[P.task_key(t)]["metrics"] for t in spec["tasks"] if t["architecture"] == arch]
            acc, auc = ([r[key] for r in rows] for key in ("accuracy", "macro_auc_ovo"))
            lines.append(f"| {arch} mean +/- sample SD | {100*statistics.mean(acc):.4f} +/- "
                         f"{100*statistics.stdev(acc):.4f} | {statistics.mean(auc):.6f} +/- {statistics.stdev(auc):.6f} |")
        acc, auc = (effects[key]["state_minus_part"] for key in ("accuracy", "macro_auc_ovo"))
        lines.append(f"| Paired State - ParT | {100*acc['mean']:+.4f} +/- {100*acc['sample_std']:.4f} | "
                     f"{auc['mean']:+.6f} +/- {auc['sample_std']:.6f} |")
        lines += ["", "## Seed-101-only ordinary-EV comparisons", ""]
        for label in ("ev_minus_part_seed101", "state_minus_ev_seed101"):
            lines.append(f"- {label}: {100*effects['accuracy'][label]:+.4f} percentage points; "
                         f"OVO AUC {effects['macro_auc_ovo'][label]:+.6f}.")
        lines += ["", "## Learning curves", "", "| Check | " + " | ".join(models) + " |",
                  "|---:|" + "---:|" * len(models)]
        for check in range(50):
            lines.append(f"| {check+1} | " + " | ".join(
                f"{100*r['validation_checks'][check]['accuracy']:.4f}%" for r in models.values()) + " |")
        lines += ["", "## QCD rejection", "", "ZB denotes zero accepted QCD events; counts remain in JSON.",
                  "", "| Signal / efficiency | " + " | ".join(models) + " |", "|---|" + "---:|"*len(models)]
        for signal, targets in next(iter(models.values()))["metrics"]["qcd_rejection"].items():
            for target in targets:
                values = [r["metrics"]["qcd_rejection"][signal][target]["rejection"] for r in models.values()]
                lines.append(f"| {signal} / {100*float(target):g}% | " +
                             " | ".join("ZB" if v is None else f"{v:.2f}" for v in values) + " |")
        lines += ["", "## Recorded cost", "",
                  "| Model / seed | Parameters | Training h | Checkpoint-validation h | AMP skips |",
                  "|---|---:|---:|---:|---:|"]
        for key, row in models.items():
            t = row["training"]
            lines.append(f"| {key} | {P.COUNTS[row['task']['architecture']]:,} | "
                         f"{t['train_seconds']/3600:.2f} | {t['validation_seconds']/3600:.2f} | {t['amp_skipped_steps']} |")
        lines += ["", "Recorded timers exclude queue time, startup/replay and some checkpoint/diagnostic overhead.", ""]
        (directory / "state_fullscale_report.md").write_text("\n".join(lines), encoding="utf-8")
        print(directory / "state_fullscale_report.md", flush=True)
        return payload

