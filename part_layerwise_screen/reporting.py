"""The 2x2 shared/layerwise x EV-off/on comparison on matched validation jets."""
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
import numpy as np
from . import protocol as P
from part_screen_common.contracts import hashed, checked_json
from part_screen_common.controls import refresh_controls, selected_training, validate_predictions


def contrasts(values):
    shared_ev = values["EV_RANGER"] - values["PART_RANGER"]
    layerwise_ev = values["EV_LAYERWISE_RANGER"] - values["PART_LAYERWISE_RANGER"]
    return dict(ev_gain_shared=shared_ev, ev_gain_layerwise=layerwise_ev,
        layerwise_gain_without_ev=values["PART_LAYERWISE_RANGER"] - values["PART_RANGER"],
        layerwise_gain_with_ev=values["EV_LAYERWISE_RANGER"] - values["EV_RANGER"],
        ev_by_layerwise_interaction=layerwise_ev - shared_ev)


def paired_accuracy(new, control):
    by_name = {r["name"]: r for r in control["predictions"]}
    if {r["name"] for r in new["predictions"]} != set(by_name):
        raise ValueError("Paired prediction files differ")
    wins = losses = ties = 0
    for row in new["predictions"]:
        with np.load(row["path"], allow_pickle=False) as a, np.load(by_name[row["name"]]["path"], allow_pickle=False) as b:
            if not np.array_equal(a["labels"], b["labels"]):
                raise ValueError("Paired file/entry labels differ")
            x, y = a["logits"].argmax(1) == a["labels"], b["logits"].argmax(1) == b["labels"]
            wins += int((x & ~y).sum())
            losses += int((~x & y).sum())
            ties += int((x == y).sum())
    return dict(wins=wins, losses=losses, ties=ties, events=wins+losses+ties,
                accuracy_difference=(wins-losses)/(wins+losses+ties))


def report(output):
    output = Path(output)
    spec = P.load_campaign(output)
    with P.file_lock(output / ".report.lock", blocking=False):
        registry = refresh_controls(output, spec["donor"])
        if not registry["ready"]:
            P.write_json(output / "reports/progress.json", dict(complete=False, reason="donor_controls_pending",
                controls={k: r["status"] for k, r in registry["controls"].items()}))
            print("Controls pending: no final comparison written. Retry --mode report when they finish.")
            return False
        models = {name: dict(reused=True, parameters=spec["donor"]["parameters"][name],
            training=row["training"], evaluation=row["evaluation"]) for name, row in registry["controls"].items()}
        lock = P.read_json(output / "selection/locked_checkpoints.json")
        if lock.get("campaign_sha256") != spec["content_hash"] or len(lock["checkpoints"]) != len(spec["tasks"]):
            raise ValueError("Checkpoint lock differs")
        for index, task in enumerate(spec["tasks"]):
            run = P.task_directory(output, index)
            training = selected_training(run, task, spec["content_hash"], spec["recipes"][task["recipe"]], spec["donor"]["runtime"])
            expected = dict(task=task, checkpoint=training["checkpoint"], sha256=training["checkpoint_sha256"])
            if lock["checkpoints"][index] != expected:
                raise ValueError("Selected checkpoint changed after lock")
            if training["optimizer"]["parameters"] != spec["parameters"][task["architecture"]]:
                raise ValueError("Optimizer parameter coverage differs")
            evaluation = validate_predictions(run, task, spec["content_hash"], training["checkpoint_sha256"],
                spec["inventory"]["splits"]["val"]["files"], population=spec["recipes"][task["recipe"]]["evaluation_population"])
            models[task["run_id"]] = dict(reused=False, parameters=spec["parameters"][task["architecture"]],
                training=training, evaluation=evaluation, startup=P.read_json(run / "startup.json"))
        base = models["PART_RANGER"]["evaluation"]
        if any(row["evaluation"]["labels_by_file"] != base["labels_by_file"] for row in models.values()):
            raise ValueError("Comparison event labels/coverage differ")
        effects = {metric: contrasts({name: r["evaluation"]["metrics"][metric] for name, r in models.items()})
                   for metric in ("accuracy", "macro_auc_ovo", "macro_auc_ovr", "loss")}
        pairs = {label: paired_accuracy(models[a]["evaluation"], models[b]["evaluation"])
            for label, a, b in (("ev_shared", "EV_RANGER", "PART_RANGER"),
                ("ev_layerwise", "EV_LAYERWISE_RANGER", "PART_LAYERWISE_RANGER"),
                ("layerwise_without_ev", "PART_LAYERWISE_RANGER", "PART_RANGER"),
                ("layerwise_with_ev", "EV_LAYERWISE_RANGER", "EV_RANGER"))}
        identity = hashed(dict(campaign_sha256=spec["content_hash"], controls_sha256=registry["content_hash"],
            checkpoints={name: r["training"]["checkpoint_sha256"] for name, r in models.items()},
            metrics={name: r["evaluation"]["metrics_sha256"] for name, r in models.items()}))
        identity_path = output / "selection/comparison_inputs.json"
        if identity_path.exists() and checked_json(identity_path) != identity:
            raise ValueError("Completed comparison artifacts changed")
        P.write_json(identity_path, identity)
        directory = output / "reports"
        result = dict(schema="part_layerwise_report_v1", complete=True, campaign_sha256=spec["content_hash"],
            population="official_validation_5M", single_seed_exploratory=True, admission=spec["admission"],
            comparison_sha256=identity["content_hash"], models=models, contrasts=effects, paired_accuracy=pairs)
        P.write_json(directory / "layerwise_screen_report.json", result)
        lines = ["# Shared versus layerwise attention bias", "",
            "Seed 101; matched short Ranger recipe. Screening validation, not independent test results. Only the layerwise models were trained here. No separate GPU preflight was requested.", "",
            "| Model | Reused | Parameters | Accuracy (%) | Delta ParT (pp) | Macro OVO AUC | Macro OVR AUC | Selected check |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name in ("PART_RANGER", "EV_RANGER", "PART_LAYERWISE_RANGER", "EV_LAYERWISE_RANGER"):
            row = models[name]
            m = row["evaluation"]["metrics"]
            lines.append(f"| {name} | {row['reused']} | {row['parameters']:,} | {100*m['accuracy']:.4f} | "
                f"{100*(m['accuracy']-base['metrics']['accuracy']):+.4f} | {m['macro_auc_ovo']:.6f} | "
                f"{m['macro_auc_ovr']:.6f} | {row['training']['best_check']} |")
        lines += ["", "## Matched contrasts", "", "| Contrast | Accuracy delta (pp) | OVO AUC delta |",
                  "|---|---:|---:|"]
        for name, value in effects["accuracy"].items():
            lines.append(f"| {name} | {100*value:+.4f} | {effects['macro_auc_ovo'][name]:+.6f} |")
        lines += ["", "Positive interaction means the EV accuracy gain is larger with layerwise bias. Single-seed point differences do not establish reproducibility.", "",
            "## QCD rejection", "", "ZB means zero QCD acceptances; counts are retained in JSON.", "",
            "| Signal / efficiency | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
        for signal, targets in base["metrics"]["qcd_rejection"].items():
            for target in targets:
                values = [row["evaluation"]["metrics"]["qcd_rejection"][signal][target]["rejection"] for row in models.values()]
                lines.append(f"| {signal} / {100*float(target):g}% | " + " | ".join("ZB" if v is None else f"{v:.2f}" for v in values) + " |")
        with NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
            handle.write("\n".join(lines) + "\n")
            temporary = handle.name
        os.replace(temporary, directory / "layerwise_screen_report.md")
        print(directory / "layerwise_screen_report.md")
        return True
