"""One-seed screening contrasts on a single common validation population."""
from pathlib import Path
from . import protocol as P


def contrasts(values):
    ev = values["EV_RANGER"] - values["PART_RANGER"]
    adam = values["EV_ADAMW"] - values["PART_ADAMW"]
    return {
        "ev_vs_part_ranger": ev,
        "context_vs_ev": values["EV_CONTEXT_RANGER"] - values["EV_RANGER"],
        "gate_vs_ev": values["EV_GATE_RANGER"] - values["EV_RANGER"],
        "combined_vs_ev": values["EV_CONTEXT_GATE_RANGER"] - values["EV_RANGER"],
        "context_gate_interaction": values["EV_CONTEXT_GATE_RANGER"] - values["EV_CONTEXT_RANGER"] - values["EV_GATE_RANGER"] + values["EV_RANGER"],
        "ev_vs_part_adamw": adam,
        "optimizer_recipe_ev_interaction": adam - ev,
    }


def report(output):
    output = Path(output)
    spec = P.load_campaign(output)
    models, identities, labels = {}, [], []
    import numpy as np
    for index, task in enumerate(spec["tasks"]):
        run = P.task_directory(output, index)
        row = P.read_json(run / "full_validation/metrics.json")
        completed = P.read_json(run / "training_complete.json")
        if (row["task"] != task or row["campaign_sha256"] != spec["content_hash"]
                or row["checkpoint_sha256"] != P.file_sha256(run / "best_model_val.pt")
                or row["checkpoint_sha256"] != completed["best_checkpoint_sha256"]):
            raise ValueError("Report task/checkpoint/campaign mismatch")
        identities.append([r["contract"]["file"] for r in row["identity_manifest"]])
        if identities[-1] != sorted(spec["inventory"]["splits"]["val"]["files"], key=lambda r: r["name"]):
            raise ValueError("Full-validation identity list differs from the frozen population")
        label_parts = []
        for number, record in enumerate(row["identity_manifest"]):
            path = run / "full_validation" / f"file_{number:03d}.npz"
            if P.file_sha256(path) != record["sha256"]:
                raise ValueError("Prediction file changed after metric calculation")
            with np.load(path, allow_pickle=False) as saved:
                label_parts.append(saved["labels"])
        labels.append(np.concatenate(label_parts))
        population = spec["recipes"][task["recipe"]]["evaluation_population"]
        if (len(labels[-1]) != population or row["metrics"]["events"] != population
                or not np.array_equal(np.bincount(labels[-1], minlength=10), np.full(10, population//10))):
            raise ValueError("Report population size/class balance differs")
        checks = [P.read_json(run / "validation" / f"check_{c:02d}.json") for c in range(1,11)]
        models[task["run_id"]] = dict(task=task, parameters=spec["parameters"][task["architecture"]],
            full_validation=row["metrics"], validation_checks=checks, training=completed,
            benchmark=P.read_json(output / "benchmarks" / (task["run_id"] + ".json")),
            diagnostics=[P.read_json(p) for p in sorted((run / "diagnostics").glob("check_*.json"))])
    if any(i != identities[0] for i in identities) or any(not np.array_equal(y, labels[0]) for y in labels):
        raise ValueError("Models did not evaluate the same file/entry identities and labels")
    effects = {metric: contrasts({name: row["full_validation"][metric] for name,row in models.items()})
               for metric in ("accuracy", "macro_auc_ovo", "macro_auc_ovr", "loss")}
    curves = {metric: [contrasts({name: row["validation_checks"][c][metric] for name,row in models.items()})
                       for c in range(10)] for metric in ("accuracy", "macro_auc_ovo", "macro_auc_ovr", "loss")}
    directory = output / "reports"
    P.write_json(directory / "screen_report.json", dict(schema=P.SCHEMA, campaign_sha256=spec["content_hash"],
        population="official_validation_5M", single_seed_exploratory=True, models=models,
        contrasts=effects, per_check_contrasts=curves))
    text = ["# EdgeValue context and relation-gating screen", "",
        "All results are single-seed exploratory validation results, not test results. "
        "The 5M evaluation overlaps checkpoint validation and is not an independent holdout. "
        "AdamW comparisons change the complete optimizer recipe, including decay, clipping and schedule.", "",
        "| Run | Parameters | Accuracy (%) | Δ ParT Ranger (pp) | OVO AUC | OVR AUC | CE | Selected check |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    baseline = models["PART_RANGER"]["full_validation"]["accuracy"]
    for name, row in models.items():
        m = row["full_validation"]
        text.append(f"| {name} | {row['parameters']:,} | {100*m['accuracy']:.4f} | {100*(m['accuracy']-baseline):+.4f} | {m['macro_auc_ovo']:.6f} | {m['macro_auc_ovr']:.6f} | {m['loss']:.5f} | {row['training']['best_check']} |")
    text += ["", "## Matched contrasts", "", "| Contrast | Accuracy Δ (pp) | OVO AUC Δ |", "|---|---:|---:|"]
    for name, value in effects["accuracy"].items():
        text.append(f"| {name} | {100*value:+.4f} | {effects['macro_auc_ovo'][name]:+.6f} |")
    text += ["", "## QCD rejection", "", "ZB means zero accepted QCD events; its rejection is not replaced by a finite number. Passing counts are retained in JSON.", "",
        "| Signal / efficiency | " + " | ".join(models) + " |", "|---|" + "---:|"*len(models)]
    for signal, targets in models["PART_RANGER"]["full_validation"]["qcd_rejection"].items():
        for target in targets:
            cells = []
            for row in models.values():
                value = row["full_validation"]["qcd_rejection"][signal][target]["rejection"]
                cells.append("ZB" if value is None else f"{value:.2f}")
            text.append(f"| {signal} / {100*float(target):g}% | " + " | ".join(cells) + " |")
    text += ["", "## Cost", "", "| Run | Benchmark jets/s | Peak full-length GiB | Training h | Validation h | AMP skips |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in models.items():
        b, t = row["benchmark"], row["training"]
        text.append(f"| {name} | {b['jets_per_second']:.1f} | {b['full_length_peak_bytes']/1024**3:.2f} | {t['train_seconds']/3600:.2f} | {t['validation_seconds']/3600:.2f} | {t['amp_skipped_steps']} |")
    text += ["", "Timing excludes queue time and restart data replay. Learning curves use matched check numbers, not each run's best point at unequal progress.", ""]
    (directory / "screen_report.md").write_text("\n".join(text), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(17,4), layout="constrained")
    for name, row in models.items():
        axes[0].plot(range(1,11), [100*c["accuracy"] for c in row["validation_checks"]], label=name)
        axes[1].plot(range(1,11), [c["macro_auc_ovo"] for c in row["validation_checks"]])
    for contrast in effects["accuracy"]:
        axes[2].plot(range(1,11), [100*c[contrast] for c in curves["accuracy"]], label=contrast)
    for axis in axes:
        axis.set_xlabel("Validation check (20,000 updates)")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].set_ylabel("Macro OVO AUC")
    axes[2].set_ylabel("Matched accuracy difference (pp)")
    axes[2].axhline(0, color="black", linewidth=.6)
    axes[0].legend(fontsize=6)
    axes[2].legend(fontsize=5)
    figure.savefig(directory / "learning_curves.pdf")
    figure.savefig(directory / "learning_curves.png", dpi=180)
    plt.close(figure)
    print(directory / "screen_report.md", flush=True)
