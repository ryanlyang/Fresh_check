"""Join new grouped results and verified read-only controls by event identity."""
from pathlib import Path
import numpy as np
from . import protocol as P
from part_screen_common.contracts import hashed, checked_json
from part_screen_common.controls import refresh_controls, selected_training, validate_predictions


def contrasts(values):
    baseline = values["PART_RANGER"]
    result = {f"{name}_minus_PART_RANGER": value - baseline for name, value in values.items() if name != "PART_RANGER"}
    for a, b, label in (("PART_GROUP_FIXED_RANGER", "PART_GROUP_RANDOM_RANGER", "fixed_minus_random"),
                         ("PART_GROUP_HYBRID_RANGER", "PART_GROUP_LEARNED_RANGER", "hybrid_minus_learned")):
        if a in values and b in values:
            result[label] = values[a] - values[b]
    if "EV_RANGER" in values:
        for name in values:
            if name.startswith("PART_GROUP_"):
                result[f"{name}_minus_EV_RANGER"] = values[name] - values["EV_RANGER"]
    return result


def paired_accuracy(new, control):
    by_name = {r["name"]: r for r in control["predictions"]}
    wins = losses = ties = 0
    for row in new["predictions"]:
        old = by_name[row["name"]]
        with np.load(row["path"], allow_pickle=False) as a, np.load(old["path"], allow_pickle=False) as b:
            if not np.array_equal(a["labels"], b["labels"]):
                raise ValueError("Paired file/entry labels differ")
            x = a["logits"].argmax(1) == a["labels"]
            y = b["logits"].argmax(1) == b["labels"]
            wins += int((x & ~y).sum())
            losses += int((~x & y).sum())
            ties += int((x == y).sum())
    return dict(wins=wins, losses=losses, ties=ties, events=wins + losses + ties,
                accuracy_difference=(wins - losses) / (wins + losses + ties))


def report(output):
    output = Path(output)
    spec = P.load_campaign(output)
    P.require_approval(output, spec)
    with P.file_lock(output / ".report.lock", blocking=False):
        registry = refresh_controls(output, spec["donor"])
        if not registry["ready"]:
            P.write_json(output / "reports/progress.json", dict(complete=False, reason="donor_controls_pending",
                controls={k: v["status"] for k, v in registry["controls"].items()}))
            print("Required donor results are pending; no completed comparative report written.", flush=True)
            return False
        models = {}
        for name, record in registry["controls"].items():
            models[name] = dict(task=record["task"], reused=True, training=record["training"], evaluation=record["evaluation"],
                parameters=record["training"]["optimizer"]["parameters"],
                benchmark=P.read_json(Path(spec["donor"]["path"]) / "benchmarks" / f"{name}.json"))
        lock = P.read_json(output / "selection/locked_checkpoints.json")
        for index, task in enumerate(spec["tasks"]):
            run = P.task_directory(output, index)
            training = selected_training(run, task, spec["content_hash"], spec["recipes"][task["recipe"]], spec["donor"]["runtime"])
            locked = [r for r in lock["checkpoints"] if r["task"] == task]
            if (lock["campaign_sha256"] != spec["content_hash"] or len(locked) != 1
                    or locked[0]["sha256"] != training["checkpoint_sha256"]):
                raise ValueError("New selected-checkpoint lock differs")
            evaluation = validate_predictions(run, task, spec["content_hash"], training["checkpoint_sha256"],
                spec["inventory"]["splits"]["val"]["files"])
            models[task["run_id"]] = dict(task=task, reused=False, training=training, evaluation=evaluation,
                parameters=spec["parameters"][task["architecture"]],
                benchmark=P.read_json(output / "benchmarks" / f"{task['run_id']}.json"),
                diagnostics=[P.read_json(p) for p in sorted((run / "diagnostics").glob("check_*.json"))])
        base = models["PART_RANGER"]["evaluation"]
        if any(row["evaluation"]["labels_by_file"] != base["labels_by_file"] for row in models.values()):
            raise ValueError("Models did not evaluate the same file/entry labels")
        paired = {name: paired_accuracy(row["evaluation"], base) for name, row in models.items() if name != "PART_RANGER"}
        metrics = ("accuracy", "macro_auc_ovo", "macro_auc_ovr", "loss")
        effects = {metric: contrasts({n: r["evaluation"]["metrics"][metric] for n, r in models.items()}) for metric in metrics}
        comparison = hashed(dict(schema="part_group_comparison_lock_v1", campaign_sha256=spec["content_hash"],
            controls_sha256=registry["content_hash"], models={n: dict(checkpoint_sha256=r["training"]["checkpoint_sha256"],
            metrics_sha256=r["evaluation"]["metrics_sha256"], predictions=r["evaluation"]["predictions"]) for n, r in models.items()}))
        lock_path = output / "comparison_lock.json"
        if lock_path.exists() and checked_json(lock_path) != comparison:
            raise ValueError("Completed comparison inputs changed")
        if not lock_path.exists():
            P.write_json(lock_path, comparison)
        directory = output / "reports"
        result = dict(schema="part_group_report_v1", complete=True, campaign_sha256=spec["content_hash"],
            comparison_sha256=comparison["content_hash"], population="official_validation_5M",
            single_seed_exploratory=True, models=models, contrasts=effects, paired_accuracy_vs_part=paired)
        P.write_json(directory / "grouped_screen_report.json", result)
        lines = ["# Grouped-attention screen", "", "Seed 101, matched short Ranger recipe. These are screening validation results, not independent test results. Controls marked reused were not retrained.", "",
            "| Model | Reused | Parameters | Accuracy (%) | Delta ParT (pp) | OVO AUC | OVR AUC | CE | Selected check |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in models.items():
            m = row["evaluation"]["metrics"]
            lines.append(f"| {name} | {row['reused']} | {row['parameters']:,} | {100*m['accuracy']:.4f} | "
                f"{100*(m['accuracy']-base['metrics']['accuracy']):+.4f} | {m['macro_auc_ovo']:.6f} | {m['macro_auc_ovr']:.6f} | {m['loss']:.5f} | {row['training']['best_check']} |")
        lines += ["", "## Matched contrasts", "", "| Contrast | Accuracy delta (pp) | OVO AUC delta |", "|---|---:|---:|"]
        for name, value in effects["accuracy"].items():
            lines.append(f"| {name} | {100*value:+.4f} | {effects['macro_auc_ovo'][name]:+.6f} |")
        lines += ["", "## QCD rejection", "", "ZB denotes zero accepted QCD; no finite rejection is invented. Counts and achieved signal efficiency are retained in JSON.", "",
            "| Signal / efficiency | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
        for signal, targets in base["metrics"]["qcd_rejection"].items():
            for target in targets:
                values = [r["evaluation"]["metrics"]["qcd_rejection"][signal][target]["rejection"] for r in models.values()]
                lines.append(f"| {signal} / {100*float(target):g}% | " + " | ".join("ZB" if v is None else f"{v:.2f}" for v in values) + " |")
        lines += ["", "## Measured cost", "", "| Model | Train jets/s | Full-length GiB | Validation jets/s (load + forward) | Validation jets/s (including metrics) | Group probe ms/jet | Train hours | AMP skips |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in models.items():
            b, t = row["benchmark"], row["training"]["complete"]
            v = b["validation"]
            diagnostic = b.get("diagnostics", {})
            group_ms = (f"{1000*diagnostic['group_forward_seconds']/diagnostic['diagnostic_events']:.3f}"
                        if diagnostic.get("diagnostic_events", 0) else "--")
            lines.append(f"| {name} | {b['jets_per_second']:.1f} | {b['full_length_peak_bytes']/1024**3:.2f} | "
                f"{v['events']/v['inference_seconds']:.1f} | {v['events']/v['seconds']:.1f} | {group_ms} | "
                f"{t['train_seconds']/3600:.2f} | {t['amp_skipped_steps']} |")
        lines += ["", "Timing excludes queue time and restart stream replay. Group timing is an FP32 diagnostic microprobe (at most 16 jets), includes diagnostic collection/synchronization, and is not a per-training-jet estimate. Optional comparisons against EV do not isolate grouping added to EV.", ""]
        # Atomic text output using the same safe replace convention as JSON.
        from tempfile import NamedTemporaryFile
        import os
        with NamedTemporaryFile("w", encoding="utf-8", dir=directory, prefix="report.", suffix=".tmp", delete=False) as f:
            f.write("\n".join(lines))
            temp = f.name
        os.replace(temp, directory / "grouped_screen_report.md")
        plots(directory, models)
        print(directory / "grouped_screen_report.md", flush=True)
        return True


def plots(directory, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), layout="constrained")
    base = np.array([c["accuracy"] for c in models["PART_RANGER"]["training"]["checks"]])
    for name, row in models.items():
        checks = row["training"]["checks"]
        x = [c["check"] for c in checks]
        accuracy = np.array([c["accuracy"] for c in checks])
        axes[0].plot(x, 100 * accuracy, label=name)
        axes[1].plot(x, [c["macro_auc_ovo"] for c in checks])
        axes[2].plot(x, 100 * (accuracy - base))
    for axis, ylabel in zip(axes, ("Validation accuracy (%)", "Macro OVO AUC", "Delta at matched check (pp)")):
        axis.set_xlabel("Check (20,000 updates)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=5)
    axes[2].axhline(0, color="black", linewidth=.5)
    fig.savefig(directory / "learning_curves.pdf")
    plt.close(fig)
    diagnostic_dir = directory / "grouping_diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    for name, row in models.items():
        diagnostics = row.get("diagnostics", [])
        if not diagnostics:
            continue
        fig, grid = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
        axes = grid.flatten()
        checks = [d["check"] for d in diagnostics]
        axes[0].plot(checks, [d["mean_assignment_entropy"] for d in diagnostics])
        axes[0].set_ylabel("Mean assignment entropy")
        axes[1].plot(checks, [np.mean(d["effective_groups"]) for d in diagnostics])
        axes[1].set_ylabel("Effective groups")
        axes[2].plot(checks, [d["correction_to_backbone_norm"] for d in diagnostics])
        axes[2].set_ylabel("Correction/backbone norm")
        axes[3].plot(checks, [d["gamma"] for d in diagnostics])
        axes[3].set_ylabel("Residual gamma")
        axes[4].plot(checks, [np.mean(d["angular_width"]) for d in diagnostics])
        axes[4].set_ylabel("Mean angular width")
        axes[5].plot(checks, [d["mean_slot_similarity"] for d in diagnostics])
        axes[5].set_ylabel("Mean slot cosine similarity")
        axes[6].plot(checks, [np.mean(d["occupancy"], axis=0) for d in diagnostics])
        axes[6].set_ylabel("Occupancy per slot")
        axes[7].plot(checks, [np.mean(d["pt_share"], axis=0) for d in diagnostics])
        axes[7].set_ylabel("pT share per slot")
        for axis in axes:
            axis.set_xlabel("Validation check")
        fig.suptitle(name)
        fig.savefig(diagnostic_dir / f"{name}.pdf")
        plt.close(fig)
        examples = diagnostics[-1]["examples"]
        fig, axes = plt.subplots(1, len(examples["assignments"]), figsize=(14, 5), layout="constrained", squeeze=False)
        for i, (assignment, valid) in enumerate(zip(examples["assignments"], examples["valid"])):
            axes[0, i].imshow(np.asarray(assignment)[np.asarray(valid, bool)], aspect="auto", vmin=0, vmax=1)
            axes[0, i].set_xlabel("Group slot")
            axes[0, i].set_ylabel("Retained particle")
        fig.suptitle(name + " - fixed diagnostic sample, final check")
        fig.savefig(diagnostic_dir / f"{name}_assignments.pdf")
        plt.close(fig)
