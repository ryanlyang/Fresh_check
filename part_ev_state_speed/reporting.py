"""Separate accuracy preservation from walltime and forward-only speed."""
from pathlib import Path
import numpy as np
from part_ev_message_study import controls as C, protocol as historical
from . import protocol as P


def report(output, spec, engine):
    output = Path(output)
    with P.file_lock(output / ".report.lock", blocking=False):
        admission = P.require_approval(output, spec)
        donor_spec, donor = C.donor_contract(spec["donor"]["path"])
        if donor != spec["donor"]:
            raise ValueError("Control donor contract changed")
        registry, labels = C.registry(donor["path"], donor_spec, donor,
            require_complete=True, recompute=engine.classification_metrics)
        models = registry["controls"]
        root = Path(spec["state_reference"]["path"])
        state_spec = historical.load_campaign(root)
        task = next(t for t in state_spec["tasks"] if t["run_id"] == "EV_STATE_RANGER")
        training = C.validate_training(root, state_spec, task, donor["runtime"])
        models[task["run_id"]], labels[task["run_id"]] = C.validate_evaluation(
            root, state_spec, task, training, recompute=engine.classification_metrics)
        for task in P.TASKS:
            trained = C.validate_training(output, spec, task, donor["runtime"])
            models[task["run_id"]], labels[task["run_id"]] = C.validate_evaluation(
                output, spec, task, trained, recompute=engine.classification_metrics)
        if any(not np.array_equal(y, labels["EV_RANGER"]) for y in labels.values()):
            raise ValueError("Evaluation identities/labels differ")
        timing = {t["run_id"]: P.read_json(output / "benchmarks" /
            (t["run_id"] + "_speed.json")) for t in P.TASKS}
        effects = {name: {control: row["metrics"]["accuracy"] - models[control]["metrics"]["accuracy"]
                  for control in ("PART_RANGER", "EV_RANGER", "EV_STATE_RANGER")}
                  for name, row in models.items()}
        payload = dict(campaign_sha256=spec["content_hash"], models=models, accuracy_deltas=effects,
                       inference=timing, admission=admission,
                       population="official_validation_5M", single_seed_exploratory=True)
        payload["content_hash"] = P.digest(payload)
        directory = output / "reports"
        P.write_json(directory / "state_speed_report.json", payload)
        lines = ["# State EV: speed versus validation performance", "",
            "Seed 101, matched shortened Ranger recipe. Screening validation, not independent test results.",
            "Checkpoint selection uses accuracy. All accuracy rows below use the same full 5M population.", "",
            "| Model | Accuracy (%) | Delta EV (pp) | Delta original state (pp) | OVO AUC | Check |",
            "|---|---:|---:|---:|---:|---:|"]
        for name, row in models.items():
            m = row["metrics"]
            lines.append(f"| {name} | {100*m['accuracy']:.4f} | {100*effects[name]['EV_RANGER']:+.4f} | "
                         f"{100*effects[name]['EV_STATE_RANGER']:+.4f} | {m['macro_auc_ovo']:.6f} | {row['best_check']} |")
        lines += ["", "## Measured costs", "",
            "Training jets/s includes the streaming loader; forward timings exclude loading/transfers and metrics.",
            "Inference uses FP32, identical resident inputs, and fresh weights for timing only.", "",
            "| Model | Train jets/s | Train hours recorded | FP32 jets/s (B=512) | Time / EV | Time / original state | Latency B=1 (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for task, bench in zip(P.TASKS, admission["models"]):
            name = task["run_id"]
            rows = timing[name]["models"]
            speed = rows[name]["512"]
            lines.append(f"| {name} | {bench['jets_per_second']:.1f} | "
                f"{models[name]['training']['train_seconds']/3600:.2f} | {speed['jets_per_second']:.1f} | "
                f"{speed['seconds_per_batch']/rows['EV']['512']['seconds_per_batch']:.3f} | "
                f"{speed['seconds_per_batch']/rows['STATE_original']['512']['seconds_per_batch']:.3f} | "
                f"{rows[name]['1']['milliseconds_per_batch']:.3f} |")
        lines += ["", "Smaller time ratios are better. Do not infer accuracy preservation from numerical parity alone:",
                  "the late-four variants are new architectures, and optimized retraining can also differ numerically.",
                  "All configurations are reported, regardless of outcome. No speedup or retained-gain threshold was used to select results."]
        (directory / "state_speed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(directory / "state_speed_report.md", flush=True)
        return payload
