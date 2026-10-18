"""Finite validation loading and strict, dependency-light six-checkpoint reports.

Training/model imports are deliberately deferred: the evaluation driver first
selects the campaign's frozen source, which owns preprocessing and metric math.
"""
from __future__ import annotations

import math
from numbers import Real
import statistics


CLASS_NAMES = ("QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl")
SEEDS = (101, 202, 303)
ARCHITECTURES = ("part", "part_ev")
EVENTS = 5_000_000
CLASS_EVENTS = 500_000
REJECTION_TARGETS = {
    name: (("0.99",) if name == "Hqql" else ("0.995",) if name == "Tbl" else ("0.3", "0.5"))
    for name in CLASS_NAMES[1:]
}


def finite_validation_loader(inventory, file_record, batch_size=512):
    """Read every event in one frozen validation file, once and in entry order."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Validation batch_size must be a positive integer")
    import torch
    from torch.utils.data import DataLoader
    from weaver.utils.dataset import SimpleIterDataset
    from part_ev_fullscale.data import (
        _SeededDataset, _stream_seed, _validated_config, _file_entry_counts, rank_file_dict,
    )

    records = inventory["splits"]["val"]["files"]
    if sum(row == file_record for row in records) != 1:
        raise ValueError("Validation file must occur exactly once in the frozen inventory")
    selected = {"splits": {"val": {"files": [file_record]}}}
    files = rank_file_dict("val", selected)
    options = dict(
        file_dict=files, data_config_file=str(_validated_config(None)),
        for_training=False, load_range_and_fraction=((0, 1), 1),
        fetch_by_files=True, fetch_step=1, file_fraction=1, remake_weights=False,
        async_load=False, infinity_mode=False, in_memory=False,
        name="fullscale_finite_validation",
    )
    dataset = SimpleIterDataset(**options)
    if dataset.config._missing_standardization_info or dataset.config.weight_name is not None:
        raise ValueError("Official validation preprocessing must be manual and unweighted")
    seed = _stream_seed(101, "val", 0, 0)
    entries = _file_entry_counts(inventory, "val", files)
    return DataLoader(
        _SeededDataset(dataset, options, entries, seed, False),
        batch_size=batch_size, drop_last=False, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _finite(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def _count(value, name, *, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"Invalid {name}")


def _validated_records(records):
    expected = {(architecture, seed) for architecture in ARCHITECTURES for seed in SEEDS}
    if len(records) != len(expected):
        raise ValueError("Need all six matched-seed validation results")
    by_task = {}
    try:
        for record in records:
            task = record["task"]
            key = (task["architecture"], task["seed"])
            if set(task) != {"architecture", "seed"} or type(task["seed"]) is not int:
                raise ValueError("Invalid validation task")
            if key not in expected or key in by_task:
                raise ValueError("Need each of the six matched-seed tasks exactly once")
            for field in ("evaluation_sha256", "inventory_sha256", "checkpoint_sha256"):
                if not isinstance(record[field], str) or not record[field]:
                    raise ValueError(f"Missing {field}")
            metrics = record["metrics"]
            if type(metrics["events"]) is not int or metrics["events"] != EVENTS:
                raise ValueError("Validation results must cover all 5M events")
            if (set(metrics["class_support"]) != set(CLASS_NAMES)
                    or any(type(count) is not int or count != CLASS_EVENTS
                           for count in metrics["class_support"].values())):
                raise ValueError("Validation must contain exactly 500,000 events in each class")
            for field in ("accuracy", "macro_auc_ovo", "macro_auc_ovr"):
                _finite(metrics[field], field, minimum=0, maximum=1)
            rejections = metrics["qcd_rejection"]
            if set(rejections) != set(REJECTION_TARGETS):
                raise ValueError("Validation rejection classes differ from the fixed metric definition")
            for signal, targets in REJECTION_TARGETS.items():
                if set(rejections[signal]) != set(targets):
                    raise ValueError(f"Missing or unexpected rejection efficiency for {signal}")
                for target in targets:
                    row = rejections[signal][target]
                    efficiency = float(target)
                    if _finite(row["target_signal_efficiency"], "target efficiency") != efficiency:
                        raise ValueError("Rejection target efficiency differs")
                    for field in ("signal_total", "qcd_total"):
                        if type(row[field]) is not int or row[field] != CLASS_EVENTS:
                            raise ValueError("Rejection must use the full validation class populations")
                    _count(row["signal_accepted"], "signal acceptance count", maximum=CLASS_EVENTS)
                    _count(row["qcd_accepted"], "QCD acceptance count", maximum=CLASS_EVENTS)
                    if row["signal_accepted"] < math.ceil(efficiency * CLASS_EVENTS):
                        raise ValueError("Signal acceptance is below the target efficiency")
                    achieved = _finite(row["achieved_signal_efficiency"], "achieved efficiency", minimum=0, maximum=1)
                    if not math.isclose(achieved, row["signal_accepted"] / CLASS_EVENTS, abs_tol=1e-12):
                        raise ValueError("Achieved efficiency disagrees with passing counts")
                    _finite(row["threshold_log_odds"], "log-odds threshold")
                    if row["qcd_accepted"] == 0:
                        if row["rejection"] is not None:
                            raise ValueError("Zero QCD acceptance requires null empirical rejection")
                    else:
                        rejection = _finite(row["rejection"], "rejection", minimum=1)
                        if not math.isclose(rejection, CLASS_EVENTS / row["qcd_accepted"], rel_tol=1e-12):
                            raise ValueError("Rejection disagrees with QCD passing counts")
            by_task[key] = record
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete or malformed validation metrics: {exc}") from exc
    for field in ("evaluation_sha256", "inventory_sha256"):
        if len({record[field] for record in records}) != 1:
            raise ValueError(f"Validation records have mixed {field}")
    return by_task


def _summary(values):
    finite = all(value is not None for value in values.values())
    return {"per_seed": values, "mean": statistics.mean(values.values()) if finite else None,
            "seed_std": statistics.stdev(values.values()) if finite else None}


def build_report(records):
    """Aggregate complete, identically evaluated matched checkpoints, never seeds selectively."""
    by_task = _validated_records(records)
    ordered = [by_task[architecture, seed] for architecture in ARCHITECTURES for seed in SEEDS]
    models = {}
    for architecture in ARCHITECTURES:
        rows = {str(seed): by_task[architecture, seed]["metrics"] for seed in SEEDS}
        model = {field: _summary({seed: row[field] for seed, row in rows.items()})
                 for field in ("accuracy", "macro_auc_ovo", "macro_auc_ovr")}
        model["seed_count"] = len(SEEDS)
        model["qcd_rejection"] = {}
        for signal, targets in REJECTION_TARGETS.items():
            model["qcd_rejection"][signal] = {}
            for target in targets:
                selected = {seed: row["qcd_rejection"][signal][target] for seed, row in rows.items()}
                summary = _summary({seed: row["rejection"] for seed, row in selected.items()})
                summary["zero_background_seeds"] = sum(row["qcd_accepted"] == 0 for row in selected.values())
                summary["passing_counts"] = {seed: {field: row[field] for field in (
                    "signal_accepted", "signal_total", "qcd_accepted", "qcd_total")}
                    for seed, row in selected.items()}
                model["qcd_rejection"][signal][target] = summary
        models[architecture] = model

    comparisons = {}
    for field in ("accuracy", "macro_auc_ovo", "macro_auc_ovr"):
        comparisons[f"{field}_gain_percentage_points"] = _summary({
            str(seed): 100 * (by_task["part_ev", seed]["metrics"][field]
                             - by_task["part", seed]["metrics"][field]) for seed in SEEDS})
    comparisons["qcd_rejection"] = {}
    for signal, targets in REJECTION_TARGETS.items():
        comparisons["qcd_rejection"][signal] = {}
        for target in targets:
            baseline = models["part"]["qcd_rejection"][signal][target]
            ev = models["part_ev"]["qcd_rejection"][signal][target]
            ratios = {seed: (ev["per_seed"][seed] / value
                            if value is not None and ev["per_seed"][seed] is not None else None)
                      for seed, value in baseline["per_seed"].items()}
            gains = {seed: 100 * (ratio - 1) if ratio is not None else None for seed, ratio in ratios.items()}
            gain_summary = _summary(gains)
            ratio = ev["mean"] / baseline["mean"] if baseline["mean"] is not None and ev["mean"] is not None else None
            comparisons["qcd_rejection"][signal][target] = {
                "per_seed_ratio": ratios, "per_seed_gain_percent": gains,
                "mean_paired_gain_percent": gain_summary["mean"],
                "paired_gain_seed_std_percent": gain_summary["seed_std"],
                "ratio_of_means": ratio,
                "ratio_of_means_gain_percent": 100 * (ratio - 1) if ratio is not None else None,
            }
    return {"schema": "part_fullscale_validation_report_v1", "split": "validation",
            "events_per_checkpoint": EVENTS, "class_events": CLASS_EVENTS, "seeds": list(SEEDS),
            "evaluation_sha256": ordered[0]["evaluation_sha256"],
            "inventory_sha256": ordered[0]["inventory_sha256"],
            "models": models, "comparisons": comparisons, "per_seed_results": ordered}


def markdown_report(report):
    """Render validation-only results, keeping percent gains distinct from pp."""
    models, comparisons = report["models"], report["comparisons"]
    lines = ["# Full 5M validation evaluation: ParT vs ParT-EV", "",
             "These are **validation results, not final-test results**. Each checkpoint was selected "
             "using training-time validation accuracy. This evaluation uses the full 5M validation "
             "population (500,000 jets per class) and does not select new checkpoints.", "",
             "Values summarize the three matched training seeds; standard deviations describe seed "
             "variation, not confidence intervals. No statistical significance is inferred here.", "",
             "| Model | Accuracy (%) | Seed std (pp) | Macro AUC (OVO) | Seed std | Macro AUC (OVR) | Seed std |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for architecture, label in (("part", "ParT"), ("part_ev", "ParT-EV")):
        row = models[architecture]
        lines.append(f"| {label} | {100*row['accuracy']['mean']:.4f} | {100*row['accuracy']['seed_std']:.4f} | "
                     f"{row['macro_auc_ovo']['mean']:.6f} | {row['macro_auc_ovo']['seed_std']:.6f} | "
                     f"{row['macro_auc_ovr']['mean']:.6f} | {row['macro_auc_ovr']['seed_std']:.6f} |")
    accuracy = comparisons["accuracy_gain_percentage_points"]
    lines.extend(["", f"Mean matched accuracy gain: **{accuracy['mean']:+.4f} percentage points**.", "",
                  "| Seed | ParT accuracy (%) | ParT-EV accuracy (%) | Matched gain (pp) |", "|---|---:|---:|---:|"])
    for seed in map(str, SEEDS):
        lines.append(f"| {seed} | {100*models['part']['accuracy']['per_seed'][seed]:.4f} | "
                     f"{100*models['part_ev']['accuracy']['per_seed'][seed]:.4f} | {accuracy['per_seed'][seed]:+.4f} |")
    lines.extend(["", "Rejection uses signal-vs-QCD log odds, `logit(signal) - logit(QCD)`, equivalent "
                  "to ranking by `p(signal)/(p(signal)+p(QCD))`. A threshold retaining at least the "
                  "target signal fraction defines `rejection = QCD total / QCD accepted`.", "",
                  "Table gains below are **100 × (mean ParT-EV rejection / mean ParT rejection − 1)**, "
                  "not the mean of matched-seed percentage gains. Both are recorded separately."])

    def shown(row):
        return f"ZB {row['zero_background_seeds']}/3" if row["mean"] is None else f"{row['mean']:,.2f}"

    groups = [("30% signal efficiency", [(name, "0.3") for name in CLASS_NAMES[1:] if "0.3" in REJECTION_TARGETS[name]]),
              ("50% signal efficiency", [(name, "0.5") for name in CLASS_NAMES[1:] if "0.5" in REJECTION_TARGETS[name]]),
              ("Leptonic-class operating points", [("Hqql", "0.99"), ("Tbl", "0.995")])]
    for title, points in groups:
        lines.extend(["", f"## {title}", "", "| Signal | Signal efficiency | ParT rejection | ParT-EV rejection | Gain (%) |",
                      "|---|---:|---:|---:|---:|"])
        for signal, target in points:
            baseline = models["part"]["qcd_rejection"][signal][target]
            ev = models["part_ev"]["qcd_rejection"][signal][target]
            gain = comparisons["qcd_rejection"][signal][target]["ratio_of_means_gain_percent"]
            lines.append(f"| {signal} | {100*float(target):g}% | {shown(baseline)} | {shown(ev)} | "
                         f"{f'{gain:+.2f}' if gain is not None else 'undefined'} |")

    lines.extend(["", "## Matched-seed rejection gains", "",
                  "| Signal | Signal efficiency | Seed 101 (%) | Seed 202 (%) | Seed 303 (%) | Mean paired gain (%) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for signal, targets in REJECTION_TARGETS.items():
        for target in targets:
            row = comparisons["qcd_rejection"][signal][target]
            values = [row["per_seed_gain_percent"][str(seed)] for seed in SEEDS] + [row["mean_paired_gain_percent"]]
            lines.append(f"| {signal} | {100*float(target):g}% | "
                         + " | ".join("undefined" if value is None else f"{value:+.2f}" for value in values) + " |")
    lines.extend(["", "## QCD passing counts", "",
                  "Each denominator is 500,000 QCD jets. Counts below follow seeds 101 / 202 / 303; "
                  "signal passing counts and thresholds are retained in the accompanying JSON.", "",
                  "| Signal | Signal efficiency | ParT QCD accepted | ParT-EV QCD accepted |", "|---|---:|---:|---:|"])
    for signal, targets in REJECTION_TARGETS.items():
        for target in targets:
            counts = [" / ".join(str(models[architecture]["qcd_rejection"][signal][target]
                                     ["passing_counts"][str(seed)]["qcd_accepted"]) for seed in SEEDS)
                      for architecture in ARCHITECTURES]
            lines.append(f"| {signal} | {100*float(target):g}% | {' | '.join(counts)} |")
    lines.extend(["", "ZB n/3 means n seeds accepted zero QCD events: no finite empirical rejection "
                  "or finite three-seed aggregate is reported. Such seeds are never dropped from "
                  "averages; associated aggregate and paired gains remain undefined.", ""])
    return "\n".join(lines)
