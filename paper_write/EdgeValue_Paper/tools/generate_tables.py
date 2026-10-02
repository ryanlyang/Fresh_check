#!/usr/bin/env python3
"""Generate the paper's eight-configuration tables (stdlib only).

Run from any directory:
  python paper_write/EdgeValue_Paper/tools/generate_tables.py
  python paper_write/EdgeValue_Paper/tools/generate_tables.py --check

Override --source-root when the read-only evaluation archive is elsewhere.
No model is evaluated, no source archive is modified, and absent configurations
remain pending. --check validates source/metric invariants and compares every
generated artifact byte-for-byte without writing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    "C:/Users/22rya/ComputerScience/CERN/a_download_checkpoints/"
    "rpt_offline_test20m_20260925T235646Z"
)
SEEDS = (101, 202, 303)
CLASSES = ("QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl")
SIGNALS = CLASSES[1:]
TARGETS = ("0.3", "0.5", "0.99", "0.995")
HADRONIC_SIGNALS = tuple(cls for cls in SIGNALS if cls not in ("Hqql", "Tbl"))
BENCHMARK_TARGETS = {
    cls: "0.99" if cls == "Hqql" else "0.995" if cls == "Tbl" else "0.5"
    for cls in SIGNALS
}
DISPLAY_TABLES = ("benchmark", "0.3")
EVENT_COUNT = 20_000_000
CLASS_SUPPORT = 2_000_000
# The run IDs are identifiers, not a reliable encoding of architectural choices:
# in particular OFF_RPT_BASE_EDGEVALUE has layerwise pair bias.
CONFIGS = (
    ("S-S", "OFF_RPT_BASE", "standard", "shared", False, True),
    ("S-L", "OFF_RPT_BASE_LAYERWISE", "standard", "layerwise", False, False),
    ("S-S+EV", "OFF_RPT_BASE_SHARED_EDGEVALUE", "standard", "shared", True, False),
    ("S-L+EV", "OFF_RPT_BASE_EDGEVALUE", "standard", "layerwise", True, True),
    ("E-S", "OFF_RPT_SELECTED_SHARED", "standard+PT+TRACK+REGION", "shared", False, False),
    ("E-L", "OFF_RPT_SELECTED_LAYERWISE", "standard+PT+TRACK+REGION", "layerwise", False, True),
    ("E-S+EV", "OFF_RPT_SELECTED_SHARED_EDGEVALUE", "standard+PT+TRACK+REGION", "shared", True, False),
    ("E-L+EV", "OFF_RPT_SELECTED_EDGEVALUE", "standard+PT+TRACK+REGION", "layerwise", True, True),
)
CLASS_TEX = {
    "Hbb": r"$H\!\to b\bar b$",
    "Hcc": r"$H\!\to c\bar c$",
    "Hgg": r"$H\!\to gg$",
    "H4q": r"$H\!\to4q$",
    "Hqql": r"$H\!\to q q\ell\nu$",
    "Zqq": r"$Z\!\to q\bar q$",
    "Wqq": r"$W\!\to q\bar q'$",
    "Tbqq": r"$t\!\to b q\bar q'$",
    "Tbl": r"$t\!\to b\ell\nu$",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object, context: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), context)
    require(math.isfinite(value), context)
    return float(value)


def close(actual: float, expected: float, context: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), context)


def validate_content_hash(document: dict, context: str) -> None:
    payload = {key: value for key, value in document.items() if key != "content_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    require(actual == document["content_hash"], f"{context}: content hash mismatch")


def summary(values: list[float]) -> dict:
    require(len(values) == len(SEEDS), "A summary must contain exactly three seeds")
    for value in values:
        finite(value, "Nonfinite summary input")
    return {
        "per_seed": dict(zip(map(str, SEEDS), values)),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def presentation_policy() -> dict:
    return {
        "working_points_apply_identically_to_all_configurations_and_seeds": True,
        "retained_signal_efficiency_targets": list(TARGETS),
        "benchmark": {"signal_order": list(SIGNALS),
                      "signal_efficiency_by_class": dict(BENCHMARK_TARGETS)},
        "supplementary_30_percent": {
            "signal_order": list(HADRONIC_SIGNALS),
            "signal_efficiency_by_class": {cls: "0.3" for cls in HADRONIC_SIGNALS},
            "omitted_from_display_only": ["Hqql", "Tbl"],
            "omission_reason": "All completed configurations have zero accepted QCD events for these classes at 30%; original measurements remain in rejection data",
        },
    }


def read_sources(source_root: Path) -> tuple[dict, dict, dict]:
    documents = []
    provenance = {}
    for name, relative_path in (("report", "reports/full_test_report.json"),
                                ("plan", "evaluation_plan.json")):
        path = source_root / relative_path
        raw = path.read_bytes()
        document = json.loads(raw)
        validate_content_hash(document, relative_path)
        provenance[name] = {
            "archive_relative_path": relative_path,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": document["content_hash"],
        }
        documents.append(document)
    provenance["archive_name"] = source_root.name
    return documents[0], documents[1], provenance


def derive(report: dict, plan: dict, provenance: dict) -> dict:
    expected_runs = {config[1] for config in CONFIGS if config[5]}
    expected_pairs = {(run, seed) for run in expected_runs for seed in SEEDS}
    require(len(CONFIGS) == 8 and len({c[0] for c in CONFIGS}) == 8, "Eight unique IDs required")
    require(report["plan_sha256"] == plan["content_hash"], "Report/plan mismatch")
    require(report["event_count_per_model"] == plan["event_count"] == EVENT_COUNT, "Test size")
    require(report["inference_only"] is True, "Evaluation must be inference-only")
    require(plan["checkpoints_and_normalizers_refitted"] is False, "Refitted evaluation")
    require(plan["input_view"] == "offline", "Only offline evaluation is in scope")
    require(tuple(plan["class_order"]) == CLASSES, "Unexpected class order")
    require(set(report["models"]) == expected_runs, "Source run coverage changed")
    require(all(float(target) in plan["rejection_targets"] for target in TARGETS), "Targets absent")
    file_support = {index: 0 for index in range(len(CLASSES))}
    for source_file in plan["files"]:
        file_support[source_file["label"]] += source_file["entries"]
    require(all(count == CLASS_SUPPORT for count in file_support.values()), "Class support mismatch")
    tasks = {(task["run_id"], task["seed"]): task for task in plan["tasks"]}
    require(len(tasks) == len(plan["tasks"]) == 12 and set(tasks) == expected_pairs, "Plan coverage")
    records = {}
    for record in report["per_model_results"]:
        validate_content_hash(record, "Per-model result")
        task = record["task"]
        key = (task["run_id"], task["seed"])
        require(key not in records, f"Duplicate run/seed: {key}")
        require(key in tasks and task == tasks[key], f"Task/plan mismatch: {key}")
        require(record["plan_sha256"] == plan["content_hash"], "Nested plan hash mismatch")
        require(record["used_for_model_selection"] is False, "Test used for selection")
        metrics = record["metrics"]
        validate_content_hash(metrics, f"Metrics {key}")
        require(metrics["event_count"] == EVENT_COUNT, f"Event count {key}")
        require(tuple(metrics["class_order"]) == CLASSES, f"Class order {key}")
        require(metrics["split"] == "official_test_20M", f"Wrong split {key}")
        require(set(metrics["one_vs_rest_auc"]) == set(CLASSES), f"AUC coverage {key}")
        require(set(metrics["qcd_signal_rejection"]) == set(SIGNALS), f"Signal coverage {key}")
        accuracy = finite(metrics["accuracy"], f"Accuracy {key}")
        require(0 <= accuracy <= 1, f"Accuracy outside unit interval {key}")
        matrix = metrics["confusion_matrix"]
        require(len(matrix) == len(CLASSES), f"Confusion shape {key}")
        require(all(len(row) == len(CLASSES) and sum(row) == CLASS_SUPPORT for row in matrix),
                f"Confusion support {key}")
        close(accuracy, sum(matrix[i][i] for i in range(len(CLASSES))) / EVENT_COUNT,
              f"Confusion/accuracy arithmetic {key}")
        for cls, auc in metrics["one_vs_rest_auc"].items():
            require(0 <= finite(auc, f"AUC {key}/{cls}") <= 1, "AUC outside unit interval")
        for cls in SIGNALS:
            for target in TARGETS:
                point = metrics["qcd_signal_rejection"][cls][target]
                context = f"{key}/{cls}/{target}"
                require(point["qcd_support"] == point["signal_support"] == CLASS_SUPPORT, context)
                require(point["signal_class"] == cls, context)
                close(point["target_signal_efficiency"], float(target), context)
                close(point["achieved_signal_efficiency"],
                      point["signal_pass_count"] / CLASS_SUPPORT, context)
                require(point["discriminant"] == "logit_signal_minus_logit_QCD", context)
                require(point["pass_rule"] == "score_greater_than_or_equal_to_threshold", context)
                count = point["qcd_false_positive_count"]
                require(isinstance(count, int) and 0 <= count <= CLASS_SUPPORT, context)
                close(point["qcd_false_positive_rate"], count / CLASS_SUPPORT, context)
                if count == 0:
                    require(point["background_rejection"] is None
                            and point["background_rejection_is_infinite"] is True, context)
                else:
                    require(point["background_rejection_is_infinite"] is False, context)
                    close(finite(point["background_rejection"], context), CLASS_SUPPORT / count, context)
        records[key] = record
    require(set(records) == expected_pairs and len(records) == 12, "Result seed coverage")

    configurations = []
    for short_id, run_id, inputs, bias, edgevalue, available in CONFIGS:
        result = {
            "id": short_id, "source_run_id": run_id, "pair_inputs": inputs,
            "pair_bias": bias, "edgevalue": edgevalue,
            "status": "evaluated" if available else "pending",
            "accuracy_percent": None, "macro_ovr_auroc_percent": None,
            "accuracy_gain_pp": None, "macro_ovr_auroc_gain_pp": None,
            "rejection": None,
        }
        if available:
            seed_records = [records[(run_id, seed)] for seed in SEEDS]
            seed_metrics = [record["metrics"] for record in seed_records]
            result["seed_provenance"] = {
                str(seed): {"checkpoint_sha256": record["task"]["checkpoint_sha256"],
                            "model_contract_sha256": record["task"]["model_contract_sha256"],
                            "result_content_sha256": record["content_hash"],
                            "metrics_content_sha256": record["metrics"]["content_hash"]}
                for seed, record in zip(SEEDS, seed_records)
            }
            expected_families = [] if inputs == "standard" else ["PT", "TRACK", "REGION"]
            require(all(r["task"]["relation_families"] == expected_families for r in seed_records),
                    f"Relation-family mapping {run_id}")
            result["accuracy_percent"] = summary([100 * m["accuracy"] for m in seed_metrics])
            result["macro_ovr_auroc_percent"] = summary([
                100 * statistics.mean(m["one_vs_rest_auc"][cls] for cls in CLASSES)
                for m in seed_metrics
            ])
            source_summary = report["models"][run_id]
            close(result["accuracy_percent"]["mean"], 100 * source_summary["mean_accuracy"], run_id)
            close(result["accuracy_percent"]["sample_sd"],
                  100 * source_summary["seed_sample_standard_deviation"], run_id)
            for seed, metrics in zip(SEEDS, seed_metrics):
                close(metrics["accuracy"], source_summary["per_seed_accuracy"][str(seed)], run_id)
            result["rejection"] = {}
            for target in TARGETS:
                result["rejection"][target] = {}
                for cls in SIGNALS:
                    points = [m["qcd_signal_rejection"][cls][target] for m in seed_metrics]
                    values = [point["background_rejection"] for point in points]
                    zero_count = sum(point["qcd_false_positive_count"] == 0 for point in points)
                    cell = {
                        "zero_background_seed_count": zero_count,
                        "per_seed_rejection": dict(zip(map(str, SEEDS), values)),
                        "per_seed_qcd_false_positive_count": {
                            str(seed): point["qcd_false_positive_count"]
                            for seed, point in zip(SEEDS, points)
                        },
                        "mean": None, "sample_sd": None, "gain_percent": None,
                    }
                    if zero_count == 0:
                        values_summary = summary(values)
                        cell["mean"] = values_summary["mean"]
                        cell["sample_sd"] = values_summary["sample_sd"]
                        close(cell["mean"], source_summary["qcd_signal_rejection"][cls][target]
                              ["mean_finite_background_rejection"], f"Rejection summary {run_id}")
                    result["rejection"][target][cls] = cell
        configurations.append(result)

    baseline = configurations[0]
    for result in configurations:
        if result["status"] == "pending":
            continue
        for metric, gain in (("accuracy_percent", "accuracy_gain_pp"),
                             ("macro_ovr_auroc_percent", "macro_ovr_auroc_gain_pp")):
            result[gain] = result[metric]["mean"] - baseline[metric]["mean"]
        close(result["accuracy_gain_pp"],
              100 * report["models"][result["source_run_id"]]["difference_vs_baseline"], "Accuracy gain")
        for target in TARGETS:
            for cls in SIGNALS:
                cell, base = result["rejection"][target][cls], baseline["rejection"][target][cls]
                if cell["mean"] is not None and base["mean"] is not None:
                    cell["gain_percent"] = 100 * (cell["mean"] / base["mean"] - 1)

    return {
        "schema_version": 3,
        "provenance": provenance,
        "seeds": list(SEEDS), "class_order": list(CLASSES),
        "event_count_per_seed": EVENT_COUNT, "events_per_class": CLASS_SUPPORT,
        "interpretation": report["interpretation"],
        "aggregation": {
            "uncertainty": "Sample standard deviation across three training seeds (ddof=1), not a confidence interval",
            "accuracy": "100 times per-seed accuracy; arithmetic mean and sample SD across seeds",
            "macro_ovr_auroc": "100 times unweighted mean of all 10 class OvR AUROCs within each seed, then mean and sample SD across seeds",
            "overview_gain": "Difference of configuration and S-S means, in percentage points",
            "rejection": "Arithmetic mean and sample SD of per-seed QCD rejection only when all three seeds have nonzero QCD counts",
            "rejection_gain": "100 times (configuration mean rejection / S-S mean rejection - 1); not mean per-seed percentage gain",
            "zero_background": "ZB k/3: k seeds have zero accepted QCD events; no aggregate finite rejection, SD, or gain is reported",
            "pending": "No result is imputed for the four unevaluated configurations",
        },
        "source_metric_policy": plan["metric_policy"],
        "presentation_policy": presentation_policy(),
        "configurations": configurations,
    }


def display_id(configuration: dict) -> str:
    """Derive manuscript labels from architecture metadata, not historical IDs."""
    inputs = configuration["pair_inputs"]
    require(inputs in ("standard", "standard+PT+TRACK+REGION"), "Unknown pair inputs")
    bias = configuration["pair_bias"]
    require(bias in ("shared", "layerwise"), "Unknown pair bias")
    return (("Std" if inputs == "standard" else "Sel") +
            ("-S" if bias == "shared" else "-L") +
            ("+EV" if configuration["edgevalue"] else ""))


def rejection_column(configuration: dict) -> str:
    """Panel headings already identify pair inputs; spell out the remaining axes."""
    bias = configuration["pair_bias"]
    require(bias in ("shared", "layerwise"), "Unknown pair bias")
    return bias.capitalize() + (" + EV" if configuration["edgevalue"] else "")


def overview_tex(data: dict) -> str:
    lines = [
        "% Generated by tools/generate_tables.py; do not edit by hand.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Offline classification on the official 20M-jet test sample for "
        r"models trained on 1M jets. Values are means $\pm$ sample standard deviations "
        r"over three seeds; $\Delta$ is the difference from baseline \texttt{Std-S} "
        r"in percentage points. \pending\ denotes pending results. Placeholder.}",
        r"\label{tab:overview}", r"\begin{tabular}{lrrrr}", r"\toprule",
        r"ID & Accuracy (\%) & $\Delta$ (pp) & Macro AUROC (\%) & $\Delta$ (pp) \\",
        r"\midrule",
    ]
    for result in data["configurations"]:
        cells = [r"\texttt{" + display_id(result) + "}"]
        if result["status"] == "pending":
            cells += [r"\pending"] * 4
        else:
            for metric, gain in (("accuracy_percent", "accuracy_gain_pp"),
                                 ("macro_ovr_auroc_percent", "macro_ovr_auroc_gain_pp")):
                values = result[metric]
                cells.append(f"${values['mean']:.3f} \\pm {values['sample_sd']:.3f}$")
                cells.append(f"${result[gain]:+.3f}$" if result[gain] else "$0.000$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def rejection_cell(cell: dict) -> str:
    if cell["zero_background_seed_count"]:
        return "ZB " + str(cell["zero_background_seed_count"]) + "/3"
    digits = 0 if cell["mean"] >= 1000 else 1
    return f"${cell['mean']:.{digits}f} \\pm {cell['sample_sd']:.{digits}f}$"


def rejection_tex(data: dict, display: str) -> str:
    require(display in DISPLAY_TABLES, "Unknown rejection table presentation")
    if display == "benchmark":
        signals, targets = SIGNALS, BENCHMARK_TARGETS
        title, label = "QCD rejection at class-specific signal efficiencies", "benchmark"
        caption = (
            r"QCD background rejection on the official 20M-jet test sample at the "
            r"JetClass benchmark signal efficiencies~\cite{Qu2022ParticleTransformer}: "
            r"50\% for the seven hadronic classes, 99\% for $H\!\to q q\ell\nu$, "
            r"and 99.5\% for $t\!\to b\ell\nu$. "
        )
    else:
        signals = HADRONIC_SIGNALS
        targets = {cls: "0.3" for cls in signals}
        title, label = r"QCD rejection at 30\% signal efficiency", "30"
        caption = (
            r"QCD background rejection at 30\% signal efficiency on the official "
            r"20M-jet test sample for the seven hadronic classes. The two leptonic "
            r"classes appear at their benchmark efficiencies in "
            r"Table~\ref{tab:rejection-benchmark}. "
        )
    has_displayed_zero = any(
        result["status"] == "evaluated" and
        result["rejection"][targets[signal]][signal]["zero_background_seed_count"]
        for result in data["configurations"] for signal in signals
    )
    zero_note = (
        r"ZB $k/3$ denotes zero accepted QCD jets in $k$ seeds, for which "
        r"no finite aggregate is reported. " if has_displayed_zero else ""
    )
    lines = [
        "% Generated by tools/generate_tables.py; do not edit by hand.",
        r"\begin{table}[p]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{" + caption + r"Panels compare standard and selected "
        r"pair inputs. Finite values are means $\pm$ sample standard deviations over "
        r"three seeds. " + zero_note +
        r"\pending\ denotes pending results. Placeholder.}",
        r"\label{tab:rejection-" + label + "}",
    ]
    panels = (
        ("(a) Standard pair inputs", data["configurations"][:4]),
        ("(b) Selected pair inputs (standard + PT + TRACK + REGION)",
         data["configurations"][4:]),
    )
    for panel_index, (heading, configurations) in enumerate(panels):
        if panel_index:
            lines.append(r"\par\vspace{1em}")
        lines += [
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lc*{4}{c}@{}}",
            r"\multicolumn{6}{@{}l}{" + title + r"} \\",
            r"\multicolumn{6}{@{}l}{\textbf{" + heading + r"}} \\",
            r"\toprule",
            r"Signal & $\epsilon_s$ (\%) & " + " & ".join(rejection_column(result)
                                     for result in configurations) + r" \\",
            r"\midrule",
        ]
        for cls in signals:
            target = targets[cls]
            cells = [CLASS_TEX[cls], f"{100 * float(target):g}"]
            cells += [r"\pending" if result["status"] == "pending" else
                      rejection_cell(result["rejection"][target][cls])
                      for result in configurations]
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular*}"]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def validate_derived(data: dict) -> None:
    require(data["schema_version"] == 3, "Derived schema version")
    require(data["presentation_policy"] == presentation_policy(), "Rejection presentation policy drift")
    require([c["id"] for c in data["configurations"]] == [c[0] for c in CONFIGS], "ID coverage/order")
    evaluated = [c for c in data["configurations"] if c["status"] == "evaluated"]
    require(len(evaluated) == 4, "Exactly four evaluated configurations required")
    baseline = evaluated[0]
    for config, expected in zip(data["configurations"], CONFIGS):
        short_id, run_id, inputs, bias, edgevalue, available = expected
        require((config["id"], config["source_run_id"], config["pair_inputs"],
                 config["pair_bias"], config["edgevalue"], config["status"]) ==
                (short_id, run_id, inputs, bias, edgevalue,
                 "evaluated" if available else "pending"), "Paper/source configuration mapping")
        if config["status"] == "pending":
            require(all(config[key] is None for key in ("accuracy_percent", "macro_ovr_auroc_percent",
                    "accuracy_gain_pp", "macro_ovr_auroc_gain_pp", "rejection")), "Fabricated pending value")
            continue
        for metric, gain in (("accuracy_percent", "accuracy_gain_pp"),
                             ("macro_ovr_auroc_percent", "macro_ovr_auroc_gain_pp")):
            values = config[metric]
            require(set(values["per_seed"]) == set(map(str, SEEDS)), "Derived seed coverage")
            close(values["mean"], sum(values["per_seed"].values()) / 3, "Mean arithmetic")
            expected_sd = math.sqrt(sum((x - values["mean"]) ** 2
                                        for x in values["per_seed"].values()) / 2)
            close(values["sample_sd"], expected_sd, "Sample SD arithmetic")
            close(config[gain], values["mean"] - baseline[metric]["mean"], "Gain arithmetic")
        require(set(config["rejection"]) == set(TARGETS), "Derived targets")
        for target in TARGETS:
            require(set(config["rejection"][target]) == set(SIGNALS), "Derived signal coverage")
            for cls, cell in config["rejection"][target].items():
                base = baseline["rejection"][target][cls]
                if cell["zero_background_seed_count"]:
                    require(all(cell[key] is None for key in ("mean", "sample_sd", "gain_percent")),
                            "Zero-background result must not have a finite aggregate")
                else:
                    finite(cell["mean"], "Rejection mean")
                    finite(cell["sample_sd"], "Rejection SD")
                    values = list(cell["per_seed_rejection"].values())
                    close(cell["mean"], sum(values) / 3, "Rejection mean arithmetic")
                    close(cell["sample_sd"], math.sqrt(sum((x - cell["mean"]) ** 2 for x in values) / 2),
                          "Rejection sample SD arithmetic")
                    if base["mean"] is not None:
                        close(cell["gain_percent"], 100 * (cell["mean"] / base["mean"] - 1),
                              "Rejection gain arithmetic")
    # JSON must never silently contain NaN or Infinity.
    json.dumps(data, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--check", action="store_true", help="Validate and check generated files without writing")
    args = parser.parse_args()
    report, plan, provenance = read_sources(args.source_root)
    data = derive(report, plan, provenance)
    validate_derived(data)
    outputs = {
        "tables/overview.tex": overview_tex(data),
        "tables/rejection_30.tex": rejection_tex(data, "0.3"),
        # Historical filename retained; the caption and row labels specify the
        # mixed JetClass benchmark efficiencies rather than a uniform 50%.
        "tables/rejection_50.tex": rejection_tex(data, "benchmark"),
        "data/derived_metrics.json": json.dumps(data, indent=2, allow_nan=False) + "\n",
    }
    for relative_path, content in outputs.items():
        path = PAPER_ROOT / relative_path
        if args.check:
            require(path.exists() and path.read_bytes() == content.encode("utf-8"),
                    f"Stale or missing generated artifact: {relative_path}; rerun without --check")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
    print(f"{'Checked' if args.check else 'Generated'} {len(outputs)} artifacts; "
          "8 configurations (4 evaluated, 4 pending), 12 seed results, "
          "9 signals x 4 retained working points; benchmark and 7-signal 30% tables.")


if __name__ == "__main__":
    main()
