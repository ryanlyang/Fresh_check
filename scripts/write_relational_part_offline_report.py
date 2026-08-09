#!/usr/bin/env python3
"""Aggregate the sealed offline transfer evaluations and paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part.contracts import load_hashed_json, sha256_file, validate_content_hash, with_content_hash, write_immutable_json  # noqa: E402
from teacher_logit_reco.relational_part.evaluation import paired_prediction_statistics  # noqa: E402
from teacher_logit_reco.relational_part.offline_transfer import (  # noqa: E402
    OFFLINE_TRANSFER_MODEL_SPECS,
    OFFLINE_TRANSFER_REPORT_CONTRACT,
    OFFLINE_TRANSFER_SEEDS,
)


def _load_predictions(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"][0]))
        validate_content_hash(
            metadata,
            expected_contract="relational_part_offline_final_predictions_v1",
        )
        return {
            "logits": payload["logits"].copy(),
            "labels": payload["labels"].copy(),
            "predictions": payload["predictions"].copy(),
            "event_identities": payload["event_identities"].copy(),
            "metadata": metadata,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    lock = load_hashed_json(root / "selection" / "locked_finalists.json")
    evaluations = {}
    predictions = {}
    for run_id in OFFLINE_TRANSFER_MODEL_SPECS:
        for seed in OFFLINE_TRANSFER_SEEDS:
            task_root = root / "final_test" / run_id / f"seed_{seed}"
            result = load_hashed_json(task_root / "metrics.json")
            if result["lock_sha256"] != lock["content_hash"]:
                raise ValueError("offline final result belongs to another lock")
            prediction_path = task_root / "predictions.npz"
            if sha256_file(prediction_path) != result["prediction_file_sha256"]:
                raise ValueError("offline prediction bytes differ from result")
            prediction = _load_predictions(prediction_path)
            if prediction["metadata"] != result["prediction_metadata"]:
                raise ValueError("offline prediction metadata differs from result")
            evaluations[(run_id, seed)] = result
            predictions[(run_id, seed)] = prediction
    paired = {}
    for run_id in OFFLINE_TRANSFER_MODEL_SPECS:
        if run_id == "OFF_RPT_BASE":
            continue
        paired[run_id] = {}
        for seed in OFFLINE_TRANSFER_SEEDS:
            paired[run_id][str(seed)] = paired_prediction_statistics(
                predictions[(run_id, seed)],
                predictions[("OFF_RPT_BASE", seed)],
            )
    model_rows = {}
    baseline_mean = statistics.mean(
        evaluations[("OFF_RPT_BASE", seed)]["metrics"]["accuracy"]
        for seed in OFFLINE_TRANSFER_SEEDS
    )
    for run_id in OFFLINE_TRANSFER_MODEL_SPECS:
        rows = [evaluations[(run_id, seed)]["metrics"] for seed in OFFLINE_TRANSFER_SEEDS]
        accuracies = [float(row["accuracy"]) for row in rows]
        rejection = {}
        for signal in rows[0]["qcd_signal_rejection"]:
            rejection[signal] = {}
            for target in ("0.3", "0.5"):
                values = [row["qcd_signal_rejection"][signal][target]["background_rejection"] for row in rows]
                rejection[signal][target] = {
                    "per_seed": values,
                    "mean_finite_background_rejection": (
                        None if any(value is None for value in values)
                        else statistics.mean(float(value) for value in values)
                    ),
                }
        model_rows[run_id] = {
            "mean_accuracy": statistics.mean(accuracies),
            "sample_standard_deviation": statistics.stdev(accuracies),
            "mean_accuracy_difference_vs_off_rpt_base": statistics.mean(accuracies) - baseline_mean,
            "per_seed_accuracy": {str(seed): accuracy for seed, accuracy in zip(OFFLINE_TRANSFER_SEEDS, accuracies)},
            "qcd_signal_rejection": rejection,
        }
    report = with_content_hash(
        {
            "contract": OFFLINE_TRANSFER_REPORT_CONTRACT,
            "schema_version": 1,
            "campaign_sha256": campaign["content_hash"],
            "lock_sha256": lock["content_hash"],
            "input_view": "offline",
            "models": model_rows,
            "paired_statistics_vs_off_rpt_base": paired,
            "primary_hypothesis": campaign["hypotheses"]["primary"],
            "performance_gate": False,
            "all_twelve_predeclared_final_tasks_reported": True,
            "interpretation": campaign["scientific_interpretation"],
        }
    )
    write_immutable_json(root / "reports" / "offline_transfer_report.json", report)
    lines = [
        "# Offline Relational Particle Transformer Transfer",
        "",
        "All four predeclared models and all three seeds were evaluated on the matched offline final-test identities. No performance gate was applied.",
        "",
        "| Model | Mean accuracy | Difference vs OFF_RPT_BASE | Seed std. dev. |",
        "|---|---:|---:|---:|",
    ]
    for run_id, row in model_rows.items():
        lines.append(
            f"| {run_id} | {100*row['mean_accuracy']:.4f}% | {100*row['mean_accuracy_difference_vs_off_rpt_base']:+.4f} pp | {100*row['sample_standard_deviation']:.4f} pp |"
        )
    lines.extend(
        [
            "",
            "This is a predeclared offline-domain replication of the HLT architecture result. It should not be described as a globally untouched offline test because the same repository split may have been used by earlier offline experiments.",
            "",
        ]
    )
    destination = root / "reports" / "offline_transfer_report.md"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != "\n".join(lines):
            raise FileExistsError("offline Markdown report already differs")
    else:
        destination.write_text("\n".join(lines), encoding="utf-8")
    print(report["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
