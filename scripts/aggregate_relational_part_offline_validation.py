#!/usr/bin/env python3
"""Summarize validation and lock every predeclared offline final-test task."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part.contracts import load_hashed_json, sha256_file, with_content_hash, write_immutable_json  # noqa: E402
from teacher_logit_reco.relational_part.offline_transfer import (  # noqa: E402
    OFFLINE_TRANSFER_FINAL_LOCK_CONTRACT,
    OFFLINE_TRANSFER_FINAL_TASK_REGISTRY_CONTRACT,
    OFFLINE_TRANSFER_MODEL_SPECS,
    OFFLINE_TRANSFER_SEEDS,
    OFFLINE_TRANSFER_TASK_REGISTRY_CONTRACT,
    OFFLINE_TRANSFER_VALIDATION_SUMMARY_CONTRACT,
    validate_offline_task_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    binding = load_hashed_json(root / "inputs" / "offline_cache_binding.json")
    tasks = load_hashed_json(root / "registry" / "training_tasks.json", expected_contract=OFFLINE_TRANSFER_TASK_REGISTRY_CONTRACT)
    validate_offline_task_registry(tasks)
    rows = []
    by_run = {run_id: [] for run_id in OFFLINE_TRANSFER_MODEL_SPECS}
    for task in tasks["tasks"]:
        run_id, seed = task["run_id"], int(task["seed"])
        run_root = root / "runs" / run_id / f"seed_{seed}"
        registration = load_hashed_json(run_root / "checkpoint_registration.json")
        metrics = load_hashed_json(run_root / "val_select_metrics.json")
        if registration["val_select_metrics_sha256"] != metrics["content_hash"]:
            raise ValueError("validation metrics differ from checkpoint registration")
        row = {
            "run_id": run_id,
            "seed": seed,
            "accuracy": float(metrics["accuracy"]),
            "cross_entropy": float(metrics["cross_entropy"]),
            "macro_per_class_accuracy": float(metrics["macro_per_class_accuracy"]),
            "checkpoint_sha256": registration["checkpoint_sha256"],
            "checkpoint_registration_sha256": registration["content_hash"],
            "model_contract_sha256": task["model_contract_sha256"],
        }
        rows.append(row)
        by_run[run_id].append(row)
    baseline = statistics.mean(row["accuracy"] for row in by_run["OFF_RPT_BASE"])
    summaries = {}
    for run_id, run_rows in by_run.items():
        accuracies = [row["accuracy"] for row in run_rows]
        summaries[run_id] = {
            "mean_accuracy": statistics.mean(accuracies),
            "sample_standard_deviation": statistics.stdev(accuracies),
            "mean_accuracy_difference_vs_off_rpt_base": statistics.mean(accuracies) - baseline,
            "seed_count": len(run_rows),
        }
    summary = with_content_hash(
        {
            "contract": OFFLINE_TRANSFER_VALIDATION_SUMMARY_CONTRACT,
            "schema_version": 1,
            "campaign_sha256": campaign["content_hash"],
            "task_registry_sha256": tasks["content_hash"],
            "split": "stack_val",
            "rows": rows,
            "models": summaries,
            "used_for_checkpoint_selection": False,
            "used_to_remove_final_tasks": False,
            "performance_gate": False,
        }
    )
    write_immutable_json(root / "selection" / "validation_summary.json", summary)
    locked_rows = []
    final_tasks = []
    for row in rows:
        run_id, seed = row["run_id"], row["seed"]
        checkpoint = root / "runs" / run_id / f"seed_{seed}" / "best_model_val.pt"
        if sha256_file(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("checkpoint bytes differ before final lock")
        locked = {
            **row,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_registration_path": str((checkpoint.parent / "checkpoint_registration.json").resolve()),
        }
        locked_rows.append(locked)
        final_tasks.append(
            {
                "task_index": len(final_tasks),
                "run_id": run_id,
                "seed": seed,
                "relation_families": list(OFFLINE_TRANSFER_MODEL_SPECS[run_id]["relation_families"]),
                "output_dir": f"final_test/{run_id}/seed_{seed}",
            }
        )
    lock = with_content_hash(
        {
            "contract": OFFLINE_TRANSFER_FINAL_LOCK_CONTRACT,
            "schema_version": 1,
            "campaign_sha256": campaign["content_hash"],
            "validation_summary_sha256": summary["content_hash"],
            "offline_cache_binding_sha256": binding["content_hash"],
            "final_test_offline_content_sha256": binding["splits"]["final_test"]["offline_content_sha256"],
            "locked_rows": locked_rows,
            "all_predeclared_tasks_locked": True,
            "performance_gate": False,
            "final_test_opened": False,
        }
    )
    write_immutable_json(root / "selection" / "locked_finalists.json", lock)
    final_registry = with_content_hash(
        {
            "contract": OFFLINE_TRANSFER_FINAL_TASK_REGISTRY_CONTRACT,
            "schema_version": 1,
            "campaign_sha256": campaign["content_hash"],
            "lock_sha256": lock["content_hash"],
            "task_count": len(final_tasks),
            "tasks": final_tasks,
            "performance_gate": False,
        }
    )
    write_immutable_json(root / "final_test" / "task_registry.json", final_registry)
    print(lock["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
