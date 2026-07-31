#!/usr/bin/env python3
"""Derive production resource evidence only from a completed real miniature."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    RESOURCE_MEASUREMENT_EVIDENCE_CONTRACT,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    SLURM_SUBMISSION_LEDGER_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.orchestration import (  # noqa: E402
    RESOURCE_CLASSES,
)


PRODUCTION_COUNTS = {
    "model_train": 500_000,
    "scale_train": 3_000_000,
    "val_stop": 50_000,
    "design_select": 25_000,
    "design_confirm": 25_000,
    "final_select": 50_000,
    "final_test": 300_000,
}
TRAINING_NODE_FRAGMENTS = (
    "train",
    "beam",
    "auxiliary",
    "feedback",
    "combination",
    "capacity",
)
TARGET_NODE_FRAGMENTS = (
    "target",
    "tree",
    "normalization",
    "hlt_cache",
    "teacher_target",
)


def _latest_jobs(root: Path, plan, source) -> tuple[dict[str, str], list[str]]:
    jobs: dict[str, str] = {}
    hashes = []
    for path in sorted(
        (root / "job_ledgers").glob("slurm_submission_attempt_*.json")
    ):
        ledger = load_hashed_json(
            path, expected_contract=SLURM_SUBMISSION_LEDGER_CONTRACT
        )
        if (
            ledger.get("source") != source
            or ledger.get("execution_plan_sha256") != plan["content_hash"]
        ):
            raise ValueError("resource submission ledger lineage differs")
        if ledger["submission_mode"] == "dry_run":
            continue
        jobs.update(
            {
                str(node_id): str(job_id)
                for node_id, job_id in ledger["jobs"].items()
            }
        )
        hashes.append(ledger["content_hash"])
    expected = {row["node_id"] for row in plan["nodes"]}
    if set(jobs) != expected:
        raise ValueError("resource evidence lacks latest job coverage")
    return jobs, hashes


def _sacct(job_ids: list[str], *, usage: bool) -> list[dict[str, str]]:
    fields = (
        "JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,"
        "AllocTRES,Start,End"
    )
    command = ["sacct"]
    if not usage:
        command.append("-X")
    command.extend(
        ["-n", "-P", "-j", ",".join(job_ids), "-o", fields]
    )
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True
    )
    keys = (
        "job_id_raw",
        "state",
        "exit_code",
        "elapsed_seconds",
        "allocated_cpus",
        "maximum_rss",
        "allocated_tres",
        "start",
        "end",
    )
    rows = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        values = raw.split("|")
        if len(values) < len(keys):
            raise ValueError("resource sacct row is malformed")
        rows.append(dict(zip(keys, values[: len(keys)])))
    return rows


def _allocation_rows(rows, job_id: str) -> list[dict[str, str]]:
    tasks = [
        row
        for row in rows
        if row["job_id_raw"].startswith(job_id + "_")
        and "." not in row["job_id_raw"]
    ]
    selected = tasks or [
        row for row in rows if row["job_id_raw"] == job_id
    ]
    if not selected or any(
        not row["state"].rstrip("+").startswith("COMPLETED")
        or not row["exit_code"].startswith("0:")
        or int(row["elapsed_seconds"]) <= 0
        for row in selected
    ):
        raise ValueError(f"resource job is not completely measured: {job_id}")
    return selected


def _bytes(value: str) -> int:
    text = str(value).strip().upper()
    if not text:
        return 0
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", text)
    if match is None:
        raise ValueError(f"unrecognized Slurm memory value: {value}")
    scale = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }[match.group(2)]
    return int(float(match.group(1)) * scale)


def _gpu_count(tres: str) -> int:
    matches = re.findall(r"(?:^|,)gres/gpu[^=]*=([0-9]+)", tres)
    return sum(int(value) for value in matches)


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _slurm_time(seconds: float) -> str:
    total = max(60, int(math.ceil(float(seconds) / 60.0) * 60))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
        if days
        else f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    )


def _slurm_seconds(value: str) -> int:
    text = str(value)
    if "-" in text:
        days_raw, text = text.split("-", 1)
        days = int(days_raw)
    else:
        days = 0
    hours, minutes, seconds = (int(piece) for piece in text.split(":"))
    return days * 86_400 + hours * 3_600 + minutes * 60 + seconds


def _event_ratio(campaign, node_id: str) -> float:
    roles = campaign["split_roles"]
    role = (
        "scale_train"
        if node_id.startswith("scale_")
        else "final_select"
        if "stack" in node_id
        else "final_test"
        if node_id in {"final_input_preparation", "final_test"}
        else "model_train"
    )
    observed = int(roles[role])
    if observed <= 0:
        raise ValueError(f"miniature split count is nonpositive: {role}")
    return max(1.0, PRODUCTION_COUNTS[role] / observed)


def _artifact_bytes(root: Path) -> tuple[int, int]:
    checkpoints = [
        path
        for path in root.rglob("best_model_val.pt")
        if path.is_file() and not path.is_symlink()
    ]
    exports = [
        path
        for path in root.rglob("*.pt")
        if path.is_file()
        and not path.is_symlink()
        and (
            path.parent.name == "exports"
            or path.name.startswith("deployable")
        )
    ]
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoints)
    export_bytes = sum(path.stat().st_size for path in exports)
    if checkpoint_bytes <= 0 or export_bytes <= 0:
        raise ValueError("miniature checkpoint/export byte evidence is absent")
    return checkpoint_bytes, export_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--scheduler-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    if campaign["campaign_profile"] != "miniature_test":
        raise ValueError("resource evidence requires a miniature campaign")
    plan = load_hashed_json(
        root / "job_ledgers" / "production_execution_plan.json",
        expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT,
    )
    scheduler = load_hashed_json(args.scheduler_evidence)
    if (
        scheduler.get("contract")
        != "hosd_miniature_scheduler_evidence_v1"
        or scheduler.get("source") != campaign["source"]
        or scheduler.get("execution_plan_sha256") != plan["content_hash"]
    ):
        raise ValueError("resource scheduler evidence differs")
    jobs, ledger_hashes = _latest_jobs(root, plan, campaign["source"])
    allocations = _sacct(list(jobs.values()), usage=False)
    usage = _sacct(list(jobs.values()), usage=True)
    by_id = {row["node_id"]: row for row in plan["nodes"]}
    measured_nodes = {}
    intervals = []
    gpu_hours_by_stage: dict[str, float] = {}
    projected_gpu_hours_by_stage: dict[str, float] = {}
    projected_target_seconds = 0.0
    max_projected_seconds = {"cpu": 0.0, "gpu": 0.0}
    max_rss = {"cpu": 0, "gpu": 0}
    for node_id, job_id in sorted(jobs.items()):
        node = by_id[node_id]
        rows = _allocation_rows(allocations, job_id)
        resource = str(node["resource"])
        event_ratio = _event_ratio(campaign, node_id)
        epoch_ratio = (
            20.0
            if resource == "gpu"
            and any(value in node_id for value in TRAINING_NODE_FRAGMENTS)
            else 1.0
        )
        elapsed = sum(int(row["elapsed_seconds"]) for row in rows)
        projected_seconds = elapsed * event_ratio * epoch_ratio
        max_projected_seconds[resource] = max(
            max_projected_seconds[resource], projected_seconds
        )
        if resource == "cpu" and any(
            value in node_id for value in TARGET_NODE_FRAGMENTS
        ):
            projected_target_seconds += elapsed * event_ratio
        gpu_hours = sum(
            int(row["elapsed_seconds"])
            * _gpu_count(row["allocated_tres"])
            / 3600.0
            for row in rows
        )
        if resource == "gpu" and gpu_hours <= 0:
            raise ValueError(f"GPU node lacks allocated GPU evidence: {node_id}")
        if gpu_hours:
            stage = str(node["stage"])
            gpu_hours_by_stage[stage] = (
                gpu_hours_by_stage.get(stage, 0.0) + gpu_hours
            )
            projected_gpu_hours_by_stage[stage] = (
                projected_gpu_hours_by_stage.get(stage, 0.0)
                + gpu_hours * event_ratio * epoch_ratio
            )
        for row in rows:
            if not row["start"] or not row["end"]:
                raise ValueError("resource concurrency timestamps are absent")
            intervals.append((_time(row["start"]), 1))
            intervals.append((_time(row["end"]), -1))
            allocation_id = row["job_id_raw"]
            observed_rss = max(
                (
                    _bytes(value["maximum_rss"])
                    for value in usage
                    if value["job_id_raw"] == allocation_id
                    or value["job_id_raw"].startswith(allocation_id + ".")
                ),
                default=0,
            )
            max_rss[resource] = max(max_rss[resource], observed_rss)
        measured_nodes[node_id] = {
            "job_id": job_id,
            "resource": resource,
            "stage": node["stage"],
            "task_count": len(rows),
            "elapsed_task_seconds": elapsed,
            "event_scale_ratio": event_ratio,
            "epoch_scale_ratio": epoch_ratio,
            "projected_max_task_seconds": projected_seconds,
            "measured_gpu_hours": gpu_hours,
        }
    if not all(max_rss.values()):
        raise ValueError("miniature MaxRSS evidence is incomplete")
    active = maximum = 0
    for _, delta in sorted(intervals, key=lambda row: (row[0], row[1])):
        active += delta
        maximum = max(maximum, active)
    checkpoint_bytes, export_bytes = _artifact_bytes(root)
    requests = {}
    for resource, base in RESOURCE_CLASSES.items():
        memory_gib = max(
            int(str(base["memory"]).removesuffix("G")),
            int(math.ceil(max_rss[resource] * 1.5 / 1024**3)),
        )
        request = dict(base)
        request["memory"] = f"{memory_gib}G"
        request["time"] = str(base["time"])
        requests[resource] = request
    raw_digest = hashlib.sha256(
        json.dumps(
            {"allocations": allocations, "usage": usage},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    artifact = with_content_hash(
        {
            "contract": RESOURCE_MEASUREMENT_EVIDENCE_CONTRACT,
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "miniature_execution_plan_sha256": plan["content_hash"],
            "scheduler_evidence_sha256": scheduler["content_hash"],
            "submission_ledger_hashes": ledger_hashes,
            "sacct_rows_sha256": raw_digest,
            "measured_nodes": measured_nodes,
            "requests_by_class": requests,
            "projected_target_extraction_seconds": max(
                1, int(math.ceil(projected_target_seconds))
            ),
            "measured_gpu_hours_by_stage": gpu_hours_by_stage,
            "projected_gpu_hours_by_stage": projected_gpu_hours_by_stage,
            "maximum_concurrent_jobs": maximum,
            "checkpoint_bytes": checkpoint_bytes,
            "export_bytes": export_bytes,
            "maximum_rss_bytes_by_class": max_rss,
            "projection_population_counts": PRODUCTION_COUNTS,
            "training_epoch_scale_ratio": 20,
            "request_memory_safety_factor": 1.5,
            "request_time_policy": (
                "registered_stage_class_limit; tiny-miniature linear time "
                "projection is reported but is not a valid walltime estimator"
            ),
            "all_values_derived_without_scientific_metrics": True,
            "hand_authored_measurements": False,
        }
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
                "maximum_concurrent_jobs": maximum,
                "requests_by_class": requests,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
