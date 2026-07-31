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
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    AuthenticatedTreeSplit,
    RESOURCE_MEASUREMENT_EVIDENCE_CONTRACT,
    iter_authenticated_target_shard_layouts,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    SLURM_SUBMISSION_LEDGER_CONTRACT,
    load_hashed_json,
    canonical_sha256,
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
WALLTIME_SAFETY_FACTOR = 1.5
WALLTIME_ROUNDING_SECONDS = 30 * 60
MEMORY_SAFETY_FACTOR = 1.5
MEMORY_FLOOR_GIB = {"cpu": 4, "gpu": 16}
SCALE_MEMORY_PROJECTION_NODES = (
    "scale_input_prepare",
    "scale_tree_build",
    "scale_target_build",
    "scale_teacher_target_inference",
    "scale_graph_train",
)
PRODUCTION_TREE_SHARD_EVENTS = 10_000
PRODUCTION_TARGET_SHARD_EVENTS = 2_048
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


def _npz_layout_bytes(path: Path) -> int:
    """Read NPY headers only and return decoded array payload bytes."""

    total = 0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as member:
                version = np.lib.format.read_magic(member)
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, dtype = reader(member)
            total += int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    return total


def _maximum_tree_layout_128_255() -> dict[str, int]:
    """Conservative decoded bytes for one 128-leaf/255-node CA tree."""

    leaves = 128
    nodes = 2 * leaves - 1
    components = {
        "identity_and_scalar_headers": 1024,
        "leaf_to_node_and_three_assignments": 4 * leaves * 4,
        "topology_parent_left_right_depth": 4 * nodes * 4,
        "node_four_vectors": nodes * 4 * 4,
        "six_float_node_scalars": 6 * nodes * 4,
        "multiplicity_node_scalar": nodes * 4,
        "python_container_and_array_headers": 16_384,
    }
    return {
        **components,
        "leaves": leaves,
        "maximum_nodes": nodes,
        "total_bytes_per_event": sum(components.values()),
    }


def _scale_layout_ledger(root: Path, source: dict) -> dict:
    """Authenticate resident-unit byte accounting for every risky scale node."""

    input_completion = load_hashed_json(
        root / "scale_up" / "inputs" / "completion.json",
        expected_contract="hosd_scale_input_completion_v4",
    )
    tree_completion = load_hashed_json(
        root / "scale_up" / "trees" / "completion.json",
        expected_contract="hosd_scale_tree_wave_completion_v1",
    )
    target_completion = load_hashed_json(
        root / "scale_up" / "target_completion.json",
        expected_contract="hosd_scale_target_wave_completion_v1",
    )
    teacher_completion = load_hashed_json(
        root / "scale_up" / "teacher_outputs" / "completion.json",
        expected_contract="hosd_scale_teacher_output_completion_v1",
    )
    for name, artifact in (
        ("input", input_completion),
        ("tree", tree_completion),
        ("target", target_completion),
        ("teacher output", teacher_completion),
    ):
        if artifact.get("source") != source:
            raise ValueError(f"scale memory {name} completion source differs")
    if tree_completion.get("scale_input_completion_sha256") != input_completion[
        "content_hash"
    ]:
        raise ValueError("scale memory tree/input completion lineage differs")
    active_tree_resource = load_hashed_json(tree_completion["tree_resource_path"])
    active_tree_backend = load_hashed_json(tree_completion["backend_manifest_path"])
    if (
        active_tree_resource["content_hash"] != tree_completion["tree_resource_sha256"]
        or active_tree_backend["content_hash"]
        != tree_completion["backend_manifest_sha256"]
    ):
        raise ValueError("scale memory active tree parents differ")
    input_paths = [Path(row["npz_path"]) for row in input_completion["rows"]]
    input_completion_rows = {
        str(row["view_id"]): row for row in input_completion["rows"]
    }
    if len(input_paths) != 5 or len(input_completion_rows) != 5:
        raise ValueError("scale memory input completion coverage differs")
    input_rows = []
    for path in input_paths:
        manifest = load_hashed_json(
            path.with_suffix(path.suffix + ".json"),
            expected_contract="hosd_label_blind_input_view_v4",
        )
        completion_row = next(
            (
                row
                for row in input_completion["rows"]
                if Path(row["npz_path"]).resolve() == path.resolve()
            ),
            None,
        )
        if (
            manifest.get("source") != source
            or completion_row is None
            or manifest["content_hash"] != completion_row["view_manifest_sha256"]
            or manifest["npz_sha256"] != completion_row["npz_sha256"]
        ):
            raise ValueError("scale memory input-view source differs")
        store = manifest.get("mmap_store", {})
        if store.get("contract") != "hosd_npy_mmap_store_v2":
            raise ValueError("scale memory input-view storage differs")
        count = int(manifest["identity_count"])
        payload_bytes = sum(
            int(np.prod(row["shape"], dtype=np.int64))
            * int(np.dtype(row["dtype"]).itemsize)
            for row in store["members"].values()
        )
        input_rows.append(
            {
                "coordinate": (
                    "offline"
                    if manifest["replica_id"] is None
                    else f"hlt_replica_{manifest['replica_id']}"
                ),
                "manifest_sha256": manifest["content_hash"],
                "event_count": count,
                "decoded_payload_bytes": payload_bytes,
                "decoded_bytes_per_event": int(math.ceil(payload_bytes / count)),
            }
        )
    max_input_per_event = max(
        row["decoded_bytes_per_event"] for row in input_rows
    )

    tree_rows = []
    for row in tree_completion["rows"]:
        coordinate = str(row["coordinate"])
        input_key = "offline:scale_train" if coordinate == "offline" else (
            f"hlt:scale_train:r{int(coordinate)}"
        )
        input_row = input_completion_rows.get(input_key)
        path = Path(row["tree_manifest_path"])
        if input_row is None:
            raise ValueError("scale memory tree coordinate lacks its active input")
        split = AuthenticatedTreeSplit(
            path.parent,
            expected_parents={
                "hlt_content_sha256": input_row["npz_sha256"],
                "tree_resource_sha256": tree_completion["tree_resource_sha256"],
                "backend_manifest_sha256": tree_completion[
                    "backend_manifest_sha256"
                ],
            },
        )
        if split.manifest["content_hash"] != row["tree_manifest_sha256"]:
            raise ValueError("scale memory tree completion manifest differs")
        for record in split.records:
            verified_path = split.verified_shard_path(record.shard_index)
            tree_rows.append(
                {
                    "manifest_sha256": split.manifest["content_hash"],
                    "npz_sha256": record.npz_sha256,
                    "event_count": record.event_count,
                    "decoded_payload_bytes": _npz_layout_bytes(verified_path),
                }
            )
    if not tree_rows:
        raise ValueError("scale memory tree layout evidence is absent")
    max_tree_shard_events = max(row["event_count"] for row in tree_rows)
    max_tree_shard_bytes = max(row["decoded_payload_bytes"] for row in tree_rows)
    analytical_tree = _maximum_tree_layout_128_255()
    maximum_tree_bytes_per_event = int(
        analytical_tree["total_bytes_per_event"]
    )
    if any(
        int(math.ceil(row["decoded_payload_bytes"] / row["event_count"]))
        > maximum_tree_bytes_per_event
        for row in tree_rows
    ):
        raise ValueError("observed tree layout exceeds analytical 128/255 bound")

    target_rows = []
    active_target_manifests: dict[str, Path] = {}
    for target_id, completion_hash in target_completion[
        "target_completion_hashes"
    ].items():
        coordinate_root = root / "scale_up" / "targets" / target_id
        coordinate_completion = load_hashed_json(
            coordinate_root / "completion.json",
            expected_contract="hosd_scale_target_completion_v1",
        )
        if (
            coordinate_completion["content_hash"] != completion_hash
            or coordinate_completion.get("source") != source
        ):
            raise ValueError("scale memory target completion lineage differs")
        for artifact_name, expected_hash in coordinate_completion[
            "artifact_hashes"
        ].items():
            path = None
            if artifact_name == "canonical_cache":
                path = coordinate_root / "canonical" / "target_manifest.json"
            elif artifact_name.startswith("hlt_cache_"):
                replica = int(artifact_name.removeprefix("hlt_cache_"))
                path = coordinate_root / "hlt" / f"replica_{replica}" / "target_manifest.json"
            elif artifact_name.startswith("hlt_analogue_cache_"):
                replica = int(artifact_name.removeprefix("hlt_analogue_cache_"))
                path = coordinate_root / "hlt" / f"replica_{replica}" / "target_manifest.json"
            elif artifact_name.startswith("residual_cache_"):
                replica = int(artifact_name.removeprefix("residual_cache_"))
                path = coordinate_root / "residual" / f"replica_{replica}" / "target_manifest.json"
            elif artifact_name == "teacher_target_cache":
                teacher_id = (
                    "O_FULLREL"
                    if target_id == "T_OFFLINE_LOGITS_O_FULLREL"
                    else "O_BASE"
                )
                path = root / "scale_up" / "teacher_outputs" / teacher_id / "target_manifest.json"
            if path is None:
                continue
            candidate = load_hashed_json(
                path, expected_contract="hosd_target_cache_manifest_v1"
            )
            if (
                candidate["content_hash"] != expected_hash
                or candidate.get("source") != source
            ):
                raise ValueError("scale memory declared target artifact differs")
            active_target_manifests[candidate["content_hash"]] = path
    for teacher_id, expected_hash in teacher_completion[
        "teacher_output_manifest_hashes"
    ].items():
        path = root / "scale_up" / "teacher_outputs" / teacher_id / "target_manifest.json"
        manifest = load_hashed_json(
            path, expected_contract="hosd_target_cache_manifest_v1"
        )
        if manifest["content_hash"] != expected_hash or manifest.get("source") != source:
            raise ValueError("scale memory teacher-output completion differs")
        active_target_manifests[manifest["content_hash"]] = path
    for expected_hash, path in sorted(active_target_manifests.items()):
        for manifest, record, shard in iter_authenticated_target_shard_layouts(
            path.parent
        ):
            if manifest["content_hash"] != expected_hash or manifest.get("source") != source:
                raise ValueError("scale memory target layout lineage differs")
            target_rows.append(
                {
                    "manifest_sha256": manifest["content_hash"],
                    "npz_sha256": record["npz_sha256"],
                    "event_count": int(record["identity_stop"])
                    - int(record["identity_start"]),
                    "decoded_payload_bytes": _npz_layout_bytes(shard),
                }
            )
    if not target_rows:
        raise ValueError("scale memory target layout evidence is absent")
    max_target_shard_events = max(row["event_count"] for row in target_rows)
    max_target_shard_bytes = max(
        row["decoded_payload_bytes"] for row in target_rows
    )
    maximum_target_bytes_per_event = max(
        int(math.ceil(row["decoded_payload_bytes"] / row["event_count"]))
        for row in target_rows
    )
    production_tree_bytes = (
        maximum_tree_bytes_per_event * PRODUCTION_TREE_SHARD_EVENTS
    )
    production_target_bytes = (
        maximum_target_bytes_per_event * PRODUCTION_TARGET_SHARD_EVENTS
    )
    miniature_byte_accounting = {
        "scale_input_prepare": 3 * max_input_per_event,
        "scale_tree_build": (
            max_input_per_event * max_tree_shard_events
            + 2 * max_tree_shard_bytes
        ),
        "scale_target_build": (
            max_input_per_event * max_target_shard_events
            + max_tree_shard_bytes
            + 2 * max_target_shard_bytes
        ),
        "scale_teacher_target_inference": (
            max_input_per_event * max_target_shard_events
            + max_tree_shard_bytes
            + 2 * max_target_shard_bytes
        ),
        "scale_graph_train": (
            2
            * max_input_per_event
            * max(max_tree_shard_events, max_target_shard_events)
            + max_tree_shard_bytes
            + 2 * max_target_shard_bytes
        ),
    }
    production_byte_accounting = {
        "scale_input_prepare": 3 * max_input_per_event,
        "scale_tree_build": (
            max_input_per_event * PRODUCTION_TREE_SHARD_EVENTS
            + 2 * production_tree_bytes
        ),
        "scale_target_build": (
            max_input_per_event * PRODUCTION_TARGET_SHARD_EVENTS
            + production_tree_bytes
            + maximum_tree_bytes_per_event * PRODUCTION_TARGET_SHARD_EVENTS
            + 2 * production_target_bytes
        ),
        "scale_teacher_target_inference": (
            max_input_per_event * PRODUCTION_TARGET_SHARD_EVENTS
            + production_tree_bytes
            + maximum_tree_bytes_per_event * PRODUCTION_TARGET_SHARD_EVENTS
            + 2 * production_target_bytes
        ),
        "scale_graph_train": (
            2 * max_input_per_event * PRODUCTION_TREE_SHARD_EVENTS
            + 2 * production_tree_bytes
            + 2 * production_target_bytes
        ),
    }
    ledger = {
        "contract": "hosd_scale_resident_layout_ledger_v4",
        "source_sha256": canonical_sha256(source),
        "input_rows": input_rows,
        "tree_manifest_hashes": sorted(
            {row["manifest_sha256"] for row in tree_rows}
        ),
        "target_manifest_hashes": sorted(
            {row["manifest_sha256"] for row in target_rows}
        ),
        "active_completion_hashes": {
            "scale_inputs": input_completion["content_hash"],
            "scale_trees": tree_completion["content_hash"],
            "scale_targets": target_completion["content_hash"],
            "scale_teacher_outputs": teacher_completion["content_hash"],
        },
        "maximum_input_decoded_bytes_per_event": max_input_per_event,
        "maximum_tree_shard_events": max_tree_shard_events,
        "maximum_tree_shard_decoded_bytes": max_tree_shard_bytes,
        "analytical_maximum_tree_layout_128_255": analytical_tree,
        "production_tree_shard_events": PRODUCTION_TREE_SHARD_EVENTS,
        "production_tree_shard_decoded_bytes_upper_bound": production_tree_bytes,
        "maximum_target_shard_events": max_target_shard_events,
        "maximum_target_shard_decoded_bytes": max_target_shard_bytes,
        "maximum_target_decoded_bytes_per_event": maximum_target_bytes_per_event,
        "production_target_shard_events": PRODUCTION_TARGET_SHARD_EVENTS,
        "production_target_shard_decoded_bytes_upper_bound": production_target_bytes,
        "miniature_byte_accounting": miniature_byte_accounting,
        "production_byte_accounting": production_byte_accounting,
        "multiplicity_basis": {
            "scale_input_prepare": "source_decode_plus_working_copy_plus_mmap_publish",
            "scale_tree_build": "one_input_shard_plus_tree_working_and_publish_copies",
            "scale_target_build": "one_input_and_tree_shard_plus_target_working_and_publish_copies",
            "scale_teacher_target_inference": "one_mmap_input_window_plus_authenticated_tree_shard_and_target_output_copies",
            "scale_graph_train": "two_input_windows_plus_one_tree_and_two_target_windows",
        },
    }
    return {**ledger, "content_hash": canonical_sha256(ledger)}


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


def derive_requests_from_measurements(
    *,
    maximum_projected_seconds_by_class,
    maximum_rss_bytes_by_class,
):
    requests = {}
    for resource, base in RESOURCE_CLASSES.items():
        memory_gib = max(
            MEMORY_FLOOR_GIB[resource],
            int(
                math.ceil(
                    maximum_rss_bytes_by_class[resource]
                    * MEMORY_SAFETY_FACTOR
                    / 1024**3
                )
            ),
        )
        policy_memory_gib = int(str(base["memory"]).removesuffix("G"))
        if memory_gib > policy_memory_gib:
            raise ValueError(
                f"projected {resource} memory exceeds registered Tigris policy"
            )
        requested_seconds = max(
            WALLTIME_ROUNDING_SECONDS,
            int(
                math.ceil(
                    maximum_projected_seconds_by_class[resource]
                    * WALLTIME_SAFETY_FACTOR
                    / WALLTIME_ROUNDING_SECONDS
                )
                * WALLTIME_ROUNDING_SECONDS
            ),
        )
        if requested_seconds > _slurm_seconds(str(base["time"])):
            raise ValueError(
                f"projected {resource} walltime exceeds registered Tigris policy"
            )
        request = dict(base)
        request["memory"] = f"{memory_gib}G"
        request["time"] = _slurm_time(requested_seconds)
        requests[resource] = request
    return requests


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
    scale_layout_ledger = _scale_layout_ledger(root, campaign["source"])
    scale_memory_projections = {}
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
        maximum_elapsed = max(int(row["elapsed_seconds"]) for row in rows)
        projected_seconds = maximum_elapsed * event_ratio * epoch_ratio
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
        node_observed_rss = 0
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
            node_observed_rss = max(node_observed_rss, observed_rss)
        if node_id in SCALE_MEMORY_PROJECTION_NODES:
            observed_population = int(campaign["split_roles"]["scale_train"])
            production_population = int(PRODUCTION_COUNTS["scale_train"])
            miniature_unit_bytes = int(
                scale_layout_ledger["miniature_byte_accounting"][node_id]
            )
            production_unit_bytes = int(
                scale_layout_ledger["production_byte_accounting"][node_id]
            )
            population_resident = node_id == "scale_input_prepare"
            observed_resident_unit = (
                miniature_unit_bytes * observed_population
                if population_resident
                else miniature_unit_bytes
            )
            fixed = max(0, node_observed_rss - observed_resident_unit)
            projected = fixed + (
                production_unit_bytes * production_population
                if population_resident
                else production_unit_bytes
            )
            max_rss[resource] = max(max_rss[resource], projected)
            registered_limit = _bytes(RESOURCE_CLASSES[resource]["memory"])
            if projected > registered_limit:
                raise ValueError(
                    f"{node_id} projected resident memory exceeds registered "
                    "Tigris limit"
                )
            coordinate_type = {
                "scale_input_prepare": "offline_and_hlt_input_coordinates",
                "scale_tree_build": "offline_and_hlt_tree_coordinates",
                "scale_target_build": "offline_and_hlt_target_coordinates",
                "scale_teacher_target_inference": "O_FULLREL_teacher_target_coordinate",
                "scale_graph_train": "worst_case_shortlisted_scale_graph",
            }[node_id]
            production_unit_events = {
                "scale_input_prepare": 1,
                "scale_tree_build": PRODUCTION_TREE_SHARD_EVENTS,
                "scale_target_build": PRODUCTION_TARGET_SHARD_EVENTS,
                "scale_teacher_target_inference": PRODUCTION_TARGET_SHARD_EVENTS,
                "scale_graph_train": PRODUCTION_TREE_SHARD_EVENTS,
            }[node_id]
            miniature_unit_events = {
                "scale_input_prepare": 1,
                "scale_tree_build": int(
                    scale_layout_ledger["maximum_tree_shard_events"]
                ),
                "scale_target_build": int(
                    scale_layout_ledger["maximum_target_shard_events"]
                ),
                "scale_teacher_target_inference": int(
                    scale_layout_ledger["maximum_target_shard_events"]
                ),
                "scale_graph_train": max(
                    int(scale_layout_ledger["maximum_tree_shard_events"]),
                    int(scale_layout_ledger["maximum_target_shard_events"]),
                ),
            }[node_id]
            scale_memory_projections[node_id] = with_content_hash({
                "contract": "hosd_scale_resident_memory_projection_v5",
                "pilot_node_id": node_id,
                "pilot_job_id": job_id,
                "coordinate_type": coordinate_type,
                "resource_class": resource,
                "source_sha256": canonical_sha256(campaign["source"]),
                "layout_evidence_sha256": scale_layout_ledger["content_hash"],
                "observed_population": observed_population,
                "observed_maximum_rss_bytes": node_observed_rss,
                "fixed_resident_bytes": fixed,
                "resident_unit": (
                    "event_population" if population_resident else "one_shard"
                ),
                "miniature_resident_unit_events": miniature_unit_events,
                "miniature_resident_unit_bytes_upper_bound": miniature_unit_bytes,
                "production_resident_unit_events": production_unit_events,
                "production_resident_unit_bytes_upper_bound": production_unit_bytes,
                "production_population": production_population,
                "production_shard_count": int(
                    math.ceil(
                        production_population
                        / production_unit_events
                    )
                ),
                "projected_resident_bytes": projected,
                "projection_model": (
                    "fixed_plus_authenticated_per_event_population_v2"
                    if population_resident
                    else "fixed_plus_authenticated_single_shard_v2"
                ),
                "loader_storage_contracts": [
                    "hosd_label_blind_input_view_v4",
                    "hosd_npy_mmap_store_v2",
                    "hosd_authenticated_tree_split_index_v1",
                    "one_shard_resident_target_cache",
                    "one_shard_resident_region_tree_cache",
                    "hosd_native_relation_npy_store_v1",
                ],
                "registered_tigris_limit_bytes": registered_limit,
                "within_registered_tigris_limit": True,
                "representative_real_task_completed": True,
            })
        measured_nodes[node_id] = {
            "job_id": job_id,
            "resource": resource,
            "stage": node["stage"],
            "task_count": len(rows),
            "elapsed_task_seconds": elapsed,
            "maximum_elapsed_task_seconds": maximum_elapsed,
            "event_scale_ratio": event_ratio,
            "epoch_scale_ratio": epoch_ratio,
            "projected_max_task_seconds": projected_seconds,
            "measured_gpu_hours": gpu_hours,
        }
    if not all(max_rss.values()):
        raise ValueError("miniature MaxRSS evidence is incomplete")
    if set(scale_memory_projections) != set(SCALE_MEMORY_PROJECTION_NODES) or any(
        not row["observed_maximum_rss_bytes"]
        for row in scale_memory_projections.values()
    ):
        raise ValueError("representative scale-node memory pilots are incomplete")
    active = maximum = 0
    for _, delta in sorted(intervals, key=lambda row: (row[0], row[1])):
        active += delta
        maximum = max(maximum, active)
    checkpoint_bytes, export_bytes = _artifact_bytes(root)
    requests = derive_requests_from_measurements(
        maximum_projected_seconds_by_class=max_projected_seconds,
        maximum_rss_bytes_by_class=max_rss,
    )
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
            "schema_version": 7,
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
            "scale_resident_layout_ledger": scale_layout_ledger,
            "scale_resident_memory_projections": scale_memory_projections,
            "projection_population_counts": PRODUCTION_COUNTS,
            "training_epoch_scale_ratio": 20,
            "request_memory_safety_factor": MEMORY_SAFETY_FACTOR,
            "request_memory_floor_gib_by_class": MEMORY_FLOOR_GIB,
            "request_time_safety_factor": WALLTIME_SAFETY_FACTOR,
            "request_time_rounding_seconds": WALLTIME_ROUNDING_SECONDS,
            "request_time_policy": (
                "maximum measured task elapsed times scaled by authenticated "
                "event/epoch ratios and fixed safety factor; fail closed above "
                "registered Tigris class limits"
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
