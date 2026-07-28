"""Source and artifact-layout provenance for RETB."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES

from .contracts import require_git_object_id, validate_content_hash, with_content_hash


ARTIFACT_LAYOUT_CONTRACT = "retb_artifact_layout_v1"
RAW_INPUT_SCHEMA_CONTRACT = "retb_raw_input_schema_v2"

REQUIRED_DIRECTORIES = (
    "inputs",
    "inputs/hlt_replicas/replica_0",
    "inputs/hlt_replicas/replica_1",
    "inputs/hlt_replicas/replica_2",
    "inputs/hlt_replicas/replica_3",
    "inputs/region_tree",
    "inputs/normalization/offline_500k",
    "inputs/normalization/hlt_shared_500k",
    "inputs/normalization/offline_scale",
    "inputs/normalization/hlt_shared_scale",
    "inputs/normalization/token",
    "registry",
    "offline_experts",
    "offline_fusions",
    "hlt_experts",
    "native_hlt_fusions",
    "offline_targets",
    "post_selection_oracle_targets",
    "bridge_pilots",
    "bridge_target_candidates",
    "predictors",
    "predictor_bundles",
    "bundle_searches",
    "joint_predictors",
    "token_refiners",
    "final_adapters",
    "unrestricted_fusions",
    "capacity_controls",
    "scale_up",
    "selection_predictions/stack_val",
    "evaluations/stack_val_selection_metrics",
    "selection",
    "reports",
    "job_ledgers",
)


def source_snapshot(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()

    def run_text(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def run_bytes(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout

    commit = run_text("rev-parse", "HEAD")
    require_git_object_id(commit, name="source_commit")
    status = run_text("status", "--porcelain=v1", "--untracked-files=all")
    digest = hashlib.sha256()
    digest.update(b"retb_source_snapshot_v1\0")
    digest.update(run_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--"))
    untracked = [
        item
        for item in run_bytes(
            "ls-files", "--others", "--exclude-standard", "-z"
        ).split(b"\0")
        if item
    ]
    for encoded in sorted(untracked):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("untracked source path escapes repository") from exc
        digest.update(b"\0untracked\0")
        digest.update(encoded)
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return {
        "source_commit": commit,
        "source_status_sha256": digest.hexdigest(),
        "source_dirty": bool(status),
    }


def build_raw_input_schema() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": RAW_INPUT_SCHEMA_CONTRACT,
            "schema_version": 2,
            "max_constituents": 128,
            "raw_dimension": 14,
            "derived_dimension": 17,
            "raw_fields": [
                "pt",
                "eta",
                "phi",
                "energy",
                "charge",
                "is_charged_hadron",
                "is_neutral_hadron",
                "is_photon",
                "is_electron",
                "is_muon",
                "d0",
                "d0err",
                "dz",
                "dzerr",
            ],
            "required_particle_branches": list(PARTICLE_READ_BRANCHES),
            "derived_input_implementation": "jetclass_fresh.part_inputs",
            "pid_zero_hot_policy": "unknown_category",
            "pid_multi_hot_policy": "fail_preflight",
            "invalid_track_measurement_sentinel": {
                "d0": 0.0,
                "d0err": 0.0,
                "dz": 0.0,
                "dzerr": 0.0,
                "inferred_from_observed_zeros": False,
            },
            "measurement_validity_states": [
                "not_track_domain",
                "track_measurement_available",
                "track_measurement_missing",
            ],
            "constituent_matching_fields_allowed": False,
        }
    )


def build_artifact_layout() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": ARTIFACT_LAYOUT_CONTRACT,
            "schema_version": 1,
            "directories": list(REQUIRED_DIRECTORIES),
            "step1_files": [
                "campaign_spec.json",
                "inputs/split_manifest.json.gz",
                "inputs/final_select_label_manifest.json.gz",
                "inputs/validation_partition_manifest.json.gz",
                "inputs/scale_train_manifest.json.gz",
                "inputs/split_audit.json",
                "inputs/scale_train_audit.json",
                "inputs/raw_input_schema.json",
                "inputs/hlt_replica_manifest.json",
                "registry/global_determinism.json",
                "storage_measurements.json",
                "reports/retb_step1_report.json",
            ],
            "immutable_json_policy": "canonical_sha256_atomic_no_overwrite",
            "persistent_pair_matrix_allowed": False,
        }
    )


def initialize_artifact_layout(
    campaign_root: str | Path,
    layout: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(layout, expected_contract=ARTIFACT_LAYOUT_CONTRACT)
    root = Path(campaign_root)
    if root.exists() and root.is_symlink():
        raise ValueError(f"campaign root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for relative in layout["directories"]:
        path = root / str(relative)
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError(f"artifact directory is unsafe: {path}")
        if not path.exists():
            path.mkdir(parents=True)
            created.append(str(relative))
    return {
        "campaign_root": str(root.resolve()),
        "layout_sha256": str(layout["content_hash"]),
        "created_directories": created,
    }


__all__ = [
    "ARTIFACT_LAYOUT_CONTRACT",
    "RAW_INPUT_SCHEMA_CONTRACT",
    "REQUIRED_DIRECTORIES",
    "build_artifact_layout",
    "build_raw_input_schema",
    "initialize_artifact_layout",
    "source_snapshot",
]
