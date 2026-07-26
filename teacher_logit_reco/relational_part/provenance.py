"""Artifact-layout and source-provenance helpers for Step 1."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES

from .contracts import (
    require_git_object_id,
    validate_content_hash,
    with_content_hash,
)


ARTIFACT_LAYOUT_CONTRACT = "relational_part_artifact_layout_v2"
RAW_INPUT_SCHEMA_CONTRACT = "relational_part_raw_input_schema_v2"


REQUIRED_DIRECTORIES = (
    "backend",
    "inputs",
    "inputs/hlt_cache",
    "inputs/relation_tree_cache",
    "registry",
    "runs",
    "selection",
    "selection/semantic_controls",
    "final_test",
    "reports",
    "job_ledgers",
)


def build_raw_input_schema_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": RAW_INPUT_SCHEMA_CONTRACT,
            "schema_version": 2,
            "schema_id": "jetclass_hlt_raw_v1",
            "required_particle_branches": list(PARTICLE_READ_BRANCHES),
            "pid_branches": [
                "part_isChargedHadron",
                "part_isNeutralHadron",
                "part_isPhoton",
                "part_isElectron",
                "part_isMuon",
            ],
            "track_fields": {
                "d0": {
                    "branch": "part_d0val",
                    "missing_policy": "no_numeric_sentinel",
                },
                "d0err": {
                    "branch": "part_d0err",
                    "missing_policy": "no_numeric_sentinel",
                },
                "dz": {
                    "branch": "part_dzval",
                    "missing_policy": "no_numeric_sentinel",
                },
                "dzerr": {
                    "branch": "part_dzerr",
                    "missing_policy": "no_numeric_sentinel",
                },
            },
            "track_measurement_validity": [
                "charged_pid",
                "finite_d0_dz",
                "finite_strictly_positive_uncertainties",
                "not_declared_missing_sentinel",
            ],
            "pid_threshold": 0.5,
            "pid_binary_tolerance": 1.0e-6,
            "pid_zero_hot_policy": "unknown_category",
            "pid_multi_hot_policy": "fail_preflight",
            "charge_states": [-1, 0, 1],
            "charge_integer_tolerance": 1.0e-6,
            "charge_quantization": (
                "nearest_locked_state_after_tolerance_validation"
            ),
            "sentinel_inference_from_observed_distribution_allowed": False,
        }
    )


def validate_raw_input_schema_contract(schema: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        schema,
        expected_contract=RAW_INPUT_SCHEMA_CONTRACT,
    )
    semantic = dict(schema)
    semantic.pop("content_hash", None)
    semantic.pop("source", None)
    expected = build_raw_input_schema_contract()
    expected.pop("content_hash")
    if semantic != expected:
        raise ValueError("raw-input schema differs from the locked schema")
    return digest


def build_artifact_layout_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": ARTIFACT_LAYOUT_CONTRACT,
            "schema_version": 2,
            "directories": list(REQUIRED_DIRECTORIES),
            "step1_files": [
                "campaign_spec.json",
                "inputs/split_manifest.json.gz",
                "inputs/split_audit.json",
                "inputs/raw_input_schema.json",
                "inputs/hlt_expectation.json",
                "inputs/normalization_contract.json",
                "inputs/angular_tree_resource_contract.json",
                "registry/relation_family_registry.json",
                "registry/screening_registry.json",
                "registry/confirmation_architecture_registry.json",
                "registry/semantic_control_registry.json",
                "registry/global_determinism.json",
                "registry/artifact_layout.json",
                "storage_measurements.json",
                "storage_projection.json",
                "reports/relational_part_step1_report.json",
            ],
            "future_files": [
                "backend/backend_manifest.json",
                "backend/throughput_probe.json",
                "inputs/preconstruction_raw_input_audit.json",
                "inputs/hlt_cache_audit.json",
                "inputs/relation_normalization.json",
                "inputs/postconstruction_input_audit.json",
                "selection/screening_summary.json",
                "selection/confirmation_registry.json",
                "selection/confirmation_summary.json",
                "selection/locked_finalists.json",
                "reports/relational_part_report.json",
                "reports/relational_part_report.md",
            ],
            "immutable_json_policy": "canonical_sha256_atomic_no_overwrite",
            "persistent_pair_matrix_allowed": False,
        }
    )


def initialize_artifact_layout(
    campaign_root: str | Path,
    layout: Mapping[str, Any],
) -> dict[str, Any]:
    from .contracts import validate_content_hash

    validate_content_hash(layout, expected_contract=ARTIFACT_LAYOUT_CONTRACT)
    root = Path(campaign_root)
    if root.exists() and root.is_symlink():
        raise ValueError(f"campaign root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for relative in layout["directories"]:
        destination = root / str(relative)
        if destination.exists() and (
            destination.is_symlink() or not destination.is_dir()
        ):
            raise ValueError(f"artifact directory is unsafe: {destination}")
        if not destination.exists():
            destination.mkdir(parents=True)
            created.append(str(relative))
    return {
        "campaign_root": str(root.resolve()),
        "layout_sha256": str(layout["content_hash"]),
        "created_directories": created,
        "all_directories": list(layout["directories"]),
    }


def source_snapshot(repo_root: str | Path) -> dict[str, Any]:
    """Return commit and dirty-state hashes without mutating the repository."""

    root = Path(repo_root).resolve()

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    require_git_object_id(commit, name="source_commit")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    status_hash = hashlib.sha256(status.encode("utf-8")).hexdigest()
    return {
        "source_commit": commit,
        "source_status_sha256": status_hash,
        "source_dirty": bool(status),
    }


__all__ = [
    "ARTIFACT_LAYOUT_CONTRACT",
    "RAW_INPUT_SCHEMA_CONTRACT",
    "REQUIRED_DIRECTORIES",
    "build_artifact_layout_contract",
    "build_raw_input_schema_contract",
    "initialize_artifact_layout",
    "source_snapshot",
    "validate_raw_input_schema_contract",
]
