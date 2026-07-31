"""Exact Stage-B target/teacher wave coordinates and completion attestations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    STAGE_B_WAVE_COMPLETION_CONTRACT,
    STRUCTURE_TARGET_REGISTRY_CONTRACT,
    TARGET_CACHE_MANIFEST_CONTRACT,
    validate_content_hash,
    with_content_hash,
    load_hashed_json,
    write_immutable_json,
)
from .extractors import PHYSICAL_TARGET_IDS


TARGET_SPLITS = ("model_train", "val_stop", "val_design")
HLT_REPLICAS = (0, 1, 2, 3)
TEACHER_IDS = ("O_BASE", "O_FULLREL")


def _replicas_for_split(split: str) -> tuple[int, ...]:
    return HLT_REPLICAS if split == "model_train" else (0,)


def stage_b_wave_rows(
    *,
    wave_kind: str,
    target_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validate_content_hash(
        target_registry, expected_contract=STRUCTURE_TARGET_REGISTRY_CONTRACT
    )
    physical = sorted(
        str(row["target_id"])
        for row in target_registry["targets"]
        if bool(row.get("executable_current_source"))
        and str(row["target_id"]) in PHYSICAL_TARGET_IDS
    )
    if not physical:
        raise ValueError("Stage-B wave has no executable physical targets")
    if wave_kind == "canonical":
        return [
            {
                "row_id": f"CANONICAL__{split}",
                "split": split,
                "target_ids": physical,
                "artifact_kind": "canonical_offline",
            }
            for split in TARGET_SPLITS
        ]
    if wave_kind == "hlt_analogue":
        return [
            {
                "row_id": f"HLT_ANALOGUE__{split}__R{replica}",
                "split": split,
                "replica": replica,
                "target_ids": physical,
                "artifact_kind": "hlt_analogue",
            }
            for split in TARGET_SPLITS
            for replica in _replicas_for_split(split)
        ]
    if wave_kind == "teacher_output":
        return [
            {
                "row_id": f"TEACHER_OUTPUT__{split}__{teacher}",
                "split": split,
                "teacher_id": teacher,
            }
            for split in TARGET_SPLITS
            for teacher in TEACHER_IDS
        ]
    if wave_kind == "residual":
        return [
            {
                "row_id": f"RESIDUAL__{split}__R{replica}",
                "split": split,
                "replica": replica,
            }
            for split in TARGET_SPLITS
            for replica in _replicas_for_split(split)
        ]
    raise ValueError("unknown Stage-B wave kind")


def build_stage_b_wave_completion(
    *,
    wave_kind: str,
    target_registry: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact row coverage before publishing an aggregate completion."""

    rows = stage_b_wave_rows(
        wave_kind=wave_kind, target_registry=target_registry
    )
    if len(manifests) != len(rows):
        raise ValueError("Stage-B wave manifest count differs")
    expected_by_coordinate = {
        (
            row["split"],
            row.get("replica"),
            row.get("teacher_id"),
        ): row
        for row in rows
    }
    observed = {}
    for manifest in manifests:
        validate_content_hash(
            manifest, expected_contract=TARGET_CACHE_MANIFEST_CONTRACT
        )
        if manifest.get("source") != dict(source):
            raise ValueError("Stage-B cache source differs")
        teacher_id = None
        persisted = set(manifest.get("persisted_target_ids", ()))
        if wave_kind == "teacher_output":
            if "T_OFFLINE_LOGITS_O_BASE" in persisted:
                teacher_id = "O_BASE"
                expected_targets = {
                    "T_OFFLINE_LOGITS_O_BASE",
                    "T_OFFLINE_POOLED_LATENT",
                }
            elif "T_OFFLINE_LOGITS_O_FULLREL" in persisted:
                teacher_id = "O_FULLREL"
                expected_targets = {"T_OFFLINE_LOGITS_O_FULLREL"}
            else:
                raise ValueError("teacher-output target coverage differs")
            if persisted != expected_targets:
                raise ValueError("teacher-output cache target coverage differs")
        key = (
            str(manifest["split"]),
            (
                None
                if manifest.get("hlt_replica_id") is None
                else int(manifest["hlt_replica_id"])
            ),
            teacher_id,
        )
        if key not in expected_by_coordinate or key in observed:
            raise ValueError("Stage-B cache coordinate coverage differs")
        expected = expected_by_coordinate[key]
        if wave_kind in {"canonical", "hlt_analogue"}:
            target_ids = set(expected["target_ids"])
            if (
                persisted
                | set(manifest.get("streamed_target_ids", ()))
                != target_ids
                or str(manifest.get("artifact_kind"))
                != expected["artifact_kind"]
            ):
                raise ValueError("Stage-B physical target coverage differs")
        elif wave_kind == "residual":
            if (
                str(manifest.get("artifact_kind")) != "residual"
                or not persisted
            ):
                raise ValueError("Stage-B residual cache semantics differ")
        observed[key] = manifest
    if set(observed) != set(expected_by_coordinate):
        raise ValueError("Stage-B wave coordinate coverage differs")
    ordered = [
        observed[
            (
                row["split"],
                row.get("replica"),
                row.get("teacher_id"),
            )
        ]
        for row in rows
    ]
    return with_content_hash(
        {
            "contract": STAGE_B_WAVE_COMPLETION_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "wave_kind": wave_kind,
            "target_registry_sha256": target_registry["content_hash"],
            "rows": [
                {
                    **row,
                    "cache_manifest_sha256": manifest["content_hash"],
                }
                for row, manifest in zip(rows, ordered)
            ],
            "cache_manifest_hashes": {
                row["row_id"]: manifest["content_hash"]
                for row, manifest in zip(rows, ordered)
            },
            "row_count": len(rows),
            "exact_coordinate_coverage": True,
            "label_access_for_extraction": False,
            "performance_based_termination": False,
        }
    )


def stage_b_manifest_path(
    campaign_root: str | Path,
    *,
    wave_kind: str,
    row: Mapping[str, Any],
) -> Path:
    root = Path(campaign_root)
    split = str(row["split"])
    if wave_kind == "canonical":
        return root / "targets" / "canonical" / split / "target_manifest.json"
    if wave_kind == "hlt_analogue":
        return (
            root
            / "targets"
            / "hlt_analogues"
            / split
            / f"replica_{row['replica']}"
            / "target_manifest.json"
        )
    if wave_kind == "teacher_output":
        return (
            root
            / "teachers"
            / "outputs"
            / split
            / str(row["teacher_id"])
            / "target_manifest.json"
        )
    if wave_kind == "residual":
        return (
            root
            / "targets"
            / "residuals"
            / split
            / f"replica_{row['replica']}"
            / "target_manifest.json"
        )
    raise ValueError("unknown Stage-B manifest path kind")


def stage_b_completion_path(
    campaign_root: str | Path, *, wave_kind: str
) -> Path:
    root = Path(campaign_root)
    return {
        "canonical": root / "targets" / "canonical" / "target_manifest.json",
        "hlt_analogue": root
        / "targets"
        / "hlt_analogues"
        / "completion.json",
        "teacher_output": root
        / "teachers"
        / "teacher_output_cache_completion.json",
        "residual": root / "targets" / "residuals" / "completion.json",
    }[wave_kind]


def try_finalize_stage_b_wave(
    *,
    campaign_root: str | Path,
    wave_kind: str,
    target_registry: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any] | None:
    rows = stage_b_wave_rows(
        wave_kind=wave_kind, target_registry=target_registry
    )
    paths = [
        stage_b_manifest_path(campaign_root, wave_kind=wave_kind, row=row)
        for row in rows
    ]
    if not all(path.is_file() for path in paths):
        return None
    manifests = [
        load_hashed_json(
            path, expected_contract=TARGET_CACHE_MANIFEST_CONTRACT
        )
        for path in paths
    ]
    artifact = build_stage_b_wave_completion(
        wave_kind=wave_kind,
        target_registry=target_registry,
        manifests=manifests,
        source=source,
    )
    write_immutable_json(
        stage_b_completion_path(campaign_root, wave_kind=wave_kind),
        artifact,
    )
    return artifact


__all__ = [
    "HLT_REPLICAS",
    "TARGET_SPLITS",
    "TEACHER_IDS",
    "build_stage_b_wave_completion",
    "stage_b_completion_path",
    "stage_b_manifest_path",
    "stage_b_wave_rows",
    "try_finalize_stage_b_wave",
]
