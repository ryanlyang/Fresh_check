"""Canonical immutable contracts for the RETB campaign."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


CANONICAL_JSON_CONTRACT = "retb_canonical_json_v1"
CAMPAIGN_SPEC_CONTRACT = "retb_campaign_spec_v2"
STEP1_REPORT_CONTRACT = "retb_step1_report_v1"
SEMANTIC_CONTROL_POLICY = {
    "relation_zero_scope": "one_biased_expert_at_a_time",
    "relation_zero_expert_order": [
        "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"
    ],
    "wrong_event_relation_matching": "exact_valid_particle_multiplicity",
    "wrong_event_relation_self_match_allowed": False,
    "wrong_event_singleton_policy": (
        "exclude_exact_singleton_multiplicity_strata_from_both_the_"
        "perturbed_row_and_its_paired_active_reference"
    ),
    "wrong_event_minimum_derangeable_stratum_size": 2,
    "policy_frozen_before_scientific_results": True,
}
SOURCE_STATUS_HASH_POLICY = "git_diff_binary_HEAD_plus_sorted_untracked_file_bytes_v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def require_git_object_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or GIT_OBJECT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase Git object ID")
    return value


def with_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "content_hash" in payload:
        raise ValueError("unhashed payload must not contain content_hash")
    result = dict(payload)
    result["content_hash"] = canonical_sha256(result)
    return result


def validate_content_hash(
    payload: Mapping[str, Any],
    *,
    expected_contract: str | None = None,
) -> str:
    if expected_contract is not None and payload.get("contract") != expected_contract:
        raise ValueError(
            f"contract mismatch: expected {expected_contract!r}, "
            f"got {payload.get('contract')!r}"
        )
    claimed = require_sha256(payload.get("content_hash"), name="content_hash")
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    actual = canonical_sha256(unhashed)
    if actual != claimed:
        raise ValueError(
            f"artifact content hash mismatch: claimed={claimed}, computed={actual}"
        )
    return actual


def source_record(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "commit": require_git_object_id(
            snapshot.get("source_commit"), name="source_snapshot.source_commit"
        ),
        "status_sha256": require_sha256(
            snapshot.get("source_status_sha256"),
            name="source_snapshot.source_status_sha256",
        ),
        "dirty": bool(snapshot.get("source_dirty")),
        "status_hash_policy": SOURCE_STATUS_HASH_POLICY,
    }


def bind_source(
    payload: Mapping[str, Any],
    *,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(payload)
    expected = source_record(source_snapshot)
    if "source" in payload:
        if payload["source"] != expected:
            raise ValueError("artifact is bound to a different source snapshot")
        return dict(payload)
    result = dict(payload)
    result.pop("content_hash")
    result["source"] = expected
    return with_content_hash(result)


def _publish_bytes(path: str | Path, encoded: bytes) -> str:
    destination = Path(path)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError(f"immutable destination is unsafe: {destination}")
        if destination.read_bytes() != encoded:
            raise FileExistsError(
                f"refusing to overwrite different immutable artifact: {destination}"
            )
        return "already_present"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            # Array coordinates can discover complete coverage at the same
            # instant and race to close one immutable wave.  Identical bytes
            # are a successful idempotent publication; only a different or
            # unsafe winner is an integrity failure.
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != encoded
            ):
                raise FileExistsError(
                    "immutable destination appeared with different bytes: "
                    f"{destination}"
                ) from exc
            return "already_present"
    finally:
        if temporary.exists():
            temporary.unlink()
    return "published"


def write_immutable_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    gzip_compressed: bool | None = None,
) -> dict[str, Any]:
    validate_content_hash(payload)
    raw = canonical_json_bytes(dict(payload)) + b"\n"
    compress = str(path).endswith(".gz") if gzip_compressed is None else gzip_compressed
    encoded = gzip.compress(raw, compresslevel=9, mtime=0) if compress else raw
    return {
        "path": str(Path(path).resolve()),
        "content_hash": str(payload["content_hash"]),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "status": _publish_bytes(path, encoded),
    }


def write_immutable_bytes(path: str | Path, payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("immutable byte payload must be bytes")
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "status": _publish_bytes(path, payload),
    }


def load_hashed_json(
    path: str | Path,
    *,
    expected_contract: str | None = None,
) -> dict[str, Any]:
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise FileNotFoundError(f"immutable artifact is absent or unsafe: {artifact}")
    encoded = artifact.read_bytes()
    raw = gzip.decompress(encoded) if artifact.suffix == ".gz" else encoded
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {artifact}")
    validate_content_hash(payload, expected_contract=expected_contract)
    return payload


def build_campaign_spec(
    *,
    campaign_id: str,
    campaign_profile: str,
    source_snapshot: Mapping[str, Any],
    parent_artifact_hashes: Mapping[str, str],
    run_registry_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id) is None:
        raise ValueError("campaign_id contains unsafe characters")
    if campaign_profile not in {"production_500k_scale3m", "miniature_test"}:
        raise ValueError(f"unsupported campaign profile {campaign_profile!r}")
    parents = {
        str(key): require_sha256(value, name=f"parent_artifact_hashes.{key}")
        for key, value in sorted(parent_artifact_hashes.items())
    }
    required = {
        "artifact_layout",
        "final_select_label_manifest",
        "global_determinism",
        "hlt_replica_manifest",
        "raw_input_schema",
        "scale_train_manifest",
        "split_audit",
        "split_manifest",
        "storage_measurements",
        "validation_partition_manifest",
    }
    missing = sorted(required - set(parents))
    if missing:
        raise ValueError(f"campaign spec is missing parent artifacts: {missing}")
    registries = {
        str(key): require_sha256(value, name=f"run_registry_hashes.{key}")
        for key, value in sorted(run_registry_hashes.items())
    }
    if not registries:
        raise ValueError("campaign spec requires registry artifacts")
    return with_content_hash(
        {
            "contract": CAMPAIGN_SPEC_CONTRACT,
            "schema_version": 2,
            "campaign_id": campaign_id,
            "scientific_program": "relation_expert_token_bridge",
            "campaign_stage": "step1_contracts_only",
            "campaign_profile": campaign_profile,
            "scientific_results_allowed": campaign_profile.startswith("production_"),
            "source": source_record(source_snapshot),
            "parent_artifact_hashes": parents,
            "registry_hashes": registries,
            "access_policy": {
                "epoch_selection": "val_stop_only",
                "component_selection": "val_design_only",
                "finalist_selection": "stage_n_stack_val_only",
                "final_test": "sealed_until_final_test_execution_lock",
                "performance_based_run_termination": False,
            },
            "semantic_control_policy": dict(SEMANTIC_CONTROL_POLICY),
            "dimensions": {
                "classes": 10,
                "max_constituents": 128,
                "raw_particle_fields": 14,
                "derived_particle_fields": 17,
            },
            "confirmation_pipeline_seeds": [101, 202, 303],
            "screen_seed": 101,
            "canonical_json_contract": CANONICAL_JSON_CONTRACT,
            "continuation_source_snapshot_must_match_exactly": True,
        }
    )


__all__ = [
    "CAMPAIGN_SPEC_CONTRACT",
    "CANONICAL_JSON_CONTRACT",
    "SEMANTIC_CONTROL_POLICY",
    "STEP1_REPORT_CONTRACT",
    "bind_source",
    "build_campaign_spec",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_hashed_json",
    "require_git_object_id",
    "require_sha256",
    "source_record",
    "validate_content_hash",
    "with_content_hash",
    "write_immutable_bytes",
    "write_immutable_json",
]
