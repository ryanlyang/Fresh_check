"""Canonical, immutable contracts for the relational ParT campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


CANONICAL_JSON_CONTRACT = "relational_part_canonical_json_v1"
CAMPAIGN_SPEC_CONTRACT = "relational_part_campaign_spec_v5"
STEP1_REPORT_CONTRACT = "relational_part_step1_report_v5"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SOURCE_STATUS_HASH_POLICY = (
    "git_diff_binary_HEAD_plus_sorted_untracked_file_bytes_v2"
)


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


def sha256_file(path: str | Path) -> str:
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise FileNotFoundError(f"file is absent or unsafe: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def require_git_object_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or GIT_OBJECT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40- or 64-digit Git object ID")
    return value


def with_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "content_hash" in payload:
        raise ValueError("unhashed payload must not contain content_hash")
    artifact = dict(payload)
    artifact["content_hash"] = canonical_sha256(artifact)
    return artifact


def bind_source_provenance(
    payload: Mapping[str, Any],
    *,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach one code snapshot to an artifact and recompute its identity."""

    validate_content_hash(payload)
    source = {
        "commit": require_git_object_id(
            source_snapshot.get("source_commit"), name="source_snapshot.source_commit"
        ),
        "status_sha256": require_sha256(
            source_snapshot.get("source_status_sha256"),
            name="source_snapshot.source_status_sha256",
        ),
        "dirty": bool(source_snapshot.get("source_dirty")),
        "status_hash_policy": SOURCE_STATUS_HASH_POLICY,
    }
    existing_source = payload.get("source")
    if existing_source is not None:
        if existing_source != source:
            raise ValueError("artifact is already bound to a different source snapshot")
        return dict(payload)
    rebound = dict(payload)
    rebound.pop("content_hash")
    rebound["source"] = source
    return with_content_hash(rebound)


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


def load_hashed_json(
    path: str | Path,
    *,
    expected_contract: str | None = None,
) -> dict[str, Any]:
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise FileNotFoundError(f"immutable artifact is absent or unsafe: {artifact}")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {artifact}")
    validate_content_hash(payload, expected_contract=expected_contract)
    return payload


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
            raise FileExistsError(
                f"immutable destination appeared during publication: {destination}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return "published"


def write_immutable_bytes(path: str | Path, payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("immutable byte payload must be bytes")
    status = _publish_bytes(path, payload)
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": status,
    }


def write_immutable_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(payload)
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    status = _publish_bytes(path, encoded)
    return {
        "path": str(Path(path).resolve()),
        "content_hash": str(payload["content_hash"]),
        "status": status,
    }


def build_campaign_spec(
    *,
    campaign_id: str,
    source_snapshot: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    global_determinism: Mapping[str, Any],
    split_manifest_hash: str,
    hlt_cache_status: str,
    campaign_profile: str = "production_1m_125k_0_125k_500k",
) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id) is None:
        raise ValueError("campaign_id contains unsafe characters")
    source_commit = require_git_object_id(
        source_snapshot.get("source_commit"), name="source_snapshot.source_commit"
    )
    source_status = require_sha256(
        source_snapshot.get("source_status_sha256"),
        name="source_snapshot.source_status_sha256",
    )
    parents = {
        str(name): require_sha256(value, name=f"artifact_hashes.{name}")
        for name, value in sorted(artifact_hashes.items())
    }
    required = {
        "split_binding",
        "hlt_expectation",
        "raw_input_schema",
        "normalization_contract",
        "angular_tree_resource_contract",
        "relation_family_registry",
        "screening_registry",
        "confirmation_architecture_registry",
        "semantic_control_registry",
        "artifact_layout",
        "global_determinism",
        "storage_projection",
    }
    missing = sorted(required - set(parents))
    if missing:
        raise ValueError(f"campaign spec is missing parent artifacts: {missing}")
    if hlt_cache_status not in {"expected_not_built", "authenticated"}:
        raise ValueError(f"invalid hlt_cache_status {hlt_cache_status!r}")
    if hlt_cache_status == "authenticated" and "hlt_binding" not in parents:
        raise ValueError("authenticated campaign spec requires hlt_binding")
    allowed_profiles = {
        "production_1m_125k_0_125k_500k",
        "nonproduction_miniature_test",
    }
    if campaign_profile not in allowed_profiles:
        raise ValueError(f"invalid campaign_profile {campaign_profile!r}")
    scientific_results_allowed = campaign_profile.startswith("production_")
    global_determinism_hash = validate_content_hash(global_determinism)
    if parents["global_determinism"] != global_determinism_hash:
        raise ValueError(
            "campaign global-determinism artifact differs from its parent hash"
        )
    if global_determinism.get("fixed_before_scientific_results") is not True:
        raise ValueError("global deterministic conventions must be fixed pre-results")
    if global_determinism.get("model_specific_override_allowed") is not False:
        raise ValueError("model-specific deterministic-policy overrides are forbidden")

    return with_content_hash(
        {
            "contract": CAMPAIGN_SPEC_CONTRACT,
            "schema_version": 5,
            "campaign_id": campaign_id,
            "scientific_program": "relational_particle_transformer_attention_bias",
            "campaign_stage": "step1_contracts_only",
            "campaign_profile": campaign_profile,
            "scientific_results_allowed": scientific_results_allowed,
            "source": {
                "commit": source_commit,
                "status_sha256": source_status,
                "dirty": bool(source_snapshot.get("source_dirty")),
                "status_hash_policy": (
                    SOURCE_STATUS_HASH_POLICY
                ),
            },
            "split_manifest_hash": require_sha256(
                split_manifest_hash, name="split_manifest_hash"
            ),
            "hlt_cache_status": hlt_cache_status,
            "parent_artifact_hashes": parents,
            "global_determinism": dict(global_determinism),
            "global_epsilon": 1.0e-6,
            "hlt_only_inference": True,
            "offline_or_teacher_required": False,
            "final_test_scientific_access": "sealed_until_locked_confirmation",
            "continuation_source_snapshot_must_match_exactly": True,
            "canonical_json_contract": CANONICAL_JSON_CONTRACT,
        }
    )


__all__ = [
    "CAMPAIGN_SPEC_CONTRACT",
    "CANONICAL_JSON_CONTRACT",
    "STEP1_REPORT_CONTRACT",
    "bind_source_provenance",
    "build_campaign_spec",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_hashed_json",
    "require_git_object_id",
    "require_sha256",
    "sha256_file",
    "validate_content_hash",
    "with_content_hash",
    "write_immutable_bytes",
    "write_immutable_json",
]
