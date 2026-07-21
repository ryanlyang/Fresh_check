"""Canonical artifact helpers for the prediction-anchored bridge campaign.

The bridge pilot relies on small immutable JSON contracts.  This module keeps
their hashing and publication rules in one dependency-light place so later
steps cannot accidentally use formatting-dependent hashes or overwrite a
previously bound artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


PREDICTION_ANCHORED_CANONICAL_JSON_CONTRACT = (
    "prediction_anchored_canonical_json_v1"
)


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the UTF-8 canonical representation used for campaign hashes."""

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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def with_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy *payload* and append its hash, rejecting an ambiguous input hash."""

    if "content_hash" in payload:
        raise ValueError("payload already contains content_hash")
    output = dict(payload)
    output["content_hash"] = canonical_sha256(output)
    return output


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
    claimed = payload.get("content_hash")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("artifact has no valid content_hash")
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    actual = canonical_sha256(unhashed)
    if actual != claimed:
        raise ValueError(
            f"artifact content hash mismatch: expected {claimed}, computed {actual}"
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


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish JSON once, accepting only byte-semantic idempotence."""

    destination = Path(path)
    validate_content_hash(payload)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError(f"immutable destination is unsafe: {destination}")
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if canonical_json_bytes(existing) != canonical_json_bytes(dict(payload)):
            raise FileExistsError(
                f"refusing to overwrite different immutable artifact: {destination}"
            )
        return {
            "path": str(destination.resolve()),
            "content_hash": str(payload["content_hash"]),
            "status": "already_present",
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
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
            # Linking a fully flushed temporary inode is atomic and refuses an
            # existing destination, unlike os.replace which can overwrite a
            # concurrently published immutable artifact.
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"immutable destination appeared during publication: {destination}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination.resolve()),
        "content_hash": str(payload["content_hash"]),
        "status": "published",
    }


__all__ = [
    "PREDICTION_ANCHORED_CANONICAL_JSON_CONTRACT",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_hashed_json",
    "sha256_file",
    "validate_content_hash",
    "with_content_hash",
    "write_immutable_json",
]
