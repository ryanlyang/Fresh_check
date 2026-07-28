"""Authenticated storage/runtime measurements for RETB bootstrap."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .contracts import require_sha256, validate_content_hash, with_content_hash


STORAGE_MEASUREMENTS_CONTRACT = "retb_storage_measurements_v1"
REQUIRED_MEASUREMENTS = (
    "hlt_v3_compressed_bytes_per_jet",
    "region_sidecar_bytes_per_jet",
    "expert_target_fp16_bytes_per_scalar",
    "expert_target_fp32_bytes_per_scalar",
    "logits_bytes_per_jet",
    "identity_bytes_per_jet",
    "checkpoint_bytes",
    "projected_peak_concurrent_bytes",
    "available_storage_bytes",
    "cpu_degradation_jets_per_second",
    "gpu_expert_jets_per_second",
    "gpu_predictor_jets_per_second",
)


def build_storage_measurements(
    *,
    measurements: Mapping[str, float | int],
    evidence_hashes: Mapping[str, str] | None = None,
    source_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    measurement_profile: str,
) -> dict[str, Any]:
    if measurement_profile not in {"production_source_evidence", "miniature_test"}:
        raise ValueError("invalid storage measurement profile")
    normalized: dict[str, float | int] = {}
    missing = sorted(set(REQUIRED_MEASUREMENTS) - set(measurements))
    if missing:
        raise ValueError(f"storage measurements missing fields: {missing}")
    for key in REQUIRED_MEASUREMENTS:
        value = float(measurements[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"storage measurement {key} must be finite and nonnegative")
        normalized[key] = int(value) if value.is_integer() else value
    evidence_hashes = evidence_hashes or {}
    hashes = {
        str(key): require_sha256(value, name=f"evidence_hashes.{key}")
        for key, value in sorted(evidence_hashes.items())
    }
    evidence: dict[str, dict[str, Any]] = {}
    for key, raw in sorted((source_evidence or {}).items()):
        path = Path(str(raw["path"])).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"storage evidence is absent or unsafe: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        actual_bytes = int(path.stat().st_size)
        if "sha256" in raw and str(raw["sha256"]) != actual_sha:
            raise ValueError(f"storage evidence hash mismatch for {key}")
        if "bytes" in raw and int(raw["bytes"]) != actual_bytes:
            raise ValueError(f"storage evidence byte count mismatch for {key}")
        if key in hashes and hashes[key] != actual_sha:
            raise ValueError(f"storage evidence hash sources disagree for {key}")
        hashes[str(key)] = actual_sha
        evidence[str(key)] = {
            "path": str(path),
            "sha256": actual_sha,
            "bytes": actual_bytes,
            "purpose": str(raw.get("purpose", key)),
        }
    if measurement_profile == "production_source_evidence" and not evidence:
        raise ValueError(
            "production measurements require authenticated source-evidence files"
        )
    return with_content_hash(
        {
            "contract": STORAGE_MEASUREMENTS_CONTRACT,
            "schema_version": 1,
            "measurement_profile": measurement_profile,
            "measurements": normalized,
            "evidence_hashes": hashes,
            "source_evidence": evidence,
            "target_projection_formula": (
                "events*K*D*bytes_per_scalar_plus_logits_plus_identity"
            ),
            "storage_admission": (
                "projected_peak_concurrent_bytes_less_than_or_equal_to_"
                "available_storage_bytes_minus_serialized_reserve"
            ),
            "performance_measurements_select_scientific_models": False,
        }
    )


def validate_storage_measurements(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STORAGE_MEASUREMENTS_CONTRACT
    )
    build_storage_measurements(
        measurements=payload["measurements"],
        evidence_hashes=payload["evidence_hashes"],
        source_evidence=payload.get("source_evidence", {}),
        measurement_profile=str(payload["measurement_profile"]),
    )
    return digest


def miniature_storage_measurements() -> dict[str, Any]:
    return build_storage_measurements(
        measurements={
            "hlt_v3_compressed_bytes_per_jet": 1,
            "region_sidecar_bytes_per_jet": 1,
            "expert_target_fp16_bytes_per_scalar": 2,
            "expert_target_fp32_bytes_per_scalar": 4,
            "logits_bytes_per_jet": 40,
            "identity_bytes_per_jet": 16,
            "checkpoint_bytes": 1,
            "projected_peak_concurrent_bytes": 1,
            "available_storage_bytes": 2,
            "cpu_degradation_jets_per_second": 1,
            "gpu_expert_jets_per_second": 1,
            "gpu_predictor_jets_per_second": 1,
        },
        evidence_hashes={},
        source_evidence={},
        measurement_profile="miniature_test",
    )


__all__ = [
    "REQUIRED_MEASUREMENTS",
    "STORAGE_MEASUREMENTS_CONTRACT",
    "build_storage_measurements",
    "miniature_storage_measurements",
    "validate_storage_measurements",
]
