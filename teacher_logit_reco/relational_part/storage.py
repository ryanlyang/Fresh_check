"""Measured storage projection for the relational ParT campaign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import with_content_hash


RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT = "relational_part_storage_measurements_v1"
RELATIONAL_STORAGE_PROJECTION_CONTRACT = "relational_part_storage_projection_v1"
GIB = 1024**3


@dataclass(frozen=True)
class StorageMeasurements:
    hlt_sample_jets: int
    hlt_sample_bytes: int
    tree_sample_jets: int
    tree_sample_bytes: int
    checkpoint_sample_count: int
    checkpoint_sample_bytes: int
    prediction_sample_events: int
    prediction_sample_bytes: int
    fixed_overhead_bytes: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StorageMeasurements":
        fields = {
            name: int(payload[name])
            for name in (
                "hlt_sample_jets",
                "hlt_sample_bytes",
                "tree_sample_jets",
                "tree_sample_bytes",
                "checkpoint_sample_count",
                "checkpoint_sample_bytes",
                "prediction_sample_events",
                "prediction_sample_bytes",
                "fixed_overhead_bytes",
            )
        }
        result = cls(**fields)
        for name, value in fields.items():
            minimum = 0 if name == "fixed_overhead_bytes" else 1
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
        return result

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


def build_storage_measurements(
    measurements: StorageMeasurements | Mapping[str, Any],
    *,
    source_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind measured byte counts to authenticated representative files.

    Each evidence record names the measured artifact, its SHA-256, byte count,
    and semantic purpose.  This prevents a projection made from remembered or
    hard-coded estimates from being labelled as measured.
    """

    from .contracts import require_sha256

    if not isinstance(measurements, StorageMeasurements):
        measurements = StorageMeasurements.from_mapping(measurements)
    required = {"hlt_cache", "tree_sidecar", "checkpoint", "predictions"}
    if set(source_evidence) != required:
        raise ValueError(
            "storage source evidence keys mismatch: "
            f"expected={sorted(required)}, actual={sorted(source_evidence)}"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for key in sorted(required):
        record = source_evidence[key]
        path = str(record.get("path", "")).strip()
        purpose = str(record.get("purpose", "")).strip()
        byte_count = int(record.get("bytes", -1))
        if not path or not purpose or byte_count <= 0:
            raise ValueError(f"storage evidence {key} is incomplete")
        normalized[key] = {
            "path": path,
            "sha256": require_sha256(record.get("sha256"), name=f"{key}.sha256"),
            "bytes": byte_count,
            "purpose": purpose,
        }
    if normalized["hlt_cache"]["bytes"] != measurements.hlt_sample_bytes:
        raise ValueError("HLT evidence bytes do not match measured HLT bytes")
    if normalized["tree_sidecar"]["bytes"] != measurements.tree_sample_bytes:
        raise ValueError("tree evidence bytes do not match measured tree bytes")
    if normalized["checkpoint"]["bytes"] != measurements.checkpoint_sample_bytes:
        raise ValueError("checkpoint evidence bytes do not match measured checkpoint bytes")
    if normalized["predictions"]["bytes"] != measurements.prediction_sample_bytes:
        raise ValueError("prediction evidence bytes do not match measured prediction bytes")
    return with_content_hash(
        {
            "contract": RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT,
            "schema_version": 1,
            "measurements": measurements.to_dict(),
            "source_evidence": normalized,
            "all_variable_components_measured": True,
        }
    )


def _ceil_scaled(sample_bytes: int, target_count: int, sample_count: int) -> int:
    return (int(sample_bytes) * int(target_count) + int(sample_count) - 1) // int(
        sample_count
    )


def build_storage_projection(
    measurements: StorageMeasurements | Mapping[str, Any],
    *,
    available_bytes: int,
    budget_gib: int = 20,
    minimum_free_reserve_gib: int = 20,
    total_hlt_jets: int = 1_750_000,
    total_tree_jets: int = 1_750_000,
    retained_checkpoint_count: int = 63,
    retained_final_prediction_sets: int = 54,
    final_test_events: int = 500_000,
) -> dict[str, Any]:
    """Project from measurements and fail before submission if space is unsafe."""

    measurement_artifact_sha256: str | None = None
    if not isinstance(measurements, StorageMeasurements):
        if measurements.get("contract") == RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT:
            from .contracts import require_sha256, validate_content_hash

            measurement_artifact_sha256 = validate_content_hash(
                measurements,
                expected_contract=RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT,
            )
            measurements = StorageMeasurements.from_mapping(
                measurements["measurements"]
            )
        else:
            measurements = StorageMeasurements.from_mapping(measurements)
    integer_inputs = {
        "available_bytes": available_bytes,
        "budget_gib": budget_gib,
        "minimum_free_reserve_gib": minimum_free_reserve_gib,
        "total_hlt_jets": total_hlt_jets,
        "total_tree_jets": total_tree_jets,
        "retained_checkpoint_count": retained_checkpoint_count,
        "retained_final_prediction_sets": retained_final_prediction_sets,
        "final_test_events": final_test_events,
    }
    if any(int(value) < 0 for value in integer_inputs.values()):
        raise ValueError(f"storage inputs must be nonnegative: {integer_inputs}")

    components = {
        "compressed_hlt_cache": _ceil_scaled(
            measurements.hlt_sample_bytes,
            total_hlt_jets,
            measurements.hlt_sample_jets,
        ),
        "compact_region_tree_sidecars": _ceil_scaled(
            measurements.tree_sample_bytes,
            total_tree_jets,
            measurements.tree_sample_jets,
        ),
        "retained_best_checkpoints": _ceil_scaled(
            measurements.checkpoint_sample_bytes,
            retained_checkpoint_count,
            measurements.checkpoint_sample_count,
        ),
        "locked_final_test_predictions": _ceil_scaled(
            measurements.prediction_sample_bytes,
            retained_final_prediction_sets * final_test_events,
            measurements.prediction_sample_events,
        ),
        "metrics_registries_logs_overhead": int(measurements.fixed_overhead_bytes),
    }
    projected = int(sum(components.values()))
    budget_bytes = int(budget_gib) * GIB
    reserve_bytes = int(minimum_free_reserve_gib) * GIB
    available_bytes = int(available_bytes)
    checks = {
        "within_campaign_budget": projected <= budget_bytes,
        "free_reserve_preserved": available_bytes - projected >= reserve_bytes,
        "uses_measured_hlt_bytes_per_jet": True,
        "uses_measured_tree_bytes_per_jet": True,
        "no_persistent_pair_matrices": True,
    }
    if not all(checks.values()):
        raise ValueError(
            "storage preflight failed: "
            f"projected_bytes={projected}, budget_bytes={budget_bytes}, "
            f"available_bytes={available_bytes}, reserve_bytes={reserve_bytes}, "
            f"checks={checks}"
        )

    return with_content_hash(
        {
            "contract": RELATIONAL_STORAGE_PROJECTION_CONTRACT,
            "schema_version": 1,
            "measurement_source": "representative_files_and_locked_tree_probe",
            "measurement_artifact_sha256": measurement_artifact_sha256,
            "measurements": measurements.to_dict(),
            "projection_population": {
                "hlt_jets": int(total_hlt_jets),
                "tree_jets": int(total_tree_jets),
                "retained_checkpoint_count_upper_bound": int(
                    retained_checkpoint_count
                ),
                "retained_final_prediction_sets_upper_bound": int(
                    retained_final_prediction_sets
                ),
                "events_per_final_prediction_set": int(final_test_events),
            },
            "component_bytes": components,
            "projected_bytes": projected,
            "budget_bytes": budget_bytes,
            "available_bytes": available_bytes,
            "minimum_free_reserve_bytes": reserve_bytes,
            "projected_free_bytes_after_campaign": available_bytes - projected,
            "checks": checks,
            "ok": True,
        }
    )


__all__ = [
    "GIB",
    "RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT",
    "RELATIONAL_STORAGE_PROJECTION_CONTRACT",
    "StorageMeasurements",
    "build_storage_measurements",
    "build_storage_projection",
]
