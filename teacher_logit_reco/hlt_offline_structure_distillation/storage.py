"""Evidence-only cache storage and throughput policy."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import STORAGE_MEASUREMENT_CONTRACT, require_sha256, with_content_hash


# One compact offline cache, four HLT-replica caches, and four residual caches
# are the conservative persistent-family multiplicity.  Pair families are
# measured for throughput/rebuild planning but are never projected as
# persistent dense storage.
PERSISTENT_CACHE_MULTIPLICITY = 9
PRODUCTION_TRAIN_POPULATION = 500_000
PRODUCTION_DESIGN_POPULATION = 50_000 + 25_000
PROJECTION_COUNTS = {
    "production_500k": (
        PERSISTENT_CACHE_MULTIPLICITY * PRODUCTION_TRAIN_POPULATION
        + 3 * PRODUCTION_DESIGN_POPULATION
    ),
    "scale_3m": (
        PERSISTENT_CACHE_MULTIPLICITY
        * (PRODUCTION_TRAIN_POPULATION + 3_000_000)
        + 3 * PRODUCTION_DESIGN_POPULATION
    ),
}


def build_storage_measurements(
    *,
    family_measurements: Mapping[str, Mapping[str, Any]],
    available_storage_bytes: int,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if available_storage_bytes <= 0:
        raise ValueError("available_storage_bytes must be positive")
    families = []
    persistent_bytes_per_jet = 0.0
    streamed_bytes_per_jet = 0.0
    for family_id, raw in sorted(family_measurements.items()):
        sample_events = int(raw["sample_events"])
        bytes_written = int(raw["bytes_written"])
        elapsed_seconds = float(raw["elapsed_seconds"])
        valid_components = int(raw["valid_components"])
        total_components = int(raw["total_components"])
        if (
            sample_events <= 0
            or bytes_written < 0
            or elapsed_seconds <= 0
            or total_components <= 0
            or not 0 <= valid_components <= total_components
        ):
            raise ValueError(f"invalid storage evidence for {family_id}")
        bytes_per_jet = bytes_written / sample_events
        storage_class = str(raw["storage_class"])
        if storage_class.startswith("stream_") or "pair" in storage_class:
            streamed_bytes_per_jet += bytes_per_jet
        else:
            persistent_bytes_per_jet += bytes_per_jet
        families.append(
            {
                "family_id": family_id,
                "storage_class": storage_class,
                "sample_events": sample_events,
                "bytes_written": bytes_written,
                "elapsed_seconds": elapsed_seconds,
                "bytes_per_jet": bytes_per_jet,
                "extraction_jets_per_second": sample_events / elapsed_seconds,
                "maximum_shard_rebuild_seconds": float(
                    raw.get("maximum_shard_rebuild_seconds", elapsed_seconds)
                ),
                "target_mask_sparsity": 1.0 - valid_components / total_components,
            }
        )
    persistent_projections = {
        name: int(round(persistent_bytes_per_jet * count))
        for name, count in PROJECTION_COUNTS.items()
    }
    # Stage-C pair inputs are content-addressed (one physical payload per
    # target/role, regardless of probe/tap) and reconstructible.  Until the
    # pair worker itself is fully fused, include their measured transient peak
    # rather than pretending that streamed families consume zero disk.
    stage_c_pair_event_equivalents = (
        4 * PRODUCTION_TRAIN_POPULATION + PRODUCTION_DESIGN_POPULATION
    )
    transient_pair_peak = int(
        round(streamed_bytes_per_jet * stage_c_pair_event_equivalents)
    )
    projections = {
        name: persistent + transient_pair_peak
        for name, persistent in persistent_projections.items()
    }
    exceeds = any(value > available_storage_bytes for value in projections.values())
    return with_content_hash(
        {
            "contract": STORAGE_MEASUREMENT_CONTRACT,
            "schema_version": 3,
            "families": families,
            "available_storage_bytes": int(available_storage_bytes),
            "total_measured_bytes_per_jet": persistent_bytes_per_jet,
            "measured_streamed_dense_bytes_per_jet": streamed_bytes_per_jet,
            "projected_storage_bytes": projections,
            "projected_persistent_storage_bytes": persistent_projections,
            "projected_stage_c_transient_pair_peak_bytes": transient_pair_peak,
            "stage_c_transient_pair_event_equivalents": stage_c_pair_event_equivalents,
            "projection_populations": dict(PROJECTION_COUNTS),
            "persistent_cache_multiplicity": PERSISTENT_CACHE_MULTIPLICITY,
            "fixed_validation_cache_multiplicity": 3,
            "pair_target_bytes_excluded_from_persistent_projection": True,
            "pair_target_bytes_included_in_peak_projection": True,
            "probe_tap_persistent_bytes": 0,
            "probe_tap_storage_contract": (
                "stream_exact_frozen_tap_into_worker_RAM_v1"
            ),
            "probe_payload_storage_contract": (
                "hardlinked_content_addressed_probe_payload_v1"
            ),
            "duplicate_probe_payload_physical_multiplicity": 1,
            "projection_exceeds_available_storage": exceeds,
            "policy": (
                "persist_compact_jet_targets_stream_same_view_node_pair"
                if exceeds
                else "persist_compact_jet_targets_pair_targets_may_still_stream"
            ),
            "pair_target_default": "stream_same_view_node_or_pair",
            "decision_uses_scientific_results": False,
            "parent_hashes": {
                name: require_sha256(value, name=f"parent_hashes.{name}")
                for name, value in sorted(parent_hashes.items())
            },
            "source": dict(source),
        }
    )


__all__ = [
    "PERSISTENT_CACHE_MULTIPLICITY",
    "PROJECTION_COUNTS",
    "build_storage_measurements",
]
