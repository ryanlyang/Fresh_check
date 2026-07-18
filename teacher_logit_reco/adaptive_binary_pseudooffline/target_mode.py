"""Measured target-mode selection for storage-bounded ABPH campaigns."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    JetIdentity,
    load_offline_view,
    load_split_manifest,
    manifest_hash,
)
from teacher_logit_reco.architecture_view_part import load_cached_offline_view

from .compact_target_codec import encode_compact_target_arrays
from .config import canonical_hash
from .accounting_preflight import (
    audit_target_batch_feasibility,
    synthetic_edge_case_preflight,
)
from .inputs import validate_hlt_view_contract, validate_offline_view_contract
from .ram_workspace import RankLocalWorkspace
from .storage_quota import (
    ABPH_MAX_PROJECTED_PEAK_BYTES,
    ABPH_STORAGE_PROJECTION_CONTRACT,
)
from .targets import (
    ABPH_HIERARCHY_GROUPINGS,
    ABPH_TARGET_BUILDER_CONTRACT,
    ABPH_TARGET_BUILDER_VERSION,
    GROUP_FEATURE_NAMES,
    PARTICLE_TARGET_NAMES,
    ROOT_FEATURE_NAMES,
    AdaptiveBinaryHierarchyLayout,
    build_adaptive_binary_targets,
)


ABPH_TARGET_MODE_SELECTION_CONTRACT = "adaptive_binary_target_mode_selection_v1"
ABPH_SHARED_TRANSIENT_TARGET_MODE = "shared_transient_compact"
ABPH_RANK_LOCAL_TARGET_MODE = "rank_local_build"
ABPH_TARGET_MODES = (
    ABPH_SHARED_TRANSIENT_TARGET_MODE,
    ABPH_RANK_LOCAL_TARGET_MODE,
)
ABPH_TARGET_SAMPLE_MEASUREMENT_CONTRACT = "adaptive_binary_real_target_sample_v1"
ABPH_TARGET_PROJECTION_MARGIN = 1.15


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sample_indices(labels: np.ndarray, jets_per_class: int) -> np.ndarray:
    selected: list[int] = []
    for label in range(len(LABEL_NAMES)):
        positions = np.flatnonzero(labels == label)
        if len(positions) < jets_per_class:
            raise ValueError(
                f"target-mode sample requires {jets_per_class} jets for class "
                f"{LABEL_NAMES[label]}, found {len(positions)}"
            )
        selected.extend(int(value) for value in positions[:jets_per_class])
    return np.asarray(sorted(selected), dtype=np.int64)


def _measurement_arrays(targets: Any, labels: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {
        key: np.ascontiguousarray(value)
        for key, value in targets.array_dict().items()
    }
    arrays.update(
        {
            "labels": np.ascontiguousarray(labels, dtype=np.int64),
            "jet_file_indices": np.arange(len(labels), dtype=np.int32),
            "jet_entries": np.arange(len(labels), dtype=np.int64),
        }
    )
    return arrays


def measure_real_target_sample(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    data_dir: str | Path | None = None,
    split: str = "model_train",
    jets_per_class: int = 64,
    workspace: RankLocalWorkspace | None = None,
) -> dict[str, Any]:
    """Compile and encode a deterministic, class-stratified real target sample."""

    manifest = load_split_manifest(manifest_path)
    expected = len(manifest.splits[split])
    input_reservation = None
    sample_reservation = None
    if workspace is not None:
        # Refuse before loading monolithic source caches if their calibrated tensor
        # footprint cannot retain the mandatory cgroup headroom.
        expected_source_bytes = expected * (2 * 128 * 14 * 4 + 2 * 128 + 32)
        input_reservation = workspace.reserve(
            owner="target_mode_preflight",
            role="hlt_and_offline_source_views",
            expected_bytes=max(1, expected_source_bytes),
        )
        sample_reservation = workspace.reserve(
            owner="target_mode_preflight",
            role="target_sample_and_codec_scratch",
            expected_bytes=max(1, int(jets_per_class) * len(LABEL_NAMES) * 1_000_000),
        )
    try:
        hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
        offline_view = load_cached_offline_view(offline_cache_dir, split, verify_hash=True)
        if workspace is not None and input_reservation is not None:
            measured_sources = sum(
                int(value.nbytes)
                for value in (
                    hlt_view.tokens,
                    hlt_view.mask,
                    hlt_view.labels,
                    offline_view.tokens,
                    offline_view.mask,
                    offline_view.labels,
                )
            )
            workspace.commit(input_reservation, measured_bytes=measured_sources)
    except Exception:
        if workspace is not None:
            if sample_reservation is not None:
                workspace.release(sample_reservation)
            if input_reservation is not None:
                workspace.release(input_reservation)
        raise
    validate_hlt_view_contract(hlt_view, manifest, split, expected_n_jets=expected)
    validate_offline_view_contract(
        offline_view,
        manifest,
        split,
        expected_n_jets=expected,
        hlt_view=hlt_view,
    )
    indices = _sample_indices(np.asarray(hlt_view.labels), int(jets_per_class))
    sample_ids = tuple(hlt_view.jet_ids[int(index)] for index in indices)
    measurements: dict[str, dict[str, Any]] = {}
    feasibility_reports: dict[str, dict[str, Any]] = {}
    peak_logical = 0
    for grouping in ABPH_HIERARCHY_GROUPINGS:
        targets = build_adaptive_binary_targets(
            hlt_view.tokens[indices],
            hlt_view.mask[indices],
            offline_view.tokens[indices],
            offline_view.mask[indices],
            jet_ids=sample_ids,
            layout=AdaptiveBinaryHierarchyLayout(grouping=grouping),
        )
        arrays = _measurement_arrays(targets, np.asarray(hlt_view.labels)[indices])
        feasibility_reports[f"{split}/{grouping}"] = audit_target_batch_feasibility(
            targets,
            labels=np.asarray(hlt_view.labels)[indices],
        )
        encoded, codec_manifest = encode_compact_target_arrays(
            arrays,
            omitted_identity_keys=tuple(
                key for key in arrays if key.endswith("_identities")
            ),
        )
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **encoded)
        logical_bytes = sum(int(value.nbytes) for value in arrays.values())
        encoded_bytes = sum(int(value.nbytes) for value in encoded.values())
        peak_logical = max(peak_logical, logical_bytes)
        measurements[grouping] = {
            "n_jets": len(indices),
            "stored_bytes": len(buffer.getbuffer()),
            "encoded_uncompressed_bytes": encoded_bytes,
            "logical_uncompressed_bytes": logical_bytes,
            "stored_bytes_per_jet": len(buffer.getbuffer()) / len(indices),
            "logical_bytes_per_jet": logical_bytes / len(indices),
            "codec_manifest_hash": codec_manifest["manifest_hash"],
            "logical_content_hash": codec_manifest["logical_content_hash"],
        }
    if workspace is not None:
        assert sample_reservation is not None
        workspace.commit(sample_reservation, measured_bytes=max(1, peak_logical))
        workspace.release(sample_reservation)
        assert input_reservation is not None
        workspace.release(input_reservation)
    payload: dict[str, Any] = {
        "contract": ABPH_TARGET_SAMPLE_MEASUREMENT_CONTRACT,
        "builder_contract": ABPH_TARGET_BUILDER_CONTRACT,
        "builder_version": ABPH_TARGET_BUILDER_VERSION,
        "manifest_hash": manifest_hash(manifest),
        "split": split,
        "jets_per_class": int(jets_per_class),
        "n_jets": len(indices),
        "sample_global_indices": indices.tolist(),
        "sample_jet_identity_hash": jet_identity_hash(sample_ids),
        "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
        "offline_content_hash": offline_view.metadata.get("offline_content_hash"),
        "data_dir": str(data_dir or manifest.data_dir),
        "groupings": measurements,
        "feasibility_reports": feasibility_reports,
        "synthetic_edge_cases": synthetic_edge_case_preflight(),
    }
    payload["content_hash"] = canonical_hash(payload)
    return payload


def _projection_without_targets(projection: Mapping[str, Any]) -> int:
    rows = projection.get("rows")
    if not isinstance(rows, list):
        raise ValueError("storage projection has no artifact rows")
    retained = [
        row
        for row in rows
        if "target" not in str(row.get("artifact_family", "")).lower()
    ]
    if not retained:
        return 0
    final_wave = max(int(row["active_through_wave"]) for row in retained)
    return max(
        sum(
            int(row["expected_bytes"])
            + (
                int(row.get("atomic_write_overhead_bytes", 0))
                if wave == int(row["active_from_wave"])
                else 0
            )
            for row in retained
            if int(row["active_from_wave"]) <= wave <= int(row["active_through_wave"])
        )
        for wave in range(final_wave + 1)
    )


def select_target_mode(
    *,
    campaign_root: str | Path,
    campaign_mode: str,
    split_sizes: Mapping[str, int],
    measurement: Mapping[str, Any],
    storage_projection: Mapping[str, Any],
    workspace_capacity: Mapping[str, Any],
    hlt_cache_bytes: int,
    current_campaign_bytes: int,
    target_chunk_size: int,
    source_provenance_by_split: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile the immutable storage/RAM decision consumed by every worker."""

    if storage_projection.get("contract") != ABPH_STORAGE_PROJECTION_CONTRACT:
        raise ValueError("target-mode selection requires a validated storage projection")
    if measurement.get("contract") != ABPH_TARGET_SAMPLE_MEASUREMENT_CONTRACT:
        raise ValueError("target-mode selection requires a real target measurement")
    if set(source_provenance_by_split) != {"model_train", "model_val"}:
        raise ValueError("target-mode selection requires exact train/validation provenance")
    for split, row in source_provenance_by_split.items():
        if any(
            row.get(key) in {None, ""}
            for key in (
                "source_manifest_hash",
                "hlt_content_hash",
                "offline_content_hash",
                "jet_identity_hash",
            )
        ):
            raise ValueError(f"target-mode source provenance is incomplete for {split}")
    expected_measurement_hash = measurement.get("content_hash")
    if expected_measurement_hash != canonical_hash(
        {key: value for key, value in measurement.items() if key != "content_hash"}
    ):
        raise ValueError("target measurement content hash mismatch")
    target_jets = int(split_sizes["model_train"]) + int(split_sizes["model_val"])
    grouping_rows: dict[str, Any] = {}
    projected_target_bytes = 0
    maximum_logical_per_jet = 0.0
    for grouping in ABPH_HIERARCHY_GROUPINGS:
        row = measurement["groupings"][grouping]
        projected = math.ceil(
            float(row["stored_bytes_per_jet"])
            * target_jets
            * ABPH_TARGET_PROJECTION_MARGIN
        )
        projected_target_bytes += projected
        maximum_logical_per_jet = max(
            maximum_logical_per_jet, float(row["logical_bytes_per_jet"])
        )
        grouping_rows[grouping] = {
            **dict(row),
            "projected_shared_bytes": projected,
        }
    non_target_projection = _projection_without_targets(storage_projection)
    persistent_base = max(
        int(current_campaign_bytes),
        int(non_target_projection),
        int(hlt_cache_bytes),
    )
    atomic_target_overhead = math.ceil(
        maximum_logical_per_jet * int(target_chunk_size)
    )
    shared_peak = persistent_base + projected_target_bytes + atomic_target_overhead
    rank_local_target_working_set = math.ceil(
        maximum_logical_per_jet * int(target_chunk_size) * 2.0
    )
    ram_limit = int(workspace_capacity["reservation_limit_bytes"])
    if rank_local_target_working_set > ram_limit:
        raise MemoryError(
            "rank-local target compiler does not fit the verified RAM workspace with 20% headroom"
        )
    mode = (
        ABPH_SHARED_TRANSIENT_TARGET_MODE
        if shared_peak <= ABPH_MAX_PROJECTED_PEAK_BYTES
        else ABPH_RANK_LOCAL_TARGET_MODE
    )
    payload: dict[str, Any] = {
        "contract": ABPH_TARGET_MODE_SELECTION_CONTRACT,
        "campaign_root": str(Path(campaign_root).resolve()),
        "campaign_mode": str(campaign_mode),
        "selected_mode": mode,
        "selection_is_worker_overrideable": False,
        "measurement_content_hash": expected_measurement_hash,
        "storage_projection_content_hash": storage_projection.get("content_hash"),
        "source_manifest_hash": measurement.get("manifest_hash"),
        "hlt_content_hash": measurement.get("hlt_content_hash"),
        "offline_content_hash": measurement.get("offline_content_hash"),
        "source_provenance_by_split": {
            str(split): dict(row)
            for split, row in dict(source_provenance_by_split).items()
        },
        "sample_jet_identity_hash": measurement.get("sample_jet_identity_hash"),
        "data_dir": measurement.get("data_dir"),
        "target_chunk_size": int(target_chunk_size),
        "split_sizes": {key: int(value) for key, value in split_sizes.items()},
        "grouping_measurements": grouping_rows,
        "projected_shared_target_bytes": int(projected_target_bytes),
        "projected_non_target_peak_bytes": int(non_target_projection),
        "measured_current_campaign_bytes": int(current_campaign_bytes),
        "measured_hlt_cache_bytes": int(hlt_cache_bytes),
        "projected_shared_peak_bytes": int(shared_peak),
        "persistent_peak_gate_bytes": ABPH_MAX_PROJECTED_PEAK_BYTES,
        "rank_local_target_working_set_bytes": int(rank_local_target_working_set),
        "workspace_capacity": dict(workspace_capacity),
        "minimum_ram_headroom_fraction": 0.20,
    }
    payload["content_hash"] = canonical_hash(payload)
    return payload


def write_target_mode_selection(path: str | Path, payload: Mapping[str, Any]) -> None:
    if payload.get("content_hash") != canonical_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    ):
        raise ValueError("target-mode selection content hash mismatch")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)


def load_target_mode_selection(
    path: str | Path,
    *,
    campaign_root: str | Path | None = None,
    expected_mode: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != ABPH_TARGET_MODE_SELECTION_CONTRACT:
        raise ValueError("target-mode selection contract mismatch")
    if payload.get("content_hash") != canonical_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    ):
        raise ValueError("target-mode selection content hash mismatch")
    if payload.get("selected_mode") not in ABPH_TARGET_MODES:
        raise ValueError("target-mode selection contains an unknown mode")
    if payload.get("selection_is_worker_overrideable") is not False:
        raise ValueError("target-mode selection must be immutable to workers")
    for key in (
        "source_manifest_hash",
        "hlt_content_hash",
        "offline_content_hash",
        "measurement_content_hash",
        "storage_projection_content_hash",
    ):
        if payload.get(key) in {None, ""}:
            raise ValueError(f"target-mode selection lacks {key}")
    split_rows = payload.get("source_provenance_by_split")
    if not isinstance(split_rows, Mapping) or set(split_rows) != {
        "model_train",
        "model_val",
    }:
        raise ValueError("target-mode selection lacks exact train/validation provenance")
    for split, row in split_rows.items():
        if not isinstance(row, Mapping) or any(
            row.get(key) in {None, ""}
            for key in (
                "source_manifest_hash",
                "hlt_content_hash",
                "offline_content_hash",
                "jet_identity_hash",
            )
        ):
            raise ValueError(f"target-mode selection has incomplete {split} provenance")
    if campaign_root is not None and Path(str(payload.get("campaign_root", ""))).resolve() != Path(campaign_root).resolve():
        raise ValueError("target-mode selection belongs to another campaign")
    if expected_mode is not None and payload.get("selected_mode") != expected_mode:
        raise ValueError("target-mode selection differs from the required mode")
    return payload


def build_rank_local_offline_view(
    *,
    manifest_path: str | Path,
    split: str,
    identities: Sequence[JetIdentity],
    data_dir: str | Path | None = None,
) -> Any:
    """Read only one planned identity range directly from immutable ROOT inputs."""

    manifest = load_split_manifest(manifest_path)
    scoped = replace(
        manifest,
        splits={**manifest.splits, split: list(identities)},
        split_sizes={**manifest.split_sizes, split: len(identities)},
    )
    return load_offline_view(
        scoped,
        split,
        data_dir=(data_dir or manifest.data_dir),
        max_constits=manifest.max_constits,
        verify_label_branches=False,
    )


def rank_local_target_metadata(
    *,
    selection: Mapping[str, Any],
    split: str,
    grouping: str,
    n_jets: int,
    jet_identity_hash_value: str,
) -> dict[str, Any]:
    """Create deterministic aggregate metadata for an exact online target stream."""

    if selection.get("selected_mode") != ABPH_RANK_LOCAL_TARGET_MODE:
        raise ValueError("rank-local metadata requires rank_local_build selection")
    chunk_size = int(selection["target_chunk_size"])
    layout = AdaptiveBinaryHierarchyLayout(grouping=grouping)
    shards = [
        {
            "shard_index": index,
            "start": start,
            "stop": min(start + chunk_size, int(n_jets)),
            "n_jets": min(chunk_size, int(n_jets) - start),
        }
        for index, start in enumerate(range(0, int(n_jets), chunk_size))
    ]
    split_provenance = dict(selection.get("source_provenance_by_split", {})).get(
        str(split), {}
    )
    source_manifest_hash = split_provenance.get(
        "source_manifest_hash", selection["source_manifest_hash"]
    )
    hlt_content_hash = split_provenance.get(
        "hlt_content_hash", selection["hlt_content_hash"]
    )
    offline_content_hash = split_provenance.get(
        "offline_content_hash", selection["offline_content_hash"]
    )
    expected_identity_hash = split_provenance.get(
        "jet_identity_hash", jet_identity_hash_value
    )
    if expected_identity_hash != jet_identity_hash_value:
        raise ValueError("rank-local target metadata identity hash differs from active HLT")
    metadata: dict[str, Any] = {
        "cache_contract": "rank_local_exact_target_stream_v1",
        "builder_contract": ABPH_TARGET_BUILDER_CONTRACT,
        "builder_version": ABPH_TARGET_BUILDER_VERSION,
        "split": str(split),
        "grouping": layout.grouping,
        "layout": layout.to_dict(),
        "n_jets": int(n_jets),
        "n_shards": len(shards),
        "chunk_size": chunk_size,
        "source_manifest_hash": source_manifest_hash,
        "hlt_content_hash": hlt_content_hash,
        "offline_content_hash": offline_content_hash,
        "jet_identity_hash": str(jet_identity_hash_value),
        "root_feature_names": list(ROOT_FEATURE_NAMES),
        "group_feature_names": list(GROUP_FEATURE_NAMES),
        "particle_target_names": list(PARTICLE_TARGET_NAMES),
        "target_semantics": "exact deterministic online build from planned raw ROOT identities",
        "target_mode": ABPH_RANK_LOCAL_TARGET_MODE,
        "target_mode_selection_hash": selection["content_hash"],
        "shards": shards,
    }
    metadata["target_content_hash"] = canonical_hash(
        {
            "builder_contract": ABPH_TARGET_BUILDER_CONTRACT,
            "builder_version": ABPH_TARGET_BUILDER_VERSION,
            "split": split,
            "grouping": layout.grouping,
            "source_manifest_hash": metadata["source_manifest_hash"],
            "hlt_content_hash": metadata["hlt_content_hash"],
            "offline_content_hash": metadata["offline_content_hash"],
            "jet_identity_hash": metadata["jet_identity_hash"],
            "selection_hash": metadata["target_mode_selection_hash"],
        }
    )
    return metadata


def campaign_and_hlt_bytes(campaign_root: str | Path, hlt_cache_dir: str | Path) -> tuple[int, int]:
    return _directory_bytes(Path(campaign_root)), _directory_bytes(Path(hlt_cache_dir))


__all__ = [
    "ABPH_RANK_LOCAL_TARGET_MODE",
    "ABPH_SHARED_TRANSIENT_TARGET_MODE",
    "ABPH_TARGET_MODE_SELECTION_CONTRACT",
    "ABPH_TARGET_MODES",
    "ABPH_TARGET_PROJECTION_MARGIN",
    "ABPH_TARGET_SAMPLE_MEASUREMENT_CONTRACT",
    "build_rank_local_offline_view",
    "campaign_and_hlt_bytes",
    "load_target_mode_selection",
    "measure_real_target_sample",
    "rank_local_target_metadata",
    "select_target_mode",
    "write_target_mode_selection",
]
