"""Production numerical executors for the prediction-anchored campaign."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bridge_contracts import load_hashed_json, sha256_file
from .bridge_execution import (
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)
from .bridge_r0 import StreamedR0TrainConfig, train_streamed_r0
from .bridge_ram import AllocationNpzStager, AllocationRamLedger, StagedSource
from .bridge_splits import (
    PREDICTION_ANCHORED_SPLIT_CONTRACT,
    authorize_split_access,
    split_binding,
)
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, compute_local_particle_residual_fields


def _paths_from_source(source: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: str(source[name]["path"])
        for name in ("hlt_npz", "hlt_metadata", "offline_npz", "offline_metadata")
    }


def _verify_staged_source_binding(
    *,
    split: str,
    source: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    if int(report.get("n_events", -1)) != int(source["n_events"]):
        raise ValueError(f"staged {split} count differs from execution spec")
    if report.get("event_order_sha256") != source["event_order_sha256"]:
        raise ValueError(f"staged {split} order differs from execution spec")
    expected_files = {
        "hlt": source["hlt_npz"]["sha256"],
        "offline": source["offline_npz"]["sha256"],
    }
    if report.get("source_file_sha256") != expected_files:
        raise ValueError(f"staged {split} source file hash differs from execution spec")
    expected_content = {
        "hlt": source["hlt_content_hash"],
        "offline": source["offline_content_hash"],
    }
    if report.get("source_content_hash") != expected_content:
        raise ValueError(f"staged {split} content hash differs from execution spec")
    if not all(
        count == 1 for count in report.get("persistent_npz_open_counts", {}).values()
    ):
        raise RuntimeError(f"staged {split} did not open each compressed source exactly once")


def _stream_truth_batches(
    *,
    hlt: StagedSource,
    offline: StagedSource,
    indices: Sequence[int] | np.ndarray,
    batch_size: int,
    epoch: int,
    shuffle: bool,
    seed: int,
) -> Iterable[dict[str, np.ndarray]]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if order.ndim != 1 or order.size == 0:
        raise ValueError("streamed R0 split must contain at least one row")
    if shuffle:
        np.random.default_rng(int(seed) + 1_000_003 * int(epoch)).shuffle(order)
    for start in range(0, order.size, int(batch_size)):
        selected = order[start : start + int(batch_size)]
        hlt_batch = hlt.read_indices(selected, names=("tokens", "mask"))
        offline_batch = offline.read_indices(selected, names=("tokens", "mask"))
        fields, target_mask, _, _, _ = compute_local_particle_residual_fields(
            hlt_batch["tokens"],
            hlt_batch["mask"],
            offline_batch["tokens"],
            offline_batch["mask"],
            radii=DEFAULT_LOCAL_RESIDUAL_RADII,
        )
        yield {
            "hlt_tokens": hlt_batch["tokens"],
            "hlt_mask": hlt_batch["mask"],
            "target_fields": fields,
            "target_mask": target_mask,
        }


def run_streamed_r0_from_execution_spec(
    execution_spec_path: str | Path,
    *,
    output_dir: str | Path,
    ram_root: str | Path,
    allocation_id: str | None = None,
    batch_size: int = 256,
    shard_size: int = 8192,
    device: str | None = None,
    capacity_bytes: int | None = None,
    allow_unverified_test_root: bool = False,
) -> dict[str, Any]:
    """Train the ordinary residual predictor from resident raw sources.

    This is the first real campaign executor.  It produces only the R0
    weights, immutable registration, and scalar training curves.
    """

    if int(batch_size) <= 0 or int(shard_size) <= 0:
        raise ValueError("batch_size and shard_size must be positive")
    spec = load_hashed_json(
        execution_spec_path, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    validate_prediction_anchored_execution_spec(spec, verify_file_hashes=False)
    child = load_hashed_json(
        spec["child_manifest"]["path"], expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    if child["content_hash"] != spec["child_manifest"]["content_hash"]:
        raise ValueError("execution spec child manifest binding changed")
    train_parent_hash, train_split_hash = split_binding(child, "model_train")
    stop_parent_hash, stop_split_hash = split_binding(child, "model_val_stop")
    train_access = authorize_split_access(
        split_name="model_train",
        purpose="r0_training",
        parent_manifest_sha256=train_parent_hash,
        bound_split_sha256=train_split_hash,
    )
    stop_access = authorize_split_access(
        split_name="model_val_stop",
        purpose="checkpoint_selection",
        parent_manifest_sha256=stop_parent_hash,
        bound_split_sha256=stop_split_hash,
    )
    allocation = str(allocation_id or os.environ.get("SLURM_JOB_ID", "local_r0"))
    ledger = AllocationRamLedger(
        ram_root,
        allocation_id=allocation,
        capacity_bytes=capacity_bytes,
        allow_unverified_test_root=bool(allow_unverified_test_root),
    )
    try:
        stager = AllocationNpzStager(ledger, rank=0, world_size=1)
        wanted = {
            split: _paths_from_source(spec["sources"][split])
            for split in ("model_train", "model_val")
        }
        staged, stage_report = stager.stage_named_pairs(wanted, shard_size=int(shard_size))
        for split in wanted:
            _verify_staged_source_binding(
                split=split,
                source=spec["sources"][split],
                report=staged[split][2],
            )
        if not stage_report["all_persistent_npz_open_counts_equal_one"]:
            raise RuntimeError("R0 allocation violated the one-open source contract")
        train_hlt, train_offline, _ = staged["model_train"]
        val_hlt, val_offline, _ = staged["model_val"]
        train_indices = np.arange(train_hlt.n_events, dtype=np.int64)
        stop_indices = np.asarray(
            child["children"]["model_val_stop"]["parent_row_indices"], dtype=np.int64
        )
        raw_config = dict(spec["r0_training"])
        config = StreamedR0TrainConfig(output_dir=str(output_dir), **raw_config)
        if device is not None:
            config = replace(config, device=str(device))

        def train_batches(epoch: int):
            return _stream_truth_batches(
                hlt=train_hlt,
                offline=train_offline,
                indices=train_indices,
                batch_size=int(batch_size),
                epoch=int(epoch),
                shuffle=True,
                seed=int(config.seed),
            )

        def stop_batches(epoch: int):
            return _stream_truth_batches(
                hlt=val_hlt,
                offline=val_offline,
                indices=stop_indices,
                batch_size=int(batch_size),
                epoch=int(epoch),
                shuffle=False,
                seed=int(config.seed),
            )

        result = train_streamed_r0(
            config,
            train_batches=train_batches,
            model_val_stop_batches=stop_batches,
            provenance_hashes={
                "preprocessing_sha256": spec["preprocessing_sha256"],
                "target_schema_sha256": spec["target_schema_sha256"],
                "split_manifest_sha256": child["content_hash"],
            },
            matching_policy={
                "contract": "prediction_anchored_r0_training_membership_v1",
                "train_split": "model_train",
                "train_split_sha256": train_access["bound_split_sha256"],
                "checkpoint_split": "model_val_stop",
                "checkpoint_split_sha256": stop_access["bound_split_sha256"],
                "train_access_receipt_sha256": train_access["content_hash"],
                "stop_access_receipt_sha256": stop_access["content_hash"],
                "event_alignment": "exact_hlt_offline_jet_identity",
            },
        )
        result.update(
            {
                "execution_spec_sha256": spec["content_hash"],
                "allocation_id": allocation,
                "source_splits": ["model_train", "model_val"],
                "one_open_per_compressed_source": True,
                "persistent_dense_fields_written": False,
                "ram_peak_reserved_bytes": ledger.snapshot()["peak_reserved_bytes"],
                "checkpoint_sha256": sha256_file(result["checkpoint"]),
            }
        )
        return result
    finally:
        ledger.cleanup()


__all__ = ["run_streamed_r0_from_execution_spec"]
