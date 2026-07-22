"""Production numerical executors for the prediction-anchored campaign."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from jetclass_fresh.hlt_baseline import resolve_device

from .bridge_contracts import load_hashed_json, sha256_file
from .bridge_contracts import canonical_sha256, with_content_hash, write_immutable_json
from .bridge import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    BridgeScalers,
    build_bridge_recipe,
    fit_bridge_scalers,
    virtual_bridge,
)
from .bridge_execution import (
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)
from .bridge_r0 import StreamedR0TrainConfig, train_streamed_r0
from .bridge_ram import AllocationNpzStager, AllocationRamLedger, FrozenR0Runner, StagedSource
from .bridge_splits import (
    PREDICTION_ANCHORED_SPLIT_CONTRACT,
    authorize_split_access,
    split_binding,
)
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, compute_local_particle_residual_fields
from .hierarchical_global_reconstructor import fit_absolute_output_scaler
from .bridge_semantic_evidence import (
    build_bridge_quantile_reference_from_standardized_corrections,
)


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


def prepare_bridge_inputs_from_execution_spec(
    execution_spec_path: str | Path,
    *,
    r0_checkpoint_path: str | Path,
    r0_registration_path: str | Path,
    output_dir: str | Path,
    ram_root: str | Path,
    allocation_id: str | None = None,
    batch_size: int = 512,
    shard_size: int = 8192,
    device: str = "auto",
    capacity_bytes: int | None = None,
    allow_unverified_test_root: bool = False,
) -> dict[str, Any]:
    """Fit exact distillation-child scalers and publish virtual recipes."""

    spec = load_hashed_json(
        execution_spec_path, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    validate_prediction_anchored_execution_spec(spec, verify_file_hashes=False)
    child = load_hashed_json(
        spec["child_manifest"]["path"], expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    registration = load_hashed_json(r0_registration_path)
    r0_path = Path(r0_checkpoint_path)
    r0_sha = sha256_file(r0_path)
    if registration.get("checkpoint_sha256") != r0_sha:
        raise ValueError("R0 registration/checkpoint binding changed")
    if registration.get("split_manifest") != child["content_hash"]:
        raise ValueError("R0 registration uses a different split manifest")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bridge input output directory is not empty: {output}")
    allocation = str(allocation_id or os.environ.get("SLURM_JOB_ID", "local_bridge_inputs"))
    ledger = AllocationRamLedger(
        ram_root,
        allocation_id=allocation,
        capacity_bytes=capacity_bytes,
        allow_unverified_test_root=bool(allow_unverified_test_root),
    )
    try:
        stager = AllocationNpzStager(ledger, rank=0, world_size=1)
        staged, combined_report = stager.stage_named_pairs(
            {"stack_train": _paths_from_source(spec["sources"]["stack_train"])},
            shard_size=int(shard_size),
        )
        hlt, offline, stage_report = staged["stack_train"]
        _verify_staged_source_binding(
            split="stack_train",
            source=spec["sources"]["stack_train"],
            report=stage_report,
        )
        r0 = FrozenR0Runner(r0_path, device=resolve_device(str(device)))
        indices = np.asarray(
            child["children"]["stack_train_distill"]["parent_row_indices"], dtype=np.int64
        )
        if indices.size == 0:
            raise ValueError("stack_train_distill is empty")
        particle_width = int(hlt.manifest["arrays"]["tokens"]["shape"][1])
        scaler_value_bytes = (
            int(indices.size) * particle_width * 50 * 2 * np.dtype(np.float32).itemsize
        )
        working_bytes = int(np.ceil(1.10 * scaler_value_bytes)) + 1024 * 1024
        reservation = ledger.reserve(
            owner="rank0",
            role="exact_stack_train_distill_scaler_working_values",
            expected_bytes=working_bytes,
            category="derived",
        )
        ledger.commit(reservation, measured_bytes=working_bytes)

        def batches():
            for start in range(0, indices.size, int(batch_size)):
                selected = indices[start : start + int(batch_size)]
                hlt_batch = hlt.read_indices(selected, names=("tokens", "mask"))
                offline_batch = offline.read_indices(selected, names=("tokens", "mask"))
                truth, target_mask, _, _, _ = compute_local_particle_residual_fields(
                    hlt_batch["tokens"],
                    hlt_batch["mask"],
                    offline_batch["tokens"],
                    offline_batch["mask"],
                    radii=DEFAULT_LOCAL_RESIDUAL_RADII,
                )
                anchor, _ = r0.predict_numpy(hlt_batch["tokens"], hlt_batch["mask"])
                yield anchor, truth, target_mask

        scaler_parents = {
            "source_manifest_sha256": child["children"]["stack_train_distill"]["content_hash"],
            "r0_checkpoint_sha256": r0_sha,
            "target_schema_sha256": spec["target_schema_sha256"],
            "mask_sha256": stage_report["event_order_sha256"],
            "fit_code_sha256": canonical_sha256(
                {"fit_version": "physical_valid_particle_exact_channel_quantiles_v1"}
            ),
        }
        try:
            physical_scalers = fit_bridge_scalers(
                batches,
                parent_hashes=scaler_parents,
                channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
            )
        finally:
            ledger.release(reservation)
        all50_scalers = BridgeScalers(
            mu_f0=physical_scalers.mu_f0,
            sigma_f0=physical_scalers.sigma_f0,
            q99_delta=physical_scalers.q99_delta,
            sigma_delta=physical_scalers.sigma_delta,
            trust_scale=physical_scalers.trust_scale,
            epsilon=physical_scalers.epsilon,
            active=physical_scalers.q99_delta > 0,
            sparse_nonzero_fallback=physical_scalers.sparse_nonzero_fallback,
            valid_count=physical_scalers.valid_count,
            parent_hashes=scaler_parents,
            channel_policy=BRIDGE_CHANNEL_ALL50,
        )
        quantile_reservation = ledger.reserve(
            owner="rank0",
            role="exact_stack_train_distill_standardized_bridge_quantiles",
            expected_bytes=working_bytes,
            category="derived",
        )
        ledger.commit(quantile_reservation, measured_bytes=working_bytes)
        try:
            standardized_pieces = []
            # TorchBridgeScalers stores these buffers as float32.  Lock the
            # quantile reference to those exact deployed values so validation
            # does not fail merely from a float64->float32 representation cast.
            sigma_physical = np.asarray(physical_scalers.sigma_delta[:45], dtype=np.float32)
            for anchor, truth, target_mask in batches():
                bridge = virtual_bridge(
                    anchor,
                    truth,
                    target_mask,
                    rho="0.100",
                    channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
                )
                standardized_pieces.append(
                    (
                        (bridge[..., :45] - anchor[..., :45])
                        / sigma_physical
                    )[target_mask].astype(np.float32, copy=True)
                )
            quantile_reference = build_bridge_quantile_reference_from_standardized_corrections(
                np.concatenate(standardized_pieces, axis=0),
                sigma_physical,
                stack_train_distill_manifest_sha256=child["children"][
                    "stack_train_distill"
                ]["content_hash"],
            )
            standardized_pieces.clear()
        finally:
            ledger.release(quantile_reservation)
        recipe_kwargs = {
            "rho": "0.100",
            "r0_checkpoint_sha256": r0_sha,
            "hlt_source_sha256": spec["sources"]["stack_train"]["hlt_npz"]["sha256"],
            "offline_source_sha256": spec["sources"]["stack_train"]["offline_npz"]["sha256"],
            "split_manifest_sha256": child["children"]["stack_train_distill"]["content_hash"],
            "target_schema_sha256": spec["target_schema_sha256"],
            "preprocessing_sha256": spec["preprocessing_sha256"],
            "event_order_sha256": stage_report["event_order_sha256"],
            "audit_summary": {
                "parent_events": hlt.n_events,
                "fit_events": int(indices.size),
                "fit_child": "stack_train_distill",
                "one_open_verified": combined_report[
                    "all_persistent_npz_open_counts_equal_one"
                ],
            },
        }
        physical_recipe = build_bridge_recipe(
            channel_policy=BRIDGE_CHANNEL_PHYSICAL45, **recipe_kwargs
        )
        all50_recipe = build_bridge_recipe(
            channel_policy=BRIDGE_CHANNEL_ALL50, **recipe_kwargs
        )
        # A5/A5S are diagnostics, but they are part of the locked pilot.  Fit
        # their compact absolute-output bounds here from the exact same
        # stack_train_distill bridge definition.  Only the 45-channel quantile
        # artifact leaves allocation RAM; no generated field tensor does.
        absolute_reservation = ledger.reserve(
            owner="rank0",
            role="exact_stack_train_distill_absolute_bridge_quantiles",
            expected_bytes=working_bytes,
            category="derived",
        )
        ledger.commit(absolute_reservation, measured_bytes=working_bytes)

        def absolute_batches():
            for start in range(0, indices.size, int(batch_size)):
                selected = indices[start : start + int(batch_size)]
                hlt_batch = hlt.read_indices(selected, names=("tokens", "mask"))
                offline_batch = offline.read_indices(selected, names=("tokens", "mask"))
                truth, target_mask, _, _, _ = compute_local_particle_residual_fields(
                    hlt_batch["tokens"],
                    hlt_batch["mask"],
                    offline_batch["tokens"],
                    offline_batch["mask"],
                    radii=DEFAULT_LOCAL_RESIDUAL_RADII,
                )
                anchor, _ = r0.predict_numpy(hlt_batch["tokens"], hlt_batch["mask"])
                yield (
                    virtual_bridge(
                        anchor,
                        truth,
                        target_mask,
                        rho="0.100",
                        channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
                    ),
                    target_mask,
                )

        try:
            absolute_scaler = fit_absolute_output_scaler(
                absolute_batches(),
                source_manifest_sha256=spec["parent_manifest"]["sha256"],
                bridge_recipe_sha256=physical_recipe["content_hash"],
                epsilon=physical_scalers.epsilon,
            )
        finally:
            ledger.release(absolute_reservation)
        output.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "bridge_recipe_physical45.json": physical_recipe,
            "bridge_recipe_all50.json": all50_recipe,
            "bridge_scalers_physical45.json": physical_scalers.to_artifact(),
            "bridge_scalers_all50.json": all50_scalers.to_artifact(),
            "bridge_absolute_scaler_physical45.json": absolute_scaler,
            "bridge_quantile_reference_physical45.json": quantile_reference,
        }
        for name, artifact in artifacts.items():
            write_immutable_json(output / name, artifact)
        metrics = with_content_hash(
            {
                "contract": "prediction_anchored_bridge_bound_input_preparation_v1",
                "execution_spec_sha256": spec["content_hash"],
                "r0_checkpoint_sha256": r0_sha,
                "fit_child": "stack_train_distill",
                "fit_child_sha256": child["children"]["stack_train_distill"]["content_hash"],
                "fit_events": int(indices.size),
                "raw_stage": stage_report,
                "one_open_per_compressed_source": True,
                "persistent_field_tensors_written": False,
                "ram_peak_reserved_bytes": ledger.snapshot()["peak_reserved_bytes"],
                "artifact_hashes": {
                    name: value["content_hash"] for name, value in artifacts.items()
                },
            }
        )
        write_immutable_json(output / "step2_audit_metrics.json", metrics)
        names = sorted(path.name for path in output.iterdir())
        expected_names = sorted([*artifacts, "step2_audit_metrics.json"])
        if names != expected_names:
            raise RuntimeError(f"bridge input preparation wrote unexpected files: {names}")
        return {
            "ok": True,
            "output_dir": str(output),
            "persistent_artifacts": names,
            "fit_child": "stack_train_distill",
            "fit_events": int(indices.size),
            "one_open_per_compressed_source": True,
            "persistent_dense_fields_written": False,
        }
    finally:
        ledger.cleanup()


__all__ = [
    "run_streamed_r0_from_execution_spec",
    "prepare_bridge_inputs_from_execution_spec",
]
