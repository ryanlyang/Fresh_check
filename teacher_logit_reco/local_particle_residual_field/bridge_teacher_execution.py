"""Numerical teacher binding and compact target-logit caching for Stage B5."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import resolve_device

from .bridge import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    validate_bridge_recipe,
    virtual_bridge,
)
from .bridge_consumer import (
    T10_ALL50_CLEAN,
    T10_CLEAN,
    T10_ROBUST,
    TPRED,
    build_consumer_tensor_batch,
)
from .bridge_contracts import (
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    write_immutable_json,
)
from .bridge_evaluation import (
    ALL50_TEACHER_NAMESPACE,
    ALTERNATE_TEACHER_NAMESPACE,
    N3_F0_TEACHER_NAMESPACE,
    PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    PRIMARY_TEACHER_NAMESPACE,
    build_teacher_binding,
    validate_teacher_binding,
)
from .bridge_execution import (
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)
from .bridge_logits import build_live_teacher_config, cache_bound_teacher_logits
from .bridge_numerical import _paths_from_source, _verify_staged_source_binding
from .bridge_ram import AllocationNpzStager, AllocationRamLedger, FrozenR0Runner
from .bridge_splits import PREDICTION_ANCHORED_SPLIT_CONTRACT
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, compute_local_particle_residual_fields


PREDICTION_ANCHORED_TEACHER_BINDING_SET_CONTRACT = (
    "prediction_anchored_teacher_binding_set_v1"
)
PREDICTION_ANCHORED_TEACHER_CACHE_EXECUTION_CONTRACT = (
    "prediction_anchored_teacher_cache_execution_v1"
)


def _model_logits(model: Any, batch: Mapping[str, Any]) -> Any:
    return model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        indices=batch.get("indices"),
        oracle_fields=batch.get("oracle_fields"),
    )


def _load_execution_inputs(
    execution_spec_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = load_hashed_json(
        execution_spec_path,
        expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    )
    validate_prediction_anchored_execution_spec(spec, verify_file_hashes=False)
    child = load_hashed_json(
        spec["child_manifest"]["path"],
        expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT,
    )
    if child["content_hash"] != spec["child_manifest"]["content_hash"]:
        raise ValueError("execution spec child manifest binding changed")
    return spec, child


def _teacher_checkpoint(
    *,
    run_id: str,
    aggregate: Mapping[str, Any],
    consumer_root: Path,
) -> Path:
    validate_content_hash(
        aggregate,
        expected_contract=PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
    )
    if aggregate.get("run_id") != run_id:
        raise ValueError("teacher aggregate run ID changed")
    checkpoint = consumer_root / run_id / "median_weights.pt"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"published median teacher is absent or unsafe: {checkpoint}")
    if sha256_file(checkpoint) != aggregate.get("median_checkpoint_sha256"):
        raise ValueError(
            "published median teacher is not the exact replica selected on model_val_select"
        )
    return checkpoint


def bind_teacher_set_from_execution_spec(
    execution_spec_path: str | Path,
    *,
    selected_consumer_path: str | Path,
    physical45_recipe_path: str | Path,
    all50_recipe_path: str | Path,
    all50_scaler_path: str | Path,
    consumer_evaluation_root: str | Path,
    consumer_publication_root: str | Path,
    output_dir: str | Path,
    include_eligible_alternate: bool = True,
) -> dict[str, Any]:
    """Create primary/all50 and, when valid, alternate bindings from B3/B4 outputs."""

    _, child = _load_execution_inputs(execution_spec_path)
    selected = load_hashed_json(
        selected_consumer_path,
        expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    )
    if selected.get("status") != "CONFIRMED_LOCKED":
        raise PermissionError("Stage B5 requires the confirmed selected consumer")
    physical_recipe = load_hashed_json(physical45_recipe_path)
    all50_recipe = load_hashed_json(all50_recipe_path)
    validate_bridge_recipe(physical_recipe)
    validate_bridge_recipe(all50_recipe)
    if physical_recipe.get("channel_policy") != BRIDGE_CHANNEL_PHYSICAL45:
        raise ValueError("primary recipe is not physical45")
    if all50_recipe.get("channel_policy") != BRIDGE_CHANNEL_ALL50:
        raise ValueError("all50 recipe is not all50")
    all50_scaler = load_hashed_json(all50_scaler_path)
    evidence_root = Path(consumer_evaluation_root)
    publication_root = Path(consumer_publication_root)
    validation_hashes = {
        "model_val_select": child["children"]["model_val_select"]["content_hash"],
        "stack_val_consumer": child["children"]["stack_val_consumer"]["content_hash"],
    }

    selected_run = str(selected["selected_consumer_recipe"])
    if selected_run not in {T10_CLEAN, T10_ROBUST}:
        raise ValueError("primary selection is not a physical45 T10 recipe")
    primary_aggregate = selected["recipe_aggregate_metrics"]
    primary_checkpoint = Path(str(selected.get("checkpoint_path", "")))
    if primary_checkpoint.is_symlink() or not primary_checkpoint.is_file():
        raise FileNotFoundError("selected primary checkpoint is absent or unsafe")
    # Re-resolve through the publication root as an additional low-storage
    # campaign invariant; B4 may not point at a transient allocation path.
    expected_primary = _teacher_checkpoint(
        run_id=selected_run,
        aggregate=primary_aggregate,
        consumer_root=publication_root,
    )
    if primary_checkpoint.resolve() != expected_primary.resolve():
        raise ValueError("selected primary checkpoint is not the retained median publication")

    all50_aggregate = load_hashed_json(
        evidence_root / T10_ALL50_CLEAN / "selection_aggregate.json",
        expected_contract=PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
    )
    all50_checkpoint = _teacher_checkpoint(
        run_id=T10_ALL50_CLEAN,
        aggregate=all50_aggregate,
        consumer_root=publication_root,
    )
    bindings: dict[str, dict[str, Any]] = {
        "primary": build_teacher_binding(
            binding_kind="primary",
            run_id=selected_run,
            aggregate=primary_aggregate,
            checkpoint_path=str(primary_checkpoint),
            checkpoint_sha256=sha256_file(primary_checkpoint),
            channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
            validation_manifest_hashes=validation_hashes,
            target_cache_namespace=PRIMARY_TEACHER_NAMESPACE,
            bridge_recipe_sha256=physical_recipe["content_hash"],
            primary_selection=selected,
        ),
        "all50": build_teacher_binding(
            binding_kind="all50",
            run_id=T10_ALL50_CLEAN,
            aggregate=all50_aggregate,
            checkpoint_path=str(all50_checkpoint),
            checkpoint_sha256=sha256_file(all50_checkpoint),
            channel_policy=BRIDGE_CHANNEL_ALL50,
            validation_manifest_hashes=validation_hashes,
            target_cache_namespace=ALL50_TEACHER_NAMESPACE,
            bridge_recipe_sha256=all50_recipe["content_hash"],
            all50_scaler_artifact=all50_scaler,
        ),
    }

    alternate_status = "NOT_REQUESTED"
    alternate_run = T10_ROBUST if selected_run == T10_CLEAN else T10_CLEAN
    if include_eligible_alternate:
        alternate_aggregate = load_hashed_json(
            evidence_root / alternate_run / "selection_aggregate.json",
            expected_contract=PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
        )
        if bool(alternate_aggregate.get("eligible")):
            alternate_checkpoint = _teacher_checkpoint(
                run_id=alternate_run,
                aggregate=alternate_aggregate,
                consumer_root=publication_root,
            )
            bindings["alternate"] = build_teacher_binding(
                binding_kind="alternate",
                run_id=alternate_run,
                aggregate=alternate_aggregate,
                checkpoint_path=str(alternate_checkpoint),
                checkpoint_sha256=sha256_file(alternate_checkpoint),
                channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
                validation_manifest_hashes=validation_hashes,
                target_cache_namespace=ALTERNATE_TEACHER_NAMESPACE,
                bridge_recipe_sha256=physical_recipe["content_hash"],
            )
            alternate_status = "BOUND_ELIGIBLE"
        else:
            alternate_status = "SKIPPED_INVALID_PARENT"

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"teacher binding directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for kind, binding in bindings.items():
        write_immutable_json(output / f"{kind}.json", binding)
    return {
        "ok": True,
        "contract": PREDICTION_ANCHORED_TEACHER_BINDING_SET_CONTRACT,
        "selected_consumer_sha256": selected["content_hash"],
        "binding_paths": {
            kind: str(output / f"{kind}.json") for kind in bindings
        },
        "binding_hashes": {
            kind: binding["content_hash"] for kind, binding in bindings.items()
        },
        "alternate_run_id": alternate_run,
        "alternate_status": alternate_status,
        "bindings_created_before_cache": True,
    }


def _event_ids(source: Any, indices: np.ndarray) -> list[str]:
    raw = source.read_indices(
        indices, names=("labels", "jet_file_indices", "jet_entries")
    )
    files = [str(value) for value in source.manifest["jet_files"]]
    return [
        f"{files[int(file_index)]}\0{int(entry)}\0{int(label)}"
        for file_index, entry, label in zip(
            raw["jet_file_indices"], raw["jet_entries"], raw["labels"]
        )
    ]


class _BoundTeacherForward:
    def __init__(
        self,
        *,
        model: Any,
        checkpoint_sha256: str,
        channel_policy: str,
        device: Any,
    ) -> None:
        self.model = model
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.channel_policy = str(channel_policy)
        self.device = device

    def __call__(self, batch: Mapping[str, Any], condition: str) -> Any:
        if condition == "f0":
            fields = batch["f0"]
        elif condition == "bridge_0.100":
            fields = virtual_bridge(
                batch["f0"],
                batch["f_true"],
                batch["mask"],
                rho="0.100",
                channel_policy=self.channel_policy,
            )
        else:
            raise ValueError(f"unsupported target-logit field condition: {condition!r}")
        tensor_batch = build_consumer_tensor_batch(
            tokens=batch["tokens"],
            mask=batch["mask"],
            labels=batch["labels"],
            f0=fields,
            f_true=fields,
            run_id=TPRED,
            device=self.device,
        )
        return _model_logits(self.model, tensor_batch)


def cache_teacher_logits_from_execution_spec(
    execution_spec_path: str | Path,
    *,
    binding_path: str | Path,
    namespace: str,
    selected_consumer_path: str | Path | None,
    r0_checkpoint_path: str | Path,
    r0_registration_path: str | Path,
    bridge_recipe_path: str | Path,
    output_root: str | Path,
    ram_root: str | Path,
    allocation_id: str | None = None,
    device: str = "auto",
    batch_size: int = 512,
    shard_size: int = 8192,
    capacity_bytes: int | None = None,
    allow_unverified_test_root: bool = False,
    model_loader: Callable[[str | Path, Any], tuple[Any, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Forward one exact bound teacher on stack_train_distill and cache logits only."""

    import torch
    from .fusion import load_local_residual_field_tagger_from_checkpoint

    if int(batch_size) <= 0 or int(shard_size) <= 0:
        raise ValueError("batch_size and shard_size must be positive")
    spec, child = _load_execution_inputs(execution_spec_path)
    binding = load_hashed_json(binding_path)
    primary_required = binding.get("binding_kind") == "primary"
    if primary_required and not selected_consumer_path:
        raise PermissionError("primary/N3 target caching requires selected_bridge_consumer.json")
    selected = (
        load_hashed_json(
            selected_consumer_path,
            expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
        )
        if selected_consumer_path
        else None
    )
    validate_teacher_binding(
        binding,
        expected_kind=str(binding["binding_kind"]),
        primary_selection=selected,
    )
    expected_namespace = (
        {PRIMARY_TEACHER_NAMESPACE, N3_F0_TEACHER_NAMESPACE}
        if primary_required
        else {str(binding["target_cache_namespace"])}
    )
    if str(namespace) not in expected_namespace:
        raise ValueError("requested namespace cannot use this teacher binding")
    recipe = load_hashed_json(bridge_recipe_path)
    validate_bridge_recipe(recipe)
    if recipe["content_hash"] != binding.get("bridge_recipe_sha256"):
        raise ValueError("teacher binding and bridge recipe differ")
    if recipe.get("channel_policy") != binding.get("channel_policy"):
        raise ValueError("teacher binding and bridge channel policy differ")

    r0_path = Path(r0_checkpoint_path)
    r0_sha = sha256_file(r0_path)
    registration = load_hashed_json(r0_registration_path)
    if registration.get("checkpoint_sha256") != r0_sha:
        raise ValueError("R0 registration/checkpoint binding changed")
    if registration.get("split_manifest") != child["content_hash"]:
        raise ValueError("R0 registration uses another child manifest")
    if recipe.get("parent_hashes", {}).get("r0_checkpoint_sha256") != r0_sha:
        raise ValueError("bridge recipe uses another R0 checkpoint")

    torch_device = resolve_device(str(device))
    loader = model_loader or (
        lambda path, selected_device: load_local_residual_field_tagger_from_checkpoint(
            path, device=selected_device
        )
    )
    model, _ = loader(binding["checkpoint_path"], torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    callback = _BoundTeacherForward(
        model=model,
        checkpoint_sha256=binding["checkpoint_sha256"],
        channel_policy=binding["channel_policy"],
        device=torch_device,
    )
    live = build_live_teacher_config(binding, primary_selection=selected)
    indices = np.asarray(
        child["children"]["stack_train_distill"]["parent_row_indices"],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise ValueError("stack_train_distill is empty")

    allocation = str(allocation_id or os.environ.get("SLURM_JOB_ID", "local_teacher_cache"))
    ledger = AllocationRamLedger(
        ram_root,
        allocation_id=allocation,
        capacity_bytes=capacity_bytes,
        allow_unverified_test_root=bool(allow_unverified_test_root),
    )
    try:
        stager = AllocationNpzStager(ledger, rank=0, world_size=1)
        staged, combined = stager.stage_named_pairs(
            {"stack_train": _paths_from_source(spec["sources"]["stack_train"])},
            shard_size=int(shard_size),
        )
        hlt, offline, stage_report = staged["stack_train"]
        _verify_staged_source_binding(
            split="stack_train",
            source=spec["sources"]["stack_train"],
            report=stage_report,
        )
        if not combined["all_persistent_npz_open_counts_equal_one"]:
            raise RuntimeError("teacher cache allocation violated the one-open contract")
        r0 = FrozenR0Runner(r0_path, device=torch_device)

        def batches():
            for start in range(0, indices.size, int(batch_size)):
                chosen = indices[start : start + int(batch_size)]
                hlt_batch = hlt.read_indices(chosen, names=("tokens", "mask", "labels"))
                offline_batch = offline.read_indices(chosen, names=("tokens", "mask"))
                truth, target_mask, _, _, _ = compute_local_particle_residual_fields(
                    hlt_batch["tokens"],
                    hlt_batch["mask"],
                    offline_batch["tokens"],
                    offline_batch["mask"],
                    radii=DEFAULT_LOCAL_RESIDUAL_RADII,
                )
                if not np.array_equal(target_mask, hlt_batch["mask"]):
                    raise ValueError("stack_train_distill target mask changed")
                anchor, _ = r0.predict_numpy(
                    hlt_batch["tokens"], hlt_batch["mask"]
                )
                yield {
                    "tokens": hlt_batch["tokens"],
                    "mask": target_mask,
                    "labels": hlt_batch["labels"],
                    "event_ids": _event_ids(hlt, chosen),
                    "f0": anchor,
                    "f_true": truth,
                }

        with torch.no_grad():
            manifest = cache_bound_teacher_logits(
                binding=binding,
                checkpoint_path=binding["checkpoint_path"],
                batches=batches(),
                forward_fn=callback,
                stack_train_distill_manifest_sha256=child["children"][
                    "stack_train_distill"
                ]["content_hash"],
                class_order=spec["class_names"],
                temperature_convention="raw_logits__temperature_applied_in_kd_loss",
                output_root=output_root,
                live_teacher_config=live,
                namespace=str(namespace),
                primary_selection=selected,
            )
        return {
            "ok": True,
            "contract": PREDICTION_ANCHORED_TEACHER_CACHE_EXECUTION_CONTRACT,
            "cache_namespace": str(namespace),
            "cache_manifest_sha256": manifest["content_hash"],
            "teacher_binding_sha256": binding["content_hash"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "stack_train_distill_manifest_sha256": child["children"][
                "stack_train_distill"
            ]["content_hash"],
            "event_count": int(indices.size),
            "one_open_per_compressed_source": True,
            "persistent_dense_fields_written": False,
            "persistent_arrays": ["logits", "labels", "event_identity_hashes"],
            "ram_peak_reserved_bytes": ledger.snapshot()["peak_reserved_bytes"],
        }
    finally:
        ledger.cleanup()


__all__ = [
    "PREDICTION_ANCHORED_TEACHER_BINDING_SET_CONTRACT",
    "PREDICTION_ANCHORED_TEACHER_CACHE_EXECUTION_CONTRACT",
    "bind_teacher_set_from_execution_spec",
    "cache_teacher_logits_from_execution_spec",
]
