"""Matched-compute bridge-consumer training contracts.

The campaign keeps continuation state in allocation RAM and publishes only one
ordered-median weights payload after all three paired replicas have metrics.
This module is independent of Weaver at import time so its lineage, sampling,
and publication invariants remain CPU-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .bridge import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    physical_loss_groups,
    virtual_bridge,
)
from .bridge_campaign import PAIRED_SEED_IDS, record_registry_measurements
from .bridge_contracts import (
    build_total_sized_publication,
    canonical_sha256,
    immutable_json_bytes,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)


PREDICTION_ANCHORED_CONSUMER_CONFIG_CONTRACT = "prediction_anchored_consumer_config_v1"
PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT = "prediction_anchored_consumer_run_v1"
PREDICTION_ANCHORED_CONTINUATION_PLAN_CONTRACT = "prediction_anchored_continuation_plan_v1"
PREDICTION_ANCHORED_BRANCH_LINEAGE_CONTRACT = "prediction_anchored_tpred_branch_lineage_v1"
PREDICTION_ANCHORED_ROBUST_SAMPLER_CONTRACT = "prediction_anchored_robust_bridge_sampler_v1"
PREDICTION_ANCHORED_CORRUPTION_SCALE_CONTRACT = "prediction_anchored_corruption_scale_v1"
PREDICTION_ANCHORED_REPLICA_AGGREGATE_CONTRACT = "prediction_anchored_replica_aggregate_v1"
PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT = "prediction_anchored_consumer_publication_v1"
PREDICTION_ANCHORED_MEASURED_UPSTREAM_CONTRACT = "prediction_anchored_measured_upstream_v1"
PREDICTION_ANCHORED_CONSUMER_REPLICA_MANIFEST_CONTRACT = (
    "prediction_anchored_consumer_replica_manifest_v1"
)
PREDICTION_ANCHORED_BRANCH_EXECUTION_CONTRACT = (
    "prediction_anchored_tpred_branch_execution_v1"
)

A0_C250 = "A0_C250"
A0_C250_LONG = "A0_C250_LONG"
A0_S500 = "A0_S500"
TPRED = "Tpred"
TPRED_CONTINUE = "Tpred_continue"
T10_CLEAN = "T10_clean"
T10_ROBUST = "T10_robust"
T10_ALL50_CLEAN = "T10_all50_clean"
STEP3_RUN_IDS = (
    A0_C250,
    A0_C250_LONG,
    A0_S500,
    TPRED,
    TPRED_CONTINUE,
    T10_CLEAN,
    T10_ROBUST,
    T10_ALL50_CLEAN,
)
TPRED_BRANCH_RUN_IDS = (TPRED_CONTINUE, T10_CLEAN, T10_ROBUST, T10_ALL50_CLEAN)

FIELD_CONDITION_A0 = "hlt_only"
FIELD_CONDITION_F0 = "f0"
FIELD_CONDITION_BRIDGE = "bridge_0.100"
FIELD_CONDITION_ROBUST = "robust_60_20_15_5"

ROBUST_EXACT_BRIDGE = "exact_bridge_0.100"
ROBUST_EXACT_F0 = "exact_f0"
ROBUST_UNIFORM_RESPONSE = "uniform_bridge_0.000_0.100"
ROBUST_LIGHT_CORRUPTION = "light_corruption_bridge_0.100"
ROBUST_CONDITIONS = (
    ROBUST_EXACT_BRIDGE,
    ROBUST_EXACT_F0,
    ROBUST_UNIFORM_RESPONSE,
    ROBUST_LIGHT_CORRUPTION,
)
ROBUST_COUNTS_PER_100 = {
    ROBUST_EXACT_BRIDGE: 60,
    ROBUST_EXACT_F0: 20,
    ROBUST_UNIFORM_RESPONSE: 15,
    ROBUST_LIGHT_CORRUPTION: 5,
}


@dataclass(frozen=True)
class ConsumerRunSpec:
    run_id: str
    train_split: str
    parent: str
    field_condition: str
    channel_policy: str | None
    budget_kind: str
    unique_jet_count: int
    paired_seed_ids: tuple[int, ...] = PAIRED_SEED_IDS

    def __post_init__(self) -> None:
        if self.run_id not in STEP3_RUN_IDS:
            raise ValueError(f"unknown Step 3 consumer run {self.run_id!r}")
        if tuple(self.paired_seed_ids) != tuple(PAIRED_SEED_IDS):
            raise ValueError("scientific consumer runs require paired seeds 101/202/303")
        if self.channel_policy not in {None, BRIDGE_CHANNEL_PHYSICAL45, BRIDGE_CHANNEL_ALL50}:
            raise ValueError("invalid consumer channel policy")
        if int(self.unique_jet_count) not in {250_000, 500_000}:
            raise ValueError("consumer fairness rows must use exactly 250k or 500k unique jets")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT,
            **asdict(self),
            "paired_seed_ids": list(self.paired_seed_ids),
        }


def consumer_run_specs() -> dict[str, ConsumerRunSpec]:
    specs = (
        ConsumerRunSpec(A0_C250, "stack_train_consumer", "reference_hlt_part", FIELD_CONDITION_A0, None, "baseline", 250_000),
        ConsumerRunSpec(A0_C250_LONG, "stack_train_consumer", A0_C250, FIELD_CONDITION_A0, None, "bridge_continuation", 250_000),
        ConsumerRunSpec(A0_S500, "stack_train_union", "reference_hlt_part", FIELD_CONDITION_A0, None, "baseline", 500_000),
        ConsumerRunSpec(TPRED, "stack_train_consumer", "reference_hlt_part", FIELD_CONDITION_F0, BRIDGE_CHANNEL_PHYSICAL45, "baseline", 250_000),
        ConsumerRunSpec(TPRED_CONTINUE, "stack_train_consumer", TPRED, FIELD_CONDITION_F0, BRIDGE_CHANNEL_PHYSICAL45, "bridge_continuation", 250_000),
        ConsumerRunSpec(T10_CLEAN, "stack_train_consumer", TPRED, FIELD_CONDITION_BRIDGE, BRIDGE_CHANNEL_PHYSICAL45, "bridge_continuation", 250_000),
        ConsumerRunSpec(T10_ROBUST, "stack_train_consumer", TPRED, FIELD_CONDITION_ROBUST, BRIDGE_CHANNEL_PHYSICAL45, "bridge_continuation", 250_000),
        ConsumerRunSpec(T10_ALL50_CLEAN, "stack_train_consumer", TPRED, FIELD_CONDITION_BRIDGE, BRIDGE_CHANNEL_ALL50, "bridge_continuation", 250_000),
    )
    return {spec.run_id: spec for spec in specs}


def build_consumer_replica_manifest(config: "ConsumerCampaignConfig") -> dict[str, Any]:
    """Materialize the locked eight-row, three-paired-replica Step 3 design."""

    rows: list[dict[str, Any]] = []
    for run_id in STEP3_RUN_IDS:
        spec = consumer_run_specs()[run_id]
        steps = (
            config.bridge_finetune_steps
            if spec.budget_kind == "bridge_continuation"
            else config.baseline_steps
        )
        rows.append(
            {
                **spec.to_dict(),
                "optimizer_steps": int(steps),
                "replicas": [
                    {
                        "seed_id": int(seed),
                        "replica_id": f"{run_id}__seed{seed}",
                        "persistent_metrics": True,
                        "persistent_weights": "ordered_median_only_after_aggregate",
                    }
                    for seed in PAIRED_SEED_IDS
                ],
            }
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_REPLICA_MANIFEST_CONTRACT,
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "configuration_count": len(rows),
            "replica_count": len(rows) * len(PAIRED_SEED_IDS),
            "campaign_config_sha256": config.to_artifact()["content_hash"],
            "runs": rows,
            "tpred_branch_run_ids": list(TPRED_BRANCH_RUN_IDS),
            "tpred_branch_state": "same_allocation_ram_snapshot",
            "nonmedian_weights_persisted": False,
            "optimizer_scheduler_state_persisted": False,
            "generated_dense_fields_persisted": False,
        }
    )


@dataclass(frozen=True)
class ConsumerCampaignConfig:
    baseline_steps: int
    bridge_finetune_steps: int
    batch_size: int
    evaluation_interval_steps: int
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    model_size: str = "base"
    input_field_dim: int = 50
    reference_input_dim: int = 17
    paired_seed_ids: tuple[int, ...] = PAIRED_SEED_IDS

    def __post_init__(self) -> None:
        for name in ("baseline_steps", "bridge_finetune_steps", "batch_size", "evaluation_interval_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.input_field_dim) != 50:
            raise ValueError("the bridge consumer must widen by exactly 50 residual fields")
        if tuple(self.paired_seed_ids) != tuple(PAIRED_SEED_IDS):
            raise ValueError("scientific consumer runs require paired seeds 101/202/303")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0 or float(self.grad_clip_norm) < 0:
            raise ValueError("invalid optimizer configuration")

    @property
    def continuation_evaluation_steps(self) -> tuple[int, ...]:
        values = list(range(self.evaluation_interval_steps, self.bridge_finetune_steps + 1, self.evaluation_interval_steps))
        if not values or values[-1] != self.bridge_finetune_steps:
            values.append(self.bridge_finetune_steps)
        return tuple(values)

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_CONSUMER_CONFIG_CONTRACT,
                **asdict(self),
                "paired_seed_ids": list(self.paired_seed_ids),
                "continuation_evaluation_steps": list(self.continuation_evaluation_steps),
                "run_specs": {key: value.to_dict() for key, value in consumer_run_specs().items()},
            }
        )


@dataclass(frozen=True)
class ContinuationBatchPlan:
    seed_id: int
    n_examples: int
    batch_size: int
    steps: int
    batches: tuple[tuple[int, ...], ...]
    evaluation_steps: tuple[int, ...]
    content_hash: str

    def __post_init__(self) -> None:
        if int(self.seed_id) not in PAIRED_SEED_IDS:
            raise ValueError("continuation plan seed is not a paired scientific seed")
        if len(self.batches) != int(self.steps):
            raise ValueError("continuation plan does not contain the exact step budget")
        if any(not batch for batch in self.batches):
            raise ValueError("continuation plan contains an empty batch")
        if any(index < 0 or index >= self.n_examples for batch in self.batches for index in batch):
            raise ValueError("continuation plan index is outside the training child")
        if any(step <= 0 or step > self.steps for step in self.evaluation_steps):
            raise ValueError("continuation evaluation step is outside the budget")
        payload = self._payload()
        if canonical_sha256(payload) != self.content_hash:
            raise ValueError("continuation batch-plan hash mismatch")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract": PREDICTION_ANCHORED_CONTINUATION_PLAN_CONTRACT,
            "seed_id": int(self.seed_id),
            "n_examples": int(self.n_examples),
            "batch_size": int(self.batch_size),
            "steps": int(self.steps),
            "batches": [list(batch) for batch in self.batches],
            "evaluation_steps": list(self.evaluation_steps),
        }

    def to_artifact(self) -> dict[str, Any]:
        return {**self._payload(), "content_hash": self.content_hash}


def build_continuation_batch_plan(
    *,
    seed_id: int,
    n_examples: int,
    batch_size: int,
    steps: int,
    evaluation_steps: Sequence[int],
) -> ContinuationBatchPlan:
    if int(seed_id) not in PAIRED_SEED_IDS:
        raise ValueError("seed_id must be one of 101/202/303")
    if min(int(n_examples), int(batch_size), int(steps)) <= 0:
        raise ValueError("n_examples, batch_size, and steps must be positive")
    rng = np.random.default_rng(int(seed_id))
    batches: list[tuple[int, ...]] = []
    permutation = np.empty(0, dtype=np.int64)
    cursor = 0
    while len(batches) < int(steps):
        if cursor >= permutation.size:
            permutation = rng.permutation(int(n_examples)).astype(np.int64)
            cursor = 0
        take = min(int(batch_size), int(n_examples))
        if cursor + take <= permutation.size:
            batch = permutation[cursor : cursor + take]
            cursor += take
        else:
            first = permutation[cursor:]
            permutation = rng.permutation(int(n_examples)).astype(np.int64)
            needed = take - first.size
            batch = np.concatenate([first, permutation[:needed]])
            cursor = needed
        batches.append(tuple(int(value) for value in batch.tolist()))
    payload = {
        "contract": PREDICTION_ANCHORED_CONTINUATION_PLAN_CONTRACT,
        "seed_id": int(seed_id),
        "n_examples": int(n_examples),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "batches": [list(batch) for batch in batches],
        "evaluation_steps": sorted(set(int(value) for value in evaluation_steps)),
    }
    return ContinuationBatchPlan(
        seed_id=int(seed_id),
        n_examples=int(n_examples),
        batch_size=int(batch_size),
        steps=int(steps),
        batches=tuple(batches),
        evaluation_steps=tuple(payload["evaluation_steps"]),
        content_hash=canonical_sha256(payload),
    )


def _state_bytes(payload: Any) -> bytes:
    import torch

    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _state_hash(payload: Any) -> str:
    return hashlib.sha256(_state_bytes(payload)).hexdigest()


def _torch_load_bytes(encoded: bytes) -> Any:
    import torch

    return torch.load(BytesIO(encoded), map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class TrainingLineageSnapshot:
    encoded_payload: bytes = field(repr=False)
    content_hash: str
    model_state_hash: str
    optimizer_state_hash: str
    scheduler_state_hash: str | None
    amp_scaler_state_hash: str | None
    batch_plan_sha256: str
    evaluation_steps: tuple[int, ...]
    dropout_stream_seed: int
    robust_sampler_seed: int

    @property
    def resident_bytes(self) -> int:
        return len(self.encoded_payload)

    def audit_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_BRANCH_LINEAGE_CONTRACT,
                "snapshot_sha256": self.content_hash,
                "resident_bytes": self.resident_bytes,
                "model_state_sha256": self.model_state_hash,
                "optimizer_state_sha256": self.optimizer_state_hash,
                "scheduler_state_sha256": self.scheduler_state_hash,
                "amp_scaler_state_sha256": self.amp_scaler_state_hash,
                "batch_plan_sha256": self.batch_plan_sha256,
                "evaluation_steps": list(self.evaluation_steps),
                "dropout_stream_seed": int(self.dropout_stream_seed),
                "robust_sampler_seed": int(self.robust_sampler_seed),
                "storage": "allocation_ram_only",
                "cross_allocation_resume": False,
            }
        )


def capture_training_lineage(
    *,
    model: Any,
    optimizer: Any,
    batch_plan: ContinuationBatchPlan,
    scheduler: Any | None = None,
    amp_scaler: Any | None = None,
    dropout_stream_seed: int,
    robust_sampler_seed: int,
) -> TrainingLineageSnapshot:
    import torch

    dropout_seed = int(dropout_stream_seed)
    robust_seed = int(robust_sampler_seed)
    if not 0 <= dropout_seed < 2**32 or not 0 <= robust_seed < 2**32:
        raise ValueError("branch RNG seeds must fit unsigned 32-bit streams")
    # The recorded continuation stream is the actual stream captured below,
    # not merely a descriptive seed in the audit artifact.
    random.seed(dropout_seed)
    np.random.seed(dropout_seed)
    torch.manual_seed(dropout_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dropout_seed)
    payload = {
        "contract": PREDICTION_ANCHORED_BRANCH_LINEAGE_CONTRACT,
        "model_state_dict": deepcopy(model.state_dict()),
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": None if scheduler is None else deepcopy(scheduler.state_dict()),
        "amp_scaler_state_dict": None if amp_scaler is None else deepcopy(amp_scaler.state_dict()),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "batch_plan_sha256": batch_plan.content_hash,
        "evaluation_steps": tuple(batch_plan.evaluation_steps),
        "dropout_stream_seed": dropout_seed,
        "robust_sampler_seed": robust_seed,
    }
    encoded = _state_bytes(payload)
    return TrainingLineageSnapshot(
        encoded_payload=encoded,
        content_hash=hashlib.sha256(encoded).hexdigest(),
        model_state_hash=_state_hash(payload["model_state_dict"]),
        optimizer_state_hash=_state_hash(payload["optimizer_state_dict"]),
        scheduler_state_hash=None if scheduler is None else _state_hash(payload["scheduler_state_dict"]),
        amp_scaler_state_hash=None if amp_scaler is None else _state_hash(payload["amp_scaler_state_dict"]),
        batch_plan_sha256=batch_plan.content_hash,
        evaluation_steps=tuple(batch_plan.evaluation_steps),
        dropout_stream_seed=dropout_seed,
        robust_sampler_seed=robust_seed,
    )


def restore_training_lineage(
    snapshot: TrainingLineageSnapshot,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    amp_scaler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    import torch

    if hashlib.sha256(snapshot.encoded_payload).hexdigest() != snapshot.content_hash:
        raise ValueError("RAM lineage snapshot hash mismatch")
    payload = _torch_load_bytes(snapshot.encoded_payload)
    if payload.get("contract") != PREDICTION_ANCHORED_BRANCH_LINEAGE_CONTRACT:
        raise ValueError("RAM lineage snapshot contract mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if (scheduler is None) != (payload["scheduler_state_dict"] is None):
        raise ValueError("scheduler lineage presence mismatch")
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if (amp_scaler is None) != (payload["amp_scaler_state_dict"] is None):
        raise ValueError("AMP scaler lineage presence mismatch")
    if amp_scaler is not None:
        amp_scaler.load_state_dict(payload["amp_scaler_state_dict"])
    if restore_rng:
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    restored = {
        "model_state_sha256": _state_hash(model.state_dict()),
        "optimizer_state_sha256": _state_hash(optimizer.state_dict()),
        "scheduler_state_sha256": None if scheduler is None else _state_hash(scheduler.state_dict()),
        "amp_scaler_state_sha256": None if amp_scaler is None else _state_hash(amp_scaler.state_dict()),
        "batch_plan_sha256": payload["batch_plan_sha256"],
        "evaluation_steps": list(payload["evaluation_steps"]),
    }
    expected = {
        "model_state_sha256": snapshot.model_state_hash,
        "optimizer_state_sha256": snapshot.optimizer_state_hash,
        "scheduler_state_sha256": snapshot.scheduler_state_hash,
        "amp_scaler_state_sha256": snapshot.amp_scaler_state_hash,
    }
    for key, value in expected.items():
        if restored[key] != value:
            raise AssertionError(f"restored branch {key} differs from Tpred terminal state")
    return restored


def branch_lineage_artifact(
    snapshot: TrainingLineageSnapshot,
    *,
    branch_run_ids: Sequence[str] = TPRED_BRANCH_RUN_IDS,
) -> dict[str, Any]:
    branch_ids = tuple(str(value) for value in branch_run_ids)
    if set(branch_ids) != set(TPRED_BRANCH_RUN_IDS) or len(branch_ids) != len(TPRED_BRANCH_RUN_IDS):
        raise ValueError("Tpred must fork exactly the four declared continuation branches")
    branch = {
        run_id: {
            "parent_snapshot_sha256": snapshot.content_hash,
            "model_state_sha256": snapshot.model_state_hash,
            "optimizer_state_sha256": snapshot.optimizer_state_hash,
            "scheduler_state_sha256": snapshot.scheduler_state_hash,
            "amp_scaler_state_sha256": snapshot.amp_scaler_state_hash,
            "batch_plan_sha256": snapshot.batch_plan_sha256,
            "evaluation_steps": list(snapshot.evaluation_steps),
            "dropout_stream_seed": snapshot.dropout_stream_seed,
            "field_sampler_rng": (
                "separate_recorded_stream" if run_id == T10_ROBUST else "none"
            ),
        }
        for run_id in branch_ids
    }
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_BRANCH_LINEAGE_CONTRACT,
            "parent_run_id": TPRED,
            "parent_snapshot_sha256": snapshot.content_hash,
            "branch_run_ids": list(branch_ids),
            "branches": branch,
            "identical_model_optimizer_scheduler_batches_dropout_eval": True,
            "robust_sampler_only_declared_difference": True,
            "storage": "allocation_ram_only",
            "cross_allocation_resume": False,
        }
    )


def _normalized_source_keys(source_state: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for original, value in source_state.items():
        key = str(original)
        while key.startswith("module.") or key.startswith("model."):
            key = key.split(".", 1)[1]
        if key.startswith("part_model."):
            key = key[len("part_model.") :]
        normalized[key] = value
    return normalized


def copy_reference_hlt_weights(
    reference_state: Mapping[str, Any],
    target_model: Any,
    *,
    added_field_dim: int,
    target_prefix: str = "part_model.",
) -> dict[str, Any]:
    """Copy a reference ParT and allow only exact +50 input-axis expansions."""

    import torch

    added = int(added_field_dim)
    if added not in {0, 50}:
        raise ValueError("A0 copies without widening; Tpred widens by exactly 50")
    source = _normalized_source_keys(reference_state)
    target = target_model.state_dict()
    updated = dict(target)
    exact: list[str] = []
    widened: list[dict[str, Any]] = []
    missing: list[str] = []
    for target_key, target_value in target.items():
        if not target_key.startswith(target_prefix):
            continue
        source_key = target_key[len(target_prefix) :]
        value = source.get(source_key)
        if value is None or not hasattr(value, "shape"):
            missing.append(target_key)
            continue
        if tuple(value.shape) == tuple(target_value.shape):
            updated[target_key] = value.detach().to(device=target_value.device, dtype=target_value.dtype).clone()
            exact.append(target_key)
            continue
        source_shape = tuple(int(v) for v in value.shape)
        target_shape = tuple(int(v) for v in target_value.shape)
        differing = [axis for axis, (left, right) in enumerate(zip(source_shape, target_shape)) if left != right]
        if (
            added != 50
            or len(source_shape) != len(target_shape)
            or len(differing) != 1
            or target_shape[differing[0]] - source_shape[differing[0]] != added
        ):
            raise ValueError(
                f"reference/target tensor {target_key} differs outside the declared +50 input expansion: "
                f"{source_shape} -> {target_shape}"
            )
        axis = differing[0]
        expanded = torch.zeros_like(target_value)
        slices = [slice(None)] * len(target_shape)
        slices[axis] = slice(0, source_shape[axis])
        expanded[tuple(slices)] = value.detach().to(device=target_value.device, dtype=target_value.dtype)
        updated[target_key] = expanded
        widened.append(
            {
                "target_key": target_key,
                "input_axis": axis,
                "source_shape": list(source_shape),
                "target_shape": list(target_shape),
                "new_entries_exact_zero": True,
            }
        )
    if missing:
        raise ValueError(f"reference checkpoint is incomplete for target ParT: {missing[:10]}")
    if added == 50 and not widened:
        raise ValueError("Tpred warm start found no +50 particle-input expansion")
    if added == 0 and widened:
        raise AssertionError("A0 unexpectedly widened an input tensor")
    target_model.load_state_dict(updated, strict=True)
    # Re-read state to make bitwise copying a tested postcondition.
    loaded = target_model.state_dict()
    for key in exact:
        source_key = key[len(target_prefix) :]
        if not torch.equal(loaded[key].detach().cpu(), source[source_key].detach().cpu().to(dtype=loaded[key].dtype)):
            raise AssertionError(f"copied HLT tensor {key} is not bitwise identical")
    for item in widened:
        key = item["target_key"]
        axis = int(item["input_axis"])
        old = int(item["source_shape"][axis])
        slices = [slice(None)] * loaded[key].ndim
        slices[axis] = slice(old, None)
        if not torch.count_nonzero(loaded[key][tuple(slices)]).item() == 0:
            raise AssertionError(f"new field-input entries for {key} are not exactly zero")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT,
            "added_field_dim": added,
            "exact_copied_keys": exact,
            "widened_input_keys": widened,
            "all_other_part_parameters_bitwise_copied": True,
            "new_field_input_entries_exact_zero": bool(added == 50),
        }
    )


def verify_initial_logit_identity(
    reference_logits: Any,
    widened_logits: Any,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> dict[str, Any]:
    import torch

    left = reference_logits.detach().float()
    right = widened_logits.detach().float()
    if left.shape != right.shape:
        raise ValueError("reference and widened logits have different shapes")
    difference = (left - right).abs()
    max_abs = float(difference.max().cpu().item()) if difference.numel() else 0.0
    if not torch.allclose(left, right, atol=float(atol), rtol=float(rtol)):
        raise AssertionError(
            f"zero-column Tpred does not reproduce reference HLT logits; max_abs={max_abs}"
        )
    return {
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_difference": max_abs,
        "identity_verified": True,
    }


def build_step3_consumer_model(
    run_id: str,
    *,
    model_size: str = "base",
    num_classes: int = 10,
    part_model: Any | None = None,
) -> Any:
    """Build the production ParT wrapper for one Step 3 consumer row.

    The eight rows share the same classifier architecture.  A0 is the only
    HLT-only input contract; every predicted/bridge row receives all 50 field
    channels, while ``channel_policy`` controls how those channels are filled.
    No clipping or tagger-side field corruption is enabled because the bridge
    provider owns the declared numerical recipe.
    """

    from .tagger import (
        RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        RESIDUAL_FIELD_SOURCE_ORACLE,
        LocalResidualFieldAugmentedParT,
        LocalResidualFieldTaggerConfig,
    )
    from .targets import local_particle_residual_field_layout

    spec = consumer_run_specs().get(str(run_id))
    if spec is None:
        raise ValueError(f"unknown Step 3 run {run_id!r}")
    names, groups, _ = local_particle_residual_field_layout()
    is_a0 = spec.field_condition == FIELD_CONDITION_A0
    config = LocalResidualFieldTaggerConfig(
        num_classes=int(num_classes),
        field_dim=0 if is_a0 else 50,
        field_source=(
            RESIDUAL_FIELD_SOURCE_HLT_ONLY if is_a0 else RESIDUAL_FIELD_SOURCE_ORACLE
        ),
        model_size=str(model_size),
        residual_field_scale=1.0,
        residual_field_clip_value=0.0,
        field_dropout=0.0,
        field_names=() if is_a0 else tuple(names),
        field_groups={} if is_a0 else {key: tuple(value) for key, value in groups.items()},
    )
    return LocalResidualFieldAugmentedParT(config, part_model=part_model)


def initialize_step3_root_from_reference(
    model: Any,
    checkpoint_path: str | Path,
    *,
    run_id: str,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Strictly initialize an A0/Tpred root from one registered HLT checkpoint."""

    import torch
    from .tagger import _extract_checkpoint_state_dict

    root_ids = {A0_C250, A0_S500, TPRED}
    if str(run_id) not in root_ids:
        raise ValueError(
            "only A0_C250, A0_S500, and Tpred start from the reference HLT checkpoint"
        )
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    report = copy_reference_hlt_weights(
        _extract_checkpoint_state_dict(payload),
        model,
        added_field_dim=(50 if str(run_id) == TPRED else 0),
    )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT,
            "run_id": str(run_id),
            "reference_checkpoint": str(path.resolve()),
            "reference_checkpoint_sha256": sha256_file(path),
            "copy_report_sha256": report["content_hash"],
            "copy_report": report,
            "model_state_sha256": _state_hash(model.state_dict()),
        }
    )


def verify_paired_a0_tpred_initialization(
    *,
    a0_initialization: Mapping[str, Any],
    tpred_initialization: Mapping[str, Any],
    reference_logits: Any,
    tpred_logits: Any,
) -> dict[str, Any]:
    """Fail closed unless paired A0/Tpred share a reference and initial logits."""

    a0_hash = str(a0_initialization.get("reference_checkpoint_sha256", ""))
    tpred_hash = str(tpred_initialization.get("reference_checkpoint_sha256", ""))
    if len(a0_hash) != 64 or a0_hash != tpred_hash:
        raise ValueError("paired A0 and Tpred did not use the same reference HLT checkpoint")
    identity = verify_initial_logit_identity(reference_logits, tpred_logits)
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT,
            "reference_checkpoint_sha256": a0_hash,
            "a0_run_id": a0_initialization.get("run_id"),
            "tpred_run_id": tpred_initialization.get("run_id"),
            "same_registered_reference": True,
            "widened_initial_logit_identity": identity,
        }
    )


def build_consumer_tensor_batch(
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    f0: np.ndarray,
    f_true: np.ndarray,
    run_id: str,
    device: Any = "cpu",
    robust_sampler: "RobustBridgeSampler | None" = None,
) -> dict[str, Any]:
    """Derive HLT-only ParT inputs and the declared live field condition."""

    import torch
    from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens

    particles = np.asarray(tokens, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    targets = np.asarray(labels, dtype=np.int64)
    if particles.ndim != 3 or particles.shape[-1] != 14:
        raise ValueError("consumer tokens must have shape [N,P,14]")
    if valid.shape != particles.shape[:2] or targets.shape != (particles.shape[0],):
        raise ValueError("consumer tokens/mask/labels do not align")
    anchor = np.asarray(f0, dtype=np.float32)
    truth = np.asarray(f_true, dtype=np.float32)
    if anchor.shape != (*valid.shape, 50) or truth.shape != anchor.shape:
        raise ValueError("consumer f0/f_true must align as [N,P,50]")
    if np.any(anchor[~valid] != 0) or np.any(truth[~valid] != 0):
        raise ValueError("consumer fields require exact-zero padding")
    fields, field_diagnostics = consumer_fields_for_run(
        run_id,
        anchor,
        truth,
        valid,
        robust_sampler=robust_sampler,
    )
    inputs = build_particle_transformer_inputs_from_tokens(
        particles,
        valid,
        labels=targets,
        source_view="fixed_hlt",
    )
    output = {
        "points": torch.as_tensor(inputs.pf_points, dtype=torch.float32, device=device),
        "features": torch.as_tensor(inputs.pf_features, dtype=torch.float32, device=device),
        "lorentz_vectors": torch.as_tensor(inputs.pf_vectors, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(inputs.pf_mask, dtype=torch.bool, device=device),
        "tokens": torch.as_tensor(particles, dtype=torch.float32, device=device),
        "raw_mask": torch.as_tensor(valid, dtype=torch.bool, device=device),
        "labels": torch.as_tensor(targets, dtype=torch.long, device=device),
        "indices": None,
        "oracle_fields": (
            None
            if fields is None
            else torch.as_tensor(fields, dtype=torch.float32, device=device)
        ),
        "field_diagnostics": field_diagnostics,
        "input_availability": "hlt_plus_declared_training_field",
        "offline_tokens_enter_part_inputs": False,
        "persistent_dense_fields_written": False,
    }
    return output


def build_provider_batch_resolver(
    provider: Any,
    *,
    run_id: str,
    device: Any = "cpu",
    robust_sampler: "RobustBridgeSampler | None" = None,
) -> Callable[[tuple[int, ...]], dict[str, Any]]:
    """Adapt the Step 2 RAM provider to the exact-step consumer trainer."""

    def resolve(indices: tuple[int, ...]) -> dict[str, Any]:
        selected = np.asarray(indices, dtype=np.int64)
        hlt = provider.hlt.read_indices(selected, names=("tokens", "mask", "labels"))
        f_true, target_mask = provider.truth_for_indices(selected)
        f0, _ = provider.r0_for_indices(selected)
        if not np.array_equal(np.asarray(hlt["mask"], dtype=bool), np.asarray(target_mask, dtype=bool)):
            raise ValueError("consumer provider changed the HLT anchor mask")
        batch = build_consumer_tensor_batch(
            tokens=hlt["tokens"],
            mask=hlt["mask"],
            labels=hlt["labels"],
            f0=f0,
            f_true=f_true,
            run_id=run_id,
            device=device,
            robust_sampler=robust_sampler,
        )
        import torch

        batch["indices"] = torch.as_tensor(selected, dtype=torch.long, device=device)
        return batch

    return resolve


def consumer_cross_entropy_loss(model: Any, batch: Mapping[str, Any], step: int) -> Any:
    """The locked Step 3 objective: supervised event cross entropy only."""

    import torch.nn.functional as functional

    del step
    logits = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        indices=batch.get("indices"),
        oracle_fields=batch.get("oracle_fields"),
    )
    return functional.cross_entropy(logits, batch["labels"])


@dataclass(frozen=True)
class RobustBridgeSamplerConfig:
    exact_bridge_probability: float = 0.60
    exact_f0_probability: float = 0.20
    uniform_response_probability: float = 0.15
    light_corruption_probability: float = 0.05
    uniform_rho_low: float = 0.0
    uniform_rho_high: float = 0.10
    noise_std_factor: float = 0.10
    field_group_dropout_probability: float = 0.05
    radius_group_dropout_probability: float = 0.025
    channel_policy: str = BRIDGE_CHANNEL_PHYSICAL45

    def __post_init__(self) -> None:
        probabilities = (
            self.exact_bridge_probability,
            self.exact_f0_probability,
            self.uniform_response_probability,
            self.light_corruption_probability,
        )
        if not math.isclose(sum(float(value) for value in probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("robust consumer mixture probabilities must sum exactly to one")
        if tuple(int(round(float(value) * 100)) for value in probabilities) != (60, 20, 15, 5):
            raise ValueError("the primary robust recipe is locked to 60/20/15/5")
        if not (0 <= self.uniform_rho_low <= self.uniform_rho_high <= 0.10):
            raise ValueError("robust response rho must remain within [0, 0.10]")
        for name in ("field_group_dropout_probability", "radius_group_dropout_probability"):
            if not 0 <= float(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be in [0,1)")
        if float(self.noise_std_factor) < 0:
            raise ValueError("noise_std_factor cannot be negative")
        if self.channel_policy != BRIDGE_CHANNEL_PHYSICAL45:
            raise ValueError("the primary robust sampler is physical45 only")

    def to_dict(self) -> dict[str, Any]:
        return {"contract": PREDICTION_ANCHORED_ROBUST_SAMPLER_CONTRACT, **asdict(self)}


def fit_bridge_corruption_scale(
    batches: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Population std of ``b_0.10-f0`` on valid physical entries."""

    count = 0
    sums = np.zeros(45, dtype=np.float64)
    squares = np.zeros(45, dtype=np.float64)
    for f0, f_true, mask in batches:
        anchor = np.asarray(f0, dtype=np.float32)
        truth = np.asarray(f_true, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if anchor.shape != truth.shape or anchor.shape[-1] != 50 or valid.shape != anchor.shape[:2]:
            raise ValueError("corruption-scale batch shapes do not align")
        if not np.isfinite(anchor).all() or not np.isfinite(truth).all():
            raise ValueError("corruption-scale inputs contain non-finite values")
        correction = (np.float32(0.1) * (truth - anchor))[..., :45][valid].astype(np.float64)
        count += int(correction.shape[0])
        sums += correction.sum(axis=0, dtype=np.float64)
        squares += np.square(correction).sum(axis=0, dtype=np.float64)
    if count <= 0:
        raise ValueError("corruption-scale fit has no valid particles")
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0))
    if not parent_hashes or any(
        len(str(value)) != 64
        or any(character not in "0123456789abcdef" for character in str(value).lower())
        for value in parent_hashes.values()
    ):
        raise ValueError("corruption-scale fit requires SHA-256 parents")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CORRUPTION_SCALE_CONTRACT,
            "fit_split": "stack_train_consumer",
            "valid_particle_count": count,
            "bridge_correction_mean": mean.tolist(),
            "bridge_correction_population_std": std.tolist(),
            "parent_hashes": dict(parent_hashes),
            "persistent_dense_fields_written": False,
        }
    )


class RobustBridgeSampler:
    """Deterministic exact-cycle 60/20/15/5 field-condition sampler."""

    def __init__(
        self,
        *,
        seed: int,
        corruption_scale: Sequence[float],
        config: RobustBridgeSamplerConfig | None = None,
    ) -> None:
        self.seed = int(seed)
        self.config = config or RobustBridgeSamplerConfig()
        scale = np.asarray(corruption_scale, dtype=np.float32)
        if scale.shape != (45,) or not np.isfinite(scale).all() or np.any(scale < 0):
            raise ValueError("robust corruption scale must contain 45 finite nonnegative values")
        self.corruption_scale = scale
        self.rng = np.random.default_rng(self.seed)
        self._condition_buffer: list[str] = []
        self.draw_counts = {name: 0 for name in ROBUST_CONDITIONS}
        self.total_draws = 0
        self.group_dropout_draws = 0
        self.group_dropout_events = 0
        self.radius_dropout_draws = 0
        self.radius_dropout_events = 0

    def _refill(self) -> None:
        cycle: list[str] = []
        for name in ROBUST_CONDITIONS:
            cycle.extend([name] * ROBUST_COUNTS_PER_100[name])
        self.rng.shuffle(cycle)
        self._condition_buffer.extend(cycle)

    def draw_conditions(self, count: int) -> tuple[str, ...]:
        requested = int(count)
        if requested < 0:
            raise ValueError("condition count cannot be negative")
        while len(self._condition_buffer) < requested:
            self._refill()
        result = tuple(self._condition_buffer[:requested])
        del self._condition_buffer[:requested]
        for name in result:
            self.draw_counts[name] += 1
        self.total_draws += requested
        return result

    def sample(
        self,
        f0: np.ndarray,
        f_true: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        anchor = np.asarray(f0, dtype=np.float32)
        truth = np.asarray(f_true, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if anchor.shape != truth.shape or anchor.ndim != 3 or anchor.shape[-1] != 50:
            raise ValueError("robust sampler requires aligned [N,P,50] fields")
        if valid.shape != anchor.shape[:2] or not np.isfinite(anchor).all() or not np.isfinite(truth).all():
            raise ValueError("robust sampler mask/finite validation failed")
        if np.any(anchor[~valid] != 0) or np.any(truth[~valid] != 0):
            raise ValueError("robust sampler requires exact-zero padding")
        conditions = self.draw_conditions(anchor.shape[0])
        output = anchor.copy()
        rhos = np.zeros(anchor.shape[0], dtype=np.float32)
        exact_bridge = virtual_bridge(anchor, truth, valid, rho="0.100", channel_policy=BRIDGE_CHANNEL_PHYSICAL45)
        for event, condition in enumerate(conditions):
            event_mask = valid[event]
            if condition == ROBUST_EXACT_F0:
                continue
            if condition == ROBUST_EXACT_BRIDGE:
                output[event] = exact_bridge[event]
                rhos[event] = 0.1
                continue
            if condition == ROBUST_UNIFORM_RESPONSE:
                rho = float(self.rng.uniform(self.config.uniform_rho_low, self.config.uniform_rho_high))
                output[event, :, :45] = anchor[event, :, :45] + np.float32(rho) * (
                    truth[event, :, :45] - anchor[event, :, :45]
                )
                output[event, ~event_mask] = 0
                rhos[event] = np.float32(rho)
                continue
            if condition != ROBUST_LIGHT_CORRUPTION:
                raise AssertionError(condition)
            output[event] = exact_bridge[event]
            rhos[event] = 0.1
            noise = self.rng.normal(size=(anchor.shape[1], 45)).astype(np.float32)
            noise *= np.float32(self.config.noise_std_factor) * self.corruption_scale[None, :]
            output[event, :, :45] += noise * event_mask[:, None]
            correction = output[event, :, :45] - anchor[event, :, :45]
            for indices in physical_loss_groups().values():
                drop = bool(self.rng.random() < self.config.field_group_dropout_probability)
                self.group_dropout_draws += 1
                self.group_dropout_events += int(drop)
                if drop:
                    correction[:, indices] = 0.0
            for radius_index in range(3):
                drop = bool(self.rng.random() < self.config.radius_group_dropout_probability)
                self.radius_dropout_draws += 1
                self.radius_dropout_events += int(drop)
                if drop:
                    correction[:, 15 * radius_index : 15 * (radius_index + 1)] = 0.0
            output[event, :, :45] = anchor[event, :, :45] + correction
            output[event, ~event_mask] = 0.0
        output[..., 45:] = anchor[..., 45:]
        output[~valid] = 0.0
        diagnostics = {
            "contract": PREDICTION_ANCHORED_ROBUST_SAMPLER_CONTRACT,
            "conditions": list(conditions),
            "rhos": rhos.tolist(),
            "batch_counts": {name: conditions.count(name) for name in ROBUST_CONDITIONS},
            "cumulative_counts": dict(self.draw_counts),
            "total_draws": self.total_draws,
            "group_dropout_draws": self.group_dropout_draws,
            "group_dropout_events": self.group_dropout_events,
            "radius_dropout_draws": self.radius_dropout_draws,
            "radius_dropout_events": self.radius_dropout_events,
            "five_pass_through_exact": bool(np.array_equal(output[..., 45:], anchor[..., 45:])),
            "config": self.config.to_dict(),
        }
        return output, diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "contract": PREDICTION_ANCHORED_ROBUST_SAMPLER_CONTRACT,
            "seed": self.seed,
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "condition_buffer": list(self._condition_buffer),
            "draw_counts": dict(self.draw_counts),
            "total_draws": self.total_draws,
            "group_dropout_draws": self.group_dropout_draws,
            "group_dropout_events": self.group_dropout_events,
            "radius_dropout_draws": self.radius_dropout_draws,
            "radius_dropout_events": self.radius_dropout_events,
            "corruption_scale": self.corruption_scale.tolist(),
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("contract") != PREDICTION_ANCHORED_ROBUST_SAMPLER_CONTRACT:
            raise ValueError("robust sampler state contract mismatch")
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("robust sampler state seed mismatch")
        if not np.array_equal(np.asarray(state["corruption_scale"], dtype=np.float32), self.corruption_scale):
            raise ValueError("robust sampler corruption scale changed")
        if state.get("config") != self.config.to_dict():
            raise ValueError("robust sampler configuration changed across resume")
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        self._condition_buffer = [str(value) for value in state["condition_buffer"]]
        self.draw_counts = {str(key): int(value) for key, value in state["draw_counts"].items()}
        self.total_draws = int(state["total_draws"])
        self.group_dropout_draws = int(state["group_dropout_draws"])
        self.group_dropout_events = int(state["group_dropout_events"])
        self.radius_dropout_draws = int(state["radius_dropout_draws"])
        self.radius_dropout_events = int(state["radius_dropout_events"])


def consumer_fields_for_run(
    run_id: str,
    f0: np.ndarray,
    f_true: np.ndarray,
    mask: np.ndarray,
    *,
    robust_sampler: RobustBridgeSampler | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    spec = consumer_run_specs().get(str(run_id))
    if spec is None:
        raise ValueError(f"unknown Step 3 run {run_id!r}")
    if spec.field_condition == FIELD_CONDITION_A0:
        return None, {"field_condition": FIELD_CONDITION_A0}
    if spec.field_condition == FIELD_CONDITION_F0:
        return np.asarray(f0, dtype=np.float32).copy(), {"field_condition": FIELD_CONDITION_F0}
    if spec.field_condition == FIELD_CONDITION_BRIDGE:
        fields = virtual_bridge(
            f0,
            f_true,
            mask,
            rho="0.100",
            channel_policy=str(spec.channel_policy),
        )
        return fields, {"field_condition": FIELD_CONDITION_BRIDGE, "channel_policy": spec.channel_policy}
    if spec.field_condition == FIELD_CONDITION_ROBUST:
        if robust_sampler is None:
            raise ValueError("T10_robust requires its separate recorded sampler")
        return robust_sampler.sample(f0, f_true, mask)
    raise AssertionError(spec.field_condition)


def run_exact_step_training(
    *,
    model: Any,
    optimizer: Any,
    batch_plan: ContinuationBatchPlan,
    batch_resolver: Callable[[tuple[int, ...]], Any],
    loss_fn: Callable[[Any, Any, int], Any],
    scheduler: Any | None = None,
    amp_scaler: Any | None = None,
    evaluation_fn: Callable[[Any, int], Mapping[str, Any]] | None = None,
    grad_clip_norm: float = 0.0,
    selection_metric: str | None = None,
    selection_cross_entropy_metric: str = "cross_entropy",
    selection_state: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the exact predeclared batches/steps and evaluation opportunities."""

    import torch

    if (selection_metric is None) != (selection_state is None):
        raise ValueError("selection_metric and selection_state must be supplied together")
    if selection_metric is not None and evaluation_fn is None:
        raise ValueError("model_val_stop checkpoint selection requires evaluation_fn")
    model.train()
    initial_rng_hash = _state_hash(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    )
    losses: list[float] = []
    evaluations: list[dict[str, Any]] = []
    batch_hashes: list[str] = []
    evaluation_set = set(batch_plan.evaluation_steps)
    best_primary: float | None = None
    primary_tolerance = 0.0001
    cross_entropy_tolerance = 1.0e-6
    for step_index, indices in enumerate(batch_plan.batches, start=1):
        batch = batch_resolver(indices)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch, step_index)
        if not bool(torch.isfinite(loss).detach().cpu().item()):
            raise FloatingPointError(f"non-finite consumer loss at step {step_index}")
        if amp_scaler is None:
            loss.backward()
            if float(grad_clip_norm) > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                if not bool(torch.isfinite(norm).detach().cpu().item()):
                    raise FloatingPointError(f"non-finite consumer gradients at step {step_index}")
            optimizer.step()
        else:
            amp_scaler.scale(loss).backward()
            if float(grad_clip_norm) > 0:
                amp_scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                if not bool(torch.isfinite(norm).detach().cpu().item()):
                    raise FloatingPointError(f"non-finite consumer gradients at step {step_index}")
            amp_scaler.step(optimizer)
            amp_scaler.update()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu().item()))
        batch_hashes.append(canonical_sha256(list(indices)))
        if step_index in evaluation_set:
            if evaluation_fn is None:
                evaluations.append({"step": step_index})
            else:
                model.eval()
                with torch.no_grad():
                    metrics = dict(evaluation_fn(model, step_index))
                evaluations.append({"step": step_index, "metrics": metrics})
                if selection_metric is not None and selection_state is not None:
                    primary = _metric(metrics, selection_metric)
                    cross_entropy = _metric(metrics, selection_cross_entropy_metric)
                    best_primary = (
                        primary if best_primary is None else max(best_primary, primary)
                    )
                    current_primary = selection_state.get("primary_value")
                    current_ce = selection_state.get("cross_entropy_value")
                    current_step = selection_state.get("step")
                    current_in_pool = (
                        current_primary is not None
                        and best_primary - float(current_primary) <= primary_tolerance
                    )
                    candidate_in_pool = (
                        best_primary - float(primary) <= primary_tolerance
                    )
                    choose_candidate = candidate_in_pool and not current_in_pool
                    if candidate_in_pool and current_in_pool:
                        assert current_ce is not None and current_step is not None
                        choose_candidate = (
                            cross_entropy < float(current_ce) - cross_entropy_tolerance
                            or (
                                abs(cross_entropy - float(current_ce))
                                <= cross_entropy_tolerance
                                and step_index < int(current_step)
                            )
                        )
                    if choose_candidate:
                        selection_state.clear()
                        selection_state.update(
                            {
                                "step": int(step_index),
                                "primary_metric": str(selection_metric),
                                "primary_value": float(primary),
                                "cross_entropy_metric": str(selection_cross_entropy_metric),
                                "cross_entropy_value": float(cross_entropy),
                                "accuracy_pool_tolerance": primary_tolerance,
                                "cross_entropy_tolerance": cross_entropy_tolerance,
                                "model_state_dict": deepcopy(model.state_dict()),
                            }
                        )
                model.train()
    if len(losses) != batch_plan.steps:
        raise AssertionError("consumer training did not consume its exact optimizer-step budget")
    if tuple(item["step"] for item in evaluations) != tuple(batch_plan.evaluation_steps):
        raise AssertionError("consumer training changed evaluation opportunities")
    if selection_state is not None and "model_state_dict" not in selection_state:
        raise AssertionError("no model_val_stop checkpoint was selected")
    if selection_state is not None:
        selection_state["best_primary_value"] = float(best_primary)
    final_rng_hash = _state_hash(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    )
    return {
        "contract": PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT,
        "optimizer_steps": len(losses),
        "batch_plan_sha256": batch_plan.content_hash,
        "batch_hashes": batch_hashes,
        "evaluation_steps": [item["step"] for item in evaluations],
        "evaluations": evaluations,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "model_state_sha256": _state_hash(model.state_dict()),
        "optimizer_state_sha256": _state_hash(optimizer.state_dict()),
        "scheduler_state_sha256": None if scheduler is None else _state_hash(scheduler.state_dict()),
        "amp_scaler_state_sha256": None if amp_scaler is None else _state_hash(amp_scaler.state_dict()),
        "initial_global_rng_sha256": initial_rng_hash,
        "final_global_rng_sha256": final_rng_hash,
        "selected_checkpoint_step": (
            None if selection_state is None else int(selection_state["step"])
        ),
        "selected_checkpoint_model_state_sha256": (
            None
            if selection_state is None
            else _state_hash(selection_state["model_state_dict"])
        ),
    }


@dataclass
class ContinuationBranch:
    """In-RAM executable state for one exact Tpred continuation branch."""

    run_id: str
    model: Any
    optimizer: Any
    batch_resolver: Callable[[tuple[int, ...]], Any]
    loss_fn: Callable[[Any, Any, int], Any] = consumer_cross_entropy_loss
    scheduler: Any | None = None
    amp_scaler: Any | None = None
    evaluation_fn: Callable[[Any, int], Mapping[str, Any]] | None = None
    robust_sampler: RobustBridgeSampler | None = None
    selection_metric: str = "accuracy"
    selection_cross_entropy_metric: str = "cross_entropy"

    def __post_init__(self) -> None:
        if self.run_id not in TPRED_BRANCH_RUN_IDS:
            raise ValueError("continuation branch must be one of the four Tpred children")
        if self.evaluation_fn is None:
            raise ValueError("scientific continuation branches require model_val_stop evaluation")
        if self.run_id == T10_ROBUST and self.robust_sampler is None:
            raise ValueError("T10_robust must expose its separate recorded sampler state")
        if self.run_id != T10_ROBUST and self.robust_sampler is not None:
            raise ValueError("only T10_robust may own a robust field-condition sampler")


def run_tpred_continuation_branches(
    snapshot: TrainingLineageSnapshot,
    *,
    batch_plan: ContinuationBatchPlan,
    branches: Sequence[ContinuationBranch],
    seed_id: int,
    grad_clip_norm: float = 0.0,
) -> dict[str, Any]:
    """Execute all four branches from the exact same RAM-only Tpred state.

    Returned checkpoint candidates remain ordinary Python objects in RAM.  The
    publication function later retains only the aggregate's ordered median.
    """

    if int(seed_id) not in PAIRED_SEED_IDS:
        raise ValueError("branch execution requires a paired seed ID")
    if batch_plan.content_hash != snapshot.batch_plan_sha256:
        raise ValueError("branch execution batch plan differs from the captured Tpred lineage")
    by_id = {branch.run_id: branch for branch in branches}
    if len(by_id) != len(branches) or set(by_id) != set(TPRED_BRANCH_RUN_IDS):
        raise ValueError("branch execution requires each of the four Tpred children exactly once")

    reports: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    restored_states: dict[str, dict[str, Any]] = {}
    robust_sampler_state: dict[str, Any] | None = None
    for run_id in TPRED_BRANCH_RUN_IDS:
        branch = by_id[run_id]
        if branch.robust_sampler is not None:
            if int(branch.robust_sampler.seed) != int(snapshot.robust_sampler_seed):
                raise ValueError("T10_robust sampler seed differs from the recorded branch stream")
            if int(branch.robust_sampler.total_draws) != 0:
                raise ValueError("T10_robust sampler must begin from its pristine recorded stream")
        restored = restore_training_lineage(
            snapshot,
            model=branch.model,
            optimizer=branch.optimizer,
            scheduler=branch.scheduler,
            amp_scaler=branch.amp_scaler,
            restore_rng=True,
        )
        restored_states[run_id] = restored
        selected_state: dict[str, Any] = {}
        report = run_exact_step_training(
            model=branch.model,
            optimizer=branch.optimizer,
            scheduler=branch.scheduler,
            amp_scaler=branch.amp_scaler,
            batch_plan=batch_plan,
            batch_resolver=branch.batch_resolver,
            loss_fn=branch.loss_fn,
            evaluation_fn=branch.evaluation_fn,
            grad_clip_norm=grad_clip_norm,
            selection_metric=branch.selection_metric,
            selection_cross_entropy_metric=branch.selection_cross_entropy_metric,
            selection_state=selected_state,
        )
        reports[run_id] = report
        model_config = getattr(branch.model, "config", None)
        candidates[run_id] = {
            "checkpoint_contract": PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT,
            "run_id": run_id,
            "seed_id": int(seed_id),
            "model_config": (
                model_config.to_dict()
                if hasattr(model_config, "to_dict")
                else deepcopy(model_config)
            ),
            "model_state_dict": deepcopy(selected_state["model_state_dict"]),
            "selected_model_val_stop_step": int(selected_state["step"]),
            "epoch": int(selected_state["step"]),
            "parent_hashes": {"tpred_ram_snapshot_sha256": snapshot.content_hash},
            "channel_policy": consumer_run_specs()[run_id].channel_policy,
            "weights_only": True,
        }
        if branch.robust_sampler is not None:
            robust_sampler_state = branch.robust_sampler.state_dict()

    batch_hashes = {tuple(report["batch_hashes"]) for report in reports.values()}
    eval_steps = {tuple(report["evaluation_steps"]) for report in reports.values()}
    initial_rng = {report["initial_global_rng_sha256"] for report in reports.values()}
    if len(batch_hashes) != 1 or len(eval_steps) != 1 or len(initial_rng) != 1:
        raise AssertionError(
            "Tpred branches changed paired batches, evaluation opportunities, or initial RNG"
        )
    if any(
        state["model_state_sha256"] != snapshot.model_state_hash
        or state["optimizer_state_sha256"] != snapshot.optimizer_state_hash
        or state["scheduler_state_sha256"] != snapshot.scheduler_state_hash
        for state in restored_states.values()
    ):
        raise AssertionError("a continuation branch did not restore the exact Tpred state")

    audit = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_BRANCH_EXECUTION_CONTRACT,
            "seed_id": int(seed_id),
            "parent_snapshot_sha256": snapshot.content_hash,
            "batch_plan_sha256": batch_plan.content_hash,
            "branch_run_ids": list(TPRED_BRANCH_RUN_IDS),
            "optimizer_steps_each": int(batch_plan.steps),
            "evaluation_steps": list(batch_plan.evaluation_steps),
            "identical_initial_model_optimizer_scheduler_amp": True,
            "identical_batches_and_evaluation_opportunities": True,
            "identical_dropout_global_rng_start": True,
            "robust_sampler_rng_seed": int(snapshot.robust_sampler_seed),
            "robust_sampler_final_state_sha256": (
                None if robust_sampler_state is None else _state_hash(robust_sampler_state)
            ),
            "candidate_weights_residency": "allocation_ram_only",
            "optimizer_scheduler_state_persisted": False,
            "generated_dense_fields_persisted": False,
            "reports": reports,
        }
    )
    return {"audit": audit, "reports": reports, "candidate_weights": candidates}


def run_a0_long_from_ram_lineage(
    snapshot: TrainingLineageSnapshot,
    *,
    batch_plan: ContinuationBatchPlan,
    model: Any,
    optimizer: Any,
    batch_resolver: Callable[[tuple[int, ...]], Any],
    loss_fn: Callable[[Any, Any, int], Any] = consumer_cross_entropy_loss,
    scheduler: Any | None = None,
    amp_scaler: Any | None = None,
    evaluation_fn: Callable[[Any, int], Mapping[str, Any]] | None = None,
    grad_clip_norm: float = 0.0,
    seed_id: int,
    selection_metric: str = "accuracy",
    selection_cross_entropy_metric: str = "cross_entropy",
) -> dict[str, Any]:
    """Run A0_C250_LONG as an exact-budget continuation of A0_C250."""

    if int(seed_id) not in PAIRED_SEED_IDS:
        raise ValueError("A0_C250_LONG requires a paired seed ID")
    if snapshot.batch_plan_sha256 != batch_plan.content_hash:
        raise ValueError("A0 long continuation batch plan differs from its RAM lineage")
    restored = restore_training_lineage(
        snapshot,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=amp_scaler,
        restore_rng=True,
    )
    selected_state: dict[str, Any] = {}
    report = run_exact_step_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=amp_scaler,
        batch_plan=batch_plan,
        batch_resolver=batch_resolver,
        loss_fn=loss_fn,
        evaluation_fn=evaluation_fn,
        grad_clip_norm=grad_clip_norm,
        selection_metric=selection_metric,
        selection_cross_entropy_metric=selection_cross_entropy_metric,
        selection_state=selected_state,
    )
    return {
        "report": report,
        "lineage": with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_BRANCH_EXECUTION_CONTRACT,
                "run_id": A0_C250_LONG,
                "seed_id": int(seed_id),
                "parent_run_id": A0_C250,
                "parent_snapshot_sha256": snapshot.content_hash,
                "restored_state": restored,
                "optimizer_steps": int(batch_plan.steps),
                "batch_plan_sha256": batch_plan.content_hash,
                "warm_start_not_scratch": True,
                "candidate_weights_residency": "allocation_ram_only",
            }
        ),
        "candidate_weights": {
            "checkpoint_contract": PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT,
            "run_id": A0_C250_LONG,
            "seed_id": int(seed_id),
            "model_config": (
                model.config.to_dict()
                if hasattr(getattr(model, "config", None), "to_dict")
                else deepcopy(getattr(model, "config", None))
            ),
            "model_state_dict": deepcopy(selected_state["model_state_dict"]),
            "selected_model_val_stop_step": int(selected_state["step"]),
            "epoch": int(selected_state["step"]),
            "parent_hashes": {"a0_c250_ram_snapshot_sha256": snapshot.content_hash},
            "channel_policy": None,
            "weights_only": True,
        },
    }


@dataclass(frozen=True)
class ReplicaResult:
    run_id: str
    seed_id: int
    metrics: Mapping[str, Any]
    weights_payload: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        if self.run_id not in STEP3_RUN_IDS:
            raise ValueError("replica has unknown Step 3 run ID")
        if int(self.seed_id) not in PAIRED_SEED_IDS:
            raise ValueError("replica seed is not paired")
        if "model_state_dict" not in self.weights_payload:
            raise ValueError("replica weights payload is incomplete")


def _metric(metrics: Mapping[str, Any], dotted: str) -> float:
    value: Any = metrics
    for part in str(dotted).split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"replica metrics are missing {dotted!r}")
        value = value[part]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"replica metric {dotted!r} is non-finite")
    return result


def aggregate_paired_replicas(
    replicas: Sequence[ReplicaResult],
    *,
    primary_metric: str = "model_val_stop.accuracy",
    cross_entropy_metric: str = "model_val_stop.cross_entropy",
    gain_metric: str | None = None,
) -> dict[str, Any]:
    if len(replicas) != 3:
        raise ValueError("scientific aggregate requires exactly three replicas")
    run_ids = {item.run_id for item in replicas}
    seeds = {int(item.seed_id) for item in replicas}
    if len(run_ids) != 1 or seeds != set(PAIRED_SEED_IDS):
        raise ValueError("replicas must share one run and contain paired seeds 101/202/303")
    scored = [
        (
            _metric(item.metrics, primary_metric),
            0.0 if gain_metric is None else _metric(item.metrics, gain_metric),
            -_metric(item.metrics, cross_entropy_metric),
            int(item.seed_id),
            item,
        )
        for item in replicas
    ]
    scored.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    median = scored[1][4]
    primary = np.asarray([row[0] for row in scored], dtype=np.float64)
    gains = None if gain_metric is None else np.asarray([row[1] for row in scored], dtype=np.float64)
    cross_entropy = np.asarray([-row[2] for row in scored], dtype=np.float64)
    payload = {
        "contract": PREDICTION_ANCHORED_REPLICA_AGGREGATE_CONTRACT,
        "run_id": next(iter(run_ids)),
        "paired_seed_ids": list(PAIRED_SEED_IDS),
        "primary_metric": primary_metric,
        "gain_metric": gain_metric,
        "cross_entropy_metric": cross_entropy_metric,
        "primary_mean": float(primary.mean()),
        "primary_sample_std": float(primary.std(ddof=1)),
        "primary_population_std": float(primary.std(ddof=0)),
        "gain_mean": None if gains is None else float(gains.mean()),
        "gain_sample_std": None if gains is None else float(gains.std(ddof=1)),
        "cross_entropy_mean": float(cross_entropy.mean()),
        "cross_entropy_sample_std": float(cross_entropy.std(ddof=1)),
        "ordering": [
            primary_metric,
            gain_metric or "constant_zero",
            f"negative({cross_entropy_metric})",
            "seed_id",
        ],
        "ordered_seed_ids": [int(row[4].seed_id) for row in scored],
        "median_seed_id": int(median.seed_id),
        "best_seed_id": int(scored[-1][4].seed_id),
        "median_is_lucky_best": bool(median.seed_id == scored[-1][4].seed_id),
        "replica_metrics": [
            {"seed_id": int(item.seed_id), "metrics": deepcopy(dict(item.metrics))}
            for item in sorted(replicas, key=lambda value: value.seed_id)
        ],
    }
    return with_content_hash(payload)


def weights_only_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "amp_scaler_state_dict",
        "generated_fields",
        "f_true",
        "f0",
        "h0",
        "bridge_fields",
    }
    if "model_state_dict" not in payload:
        raise ValueError("checkpoint has no model_state_dict")
    output = {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in forbidden and key in {
            "checkpoint_contract",
            "model_contract",
            "model_config",
            "model_state_dict",
            "epoch",
            "metrics",
            "run_id",
            "seed_id",
            "parent_hashes",
            "channel_policy",
        }
    }
    output["model_state_dict"] = deepcopy(payload["model_state_dict"])
    output["weights_only"] = True
    output["optimizer_state_persisted"] = False
    output["scheduler_state_persisted"] = False
    output["generated_fields_persisted"] = False
    return output


def measure_weights_payload_bytes(payload: Mapping[str, Any]) -> int:
    measured = len(_state_bytes(weights_only_payload(payload)))
    if measured <= 0:
        raise AssertionError("serialized weights payload is empty")
    return measured


def publish_paired_replicas(
    replicas: Sequence[ReplicaResult],
    *,
    output_dir: str | Path,
    primary_metric: str = "model_val_stop.accuracy",
    cross_entropy_metric: str = "model_val_stop.cross_entropy",
    gain_metric: str | None = None,
    reservation_bytes: int | None = None,
) -> dict[str, Any]:
    import torch

    aggregate = aggregate_paired_replicas(
        replicas,
        primary_metric=primary_metric,
        cross_entropy_metric=cross_entropy_metric,
        gain_metric=gain_metric,
    )
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"consumer publication directory is not empty: {root}")
    median_seed = int(aggregate["median_seed_id"])
    median = next(item for item in replicas if int(item.seed_id) == median_seed)
    retained = weights_only_payload(median.weights_payload)
    retained["run_id"] = median.run_id
    retained["seed_id"] = median_seed
    encoded_checkpoint = _state_bytes(retained)
    checkpoint_sha256 = hashlib.sha256(encoded_checkpoint).hexdigest()
    aggregate_bytes = immutable_json_bytes(aggregate)
    metrics_artifact, publication_bytes = build_total_sized_publication(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT,
            "run_id": median.run_id,
            "aggregate_sha256": aggregate["content_hash"],
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "median_seed_id": median_seed,
            "retained_checkpoint": "median_weights.pt",
            "retained_checkpoint_sha256": checkpoint_sha256,
            "measured_state_bytes": len(encoded_checkpoint),
            "reserved_bytes": reservation_bytes,
            "reservation_enforced_before_publication": reservation_bytes is not None,
            "nonmedian_weights_persisted": False,
            "optimizer_state_persisted": False,
            "scheduler_state_persisted": False,
            "generated_fields_persisted": False,
            "persistent_artifact_allowlist": ["aggregate_metrics.json", "median_weights.pt", "publication.json"],
        },
        other_artifact_bytes=len(encoded_checkpoint) + len(aggregate_bytes),
    )
    total_bytes = int(metrics_artifact["measured_total_persistent_bytes"])
    if reservation_bytes is not None and total_bytes > int(reservation_bytes):
        raise PermissionError("consumer publication directory exceeds its predeclared run reservation")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "median_weights.pt"
    for path, encoded in (
        (checkpoint_path, encoded_checkpoint),
        (root / "aggregate_metrics.json", aggregate_bytes),
        (root / "publication.json", publication_bytes),
    ):
        with path.open("xb") as handle:
            handle.write(encoded)
    measured_bytes = len(encoded_checkpoint)
    names = sorted(path.name for path in root.iterdir())
    if names != ["aggregate_metrics.json", "median_weights.pt", "publication.json"]:
        raise RuntimeError(f"consumer publication wrote unexpected artifacts: {names}")
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if any(key in loaded for key in ("optimizer_state_dict", "scheduler_state_dict", "generated_fields")):
        raise AssertionError("published consumer checkpoint contains forbidden training/field state")
    return {
        "ok": True,
        "contract": PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT,
        "run_id": median.run_id,
        "median_seed_id": median_seed,
        "checkpoint": str(checkpoint_path),
        "aggregate": str(root / "aggregate_metrics.json"),
        "publication": str(root / "publication.json"),
        "measured_state_bytes": measured_bytes,
        "measured_total_persistent_bytes": total_bytes,
        "checkpoint_sha256": checkpoint_sha256,
        "persistent_artifacts": names,
    }


def publish_evaluated_teacher_replica(
    *,
    run_id: str,
    selection_aggregate: Mapping[str, Any],
    replica_checkpoint_paths: Mapping[int, str | Path],
    output_dir: str | Path,
    reservation_bytes: int | None = None,
) -> dict[str, Any]:
    """Retain the exact evaluation-selected teacher checkpoint bytes.

    Teacher selection evidence records the SHA-256 of the temporary replica
    that was actually forwarded on ``model_val_select``.  Re-serializing that
    payload under ``median_weights.pt`` would produce a different byte hash and
    break the later selection -> binding -> cache identity chain.  This
    publisher therefore copies the selected file byte-for-byte and rejects any
    disagreement with the immutable evaluation aggregate.
    """

    from .bridge_evaluation import (
        PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
    )
    from .bridge_contracts import validate_content_hash

    validate_content_hash(
        selection_aggregate,
        expected_contract=PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
    )
    if str(selection_aggregate.get("run_id")) != str(run_id):
        raise ValueError("teacher selection aggregate belongs to another run")
    if run_id not in {T10_CLEAN, T10_ROBUST, T10_ALL50_CLEAN}:
        raise ValueError("evaluated teacher publication requires a T10 recipe")
    seed = int(selection_aggregate["median_seed_id"])
    if seed not in PAIRED_SEED_IDS or set(int(value) for value in replica_checkpoint_paths) != set(
        PAIRED_SEED_IDS
    ):
        raise ValueError("teacher publication requires all paired replica paths")
    source = Path(replica_checkpoint_paths[seed])
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"selected teacher replica is absent or unsafe: {source}")
    expected_sha = str(selection_aggregate["median_checkpoint_sha256"])
    if sha256_file(source) != expected_sha:
        raise ValueError("selected teacher replica bytes disagree with evaluation evidence")
    measured_source_bytes = int(source.stat().st_size)

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"consumer publication directory is not empty: {root}")
    import torch
    loaded = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping) or "model_state_dict" not in loaded:
        raise ValueError("published teacher checkpoint is not a weights payload")
    forbidden = {
        "optimizer_state_dict", "scheduler_state_dict", "amp_scaler_state_dict",
        "generated_fields", "f_true", "f0", "h0", "bridge_fields",
    }
    if forbidden.intersection(loaded):
        raise ValueError("evaluated teacher checkpoint contains forbidden persistent state")

    aggregate_bytes = immutable_json_bytes(selection_aggregate)
    publication, publication_bytes = build_total_sized_publication(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT,
            "run_id": str(run_id),
            "aggregate_sha256": selection_aggregate["content_hash"],
            "aggregate_contract": PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "median_seed_id": seed,
            "retained_checkpoint": "median_weights.pt",
            "retained_checkpoint_sha256": expected_sha,
            "measured_state_bytes": measured_source_bytes,
            "reserved_bytes": reservation_bytes,
            "reservation_enforced_before_publication": reservation_bytes is not None,
            "byte_exact_evaluated_replica": True,
            "nonmedian_weights_persisted": False,
            "optimizer_state_persisted": False,
            "scheduler_state_persisted": False,
            "generated_fields_persisted": False,
            "persistent_artifact_allowlist": [
                "aggregate_metrics.json", "median_weights.pt", "publication.json"
            ],
        },
        other_artifact_bytes=measured_source_bytes + len(aggregate_bytes),
    )
    total_bytes = int(publication["measured_total_persistent_bytes"])
    if reservation_bytes is not None and total_bytes > int(reservation_bytes):
        raise PermissionError("evaluated teacher publication directory exceeds its predeclared run reservation")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "median_weights.pt"
    with source.open("rb") as reader, checkpoint_path.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    for path, encoded in (
        (root / "aggregate_metrics.json", aggregate_bytes),
        (root / "publication.json", publication_bytes),
    ):
        with path.open("xb") as handle:
            handle.write(encoded)
    if sha256_file(checkpoint_path) != expected_sha:
        raise RuntimeError("byte-exact teacher checkpoint publication changed its SHA-256")
    names = sorted(path.name for path in root.iterdir())
    if names != ["aggregate_metrics.json", "median_weights.pt", "publication.json"]:
        raise RuntimeError(f"teacher publication wrote unexpected artifacts: {names}")
    return {
        "ok": True,
        "contract": PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT,
        "run_id": str(run_id),
        "median_seed_id": seed,
        "checkpoint": str(checkpoint_path),
        "aggregate": str(root / "aggregate_metrics.json"),
        "publication": str(root / "publication.json"),
        "checkpoint_sha256": expected_sha,
        "measured_total_persistent_bytes": total_bytes,
        "byte_exact_evaluated_replica": True,
        "persistent_artifacts": names,
    }


def record_step3_registry_measurements(
    registry: Mapping[str, Any],
    representative_payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(representative_payloads) != set(STEP3_RUN_IDS):
        missing = sorted(set(STEP3_RUN_IDS) - set(representative_payloads))
        extra = sorted(set(representative_payloads) - set(STEP3_RUN_IDS))
        raise ValueError(f"Step 3 measurement payloads must cover all eight rows; missing={missing}, extra={extra}")
    measured = {
        run_id: measure_weights_payload_bytes(payload)
        for run_id, payload in representative_payloads.items()
    }
    updated = record_registry_measurements(registry, measured)
    artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_MEASURED_UPSTREAM_CONTRACT,
            "input_registry_sha256": registry["content_hash"],
            "updated_registry_sha256": updated["content_hash"],
            "measured_state_bytes": measured,
            "serialization_headroom_recorded_by_registry": True,
            "all_step3_rows_measured": True,
        }
    )
    return updated, artifact


__all__ = [
    "PREDICTION_ANCHORED_CONSUMER_CONFIG_CONTRACT",
    "PREDICTION_ANCHORED_CONSUMER_RUN_CONTRACT",
    "PREDICTION_ANCHORED_CONTINUATION_PLAN_CONTRACT",
    "PREDICTION_ANCHORED_BRANCH_LINEAGE_CONTRACT",
    "PREDICTION_ANCHORED_ROBUST_SAMPLER_CONTRACT",
    "PREDICTION_ANCHORED_CORRUPTION_SCALE_CONTRACT",
    "PREDICTION_ANCHORED_REPLICA_AGGREGATE_CONTRACT",
    "PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT",
    "PREDICTION_ANCHORED_MEASURED_UPSTREAM_CONTRACT",
    "PREDICTION_ANCHORED_CONSUMER_REPLICA_MANIFEST_CONTRACT",
    "PREDICTION_ANCHORED_BRANCH_EXECUTION_CONTRACT",
    "A0_C250",
    "A0_C250_LONG",
    "A0_S500",
    "TPRED",
    "TPRED_CONTINUE",
    "T10_CLEAN",
    "T10_ROBUST",
    "T10_ALL50_CLEAN",
    "STEP3_RUN_IDS",
    "TPRED_BRANCH_RUN_IDS",
    "ROBUST_CONDITIONS",
    "ConsumerRunSpec",
    "ConsumerCampaignConfig",
    "ContinuationBatchPlan",
    "TrainingLineageSnapshot",
    "RobustBridgeSamplerConfig",
    "RobustBridgeSampler",
    "ContinuationBranch",
    "ReplicaResult",
    "consumer_run_specs",
    "build_consumer_replica_manifest",
    "build_continuation_batch_plan",
    "capture_training_lineage",
    "restore_training_lineage",
    "branch_lineage_artifact",
    "copy_reference_hlt_weights",
    "verify_initial_logit_identity",
    "build_step3_consumer_model",
    "initialize_step3_root_from_reference",
    "verify_paired_a0_tpred_initialization",
    "build_consumer_tensor_batch",
    "build_provider_batch_resolver",
    "consumer_cross_entropy_loss",
    "fit_bridge_corruption_scale",
    "consumer_fields_for_run",
    "run_exact_step_training",
    "run_tpred_continuation_branches",
    "run_a0_long_from_ram_lineage",
    "aggregate_paired_replicas",
    "weights_only_payload",
    "measure_weights_payload_bytes",
    "publish_paired_replicas",
    "record_step3_registry_measurements",
]
