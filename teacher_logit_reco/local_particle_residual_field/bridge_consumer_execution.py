"""End-to-end numerical execution of the eight paired bridge consumers."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import resolve_device, set_training_seed

from .bridge import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    BRIDGE_CONTROLS,
    RESPONSE_RHOS,
    apply_bridge_control,
    build_matched_wrong_event_map,
    fit_matched_bin_edges,
    validate_bridge_recipe,
    virtual_bridge,
)
from .bridge_consumer import (
    A0_C250,
    A0_C250_LONG,
    A0_S500,
    PAIRED_SEED_IDS,
    STEP3_RUN_IDS,
    T10_ALL50_CLEAN,
    T10_CLEAN,
    T10_ROBUST,
    TPRED,
    TPRED_BRANCH_RUN_IDS,
    TPRED_CONTINUE,
    ConsumerCampaignConfig,
    ContinuationBranch,
    ReplicaResult,
    RobustBridgeSampler,
    build_consumer_tensor_batch,
    build_continuation_batch_plan,
    build_step3_consumer_model,
    capture_training_lineage,
    consumer_cross_entropy_loss,
    fit_bridge_corruption_scale,
    initialize_step3_root_from_reference,
    run_a0_long_from_ram_lineage,
    run_exact_step_training,
    run_tpred_continuation_branches,
    verify_paired_a0_tpred_initialization,
)
from .bridge_contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)
from .bridge_evaluation import (
    ConsumerReplicaEvaluation,
    aggregate_consumer_evaluations,
    classification_metrics,
    evaluate_bound_consumer_conditions,
    finalize_consumer_confirmation,
)
from .bridge_execution import (
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)
from .bridge_numerical import _paths_from_source, _verify_staged_source_binding
from .bridge_ram import (
    AllocationNpzStager,
    AllocationRamLedger,
    FrozenR0Runner,
    StagedSource,
)
from .bridge_splits import (
    PREDICTION_ANCHORED_SPLIT_CONTRACT,
    authorize_split_access,
    build_manifest_validation_unlock,
    claim_split_access,
    split_binding,
)
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, compute_local_particle_residual_fields


PREDICTION_ANCHORED_CONSUMER_EXECUTION_CONTRACT = (
    "prediction_anchored_consumer_numerical_execution_v1"
)


class _ResidentBridgeParent:
    """Allocation-only f0/f_true bank; no dense tensor is persisted."""

    def __init__(
        self,
        *,
        name: str,
        hlt: StagedSource,
        offline: StagedSource,
        r0: FrozenR0Runner,
        ledger: AllocationRamLedger,
        generation_batch_size: int,
    ) -> None:
        self.name = str(name)
        self.hlt = hlt
        self.offline = offline
        n_events = hlt.n_events
        particle_width = int(hlt.manifest["arrays"]["tokens"]["shape"][1])
        expected = n_events * particle_width * 50 * np.dtype(np.float32).itemsize * 2
        expected += n_events * particle_width * np.dtype(bool).itemsize
        self.reservation_id = ledger.reserve(
            owner="rank0",
            role=f"resident_f0_ftrue:{self.name}",
            expected_bytes=expected,
            category="derived",
        )
        self.f0 = np.empty((n_events, particle_width, 50), dtype=np.float32)
        self.f_true = np.empty_like(self.f0)
        self.mask = np.empty((n_events, particle_width), dtype=bool)
        step = int(generation_batch_size)
        if step <= 0:
            raise ValueError("generation_batch_size must be positive")
        for start in range(0, n_events, step):
            stop = min(start + step, n_events)
            hlt_batch = hlt.read_range(start, stop, names=("tokens", "mask"))
            offline_batch = offline.read_range(start, stop, names=("tokens", "mask"))
            truth, target_mask, _, _, _ = compute_local_particle_residual_fields(
                hlt_batch["tokens"],
                hlt_batch["mask"],
                offline_batch["tokens"],
                offline_batch["mask"],
                radii=DEFAULT_LOCAL_RESIDUAL_RADII,
            )
            anchor, _ = r0.predict_numpy(hlt_batch["tokens"], hlt_batch["mask"])
            self.f0[start:stop] = anchor
            self.f_true[start:stop] = truth
            self.mask[start:stop] = target_mask
        if not np.isfinite(self.f0).all() or not np.isfinite(self.f_true).all():
            raise ValueError(f"resident bridge parent {name} contains non-finite fields")
        if np.any(self.f0[~self.mask] != 0) or np.any(self.f_true[~self.mask] != 0):
            raise ValueError(f"resident bridge parent {name} changed padding semantics")
        ledger.commit(self.reservation_id, measured_bytes=expected)

    def batch(self, indices: Sequence[int]) -> dict[str, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        hlt = self.hlt.read_indices(selected, names=("tokens", "mask", "labels"))
        if not np.array_equal(hlt["mask"], self.mask[selected]):
            raise ValueError(f"resident bridge parent {self.name} mask changed")
        return {
            **hlt,
            "f0": self.f0[selected],
            "f_true": self.f_true[selected],
            "indices": selected,
        }


def _config_from_spec(spec: Mapping[str, Any]) -> ConsumerCampaignConfig:
    values = dict(spec["consumer_training"])
    values["paired_seed_ids"] = tuple(values.get("paired_seed_ids", PAIRED_SEED_IDS))
    return ConsumerCampaignConfig(**values)


def _evaluation_steps(total_steps: int, interval: int) -> tuple[int, ...]:
    values = list(range(int(interval), int(total_steps) + 1, int(interval)))
    if not values or values[-1] != int(total_steps):
        values.append(int(total_steps))
    return tuple(values)


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


def _resolver(
    parent: _ResidentBridgeParent,
    *,
    parent_indices: np.ndarray,
    run_id: str,
    device: Any,
    robust_sampler: RobustBridgeSampler | None = None,
) -> Callable[[tuple[int, ...]], dict[str, Any]]:
    def resolve(local_indices: tuple[int, ...]) -> dict[str, Any]:
        local = np.asarray(local_indices, dtype=np.int64)
        selected = parent_indices[local]
        raw = parent.batch(selected)
        batch = build_consumer_tensor_batch(
            tokens=raw["tokens"],
            mask=raw["mask"],
            labels=raw["labels"],
            f0=raw["f0"],
            f_true=raw["f_true"],
            run_id=run_id,
            device=device,
            robust_sampler=robust_sampler,
        )
        import torch

        batch["indices"] = torch.as_tensor(selected, dtype=torch.long, device=device)
        return batch

    return resolve


def _quick_evaluate(
    model: Any,
    parent: _ResidentBridgeParent,
    *,
    parent_indices: np.ndarray,
    run_id: str,
    device: Any,
    batch_size: int,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    eval_run = T10_CLEAN if run_id == T10_ROBUST else run_id
    logits_parts = []
    labels_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(parent_indices), int(batch_size)):
            selected = parent_indices[start : start + int(batch_size)]
            raw = parent.batch(selected)
            batch = build_consumer_tensor_batch(
                tokens=raw["tokens"],
                mask=raw["mask"],
                labels=raw["labels"],
                f0=raw["f0"],
                f_true=raw["f_true"],
                run_id=eval_run,
                device=device,
            )
            logits_parts.append(_model_logits(model, batch).detach().float().cpu())
            labels_parts.append(batch["labels"].detach().cpu())
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    return {
        "accuracy": float((logits.argmax(dim=1) == labels).float().mean().item()),
        "cross_entropy": float(functional.cross_entropy(logits, labels).item()),
    }


def _candidate(
    *,
    model: Any,
    state: Mapping[str, Any],
    run_id: str,
    seed_id: int,
    parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "checkpoint_contract": "prediction_anchored_consumer_run_v1",
        "run_id": run_id,
        "seed_id": int(seed_id),
        "model_config": config.to_dict() if hasattr(config, "to_dict") else deepcopy(config),
        "model_state_dict": deepcopy(state["model_state_dict"]),
        "epoch": int(state["step"]),
        "parent_hashes": dict(parent_hashes),
        "channel_policy": (
            BRIDGE_CHANNEL_ALL50 if run_id == T10_ALL50_CLEAN else
            BRIDGE_CHANNEL_PHYSICAL45 if run_id not in {A0_C250, A0_C250_LONG, A0_S500} else None
        ),
        "weights_only": True,
    }


def _selected_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    step = int(report["selected_checkpoint_step"])
    for row in report["evaluations"]:
        if int(row["step"]) == step:
            return dict(row["metrics"])
    raise AssertionError("selected consumer evaluation is absent from its report")


def _event_ids(source: StagedSource, indices: np.ndarray) -> list[str]:
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


class _BoundConditionForward:
    def __init__(
        self,
        *,
        model: Any,
        checkpoint_sha256: str,
        channel_policy: str,
        device: Any,
        control_seed: int,
    ) -> None:
        self.model = model
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.channel_policy = str(channel_policy)
        self.device = device
        self.control_seed = int(control_seed)

    def __call__(self, batch: Mapping[str, Any], condition: str) -> Any:
        anchor = batch["f0"]
        truth = batch["f_true"]
        mask = batch["mask"]
        if condition == "f0":
            fields = anchor
        elif condition.startswith("rho_"):
            fields = virtual_bridge(
                anchor,
                truth,
                mask,
                rho=condition.removeprefix("rho_"),
                channel_policy=self.channel_policy,
            )
        elif condition == "event_shuffled_delta":
            fields = batch[
                "event_shuffled_fields_all50"
                if self.channel_policy == BRIDGE_CHANNEL_ALL50
                else "event_shuffled_fields_physical45"
            ]
        elif condition in BRIDGE_CONTROLS:
            fields = apply_bridge_control(
                anchor,
                truth,
                mask,
                control_type=condition,
                seed=self.control_seed,
                event_ids=batch["event_ids"],
                channel_policy=self.channel_policy,
            )
        elif condition == "oracle_physical45":
            fields = virtual_bridge(
                anchor, truth, mask, rho="1.000", channel_policy=BRIDGE_CHANNEL_PHYSICAL45
            )
        elif condition == "oracle_all50":
            fields = virtual_bridge(
                anchor, truth, mask, rho="1.000", channel_policy=BRIDGE_CHANNEL_ALL50
            )
        elif condition == "zero_field_consumer_diagnostic":
            fields = np.zeros_like(anchor)
        else:
            raise ValueError(f"unknown consumer evaluation condition {condition!r}")
        tensor_batch = build_consumer_tensor_batch(
            tokens=batch["tokens"],
            mask=mask,
            labels=batch["labels"],
            f0=fields,
            f_true=fields,
            run_id=TPRED,
            device=self.device,
        )
        self.model.eval()
        return _model_logits(self.model, tensor_batch)


def _evaluation_batches(
    parent: _ResidentBridgeParent,
    *,
    parent_indices: np.ndarray,
    batch_size: int,
    control_seed: int,
    frozen_bin_edges: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    raw = parent.hlt.read_indices(
        parent_indices,
        names=("tokens", "mask", "labels", "jet_file_indices", "jet_entries"),
    )
    f0 = parent.f0[parent_indices]
    truth = parent.f_true[parent_indices]
    event_ids = _event_ids(parent.hlt, parent_indices)
    wrong_map = build_matched_wrong_event_map(
        tokens=raw["tokens"],
        mask=raw["mask"],
        labels=raw["labels"],
        event_ids=event_ids,
        seed=int(control_seed),
        bin_edges=frozen_bin_edges,
        logical_block_size=int(parent.hlt.shard_size),
        source_block_ids=np.asarray(parent_indices, dtype=np.int64)
        // int(parent.hlt.shard_size),
    )
    event_shuffled_physical45 = apply_bridge_control(
        f0,
        truth,
        raw["mask"],
        control_type="event_shuffled_delta",
        seed=int(control_seed),
        event_ids=event_ids,
        wrong_event_map=wrong_map,
        channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
    )
    event_shuffled_all50 = apply_bridge_control(
        f0,
        truth,
        raw["mask"],
        control_type="event_shuffled_delta",
        seed=int(control_seed),
        event_ids=event_ids,
        wrong_event_map=wrong_map,
        channel_policy=BRIDGE_CHANNEL_ALL50,
    )
    batches = []
    for start in range(0, len(parent_indices), int(batch_size)):
        stop = min(start + int(batch_size), len(parent_indices))
        batches.append(
            {
                "tokens": raw["tokens"][start:stop],
                "mask": raw["mask"][start:stop],
                "labels": raw["labels"][start:stop],
                "event_ids": event_ids[start:stop],
                "f0": f0[start:stop],
                "f_true": truth[start:stop],
                "event_shuffled_fields_physical45": event_shuffled_physical45[start:stop],
                "event_shuffled_fields_all50": event_shuffled_all50[start:stop],
            }
        )
    return batches


def run_consumer_campaign_from_execution_spec(
    execution_spec_path: str | Path,
    *,
    r0_checkpoint_path: str | Path,
    r0_registration_path: str | Path,
    physical45_recipe_path: str | Path,
    all50_recipe_path: str | Path,
    replica_output_dir: str | Path,
    evaluation_output_dir: str | Path,
    ram_root: str | Path,
    allocation_id: str | None = None,
    device: str = "auto",
    shard_size: int = 8192,
    generation_batch_size: int = 512,
    evaluation_batch_size: int = 512,
    bootstrap_resamples: int = 4_000,
    capacity_bytes: int | None = None,
    allow_unverified_test_root: bool = False,
    model_factory: Callable[[str, ConsumerCampaignConfig], Any] | None = None,
) -> dict[str, Any]:
    """Execute all 24 paired consumer replicas in one source-sharing allocation."""

    import torch

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
        raise ValueError("R0 was trained against a different child split manifest")
    physical_recipe = load_hashed_json(physical45_recipe_path)
    all50_recipe = load_hashed_json(all50_recipe_path)
    validate_bridge_recipe(physical_recipe)
    validate_bridge_recipe(all50_recipe)
    if physical_recipe["channel_policy"] != BRIDGE_CHANNEL_PHYSICAL45:
        raise ValueError("primary recipe is not physical45")
    if all50_recipe["channel_policy"] != BRIDGE_CHANNEL_ALL50:
        raise ValueError("all-50 recipe is not all50")
    if (
        physical_recipe["parent_hashes"]["r0_checkpoint_sha256"] != r0_sha
        or all50_recipe["parent_hashes"]["r0_checkpoint_sha256"] != r0_sha
    ):
        raise ValueError("bridge recipe is bound to a different R0 checkpoint")
    config = _config_from_spec(spec)
    torch_device = resolve_device(str(device))
    make_model = model_factory or (
        lambda run_id, cfg: build_step3_consumer_model(
            run_id,
            model_size=cfg.model_size,
            num_classes=len(spec["class_names"]),
        )
    )
    allocation = str(allocation_id or os.environ.get("SLURM_JOB_ID", "local_consumers"))
    ledger = AllocationRamLedger(
        ram_root,
        allocation_id=allocation,
        capacity_bytes=capacity_bytes,
        allow_unverified_test_root=bool(allow_unverified_test_root),
    )
    replica_root = Path(replica_output_dir)
    evidence_root = Path(evaluation_output_dir)
    if (replica_root.exists() and any(replica_root.iterdir())) or (
        evidence_root.exists() and any(evidence_root.iterdir())
    ):
        raise FileExistsError("consumer replica/evaluation output must be empty")
    resident: list[_ResidentBridgeParent] = []
    try:
        stager = AllocationNpzStager(ledger, rank=0, world_size=1)
        wanted = {
            split: _paths_from_source(spec["sources"][split])
            for split in ("stack_train", "model_val")
        }
        staged, stage_report = stager.stage_named_pairs(wanted, shard_size=int(shard_size))
        for split in wanted:
            _verify_staged_source_binding(
                split=split,
                source=spec["sources"][split],
                report=staged[split][2],
            )
        r0 = FrozenR0Runner(r0_path, device=torch_device)
        stack = _ResidentBridgeParent(
            name="stack_train",
            hlt=staged["stack_train"][0],
            offline=staged["stack_train"][1],
            r0=r0,
            ledger=ledger,
            generation_batch_size=int(generation_batch_size),
        )
        model_val = _ResidentBridgeParent(
            name="model_val",
            hlt=staged["model_val"][0],
            offline=staged["model_val"][1],
            r0=r0,
            ledger=ledger,
            generation_batch_size=int(generation_batch_size),
        )
        resident.extend((stack, model_val))
        consumer_indices = np.asarray(
            child["children"]["stack_train_consumer"]["parent_row_indices"], dtype=np.int64
        )
        union_indices = np.arange(stack.hlt.n_events, dtype=np.int64)
        stop_indices = np.asarray(
            child["children"]["model_val_stop"]["parent_row_indices"], dtype=np.int64
        )
        select_indices = np.asarray(
            child["children"]["model_val_select"]["parent_row_indices"], dtype=np.int64
        )
        corruption = fit_bridge_corruption_scale(
            [(stack.f0[consumer_indices], stack.f_true[consumer_indices], stack.mask[consumer_indices])],
            parent_hashes={
                "execution_spec_sha256": spec["content_hash"],
                "r0_checkpoint_sha256": r0_sha,
            },
        )
        corruption_scale = corruption["bridge_correction_population_std"]
        parent_hashes = {
            "execution_spec_sha256": spec["content_hash"],
            "r0_checkpoint_sha256": r0_sha,
            "reference_hlt_checkpoint_sha256": spec["baseline_checkpoint"]["sha256"],
        }
        candidate_by_run_seed: dict[str, dict[int, dict[str, Any]]] = {
            run_id: {} for run_id in STEP3_RUN_IDS
        }
        metrics_by_run_seed: dict[str, dict[int, dict[str, Any]]] = {
            run_id: {} for run_id in STEP3_RUN_IDS
        }
        for seed in PAIRED_SEED_IDS:
            set_training_seed(int(seed))
            baseline_plan_consumer = build_continuation_batch_plan(
                seed_id=seed,
                n_examples=len(consumer_indices),
                batch_size=config.batch_size,
                steps=config.baseline_steps,
                evaluation_steps=_evaluation_steps(
                    config.baseline_steps, config.evaluation_interval_steps
                ),
            )
            baseline_plan_union = build_continuation_batch_plan(
                seed_id=seed,
                n_examples=len(union_indices),
                batch_size=config.batch_size,
                steps=config.baseline_steps,
                evaluation_steps=_evaluation_steps(
                    config.baseline_steps, config.evaluation_interval_steps
                ),
            )
            continuation_plan = build_continuation_batch_plan(
                seed_id=seed,
                n_examples=len(consumer_indices),
                batch_size=config.batch_size,
                steps=config.bridge_finetune_steps,
                evaluation_steps=config.continuation_evaluation_steps,
            )

            roots: dict[str, tuple[Any, Any, Any, dict[str, Any], Mapping[str, Any]]] = {}
            for run_id, indices, plan in (
                (A0_C250, consumer_indices, baseline_plan_consumer),
                (TPRED, consumer_indices, baseline_plan_consumer),
                (A0_S500, union_indices, baseline_plan_union),
            ):
                model = make_model(run_id, config).to(torch_device)
                initialization = initialize_step3_root_from_reference(
                    model,
                    spec["baseline_checkpoint"]["path"],
                    run_id=run_id,
                    map_location=torch_device,
                )
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                resolver = _resolver(
                    stack,
                    parent_indices=indices,
                    run_id=run_id,
                    device=torch_device,
                )
                # Root rows with the same paired seed begin optimization from
                # the same dropout/stochastic-depth stream. Model construction
                # is intentionally outside this stream reset.
                set_training_seed(int(seed))
                selected: dict[str, Any] = {}
                report = run_exact_step_training(
                    model=model,
                    optimizer=optimizer,
                    batch_plan=plan,
                    batch_resolver=resolver,
                    loss_fn=consumer_cross_entropy_loss,
                    evaluation_fn=lambda current, step, rid=run_id: _quick_evaluate(
                        current,
                        model_val,
                        parent_indices=stop_indices,
                        run_id=rid,
                        device=torch_device,
                        batch_size=evaluation_batch_size,
                    ),
                    grad_clip_norm=config.grad_clip_norm,
                    selection_metric="accuracy",
                    selection_cross_entropy_metric="cross_entropy",
                    selection_state=selected,
                )
                candidate_by_run_seed[run_id][seed] = _candidate(
                    model=model,
                    state=selected,
                    run_id=run_id,
                    seed_id=seed,
                    parent_hashes=parent_hashes | {
                        "initialization_sha256": initialization["content_hash"]
                    },
                )
                metrics_by_run_seed[run_id][seed] = {
                    "model_val_stop": _selected_metrics(report),
                    "optimizer_steps": report["optimizer_steps"],
                    "batch_plan_sha256": report["batch_plan_sha256"],
                }
                roots[run_id] = (model, optimizer, resolver, report, initialization)

            # Same-reference/zero-column identity is checked before any result
            # is allowed to leave RAM.  Rebuild roots because the trained roots
            # are no longer at initialization.
            a0_probe = make_model(A0_C250, config).to(torch_device)
            a0_init = initialize_step3_root_from_reference(
                a0_probe, spec["baseline_checkpoint"]["path"], run_id=A0_C250, map_location=torch_device
            )
            tpred_probe = make_model(TPRED, config).to(torch_device)
            tpred_init = initialize_step3_root_from_reference(
                tpred_probe, spec["baseline_checkpoint"]["path"], run_id=TPRED, map_location=torch_device
            )
            probe_raw = stack.batch(consumer_indices[: min(config.batch_size, len(consumer_indices))])
            a0_batch = build_consumer_tensor_batch(
                tokens=probe_raw["tokens"], mask=probe_raw["mask"], labels=probe_raw["labels"],
                f0=probe_raw["f0"], f_true=probe_raw["f_true"], run_id=A0_C250, device=torch_device,
            )
            tpred_batch = build_consumer_tensor_batch(
                tokens=probe_raw["tokens"], mask=probe_raw["mask"], labels=probe_raw["labels"],
                f0=probe_raw["f0"], f_true=probe_raw["f_true"], run_id=TPRED, device=torch_device,
            )
            a0_probe.eval(); tpred_probe.eval()
            with torch.no_grad():
                verify_paired_a0_tpred_initialization(
                    a0_initialization=a0_init,
                    tpred_initialization=tpred_init,
                    reference_logits=_model_logits(a0_probe, a0_batch),
                    tpred_logits=_model_logits(tpred_probe, tpred_batch),
                )

            # A0 long branches from the exact terminal A0 state.
            a0_model, a0_optimizer, _, _, _ = roots[A0_C250]
            a0_snapshot = capture_training_lineage(
                model=a0_model,
                optimizer=a0_optimizer,
                batch_plan=continuation_plan,
                dropout_stream_seed=int(seed),
                robust_sampler_seed=int(seed) + 70_000,
            )
            a0_long_model = make_model(A0_C250_LONG, config).to(torch_device)
            a0_long_optimizer = torch.optim.AdamW(
                a0_long_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            a0_long = run_a0_long_from_ram_lineage(
                a0_snapshot,
                batch_plan=continuation_plan,
                model=a0_long_model,
                optimizer=a0_long_optimizer,
                batch_resolver=_resolver(
                    stack, parent_indices=consumer_indices, run_id=A0_C250_LONG, device=torch_device
                ),
                evaluation_fn=lambda current, step: _quick_evaluate(
                    current, model_val, parent_indices=stop_indices, run_id=A0_C250_LONG,
                    device=torch_device, batch_size=evaluation_batch_size,
                ),
                grad_clip_norm=config.grad_clip_norm,
                seed_id=seed,
            )
            candidate_by_run_seed[A0_C250_LONG][seed] = a0_long["candidate_weights"] | {
                "parent_hashes": parent_hashes | a0_long["candidate_weights"]["parent_hashes"]
            }
            metrics_by_run_seed[A0_C250_LONG][seed] = {
                "model_val_stop": _selected_metrics(a0_long["report"]),
                "optimizer_steps": a0_long["report"]["optimizer_steps"],
                "batch_plan_sha256": a0_long["report"]["batch_plan_sha256"],
            }

            # Four Tpred children restore the same terminal model/optimizer/RNG.
            tpred_model, tpred_optimizer, _, _, _ = roots[TPRED]
            tpred_snapshot = capture_training_lineage(
                model=tpred_model,
                optimizer=tpred_optimizer,
                batch_plan=continuation_plan,
                dropout_stream_seed=int(seed),
                robust_sampler_seed=int(seed) + 80_000,
            )
            branches = []
            for run_id in TPRED_BRANCH_RUN_IDS:
                branch_model = make_model(run_id, config).to(torch_device)
                branch_optimizer = torch.optim.AdamW(
                    branch_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
                )
                robust = None
                if run_id == T10_ROBUST:
                    robust = RobustBridgeSampler(
                        seed=tpred_snapshot.robust_sampler_seed,
                        corruption_scale=corruption_scale,
                    )
                branches.append(
                    ContinuationBranch(
                        run_id=run_id,
                        model=branch_model,
                        optimizer=branch_optimizer,
                        batch_resolver=_resolver(
                            stack,
                            parent_indices=consumer_indices,
                            run_id=run_id,
                            device=torch_device,
                            robust_sampler=robust,
                        ),
                        evaluation_fn=lambda current, step, rid=run_id: _quick_evaluate(
                            current, model_val, parent_indices=stop_indices, run_id=rid,
                            device=torch_device, batch_size=evaluation_batch_size,
                        ),
                        robust_sampler=robust,
                    )
                )
            branch_result = run_tpred_continuation_branches(
                tpred_snapshot,
                batch_plan=continuation_plan,
                branches=branches,
                seed_id=seed,
                grad_clip_norm=config.grad_clip_norm,
            )
            for run_id in TPRED_BRANCH_RUN_IDS:
                candidate = branch_result["candidate_weights"][run_id]
                candidate["parent_hashes"] = parent_hashes | candidate["parent_hashes"]
                candidate_by_run_seed[run_id][seed] = candidate
                metrics_by_run_seed[run_id][seed] = {
                    "model_val_stop": _selected_metrics(branch_result["reports"][run_id]),
                    "optimizer_steps": branch_result["reports"][run_id]["optimizer_steps"],
                    "batch_plan_sha256": branch_result["reports"][run_id]["batch_plan_sha256"],
                }

        # One resident model_val_select view is shared across all condition
        # forwards and all three teacher recipes.
        matching_fit = stack.hlt.read_indices(
            consumer_indices, names=("tokens", "mask")
        )
        frozen_matching_bin_edges = fit_matched_bin_edges(
            matching_fit["tokens"], matching_fit["mask"]
        )
        evaluation_batches = _evaluation_batches(
            model_val,
            parent_indices=select_indices,
            batch_size=int(evaluation_batch_size),
            control_seed=9_181_101,
            frozen_bin_edges=frozen_matching_bin_edges,
        )
        evaluation_objects: dict[str, list[ConsumerReplicaEvaluation]] = {
            T10_CLEAN: [], T10_ROBUST: [], T10_ALL50_CLEAN: []
        }
        replica_root.mkdir(parents=True, exist_ok=False)
        for seed in PAIRED_SEED_IDS:
            # The matched-compute row is evaluated on f0 once per seed.
            matched_model = make_model(TPRED_CONTINUE, config).to(torch_device)
            matched_model.load_state_dict(candidate_by_run_seed[TPRED_CONTINUE][seed]["model_state_dict"], strict=True)
            matched_metrics = _quick_evaluate(
                matched_model,
                model_val,
                parent_indices=select_indices,
                run_id=TPRED_CONTINUE,
                device=torch_device,
                batch_size=evaluation_batch_size,
            )
            for run_id in (T10_CLEAN, T10_ROBUST, T10_ALL50_CLEAN):
                candidate = candidate_by_run_seed[run_id][seed]
                checkpoint_path = replica_root / f"{run_id}__seed{seed}.pt"
                torch.save(candidate, checkpoint_path)
                checkpoint_sha = sha256_file(checkpoint_path)
                model = make_model(run_id, config).to(torch_device)
                model.load_state_dict(candidate["model_state_dict"], strict=True)
                policy = (
                    BRIDGE_CHANNEL_ALL50 if run_id == T10_ALL50_CLEAN else BRIDGE_CHANNEL_PHYSICAL45
                )
                callback = _BoundConditionForward(
                    model=model,
                    checkpoint_sha256=checkpoint_sha,
                    channel_policy=policy,
                    device=torch_device,
                    control_seed=9_181_101,
                )
                recipe = all50_recipe if policy == BRIDGE_CHANNEL_ALL50 else physical_recipe
                evaluation = evaluate_bound_consumer_conditions(
                    run_id=run_id,
                    seed_id=seed,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha,
                    recipe_sha256=recipe["content_hash"],
                    split_sha256=child["children"]["model_val_select"]["content_hash"],
                    ram_audit_sha256=stage_report["content_hash"],
                    class_order=spec["class_names"],
                    batches=evaluation_batches,
                    forward_fn=callback,
                    matched_compute_f0_accuracy=matched_metrics["accuracy"],
                    bootstrap_resamples=int(bootstrap_resamples),
                )
                evaluation_objects[run_id].append(evaluation)
                metrics_by_run_seed[run_id][seed]["model_val_select"] = {
                    "bridge_accuracy": evaluation.artifact["bridge_0p10"]["accuracy"],
                    "bridge_cross_entropy": evaluation.artifact["bridge_0p10"]["cross_entropy"],
                    "f0_accuracy": evaluation.artifact["f0"]["accuracy"],
                    "same_consumer_bridge_gain": evaluation.artifact["delta_same"],
                }

        # Persist all RAM-local replica pairs only after all scientific checks.
        for run_id in STEP3_RUN_IDS:
            for seed in PAIRED_SEED_IDS:
                checkpoint_path = replica_root / f"{run_id}__seed{seed}.pt"
                if not checkpoint_path.exists():
                    torch.save(candidate_by_run_seed[run_id][seed], checkpoint_path)
                (replica_root / f"{run_id}__seed{seed}.metrics.json").write_text(
                    json.dumps(metrics_by_run_seed[run_id][seed], allow_nan=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        evidence_root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(evidence_root / "corruption_scale.json", corruption)
        aggregates = {}
        for run_id, evaluations in evaluation_objects.items():
            run_root = evidence_root / run_id
            run_root.mkdir()
            for evaluation in evaluations:
                write_immutable_json(
                    run_root / f"seed_{evaluation.artifact['seed_id']}.json",
                    evaluation.artifact,
                )
            aggregate = aggregate_consumer_evaluations(
                evaluations, bootstrap_resamples=int(bootstrap_resamples)
            )
            write_immutable_json(run_root / "selection_aggregate.json", aggregate)
            aggregates[run_id] = {
                "path": str(run_root / "selection_aggregate.json"),
                "content_hash": aggregate["content_hash"],
                "eligible": aggregate["eligible"],
            }
        return {
            "ok": True,
            "contract": PREDICTION_ANCHORED_CONSUMER_EXECUTION_CONTRACT,
            "execution_spec_sha256": spec["content_hash"],
            "r0_checkpoint_sha256": r0_sha,
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "run_ids": list(STEP3_RUN_IDS),
            "replica_count": len(STEP3_RUN_IDS) * len(PAIRED_SEED_IDS),
            "replica_output_dir": str(replica_root),
            "evaluation_output_dir": str(evidence_root),
            "selection_aggregates": aggregates,
            "one_open_per_compressed_source": bool(
                stage_report["all_persistent_npz_open_counts_equal_one"]
            ),
            "resident_field_banks_persistent": False,
            "persistent_dense_fields_written": False,
            "ram_peak_reserved_bytes": ledger.snapshot()["peak_reserved_bytes"],
        }
    finally:
        ledger.cleanup()


def confirm_selected_consumer_from_execution_spec(
    execution_spec_path: str | Path,
    *,
    preconfirmation_path: str | Path,
    r0_checkpoint_path: str | Path,
    r0_registration_path: str | Path,
    physical45_recipe_path: str | Path,
    output_dir: str | Path,
    ram_root: str | Path,
    allocation_id: str | None = None,
    device: str = "auto",
    batch_size: int = 512,
    shard_size: int = 8192,
    capacity_bytes: int | None = None,
    allow_unverified_test_root: bool = False,
    model_loader: Callable[[str | Path, Any], tuple[Any, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Open stack_val_consumer once and confirm the already locked median teacher."""

    import torch
    from .fusion import load_local_residual_field_tagger_from_checkpoint

    spec = load_hashed_json(
        execution_spec_path, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    validate_prediction_anchored_execution_spec(spec, verify_file_hashes=False)
    child = load_hashed_json(
        spec["child_manifest"]["path"], expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    preconfirmation = load_hashed_json(preconfirmation_path)
    if preconfirmation.get("status") != "LOCKED_AWAITING_STACK_VAL_CONSUMER":
        raise PermissionError("consumer preconfirmation is not locked for one-shot confirmation")
    recipe = load_hashed_json(physical45_recipe_path)
    validate_bridge_recipe(recipe)
    if recipe["content_hash"] != preconfirmation.get("bridge_recipe_sha256"):
        raise ValueError("preconfirmation and physical45 recipe differ")
    r0_path = Path(r0_checkpoint_path)
    r0_sha = sha256_file(r0_path)
    registration = load_hashed_json(r0_registration_path)
    if registration.get("checkpoint_sha256") != r0_sha:
        raise ValueError("R0 registration/checkpoint binding changed")
    checkpoint = Path(str(preconfirmation.get("checkpoint_path", "")))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError("selected median consumer checkpoint is absent or unsafe")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != preconfirmation.get("checkpoint_sha256"):
        raise ValueError("selected median consumer checkpoint bytes changed")
    unlock = build_manifest_validation_unlock(
        child,
        split_name="stack_val_consumer",
        selection_sha256=preconfirmation["content_hash"],
    )
    parent_hash, split_hash = split_binding(child, "stack_val_consumer")
    authorization = authorize_split_access(
        split_name="stack_val_consumer",
        purpose="consumer_confirmation",
        parent_manifest_sha256=parent_hash,
        bound_split_sha256=split_hash,
        unlock=unlock,
    )
    output = Path(output_dir)
    access_path = output / "access_receipts" / "stack_val_consumer.json"
    access_publication = claim_split_access(access_path, authorization)
    access = authorization
    torch_device = resolve_device(str(device))
    allocation = str(allocation_id or os.environ.get("SLURM_JOB_ID", "local_consumer_confirm"))
    ledger = AllocationRamLedger(
        ram_root,
        allocation_id=allocation,
        capacity_bytes=capacity_bytes,
        allow_unverified_test_root=bool(allow_unverified_test_root),
    )
    try:
        stager = AllocationNpzStager(ledger, rank=0, world_size=1)
        staged, combined = stager.stage_named_pairs(
            {"stack_val": _paths_from_source(spec["sources"]["stack_val"])},
            shard_size=int(shard_size),
        )
        hlt, offline, stage_report = staged["stack_val"]
        _verify_staged_source_binding(
            split="stack_val", source=spec["sources"]["stack_val"], report=stage_report
        )
        r0 = FrozenR0Runner(r0_path, device=torch_device)
        loader = model_loader or (
            lambda path, selected_device: load_local_residual_field_tagger_from_checkpoint(
                path, device=selected_device
            )
        )
        model, _ = loader(checkpoint, torch_device)
        indices = np.asarray(
            child["children"]["stack_val_consumer"]["parent_row_indices"], dtype=np.int64
        )
        f0_logits = []
        bridge_logits = []
        label_parts = []
        model.eval()
        with torch.no_grad():
            for start in range(0, indices.size, int(batch_size)):
                selected = indices[start : start + int(batch_size)]
                hlt_batch = hlt.read_indices(selected, names=("tokens", "mask", "labels"))
                offline_batch = offline.read_indices(selected, names=("tokens", "mask"))
                truth, target_mask, _, _, _ = compute_local_particle_residual_fields(
                    hlt_batch["tokens"], hlt_batch["mask"],
                    offline_batch["tokens"], offline_batch["mask"],
                    radii=DEFAULT_LOCAL_RESIDUAL_RADII,
                )
                anchor, _ = r0.predict_numpy(hlt_batch["tokens"], hlt_batch["mask"])
                if not np.array_equal(target_mask, hlt_batch["mask"]):
                    raise ValueError("stack_val_consumer target mask changed")
                bridge = virtual_bridge(
                    anchor, truth, target_mask, rho="0.100", channel_policy=BRIDGE_CHANNEL_PHYSICAL45
                )
                logits_for = []
                for fields in (anchor, bridge):
                    tensor_batch = build_consumer_tensor_batch(
                        tokens=hlt_batch["tokens"], mask=hlt_batch["mask"], labels=hlt_batch["labels"],
                        f0=fields, f_true=fields, run_id=TPRED, device=torch_device,
                    )
                    logits_for.append(_model_logits(model, tensor_batch).detach().float().cpu().numpy())
                f0_logits.append(logits_for[0])
                bridge_logits.append(logits_for[1])
                label_parts.append(hlt_batch["labels"])
        labels = np.concatenate(label_parts)
        f0_metrics = classification_metrics(
            np.concatenate(f0_logits), labels, class_order=spec["class_names"]
        )
        bridge_metrics = classification_metrics(
            np.concatenate(bridge_logits), labels, class_order=spec["class_names"]
        )
        confirmation_metrics = with_content_hash(
            {
                "contract": "prediction_anchored_consumer_confirmation_metrics_v1",
                "checkpoint_sha256": checkpoint_sha,
                "bridge_recipe_sha256": recipe["content_hash"],
                "split_sha256": split_hash,
                "access_receipt_sha256": access["content_hash"],
                "event_count": int(indices.size),
                "f0_accuracy": f0_metrics["accuracy"],
                "bridge_0p10_accuracy": bridge_metrics["accuracy"],
                "f0_cross_entropy": f0_metrics["cross_entropy"],
                "bridge_0p10_cross_entropy": bridge_metrics["cross_entropy"],
                "provenance_valid": True,
                "same_checkpoint_both_conditions": True,
                "one_open_per_compressed_source": combined[
                    "all_persistent_npz_open_counts_equal_one"
                ],
                "persistent_per_event_outcomes_written": False,
            }
        )
        write_immutable_json(output / "consumer_confirmation_metrics.json", confirmation_metrics)
        final = finalize_consumer_confirmation(
            preconfirmation,
            confirmation_metrics,
            access_receipt=access,
            output_dir=output,
        )
        return {
            "ok": final.get("status") == "CONFIRMED_LOCKED",
            "status": final.get("status"),
            "selected_consumer": (
                str(output / "selected_bridge_consumer.json")
                if final.get("status") == "CONFIRMED_LOCKED"
                else None
            ),
            "confirmation_metrics": str(output / "consumer_confirmation_metrics.json"),
            "access_receipt": str(access_path),
            "access_receipt_publication": access_publication,
            "f0_accuracy": f0_metrics["accuracy"],
            "bridge_0p10_accuracy": bridge_metrics["accuracy"],
            "persistent_dense_fields_written": False,
        }
    finally:
        ledger.cleanup()


__all__ = [
    "PREDICTION_ANCHORED_CONSUMER_EXECUTION_CONTRACT",
    "confirm_selected_consumer_from_execution_spec",
    "run_consumer_campaign_from_execution_spec",
]
