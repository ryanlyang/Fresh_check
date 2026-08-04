"""Fixed-budget, resumable Stage-D auxiliary training and evaluation."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.determinism import (
    optimizer_update_counts,
    scheduled_learning_rate,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)

from .auxiliary import (
    PAIR_TARGETS,
    auxiliary_objective,
    build_sampled_pair_batch,
)
from .baselines import HOSDTrainingProtocol
from .contracts import (
    AUXILIARY_CHECKPOINT_CONTRACT,
    AUXILIARY_COMPLETION_CONTRACT,
    AUXILIARY_PREDICTION_CONTRACT,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .stage_c_training import _balanced_metrics, _move, _restore_rng_state, _rng_state

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _forward_inputs(batch: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    vectors = batch.get("lorentz_vectors", batch.get("vectors"))
    if vectors is None:
        raise ValueError("Stage-D batch lacks Lorentz vectors")
    points = batch.get("points")
    if points is None:
        points = batch["features"][:, 15:17]
    return points, batch["features"], vectors, batch["mask"]


def _preferred_epoch(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        rows,
        key=lambda row: (
            -float(row["val_stop"]["balanced_accuracy"]),
            float(row["val_stop"]["cross_entropy"]),
            int(row["epoch"]),
        ),
    )


def _finite_gradient_norm(
    loss: Any, parameters: Sequence[Any], *, retain_graph: bool
) -> float:
    active = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not active or not bool(loss.requires_grad):
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        active,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    value = float(torch.sqrt(squared).cpu())
    if not math.isfinite(value):
        raise FloatingPointError("Stage-D diagnostic gradient is nonfinite")
    return value


def _loss_for_batch(
    *,
    model: Any,
    batch: Mapping[str, Any],
    row: Mapping[str, Any],
    component_group_ids: Sequence[str],
    epoch: int,
    sample_pairs: bool,
) -> tuple[Any, dict[str, Any], Any]:
    target = batch["target"]
    target_mask = batch["target_mask"].bool()
    sampled = None
    sampled_indices = None
    if row["target_id"] in PAIR_TARGETS and sample_pairs:
        sampled = build_sampled_pair_batch(
            target,
            target_mask,
            identities=[str(value) for value in batch["identities"]],
            epoch=epoch,
            target_id=row["target_id"],
        )
        sampled_indices = sampled.indices()
    if hasattr(model, "forward_with_feedback"):
        logits, prediction = model.forward_with_feedback(
            *_forward_inputs(batch),
            direct_pair_features=(
                batch.get("direct_pair_features")
                if batch.get("direct_pair_features") is not None
                else None
            ),
            direct_pair_mask=batch.get("direct_pair_mask"),
            raw_tokens=batch.get("raw_tokens"),
            region_trees=batch.get("region_trees"),
            oracle_feedback=batch.get("oracle_feedback"),
            predicted_feedback_override=batch.get("predicted_feedback_override"),
        )
        if sampled_indices is not None:
            index = (
                sampled_indices["event_indices"].long(),
                sampled_indices["left_indices"].long(),
                sampled_indices["right_indices"].long(),
            )
            prediction = {**prediction, "value": prediction["value"][index]}
    else:
        logits, prediction = model.forward_with_aux(
            *_forward_inputs(batch), sampled_pair_indices=sampled_indices
        )
    if not bool(row.get("semantic_loss_enabled", True)):
        classification = torch.nn.functional.cross_entropy(
            logits, batch["labels"].long()
        )
        auxiliary = classification * 0.0
        total = classification
        pieces = {
            "classification_loss": classification,
            "auxiliary_loss": auxiliary,
            "weighted_auxiliary_loss": auxiliary,
            "total_loss": total,
            "semantic_loss_omitted": True,
        }
    else:
        total, pieces = auxiliary_objective(
            logits=logits,
            labels=batch["labels"],
            prediction=prediction,
            target=sampled.target if sampled is not None else target,
            target_mask=sampled.mask if sampled is not None else target_mask,
            target_id=row["target_id"],
            parameterization=row["parameterization"],
            auxiliary_weight=float(row["auxiliary_weight"]),
            component_group_ids=component_group_ids,
            sampled_pair_batch=sampled,
        )
    return total, pieces, logits


def evaluate_auxiliary(
    model: Any,
    loader: Any,
    *,
    row: Mapping[str, Any],
    component_group_ids: Sequence[str],
    split: str,
    device: Any,
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    model.to(resolved_device)
    was_training = bool(model.training)
    model.eval()
    logits, labels = [], []
    auxiliary_sum, event_sum = 0.0, 0
    try:
        with torch.no_grad():
            for raw in loader:
                batch = _move(raw, resolved_device)
                _, pieces, value = _loss_for_batch(
                    model=model,
                    batch=batch,
                    row=row,
                    component_group_ids=component_group_ids,
                    epoch=1,
                    sample_pairs=False,
                )
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Stage-D evaluation logits are nonfinite")
                events = int(batch["labels"].numel())
                logits.append(value.float().cpu().numpy())
                labels.append(batch["labels"].cpu().numpy())
                auxiliary_sum += float(pieces["auxiliary_loss"].cpu()) * events
                event_sum += events
    finally:
        if was_training:
            model.train()
    values = np.concatenate(logits)
    truth = np.concatenate(labels)
    classification = (
        evaluate_classification(values, truth, split=split)
        if split in {"design_select", "design_confirm"}
        else _balanced_metrics(values, truth)
    )
    return {
        "classification_metrics": classification,
        "auxiliary_loss": auxiliary_sum / event_sum,
        "event_count": event_sum,
        "pair_evaluation": (
            "all_applicable_pairs"
            if row["target_id"] in PAIR_TARGETS
            else "not_applicable"
        ),
    }


def train_stage_d_auxiliary(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    design_select_loader: Any,
    output_dir: str | Path,
    row: Mapping[str, Any],
    component_group_ids: Sequence[str],
    stage_d_plan_sha256: str,
    campaign_spec_sha256: str,
    lineage_hashes: Mapping[str, str],
    protocol: HOSDTrainingProtocol,
    source: Mapping[str, Any],
    deployed_analytical_flops: float,
    deployed_parameter_count: int | None = None,
    deployed_operation_profile: Mapping[str, Any] | None = None,
    device: str | Any = "cpu",
    resume: bool = True,
    training_gpu_hours_override: float | None = None,
    checkpoint_contract: str = AUXILIARY_CHECKPOINT_CONTRACT,
    completion_contract: str = AUXILIARY_COMPLETION_CONTRACT,
    prediction_contract: str = AUXILIARY_PREDICTION_CONTRACT,
    prediction_schema_version: int = 1,
    completion_schema_version: int = 1,
    plan_hash_field: str = "stage_d_plan_sha256",
    stage_label: str = "Stage-D",
    completion_filename: str = "auxiliary_completion.json",
    curves_contract: str = "hosd_auxiliary_training_curves_v1",
    evaluation_split: str = "design_select",
) -> dict[str, Any]:
    """Run every epoch; poor target or tagging metrics never terminate a row."""

    if torch is None:
        raise RuntimeError("PyTorch is required for Stage-D training")
    protocol.validate()
    if evaluation_split not in {"design_select", "design_confirm"}:
        raise ValueError(f"{stage_label} evaluation split differs")
    evaluation_result_filename = f"{evaluation_split}_result.json"
    evaluation_hash_field = f"{evaluation_split}_result_sha256"
    if not bool(row.get("resolved")):
        raise ValueError("Stage-D trainer cannot execute an unresolved row")
    if (
        not math.isfinite(float(deployed_analytical_flops))
        or float(deployed_analytical_flops) <= 0
    ):
        raise ValueError("Stage-D deployed analytical FLOPs must be finite and positive")
    operation_profile = (
        None
        if deployed_operation_profile is None
        else dict(deployed_operation_profile)
    )
    operation_profile_sha256 = None
    if operation_profile is not None:
        validate_content_hash(operation_profile)
        operation_profile_sha256 = require_sha256(
            operation_profile["content_hash"], name="deployed_operation_profile"
        )
    if row["row_kind"] in {"TARGET_MEAN", "GLOBAL_SHUFFLE", "WITHIN_CLASS_SHUFFLE"}:
        expected_control = {
            "TARGET_MEAN": "target_mean",
            "GLOBAL_SHUFFLE": "global_shuffle",
            "WITHIN_CLASS_SHUFFLE": "within_class_shuffle",
        }[row["row_kind"]]
        observed = getattr(train_loader.dataset, "control_kind", None)
        if observed != expected_control:
            raise ValueError("Stage-D null row did not receive its distinct control cache")
    checked_lineage = {
        str(key): require_sha256(value, name=f"lineage.{key}")
        for key, value in sorted(lineage_hashes.items())
    }
    task_component_seed = int(
        row.get(
            "head_component_seed",
            row.get("feedback_component_seed", -1),
        )
    )
    if task_component_seed < 0:
        raise ValueError(f"{stage_label} row lacks its task component seed")
    dataset = train_loader.dataset
    while not hasattr(dataset, "logical_role") and hasattr(dataset, "base_dataset"):
        dataset = dataset.base_dataset
    training_role = str(getattr(dataset, "logical_role", "model_train"))
    from .stage_d_data_factory import (
        DATA_ORDER_CONTRACT,
        data_order_seed,
        sampler_contract,
    )

    training_sampler_seed = data_order_seed(int(row["pipeline_seed"]), training_role)
    observed_sampler_seed = int(
        getattr(train_loader.sampler, "seed", training_sampler_seed)
    )
    if observed_sampler_seed != training_sampler_seed:
        raise ValueError(f"{stage_label} sampler order differs from pipeline contract")
    training_sampler_contract = str(
        getattr(train_loader.sampler, "contract", sampler_contract(training_role))
    )
    if training_sampler_contract != sampler_contract(training_role):
        raise ValueError(f"{stage_label} sampler implementation differs")
    if not checked_lineage:
        raise ValueError("Stage-D training lineage must not be empty")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    best_path, last_path = root / "best_model_val.pt", root / "last.pt"
    completion_path = root / completion_filename
    if completion_path.exists():
        from .contracts import load_hashed_json

        completion = load_hashed_json(
            completion_path, expected_contract=completion_contract
        )
        if (
            completion.get("source") != dict(source)
            or completion.get("row_id") != row["row_id"]
            or completion.get(plan_hash_field) != stage_d_plan_sha256
            or completion.get("campaign_spec_sha256") != campaign_spec_sha256
            or completion.get("lineage_hashes") != checked_lineage
            or int(completion.get("pipeline_seed", -1))
            != int(row["pipeline_seed"])
            or int(completion.get("encoder_component_seed", -1))
            != int(row["encoder_component_seed"])
            or int(completion.get("task_component_seed", -1))
            != task_component_seed
            or completion.get("data_order_contract") != DATA_ORDER_CONTRACT
            or completion.get("training_sampler_contract")
            != training_sampler_contract
            or int(completion.get("training_sampler_seed", -1))
            != training_sampler_seed
            or completion.get("deployed_operation_profile_sha256")
            != operation_profile_sha256
        ):
            raise ValueError("reusable Stage-D completion lineage differs")
        checkpoint = root / completion["checkpoint_file"]
        if (
            not checkpoint.is_file()
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != completion["checkpoint_sha256"]
        ):
            raise ValueError("reusable Stage-D checkpoint bytes differ")
        result = load_hashed_json(
            root / evaluation_result_filename,
            expected_contract=prediction_contract,
        )
        if (
            result["content_hash"] != completion[evaluation_hash_field]
            or float(result["deployed_analytical_flops"])
            != float(deployed_analytical_flops)
            or (
                deployed_parameter_count is not None
                and int(result["deployed_parameter_count"])
                != int(deployed_parameter_count)
            )
            or (
                training_gpu_hours_override is not None
                and float(result["training_gpu_hours"])
                != float(training_gpu_hours_override)
            )
            or result.get("deployed_operation_profile") != operation_profile
        ):
            raise ValueError("reusable Stage-D result/capacity semantics differ")
        return completion

    resolved = torch.device(device)
    model.to(resolved)
    random.seed(task_component_seed)
    np.random.seed(task_component_seed % (2**32))
    torch.manual_seed(task_component_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.base_learning_rate,
        betas=(protocol.beta1, protocol.beta2),
        weight_decay=protocol.weight_decay,
    )
    counts = optimizer_update_counts(
        training_event_count=len(train_loader.dataset),
        maximum_epochs=protocol.maximum_epochs,
        microbatch_size=protocol.microbatch_size,
        gradient_accumulation_steps=protocol.gradient_accumulation_steps,
    )
    rows: list[dict[str, Any]] = []
    candidate_states: dict[int, dict[str, Any]] = {}
    start_epoch, update_ordinal = 1, 0
    started = time.perf_counter()
    elapsed_before = 0.0
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        expected = {
            "contract": checkpoint_contract,
            "row_id": row["row_id"],
            plan_hash_field: stage_d_plan_sha256,
            "campaign_spec_sha256": campaign_spec_sha256,
            "source": dict(source),
            "lineage_hashes": checked_lineage,
            "pipeline_seed": int(row["pipeline_seed"]),
            "encoder_component_seed": int(row["encoder_component_seed"]),
            "task_component_seed": task_component_seed,
            "data_order_contract": DATA_ORDER_CONTRACT,
            "training_sampler_contract": training_sampler_contract,
            "training_sampler_seed": training_sampler_seed,
            "deployed_operation_profile_sha256": operation_profile_sha256,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{stage_label} resume lineage differs")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        rows = list(state["rows"])
        candidate_states = {
            int(key): value for key, value in state["candidate_states"].items()
        }
        start_epoch = int(state["epoch_completed"]) + 1
        update_ordinal = int(state["optimizer_update_ordinal"])
        elapsed_before = float(state.get("elapsed_training_seconds", 0.0))
        _restore_rng_state(state["rng_state"])

    for epoch in range(start_epoch, protocol.maximum_epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_events = 0
        objective_sum, class_sum, aux_sum, event_sum = 0.0, 0.0, 0.0, 0
        gradient_diagnostic = None
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, resolved)
            if hasattr(model, "set_update"):
                model.set_update(
                    min(
                        update_ordinal + 1,
                        counts["total_optimizer_updates"],
                    ),
                    counts["total_optimizer_updates"],
                )
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                loss, pieces, _ = _loss_for_batch(
                    model=model,
                    batch=batch,
                    row=row,
                    component_group_ids=component_group_ids,
                    epoch=epoch,
                    sample_pairs=True,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Stage-D training objective is nonfinite")
            if batch_index == 1:
                shared = model.shared_parameters()
                head = model.head_parameters()
                gradient_diagnostic = {
                    "classification_shared_gradient_norm": _finite_gradient_norm(
                        pieces["classification_loss"], shared, retain_graph=True
                    ),
                    "auxiliary_shared_gradient_norm": _finite_gradient_norm(
                        pieces["auxiliary_loss"], shared, retain_graph=True
                    ),
                    "auxiliary_head_gradient_norm": _finite_gradient_norm(
                        pieces["auxiliary_loss"], head, retain_graph=True
                    ),
                    "stop_encoder": row["row_kind"] == "STOP_ENCODER",
                }
                if (
                    row["row_kind"] == "STOP_ENCODER"
                    and gradient_diagnostic["auxiliary_shared_gradient_norm"] != 0.0
                ):
                    raise RuntimeError("STOP_ENCODER leaked auxiliary shared gradients")
            events = int(batch["labels"].numel())
            (loss * events).backward()
            objective_sum += float(loss.detach().cpu()) * events
            class_sum += float(pieces["classification_loss"].detach().cpu()) * events
            aux_sum += float(pieces["auxiliary_loss"].detach().cpu()) * events
            event_sum += events
            accumulated_events += events
            step_now = (
                batch_index % protocol.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not step_now:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_events)
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError("Stage-D training gradient is nonfinite")
            update_ordinal += 1
            learning_rate = scheduled_learning_rate(
                update_ordinal=update_ordinal,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=protocol.base_learning_rate,
                minimum_learning_rate=protocol.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_events = 0
        validation = evaluate_auxiliary(
            model,
            val_stop_loader,
            row=row,
            component_group_ids=component_group_ids,
            split="val_stop",
            device=resolved,
        )
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update_ordinal": update_ordinal,
                "train_objective": objective_sum / event_sum,
                "train_classification_loss": class_sum / event_sum,
                "train_auxiliary_loss": aux_sum / event_sum,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "val_stop": validation["classification_metrics"],
                "val_stop_auxiliary_loss": validation["auxiliary_loss"],
                "gradient_path_diagnostic": gradient_diagnostic,
            }
        )
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        candidate_states[epoch] = state
        selected = _preferred_epoch(rows)
        selected_epoch = int(selected["epoch"])
        candidate_states = {
            key: value
            for key, value in candidate_states.items()
            if key in {selected_epoch, epoch}
        }
        _atomic_torch_save(
            {
                "contract": checkpoint_contract,
                "schema_version": 1,
                "kind": "selected_inference",
                "row_id": row["row_id"],
                "target_id": row["target_id"],
                "source": dict(source),
                "lineage_hashes": checked_lineage,
                plan_hash_field: stage_d_plan_sha256,
                "campaign_spec_sha256": campaign_spec_sha256,
                "pipeline_seed": int(row["pipeline_seed"]),
                "encoder_component_seed": int(row["encoder_component_seed"]),
                "task_component_seed": task_component_seed,
                "data_order_contract": DATA_ORDER_CONTRACT,
                "training_sampler_contract": training_sampler_contract,
                "training_sampler_seed": training_sampler_seed,
                "deployed_operation_profile_sha256": operation_profile_sha256,
                "epoch": selected_epoch,
                "model_state_dict": candidate_states[selected_epoch],
                "selection_metrics": selected["val_stop"],
            },
            best_path,
        )
        _atomic_torch_save(
            {
                "contract": checkpoint_contract,
                "schema_version": 1,
                "kind": "resumable_last",
                "row_id": row["row_id"],
                "source": dict(source),
                "lineage_hashes": checked_lineage,
                plan_hash_field: stage_d_plan_sha256,
                "campaign_spec_sha256": campaign_spec_sha256,
                "pipeline_seed": int(row["pipeline_seed"]),
                "encoder_component_seed": int(row["encoder_component_seed"]),
                "task_component_seed": task_component_seed,
                "data_order_contract": DATA_ORDER_CONTRACT,
                "training_sampler_contract": training_sampler_contract,
                "training_sampler_seed": training_sampler_seed,
                "deployed_operation_profile_sha256": operation_profile_sha256,
                "epoch_completed": epoch,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_update_ordinal": update_ordinal,
                "rows": rows,
                "candidate_states": candidate_states,
                "rng_state": _rng_state(),
                "elapsed_training_seconds": (
                    elapsed_before + time.perf_counter() - started
                ),
            },
            last_path,
        )

    selected = _preferred_epoch(rows)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    evaluation = evaluate_auxiliary(
        model,
        design_select_loader,
        row=row,
        component_group_ids=component_group_ids,
        split=evaluation_split,
        device=resolved,
    )
    checkpoint_sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    deployed_count = (
        sum(parameter.numel() for parameter in model.classifier.parameters())
        if deployed_parameter_count is None
        else int(deployed_parameter_count)
    )
    gpu_hours = (
        float(training_gpu_hours_override)
        if training_gpu_hours_override is not None
        else (elapsed_before + time.perf_counter() - started) / 3600.0
    )
    result = with_content_hash(
        {
            "contract": prediction_contract,
            "schema_version": int(prediction_schema_version),
            "source": dict(source),
            plan_hash_field: stage_d_plan_sha256,
            "campaign_spec_sha256": campaign_spec_sha256,
            "row_id": row["row_id"],
            "target_id": row["target_id"],
            "parameterization": row["parameterization"],
            "auxiliary_weight": row["auxiliary_weight"],
            "row_kind": row["row_kind"],
            "selection_eligible": bool(row["selection_eligible"]),
            "checkpoint_sha256": checkpoint_sha,
            evaluation_split: evaluation,
            "deployed_analytical_flops": float(deployed_analytical_flops),
            "deployed_parameter_count": deployed_count,
            "training_gpu_hours": gpu_hours,
            "pipeline_seed": int(row["pipeline_seed"]),
            "encoder_component_seed": int(row["encoder_component_seed"]),
            "task_component_seed": task_component_seed,
            "data_order_contract": DATA_ORDER_CONTRACT,
            "training_sampler_contract": training_sampler_contract,
            "training_sampler_seed": training_sampler_seed,
            "deployed_operation_profile": operation_profile,
            "target_error_cross_family_tie_breaker": False,
            **{
                key: row[key]
                for key in (
                    "interface",
                    "gradient_path",
                    "control",
                    "deployable",
                )
                if key in row
            },
        }
    )
    write_immutable_json(root / evaluation_result_filename, result)
    completion = with_content_hash(
        {
            "contract": completion_contract,
            "schema_version": int(completion_schema_version),
            "source": dict(source),
            "row_id": row["row_id"],
            "target_id": row["target_id"],
            "row_kind": row["row_kind"],
            plan_hash_field: require_sha256(
                stage_d_plan_sha256, name="stage_d_plan_sha256"
            ),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "lineage_hashes": checked_lineage,
            "pipeline_seed": int(row["pipeline_seed"]),
            "encoder_component_seed": int(row["encoder_component_seed"]),
            "task_component_seed": task_component_seed,
            "data_order_contract": DATA_ORDER_CONTRACT,
            "training_sampler_contract": training_sampler_contract,
            "training_sampler_seed": training_sampler_seed,
            "deployed_operation_profile_sha256": operation_profile_sha256,
            "checkpoint_file": best_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "selected_epoch": int(selected["epoch"]),
            "selected_val_stop": dict(selected["val_stop"]),
            evaluation_hash_field: result["content_hash"],
            "epochs_completed": len(rows),
            "optimizer_updates_completed": update_ordinal,
            "early_stopping": False,
            "performance_based_termination": False,
            "future_rows_cancelled_for_performance": False,
            "classification_isolated": True,
            "auxiliary_head_removed_from_deployed_counts": True,
            "hlt_only_inference": True,
            **{
                key: row[key]
                for key in (
                    "interface",
                    "gradient_path",
                    "control",
                    "deployable",
                )
                if key in row
            },
        }
    )
    write_immutable_json(
        root / "training_curves.json",
        with_content_hash(
            {
                "contract": curves_contract,
                "schema_version": 1,
                "source": dict(source),
                "row_id": row["row_id"],
                "rows": rows,
                "selected_epoch": int(selected["epoch"]),
                "fixed_epoch_budget": True,
            }
        ),
    )
    write_immutable_json(completion_path, completion)
    if last_path.exists():
        last_path.unlink()
    return completion


__all__ = ["evaluate_auxiliary", "train_stage_d_auxiliary"]
