"""Complete optimizer-step memory probe used by runtime batch calibration."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from .config import canonical_hash
from .runtime_batch import FullStepBatchMeasurement, exact_accumulation_steps
from .runtime_batch import ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER


ABPH_RUNTIME_PROBE_EVIDENCE_BYTES = 8192


def _distributed_success_consensus(succeeded: bool, *, device: Any) -> bool:
    """Return true only when every rank completed the current local phase."""

    import torch

    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return bool(succeeded)
    flag = torch.tensor(
        [1 if succeeded else 0],
        dtype=torch.int32,
        device=device,
    )
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _gather_fixed_evidence(
    local_evidence: Mapping[str, Any],
    *,
    measured_world: int,
    device: Any,
) -> tuple[dict[str, Any], ...]:
    """Gather bounded evidence without NCCL object-size negotiation."""

    import torch

    encoded = json.dumps(
        dict(local_evidence), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    maximum = int(ABPH_RUNTIME_PROBE_EVIDENCE_BYTES)
    if len(encoded) > maximum - 4:
        raise ValueError("runtime probe evidence exceeds its fixed control payload")
    payload = bytearray(maximum)
    payload[:4] = len(encoded).to_bytes(4, byteorder="little", signed=False)
    payload[4 : 4 + len(encoded)] = encoded
    local_tensor = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local_tensor) for _ in range(int(measured_world))]
    torch.distributed.all_gather(gathered, local_tensor)

    rows: list[dict[str, Any]] = []
    for rank, tensor in enumerate(gathered):
        raw = bytes(tensor.detach().cpu().tolist())
        length = int.from_bytes(raw[:4], byteorder="little", signed=False)
        if length > maximum - 4:
            raise RuntimeError(
                f"rank {rank} supplied an invalid runtime probe evidence length"
            )
        value = json.loads(raw[4 : 4 + length].decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"rank {rank} runtime probe evidence is not a mapping")
        rows.append(value)
    return tuple(rows)


def measure_full_optimizer_step(
    *,
    stage_family: str,
    variant_name: str,
    resolved_variant_config_hash: str,
    runtime_provenance_hash: str,
    slurm_job_id: str,
    slurm_job_account: str,
    slurm_job_partition: str,
    local_batch_size: int,
    requested_world_size: int,
    model: Any,
    optimizer: Any,
    ema: Any,
    batch_factory: Callable[[int], Any],
    forward_loss: Callable[[Any, Any], Mapping[str, Any]],
    active_parameter_groups: Sequence[str],
    device: Any,
    gradient_clip_norm: float,
    find_unused_parameters: bool,
    largest_path_exercised: bool,
    prefetch_buffers: Sequence[Any],
    pinned_memory_staging: bool,
) -> FullStepBatchMeasurement:
    """Run one full candidate update inside the caller's active DDP topology.

    ``forward_loss`` must return a standard tensor mapping with scalar
    ``total_loss``. Custom result objects are rejected so DDP can discover the
    differentiable output graph reliably.
    """

    import torch

    local = int(local_batch_size)
    accumulation = exact_accumulation_steps(
        stage_family,
        world_size=int(requested_world_size),
        local_batch_size=local,
    )
    distributed = bool(
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    measured_world = int(torch.distributed.get_world_size()) if distributed else 1
    backend = str(torch.distributed.get_backend()) if distributed else "none"
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        total_memory = 1

    optimizer.zero_grad(set_to_none=True)
    forward_completed = False
    backward_completed = False
    clipped = False
    stepped = False
    failure: str | None = None
    try:
        completed_forward_steps = 0
        completed_backward_steps = 0
        for _ in range(accumulation):
            output = None
            local_forward_failure: str | None = None
            try:
                output = forward_loss(model, batch_factory(local))
                if not isinstance(output, Mapping) or "total_loss" not in output:
                    raise TypeError(
                        "full-step probe forward must return a tensor mapping "
                        "with total_loss"
                    )

                def tensor_leaves(value: Any) -> bool:
                    if isinstance(value, Mapping):
                        return all(tensor_leaves(item) for item in value.values())
                    if isinstance(value, (tuple, list)):
                        return all(tensor_leaves(item) for item in value)
                    return bool(torch.is_tensor(value))

                if not tensor_leaves(output):
                    raise TypeError(
                        "full-step probe output mapping may contain only tensors"
                    )
                loss = output["total_loss"]
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        "full-step probe total_loss is not a finite scalar"
                    )
            except Exception as exc:
                local_forward_failure = f"{type(exc).__name__}: {exc}"

            all_forward_ok = _distributed_success_consensus(
                local_forward_failure is None,
                device=device,
            )
            if not all_forward_ok:
                failure = (
                    local_forward_failure
                    or "peer rank failed before synchronized backward"
                )
                break
            completed_forward_steps += 1

            local_backward_failure: str | None = None
            try:
                assert output is not None
                (output["total_loss"] / float(accumulation)).backward()
            except Exception as exc:
                local_backward_failure = f"{type(exc).__name__}: {exc}"
            all_backward_ok = _distributed_success_consensus(
                local_backward_failure is None,
                device=device,
            )
            if not all_backward_ok:
                failure = (
                    local_backward_failure
                    or "peer rank failed during synchronized backward"
                )
                break
            completed_backward_steps += 1

        forward_completed = completed_forward_steps == accumulation
        backward_completed = completed_backward_steps == accumulation
        if failure is None:
            active_named = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            active = [parameter for _name, parameter in active_named]
            gradients = [
                parameter.grad
                for parameter in active
                if parameter.grad is not None
            ]
            if not gradients:
                raise FloatingPointError("full-step probe gradients are absent")
            nonfinite = [
                (name, int((~torch.isfinite(parameter.grad)).sum().item()))
                for name, parameter in active_named
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            if nonfinite:
                preview = ", ".join(
                    f"{name}({count})" for name, count in nonfinite[:8]
                )
                raise FloatingPointError(
                    "full-step probe gradients are nonfinite: " + preview
                )
            norm = torch.nn.utils.clip_grad_norm_(
                active, float(gradient_clip_norm)
            )
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(
                    "full-step probe clipped gradient norm is nonfinite"
                )
            clipped = True
            optimizer.step()
            stepped = True
            if hasattr(ema, "update"):
                ema_source = model.module if is_ddp else model
                # DDP wraps ReconstructorTrainingModule, whose child ``model`` is
                # the reconstructor tracked by EMA.
                nested_model = getattr(ema_source, "model", None)
                if isinstance(nested_model, torch.nn.Module):
                    ema_source = nested_model
                ema.update(ema_source)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        optimizer.zero_grad(set_to_none=True)

    adam_state = bool(optimizer.state) and all(bool(state) for state in optimizer.state.values())
    if is_cuda:
        peak = int(torch.cuda.max_memory_reserved(device))
        free = max(0, total_memory - peak)
    else:
        peak = 0
        free = total_memory
    local_evidence = {
        "stage_family": stage_family,
        "local_batch_size": local,
        "forward_completed": forward_completed,
        "backward_completed": backward_completed,
        "gradients_clipped": clipped,
        "optimizer_step_completed": stepped,
        "adamw_state_materialized": adam_state,
        "peak_device_memory_bytes": peak,
        "free_device_memory_bytes_at_peak": free,
        "failure": failure,
    }
    local_hash = canonical_hash(local_evidence)
    if distributed:
        rank_evidence = _gather_fixed_evidence(
            local_evidence,
            measured_world=measured_world,
            device=device,
        )
        rank_hashes = tuple(canonical_hash(value) for value in rank_evidence)
        forward_completed = all(bool(value["forward_completed"]) for value in rank_evidence)
        backward_completed = all(bool(value["backward_completed"]) for value in rank_evidence)
        clipped = all(bool(value["gradients_clipped"]) for value in rank_evidence)
        stepped = all(bool(value["optimizer_step_completed"]) for value in rank_evidence)
        adam_state = all(bool(value["adamw_state_materialized"]) for value in rank_evidence)
        peak = max(int(value["peak_device_memory_bytes"]) for value in rank_evidence)
        free = min(int(value["free_device_memory_bytes_at_peak"]) for value in rank_evidence)
        failures = [str(value["failure"]) for value in rank_evidence if value["failure"]]
        failure = "; ".join(failures) if failures else None
    else:
        rank_hashes = (local_hash,)

    return FullStepBatchMeasurement(
        stage_family=stage_family,
        variant_name=str(variant_name),
        resolved_variant_config_hash=str(resolved_variant_config_hash),
        runtime_provenance_hash=str(runtime_provenance_hash),
        measurement_producer=ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER,
        slurm_job_id=str(slurm_job_id),
        slurm_job_account=str(slurm_job_account),
        slurm_job_partition=str(slurm_job_partition),
        local_batch_size=local,
        accumulation_steps=accumulation,
        requested_world_size=int(requested_world_size),
        measured_world_size=measured_world,
        rank_count=measured_world,
        distributed_backend=backend,
        successful=failure is None and stepped,
        all_ranks_completed=failure is None,
        process_group_initialized=distributed,
        ddp_gradient_buckets_initialized=is_ddp,
        find_unused_parameters_exercised=(
            bool(find_unused_parameters) if requested_world_size > 1 else True
        ),
        active_parameter_groups=tuple(str(value) for value in active_parameter_groups),
        forward_completed=forward_completed,
        backward_completed=backward_completed,
        gradients_unscaled=backward_completed,
        gradients_clipped=clipped,
        optimizer_step_completed=stepped,
        adamw_state_materialized=adam_state,
        online_model_resident=True,
        ema_model_resident=bool(getattr(ema, "shadow", None)),
        prefetch_buffer_count=len(prefetch_buffers),
        pinned_memory_staging=bool(pinned_memory_staging),
        largest_path_exercised=bool(largest_path_exercised),
        total_device_memory_bytes=total_memory,
        peak_device_memory_bytes=peak,
        free_device_memory_bytes_at_peak=free,
        rank_measurement_hashes=rank_hashes,
        failure=failure,
    )


__all__ = [
    "ABPH_RUNTIME_PROBE_EVIDENCE_BYTES",
    "measure_full_optimizer_step",
]
