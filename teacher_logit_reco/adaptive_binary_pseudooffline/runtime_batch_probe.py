"""Complete optimizer-step memory probe used by runtime batch calibration."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .config import canonical_hash
from .runtime_batch import FullStepBatchMeasurement, exact_accumulation_steps


def measure_full_optimizer_step(
    *,
    stage_family: str,
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
        for _ in range(accumulation):
            output = forward_loss(model, batch_factory(local))
            if not isinstance(output, Mapping) or "total_loss" not in output:
                raise TypeError(
                    "full-step probe forward must return a tensor mapping with total_loss"
                )
            if not all(torch.is_tensor(value) for value in output.values()):
                raise TypeError("full-step probe output mapping may contain only tensors")
            loss = output["total_loss"]
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("full-step probe total_loss is not a finite scalar")
            forward_completed = True
            (loss / float(accumulation)).backward()
        backward_completed = True
        active = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradients = [parameter.grad for parameter in active if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError("full-step probe gradients are absent or nonfinite")
        norm = torch.nn.utils.clip_grad_norm_(active, float(gradient_clip_norm))
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("full-step probe clipped gradient norm is nonfinite")
        clipped = True
        optimizer.step()
        stepped = True
        if hasattr(ema, "update"):
            ema.update(model.module if is_ddp else model)
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
        gathered: list[dict[str, Any] | None] = [None for _ in range(measured_world)]
        torch.distributed.all_gather_object(gathered, local_evidence)
        rank_evidence = tuple(dict(value or {}) for value in gathered)
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


__all__ = ["measure_full_optimizer_step"]
