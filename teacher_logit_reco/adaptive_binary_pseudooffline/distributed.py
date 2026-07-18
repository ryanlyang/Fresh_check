"""Distributed optimization primitives for adaptive-binary reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import os
import socket
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch

from .runtime_profile import profile_span


ABPH_DISTRIBUTED_RUNTIME_CONTRACT = "adaptive_binary_distributed_runtime_v1"
ABPH_DDP_FORWARD_CONTRACT = "adaptive_binary_ddp_tensor_forward_v1"
ABPH_DDP_ERROR_SUMMARY_LIMIT = 1024


def _environment_integer(*names: str, default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip():
            return int(raw)
    return int(default)


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device_type: str
    initialized_here: bool = False

    def __post_init__(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("distributed rank/world size is invalid")
        if self.local_rank < 0:
            raise ValueError("distributed local rank is invalid")
        if self.world_size > 1 and self.backend not in {"nccl", "gloo"}:
            raise ValueError("distributed backend must be nccl or gloo")
        if self.world_size == 1 and self.backend not in {"none", "nccl", "gloo"}:
            raise ValueError("single-rank backend is invalid")

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": ABPH_DISTRIBUTED_RUNTIME_CONTRACT,
            **asdict(self),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node_id": os.environ.get("SLURM_NODEID"),
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
        }


def distributed_environment(*, requested_world_size: int | None = None) -> tuple[int, int, int]:
    world_size = _environment_integer(
        "WORLD_SIZE", "SLURM_NTASKS", default=requested_world_size or 1
    )
    rank = _environment_integer("RANK", "SLURM_PROCID", default=0)
    local_rank = _environment_integer("LOCAL_RANK", "SLURM_LOCALID", default=0)
    if requested_world_size is not None and world_size != int(requested_world_size):
        raise RuntimeError(
            f"observed distributed world size {world_size}, expected {requested_world_size}"
        )
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise RuntimeError("distributed environment rank topology is invalid")
    return rank, world_size, local_rank


def initialize_distributed_runtime(
    *, requested_world_size: int, device: Any
) -> DistributedRuntime:
    torch = require_torch()
    rank, world_size, local_rank = distributed_environment(
        requested_world_size=int(requested_world_size)
    )
    device_type = str(getattr(device, "type", device))
    if world_size == 1:
        return DistributedRuntime(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend="none",
            device_type=device_type,
        )
    if not torch.distributed.is_available():
        raise RuntimeError("PyTorch distributed support is unavailable")
    if device_type != "cuda":
        backend = "gloo"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL ABPH training requires CUDA")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    initialized_here = False
    if not torch.distributed.is_initialized():
        for name in ("MASTER_ADDR", "MASTER_PORT"):
            if not os.environ.get(name):
                raise RuntimeError(f"distributed ABPH runtime requires {name}")
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method="env://",
            timeout=timedelta(
                seconds=int(os.environ.get("ABPH_DDP_TIMEOUT_SECONDS", "1800"))
            ),
        )
        initialized_here = True
    observed_backend = str(torch.distributed.get_backend())
    if (
        int(torch.distributed.get_rank()) != rank
        or int(torch.distributed.get_world_size()) != world_size
        or observed_backend != backend
    ):
        raise RuntimeError("initialized process group differs from the requested runtime")
    return DistributedRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
        device_type=device_type,
        initialized_here=initialized_here,
    )


def destroy_distributed_runtime(runtime: DistributedRuntime) -> None:
    torch = require_torch()
    if (
        runtime.distributed
        and runtime.initialized_here
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def abort_distributed_runtime(runtime: DistributedRuntime) -> None:
    """Tear down a failed process group without entering another collective."""

    torch = require_torch()
    if not runtime.distributed or not torch.distributed.is_initialized():
        return
    try:
        default_group = torch.distributed.distributed_c10d._get_default_group()
        abort = getattr(default_group, "abort", None)
        if callable(abort):
            abort()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def barrier(runtime: DistributedRuntime) -> None:
    torch = require_torch()
    if runtime.distributed:
        torch.distributed.barrier()


def all_gather_objects(
    runtime: DistributedRuntime, value: Any
) -> tuple[Any, ...]:
    """Gather one pickle-safe control payload from every rank in rank order."""

    if not runtime.distributed:
        return (value,)
    torch = require_torch()
    gathered: list[Any] = [None] * runtime.world_size
    torch.distributed.all_gather_object(gathered, value)
    return tuple(gathered)


def broadcast_object(
    runtime: DistributedRuntime, value: Any, *, source_rank: int = 0
) -> Any:
    """Broadcast a rank-zero control/checkpoint payload to every rank."""

    if not runtime.distributed:
        return value
    torch = require_torch()
    values = [value if runtime.rank == int(source_rank) else None]
    torch.distributed.broadcast_object_list(values, src=int(source_rank))
    return values[0]


def all_reduce_float64_pair(
    runtime: DistributedRuntime,
    numerator: float,
    denominator: float,
    *,
    device: Any,
) -> tuple[float, float]:
    """Reduce a reviewed additive numerator/denominator pair exactly once."""

    if not runtime.distributed:
        return float(numerator), float(denominator)
    torch = require_torch()
    tensor = torch.tensor(
        [float(numerator), float(denominator)],
        dtype=torch.float64,
        device=device if runtime.backend == "nccl" else "cpu",
    )
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor[0].item()), float(tensor[1].item())


def all_reduce_min_bool(
    runtime: DistributedRuntime, value: bool, *, device: Any
) -> bool:
    torch = require_torch()
    if not runtime.distributed:
        return bool(value)
    tensor = torch.tensor(
        1 if value else 0,
        dtype=torch.int32,
        device=device if runtime.backend == "nccl" else "cpu",
    )
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return bool(int(tensor.item()))


def all_reduce_sum_int(
    runtime: DistributedRuntime, value: int, *, device: Any
) -> int:
    torch = require_torch()
    if not runtime.distributed:
        return int(value)
    tensor = torch.tensor(
        int(value),
        dtype=torch.int64,
        device=device if runtime.backend == "nccl" else "cpu",
    )
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def gather_error_summaries(
    runtime: DistributedRuntime,
    *,
    phase: str,
    error: BaseException | str | None,
    structural: bool,
) -> tuple[dict[str, Any], ...]:
    local = {
        "rank": runtime.rank,
        "phase": str(phase),
        "structural": bool(structural),
        "error_type": None if error is None else type(error).__name__,
        "message": None
        if error is None
        else str(error)[:ABPH_DDP_ERROR_SUMMARY_LIMIT],
    }
    if not runtime.distributed:
        return (local,)
    torch = require_torch()
    gathered: list[dict[str, Any] | None] = [None] * runtime.world_size
    torch.distributed.all_gather_object(gathered, local)
    return tuple(dict(item) for item in gathered if item is not None)


def any_structural_error(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(bool(row.get("structural")) for row in rows)


def parameter_state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_common_parameter_state(
    runtime: DistributedRuntime, model: Any
) -> str:
    local_hash = parameter_state_hash(model)
    if not runtime.distributed:
        return local_hash
    torch = require_torch()
    gathered: list[str | None] = [None] * runtime.world_size
    torch.distributed.all_gather_object(gathered, local_hash)
    if any(value != local_hash for value in gathered):
        raise RuntimeError("ABPH model parameter hashes differ across ranks")
    return local_hash


def _detach_metadata(value: Any) -> Any:
    torch = require_torch()
    if isinstance(value, Mapping):
        return {str(key): _detach_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_detach_metadata(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor
    if isinstance(value, np.generic):
        return value.item()
    return value


class ReconstructorTrainingModule(require_torch().nn.Module):
    """DDP-visible module that owns both model forward and objective composition."""

    def __init__(
        self,
        model: Any,
        step_function: Callable[[Any, Any, Any], Any],
        compose_function: Callable[[Any, Any, Any], Any],
        loss_weights: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.step_function = step_function
        self.compose_function = compose_function
        self.loss_weights = loss_weights
        self.last_metadata: dict[str, Any] | None = None

    def forward(self, batch: Any, context: Any) -> Mapping[str, Any]:
        torch = require_torch()
        self.last_metadata = None
        result = self.step_function(self.model, batch, context)
        parameter = next(self.model.parameters())
        with profile_span(getattr(context, "runtime_profiler", None), "loss_composition"):
            with torch.autocast(device_type=parameter.device.type, enabled=False):
                composed = self.compose_function(result, context, self.loss_weights)
        finite_tensors = (
            composed.total,
            *tuple(composed.raw_terms.values()),
            *tuple(composed.weighted_terms.values()),
            *tuple(
                torch.as_tensor(value).detach() for value in result.tensors_to_check
            ),
        )
        batch_size_tensor = torch.tensor(
            int(result.batch_size), dtype=torch.int64, device=composed.total.device
        )
        self.last_metadata = {
            "contract": ABPH_DDP_FORWARD_CONTRACT,
            "required_terms": tuple(composed.required_terms),
            "effective_weights": dict(composed.effective_weights),
            "metrics": _detach_metadata(result.metrics),
            "batch_size": int(result.batch_size),
        }
        return {
            "total_loss": composed.total,
            "raw_loss_terms": dict(composed.raw_terms),
            "weighted_loss_terms": dict(composed.weighted_terms),
            "finite_check_tensors": finite_tensors,
            "batch_size_tensor": batch_size_tensor,
        }


def require_standard_tensor_mapping(value: Any) -> Mapping[str, Any]:
    torch = require_torch()
    if not isinstance(value, Mapping):
        raise TypeError("DDP reconstructor forward must return a standard mapping")
    required = {
        "total_loss",
        "raw_loss_terms",
        "weighted_loss_terms",
        "finite_check_tensors",
        "batch_size_tensor",
    }
    if set(value) != required:
        raise KeyError("DDP reconstructor forward mapping has the wrong fields")

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
            return
        if not isinstance(item, torch.Tensor):
            raise TypeError("DDP forward output contains a non-tensor leaf")

    visit(value)
    if value["total_loss"].numel() != 1:
        raise ValueError("DDP total_loss must be scalar")
    return value


def tensor_mapping_is_finite(value: Mapping[str, Any]) -> bool:
    torch = require_torch()
    checked = require_standard_tensor_mapping(value)
    tensors = (
        checked["total_loss"],
        *tuple(checked["finite_check_tensors"]),
    )
    return all(bool(torch.isfinite(item).all()) for item in tensors)


def build_stage_ddp_wrapper(
    module: ReconstructorTrainingModule,
    runtime: DistributedRuntime,
    *,
    device: Any,
    find_unused_parameters: bool = True,
) -> Any:
    torch = require_torch()
    if not runtime.distributed:
        return module
    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": bool(find_unused_parameters),
    }
    if str(getattr(device, "type", device)) == "cuda":
        kwargs.update(device_ids=[runtime.local_rank], output_device=runtime.local_rank)
    return torch.nn.parallel.DistributedDataParallel(module, **kwargs)


__all__ = [
    "ABPH_DDP_ERROR_SUMMARY_LIMIT",
    "ABPH_DDP_FORWARD_CONTRACT",
    "ABPH_DISTRIBUTED_RUNTIME_CONTRACT",
    "DistributedRuntime",
    "ReconstructorTrainingModule",
    "all_gather_objects",
    "all_reduce_float64_pair",
    "all_reduce_min_bool",
    "all_reduce_sum_int",
    "any_structural_error",
    "barrier",
    "broadcast_object",
    "build_stage_ddp_wrapper",
    "destroy_distributed_runtime",
    "distributed_environment",
    "gather_error_summaries",
    "initialize_distributed_runtime",
    "parameter_state_hash",
    "require_standard_tensor_mapping",
    "tensor_mapping_is_finite",
    "verify_common_parameter_state",
]
