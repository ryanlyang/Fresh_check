"""Exact parameter inventory and reproducible measured resource profiling."""

from __future__ import annotations

import statistics
import time
from typing import Any, Mapping

from .contracts import require_sha256, with_content_hash
from .evaluation import model_forward

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


RESOURCE_PROFILE_CONTRACT = "relational_part_resource_profile_v1"


def parameter_profile(model: Any) -> dict[str, Any]:
    records = []
    total = trainable = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
        records.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "count": count,
                "trainable": bool(parameter.requires_grad),
            }
        )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "nontrainable_parameters": total - trainable,
        "active_parameter_records": records,
    }


class _FlopCounter:
    def __init__(self) -> None:
        self.by_kind: dict[str, int] = {}
        self.handles: list[Any] = []

    def add(self, kind: str, count: int) -> None:
        self.by_kind[kind] = self.by_kind.get(kind, 0) + int(count)

    def install(self, model: Any) -> None:
        module = torch.nn

        def linear(layer, inputs, output):
            del layer
            vectors = int(inputs[0].numel() // inputs[0].shape[-1])
            self.add(
                "linear",
                2 * vectors * int(inputs[0].shape[-1]) * int(output.shape[-1]),
            )

        def conv1d(layer, inputs, output):
            kernel = int(layer.kernel_size[0])
            per_output = kernel * int(layer.in_channels) // int(layer.groups)
            self.add("conv1d", 2 * int(output.numel()) * per_output)

        def mha(layer, inputs, output):
            del output
            query = inputs[0]
            if bool(layer.batch_first):
                batch, query_count, embed = map(int, query.shape)
                key_count = int(inputs[1].shape[1])
            else:
                query_count, batch, embed = map(int, query.shape)
                key_count = int(inputs[1].shape[0])
            heads = int(layer.num_heads)
            head_dim = embed // heads
            projection = 2 * batch * (
                query_count * embed * embed
                + 2 * key_count * embed * embed
                + query_count * embed * embed
            )
            attention = 4 * batch * heads * query_count * key_count * head_dim
            self.add("multihead_attention", projection + attention)

        for child in model.modules():
            if isinstance(child, module.Linear):
                self.handles.append(child.register_forward_hook(linear))
            elif isinstance(child, module.Conv1d):
                self.handles.append(child.register_forward_hook(conv1d))
            elif isinstance(child, module.MultiheadAttention):
                self.handles.append(child.register_forward_hook(mha))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def profile_model_resources(
    model: Any,
    example_batch: Mapping[str, Any],
    *,
    device: str | Any = "cpu",
    warmup_repetitions: int = 2,
    measured_repetitions: int = 5,
    model_contract_sha256: str,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for resource profiling")
    if warmup_repetitions < 0 or measured_repetitions <= 0:
        raise ValueError("invalid profiling repetition counts")
    resolved = torch.device(device)
    moved = {
        name: value.to(resolved) if isinstance(value, torch.Tensor) else value
        for name, value in example_batch.items()
        if name != "labels"
    }
    was_training = bool(model.training)
    model.to(resolved).eval()
    counter = _FlopCounter()
    counter.install(model)
    with torch.no_grad():
        model_forward(model, moved)
    counter.remove()

    def synchronize() -> None:
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)

    with torch.no_grad():
        for _ in range(warmup_repetitions):
            model_forward(model, moved)
        synchronize()
        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)
            baseline_memory = int(torch.cuda.memory_allocated(resolved))
        else:
            baseline_memory = None
        durations = []
        for _ in range(measured_repetitions):
            start = time.perf_counter()
            model_forward(model, moved)
            synchronize()
            durations.append((time.perf_counter() - start) * 1000.0)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(resolved)) - baseline_memory
            if resolved.type == "cuda"
            else None
        )
    if was_training:
        model.train()
    batch_size = int(next(iter(moved.values())).shape[0])
    flops = int(sum(counter.by_kind.values()))
    return with_content_hash(
        {
            "contract": RESOURCE_PROFILE_CONTRACT,
            "schema_version": 1,
            "model_contract_sha256": require_sha256(
                model_contract_sha256, name="model_contract_sha256"
            ),
            "device": str(resolved),
            "batch_size": batch_size,
            **parameter_profile(model),
            "forward_flops": flops,
            "forward_flops_per_event": flops / batch_size,
            "flop_convention": {
                "multiply_add": 2,
                "included": [
                    "Linear",
                    "Conv1d",
                    "MultiheadAttention_qkv_output_and_attention_matmuls",
                ],
                "excluded": [
                    "normalization",
                    "activation",
                    "softmax",
                    "raw_relation_arithmetic",
                ],
            },
            "latency_ms": {
                "warmup_repetitions": int(warmup_repetitions),
                "measured_repetitions": int(measured_repetitions),
                "samples": durations,
                "median": float(statistics.median(durations)),
                "mean": float(statistics.fmean(durations)),
            },
            "peak_incremental_device_memory_bytes": peak_memory,
        }
    )


def build_resource_profile_contract(*, global_determinism_sha256: str) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": RESOURCE_PROFILE_CONTRACT,
            "schema_version": 1,
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "parameter_count": "all_active_named_parameters_exact_numel",
            "trainable_parameter_count": "requires_grad_true_exact_numel",
            "flop_convention": "multiply_and_add_each_count_as_one_operation",
            "latency_clock": "perf_counter_with_device_synchronization",
            "gpu_peak_memory": "max_allocated_minus_premeasurement_allocated",
            "cpu_peak_memory": None,
        }
    )


__all__ = [
    "RESOURCE_PROFILE_CONTRACT",
    "build_resource_profile_contract",
    "parameter_profile",
    "profile_model_resources",
]
