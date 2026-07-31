"""Measured and analytical HOSD efficiency evidence."""

from __future__ import annotations

import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Mapping

from .contracts import (
    EFFICIENCY_PROFILE_CONTRACT,
    require_sha256,
    with_content_hash,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


WARMUP_REPETITIONS = 200
MEASURED_REPETITIONS = 1_000


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    coordinate = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(coordinate))
    upper = int(math.ceil(coordinate))
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _forward(model: Any, batch: Mapping[str, Any]) -> Any:
    vectors = batch.get("lorentz_vectors", batch.get("vectors"))
    points = batch.get("points")
    if points is None:
        points = batch["features"][:, 15:17]
    return model(points, batch["features"], vectors, batch["mask"])


def measure_deployable_efficiency(
    *,
    model: Any,
    representative_batches: Mapping[int, Mapping[str, Any]],
    production_batch_size: int,
    graph_id: str,
    seed: int,
    export_path: str | Path,
    export_sha256: str,
    analytical_training_flops: int,
    analytical_inference_flops_by_batch: Mapping[int, int],
    complete_parameter_count: int,
    target_cache_bytes_per_jet: float,
    training_gpu_hours: float,
    clock_power_mode: str,
    training_evidence: Mapping[str, Any],
    analytical_training_flop_convention: str,
    source: Mapping[str, Any],
    device: str | Any = "cuda",
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for efficiency measurement")
    expected_batches = {1, 128, int(production_batch_size)}
    if set(representative_batches) != expected_batches:
        raise ValueError("efficiency batches must cover 1, 128, and production")
    if set(int(key) for key in analytical_inference_flops_by_batch) != expected_batches:
        raise ValueError("analytical inference FLOP batch coverage differs")
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("authoritative HOSD latency requires a CUDA GPU")
    model.to(resolved).eval()
    deployed_parameters = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if (
        deployed_parameters <= 0
        or int(complete_parameter_count) < deployed_parameters
        or int(analytical_training_flops) <= 0
        or any(int(value) <= 0 for value in analytical_inference_flops_by_batch.values())
    ):
        raise ValueError("efficiency capacity values differ")
    if not training_evidence or not analytical_training_flop_convention:
        raise ValueError("efficiency training evidence differs")
    torch.cuda.reset_peak_memory_stats(resolved)
    measurements = {}
    with torch.no_grad():
        for batch_size in sorted(expected_batches):
            batch = _move_batch(representative_batches[batch_size], resolved)
            observed = int(batch["features"].shape[0])
            if observed != batch_size:
                raise ValueError("efficiency representative batch size differs")
            for _ in range(WARMUP_REPETITIONS):
                output = _forward(model, batch)
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError("efficiency warmup logits are nonfinite")
            torch.cuda.synchronize(resolved)
            elapsed = []
            for _ in range(MEASURED_REPETITIONS):
                started = time.perf_counter_ns()
                output = _forward(model, batch)
                torch.cuda.synchronize(resolved)
                elapsed.append((time.perf_counter_ns() - started) / 1e6)
            measurements[str(batch_size)] = {
                "batch_size": batch_size,
                "warmup_repetitions": WARMUP_REPETITIONS,
                "measured_repetitions": MEASURED_REPETITIONS,
                "median_milliseconds": statistics.median(elapsed),
                "p90_milliseconds": _percentile(elapsed, 0.90),
                "p95_milliseconds": _percentile(elapsed, 0.95),
                "examples_per_second_from_median": (
                    batch_size / (statistics.median(elapsed) / 1000.0)
                ),
            }
    properties = torch.cuda.get_device_properties(resolved)
    device_uuid = getattr(properties, "uuid", None)
    return with_content_hash(
        {
            "contract": EFFICIENCY_PROFILE_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "graph_id": str(graph_id),
            "seed": int(seed),
            "export_sha256": require_sha256(
                export_sha256, name="export_sha256"
            ),
            "export_size_bytes": Path(export_path).stat().st_size,
            "complete_trainable_parameters": int(complete_parameter_count),
            "deployed_trainable_parameters": deployed_parameters,
            "target_head_removal_parameter_savings": (
                int(complete_parameter_count) - deployed_parameters
            ),
            "analytical_training_flops": int(analytical_training_flops),
            "analytical_inference_flops_by_batch": {
                str(key): int(value)
                for key, value in sorted(analytical_inference_flops_by_batch.items())
            },
            "target_cache_bytes_per_jet": float(target_cache_bytes_per_jet),
            "training_gpu_hours": float(training_gpu_hours),
            "training_evidence": dict(training_evidence),
            "analytical_training_flop_convention": str(
                analytical_training_flop_convention
            ),
            "input_multiplicity": int(
                representative_batches[1]["features"].shape[-1]
            ),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(resolved)),
            "latency": measurements,
            "single_jet_latency_key": "1",
            "production_batch_size": int(production_batch_size),
            "hardware_software": {
                "gpu_name": properties.name,
                "gpu_uuid": None if device_uuid is None else str(device_uuid),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "driver_version": torch._C._cuda_getDriverVersion()
                if hasattr(torch._C, "_cuda_getDriverVersion")
                else None,
                "cuda_runtime": torch.version.cuda,
                "pytorch": torch.__version__,
                "python_platform": platform.platform(),
                "export_backend": "pytorch_state_dict_eager",
                "precision": "FP32",
                "clock_power_mode": str(clock_power_mode),
            },
            "unlike_hardware_comparison_allowed": False,
            "latency_selection_tiebreaker": False,
        }
    )


__all__ = [
    "MEASURED_REPETITIONS",
    "WARMUP_REPETITIONS",
    "measure_deployable_efficiency",
]
