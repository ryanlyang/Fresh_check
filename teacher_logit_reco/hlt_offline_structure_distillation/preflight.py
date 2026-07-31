"""Research-compute runtime and storage preflight for HOSD."""

from __future__ import annotations

import importlib
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping

from .contracts import (
    RESOURCE_PREFLIGHT_CONTRACT,
    RESOURCE_MEASUREMENTS_CONTRACT,
    STORAGE_MEASUREMENT_CONTRACT,
    validate_content_hash,
    with_content_hash,
)


REQUIRED_SCALE_MEMORY_PROJECTION_NODES = frozenset(
    {
        "scale_input_prepare",
        "scale_tree_build",
        "scale_target_build",
        "scale_teacher_target_inference",
        "scale_graph_train",
    }
)


def run_resource_preflight(
    *,
    campaign_root: str | Path,
    storage_measurements: Mapping[str, Any],
    profile: str,
    source: Mapping[str, Any],
    require_cuda: bool,
    resource_measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_content_hash(
        storage_measurements, expected_contract=STORAGE_MEASUREMENT_CONTRACT
    )
    if storage_measurements.get("source") != dict(source):
        raise ValueError("resource preflight storage source differs")
    if profile == "production_500k_scale3m":
        if resource_measurements is None:
            raise ValueError(
                "production preflight requires miniature resource measurements"
            )
        validate_content_hash(
            resource_measurements,
            expected_contract=RESOURCE_MEASUREMENTS_CONTRACT,
        )
        if resource_measurements.get("source") != dict(source):
            raise ValueError("resource measurement source differs")
        projections = resource_measurements.get(
            "scale_resident_memory_projections"
        )
        if (
            not isinstance(projections, Mapping)
            or set(projections) != set(REQUIRED_SCALE_MEMORY_PROJECTION_NODES)
        ):
            raise ValueError("production preflight lacks scale-node pilots")
        for node_id, projection in projections.items():
            expected_unit_events = {
                "scale_input_prepare": 1,
                "scale_tree_build": 10_000,
                "scale_target_build": 2_048,
                "scale_graph_train": 10_000,
            }[node_id]
            if (
                not isinstance(projection, Mapping)
                or projection.get("pilot_node_id") != node_id
                or projection.get("representative_real_task_completed")
                is not True
                or projection.get("within_registered_tigris_limit") is not True
                or int(projection.get("projected_resident_bytes", 0)) <= 0
                or int(projection.get("projected_resident_bytes", 0))
                > int(projection.get("registered_tigris_limit_bytes", 0))
                or int(
                    projection.get("production_resident_unit_events", 0)
                )
                != expected_unit_events
            ):
                raise ValueError(
                    f"production preflight scale pilot differs: {node_id}"
                )
    elif resource_measurements is not None:
        raise ValueError("miniature preflight cannot use production measurements")
    root = Path(campaign_root).resolve()
    if not bool(sys.flags.no_user_site):
        raise RuntimeError(
            "resource preflight requires PYTHONNOUSERSITE=1 / python -s"
        )
    usage = shutil.disk_usage(root)
    projected = int(
        storage_measurements.get("projected_storage_bytes", {}).get(
            "scale_3m"
            if profile == "production_500k_scale3m"
            else "production_500k",
            0,
        )
    )
    if projected <= 0:
        raise ValueError("storage measurements lack a positive campaign projection")
    minimum_free = int(math_ceil_ratio(projected * 6, 5))
    if usage.free < minimum_free:
        raise RuntimeError(
            f"insufficient storage: free={usage.free}, required={minimum_free}"
        )
    imports = {}
    for name in ("numpy", "torch", "uproot", "awkward", "ninja", "weaver"):
        module = importlib.import_module(name)
        imports[name] = str(getattr(module, "__version__", "present"))
    torch = importlib.import_module("torch")
    cuda = bool(torch.cuda.is_available())
    if require_cuda and not cuda:
        raise RuntimeError("GPU preflight requires an allocated CUDA device")
    gpu = None
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "gh200_expected_match": "GH200" in properties.name.upper()
            or "HOPPER" in properties.name.upper(),
        }
        if require_cuda and profile == "production_500k_scale3m" and not gpu[
            "gh200_expected_match"
        ]:
            raise RuntimeError("production GPU preflight did not allocate a GH200")
    return with_content_hash(
        {
            "contract": RESOURCE_PREFLIGHT_CONTRACT,
            "schema_version": 10,
            "source": dict(source),
            "profile": profile,
            "storage_measurements_sha256": storage_measurements[
                "content_hash"
            ],
            "resource_measurements_sha256": (
                None
                if resource_measurements is None
                else resource_measurements["content_hash"]
            ),
            "resource_projections": (
                None
                if resource_measurements is None
                else {
                    key: resource_measurements[key]
                    for key in (
                        "projected_target_extraction_seconds",
                        "projected_gpu_hours_by_stage",
                        "maximum_concurrent_jobs",
                        "checkpoint_bytes",
                        "export_bytes",
                        "requests_by_class",
                        "scale_resident_layout_ledger",
                        "scale_resident_memory_projections",
                    )
                }
            ),
            "campaign_root": str(root),
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "minimum_3_10": sys.version_info >= (3, 10),
            },
            "imports": imports,
            "cuda_available": cuda,
            "gpu": gpu,
            "storage": {
                "free_bytes": usage.free,
                "projected_bytes": projected,
                "safety_factor_numerator": 6,
                "safety_factor_denominator": 5,
                "minimum_free_bytes": minimum_free,
                "passed": True,
            },
            "user_site_disabled": bool(sys.flags.no_user_site),
            "runtime_ready": True,
        }
    )


def math_ceil_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("preflight ratio denominator must be positive")
    return (int(numerator) + int(denominator) - 1) // int(denominator)


__all__ = ["math_ceil_ratio", "run_resource_preflight"]
