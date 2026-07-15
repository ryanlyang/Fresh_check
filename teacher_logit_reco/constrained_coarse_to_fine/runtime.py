"""Runtime compatibility helpers for constrained coarse-to-fine jobs.

Tigris' ARM CUDA environment ships a PyTorch build whose optional
``torch._native`` Triton overrides try to JIT-compile a small CUDA utility
through the site compiler wrapper. That wrapper cannot build Triton's helper
module, so an otherwise ordinary backward pass fails before the first update.
The standard PyTorch CUDA implementations remain available. This module
selectively removes only those optional Triton overrides and probes the
fallback before a long training job loads its data.
"""

from __future__ import annotations

import importlib
import os
import platform
from typing import Any

import torch


TORCH_NATIVE_TRITON_COMPAT_CONTRACT = "constrained_c2f_torch_native_triton_fallback_v1"
TORCH_NATIVE_TRITON_MODE_ENV = "CONSTRAINED_C2F_TORCH_NATIVE_TRITON"
TORCH_NATIVE_TRITON_PROBE_ENV = "CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE"

_VALID_MODES = {"auto", "disable", "keep"}


def _env_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean environment value, got {value!r}")


def _resolved_mode(mode: str | None) -> str:
    resolved = (mode or os.environ.get(TORCH_NATIVE_TRITON_MODE_ENV, "auto")).strip().lower()
    if resolved not in _VALID_MODES:
        raise ValueError(
            f"{TORCH_NATIVE_TRITON_MODE_ENV} must be one of {sorted(_VALID_MODES)}, got {resolved!r}"
        )
    return resolved


def _should_disable(mode: str) -> bool:
    if mode == "disable":
        return True
    if mode == "keep":
        return False
    # The observed compiler failure is specific to the Tigris ARM workers.
    return platform.machine().strip().lower() in {"aarch64", "arm64"}


def _active_triton_override_count(registry: Any) -> int | None:
    """Return active Triton override count when the registry exposes it."""

    graphs = getattr(registry, "_graphs", None)
    if not isinstance(graphs, dict):
        return None
    return sum(
        1
        for nodes in graphs.values()
        for node in nodes
        if getattr(node, "dsl_name", None) == "triton" and bool(getattr(node, "active", False))
    )


def _probe_bmm_backward(device: torch.device) -> None:
    """Exercise the backward path that previously selected Triton's override."""

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    try:
        left = torch.randn((2, 3, 4), device=device, requires_grad=True)
        right = torch.randn((2, 4, 5), device=device, requires_grad=True)
        torch.bmm(left, right).square().mean().backward()
        torch.cuda.synchronize(device)
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)


def configure_torch_native_triton_fallback(
    device: str | torch.device,
    *,
    mode: str | None = None,
    probe: bool | None = None,
) -> dict[str, Any]:
    """Disable broken optional Triton-native overrides when the policy selects it.

    ``auto`` is deliberately narrow: it applies only on ARM CUDA workers,
    where the observed Tigris compiler failure occurs. ``disable`` forces the
    portable PyTorch fallback on any CUDA host; ``keep`` preserves the native
    overrides. A CUDA ``bmm`` backward probe makes a bad runtime fail before
    expensive cache loading or training work begins.
    """

    resolved_device = torch.device(device)
    resolved_mode = _resolved_mode(mode)
    probe_enabled = _env_bool(os.environ.get(TORCH_NATIVE_TRITON_PROBE_ENV), default=True) if probe is None else bool(probe)
    report: dict[str, Any] = {
        "contract": TORCH_NATIVE_TRITON_COMPAT_CONTRACT,
        "requested_mode": resolved_mode,
        "device": str(resolved_device),
        "platform_machine": platform.machine(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fallback_selected": False,
        "probe_requested": probe_enabled,
        "probe_passed": False,
    }
    if resolved_device.type != "cuda":
        report["reason"] = "non_cuda_device"
        return report
    if not _should_disable(resolved_mode):
        report["reason"] = "policy_keeps_native_overrides"
        return report

    try:
        registry = importlib.import_module("torch._native.registry")
    except ModuleNotFoundError:
        # Older PyTorch versions have no optional native override layer and
        # already use their standard CUDA kernels.
        report["reason"] = "torch_native_registry_unavailable"
        return report

    operations = list(registry.get_dsl_operations("triton"))
    report["registered_triton_operations"] = operations
    if not operations:
        report["reason"] = "no_triton_native_overrides_registered"
        return report

    registry.deregister_op_overrides(disable_dsl_names="triton")
    active_count = _active_triton_override_count(registry)
    report["active_triton_overrides_after_disable"] = active_count
    if active_count is not None and active_count:
        raise RuntimeError(
            "failed to disable all torch._native Triton overrides; refusing to start a C2F CUDA job"
        )
    report["fallback_selected"] = True
    report["reason"] = "triton_native_overrides_disabled"
    if probe_enabled:
        _probe_bmm_backward(resolved_device)
        report["probe_passed"] = True
    return report


__all__ = [
    "TORCH_NATIVE_TRITON_COMPAT_CONTRACT",
    "TORCH_NATIVE_TRITON_MODE_ENV",
    "TORCH_NATIVE_TRITON_PROBE_ENV",
    "configure_torch_native_triton_fallback",
]
