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

import hashlib
import importlib
import json
import os
import platform
from pathlib import Path
import subprocess
from typing import Any, Mapping
from contextlib import nullcontext

import torch


TORCH_NATIVE_TRITON_COMPAT_CONTRACT = "constrained_c2f_torch_native_triton_fallback_v1"
TORCH_NATIVE_TRITON_MODE_ENV = "CONSTRAINED_C2F_TORCH_NATIVE_TRITON"
TORCH_NATIVE_TRITON_PROBE_ENV = "CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE"

C2F_RUNTIME_PROFILE_CONTRACT = "constrained_c2f_runtime_profile_v1"
C2F_CODE_ENVIRONMENT_CONTRACT = "constrained_c2f_code_environment_v1"

C2F_RUNTIME_PROFILES = frozenset(
    {
        "fp32_reference",
        "fp16_diagnostic",
        "bf16_calibration",
        "bf16_exploratory_pilot_v1",
        "accelerated_candidate_v1",
        "accelerated_approved_v1",
    }
)
C2F_PRECISION_MODES = frozenset(
    {
        "fp32",
        "bf16_forward_fp32_loss",
        "fp16_forward_fp32_loss",
    }
)
_PROFILE_PRECISION_MODES = {
    "fp32_reference": "fp32",
    "fp16_diagnostic": "fp16_forward_fp32_loss",
    "bf16_calibration": "bf16_forward_fp32_loss",
    "bf16_exploratory_pilot_v1": "bf16_forward_fp32_loss",
    "accelerated_candidate_v1": "bf16_forward_fp32_loss",
    "accelerated_approved_v1": "bf16_forward_fp32_loss",
}

_VALID_MODES = {"auto", "disable", "keep"}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_runtime_profile(profile: str) -> str:
    """Return a validated named C2F execution profile."""

    normalized = str(profile).strip().lower()
    if normalized not in C2F_RUNTIME_PROFILES:
        raise ValueError(f"unknown C2F runtime profile {profile!r}; expected one of {sorted(C2F_RUNTIME_PROFILES)}")
    return normalized


def normalize_precision_mode(precision_mode: str) -> str:
    """Return a validated C2F precision declaration."""

    normalized = str(precision_mode).strip().lower()
    if normalized not in C2F_PRECISION_MODES:
        raise ValueError(
            f"unknown C2F precision mode {precision_mode!r}; expected one of {sorted(C2F_PRECISION_MODES)}"
        )
    return normalized


def precision_mode_metadata(precision_mode: str) -> dict[str, Any]:
    """Describe the intended autocast/scaler contract for a precision mode."""

    normalized = normalize_precision_mode(precision_mode)
    if normalized == "fp32":
        return {
            "precision_mode": normalized,
            "autocast_enabled": False,
            "autocast_dtype": None,
            "grad_scaler_enabled": False,
        }
    dtype = "bfloat16" if normalized.startswith("bf16_") else "float16"
    return {
        "precision_mode": normalized,
        "autocast_enabled": True,
        "autocast_dtype": dtype,
        "grad_scaler_enabled": normalized.startswith("fp16_"),
    }


def precision_execution_state(precision_mode: str, device: str | torch.device) -> dict[str, Any]:
    """Resolve the actual autocast/scaler state for a specific runtime device."""

    requested = precision_mode_metadata(precision_mode)
    resolved_device = torch.device(device)
    cuda_available = resolved_device.type == "cuda" and torch.cuda.is_available()
    autocast_enabled = bool(requested["autocast_enabled"] and cuda_available)
    grad_scaler_enabled = bool(requested["grad_scaler_enabled"] and cuda_available)
    return {
        **requested,
        "device": str(resolved_device),
        "autocast_enabled": autocast_enabled,
        "autocast_dtype": requested["autocast_dtype"] if autocast_enabled else None,
        "grad_scaler_enabled": grad_scaler_enabled,
        "model_parameter_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "loss_dtype": "float32",
        "matching_cost_dtype": "float32",
    }


def precision_autocast_context(precision_mode: str, device: str | torch.device):
    """Return the explicitly typed forward autocast context for C2F models."""

    execution = precision_execution_state(precision_mode, device)
    if not execution["autocast_enabled"]:
        return nullcontext()
    dtype = torch.bfloat16 if execution["autocast_dtype"] == "bfloat16" else torch.float16
    try:
        return torch.amp.autocast("cuda", dtype=dtype, enabled=True)
    except AttributeError:  # pragma: no cover - compatibility for older torch releases
        return torch.cuda.amp.autocast(dtype=dtype, enabled=True)


def precision_autocast_disabled_context(device: str | torch.device):
    """Return an explicit disabled-autocast context for the FP32 objective."""

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=False)
    except AttributeError:  # pragma: no cover - compatibility for older torch releases
        return torch.cuda.amp.autocast(enabled=False)


def precision_grad_scaler(precision_mode: str, device: str | torch.device):
    """Create a gradient scaler only for the explicit FP16 diagnostic mode."""

    execution = precision_execution_state(precision_mode, device)
    if not execution["grad_scaler_enabled"]:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # pragma: no cover - compatibility for older torch releases
        return torch.cuda.amp.GradScaler(enabled=True)


def build_runtime_profile(
    *,
    profile: str,
    precision_mode: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    learning_rate: float,
    hlt_encoder_lr_scale: float,
    weight_decay: float,
    grad_clip_norm: float,
    lr_schedule: str,
    warmup_fraction: float,
    min_lr_ratio: float,
    min_epochs: int,
    early_stop_patience: int,
    fixed_horizon: bool,
    max_epochs: int,
    hungarian_workers: int,
    hungarian_executor: str,
) -> dict[str, Any]:
    """Build the canonical, hash-bound execution profile persisted by C2F jobs.

    Step 1 only declares the profile contract. Later acceleration steps consume
    the stored fields to implement the matching scheduler, precision, and
    Hungarian execution behavior without changing their identity in metadata.
    """

    normalized_profile = normalize_runtime_profile(profile)
    normalized_precision = normalize_precision_mode(precision_mode)
    expected_precision = _PROFILE_PRECISION_MODES[normalized_profile]
    if normalized_precision != expected_precision:
        raise ValueError(
            f"runtime profile {normalized_profile!r} requires precision mode "
            f"{expected_precision!r}, got {normalized_precision!r}"
        )
    normalized_schedule = str(lr_schedule).strip().lower()
    if normalized_schedule not in {"constant", "warmup_cosine"}:
        raise ValueError("lr_schedule must be 'constant' or 'warmup_cosine'")
    normalized_executor = str(hungarian_executor).strip().lower()
    if normalized_executor not in {"serial", "thread"}:
        raise ValueError("hungarian_executor must be 'serial' or 'thread'")
    if prefetch_factor is not None and int(prefetch_factor) <= 0:
        raise ValueError("prefetch_factor must be positive when supplied")
    if int(num_workers) == 0 and prefetch_factor is not None:
        raise ValueError("prefetch_factor requires num_workers > 0")
    if not 0.0 < float(warmup_fraction) <= 1.0:
        raise ValueError("warmup_fraction must be in (0, 1]")
    if not 0.0 < float(min_lr_ratio) <= 1.0:
        raise ValueError("min_lr_ratio must be in (0, 1]")
    if int(min_epochs) < 0:
        raise ValueError("min_epochs must be nonnegative")
    if int(max_epochs) <= 0:
        raise ValueError("max_epochs must be positive")
    if int(min_epochs) > int(max_epochs):
        raise ValueError("min_epochs cannot exceed max_epochs")
    if int(hungarian_workers) <= 0:
        raise ValueError("hungarian_workers must be positive")

    payload: dict[str, Any] = {
        "contract": C2F_RUNTIME_PROFILE_CONTRACT,
        "name": normalized_profile,
        "precision": precision_mode_metadata(normalized_precision),
        "batch": {
            "train": int(batch_size),
            "eval": int(eval_batch_size),
        },
        "input_pipeline": {
            "num_workers": int(num_workers),
            "prefetch_factor": None if prefetch_factor is None else int(prefetch_factor),
        },
        "optimizer": {
            "learning_rate": float(learning_rate),
            "hlt_encoder_lr_scale": float(hlt_encoder_lr_scale),
            "weight_decay": float(weight_decay),
            "grad_clip_norm": float(grad_clip_norm),
        },
        "scheduler": {
            "name": normalized_schedule,
            "warmup_fraction": float(warmup_fraction),
            "min_lr_ratio": float(min_lr_ratio),
            "min_epochs": int(min_epochs),
            "early_stop_patience": int(early_stop_patience),
            "fixed_horizon": bool(fixed_horizon),
            "max_epochs": int(max_epochs),
        },
        "hungarian": {
            "workers": int(hungarian_workers),
            "executor": normalized_executor,
        },
    }
    payload["runtime_profile_hash"] = _canonical_sha256(payload)
    return payload


def collect_code_environment(project_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Collect the reproducibility fingerprint required by candidate profiles."""

    root = Path(project_dir) if project_dir is not None else Path(__file__).resolve().parents[2]
    commit: str | None = None
    status = ""
    git_error: str | None = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        git_error = str(error)
    try:
        import scipy

        scipy_version: str | None = scipy.__version__
    except ImportError:
        scipy_version = None
    payload: dict[str, Any] = {
        "contract": C2F_CODE_ENVIRONMENT_CONTRACT,
        "source_commit": commit,
        "source_tree_clean": bool(commit is not None and not status),
        "source_status_hash": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "scipy_version": scipy_version,
    }
    if git_error is not None:
        payload["git_error"] = git_error
    payload["code_environment_hash"] = _canonical_sha256(payload)
    return payload


def profile_requires_clean_source(profile: str) -> bool:
    """Return whether the named profile may be created only from a clean tree."""

    return normalize_runtime_profile(profile) in {"accelerated_candidate_v1", "accelerated_approved_v1"}


def profile_requires_last_checkpoint(profile: str) -> bool:
    """Return whether a profile must persist an epoch-boundary resume point."""

    return normalize_runtime_profile(profile) in {"accelerated_candidate_v1", "accelerated_approved_v1"}


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
    "C2F_CODE_ENVIRONMENT_CONTRACT",
    "C2F_PRECISION_MODES",
    "C2F_RUNTIME_PROFILE_CONTRACT",
    "C2F_RUNTIME_PROFILES",
    "TORCH_NATIVE_TRITON_COMPAT_CONTRACT",
    "TORCH_NATIVE_TRITON_MODE_ENV",
    "TORCH_NATIVE_TRITON_PROBE_ENV",
    "build_runtime_profile",
    "collect_code_environment",
    "configure_torch_native_triton_fallback",
    "normalize_precision_mode",
    "normalize_runtime_profile",
    "precision_autocast_context",
    "precision_autocast_disabled_context",
    "precision_execution_state",
    "precision_grad_scaler",
    "precision_mode_metadata",
    "profile_requires_clean_source",
    "profile_requires_last_checkpoint",
]
