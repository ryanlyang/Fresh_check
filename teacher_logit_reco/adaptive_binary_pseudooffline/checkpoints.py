"""Compact selected-checkpoint and ephemeral resume contracts for ABPH."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from .storage_quota import (
    ABPH_CACHE_HEAVY_STORAGE_PROFILE,
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    write_quota_managed_bytes,
)


ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT = (
    "adaptive_binary_pseudooffline_compact_selected_checkpoint_v1"
)
ABPH_EPHEMERAL_RESUME_CHECKPOINT_CONTRACT = (
    "adaptive_binary_pseudooffline_ephemeral_resume_checkpoint_v1"
)
ABPH_RESTART_SEMANTICS_WARM_START = "selected_weights_warm_start_not_exact_resume"
ABPH_RESTART_SEMANTICS_EXACT = "optimizer_exact_within_live_allocation_only"

_SELECTED_ROLES = frozenset({"best_stage_model_val", "best_model_val"})
_FORBIDDEN_COMPACT_KEYS = frozenset(
    {
        "online_model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "ema_state_dict",
        "rng_state",
        "train_source_state_dict",
        "distributed_checkpoint_state",
    }
)


def active_storage_profile() -> str:
    return os.environ.get("ABPH_STORAGE_PROFILE", ABPH_CACHE_HEAVY_STORAGE_PROFILE)


def streaming_storage_enabled() -> bool:
    return active_storage_profile() == ABPH_STREAMING_STORAGE_PROFILE


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _tensor_hash(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("model state values must be tensors")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    if tensor.numel():
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_state_content_hash(state: Mapping[str, Any]) -> str:
    rows = [
        {"name": str(name), "tensor_sha256": _tensor_hash(value)}
        for name, value in sorted(state.items())
    ]
    return _canonical_hash({"tensors": rows})


def _compact_content_hash(payload: Mapping[str, Any]) -> str:
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"content_hash", "model_state_dict", "component_state_dicts"}
    }
    components = payload.get("component_state_dicts", {})
    return _canonical_hash(
        {
            "metadata": metadata,
            "model_state_content_hash": model_state_content_hash(
                payload["model_state_dict"]
            ),
            "component_state_content_hashes": {
                str(name): model_state_content_hash(state)
                for name, state in sorted(dict(components).items())
            },
        }
    )


def build_compact_selected_checkpoint(
    *,
    model_state_dict: Mapping[str, Any],
    checkpoint_role: str,
    model_metadata: Mapping[str, Any],
    resolved_variant_config: Mapping[str, Any],
    resolved_variant_config_hash: str,
    validation: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    runtime_contracts: Mapping[str, Any] | None = None,
    schedule_contracts: Mapping[str, Any] | None = None,
    component_state_dicts: Mapping[str, Mapping[str, Any]] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one self-verifying selected artifact with one model state."""

    if checkpoint_role not in _SELECTED_ROLES:
        raise ValueError("compact checkpoints must be model-val selected")
    if not model_state_dict:
        raise ValueError("compact selected checkpoint requires a nonempty model state")
    if not resolved_variant_config_hash:
        raise ValueError("compact selected checkpoint requires a configuration hash")
    extra = dict(extra_metadata or {})
    overlap = _FORBIDDEN_COMPACT_KEYS.intersection(extra)
    if overlap or "model_state_dict" in extra or "content_hash" in extra:
        raise ValueError(
            "compact selected metadata contains forbidden training state: "
            + ", ".join(sorted(overlap | ({"model_state_dict"} if "model_state_dict" in extra else set())))
        )
    payload: dict[str, Any] = {
        "checkpoint_contract": ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT,
        "checkpoint_role": checkpoint_role,
        "selection_split": "model_val",
        "selection_mode": "rollout",
        "model_state_dict": {
            str(name): value.detach().cpu().clone()
            for name, value in model_state_dict.items()
        },
        "component_state_dicts": {
            str(component): {
                str(name): value.detach().cpu().clone()
                for name, value in state.items()
            }
            for component, state in dict(component_state_dicts or {}).items()
        },
        "model_metadata": _jsonable(model_metadata),
        "resolved_variant_config": _jsonable(resolved_variant_config),
        "resolved_variant_config_hash": str(resolved_variant_config_hash),
        "validation": _jsonable(validation),
        "provenance": _jsonable(provenance),
        "runtime_contracts": _jsonable(runtime_contracts or {}),
        "schedule_contracts": _jsonable(schedule_contracts or {}),
        "storage_profile": ABPH_STREAMING_STORAGE_PROFILE,
        "restart_semantics": ABPH_RESTART_SEMANTICS_WARM_START,
        "exact_resume_supported": False,
        "final_test_loaded": False,
        **_jsonable(extra),
    }
    payload["model_state_content_hash"] = model_state_content_hash(
        payload["model_state_dict"]
    )
    payload["component_state_content_hashes"] = {
        name: model_state_content_hash(state)
        for name, state in payload["component_state_dicts"].items()
    }
    payload["content_hash"] = _compact_content_hash(payload)
    validate_compact_selected_checkpoint(payload)
    return payload


def validate_compact_selected_checkpoint(
    payload: Mapping[str, Any], *, require_selected: bool = True
) -> None:
    if payload.get("checkpoint_contract") != ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT:
        raise ValueError("compact selected checkpoint contract mismatch")
    if require_selected and payload.get("checkpoint_role") not in _SELECTED_ROLES:
        raise ValueError("compact checkpoint is not model-val selected")
    forbidden = sorted(_FORBIDDEN_COMPACT_KEYS.intersection(payload))
    if forbidden:
        raise ValueError(
            "compact selected checkpoint contains forbidden fields: "
            + ", ".join(forbidden)
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("compact selected checkpoint has no model state")
    state_hash = model_state_content_hash(state)
    if payload.get("model_state_content_hash") != state_hash:
        raise ValueError("compact selected checkpoint model-state hash mismatch")
    components = payload.get("component_state_dicts", {})
    if not isinstance(components, Mapping):
        raise ValueError("compact selected checkpoint component states are invalid")
    component_hashes = {
        str(name): model_state_content_hash(component_state)
        for name, component_state in components.items()
    }
    if payload.get("component_state_content_hashes") != component_hashes:
        raise ValueError("compact selected checkpoint component-state hash mismatch")
    if payload.get("content_hash") != _compact_content_hash(payload):
        raise ValueError("compact selected checkpoint content hash mismatch")
    if payload.get("exact_resume_supported") is not False:
        raise ValueError("compact selected checkpoint must forbid exact resume")
    if payload.get("restart_semantics") != ABPH_RESTART_SEMANTICS_WARM_START:
        raise ValueError("compact selected checkpoint restart semantics mismatch")
    if payload.get("final_test_loaded") is not False:
        raise ValueError("compact selected checkpoint violates final-test isolation")


def load_torch_checkpoint(path: str | Path, *, device: Any = "cpu") -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - old research PyTorch
        payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint {path} is not a mapping")
    if payload.get("checkpoint_contract") == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT:
        validate_compact_selected_checkpoint(payload)
    return payload


def selected_model_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("checkpoint_contract") == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT:
        validate_compact_selected_checkpoint(payload)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("selected checkpoint has no model_state_dict")
    return state


def selected_checkpoint_provenance(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = Path(path)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {
        "path": str(resolved.resolve()),
        "file_sha256": digest,
        "checkpoint_contract": payload.get("checkpoint_contract"),
        "checkpoint_role": payload.get("checkpoint_role"),
        "content_hash": payload.get("content_hash"),
        "model_state_content_hash": (
            payload.get("model_state_content_hash")
            or model_state_content_hash(selected_model_state(payload))
        ),
        "restart_semantics": payload.get(
            "restart_semantics", "legacy_selected_weights_warm_start"
        ),
        "exact_resume": False,
    }


def _serialize(payload: Mapping[str, Any]) -> bytes:
    import torch

    stream = BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_selected_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    campaign_root: str | Path | None = None,
    artifact_role: str = "selected_checkpoint",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Write selected state atomically and quota-account it in streaming mode."""

    destination = Path(path)
    if payload.get("checkpoint_contract") == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT:
        validate_compact_selected_checkpoint(payload)
    data = _serialize(payload)
    if streaming_storage_enabled():
        if campaign_root is None:
            raise ValueError("streaming selected checkpoint writes require campaign_root")
        provenance_hash = str(
            payload.get("content_hash")
            or _canonical_hash(dict(payload.get("provenance", {})))
        )
        return write_quota_managed_bytes(
            campaign_root,
            destination,
            data,
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            artifact_role=artifact_role,
            source_provenance_hash=provenance_hash,
            run_id=run_id or os.environ.get("SLURM_JOB_ID", "local"),
            profile=ABPH_STREAMING_STORAGE_PROFILE,
        )
    _atomic_bytes(destination, data)
    return {
        "status": "committed",
        "destination": str(destination.resolve()),
        "actual_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "artifact_role": artifact_role,
    }


def ephemeral_checkpoint_path(*, variant_name: str, filename: str = "last.pt") -> Path:
    root = os.environ.get("ABPH_RAM_WORKSPACE")
    if not root:
        raise RuntimeError(
            "streaming checkpoint state requires a verified ABPH_RAM_WORKSPACE"
        )
    safe = variant_name.replace("/", "_").replace("\\", "_")
    path = Path(root).resolve() / "checkpoints" / safe / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def require_exact_resume_checkpoint(payload: Mapping[str, Any]) -> None:
    if payload.get("checkpoint_contract") == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT:
        raise ValueError(
            "compact selected checkpoints are warm starts, not exact optimizer resumes"
        )
    if payload.get("exact_resume_supported") is False:
        raise ValueError("checkpoint explicitly forbids exact optimizer resume")


__all__ = [
    "ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT",
    "ABPH_EPHEMERAL_RESUME_CHECKPOINT_CONTRACT",
    "ABPH_RESTART_SEMANTICS_EXACT",
    "ABPH_RESTART_SEMANTICS_WARM_START",
    "active_storage_profile",
    "build_compact_selected_checkpoint",
    "ephemeral_checkpoint_path",
    "load_torch_checkpoint",
    "model_state_content_hash",
    "require_exact_resume_checkpoint",
    "selected_checkpoint_provenance",
    "selected_model_state",
    "streaming_storage_enabled",
    "validate_compact_selected_checkpoint",
    "write_selected_checkpoint",
]
