"""Deterministic slot-query coordinates for frozen offline targets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bridge_targets import BridgeProjection


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "PyTorch is required to resolve target slot queries"
        ) from error
    return torch


def target_slot_queries(
    checkpoint_path: str | Path,
    *,
    target_mode: str,
) -> np.ndarray:
    """Return the exact query coordinate consumed by a token predictor.

    T0/T1 predictors use the frozen tokenizer queries directly. T2 target
    tokens live after ``BridgeProjection`` and use the immutable projection
    of those queries.
    """

    module = _require_torch()
    payload = module.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise ValueError("target checkpoint payload differs")
    if target_mode == "T0_PURE":
        state = payload.get("model_state_dict", payload)
    elif target_mode in {
        "T1_ANCHORED_BRIDGE",
        "T1_TASK_BRIDGE",
        "T2_PROJECT",
    }:
        state = payload.get("offline_target_state_dict")
    else:
        raise ValueError("target slot-query mode differs")
    if not isinstance(state, Mapping):
        raise ValueError("target checkpoint state differs")
    suffixes = (
        "expert_model.tokenizer.slot_queries",
        "tokenizer.slot_queries",
    )
    matches = [
        value.detach().float().cpu()
        for name, value in state.items()
        if isinstance(value, module.Tensor)
        and any(str(name).endswith(suffix) for suffix in suffixes)
    ]
    if len(matches) != 1:
        raise ValueError("target checkpoint slot-query identity is ambiguous")
    queries = matches[0]
    if queries.ndim != 2 or int(queries.shape[0]) not in {1, 2, 4, 8, 16}:
        raise ValueError("target checkpoint slot-query shape differs")
    if target_mode == "T2_PROJECT":
        projection_state = {
            str(name)[len("projection.") :]: value
            for name, value in state.items()
            if str(name).startswith("projection.")
        }
        up = projection_state.get("up.weight")
        if not isinstance(up, module.Tensor) or up.ndim != 2:
            raise ValueError("T2 checkpoint projection state differs")
        projection = BridgeProjection(
            int(queries.shape[1]), int(up.shape[1])
        )
        projection.load_state_dict(projection_state, strict=True)
        projection.eval()
        with module.no_grad():
            queries = projection(queries)
    array = np.ascontiguousarray(
        queries.detach().float().cpu().numpy(), dtype=np.float32
    )
    if not np.isfinite(array).all():
        raise ValueError("target slot queries are nonfinite")
    return array


def target_slot_query_sha256(
    checkpoint_path: str | Path,
    *,
    target_mode: str,
) -> str:
    values = target_slot_queries(
        checkpoint_path, target_mode=target_mode
    )
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


__all__ = ["target_slot_queries", "target_slot_query_sha256"]
