"""Locked RETB uniform and heterogeneous summary-token shapes."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import canonical_sha256, validate_content_hash, with_content_hash
from .registry import EXPERT_ORDER, TOKEN_SHAPES


TOKEN_SHAPE_CONTRACT = "retb_token_shape_registry_v1"
ALLOWED_HETEROGENEOUS_K = (1, 2, 4, 8, 16)
HETEROGENEOUS_SLOT_CAP = 56
HET_PHYSICS = {
    "BASE4": 4,
    "PT": 8,
    "TRACK": 16,
    "PID": 4,
    "CHARGE": 4,
    "DENSITY": 4,
    "REGION": 16,
}


def build_token_shape_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": TOKEN_SHAPE_CONTRACT,
            "schema_version": 1,
            "uniform_shapes": TOKEN_SHAPES,
            "uniform_order": list(TOKEN_SHAPES),
            "token_dimensions": [64, 128],
            "equal_scalar_budget_comparison": {
                "left": "S8_128",
                "right": "S16_64",
                "scalar_count": 1024,
            },
            "heterogeneous": {
                "HET_PHYSICS": HET_PHYSICS,
                "HET_SELECTED": {
                    "allowed_K": list(ALLOWED_HETEROGENEOUS_K),
                    "dimension": 128,
                    "slot_cap": HETEROGENEOUS_SLOT_CAP,
                    "selection_deferred_to_stage_C": True,
                },
                "HET_BEAM": {
                    "allowed_K": list(ALLOWED_HETEROGENEOUS_K),
                    "dimension": 128,
                    "slot_cap": HETEROGENEOUS_SLOT_CAP,
                    "beam_width": 32,
                    "selection_deferred_to_stage_C": True,
                },
            },
            "expert_order": list(EXPERT_ORDER),
            "slot_identity": "learned_query_index",
            "slot_matching": None,
        }
    )


def validate_token_shape_contract(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(payload, expected_contract=TOKEN_SHAPE_CONTRACT)
    semantic = dict(payload)
    semantic.pop("content_hash", None)
    semantic.pop("source", None)
    expected = build_token_shape_contract()
    expected.pop("content_hash")
    if canonical_sha256(semantic) != canonical_sha256(expected):
        raise ValueError("token-shape registry differs from the locked contract")
    return digest


def resolve_uniform_shape(shape_id: str) -> tuple[int, int]:
    try:
        row = TOKEN_SHAPES[str(shape_id)]
    except KeyError as exc:
        raise ValueError(f"unknown uniform token shape {shape_id!r}") from exc
    return int(row["K"]), int(row["D"])


def validate_heterogeneous_allocation(
    allocation: Mapping[str, int],
    *,
    require_exact_cap: bool = False,
) -> dict[str, int]:
    if set(allocation) != set(EXPERT_ORDER):
        raise ValueError("heterogeneous allocation must cover the seven experts")
    canonical = {name: int(allocation[name]) for name in EXPERT_ORDER}
    invalid = {
        name: value
        for name, value in canonical.items()
        if value not in ALLOWED_HETEROGENEOUS_K
    }
    if invalid:
        raise ValueError(f"heterogeneous slot counts are invalid: {invalid}")
    total = sum(canonical.values())
    if total > HETEROGENEOUS_SLOT_CAP:
        raise ValueError("heterogeneous allocation exceeds the 56-slot cap")
    if require_exact_cap and total != HETEROGENEOUS_SLOT_CAP:
        raise ValueError("heterogeneous allocation must use exactly 56 slots")
    return canonical


def resolve_expert_shapes(
    *,
    uniform_shape_id: str | None = None,
    heterogeneous_allocation: Mapping[str, int] | None = None,
) -> dict[str, tuple[int, int]]:
    if (uniform_shape_id is None) == (heterogeneous_allocation is None):
        raise ValueError("select exactly one uniform or heterogeneous shape")
    if uniform_shape_id is not None:
        shape = resolve_uniform_shape(uniform_shape_id)
        return {expert: shape for expert in EXPERT_ORDER}
    allocation = validate_heterogeneous_allocation(
        heterogeneous_allocation or {}
    )
    return {expert: (allocation[expert], 128) for expert in EXPERT_ORDER}


__all__ = [
    "ALLOWED_HETEROGENEOUS_K",
    "HETEROGENEOUS_SLOT_CAP",
    "HET_PHYSICS",
    "TOKEN_SHAPE_CONTRACT",
    "build_token_shape_contract",
    "resolve_expert_shapes",
    "resolve_uniform_shape",
    "validate_heterogeneous_allocation",
    "validate_token_shape_contract",
]
