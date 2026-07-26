"""Exact symbolic capacity controls for the locked Weaver pair stem."""

from __future__ import annotations

import bisect
from typing import Any, Iterable

from .contracts import with_content_hash


WIDE_CAPACITY_CONTRACT = "relational_part_wide_capacity_match_v1"
PAIR_PARAMETER_FORMULA_VERSION = "weaver_pair_embed_bn_conv_v1"
REFERENCE_WIDTHS = (64, 64, 64)
PAIR_HEADS = 8
FULL_RELATION_ENCODED_DIMENSION = 58


def _linear_parameters(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim + output_dim


def family_encoder_parameter_count() -> dict[str, int]:
    counts = {
        "PT": _linear_parameters(10, 32) + 32 + _linear_parameters(32, 8),
        "TRACK": (
            _linear_parameters(7, 32) + 32 + _linear_parameters(32, 16)
            + _linear_parameters(81, 48) + 48 + _linear_parameters(48, 12)
        ),
        "PID": 6 * 8 + 6 * 8 + 36 * 8,
        "CHARGE": (
            9 * 4 + _linear_parameters(12, 32) + 32
            + _linear_parameters(32, 6)
        ),
        "DENSITY": (
            _linear_parameters(66, 32) + 32 + _linear_parameters(32, 12)
        ),
        "REGION": (
            _linear_parameters(41, 32) + 32 + _linear_parameters(32, 12)
        ),
    }
    return counts


def pair_encoder_parameter_count(
    input_dimension: int,
    widths: tuple[int, int, int],
) -> int:
    """BatchNorm(input), three Conv+BN blocks, and final head Conv."""

    w1, w2, w3 = map(int, widths)
    return (
        2 * int(input_dimension)
        + (int(input_dimension) * w1 + w1 + 2 * w1)
        + (w1 * w2 + w2 + 2 * w2)
        + (w2 * w3 + w3 + 2 * w3)
        + (w3 * PAIR_HEADS + PAIR_HEADS + 2 * PAIR_HEADS)
    )


def pair_encoder_flops(
    input_dimension: int,
    widths: tuple[int, int, int],
    *,
    valid_particles: int = 128,
) -> int:
    """Analytical multiply-add FLOPs for all directed valid pairs."""

    w1, w2, w3 = map(int, widths)
    per_pair = 2 * (
        int(input_dimension) * w1
        + w1 * w2
        + w2 * w3
        + w3 * PAIR_HEADS
    )
    return int(valid_particles) ** 2 * per_pair


def full_active_incremental_parameters() -> int:
    family_parameters = sum(family_encoder_parameter_count().values())
    stem_increment = pair_encoder_parameter_count(
        4 + FULL_RELATION_ENCODED_DIMENSION, REFERENCE_WIDTHS
    ) - pair_encoder_parameter_count(4, REFERENCE_WIDTHS)
    return family_parameters + stem_increment


def select_wide_widths(
    *,
    target_increment: int | None = None,
    candidates: Iterable[int] = range(64, 257),
) -> dict[str, Any]:
    target = (
        full_active_incremental_parameters()
        if target_increment is None
        else int(target_increment)
    )
    base_parameters = pair_encoder_parameter_count(4, REFERENCE_WIDTHS)
    values = sorted(set(int(value) for value in candidates))
    best: tuple[Any, ...] | None = None
    for w1 in values:
        for w2 in values:
            # For fixed w1,w2 the parameter count is affine in w3.
            without_w3 = pair_encoder_parameter_count(4, (w1, w2, 0))
            desired = (target + base_parameters - without_w3) / (w2 + 11)
            position = bisect.bisect_left(values, desired)
            candidate_w3 = {
                values[max(0, min(len(values) - 1, position + offset))]
                for offset in (-1, 0, 1)
            }
            for w3 in candidate_w3:
                widths = (int(w1), int(w2), int(w3))
                increment = pair_encoder_parameter_count(4, widths) - base_parameters
                mismatch = abs(increment - target)
                flops = pair_encoder_flops(4, widths)
                key = (mismatch, flops, sum(widths), widths)
                if best is None or key < best[0]:
                    best = (key, increment, flops)
    if best is None:
        raise ValueError("wide capacity candidate set is empty")
    key, increment, flops = best
    widths = key[3]
    relative = key[0] / max(target, 1)
    if relative > .02:
        raise ValueError("wide incremental capacity mismatch exceeds two percent")
    return with_content_hash(
        {
            "contract": WIDE_CAPACITY_CONTRACT,
            "schema_version": 1,
            "parameter_formula_version": PAIR_PARAMETER_FORMULA_VERSION,
            "reference_widths": list(REFERENCE_WIDTHS),
            "candidate_range_inclusive": [64, 256],
            "full_relation_encoded_dimension": FULL_RELATION_ENCODED_DIMENSION,
            "family_encoder_parameters": family_encoder_parameter_count(),
            "target_full_incremental_parameters": target,
            "selected_widths": list(widths),
            "wide_incremental_parameters": increment,
            "absolute_incremental_mismatch": key[0],
            "relative_incremental_mismatch": relative,
            "analytical_pair_encoder_flops_at_128": flops,
            "tie_breaks": [
                "minimum_absolute_incremental_parameter_mismatch",
                "lower_analytically_calculated_pair_encoder_FLOPs_at_128_valid_particles",
                "smaller_width_sum",
                "smaller_width_tuple_lexicographically",
            ],
        }
    )


__all__ = [
    "FULL_RELATION_ENCODED_DIMENSION",
    "PAIR_PARAMETER_FORMULA_VERSION",
    "REFERENCE_WIDTHS",
    "WIDE_CAPACITY_CONTRACT",
    "family_encoder_parameter_count",
    "full_active_incremental_parameters",
    "pair_encoder_flops",
    "pair_encoder_parameter_count",
    "select_wide_widths",
]
