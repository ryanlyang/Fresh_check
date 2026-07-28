"""Joint Stage-E target-coordinate and coordinate-specific fusion selector."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .contracts import require_sha256, with_content_hash
from .registry import EXPERT_ORDER


BRIDGE_COORDINATE_SELECTION_CONTRACT = (
    "retb_joint_bridge_target_coordinate_selection_v1"
)
ELIGIBLE_TOKEN_MODES = {
    "T0_PURE",
    "T1_ANCHORED_BRIDGE",
    "T1_TASK_BRIDGE",
    "T2_PROJECT",
}


def _retain(
    rows: Sequence[Mapping[str, Any]], *, width: int
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("bridge coordinate selector has no candidates")
    maximum = max(float(row["accuracy"]) for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if maximum - float(row["accuracy"]) <= 0.0001
    ]
    return sorted(
        eligible,
        key=lambda row: (
            float(row["cross_entropy"]),
            float(row.get("measured_flops", 0.0)),
            int(row.get("parameter_count", 0)),
            tuple(row["target_tuple"]),
        ),
    )[: int(width)]


def select_joint_bridge_coordinates(
    *,
    eligible_modes: Mapping[str, Sequence[str]],
    default_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    pooled_scorer: Callable[[tuple[str, ...], int], Mapping[str, Any]],
    transformer_scorer: Callable[[tuple[str, ...], int], Mapping[str, Any]],
    shape_id: str,
    eligibility_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if (
        set(eligible_modes) != set(EXPERT_ORDER)
        or set(default_metrics) != set(EXPERT_ORDER)
        or set(eligibility_hashes) != set(EXPERT_ORDER)
    ):
        raise ValueError("bridge coordinate selector expert coverage differs")
    defaults = []
    for expert in EXPERT_ORDER:
        modes = list(eligible_modes[expert])
        if (
            not modes
            or "T0_PURE" not in modes
            or any(mode not in ELIGIBLE_TOKEN_MODES for mode in modes)
            or set(eligibility_hashes[expert]) != set(modes)
        ):
            raise ValueError("bridge coordinate eligibility differs")
        for mode in modes:
            require_sha256(
                eligibility_hashes[expert][mode],
                name=f"eligibility_hashes.{expert}.{mode}",
            )
        maximum = max(
            float(default_metrics[expert][mode]["accuracy"]) for mode in modes
        )
        close = [
            mode
            for mode in modes
            if maximum - float(default_metrics[expert][mode]["accuracy"])
            <= 0.0001
        ]
        defaults.append(
            min(
                close,
                key=lambda mode: (
                    float(default_metrics[expert][mode]["cross_entropy"]),
                    float(
                        default_metrics[expert][mode].get(
                            "measured_flops", 0.0
                        )
                    ),
                    int(
                        default_metrics[expert][mode].get(
                            "parameter_count", 0
                        )
                    ),
                    mode,
                ),
            )
        )
    beam = [tuple(defaults)]
    trace = []
    for depth, expert in enumerate(EXPERT_ORDER):
        scored = []
        seen = set()
        for retained in beam:
            for mode in eligible_modes[expert]:
                proposal = list(retained)
                proposal[depth] = mode
                target_tuple = tuple(proposal)
                if target_tuple in seen:
                    continue
                seen.add(target_tuple)
                score = pooled_scorer(target_tuple, 41703)
                scored.append(
                    {
                        "target_tuple": target_tuple,
                        "accuracy": float(score["accuracy"]),
                        "cross_entropy": float(score["cross_entropy"]),
                        "normalized_token_error": float(
                            score.get("normalized_token_error", 0.0)
                        ),
                        "measured_flops": float(
                            score.get("measured_flops", 0.0)
                        ),
                        "parameter_count": int(
                            score.get("parameter_count", 0)
                        ),
                        "readout_sha256": require_sha256(
                            score["readout_sha256"], name="readout_sha256"
                        ),
                    }
                )
        retained_rows = _retain(scored, width=16)
        retained_hashes = [row["readout_sha256"] for row in retained_rows]
        if len(retained_hashes) != len(set(retained_hashes)):
            raise ValueError("provisional target tuples reused a pooled readout")
        beam = [tuple(row["target_tuple"]) for row in retained_rows]
        trace.append(
            {
                "depth": depth,
                "expert": expert,
                "retained_target_tuples": [
                    list(row["target_tuple"]) for row in retained_rows
                ],
                "readout_hashes": [
                    row["readout_sha256"] for row in retained_rows
                ],
            }
        )
    top_four = [tuple(row["target_tuple"]) for row in retained_rows[:4]]
    common_modes = sorted(
        set.intersection(*[set(eligible_modes[name]) for name in EXPERT_ORDER])
    )
    homogeneous = [tuple([mode] * len(EXPERT_ORDER)) for mode in common_modes]
    locked_tuples = []
    for target_tuple in [*top_four, *homogeneous]:
        if target_tuple not in locked_tuples:
            locked_tuples.append(target_tuple)
    fusion_rows = []
    for target_tuple in locked_tuples:
        score = transformer_scorer(target_tuple, 41703)
        fusion_rows.append(
            {
                "target_tuple": target_tuple,
                "accuracy": float(score["accuracy"]),
                "cross_entropy": float(score["cross_entropy"]),
                "normalized_token_error": float(
                    score.get("normalized_token_error", 0.0)
                ),
                "measured_flops": float(score.get("measured_flops", 0.0)),
                "parameter_count": int(score.get("parameter_count", 0)),
                "fusion_sha256": require_sha256(
                    score["fusion_sha256"], name="fusion_sha256"
                ),
                "normalizer_set_sha256": require_sha256(
                    score["normalizer_set_sha256"],
                    name="normalizer_set_sha256",
                ),
                "target_cache_namespace": str(score["target_cache_namespace"]),
            }
        )
    for field in (
        "fusion_sha256",
        "normalizer_set_sha256",
        "target_cache_namespace",
    ):
        values = [row[field] for row in fusion_rows]
        if len(values) != len(set(values)):
            raise ValueError(
                f"locked coordinate systems reused {field}"
            )
    selected = _retain(fusion_rows, width=1)[0]
    return with_content_hash(
        {
            "contract": BRIDGE_COORDINATE_SELECTION_CONTRACT,
            "schema_version": 1,
            "shape_id": str(shape_id),
            "expert_order": list(EXPERT_ORDER),
            "readout_seed": 41703,
            "beam_width": 16,
            "global_accuracy_window": 0.0001,
            "default_tuple": defaults,
            "beam_trace": trace,
            "top_four_mixed_target_tuples": [list(row) for row in top_four],
            "homogeneous_target_tuples": [list(row) for row in homogeneous],
            "locked_coordinate_systems": [
                {
                    **row,
                    "target_tuple": list(row["target_tuple"]),
                    "coordinate_contract_sha256": with_content_hash(
                        {
                            "contract": "retb_locked_target_coordinate_v1",
                            "schema_version": 1,
                            "shape_id": str(shape_id),
                            "target_tuple": list(row["target_tuple"]),
                            "fusion_sha256": row["fusion_sha256"],
                            "normalizer_set_sha256": row[
                                "normalizer_set_sha256"
                            ],
                            "target_cache_namespace": row[
                                "target_cache_namespace"
                            ],
                        }
                    )["content_hash"],
                }
                for row in fusion_rows
            ],
            "selected_target_tuple": list(selected["target_tuple"]),
            "selected_fusion_sha256": selected["fusion_sha256"],
            "eligibility_hashes": {
                expert: dict(sorted(eligibility_hashes[expert].items()))
                for expert in EXPERT_ORDER
            },
            "new_cross_mode_tuples_after_lock_permitted": False,
            "all_negative_campaign_still_emits": True,
        }
    )


__all__ = [
    "BRIDGE_COORDINATE_SELECTION_CONTRACT",
    "select_joint_bridge_coordinates",
]
