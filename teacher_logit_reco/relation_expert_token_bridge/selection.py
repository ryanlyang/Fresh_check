"""Deterministic Stage-C loss, shape, and heterogeneous-budget selectors."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .expert_training import EXPERT_LOSS_CANDIDATES
from .registry import EXPERT_ORDER, TOKEN_SHAPES


UNIFORM_SHAPE_METRICS_CONTRACT = "retb_uniform_shape_metrics_v1"
OFFLINE_SHAPE_SELECTION_CONTRACT = "retb_offline_shape_selection_v1"
JOINT_EXPERT_LOSS_SELECTION_CONTRACT = "retb_joint_expert_loss_selection_v1"
HETEROGENEOUS_SELECTION_CONTRACT = "retb_heterogeneous_selection_v1"
PIPELINE_SEEDS = (101, 202, 303)
ALLOWED_K = (1, 2, 4, 8, 16)
HET_PHYSICS = {
    "BASE4": 4,
    "PT": 8,
    "TRACK": 16,
    "PID": 4,
    "CHARGE": 4,
    "DENSITY": 4,
    "REGION": 16,
}


def _finite_metrics(row: Mapping[str, Any]) -> None:
    values = [
        float(row["accuracy"]),
        float(row["cross_entropy"]),
        *[float(value) for value in row["per_class_efficiency"]],
    ]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("selection metric is nonfinite")
    if not 0.0 <= values[0] <= 1.0 or values[1] < 0.0:
        raise ValueError("selection metric lies outside its valid domain")
    if len(row["per_class_efficiency"]) != 10:
        raise ValueError("selection requires ten per-class efficiencies")


def build_uniform_shape_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    stage_c_run_registry_sha256: str,
    val_design_label_manifest_sha256: str,
) -> dict[str, Any]:
    expected = {
        (shape_id, seed)
        for shape_id in TOKEN_SHAPES
        for seed in PIPELINE_SEEDS
    }
    if len(rows) != 21:
        raise ValueError("uniform shape selection requires exactly 21 rows")
    checked = []
    seen = set()
    label_sha = require_sha256(
        val_design_label_manifest_sha256,
        name="val_design_label_manifest_sha256",
    )
    for raw in rows:
        row = dict(raw)
        key = (str(row["shape_id"]), int(row["pipeline_seed"]))
        if key in seen or key not in expected:
            raise ValueError("uniform shape metric key is duplicated or unknown")
        seen.add(key)
        if row.get("split") != "val_design":
            raise ValueError("uniform shape selector may read only val_design")
        if row.get("fusion_variant") != "F_TOKEN_TRANSFORMER":
            raise ValueError("uniform shape selector requires canonical fusion")
        if row.get("label_manifest_sha256") != label_sha:
            raise ValueError("uniform shape label lineage differs")
        _finite_metrics(row)
        for name in (
            "fusion_checkpoint_sha256",
            "fusion_registration_sha256",
            "frozen_cache_sha256",
            "metrics_artifact_sha256",
        ):
            require_sha256(row.get(name), name=f"{key}.{name}")
        checked.append(
            {
                "shape_id": key[0],
                "pipeline_seed": key[1],
                "split": "val_design",
                "fusion_variant": "F_TOKEN_TRANSFORMER",
                "accuracy": float(row["accuracy"]),
                "cross_entropy": float(row["cross_entropy"]),
                "per_class_efficiency": [
                    float(value) for value in row["per_class_efficiency"]
                ],
                "fusion_checkpoint_sha256": row["fusion_checkpoint_sha256"],
                "fusion_registration_sha256": row[
                    "fusion_registration_sha256"
                ],
                "frozen_cache_sha256": row["frozen_cache_sha256"],
                "metrics_artifact_sha256": row["metrics_artifact_sha256"],
                "label_manifest_sha256": label_sha,
            }
        )
    if seen != expected:
        raise ValueError("uniform shape metrics are incomplete")
    return with_content_hash(
        {
            "contract": UNIFORM_SHAPE_METRICS_CONTRACT,
            "schema_version": 1,
            "stage_c_run_registry_sha256": require_sha256(
                stage_c_run_registry_sha256,
                name="stage_c_run_registry_sha256",
            ),
            "val_design_label_manifest_sha256": label_sha,
            "shape_order": list(TOKEN_SHAPES),
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "row_count": 21,
            "rows": sorted(
                checked,
                key=lambda value: (
                    list(TOKEN_SHAPES).index(value["shape_id"]),
                    value["pipeline_seed"],
                ),
            ),
        }
    )


def validate_uniform_shape_metrics(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=UNIFORM_SHAPE_METRICS_CONTRACT
    )
    expected = build_uniform_shape_metrics(
        rows=payload.get("rows", []),
        stage_c_run_registry_sha256=payload.get(
            "stage_c_run_registry_sha256"
        ),
        val_design_label_manifest_sha256=payload.get(
            "val_design_label_manifest_sha256"
        ),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("uniform shape metrics differ")
    return digest


def select_offline_shapes(
    metrics: Mapping[str, Any],
    *,
    baseline_mean_accuracy: float | None = None,
) -> dict[str, Any]:
    validate_uniform_shape_metrics(metrics)
    rows = list(metrics.get("rows", []))
    if len(rows) != 21:
        raise ValueError("offline shape selector input is incomplete")
    aggregates = []
    for shape_id, shape in TOKEN_SHAPES.items():
        selected = [row for row in rows if row["shape_id"] == shape_id]
        if {row["pipeline_seed"] for row in selected} != set(PIPELINE_SEEDS):
            raise ValueError(f"shape {shape_id} lacks complete three-seed inputs")
        for row in selected:
            _finite_metrics(row)
        aggregates.append(
            {
                "shape_id": shape_id,
                "K": int(shape["K"]),
                "D": int(shape["D"]),
                "scalars": int(shape["K"] * shape["D"]),
                "mean_accuracy": sum(row["accuracy"] for row in selected) / 3,
                "mean_cross_entropy": (
                    sum(row["cross_entropy"] for row in selected) / 3
                ),
                "mean_per_class_efficiency": [
                    sum(row["per_class_efficiency"][index] for row in selected)
                    / 3
                    for index in range(10)
                ],
                "all_seed_metrics_finite": True,
                "parent_rows": [
                    {
                        "pipeline_seed": row["pipeline_seed"],
                        "fusion_checkpoint_sha256": row[
                            "fusion_checkpoint_sha256"
                        ],
                        "metrics_artifact_sha256": row[
                            "metrics_artifact_sha256"
                        ],
                    }
                    for row in selected
                ],
            }
        )
    maximum = max(row["mean_accuracy"] for row in aggregates)
    high_eligible = [
        row for row in aggregates if maximum - row["mean_accuracy"] <= 0.0005
    ]
    high = min(
        high_eligible,
        key=lambda row: (
            row["mean_cross_entropy"],
            row["scalars"],
            row["K"],
            row["D"],
            row["shape_id"],
        ),
    )
    compact_rows = []
    for row in sorted(
        aggregates,
        key=lambda value: (
            value["scalars"],
            value["K"],
            value["D"],
            value["shape_id"],
        ),
    ):
        worst_deficit = max(
            high["mean_per_class_efficiency"][index]
            - row["mean_per_class_efficiency"][index]
            for index in range(10)
        )
        eligible = (
            high["mean_accuracy"] - row["mean_accuracy"] <= 0.0020
            and row["mean_cross_entropy"] - high["mean_cross_entropy"] <= 0.0050
            and worst_deficit <= 0.0100
            and row["all_seed_metrics_finite"]
        )
        compact_rows.append(
            {
                "shape_id": row["shape_id"],
                "accuracy_deficit": high["mean_accuracy"]
                - row["mean_accuracy"],
                "cross_entropy_increase": row["mean_cross_entropy"]
                - high["mean_cross_entropy"],
                "worst_per_class_efficiency_deficit": worst_deficit,
                "eligible": eligible,
            }
        )
    compact = next(
        (
            next(row for row in aggregates if row["shape_id"] == audit["shape_id"])
            for audit in compact_rows
            if audit["eligible"]
        ),
        high,
    )
    baseline = (
        None
        if baseline_mean_accuracy is None
        else float(baseline_mean_accuracy)
    )
    carried = []
    for shape in ("S1_128", compact["shape_id"], high["shape_id"]):
        if shape not in carried:
            carried.append(shape)
    carried.extend(["HET_PHYSICS", "HET_SELECTED", "HET_BEAM"])
    return with_content_hash(
        {
            "contract": OFFLINE_SHAPE_SELECTION_CONTRACT,
            "schema_version": 1,
            "uniform_shape_metrics_sha256": require_sha256(
                metrics.get("content_hash"),
                name="uniform_shape_metrics.content_hash",
            ),
            "complete_shape_count": 7,
            "complete_seed_shape_rows": 21,
            "ranking": sorted(
                aggregates,
                key=lambda row: (
                    -row["mean_accuracy"],
                    row["mean_cross_entropy"],
                    row["scalars"],
                    row["K"],
                    row["D"],
                    row["shape_id"],
                ),
            ),
            "high_accuracy_window": 0.0005,
            "SHAPE_HIGH": {
                key: high[key] for key in ("shape_id", "K", "D", "scalars")
            },
            "compact_audit": compact_rows,
            "SHAPE_COMPACT": {
                key: compact[key]
                for key in ("shape_id", "K", "D", "scalars")
            },
            "carried_shapes_duplicate_free": carried,
            "baseline_mean_accuracy": baseline,
            "all_multi_expert_models_worse_than_baseline": (
                None
                if baseline is None
                else all(row["mean_accuracy"] < baseline for row in aggregates)
            ),
            "selection_emitted_despite_scientific_result": True,
        }
    )


def _retain(
    rows: Sequence[Mapping[str, Any]],
    *,
    width: int,
    tuple_field: str,
    include_slots: bool = False,
) -> list[dict[str, Any]]:
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
            *([int(row["total_slots"])] if include_slots else []),
            tuple(row[tuple_field]),
        ),
    )[: int(width)]


def select_joint_expert_losses(
    *,
    eligible_variants: Mapping[str, Sequence[str]],
    individual_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    pooled_scorer: Callable[[tuple[str, ...], int], Mapping[str, Any]],
    transformer_scorer: Callable[[tuple[str, ...], int], Mapping[str, Any]],
    shape_id: str,
) -> dict[str, Any]:
    if set(eligible_variants) != set(EXPERT_ORDER) or set(
        individual_metrics
    ) != set(EXPERT_ORDER):
        raise ValueError("joint loss selector expert coverage differs")
    defaults = []
    for expert in EXPERT_ORDER:
        variants = list(eligible_variants[expert])
        if not variants or any(v not in EXPERT_LOSS_CANDIDATES for v in variants):
            raise ValueError("joint loss selector variant is ineligible")
        maximum = max(
            float(individual_metrics[expert][variant]["accuracy"])
            for variant in variants
        )
        eligible_default = [
            variant
            for variant in variants
            if maximum
            - float(individual_metrics[expert][variant]["accuracy"])
            <= 0.0001
        ]
        defaults.append(
            min(
                eligible_default,
                key=lambda variant: (
                    float(individual_metrics[expert][variant]["cross_entropy"]),
                    float(
                        individual_metrics[expert][variant].get(
                            "measured_flops", 0.0
                        )
                    ),
                    int(
                        individual_metrics[expert][variant].get(
                            "parameter_count", 0
                        )
                    ),
                    variant,
                ),
            )
        )
    beam = [tuple(defaults)]
    trace = []
    for depth, expert in enumerate(EXPERT_ORDER):
        candidates = []
        for retained in beam:
            for variant in eligible_variants[expert]:
                proposal = list(retained)
                proposal[depth] = variant
                proposal_tuple = tuple(proposal)
                score = pooled_scorer(proposal_tuple, 41701)
                candidates.append(
                    {
                        "loss_tuple": proposal_tuple,
                        "accuracy": float(score["accuracy"]),
                        "cross_entropy": float(score["cross_entropy"]),
                        "measured_flops": float(score.get("measured_flops", 0.0)),
                        "parameter_count": int(score.get("parameter_count", 0)),
                        "readout_sha256": require_sha256(
                            score["readout_sha256"], name="readout_sha256"
                        ),
                    }
                )
        retained_rows = _retain(
            candidates, width=16, tuple_field="loss_tuple"
        )
        beam = [tuple(row["loss_tuple"]) for row in retained_rows]
        trace.append(
            {
                "depth": depth,
                "expert": expert,
                "retained_loss_tuples": [
                    list(row["loss_tuple"]) for row in retained_rows
                ],
            }
        )
    top_four = retained_rows[:4]
    finalists = []
    for row in top_four:
        score = transformer_scorer(tuple(row["loss_tuple"]), 41701)
        finalists.append(
            {
                "loss_tuple": tuple(row["loss_tuple"]),
                "accuracy": float(score["accuracy"]),
                "cross_entropy": float(score["cross_entropy"]),
                "measured_flops": float(score.get("measured_flops", 0.0)),
                "parameter_count": int(score.get("parameter_count", 0)),
                "fusion_sha256": require_sha256(
                    score["fusion_sha256"], name="fusion_sha256"
                ),
            }
        )
    selected = _retain(finalists, width=1, tuple_field="loss_tuple")[0]
    homogeneous = [
        [variant] * 7
        for variant in sorted(
            set.intersection(
                *[set(eligible_variants[name]) for name in EXPERT_ORDER]
            )
        )
    ]
    all_ce = ["ELOSS_CE"] * 7
    if all_ce not in homogeneous:
        homogeneous.append(all_ce)
    return with_content_hash(
        {
            "contract": JOINT_EXPERT_LOSS_SELECTION_CONTRACT,
            "schema_version": 1,
            "shape_id": str(shape_id),
            "expert_order": list(EXPERT_ORDER),
            "readout_seed": 41701,
            "beam_width": 16,
            "default_tuple": defaults,
            "beam_trace": trace,
            "top_four_mixed_tuples": [
                list(row["loss_tuple"]) for row in top_four
            ],
            "selected_tuple": list(selected["loss_tuple"]),
            "selected_fusion_sha256": selected["fusion_sha256"],
            "homogeneous_controls": homogeneous,
            "all_CE_control_present": all_ce in homogeneous,
            "independent_per_expert_selection_used": False,
        }
    )


def greedy_heterogeneous_allocation(
    scorer: Callable[[Mapping[str, int], int], Mapping[str, Any]],
) -> dict[str, Any]:
    seeds = (101, 202, 303)

    def matched_score(allocation: Mapping[str, int]) -> dict[str, Any]:
        rows = [dict(scorer(dict(allocation), seed)) for seed in seeds]
        return {
            "accuracy": sum(float(row["accuracy"]) for row in rows)
            / len(rows),
            "cross_entropy": sum(
                float(row["cross_entropy"]) for row in rows
            )
            / len(rows),
            "readout_sha256_by_seed": {
                str(seed): require_sha256(
                    row["readout_sha256"],
                    name=f"readout_sha256.seed_{seed}",
                )
                for seed, row in zip(seeds, rows, strict=True)
            },
        }

    allocation = {expert: 1 for expert in EXPERT_ORDER}
    current_score = matched_score(allocation)
    trace = []
    while True:
        candidates = []
        for expert_index, expert in enumerate(EXPERT_ORDER):
            current = allocation[expert]
            next_values = [value for value in ALLOWED_K if value > current]
            if not next_values:
                continue
            proposed = next_values[0]
            added = proposed - current
            if sum(allocation.values()) + added > 56:
                continue
            candidate = dict(allocation)
            candidate[expert] = proposed
            score = matched_score(candidate)
            gain = float(score["accuracy"]) - float(current_score["accuracy"])
            candidates.append(
                {
                    "expert": expert,
                    "expert_index": expert_index,
                    "allocation": candidate,
                    "added_slots": added,
                    "gain_per_added_slot": gain / added,
                    "accuracy": float(score["accuracy"]),
                    "cross_entropy": float(score["cross_entropy"]),
                    "readout_sha256_by_seed": score[
                        "readout_sha256_by_seed"
                    ],
                }
            )
        if not candidates:
            break
        selected = min(
            candidates,
            key=lambda row: (
                -row["gain_per_added_slot"],
                row["cross_entropy"],
                row["added_slots"],
                row["expert_index"],
            ),
        )
        allocation = dict(selected["allocation"])
        current_score = selected
        trace.append(
            {
                "selected_expert": selected["expert"],
                "allocation": allocation,
                "total_slots": sum(allocation.values()),
                "gain_per_added_slot": selected["gain_per_added_slot"],
                "readout_sha256_by_seed": selected[
                    "readout_sha256_by_seed"
                ],
            }
        )
    return {
        "allocation": allocation,
        "total_slots": sum(allocation.values()),
        "trace": trace,
    }


def beam_heterogeneous_allocation(
    pooled_scorer: Callable[[Mapping[str, int], int], Mapping[str, Any]],
    transformer_scorer: Callable[[Mapping[str, int], int], Mapping[str, Any]],
) -> dict[str, Any]:
    beam = [dict()]
    trace = []
    for depth, expert in enumerate(EXPERT_ORDER):
        rows = []
        remaining = len(EXPERT_ORDER) - depth - 1
        for partial in beam:
            for value in ALLOWED_K:
                proposal = {**partial, expert: value}
                if sum(proposal.values()) + remaining > 56:
                    continue
                completed = {
                    name: proposal.get(name, 1) for name in EXPERT_ORDER
                }
                score = pooled_scorer(completed, 41702)
                rows.append(
                    {
                        "partial": proposal,
                        "allocation_tuple": tuple(
                            completed[name] for name in EXPERT_ORDER
                        ),
                        "accuracy": float(score["accuracy"]),
                        "cross_entropy": float(score["cross_entropy"]),
                        "total_slots": sum(completed.values()),
                        "readout_sha256": require_sha256(
                            score["readout_sha256"], name="readout_sha256"
                        ),
                    }
                )
        retained = _retain(
            rows,
            width=32,
            tuple_field="allocation_tuple",
            include_slots=True,
        )
        beam = [dict(row["partial"]) for row in retained]
        trace.append(
            {
                "depth": depth,
                "expert": expert,
                "retained_allocations": [
                    list(row["allocation_tuple"]) for row in retained
                ],
            }
        )
    complete = []
    for allocation in beam:
        if set(allocation) != set(EXPERT_ORDER) or sum(allocation.values()) > 56:
            raise RuntimeError("heterogeneous beam produced invalid allocation")
        score = pooled_scorer(allocation, 41702)
        complete.append(
            {
                "allocation": allocation,
                "allocation_tuple": tuple(
                    allocation[name] for name in EXPERT_ORDER
                ),
                "accuracy": float(score["accuracy"]),
                "cross_entropy": float(score["cross_entropy"]),
                "total_slots": sum(allocation.values()),
            }
        )
    top_four = _retain(
        complete,
        width=4,
        tuple_field="allocation_tuple",
        include_slots=True,
    )
    transformer_rows = []
    for row in top_four:
        score = transformer_scorer(row["allocation"], 41702)
        transformer_rows.append(
            {
                **row,
                "accuracy": float(score["accuracy"]),
                "cross_entropy": float(score["cross_entropy"]),
                "fusion_sha256": require_sha256(
                    score["fusion_sha256"], name="fusion_sha256"
                ),
            }
        )
    selected = _retain(
        transformer_rows,
        width=1,
        tuple_field="allocation_tuple",
        include_slots=True,
    )[0]
    return {
        "allocation": selected["allocation"],
        "total_slots": selected["total_slots"],
        "top_four_allocations": [
            row["allocation"] for row in top_four
        ],
        "selected_fusion_sha256": selected["fusion_sha256"],
        "trace": trace,
    }


def select_heterogeneous_allocations(
    *,
    greedy_scorer: Callable[[Mapping[str, int], int], Mapping[str, Any]],
    beam_pooled_scorer: Callable[[Mapping[str, int], int], Mapping[str, Any]],
    beam_transformer_scorer: Callable[
        [Mapping[str, int], int], Mapping[str, Any]
    ],
) -> dict[str, Any]:
    greedy = greedy_heterogeneous_allocation(greedy_scorer)
    beam = beam_heterogeneous_allocation(
        beam_pooled_scorer, beam_transformer_scorer
    )
    if (
        sum(HET_PHYSICS.values()) != 56
        or greedy["total_slots"] > 56
        or beam["total_slots"] > 56
    ):
        raise RuntimeError("heterogeneous slot budget was exceeded")
    return with_content_hash(
        {
            "contract": HETEROGENEOUS_SELECTION_CONTRACT,
            "schema_version": 1,
            "expert_order": list(EXPERT_ORDER),
            "allowed_K": list(ALLOWED_K),
            "dimension": 128,
            "slot_cap": 56,
            "HET_PHYSICS": HET_PHYSICS,
            "HET_SELECTED": greedy,
            "HET_BEAM": beam,
            "greedy_pipeline_seeds": [101, 202, 303],
            "greedy_score_aggregation": (
                "arithmetic_mean_accuracy_and_cross_entropy"
            ),
            "beam_readout_seed": 41702,
            "beam_width": 32,
            "allocations_frozen_before_final_fusion_training": True,
        }
    )


__all__ = [
    "HETEROGENEOUS_SELECTION_CONTRACT",
    "HET_PHYSICS",
    "JOINT_EXPERT_LOSS_SELECTION_CONTRACT",
    "OFFLINE_SHAPE_SELECTION_CONTRACT",
    "UNIFORM_SHAPE_METRICS_CONTRACT",
    "beam_heterogeneous_allocation",
    "build_uniform_shape_metrics",
    "greedy_heterogeneous_allocation",
    "select_heterogeneous_allocations",
    "select_joint_expert_losses",
    "select_offline_shapes",
    "validate_uniform_shape_metrics",
]
