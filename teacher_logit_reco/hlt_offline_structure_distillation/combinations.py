"""Deterministic Stage-F combinations, PCGrad, and Stage-G interventions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.complementarity import (
    linear_cka,
)

from .auxiliary import select_utility_row
from .contracts import (
    COMBINATION_BEAM_COMPLETION_CONTRACT,
    COMBINATION_BEAM_PROMOTION_CONTRACT,
    COMBINATION_WAVE_COMPLETION_CONTRACT,
    COMBINATION_RESULT_CONTRACT,
    COMBINATION_SELECTION_CONTRACT,
    FEEDBACK_SELECTION_CONTRACT,
    GRADIENT_CONFLICT_CONTRACT,
    MECHANISM_CONTROL_PLAN_CONTRACT,
    MECHANISM_RESULT_CONTRACT,
    MECHANISM_SUMMARY_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    STAGE_F_PLAN_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


FAMILY_ORDER = (
    "JET",
    "COMPOSITION",
    "TRACK",
    "DENSITY",
    "CA_REGION",
    "OTHER_RELATIONS",
    "TEACHER",
    "LATENT",
)
MANDATORY_COMBINATIONS = (
    "C_PHYSICAL",
    "C_TRACK_TOPOLOGY",
    "C_PHYSICAL_KD",
    "C_PHYSICAL_LATENT",
    "C_ALL_BEST",
    "C_NATIVE_OFFLINE",
)


def target_family(target_id: str) -> str:
    value = str(target_id)
    if value == "T_OFFLINE_JET_10":
        return "JET"
    if value == "T_OFFLINE_COMPOSITION_16":
        return "COMPOSITION"
    if value in {
        "T_OFFLINE_TRACK_32",
        "T_OFFLINE_TRACK_COMPONENT_PROXY_17",
        "T_OFFLINE_RELATION_TRACK",
    }:
        return "TRACK"
    if value in {"T_OFFLINE_DENSITY_22", "T_OFFLINE_RELATION_DENSITY"}:
        return "DENSITY"
    if value in {
        "T_OFFLINE_CA_TREE_26",
        "T_OFFLINE_RELATION_REGION",
    }:
        return "CA_REGION"
    if value.startswith("T_OFFLINE_RELATION_"):
        return "OTHER_RELATIONS"
    if value.startswith("T_OFFLINE_LOGITS_"):
        return "TEACHER"
    if value == "T_OFFLINE_POOLED_LATENT":
        return "LATENT"
    raise ValueError(f"target has no combination family: {value}")


def normalize_combination_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    checked = {str(key): float(value) for key, value in weights.items()}
    if not checked:
        raise ValueError("combination must contain at least one weighted member")
    if any(not math.isfinite(value) or value < 0 for value in checked.values()):
        raise ValueError("combination weights must be finite and nonnegative")
    total = math.fsum(checked.values())
    if total <= 0:
        raise ValueError("combination weights must have positive sum")
    cap = min(total, 1.0)
    scale = cap / total
    return {key: checked[key] * scale for key in sorted(checked)}


def _combination(
    combination_id: str,
    members: Sequence[Mapping[str, Any]],
    *,
    budget: str,
    weighting: str = "W_FIXED",
) -> dict[str, Any]:
    ordered = sorted(
        (dict(member) for member in members),
        key=lambda row: (FAMILY_ORDER.index(row["family"]), row["target_id"]),
    )
    if len({row["family"] for row in ordered}) != len(ordered):
        raise ValueError("combination contains duplicate family groups")
    weights = normalize_combination_weights(
        {row["target_id"]: float(row["auxiliary_weight"]) for row in ordered}
    )
    semantics = {
        "combination_id": combination_id,
        "members": ordered,
        "normalized_weights": weights,
        "budget": budget,
        "weighting": weighting,
    }
    return {
        "graph_id": f"COMBO_{canonical_sha256(semantics)[:16]}",
        **semantics,
        "selection_eligible": budget == "FULL" and weighting == "W_FIXED",
        "fixed_epoch_budget": 5 if budget == "BEAM_5_EPOCH" else 40,
        "performance_can_omit_or_cancel": False,
    }


def _native_offline_combination(member: Mapping[str, Any]) -> dict[str, Any]:
    native = {
        "baseline_id": "H_NATIVE_REL_AUX",
        "target_dimension": 545,
        "availability_groups": 7,
        "auxiliary_weight": 0.30,
        "target_source": "runtime_HLT_relation_summaries",
    }
    normalized = normalize_combination_weights(
        {
            str(member["target_id"]): float(member["auxiliary_weight"]),
            "H_NATIVE_REL_AUX": float(native["auxiliary_weight"]),
        }
    )
    semantics = {
        "combination_id": "C_NATIVE_OFFLINE",
        "members": [dict(member)],
        "native_relation_auxiliary": native,
        "normalized_weights": normalized,
        "budget": "FULL",
        "weighting": "W_FIXED",
    }
    return {
        "graph_id": f"COMBO_{canonical_sha256(semantics)[:16]}",
        **semantics,
        "selection_eligible": True,
        "fixed_epoch_budget": 40,
        "performance_can_omit_or_cancel": False,
    }


def build_stage_f_plan(
    *,
    single_family_selection: Mapping[str, Any],
    feedback_selection: Mapping[str, Any],
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        single_family_selection,
        expected_contract=SINGLE_FAMILY_SELECTION_CONTRACT,
    )
    validate_content_hash(
        feedback_selection, expected_contract=FEEDBACK_SELECTION_CONTRACT
    )
    definitions = single_family_selection["selected_definition_by_target"]
    order = {
        row["target_id"]: int(row["ordinal"])
        for row in single_family_selection["cross_family_order"]
    }
    grouped: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILY_ORDER}
    for target_id, definition in definitions.items():
        try:
            family = target_family(target_id)
        except ValueError:
            continue
        grouped[family].append(
            {
                **dict(definition),
                "family": family,
                "selected_row_id": single_family_selection[
                    "selected_row_by_target"
                ][target_id],
                "utility_ordinal": order[target_id],
            }
        )
    missing = [family for family in FAMILY_ORDER if not grouped[family]]
    if missing:
        raise ValueError(f"combination plan lacks family groups: {missing}")
    best = {
        family: min(rows, key=lambda row: (row["utility_ordinal"], row["target_id"]))
        for family, rows in grouped.items()
    }
    physical_families = (
        "JET",
        "COMPOSITION",
        "TRACK",
        "DENSITY",
        "CA_REGION",
        "OTHER_RELATIONS",
    )
    physical = [best[family] for family in physical_families]
    all_best_members = [best[family] for family in FAMILY_ORDER]
    mandatory = [
        _combination("C_PHYSICAL", physical, budget="FULL"),
        _combination(
            "C_TRACK_TOPOLOGY",
            [best["TRACK"], best["CA_REGION"]],
            budget="FULL",
        ),
        _combination(
            "C_PHYSICAL_KD",
            [*physical, best["TEACHER"]],
            budget="FULL",
        ),
        _combination(
            "C_PHYSICAL_LATENT",
            [*physical, best["LATENT"]],
            budget="FULL",
        ),
        _combination(
            "C_ALL_BEST",
            all_best_members,
            budget="FULL",
        ),
        _native_offline_combination(
            min(
                physical,
                key=lambda row: (row["utility_ordinal"], row["target_id"]),
            )
        ),
    ]
    pcgrad_control = _combination(
        "C_ALL_BEST_PCGRAD",
        all_best_members,
        budget="FULL",
        weighting="W_PCGRAD",
    )
    # This is the immutable member pool.  The runtime beam starts at the
    # separately trained H_BASE_BEAM_BUDGET root.  At each family it reuses
    # every omit score and trains only the add expansion, so at most 12 new
    # fits are needed per family and the exact hard maximum is 8*12=96.
    beam_member_pool = [
        _combination(
            f"BEAM_SINGLE_{family}",
            [best[family]],
            budget="BEAM_5_EPOCH",
        )
        for family in FAMILY_ORDER
    ]
    return with_content_hash(
        {
            "contract": STAGE_F_PLAN_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "single_family_selection_sha256": single_family_selection[
                "content_hash"
            ],
            "feedback_selection_sha256": feedback_selection["content_hash"],
            "family_order": list(FAMILY_ORDER),
            "best_member_by_family": best,
            "mandatory_combinations": mandatory,
            "pcgrad_control": pcgrad_control,
            "beam_member_pool": beam_member_pool,
            "beam_root": {
                "graph_id": "H_BASE_BEAM_BUDGET",
                "members": [],
                "budget": "BEAM_5_EPOCH",
                "fixed_epoch_budget": 5,
                "source_result_reused_for_omit": True,
            },
            "beam_width": 12,
            "beam_reduced_budget_fit_hard_maximum": 96,
            "full_fit_hard_maximum": 10,
            "full_control_fit_count": 1,
            "total_full_execution_hard_maximum": 11,
            "top_beam_winners_full_fit": 4,
            "baseline_proxy": "H_BASE_BEAM_BUDGET",
            "pcgrad_coordinate": {
                "fixed_graph_id": next(
                    row["graph_id"]
                    for row in mandatory
                    if row["combination_id"] == "C_ALL_BEST"
                ),
                "pcgrad_graph_id": pcgrad_control["graph_id"],
                "selection_eligible": False,
            },
            "weighting_coordinates": ["W_FIXED", "W_PCGRAD"],
            "performance_can_cancel_or_omit": False,
        }
    )


def expand_combination_beam(
    *,
    stage_f_plan: Mapping[str, Any],
    family: str,
    current_beam: Sequence[Mapping[str, Any]],
    completed_new_fit_count: int,
) -> dict[str, Any]:
    """Build one add/omit wave while training only the add coordinates."""

    validate_content_hash(stage_f_plan, expected_contract=STAGE_F_PLAN_CONTRACT)
    if family not in FAMILY_ORDER:
        raise ValueError("unknown beam family")
    expected_ordinal = FAMILY_ORDER.index(family)
    if not 0 <= int(completed_new_fit_count) <= 96:
        raise ValueError("beam completed-fit count lies outside the hard cap")
    member = dict(stage_f_plan["best_member_by_family"][family])
    roots = [dict(row) for row in current_beam]
    if not roots:
        roots = [dict(stage_f_plan["beam_root"])]
    if len(roots) > int(stage_f_plan["beam_width"]):
        raise ValueError("beam root width exceeds the registered bound")
    omit = []
    add_by_hash = {}
    for root in roots:
        omit.append(root)
        existing = [dict(row) for row in root.get("members", ())]
        if any(row["family"] == family for row in existing):
            continue
        candidate_hash = canonical_sha256(
            [root["graph_id"], family, member["target_id"]]
        )
        candidate = _combination(
            f"BEAM_W{expected_ordinal}_{candidate_hash[:12]}",
            [*existing, member],
            budget="BEAM_5_EPOCH",
        )
        add_by_hash.setdefault(candidate["graph_id"], candidate)
    additions = [
        add_by_hash[key] for key in sorted(add_by_hash)
    ]
    if int(completed_new_fit_count) + len(additions) > int(
        stage_f_plan["beam_reduced_budget_fit_hard_maximum"]
    ):
        raise RuntimeError("beam expansion would exceed 96 reduced-budget fits")
    all_candidates = {
        row["graph_id"]: row for row in [*omit, *additions]
    }
    return with_content_hash(
        {
            "contract": "hosd_combination_beam_expansion_v1",
            "schema_version": 1,
            "source": dict(stage_f_plan["source"]),
            "stage_f_plan_sha256": stage_f_plan["content_hash"],
            "family": family,
            "family_ordinal": expected_ordinal,
            "omit_reuse_graph_ids": sorted(row["graph_id"] for row in omit),
            "new_fit_candidates": additions,
            "new_fit_count": len(additions),
            "completed_new_fit_count_before": int(completed_new_fit_count),
            "completed_new_fit_count_after": (
                int(completed_new_fit_count) + len(additions)
            ),
            "all_candidates": [
                all_candidates[key] for key in sorted(all_candidates)
            ],
            "candidate_count": len(all_candidates),
            "new_fits_only_for_add_branch": True,
            "hard_maximum": int(
                stage_f_plan["beam_reduced_budget_fit_hard_maximum"]
            ),
        }
    )


def advance_combination_beam(
    *,
    stage_f_plan: Mapping[str, Any],
    family: str,
    expansion: Mapping[str, Any],
    reduced_budget_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_content_hash(stage_f_plan, expected_contract=STAGE_F_PLAN_CONTRACT)
    if family not in FAMILY_ORDER:
        raise ValueError("unknown beam family")
    if (
        expansion.get("contract") != "hosd_combination_beam_expansion_v1"
        or expansion.get("stage_f_plan_sha256") != stage_f_plan["content_hash"]
        or expansion.get("family") != family
    ):
        raise ValueError("beam expansion lineage differs")
    current_beam = expansion["all_candidates"]
    by_id = {row["graph_id"]: row for row in reduced_budget_results}
    if set(by_id) != {row["graph_id"] for row in current_beam}:
        raise ValueError("beam wave result coverage differs")
    ranked = []
    for candidate in current_beam:
        result = by_id[candidate["graph_id"]]
        validate_content_hash(result, expected_contract=COMBINATION_RESULT_CONTRACT)
        ranked.append(result)
    ranked.sort(
        key=lambda row: (
            -float(
                row["design_select"]["classification_metrics"][
                    "macro_per_class_accuracy"
                ]
            ),
            float(row["design_select"]["classification_metrics"]["cross_entropy"]),
            row["graph_id"],
        )
    )
    survivors = ranked[: int(stage_f_plan["beam_width"])]
    return with_content_hash(
        {
            "contract": "hosd_combination_beam_wave_v1",
            "schema_version": 1,
            "source": dict(stage_f_plan["source"]),
            "stage_f_plan_sha256": stage_f_plan["content_hash"],
            "family": family,
            "family_ordinal": FAMILY_ORDER.index(family),
            "expansion_sha256": expansion["content_hash"],
            "input_graph_ids": sorted(by_id),
            "surviving_graph_ids": [row["graph_id"] for row in survivors],
            "surviving_candidates": [
                next(
                    candidate
                    for candidate in current_beam
                    if candidate["graph_id"] == row["graph_id"]
                )
                for row in survivors
            ],
            "result_hashes": {
                key: by_id[key]["content_hash"] for key in sorted(by_id)
            },
            "width": int(stage_f_plan["beam_width"]),
            "completed_new_fit_count": int(
                expansion["completed_new_fit_count_after"]
            ),
            "reused_omit_result_count": len(
                expansion["omit_reuse_graph_ids"]
            ),
            "all_negative_rows_retained_by_rank": True,
        }
    )


def pcgrad_project(
    task_gradients: Sequence[Sequence[Any]],
    *,
    update_ordinal: int,
    epsilon: float = 1e-12,
) -> tuple[Any, ...]:
    """Project shared gradients in canonical order with cyclic rotation."""

    if torch is None or not task_gradients:
        raise ValueError("PCGrad requires at least one tensor gradient task")
    count = len(task_gradients)
    width = len(task_gradients[0])
    if width == 0 or any(len(task) != width for task in task_gradients):
        raise ValueError("PCGrad task gradient structures differ")
    rotation = int(update_ordinal) % count
    order = list(range(count))
    order = order[rotation:] + order[:rotation]
    projected = [
        [gradient.clone() for gradient in task_gradients[index]]
        for index in range(count)
    ]
    originals = [
        [gradient.detach().clone() for gradient in task]
        for task in task_gradients
    ]
    for index in order:
        for comparison in order:
            if index == comparison:
                continue
            dot = sum(
                (left * right).sum()
                for left, right in zip(projected[index], originals[comparison])
            )
            if bool(dot < 0):
                norm = sum(
                    right.square().sum() for right in originals[comparison]
                ).clamp_min(float(epsilon))
                projected[index] = [
                    left - dot / norm * right
                    for left, right in zip(
                        projected[index], originals[comparison]
                    )
                ]
    return tuple(
        sum(projected[task][parameter] for task in range(count)) / count
        for parameter in range(width)
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x, y = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(
        right, dtype=np.float64
    ).reshape(-1)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("correlation arrays differ or are too short")
    x, y = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return 0.0 if denominator == 0 else float((x @ y) / denominator)


def build_gradient_conflict_report(
    *,
    identities: Sequence[str],
    residuals_by_family: Mapping[str, np.ndarray],
    target_errors_by_family: Mapping[str, np.ndarray],
    gradient_cosines: Mapping[str, Sequence[float]],
    representations_by_family: Mapping[str, np.ndarray],
    leave_one_out_accuracy_change: Mapping[str, float],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if len(identities) != len(set(identities)):
        raise ValueError("redundancy report identities are not unique")
    families = sorted(residuals_by_family)
    if set(families) != set(target_errors_by_family) or set(families) != set(
        representations_by_family
    ):
        raise ValueError("redundancy family coverage differs")
    pairs = {}
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            key = f"{left}__{right}"
            pairs[key] = {
                "residual_pearson": _pearson(
                    residuals_by_family[left], residuals_by_family[right]
                ),
                "residual_spearman": _pearson(
                    _average_ranks(np.asarray(residuals_by_family[left]).reshape(-1)),
                    _average_ranks(np.asarray(residuals_by_family[right]).reshape(-1)),
                ),
                "target_error_correlation": _pearson(
                    target_errors_by_family[left],
                    target_errors_by_family[right],
                ),
                "representation_linear_cka": float(
                    linear_cka(
                        np.asarray(representations_by_family[left]),
                        np.asarray(representations_by_family[right]),
                    )
                ),
            }
    gradients = {}
    for key, values in sorted(gradient_cosines.items()):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("gradient cosine series differs")
        gradients[key] = {
            "mean_cosine": float(array.mean()),
            "negative_fraction": float((array < 0).mean()),
            "min_cosine": float(array.min()),
            "max_cosine": float(array.max()),
        }
    return with_content_hash(
        {
            "contract": GRADIENT_CONFLICT_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "identity_count": len(identities),
            "identity_order_sha256": canonical_sha256(list(identities)),
            "pair_diagnostics": pairs,
            "gradient_diagnostics": gradients,
            "leave_one_family_out_accuracy_change": {
                key: float(value)
                for key, value in sorted(leave_one_out_accuracy_change.items())
            },
            "selection_effect": "report_only",
        }
    )


def promote_combination_beam_winners(
    *,
    stage_f_plan: Mapping[str, Any],
    final_wave: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the top reduced-budget beam rows as distinct full-fit graphs."""

    validate_content_hash(stage_f_plan, expected_contract=STAGE_F_PLAN_CONTRACT)
    validate_content_hash(
        final_wave, expected_contract="hosd_combination_beam_wave_v1"
    )
    if (
        final_wave.get("stage_f_plan_sha256") != stage_f_plan["content_hash"]
        or final_wave.get("family") != FAMILY_ORDER[-1]
        or int(final_wave.get("family_ordinal", -1)) != len(FAMILY_ORDER) - 1
    ):
        raise ValueError("beam promotion requires the complete final wave")
    survivors = list(final_wave["surviving_candidates"])
    excluded_baselines = [
        candidate for candidate in survivors if not candidate.get("members")
    ]
    if any(
        candidate.get("graph_id") != "H_BASE_BEAM_BUDGET"
        for candidate in excluded_baselines
    ) or len(excluded_baselines) > 1:
        raise ValueError("beam contains an unknown empty-member candidate")
    # H_BASE_BEAM_BUDGET is the separately trained reduced-budget reference.
    # It participates in every add/omit ranking wave, but it is not a
    # combination and therefore cannot consume one of the four full-fit
    # combination promotion slots.
    eligible = [candidate for candidate in survivors if candidate.get("members")]
    reduced = eligible[: int(stage_f_plan["top_beam_winners_full_fit"])]
    promoted = []
    for rank, candidate in enumerate(reduced, 1):
        graph = _combination(
            f"C_BEAM_WINNER_{rank}",
            candidate["members"],
            budget="FULL",
        )
        promoted.append(
            {
                **graph,
                "beam_rank": rank,
                "reduced_budget_graph_id": candidate["graph_id"],
            }
        )
    if len(promoted) != int(stage_f_plan["top_beam_winners_full_fit"]):
        raise ValueError("beam promotion winner coverage differs")
    return with_content_hash(
        {
            "contract": COMBINATION_BEAM_PROMOTION_CONTRACT,
            "schema_version": 2,
            "source": dict(stage_f_plan["source"]),
            "stage_f_plan_sha256": stage_f_plan["content_hash"],
            "final_wave_sha256": final_wave["content_hash"],
            "promoted_graphs": promoted,
            "promoted_graph_ids": [row["graph_id"] for row in promoted],
            "reduced_budget_graph_ids": [
                row["reduced_budget_graph_id"] for row in promoted
            ],
            "excluded_baseline_graph_ids": [
                str(row["graph_id"]) for row in excluded_baselines
            ],
            "baseline_exclusion_rule": (
                "zero_member_H_BASE_BEAM_BUDGET_is_ranked_but_not_promoted_v1"
            ),
            "winner_count": len(promoted),
            "full_fit_required": True,
            "performance_can_omit_or_cancel": False,
        }
    )


def build_combination_beam_completion(
    *,
    stage_f_plan: Mapping[str, Any],
    expansions: Sequence[Mapping[str, Any]],
    waves: Sequence[Mapping[str, Any]],
    reduced_budget_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attest all eight bounded beam waves and their complete fit coverage."""

    validate_content_hash(stage_f_plan, expected_contract=STAGE_F_PLAN_CONTRACT)
    if len(expansions) != len(FAMILY_ORDER) or len(waves) != len(FAMILY_ORDER):
        raise ValueError("combination beam family-wave coverage differs")
    expected_results = {"H_BASE_BEAM_BUDGET"}
    previous_wave = None
    for ordinal, family in enumerate(FAMILY_ORDER):
        expansion = expansions[ordinal]
        wave = waves[ordinal]
        validate_content_hash(
            expansion, expected_contract="hosd_combination_beam_expansion_v1"
        )
        validate_content_hash(
            wave, expected_contract="hosd_combination_beam_wave_v1"
        )
        expected_roots = (
            ["H_BASE_BEAM_BUDGET"]
            if previous_wave is None
            else sorted(previous_wave["surviving_graph_ids"])
        )
        if (
            expansion.get("stage_f_plan_sha256") != stage_f_plan["content_hash"]
            or expansion.get("family") != family
            or int(expansion.get("family_ordinal", -1)) != ordinal
            or sorted(expansion.get("omit_reuse_graph_ids", ())) != expected_roots
            or wave.get("expansion_sha256") != expansion["content_hash"]
            or wave.get("family") != family
        ):
            raise ValueError("combination beam wave lineage differs")
        expected_results.update(
            row["graph_id"] for row in expansion["new_fit_candidates"]
        )
        previous_wave = wave
    by_id = {str(row["graph_id"]): row for row in reduced_budget_results}
    if len(by_id) != len(reduced_budget_results) or set(by_id) != expected_results:
        raise ValueError("combination reduced-budget result coverage differs")
    for result in by_id.values():
        validate_content_hash(result, expected_contract=COMBINATION_RESULT_CONTRACT)
        if result.get("source") != stage_f_plan["source"]:
            raise ValueError("combination reduced-budget source differs")
    total = sum(int(row["new_fit_count"]) for row in expansions)
    if (
        total != int(waves[-1]["completed_new_fit_count"])
        or total > int(stage_f_plan["beam_reduced_budget_fit_hard_maximum"])
    ):
        raise ValueError("combination reduced-budget fit count differs")
    promotion = promote_combination_beam_winners(
        stage_f_plan=stage_f_plan, final_wave=waves[-1]
    )
    return with_content_hash(
        {
            "contract": COMBINATION_BEAM_COMPLETION_CONTRACT,
            "schema_version": 2,
            "source": dict(stage_f_plan["source"]),
            "stage_f_plan_sha256": stage_f_plan["content_hash"],
            "family_order": list(FAMILY_ORDER),
            "expansion_hashes": [row["content_hash"] for row in expansions],
            "wave_hashes": [row["content_hash"] for row in waves],
            "reduced_budget_result_hashes": {
                key: by_id[key]["content_hash"] for key in sorted(by_id)
            },
            "new_fit_count": total,
            "hard_maximum": int(
                stage_f_plan["beam_reduced_budget_fit_hard_maximum"]
            ),
            "promotion": promotion,
            "promotion_sha256": promotion["content_hash"],
            "all_families_completed": True,
            "all_negative_rows_retained_by_rank": True,
            "performance_based_termination": False,
        }
    )


def build_combination_wave_completion(
    *,
    stage_f_plan: Mapping[str, Any],
    wave_kind: str,
    results: Sequence[Mapping[str, Any]],
    beam_completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attest exact full-budget or PCGrad result coverage."""

    validate_content_hash(stage_f_plan, expected_contract=STAGE_F_PLAN_CONTRACT)
    if wave_kind == "FULL":
        if beam_completion is None:
            raise ValueError("full combination wave requires beam completion")
        validate_content_hash(
            beam_completion,
            expected_contract=COMBINATION_BEAM_COMPLETION_CONTRACT,
        )
        if (
            beam_completion.get("stage_f_plan_sha256")
            != stage_f_plan["content_hash"]
        ):
            raise ValueError("full combination beam lineage differs")
        expected = {
            row["graph_id"] for row in stage_f_plan["mandatory_combinations"]
        } | set(beam_completion["promotion"]["promoted_graph_ids"])
        hard_maximum = int(stage_f_plan["full_fit_hard_maximum"])
    elif wave_kind == "PCGRAD":
        if beam_completion is not None:
            raise ValueError("PCGrad completion cannot consume beam promotion")
        expected = {stage_f_plan["pcgrad_control"]["graph_id"]}
        hard_maximum = int(stage_f_plan["full_control_fit_count"])
    else:
        raise ValueError("unknown combination completion wave")
    by_id = {str(row["graph_id"]): row for row in results}
    if (
        len(by_id) != len(results)
        or set(by_id) != expected
        or len(by_id) > hard_maximum
    ):
        raise ValueError("combination wave result coverage differs")
    for result in by_id.values():
        validate_content_hash(result, expected_contract=COMBINATION_RESULT_CONTRACT)
        if (
            result.get("source") != stage_f_plan["source"]
            or result.get("stage_f_plan_sha256")
            != stage_f_plan["content_hash"]
        ):
            raise ValueError("combination wave result lineage differs")
    return with_content_hash(
        {
            "contract": COMBINATION_WAVE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "source": dict(stage_f_plan["source"]),
            "stage_f_plan_sha256": stage_f_plan["content_hash"],
            "wave_kind": wave_kind,
            "result_hashes": {
                key: by_id[key]["content_hash"] for key in sorted(by_id)
            },
            "graph_ids": sorted(by_id),
            "result_count": len(by_id),
            "hard_maximum": hard_maximum,
            "beam_completion_sha256": (
                None
                if beam_completion is None
                else beam_completion["content_hash"]
            ),
            "all_registered_rows_complete": True,
            "performance_based_termination": False,
        }
    )


def build_combination_selection(
    *,
    stage_f_plan: Mapping[str, Any],
    full_results: Sequence[Mapping[str, Any]],
    beam_winner_graph_ids: Sequence[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(stage_f_plan, expected_contract=STAGE_F_PLAN_CONTRACT)
    if len(beam_winner_graph_ids) > 4:
        raise ValueError("too many full-budget beam winners")
    results = {row["graph_id"]: row for row in full_results}
    expected = {
        row["graph_id"] for row in stage_f_plan["mandatory_combinations"]
    } | set(beam_winner_graph_ids)
    if set(results) != expected or len(results) > 10:
        raise ValueError("full combination result coverage differs")
    for result in results.values():
        validate_content_hash(result, expected_contract=COMBINATION_RESULT_CONTRACT)
    ranked = [
        {
            **row,
            "row_id": row["graph_id"],
            "target_id": row["combination_id"],
        }
        for row in results.values()
    ]
    winner = select_utility_row(ranked)
    return with_content_hash(
        {
            "contract": COMBINATION_SELECTION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "stage_f_plan_sha256": stage_f_plan["content_hash"],
            "result_hashes": {
                key: results[key]["content_hash"] for key in sorted(results)
            },
            "beam_winner_graph_ids": list(beam_winner_graph_ids),
            "selected_combination_graph_id": winner["graph_id"],
            "selected_combination_id": winner["combination_id"],
            "selected_graph_definition": {
                key: winner[key]
                for key in (
                    "graph_id",
                    "combination_id",
                    "members",
                    "normalized_weights",
                    "budget",
                    "weighting",
                )
            }
            | (
                {
                    "native_relation_auxiliary": dict(
                        winner["native_relation_auxiliary"]
                    )
                }
                if winner.get("native_relation_auxiliary") is not None
                else {}
            ),
            "selection_split": "design_select",
            "negative_gain_can_still_win": True,
        }
    )


def build_mechanism_control_plan(
    *,
    combination_selection: Mapping[str, Any],
    selected_combination: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        combination_selection, expected_contract=COMBINATION_SELECTION_CONTRACT
    )
    if (
        selected_combination["graph_id"]
        != combination_selection["selected_combination_graph_id"]
    ):
        raise ValueError("mechanism graph differs from locked combination")
    interventions = [
        *[
            {
                "intervention_id": f"LEAVE_ONE_OUT_{member['family']}",
                "kind": "leave_one_family_out",
                "family": member["family"],
            }
            for member in selected_combination["members"]
        ],
        {"intervention_id": "REMOVE_AUX_HEADS", "kind": "inference_head_removal"},
        {"intervention_id": "ZERO_FEEDBACK", "kind": "zero_feedback"},
        {"intervention_id": "SHUFFLE_FEEDBACK", "kind": "wrong_event_prediction"},
        {"intervention_id": "TARGET_MEAN", "kind": "target_mean"},
        {"intervention_id": "UNRESTRICTED", "kind": "matched_capacity"},
        {"intervention_id": "ABS_RES_HET", "kind": "parameterization"},
        {"intervention_id": "ERROR_GAIN_TRACKING", "kind": "eventwise_correlation"},
    ]
    executable = []
    for row in interventions:
        definition = dict(row)
        if row["kind"] == "leave_one_family_out":
            members = [
                member
                for member in selected_combination["members"]
                if member["family"] != row["family"]
            ]
            if not members:
                definition["execution"] = {
                    "worker": "emit_registered_not_applicable",
                    "reason": "leave_one_out_would_remove_all_auxiliaries",
                }
            else:
                graph = _combination(
                    row["intervention_id"],
                    members,
                    budget="FULL",
                    weighting=str(selected_combination["weighting"]),
                )
                graph["selection_eligible"] = False
                definition["execution"] = {
                    "worker": "train_hosd_combination",
                    "graph": graph,
                    "split": "design_confirm",
                    "retrain_from_common_initialization": True,
                }
        elif row["kind"] == "inference_head_removal":
            definition["execution"] = {
                "worker": "evaluate_hosd_mechanism",
                "mode": "head_removal_parity",
                "checkpoint_graph_id": selected_combination["graph_id"],
                "exact_logits_required": True,
            }
        elif row["kind"] in {
            "zero_feedback",
            "wrong_event_prediction",
            "target_mean",
            "matched_capacity",
            "parameterization",
        }:
            definition["execution"] = {
                "worker": "evaluate_hosd_mechanism",
                "mode": "authenticated_registered_control_join",
                "required_control_kind": row["kind"],
                "retrain_if_registered_control_missing": True,
            }
        else:
            definition["execution"] = {
                "worker": "evaluate_hosd_mechanism",
                "mode": "identity_paired_eventwise_error_gain",
                "split": "design_confirm",
            }
        executable.append(definition)
    return with_content_hash(
        {
            "contract": MECHANISM_CONTROL_PLAN_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "combination_selection_sha256": combination_selection["content_hash"],
            "selected_graph_id": selected_combination["graph_id"],
            "evaluation_split": "design_confirm",
            "selection_eligible": False,
            "interventions": executable,
            "intervention_count": len(executable),
            "can_reopen_selection": False,
        }
    )


def build_mechanism_result(
    *,
    plan: Mapping[str, Any],
    intervention_id: str,
    status: str,
    measurements: Mapping[str, Any],
    evidence_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(plan, expected_contract=MECHANISM_CONTROL_PLAN_CONTRACT)
    rows = {
        str(row["intervention_id"]): row for row in plan["interventions"]
    }
    if intervention_id not in rows:
        raise ValueError("mechanism intervention is absent from the locked plan")
    if status not in {"completed", "not_applicable"}:
        raise ValueError("mechanism status must be completed/not_applicable")
    if status == "completed" and not measurements:
        raise ValueError("completed mechanism result has no measurements")
    if not evidence_hashes:
        raise ValueError("mechanism result has no authenticated evidence")
    return with_content_hash(
        {
            "contract": MECHANISM_RESULT_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "mechanism_plan_sha256": plan["content_hash"],
            "intervention_id": intervention_id,
            "intervention": rows[intervention_id],
            "status": status,
            "measurements": dict(measurements),
            "evidence_hashes": {
                key: require_sha256(value, name=f"evidence.{key}")
                for key, value in sorted(evidence_hashes.items())
            },
            "evaluation_split": "design_confirm",
            "selection_eligible": False,
        }
    )


def build_mechanism_summary(
    *,
    plan: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        plan, expected_contract=MECHANISM_CONTROL_PLAN_CONTRACT
    )
    by_id = {row["intervention_id"]: row for row in results}
    expected = {row["intervention_id"] for row in plan["interventions"]}
    if set(by_id) != expected:
        raise ValueError("mechanism result coverage differs")
    for row in by_id.values():
        validate_content_hash(row, expected_contract=MECHANISM_RESULT_CONTRACT)
        if (
            row.get("source") != dict(source)
            or row.get("mechanism_plan_sha256") != plan["content_hash"]
            or row.get("evaluation_split") != "design_confirm"
            or bool(row.get("selection_eligible"))
            or row.get("status") not in {"completed", "not_applicable"}
        ):
            raise ValueError("mechanism result lineage differs")
    return with_content_hash(
        {
            "contract": MECHANISM_SUMMARY_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "mechanism_plan_sha256": plan["content_hash"],
            "evaluation_split": "design_confirm",
            "results": [by_id[key] for key in sorted(by_id)],
            "result_hashes": {
                key: by_id[key]["content_hash"] for key in sorted(by_id)
            },
            "all_interventions_complete": True,
            "selection_reopened": False,
            "negative_or_null_mechanism_results_reported": True,
        }
    )


__all__ = [
    "FAMILY_ORDER",
    "MANDATORY_COMBINATIONS",
    "advance_combination_beam",
    "build_combination_beam_completion",
    "build_combination_wave_completion",
    "expand_combination_beam",
    "promote_combination_beam_winners",
    "build_combination_selection",
    "build_gradient_conflict_report",
    "build_mechanism_control_plan",
    "build_mechanism_result",
    "build_mechanism_summary",
    "build_stage_f_plan",
    "normalize_combination_weights",
    "pcgrad_project",
    "target_family",
]
