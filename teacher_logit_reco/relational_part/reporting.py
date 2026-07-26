"""Role-aware JSON and Markdown reporting for the sealed relational campaign."""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .evaluation import (
    EVALUATION_CONTRACT,
    FINAL_EVALUATION_CONTRACT,
    PAIRED_STATISTICS_CONTRACT,
)
from .selection import (
    CONFIRMATION_SUMMARY_CONTRACT,
    LOCKED_FINALISTS_CONTRACT,
    validate_locked_finalists,
)


REPORT_CONTRACT = "relational_part_report_v2"


def build_relational_part_report(
    *,
    locked_finalists: Mapping[str, Any],
    confirmation_summary: Mapping[str, Any],
    final_evaluations: Sequence[Mapping[str, Any]],
    paired_statistics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    lock_sha = validate_locked_finalists(
        locked_finalists,
        campaign_spec_sha256=locked_finalists["campaign_spec_sha256"],
        split_manifest_sha256=locked_finalists["split_manifest_sha256"],
        hlt_cache_hashes=locked_finalists["hlt_cache_hashes"],
    )
    confirmation_sha = validate_content_hash(
        confirmation_summary, expected_contract=CONFIRMATION_SUMMARY_CONTRACT
    )
    if locked_finalists.get("confirmation_summary_sha256") != confirmation_sha:
        raise ValueError("report lock and confirmation summary disagree")
    expected_rows = {
        str(row["run_id"]): row for row in locked_finalists["evaluation_rows"]
    }
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for evaluation in final_evaluations:
        validate_content_hash(
            evaluation, expected_contract=FINAL_EVALUATION_CONTRACT
        )
        if evaluation.get("locked_finalists_sha256") != lock_sha:
            raise ValueError("final evaluation belongs to another finalist lock")
        if evaluation.get("campaign_spec_sha256") != locked_finalists[
            "campaign_spec_sha256"
        ]:
            raise ValueError("final evaluation belongs to another campaign")
        if evaluation.get("split_manifest_sha256") != locked_finalists[
            "split_manifest_sha256"
        ]:
            raise ValueError("final evaluation belongs to another split")
        if evaluation.get("final_test_hlt_cache_sha256") != locked_finalists[
            "hlt_cache_hashes"
        ]["final_test"]:
            raise ValueError("final evaluation belongs to another final HLT cache")
        if evaluation.get("final_test_used_for_selection") is not False:
            raise ValueError("final evaluation is marked as selection input")
        validate_content_hash(
            evaluation.get("metrics", {}),
            expected_contract=EVALUATION_CONTRACT,
        )
        if evaluation["metrics"].get("split") != "final_test":
            raise ValueError("report metric split is not final_test")
        if int(evaluation.get("event_count", -1)) != int(
            evaluation["metrics"].get("event_count", -2)
        ):
            raise ValueError("final evaluation event count drifted")
        profile = evaluation.get("parameter_and_flop_profile")
        if not isinstance(profile, Mapping) or not {
            "trainable_parameters",
            "forward_flops_per_event",
            "latency_ms",
            "peak_incremental_device_memory_bytes",
        }.issubset(profile):
            raise ValueError("final evaluation resource profile is incomplete")
        run_id = str(evaluation["run_id"])
        seed = int(evaluation["seed"])
        if run_id not in expected_rows or seed not in (101, 202, 303):
            raise ValueError("final evaluation is absent from the lock")
        if evaluation["checkpoint_sha256"] != expected_rows[run_id][
            "checkpoint_hashes"
        ][str(seed)]:
            raise ValueError("report final evaluation checkpoint mismatch")
        if evaluation.get(
            "checkpoint_registration_sha256"
        ) != expected_rows[run_id]["checkpoint_registration_hashes"][str(seed)]:
            raise ValueError("report checkpoint registration mismatch")
        if evaluation.get("model_contract_sha256") != expected_rows[run_id][
            "model_contract_sha256"
        ]:
            raise ValueError("report model contract mismatch")
        if (
            evaluation.get("lineage_authenticated") is not True
            or dict(evaluation.get("checkpoint_lineage_hashes", {}))
            != dict(expected_rows[run_id]["lineage_hashes"])
        ):
            raise ValueError("report final lineage authentication failed")
        grouped.setdefault(run_id, {})[seed] = evaluation
    if set(grouped) != set(expected_rows) or any(
        set(rows) != {101, 202, 303} for rows in grouped.values()
    ):
        raise ValueError("report requires every locked run at all three seeds")
    baseline = grouped["RPT_BASE"]
    rows = []
    for run_id, locked_row in expected_rows.items():
        accuracy = [
            float(grouped[run_id][seed]["metrics"]["accuracy"])
            for seed in (101, 202, 303)
        ]
        deltas = [
            accuracy[index]
            - float(baseline[seed]["metrics"]["accuracy"])
            for index, seed in enumerate((101, 202, 303))
        ]
        statistics_by_seed = paired_statistics.get(run_id, {})
        if run_id != "RPT_BASE":
            if set(map(int, statistics_by_seed)) != {101, 202, 303}:
                raise ValueError(f"{run_id} lacks paired statistics at every seed")
            for seed_key, value in statistics_by_seed.items():
                validate_content_hash(
                    value, expected_contract=PAIRED_STATISTICS_CONTRACT
                )
                seed = int(seed_key)
                if (
                    value.get("candidate_run_id") != run_id
                    or value.get("baseline_run_id") != "RPT_BASE"
                    or int(value.get("seed", -1)) != seed
                ):
                    raise ValueError("paired statistic run/seed binding drifted")
                candidate_accuracy = float(
                    grouped[run_id][seed]["metrics"]["accuracy"]
                )
                baseline_accuracy = float(
                    baseline[seed]["metrics"]["accuracy"]
                )
                if (
                    float(value["candidate_accuracy"]) != candidate_accuracy
                    or float(value["baseline_accuracy"]) != baseline_accuracy
                    or float(value["paired_absolute_accuracy_difference"])
                    != candidate_accuracy - baseline_accuracy
                ):
                    raise ValueError("paired statistic accuracy values drifted")
        rows.append(
            {
                "run_id": run_id,
                "configuration_role": locked_row["configuration_role"],
                "relational_selection_eligible": locked_row[
                    "relational_selection_eligible"
                ],
                "mean_final_test_accuracy": float(statistics.fmean(accuracy)),
                "final_test_accuracy_sample_standard_deviation": float(
                    statistics.stdev(accuracy)
                ),
                "per_seed_final_test_accuracy": {
                    str(seed): accuracy[index]
                    for index, seed in enumerate((101, 202, 303))
                },
                "per_seed_complete_metrics": {
                    str(seed): dict(grouped[run_id][seed]["metrics"])
                    for seed in (101, 202, 303)
                },
                "per_seed_resource_profiles": {
                    str(seed): dict(
                        grouped[run_id][seed]["parameter_and_flop_profile"]
                    )
                    for seed in (101, 202, 303)
                },
                "per_seed_matched_baseline_difference": {
                    str(seed): deltas[index]
                    for index, seed in enumerate((101, 202, 303))
                },
                "mean_final_test_matched_baseline_difference": float(
                    statistics.fmean(deltas)
                ),
                "paired_statistics": {
                    str(seed): dict(statistics_by_seed[str(seed)])
                    if str(seed) in statistics_by_seed
                    else dict(statistics_by_seed[seed])
                    for seed in (101, 202, 303)
                }
                if run_id != "RPT_BASE"
                else {},
                "paired_seed_summary": (
                    {
                        "mean_paired_absolute_accuracy_difference": float(
                            statistics.fmean(
                                float(value[
                                    "paired_absolute_accuracy_difference"
                                ])
                                for value in statistics_by_seed.values()
                            )
                        ),
                        "sample_standard_deviation": float(
                            statistics.stdev(
                                float(value[
                                    "paired_absolute_accuracy_difference"
                                ])
                                for value in statistics_by_seed.values()
                            )
                        ),
                    }
                    if run_id != "RPT_BASE"
                    else {}
                ),
            }
        )
    winner_id = str(locked_finalists["nominal_relational_winner_id"])
    winner = next(row for row in rows if row["run_id"] == winner_id)
    confirmation_gain = bool(
        locked_finalists["confirmation_gain_positive"]
    )
    capacity_reproduces = bool(
        locked_finalists["capacity_control_reproduces_gain"]
    )
    final_positive = (
        winner["mean_final_test_matched_baseline_difference"] > 0.0
    )
    hlt_only = all(
        evaluation.get("hlt_only_inference") is True
        for evaluation in final_evaluations
    )
    lineage_authenticated = all(
        evaluation.get("lineage_authenticated") is True
        and evaluation.get("checkpoint_registration_sha256")
        == expected_rows[str(evaluation["run_id"])][
            "checkpoint_registration_hashes"
        ][str(int(evaluation["seed"]))]
        for evaluation in final_evaluations
    )
    success = (
        confirmation_gain
        and not capacity_reproduces
        and final_positive
        and hlt_only
        and lineage_authenticated
    )
    confirmation_winner = next(
        row
        for row in confirmation_summary["rows"]
        if row["run_id"] == winner_id
    )
    strong = (
        success
        and confirmation_winner["mean_matched_seed_accuracy_difference"] >= 0.003
        and confirmation_winner["seeds_beating_matched_baseline"] == 3
    )
    return with_content_hash(
        {
            "contract": REPORT_CONTRACT,
            "schema_version": 2,
            "locked_finalists_sha256": lock_sha,
            "confirmation_summary_sha256": confirmation_sha,
            "campaign_spec_sha256": require_sha256(
                locked_finalists["campaign_spec_sha256"],
                name="campaign_spec_sha256",
            ),
            "baseline_id": "RPT_BASE",
            "nominal_relational_winner_id": winner_id,
            "confirmation_gain_positive": confirmation_gain,
            "capacity_control_reproduces_gain": capacity_reproduces,
            "winner_final_test_gain_positive": final_positive,
            "all_final_inference_hlt_only": hlt_only,
            "all_final_lineage_authenticated": lineage_authenticated,
            "positive_architecture_result": success,
            "particularly_strong_result": strong,
            "fully_negative_campaign_completed_validly": not confirmation_gain,
            "rows": rows,
            "claims": {
                "selection_used_final_test": False,
                "final_test_reporting_only": True,
                "compound_architecture_results_labelled": True,
                "validation_and_seed_uncertainty_both_reported": True,
            },
        }
    )


def render_relational_part_markdown(report: Mapping[str, Any]) -> str:
    validate_content_hash(report, expected_contract=REPORT_CONTRACT)
    lines = [
        "# Relational Particle Transformer report",
        "",
        f"- Baseline: `{report['baseline_id']}`",
        f"- Nominal relational winner: `{report['nominal_relational_winner_id']}`",
        f"- Positive validation gain: `{str(report['confirmation_gain_positive']).lower()}`",
        f"- Capacity control reproduces gain: `{str(report['capacity_control_reproduces_gain']).lower()}`",
        f"- Positive sealed-test gain: `{str(report['winner_final_test_gain_positive']).lower()}`",
        f"- Positive architecture result: `{str(report['positive_architecture_result']).lower()}`",
        "",
        "Final-test results were used for reporting only, never model selection.",
        "",
        "| Run | Role | Mean accuracy | Mean matched Δ | Seed spread |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| `{}` | {} | {:.6f} | {:+.6f} | {:.6f} |".format(
                row["run_id"],
                row["configuration_role"],
                row["mean_final_test_accuracy"],
                row["mean_final_test_matched_baseline_difference"],
                row["final_test_accuracy_sample_standard_deviation"],
            )
        )
    lines.extend(
        [
            "",
            "Inference dependencies were authenticated as HLT-only for every row.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "REPORT_CONTRACT",
    "build_relational_part_report",
    "render_relational_part_markdown",
]
