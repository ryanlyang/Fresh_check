#!/usr/bin/env python3
"""Aggregate the fixed three-seed TRACK+CHARGE+DENSITY fusion diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


CONTRACT = "relational_part_posthoc_fusion_three_seed_summary_v1"
SEED_ARTIFACT_CONTRACT = "relational_part_posthoc_fusion_comparison_v2"
COMPARISON_ID = "track_charge_density"
EXPECTED_SEEDS = (101, 202, 303)
CONTROL_RUN_IDS = (
    "RPT_BASE",
    "RPT_BASE_WIDE_MAX",
    "RPT_FULL_ZERO_REL",
)
RELATION_RUN_IDS = (
    "RPT_TRACK",
    "RPT_CHARGE",
    "RPT_DENSITY",
)


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(float(value) for value in values))


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accuracies = [float(row["accuracy"]) for row in rows]
    return {
        "mean_accuracy": _mean(accuracies),
        "median_accuracy": float(statistics.median(accuracies)),
        "accuracy_sample_standard_deviation": float(statistics.stdev(accuracies)),
        "mean_cross_entropy": _mean(
            [float(row["cross_entropy"]) for row in rows]
        ),
        "mean_brier_score": _mean(
            [float(row["brier_score"]) for row in rows]
        ),
        "mean_ece_15_bin_top_label": _mean(
            [float(row["ece_15_bin_top_label"]["value"]) for row in rows]
        ),
    }


def _pooled_rejection_at_75(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    class_names = tuple(str(value) for value in rows[0])
    if any(tuple(str(value) for value in row) != class_names for row in rows):
        raise ValueError("seed artifacts have different rejection class ordering")
    result = {}
    for class_name in class_names:
        cells = [row[class_name] for row in rows]
        false_positives = sum(
            int(cell["qcd_false_positive_count"]) for cell in cells
        )
        qcd_support = sum(int(cell["qcd_support"]) for cell in cells)
        result[class_name] = {
            "target_signal_efficiency": 0.75,
            "pooled_qcd_false_positive_count": false_positives,
            "pooled_qcd_support": qcd_support,
            "pooled_qcd_false_positive_rate": (
                float(false_positives / qcd_support)
            ),
            "pooled_background_rejection": (
                None
                if false_positives == 0
                else float(qcd_support / false_positives)
            ),
            "pooled_background_rejection_is_infinite": false_positives == 0,
            "pooling_rule": (
                "sum_qcd_false_positives_and_support_across_seed_specific_"
                "thresholds"
            ),
        }
    return result


def aggregate_seed_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(artifacts) != len(EXPECTED_SEEDS):
        raise ValueError("exactly three seed artifacts are required")
    by_seed = {int(row["seed"]): row for row in artifacts}
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS or len(by_seed) != len(artifacts):
        raise ValueError("fusion artifacts must cover seeds 101, 202, and 303 once")
    ordered = [by_seed[seed] for seed in EXPECTED_SEEDS]
    reference = ordered[0]
    invariant_names = (
        "campaign_spec_sha256",
        "global_determinism_sha256",
        "hlt_cache_hashes",
        "event_identity_hashes",
        "class_order",
        "control_member_run_ids",
        "relation_member_run_ids",
        "calibration_split",
        "evaluation_split",
    )
    for artifact in ordered:
        if artifact.get("contract") != SEED_ARTIFACT_CONTRACT:
            raise ValueError("seed fusion artifact contract differs")
        if artifact.get("comparison_id") != COMPARISON_ID:
            raise ValueError("seed fusion comparison differs")
        if artifact.get("official_campaign_metric") is not False:
            raise ValueError("supplemental fusion was marked as official")
        if artifact.get("eligible_for_model_selection") is not False:
            raise ValueError("supplemental fusion was marked selection-eligible")
        if artifact.get("final_test_opened") is not False:
            raise ValueError("supplemental fusion opened the final test")
        if tuple(artifact["control_member_run_ids"]) != CONTROL_RUN_IDS:
            raise ValueError("control fusion members differ")
        if tuple(artifact["relation_member_run_ids"]) != RELATION_RUN_IDS:
            raise ValueError("relation fusion members differ")
        for name in invariant_names:
            if artifact[name] != reference[name]:
                raise ValueError(f"seed fusion invariant differs: {name}")

    control_metrics = [
        row["fusion_results"]["CONTROL_LOGIT_FUSION"]["primary_metrics"]
        for row in ordered
    ]
    relation_metrics = [
        row["fusion_results"]["RELATION_LOGIT_FUSION"]["primary_metrics"]
        for row in ordered
    ]
    paired = [
        row["primary_relation_minus_control_paired_statistics"] for row in ordered
    ]
    differences = [
        float(row["paired_absolute_accuracy_difference"]) for row in paired
    ]
    per_class_names = tuple(
        str(value) for value in paired[0]["per_class_paired_accuracy_difference"]
    )
    if any(
        tuple(str(value) for value in row["per_class_paired_accuracy_difference"])
        != per_class_names
        for row in paired
    ):
        raise ValueError("paired per-class result ordering differs")

    return {
        "comparison_id": COMPARISON_ID,
        "seed_order": list(EXPECTED_SEEDS),
        "control_member_run_ids": list(CONTROL_RUN_IDS),
        "relation_member_run_ids": list(RELATION_RUN_IDS),
        "calibration_split": "val_stop",
        "evaluation_split": "val_select",
        "final_test_opened": False,
        "scientific_status": "supplemental_post_hoc_diagnostic_only",
        "official_campaign_metric": False,
        "eligible_for_model_selection": False,
        "control_fusion_three_seed_metrics": _metric_summary(control_metrics),
        "relation_fusion_three_seed_metrics": _metric_summary(relation_metrics),
        "relation_minus_control_accuracy": {
            "per_seed": {
                str(seed): float(value)
                for seed, value in zip(EXPECTED_SEEDS, differences, strict=True)
            },
            "mean": _mean(differences),
            "median": float(statistics.median(differences)),
            "sample_standard_deviation": float(statistics.stdev(differences)),
            "seeds_relation_fusion_beats_control_fusion": sum(
                value > 0.0 for value in differences
            ),
        },
        "mean_per_class_paired_accuracy_difference": {
            class_name: _mean(
                [
                    float(row["per_class_paired_accuracy_difference"][class_name])
                    for row in paired
                ]
            )
            for class_name in per_class_names
        },
        "control_fusion_pooled_qcd_rejection_at_0p75": _pooled_rejection_at_75(
            [
                row["fusion_results"]["CONTROL_LOGIT_FUSION"][
                    "primary_qcd_signal_rejection_at_0p75"
                ]
                for row in ordered
            ]
        ),
        "relation_fusion_pooled_qcd_rejection_at_0p75": _pooled_rejection_at_75(
            [
                row["fusion_results"]["RELATION_LOGIT_FUSION"][
                    "primary_qcd_signal_rejection_at_0p75"
                ]
                for row in ordered
            ]
        ),
        "per_seed_paired_statistics": {
            str(seed): row
            for seed, row in zip(EXPECTED_SEEDS, paired, strict=True)
        },
        "campaign_spec_sha256": reference["campaign_spec_sha256"],
        "global_determinism_sha256": reference["global_determinism_sha256"],
        "hlt_cache_hashes": reference["hlt_cache_hashes"],
        "event_identity_hashes": reference["event_identity_hashes"],
        "seed_artifact_hashes": {
            str(seed): row["content_hash"]
            for seed, row in zip(EXPECTED_SEEDS, ordered, strict=True)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    campaign_root = args.campaign_root.resolve()
    sys.path.insert(0, str(source_root))
    from teacher_logit_reco.relational_part import (
        load_hashed_json,
        sha256_file,
        validate_content_hash,
        with_content_hash,
        write_immutable_json,
    )

    campaign = load_hashed_json(campaign_root / "campaign_spec.json")
    paths = [
        campaign_root
        / "supplemental_diagnostics"
        / "fusion"
        / f"control_vs_{COMPARISON_ID}_seed_{seed}.json"
        for seed in EXPECTED_SEEDS
    ]
    artifacts = [
        load_hashed_json(path, expected_contract=SEED_ARTIFACT_CONTRACT)
        for path in paths
    ]
    fields = aggregate_seed_artifacts(artifacts)
    if fields["campaign_spec_sha256"] != campaign["content_hash"]:
        raise ValueError("fusion artifacts refer to another campaign")
    fields.update(
        {
            "contract": CONTRACT,
            "schema_version": 1,
            "aggregator_script_sha256": sha256_file(Path(__file__)),
        }
    )
    artifact = with_content_hash(fields)
    output = args.output or (
        campaign_root
        / "supplemental_diagnostics"
        / "fusion"
        / f"control_vs_{COMPARISON_ID}_three_seed.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_content_hash(existing, expected_contract=CONTRACT)
        if existing != artifact:
            raise FileExistsError(
                f"three-seed fusion summary differs from existing artifact: {output}"
            )
    else:
        write_immutable_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
