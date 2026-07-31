"""Complete JSON/Markdown reporting for RETB Stage L."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .confirmation import (
    CONFIRMATION_SUMMARY_CONTRACT,
    SCALE_SHORTLIST_CONTRACT,
    SHORTLISTED_CONTROLS_CONTRACT,
    validate_shortlisted_500k_controls,
)
from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .step13 import FAILURE_INTERPRETATIONS


STAGE_L_REPORT_CONTRACT = "retb_stage_l_confirmation_report_v3"


def _interpretations(
    confirmation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = list(confirmation["rows"])
    categories = {}
    for row in rows:
        categories.setdefault(row["semantic_category"], []).append(row)
    faithful = [
        row
        for category in ("FROZEN_RECONSTRUCTION", "TOKEN_REFINER")
        for row in categories.get(category, [])
    ]
    unrestricted = categories.get("UNRESTRICTED_FUSION", [])
    native = categories.get("NATIVE_HLT_FUSION", [])
    statuses = {
        "NO_MODEL_IMPROVES": (
            "triggered"
            if confirmation["all_candidates_worse_than_baseline"]
            else "not_triggered"
        )
    }
    if unrestricted:
        best_unrestricted = max(
            unrestricted, key=lambda row: row["mean_accuracy"]
        )
        faithful_positive = any(row.get("gain_positive") for row in faithful)
        statuses["UNRESTRICTED_WITHOUT_FAITHFUL_GAIN"] = (
            "triggered"
            if not faithful_positive and (
            not native
            or best_unrestricted["mean_accuracy"]
            > max(row["mean_accuracy"] for row in native)
            )
            else "not_triggered"
        )
    return [
        {
            "id": row["interpretation_id"],
            "status": statuses.get(
                row["interpretation_id"], "not_evaluated_by_stage_l"
            ),
            "triggered": (
                None
                if row["interpretation_id"] not in statuses
                else statuses[row["interpretation_id"]] == "triggered"
            ),
            "interpretation": row["meaning"],
        }
        for row in FAILURE_INTERPRETATIONS
    ]


def render_stage_l_markdown(
    *,
    confirmation: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    shortlisted_controls: Mapping[str, Any],
    interpretations: list[Mapping[str, Any]],
) -> str:
    lines = [
        "# RETB Stage-L 500k confirmation and scale shortlist",
        "",
        "Selection population: `val_design`; matched pipeline seeds: "
        "`101, 202, 303`.",
        "",
        "No performance result stopped execution. Neither `stack_val` nor "
        "`final_test` was consumed.",
        "",
        "## Complete three-seed graph results",
        "",
        "| Graph | Category | Accuracy mean | Accuracy SD | Cross entropy | "
        "Mean log rejection | Delta accuracy vs baseline | Shortlisted |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    shortlisted = set(shortlist["SCALE_SHORTLIST"])
    for row in confirmation["rows"]:
        paired = row["paired_vs_named_baseline"]
        delta = (
            "n/a"
            if paired is None
            else f"{paired['mean_accuracy_difference']:.8f}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    row["graph_id"],
                    row["semantic_category"],
                    f"{row['mean_accuracy']:.8f}",
                    f"{row['accuracy_sample_standard_deviation']:.8f}",
                    f"{row['mean_cross_entropy']:.8f}",
                    (
                        f"{row['mean_log_Jeffreys_selection_rejection']:.8f}"
                    ),
                    delta,
                    "yes" if row["graph_id"] in shortlisted else "no",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Matched pipeline-seed results",
            "",
            "| Graph | Seed | Accuracy | Cross entropy | Mean log rejection "
            "| Accuracy delta vs named baseline | Paired delta 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in confirmation["rows"]:
        paired = row["paired_vs_named_baseline"]
        differences = (
            [None] * len(row["seed_rows"])
            if paired is None
            else paired["per_seed_accuracy_difference"]
        )
        for seed_row, difference in zip(
            row["seed_rows"], differences, strict=True
        ):
            delta = "n/a" if difference is None else f"{difference:.8f}"
            interval = seed_row["paired_statistics"]["bootstrap"][
                "paired_accuracy_difference_interval_95"
            ]
            interval_text = (
                f"[{float(interval[0]):.8f}, {float(interval[1]):.8f}]"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        row["graph_id"],
                        str(seed_row["pipeline_seed"]),
                        f"{seed_row['accuracy']:.8f}",
                        f"{seed_row['cross_entropy']:.8f}",
                        (
                            f"{seed_row['mean_log_Jeffreys_selection_rejection']:.8f}"
                        ),
                        delta,
                        interval_text,
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Locked scale shortlist",
            "",
            f"- Accuracy top three: `{', '.join(shortlist['ACC_SCALE_TOP3'])}`",
            f"- Rejection top three: `{', '.join(shortlist['REJ_SCALE_TOP3'])}`",
            f"- Canonical union: `{', '.join(shortlist['SCALE_SHORTLIST'])}`",
            f"- `SHAPE_BRIDGE`: `{shortlist['SHAPE_BRIDGE']['shape_id']}`",
            "",
            "The lock contains graph definitions only. It contains no 3M "
            "checkpoint and does not name a final finalist.",
            "",
            "## Post-shortlist matched controls",
            "",
            "| Graph | H_MONO_PARAM accuracy | H_MONO_FLOP accuracy | "
            "H_BASE_LONG accuracy | Capacity reproduces gain |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for row in shortlisted_controls["rows"]:
        lines.append(
            f"| {row['graph_id']} | "
            f"{row['mean_accuracy_by_control']['H_MONO_PARAM']:.8f} | "
            f"{row['mean_accuracy_by_control']['H_MONO_FLOP']:.8f} | "
            f"{row['mean_accuracy_by_control']['H_BASE_LONG']:.8f} | "
            f"{'yes' if row['capacity_control_reproduces_gain'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "These controls were resolved after the shortlist lock and did "
            "not alter membership.",
            "",
            "## Predeclared interpretation",
            "",
        ]
    )
    for row in interpretations:
        lines.append(
            f"- `{row['id']}` ({row['status']}): "
            f"{row['interpretation']}"
        )
    return "\n".join(lines) + "\n"


def build_stage_l_report(
    *,
    confirmation: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    shortlisted_controls: Mapping[str, Any],
    step13_bundle_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    confirmation_sha = validate_content_hash(
        confirmation, expected_contract=CONFIRMATION_SUMMARY_CONTRACT
    )
    shortlist_sha = validate_content_hash(
        shortlist, expected_contract=SCALE_SHORTLIST_CONTRACT
    )
    controls_sha = validate_shortlisted_500k_controls(
        shortlisted_controls,
        locked_scale_shortlist=shortlist,
    )
    if (
        shortlist["parent_hashes"]["confirmation_summary"]
        != confirmation_sha
        or confirmation.get("source") != shortlist.get("source")
        or shortlisted_controls.get("source") != shortlist.get("source")
        or confirmation.get("stack_val_consumed")
        or confirmation.get("final_test_consumed")
        or shortlist.get("stack_val_consumed")
        or shortlist.get("final_test_consumed")
    ):
        raise ValueError("Stage-L report lineage/access differs")
    interpretations = _interpretations(confirmation)
    markdown = render_stage_l_markdown(
        confirmation=confirmation,
        shortlist=shortlist,
        shortlisted_controls=shortlisted_controls,
        interpretations=interpretations,
    )
    artifact = bind_source(
        with_content_hash(
            {
                "contract": STAGE_L_REPORT_CONTRACT,
                "schema_version": 3,
                "parents": {
                    "step13_bundle": require_sha256(
                        step13_bundle_sha256,
                        name="step13_bundle_sha256",
                    ),
                    "confirmation_summary": confirmation_sha,
                    "locked_scale_shortlist": shortlist_sha,
                    "shortlisted_500k_controls": controls_sha,
                },
                "complete_graph_rows": confirmation["rows"],
                "complete_matched_seed_coverage": confirmation[
                    "complete_matched_seed_coverage"
                ],
                "ACC_SCALE_TOP3": shortlist["ACC_SCALE_TOP3"],
                "REJ_SCALE_TOP3": shortlist["REJ_SCALE_TOP3"],
                "SCALE_SHORTLIST": shortlist["SCALE_SHORTLIST"],
                "SHAPE_BRIDGE": shortlist["SHAPE_BRIDGE"],
                "failure_interpretations": interpretations,
                "shortlisted_graph_controls": shortlisted_controls["rows"],
                "all_candidates_worse_than_baseline": confirmation[
                    "all_candidates_worse_than_baseline"
                ],
                "markdown_sha256": hashlib.sha256(
                    markdown.encode("utf-8")
                ).hexdigest(),
                "scientific_underperformance_blocked_execution": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return artifact, markdown


def publish_stage_l_report(
    *,
    output_dir: str | Path,
    artifact: Mapping[str, Any],
    markdown: str,
) -> dict[str, Any]:
    validate_content_hash(
        artifact, expected_contract=STAGE_L_REPORT_CONTRACT
    )
    encoded = markdown.encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != artifact["markdown_sha256"]:
        raise ValueError("Stage-L Markdown bytes differ")
    root = Path(output_dir)
    return {
        "json": write_immutable_json(
            root / "retb_stage_l_report.json", artifact
        ),
        "markdown": write_immutable_bytes(
            root / "retb_stage_l_report.md", encoded
        ),
    }


__all__ = [
    "STAGE_L_REPORT_CONTRACT",
    "build_stage_l_report",
    "publish_stage_l_report",
    "render_stage_l_markdown",
]
