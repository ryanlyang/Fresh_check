"""Complete Stage-M/N JSON and Markdown reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    bind_source,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .final_seal import (
    FINAL_TEST_EVALUATION_CONTRACT,
    FINAL_TEST_EXECUTION_LOCK_CONTRACT,
)
from .scale_up import SCALE_COMPLETION_CONTRACT
from .stage_n_selection import LOCKED_SCALE_FINALISTS_CONTRACT
from .semantic_evidence import (
    SEMANTIC_CONTROLS_CONTRACT,
    validate_stage_k_semantic_controls,
)


STAGE_MN_REPORT_CONTRACT = "retb_stage_mn_final_report_v3"


def render_stage_mn_markdown(
    *,
    scale_completion: Mapping[str, Any],
    locked_scale_finalists: Mapping[str, Any],
    final_evaluation: Mapping[str, Any],
    semantic_controls: Mapping[str, Any],
) -> str:
    lines = [
        "# RETB Stage-M/N scale-up and sealed final evaluation",
        "",
        "Every locked 500k shortlist graph was retrained at 3M across "
        "pipeline seeds `101, 202, 303`. No performance result stopped the "
        "workflow.",
        "",
        "## Locked 3M finalists",
        "",
        f"- Accuracy finalist: `{locked_scale_finalists['ACCURACY_FINALIST']}`",
        f"- Rejection finalist: `{locked_scale_finalists['REJECTION_FINALIST']}`",
        f"- Same graph won both: `{locked_scale_finalists['same_graph_won_both']}`",
        "",
        "## Pre-stack 3M confirmation",
        "",
        "| Graph | Seed | Accuracy | Cross entropy | Mean log rejection |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in scale_completion["runs"]:
        metrics = row["pre_stack_confirmation"]["metrics"]
        lines.append(
            f"| {row['graph_id']} | {row['pipeline_seed']} | "
            f"{metrics['accuracy']:.8f} | {metrics['cross_entropy']:.8f} | "
            f"{row['pre_stack_confirmation']['mean_log_Jeffreys_selection_rejection']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Sealed final-test rows",
            "",
            "| Row | Accuracy | Cross entropy | Macro accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for row_id in final_evaluation["evaluated_row_ids"]:
        metrics = final_evaluation["classification_metrics"][row_id]
        lines.append(
            f"| {row_id} | {metrics['accuracy']:.8f} | "
            f"{metrics['cross_entropy']:.8f} | "
            f"{metrics['macro_per_class_accuracy']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Paired finalist statistics",
            "",
        ]
    )
    for row_id, paired in sorted(
        final_evaluation["paired_finalist_statistics"].items()
    ):
        interval = paired["bootstrap"][
            "paired_accuracy_difference_interval_95"
        ]
        rejection_interval = paired["bootstrap"][
            "paired_mean_log_rejection_difference_interval_95"
        ]
        lines.append(
            f"- `{row_id}`: accuracy delta "
            f"`{paired['central']['paired_accuracy_difference']:.8f}` "
            f"(95% CI `[{interval[0]:.8f}, {interval[1]:.8f}]`); "
            f"mean-log-rejection delta "
            f"`{paired['central']['mean_log_rejection_difference']:.8f}` "
            f"(95% CI `[{rejection_interval[0]:.8f}, "
            f"{rejection_interval[1]:.8f}]`)."
        )
    if final_evaluation["paired_between_distinct_finalists"]:
        lines.extend(
            [
                "",
                "## Paired difference between distinct finalists",
                "",
            ]
        )
        for row_id, paired in sorted(
            final_evaluation[
                "paired_between_distinct_finalists"
            ].items()
        ):
            interval = paired["bootstrap"][
                "paired_mean_log_rejection_difference_interval_95"
            ]
            lines.append(
                f"- `{row_id}`: mean-log-rejection delta "
                f"`{paired['central']['mean_log_rejection_difference']:.8f}` "
                f"(95% CI `[{interval[0]:.8f}, {interval[1]:.8f}]`)."
            )
    lines.extend(
        [
            "",
            "## Semantic and causal controls (val_design only)",
            "",
            "These perturbations were evaluated after selection and never "
            "selected a checkpoint. Deltas are control minus the serialized "
            "active/reference condition.",
            "",
            "| Control | Evaluation coordinate | Accuracy | Δ accuracy | "
            "Cross entropy | Δ cross entropy |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for control in semantic_controls["rows"]:
        control_id = str(control["control_id"])
        for index, record in enumerate(control["metric_records"]):
            metrics = record.get("metrics") or record.get("learned_metrics")
            deltas = record.get("metric_deltas") or record.get(
                "learned_minus_fixed"
            )
            coordinate_parts = [
                f"{key}={record[key]}"
                for key in (
                    "shape_role", "pipeline_seed", "consumer_kind",
                    "expert_id", "condition_id", "candidate_id",
                )
                if key in record
            ]
            coordinate = ", ".join(coordinate_parts) or f"record={index}"
            if metrics is None:
                lines.append(
                    f"| {control_id} | {coordinate}; diagnostics-only | "
                    "NA | NA | NA | NA |"
                )
                diagnostic = {
                    key: value for key, value in record.items()
                    if key != "source_artifact_sha256"
                }
                lines.append(
                    f"<!-- {control_id} {coordinate}: "
                    f"{json.dumps(diagnostic, sort_keys=True)} -->"
                )
                continue
            accuracy_delta = (
                "NA" if deltas is None else
                f"{float(deltas['accuracy_control_minus_reference']):.8f}"
            )
            ce_delta = (
                "NA" if deltas is None else
                f"{float(deltas['cross_entropy_control_minus_reference']):.8f}"
            )
            lines.append(
                f"| {control_id} | {coordinate} | "
                f"{float(metrics['accuracy']):.8f} | {accuracy_delta} | "
                f"{float(metrics['cross_entropy']):.8f} | {ce_delta} |"
            )
    lines.extend(
        [
            "",
            "### Frozen reconstruction, token refiner, and unrestricted fusion",
            "",
            "| Shape | Seed | Condition | Accuracy | Delta vs frozen | "
            "Cross entropy | Delta vs frozen |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for record in semantic_controls["reconstruction_metric_records"]:
        for condition, metrics in record["condition_metrics"].items():
            deltas = record[
                "condition_metric_deltas_vs_frozen_reconstruction"
            ].get(condition)
            accuracy_delta = 0.0 if deltas is None else float(
                deltas["accuracy_control_minus_reference"]
            )
            ce_delta = 0.0 if deltas is None else float(
                deltas["cross_entropy_control_minus_reference"]
            )
            lines.append(
                f"| {record['shape_role']} | {record['pipeline_seed']} | "
                f"{condition} | {float(metrics['accuracy']):.8f} | "
                f"{accuracy_delta:.8f} | "
                f"{float(metrics['cross_entropy']):.8f} | {ce_delta:.8f} |"
            )
    lines.extend(
        [
            "",
            "The final-test result did not select or replace any graph. "
            "All reported rows were sealed by the execution lock.",
            "",
        ]
    )
    return "\n".join(lines)


def build_stage_mn_report(
    *,
    scale_completion: Mapping[str, Any],
    locked_scale_finalists: Mapping[str, Any],
    execution_lock: Mapping[str, Any],
    final_evaluation: Mapping[str, Any],
    semantic_controls: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    scale_sha = validate_content_hash(
        scale_completion, expected_contract=SCALE_COMPLETION_CONTRACT
    )
    finalist_sha = validate_content_hash(
        locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    execution_sha = validate_content_hash(
        execution_lock,
        expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    )
    final_sha = validate_content_hash(
        final_evaluation,
        expected_contract=FINAL_TEST_EVALUATION_CONTRACT,
    )
    semantic_sha = validate_stage_k_semantic_controls(
        semantic_controls,
        expected_source=scale_completion["source"],
    )
    if (
        locked_scale_finalists["lineage_hashes"]["scale_completion"]
        != scale_sha
        or execution_lock["parent_hashes"]["locked_scale_finalists"]
        != finalist_sha
        or final_evaluation["final_test_execution_lock_sha256"]
        != execution_sha
        or len(
            {
                repr(row.get("source"))
                for row in (
                    scale_completion,
                    locked_scale_finalists,
                    execution_lock,
                    final_evaluation,
                    semantic_controls,
                )
            }
        )
        != 1
    ):
        raise ValueError("Stage-M/N report lineage differs")
    markdown = render_stage_mn_markdown(
        scale_completion=scale_completion,
        locked_scale_finalists=locked_scale_finalists,
        final_evaluation=final_evaluation,
        semantic_controls=semantic_controls,
    )
    artifact = bind_source(
        with_content_hash(
            {
                "contract": STAGE_MN_REPORT_CONTRACT,
                "schema_version": 3,
                "parents": {
                    "scale_completion": scale_sha,
                    "locked_scale_finalists": finalist_sha,
                    "final_test_execution_lock": execution_sha,
                    "sealed_final_test_evaluation": final_sha,
                    "semantic_controls": semantic_sha,
                },
                "ACCURACY_FINALIST": locked_scale_finalists[
                    "ACCURACY_FINALIST"
                ],
                "REJECTION_FINALIST": locked_scale_finalists[
                    "REJECTION_FINALIST"
                ],
                "scale_runs": scale_completion["runs"],
                "final_classification_metrics": final_evaluation[
                    "classification_metrics"
                ],
                "paired_finalist_statistics": final_evaluation[
                    "paired_finalist_statistics"
                ],
                "paired_between_distinct_finalists": final_evaluation[
                    "paired_between_distinct_finalists"
                ],
                "semantic_controls": semantic_controls["rows"],
                "reconstruction_metric_records": semantic_controls[
                    "reconstruction_metric_records"
                ],
                "markdown_sha256": hashlib.sha256(
                    markdown.encode("utf-8")
                ).hexdigest(),
                "performance_based_termination": False,
                "test_result_selected_replacement": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return artifact, markdown


def publish_stage_mn_report(
    *,
    output_dir: str | Path,
    artifact: Mapping[str, Any],
    markdown: str,
) -> dict[str, Any]:
    validate_content_hash(
        artifact, expected_contract=STAGE_MN_REPORT_CONTRACT
    )
    encoded = markdown.encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != artifact["markdown_sha256"]:
        raise ValueError("Stage-M/N Markdown bytes differ")
    root = Path(output_dir)
    return {
        "json": write_immutable_json(
            root / "retb_stage_mn_final_report.json", artifact
        ),
        "markdown": write_immutable_bytes(
            root / "retb_stage_mn_final_report.md", encoded
        ),
    }


__all__ = [
    "STAGE_MN_REPORT_CONTRACT",
    "build_stage_mn_report",
    "publish_stage_mn_report",
    "render_stage_mn_markdown",
]
