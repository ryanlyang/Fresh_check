#!/usr/bin/env python3
"""Aggregate real three-seed controls for every locked 500k shortlist graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    CONFIRMATION_SUMMARY_CONTRACT,
    SCALE_SHORTLIST_CONTRACT,
    build_shortlisted_500k_controls,
    validate_shortlisted_500k_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_capacity_controls import (  # noqa: E402
    HLT_CAPACITY_CONTROL_EXPORT_CONTRACT,
    HLT_CAPACITY_CONTROL_ROW_CONTRACT,
    validate_hlt_capacity_control_row,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_l_reporting import (  # noqa: E402
    build_stage_l_report,
    publish_stage_l_report,
)
from teacher_logit_reco.relation_expert_token_bridge.step13 import (  # noqa: E402
    STEP13_BUNDLE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)

KINDS = ("H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG")
SEEDS = (101, 202, 303)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--confirmation-summary", required=True, type=Path)
    parser.add_argument("--control-row", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    confirmation = load_hashed_json(
        args.confirmation_summary,
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    step13 = load_hashed_json(
        root / "registry" / "retb_step13_confirmation_shortlist_bundle.json",
        expected_contract=STEP13_BUNDLE_CONTRACT,
    )
    if any(
        artifact.get("source") != campaign.get("source")
        for artifact in (shortlist, confirmation, step13)
    ):
        raise ValueError("500k control aggregation source differs")

    loaded: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for path in args.control_row:
        row = load_hashed_json(
            path, expected_contract=HLT_CAPACITY_CONTROL_ROW_CONTRACT
        )
        validate_hlt_capacity_control_row(row)
        key = (
            str(row["owner_finalist_graph_id"]),
            str(row["control_kind"]),
            int(row["pipeline_seed"]),
        )
        if row.get("source") != campaign.get("source") or key in loaded:
            raise ValueError("500k control row source/identity differs")
        loaded[key] = (path.resolve(), row)

    expected = {
        (graph_id, kind, seed)
        for graph_id in shortlist["SCALE_SHORTLIST"]
        for kind in KINDS
        for seed in SEEDS
    }
    if set(loaded) != expected:
        raise ValueError("500k control wave coverage differs")
    confirmation_by_graph = {
        row["graph_id"]: row for row in confirmation["rows"]
    }
    snapshot = source_snapshot(REPO_ROOT)
    rows = []
    for graph_id in shortlist["SCALE_SHORTLIST"]:
        metric_rows: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KINDS}
        hashes: dict[str, list[str]] = {kind: [] for kind in KINDS}
        for kind in KINDS:
            for seed in SEEDS:
                path, control = loaded[(graph_id, kind, seed)]
                metrics = load_hashed_json(
                    path.parent / "training" / "val_design_metrics.json"
                )
                registration = load_hashed_json(
                    path.parent / "training" / "registration.json"
                )
                deployable = load_hashed_json(
                    path.parent / "deployable_control.json",
                    expected_contract=HLT_CAPACITY_CONTROL_EXPORT_CONTRACT,
                )
                if (
                    registration["content_hash"]
                    != control["training_registration_sha256"]
                    or registration["val_design_metrics_sha256"]
                    != metrics["content_hash"]
                    or deployable["content_hash"]
                    != control["deployable_export_sha256"]
                    or deployable["training_registration_sha256"]
                    != registration["content_hash"]
                    or any(
                        artifact.get("source") != campaign.get("source")
                        for artifact in (metrics, registration, deployable)
                    )
                ):
                    raise ValueError("500k control metrics lineage differs")
                metric_rows[kind].append(
                    {
                        "pipeline_seed": seed,
                        "accuracy": float(metrics["accuracy"]),
                        "metrics_sha256": metrics["content_hash"],
                        "training_row_sha256": control["content_hash"],
                    }
                )
                hashes[kind].append(control["content_hash"])
        means = {
            kind: sum(item["accuracy"] for item in metric_rows[kind]) / 3.0
            for kind in KINDS
        }
        graph_accuracy = float(confirmation_by_graph[graph_id]["mean_accuracy"])
        metrics_artifact = bind_source(
            with_content_hash(
                {
                    "contract": "retb_shortlisted_500k_control_metrics_v1",
                    "schema_version": 1,
                    "graph_id": graph_id,
                    "locked_scale_shortlist_sha256": shortlist["content_hash"],
                    "control_rows": metric_rows,
                    "mean_accuracy_by_control": means,
                    "shortlisted_graph_mean_accuracy": graph_accuracy,
                    "capacity_control_reproduces_gain": max(
                        means["H_MONO_PARAM"], means["H_MONO_FLOP"]
                    )
                    >= graph_accuracy,
                    "comparison_rule": (
                        "maximum_three_seed_monolithic_mean_accuracy_greater_"
                        "than_or_equal_to_shortlisted_graph_mean_accuracy"
                    ),
                    "performance_result_blocked_stage_M": False,
                    "stack_val_consumed": False,
                    "final_test_consumed": False,
                }
            ),
            source_snapshot=snapshot,
        )
        metrics_path = args.output.parent / "metrics" / f"{graph_id}.json"
        write_immutable_json(metrics_path, metrics_artifact)
        capacity = load_hashed_json(
            root
            / "exports"
            / shortlist["locked_graph_definitions"][graph_id]["configuration"][
                "run_ids_by_seed"
            ]["101"]
            / "complete_graph_capacity.json"
        )
        rows.append(
            {
                "graph_id": graph_id,
                "complete_graph_capacity_sha256": capacity["content_hash"],
                "monolithic_parameter_control_sha256": canonical_sha256(
                    hashes["H_MONO_PARAM"]
                ),
                "monolithic_flop_control_sha256": canonical_sha256(
                    hashes["H_MONO_FLOP"]
                ),
                "H_BASE_LONG_label_exposure_control_sha256": canonical_sha256(
                    hashes["H_BASE_LONG"]
                ),
                "control_metrics_artifact_sha256": metrics_artifact["content_hash"],
                "capacity_control_reproduces_gain": metrics_artifact[
                    "capacity_control_reproduces_gain"
                ],
                "training_row_hashes_by_kind": hashes,
                "mean_accuracy_by_control": means,
            }
        )
    controls = bind_source(
        build_shortlisted_500k_controls(
            locked_scale_shortlist=shortlist, rows=rows
        ),
        source_snapshot=snapshot,
    )
    validate_shortlisted_500k_controls(
        controls, locked_scale_shortlist=shortlist
    )
    publication = write_immutable_json(args.output, controls)
    report, markdown = build_stage_l_report(
        confirmation=confirmation,
        shortlist=shortlist,
        shortlisted_controls=controls,
        step13_bundle_sha256=step13["content_hash"],
        source_snapshot=snapshot,
    )
    report_publication = publish_stage_l_report(
        output_dir=args.report_output_dir,
        artifact=report,
        markdown=markdown,
    )
    print(
        json.dumps(
            {
                "publication": publication,
                "report_publication": report_publication,
                "graph_count": len(rows),
                "training_row_count": len(loaded),
                "performance_result_blocked_stage_M": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
