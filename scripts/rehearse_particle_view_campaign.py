#!/usr/bin/env python3
"""Run the deterministic miniature filesystem rehearsal for Step 10."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    LOGICAL_NODE_LAYOUT,
    ParticleViewRunSpec,
    build_node_completion,
    build_particle_view_production_graph,
    build_particle_view_registry,
    build_quality_warning,
    reconcile_particle_view_production_graph,
    submit_particle_view_graph,
    with_content_hash,
    write_immutable_json,
    write_quality_warning_summary,
    write_quality_warning_jsonl,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser


def _miniature_registry():
    specs = []
    previous = None
    for index, (_, stages, _) in enumerate(LOGICAL_NODE_LAYOUT):
        stage = stages[0]
        run_id = f"mini_{index:02d}_{stage}"
        nontraining = stage in {"source", "stack", "report_export", "final_test"}
        selectable = stage == "final_test"
        specs.append(
            ParticleViewRunSpec(
                run_id=run_id,
                stage=stage,
                scientific_role=f"miniature_{stage}",
                selection_family=(
                    "pre_stage_g_deployable" if selectable else "infrastructure"
                ),
                parent_run_ids=(() if previous is None else (previous,)),
                uses_labels=not nontraining,
                train_split=None if nontraining else "train",
                selectable=selectable,
                final_test_eligible=selectable,
            )
        )
        previous = run_id
    return build_particle_view_registry(
        unified_split_manifest_sha256=_sha("mini-manifest"),
        train_identity_sha256=_sha("mini-train"),
        run_specs=specs,
        campaign_id="particle_view_miniature_rehearsal_v1",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry = _miniature_registry()
    commands = {
        node_id: [sys.executable, "-c", f"print('rehearsal {node_id}')"]
        for node_id, _, _ in LOGICAL_NODE_LAYOUT
    }
    graph = build_particle_view_production_graph(
        registry=registry,
        artifact_root=str(root),
        source_commit="1" * 40,
        command_catalog=commands,
        graph_id="particle_view_miniature_rehearsal_v1",
    )
    reconciliation = reconcile_particle_view_production_graph(
        graph=graph, registry=registry
    )
    registry_path = root / "registry.json"
    graph_path = root / "production_graph.json"
    write_immutable_json(registry_path, registry)
    write_immutable_json(graph_path, graph)
    write_immutable_json(root / "graph_reconciliation.json", reconciliation)

    support = with_content_hash(
        {"contract": "particle_view_rehearsal_metric_v1", "accuracy": 0.5}
    )
    support_path = root / "metrics" / "weak_metric.json"
    write_immutable_json(support_path, support)
    warning = build_quality_warning(
        warning_code="WARN_WEAK_RECOVERY",
        severity="warning",
        graph_node="pv05_predictor_loss_packs",
        configuration_id="mini_predictor",
        seed=101,
        split="model_val_select",
        observed_value=0.0,
        reference_value=0.1,
        warning_threshold=0.01,
        interpretation="Miniature warning-continuation fixture.",
        suggested_diagnostic="Inspect the rehearsal metric.",
        supporting_artifacts=[
            {"path": str(support_path), "sha256": support["content_hash"]}
        ],
        source_commit="1" * 40,
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    warning_path = root / "quality_warnings" / "rehearsal_warning.json"
    write_immutable_json(warning_path, warning)
    for node_id, _, _ in LOGICAL_NODE_LAYOUT:
        write_quality_warning_jsonl(
            root
            / "quality_warnings"
            / node_id
            / "quality_warnings.jsonl",
            [warning] if node_id == "pv05_predictor_loss_packs" else [],
        )
    warning_summary = write_quality_warning_summary(
        output_dir=root, warnings=[warning]
    )

    for node_id, _, _ in LOGICAL_NODE_LAYOUT:
        completion = build_node_completion(
            graph=graph,
            node_id=node_id,
            output_artifacts=[
                {"path": str(support_path), "sha256": support["content_hash"]}
            ],
            warning_sha256=(
                [warning["content_hash"]]
                if node_id == "pv05_predictor_loss_packs"
                else []
            ),
            rehearsal=True,
        )
        write_immutable_json(
            root / "node_completions" / f"{node_id}.json", completion
        )

    clean_ledger = submit_particle_view_graph(
        graph=graph,
        graph_path=str(graph_path),
        mode="dry_run",
    )
    existing = {
        node_id: {"job_id": str(90_000 + index), "state": "COMPLETED"}
        for index, (node_id, _, _) in enumerate(LOGICAL_NODE_LAYOUT[:6])
    }
    recovery_ledger = submit_particle_view_graph(
        graph=graph,
        graph_path=str(graph_path),
        existing_jobs=existing,
        mode="dry_run",
    )
    first_recovered = recovery_ledger["records"][6]
    warning_did_not_block_descendants = (
        first_recovered["node_id"] == "pv06_confirmation_selection"
        and first_recovered["action"] == "submit"
        and first_recovered["dependency_job_ids"] == []
        and all(
            "warning" not in token.lower()
            for token in first_recovered["command"]
        )
    )
    if not warning_did_not_block_descendants:
        raise RuntimeError("quality warning incorrectly blocked campaign recovery")
    write_immutable_json(root / "clean_start_ledger.json", clean_ledger)
    write_immutable_json(root / "recovery_ledger.json", recovery_ledger)
    report = with_content_hash(
        {
            "contract": "particle_view_rehearsal_report_v1",
            "status": "PASS",
            "registry_sha256": registry["content_hash"],
            "graph_sha256": graph["content_hash"],
            "reconciliation_sha256": reconciliation["content_hash"],
            "logical_node_count": len(LOGICAL_NODE_LAYOUT),
            "completion_count": len(LOGICAL_NODE_LAYOUT),
            "clean_start_planned_count": clean_ledger["planned_submit_count"],
            "recovery_reused_completed_count": 6,
            "recovery_planned_count": recovery_ledger["planned_submit_count"],
            "warning_count": warning_summary["warning_count"],
            "warning_aggregate_exit_code": 0,
            "warning_did_not_block_descendants": warning_did_not_block_descendants,
            "python_no_user_site_required": True,
            "tigris_account": "reu-aisocial",
        }
    )
    write_immutable_json(root / "rehearsal_report.json", report)
    print(
        f"status=PASS nodes={report['logical_node_count']} "
        f"warnings={report['warning_count']} "
        f"content_hash={report['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
