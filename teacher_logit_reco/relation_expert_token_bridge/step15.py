"""Step-15 production-DAG bundle publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    bind_source,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .production import (
    JOB_LEDGER_CONTRACT,
    PRODUCTION_GRAPH_CONTRACT,
    STEP15_BUNDLE_CONTRACT,
    build_step15_bundle,
    validate_job_ledger,
    validate_production_graph,
)


STEP15_PREFLIGHT_REPORT_CONTRACT = "retb_step15_preflight_report_v7"


def build_step15_preflight_report(
    *,
    production_graph: Mapping[str, Any],
    dry_run_ledger: Mapping[str, Any],
    step15_bundle: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    graph_sha = validate_production_graph(production_graph)
    ledger_sha = validate_job_ledger(
        dry_run_ledger, production_graph=production_graph
    )
    validate_content_hash(step15_bundle, expected_contract=STEP15_BUNDLE_CONTRACT)
    if (
        step15_bundle["production_graph_sha256"] != graph_sha
        or step15_bundle["dry_run_job_ledger_sha256"] != ledger_sha
    ):
        raise ValueError("Step-15 preflight lineage differs")
    return bind_source(
        with_content_hash({
            "contract": STEP15_PREFLIGHT_REPORT_CONTRACT,
            "schema_version": 7,
            "campaign_id": production_graph["campaign_id"],
            "production_graph_sha256": graph_sha,
            "dry_run_job_ledger_sha256": ledger_sha,
            "step15_bundle_sha256": step15_bundle["content_hash"],
            "checks": {
                "all_stages_A_through_N_present": True,
                "both_stage_n_selectors_present": True,
                "scale_up_present": True,
                "bounded_arrays_present": True,
                "resource_probes_present": True,
                "resumable_targets_present": True,
                "dynamic_continuation_present": True,
                "dynamic_manifest_execution_requires_binding_receipt": True,
                "stage_f_j_parent_completeness_revalidated": True,
                "stage_f_j_resumable_rows_revalidate_output_hashes": True,
                "stage_k_m_parent_completeness_revalidated": True,
                "stage_l_m_registration_only_rows_forbidden": True,
                "negative_control_or_scale_results_continue": True,
                "all_shortlisted_graph_seed_rows_required": True,
                "sealed_stage_n_parent_completeness_revalidated": True,
                "final_test_evaluation_exactly_once": True,
                "final_test_result_cannot_replace_finalist": True,
                "monitoring_and_ledger_present": True,
                "node_execution_registry_present": True,
                "automatic_manifest_producer_coverage_complete": True,
                "performance_based_termination_disabled": True,
                "provenance_failures_block_dependents": True,
                "negative_campaign_reaches_final_report": True,
            },
            "authoritative_tigris_smoke_required_before_production": True,
        }),
        source_snapshot=source_snapshot,
    )


def build_step15_contract_bundle(
    *,
    production_graph: Mapping[str, Any],
    dry_run_ledger: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        production_graph["source_commit"] != source_snapshot["source_commit"]
        or production_graph["source_status_sha256"]
        != source_snapshot["source_status_sha256"]
    ):
        raise ValueError("Step-15 production graph source differs")
    step15 = bind_source(
        build_step15_bundle(
            production_graph=production_graph,
            dry_run_ledger=dry_run_ledger,
        ),
        source_snapshot=source_snapshot,
    )
    report = build_step15_preflight_report(
        production_graph=production_graph,
        dry_run_ledger=dry_run_ledger,
        step15_bundle=step15,
        source_snapshot=source_snapshot,
    )
    return {
        "production_graph": dict(production_graph),
        "dry_run_job_ledger": dict(dry_run_ledger),
        "step15_bundle": step15,
        "step15_preflight_report": report,
    }


def validate_step15_contract_bundle(bundle: Mapping[str, Any]) -> str:
    graph = bundle["production_graph"]
    ledger = bundle["dry_run_job_ledger"]
    graph_sha = validate_production_graph(graph)
    validate_content_hash(graph, expected_contract=PRODUCTION_GRAPH_CONTRACT)
    ledger_sha = validate_job_ledger(ledger, production_graph=graph)
    validate_content_hash(ledger, expected_contract=JOB_LEDGER_CONTRACT)
    expected = build_step15_bundle(
        production_graph=graph, dry_run_ledger=ledger
    )
    step15 = bundle["step15_bundle"]
    expected.pop("content_hash")
    actual = dict(step15)
    actual.pop("content_hash")
    actual.pop("source", None)
    if actual != expected:
        raise ValueError("Step-15 bundle semantics differ")
    report = bundle["step15_preflight_report"]
    validate_content_hash(
        report, expected_contract=STEP15_PREFLIGHT_REPORT_CONTRACT
    )
    if (
        report["production_graph_sha256"] != graph_sha
        or report["dry_run_job_ledger_sha256"] != ledger_sha
        or report["step15_bundle_sha256"] != step15["content_hash"]
    ):
        raise ValueError("Step-15 report lineage differs")
    return str(step15["content_hash"])


def publish_step15_contract_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    validate_step15_contract_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "production_graph": root / "job_ledgers" / "production_graph.json",
        "dry_run_job_ledger": (
            root / "job_ledgers" / "graph_resolution_ledger.json"
        ),
        "step15_bundle": (
            root / "registry" / "retb_step15_production_bundle.json"
        ),
        "step15_preflight_report": (
            root / "reports" / "retb_step15_preflight_report.json"
        ),
    }
    return {
        name: write_immutable_json(paths[name], bundle[name])
        for name in paths
    }


__all__ = [
    "STEP15_PREFLIGHT_REPORT_CONTRACT",
    "build_step15_contract_bundle",
    "build_step15_preflight_report",
    "publish_step15_contract_bundle",
    "validate_step15_contract_bundle",
]
