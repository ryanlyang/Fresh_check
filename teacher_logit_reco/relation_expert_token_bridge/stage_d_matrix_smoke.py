"""Authenticated full-matrix miniature execution for RETB Stages A--D."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    require_sha256,
    source_record,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .hlt_experts import DUAL_WEIGHTS, HLT_MODES
from .native_fusion import NATIVE_FUSION_VARIANTS
from .production import validate_production_graph
from .registry import EXPERT_ORDER
from .replicas import REALIZATION_POLICIES
from .static_experiments import STATIC_EXPERIMENT_PLAN_CONTRACT
from .step6 import STAGE_D_SHAPES


STAGE_D_MATRIX_SMOKE_SCOPE_CONTRACT = "retb_stage_d_matrix_smoke_scope_v1"
STAGE_D_MATRIX_SMOKE_LEDGER_CONTRACT = "retb_stage_d_matrix_smoke_ledger_v1"
STAGE_D_MATRIX_SMOKE_REPORT_CONTRACT = "retb_stage_d_matrix_smoke_report_v1"
STAGE_D_MATRIX_SMOKE_STAGES = "ABCD"
STAGE_D_MATRIX_SMOKE_TERMINAL_NODE = "native_hlt_fusion_training"
STAGE_D_MATRIX_SMOKE_COUNTS = {
    "offline_expert_training": 147,
    "offline_expert_confirmation": 147,
    "offline_fusion_cache": 63,
    "offline_fusion_training": 49,
    "native_hlt_expert_training": 541,
    "native_hlt_fusion_training": 30,
}


def _source(production_graph: Mapping[str, Any]) -> dict[str, Any]:
    return source_record(
        {
            "source_commit": str(production_graph["source_commit"]),
            "source_status_sha256": str(
                production_graph["source_status_sha256"]
            ),
            "source_dirty": False,
        }
    )


def stage_d_matrix_smoke_node_ids(
    production_graph: Mapping[str, Any],
) -> list[str]:
    """Return the exact dependency-closed production prefix through Stage D."""
    validate_production_graph(production_graph)
    selected = [
        str(node["node_id"])
        for node in production_graph["nodes"]
        if str(node["stage"]) in STAGE_D_MATRIX_SMOKE_STAGES
    ]
    selected_set = set(selected)
    if not selected or selected[-1] != STAGE_D_MATRIX_SMOKE_TERMINAL_NODE:
        raise ValueError("Stage-D matrix smoke terminal node differs")
    for node in production_graph["nodes"]:
        if node["node_id"] not in selected_set:
            continue
        if any(parent not in selected_set for parent in node["dependencies"]):
            raise ValueError("Stage-D matrix smoke prefix is not dependency closed")
    return selected


def build_stage_d_matrix_smoke_scope(
    *, production_graph: Mapping[str, Any]
) -> dict[str, Any]:
    graph_sha = validate_production_graph(production_graph)
    if (
        production_graph.get("campaign_profile")
        != "nonproduction_miniature_test"
        or production_graph.get("scientific_results_allowed") is not False
    ):
        raise ValueError("Stage-D matrix smoke requires the miniature graph")
    nodes = stage_d_matrix_smoke_node_ids(production_graph)
    return with_content_hash(
        {
            "contract": STAGE_D_MATRIX_SMOKE_SCOPE_CONTRACT,
            "schema_version": 1,
            "campaign_id": str(production_graph["campaign_id"]),
            "campaign_root": str(Path(production_graph["campaign_root"])),
            "production_graph_sha256": graph_sha,
            "campaign_profile": "nonproduction_miniature_test",
            "submitted_stages": list(STAGE_D_MATRIX_SMOKE_STAGES),
            "submitted_node_ids": nodes,
            "submitted_node_count": len(nodes),
            "terminal_node_id": STAGE_D_MATRIX_SMOKE_TERMINAL_NODE,
            "expected_static_matrix_counts": dict(STAGE_D_MATRIX_SMOKE_COUNTS),
            "all_541_native_hlt_configurations_required": True,
            "all_30_native_fusion_configurations_required": True,
            "miniature_epoch_budget_per_trainable_row": 2,
            "one_forward_backward_optimizer_update_minimum": True,
            "validation_and_output_reload_required": True,
            "scientific_performance_claimed": False,
            "scientific_underperformance_blocks_continuation": False,
            "runtime_lineage_or_nonfinite_failure_blocks_continuation": True,
            "final_test_submitted": False,
            "production_evidence_eligible": False,
            "source": _source(production_graph),
        }
    )


def validate_stage_d_matrix_smoke_scope(
    payload: Mapping[str, Any], *, production_graph: Mapping[str, Any]
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_D_MATRIX_SMOKE_SCOPE_CONTRACT
    )
    expected = build_stage_d_matrix_smoke_scope(
        production_graph=production_graph
    )
    if dict(payload) != expected:
        raise ValueError("Stage-D matrix smoke scope semantics differ")
    return digest


def summarize_stage_d_matrix(
    static_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and summarize the complete Stage-D static coordinate surface."""
    validate_content_hash(
        static_plan, expected_contract=STATIC_EXPERIMENT_PLAN_CONTRACT
    )
    counts = {
        name: len(static_plan["groups"][name])
        for name in STAGE_D_MATRIX_SMOKE_COUNTS
    }
    if counts != STAGE_D_MATRIX_SMOKE_COUNTS:
        raise ValueError("Stage-D matrix smoke static counts differ")
    native = list(static_plan["groups"]["native_hlt_expert_training"])
    fusions = list(static_plan["groups"]["native_hlt_fusion_training"])
    if len({str(row["run_id"]) for row in native}) != len(native):
        raise ValueError("Stage-D native run IDs are not unique")
    if any(
        row.get("scientific_underperformance_skips_row") is not False
        for row in (*native, *fusions)
    ):
        raise ValueError("Stage-D matrix gained a performance-based row gate")
    expert_rows = [
        row for row in native
        if row["configuration"].get("kind") == "NATIVE_HLT_EXPERT"
    ]
    control_rows = [
        row for row in native
        if row["configuration"].get("kind") == "NATIVE_HLT_MATCHED_CONTROL"
    ]
    experts = sorted(
        {str(row["configuration"]["expert_id"]) for row in expert_rows}
    )
    modes = sorted({str(row["configuration"]["mode"]) for row in expert_rows})
    policies = sorted(
        {str(row["configuration"]["realization_policy"]) for row in expert_rows}
    )
    shapes = sorted(
        {str(row["configuration"]["shape_id"]) for row in expert_rows}
    )
    measurement = sorted(
        {bool(row["configuration"]["measurement_embedding"]) for row in expert_rows}
    )
    dual_weights = sorted(
        {
            (
                float(row["configuration"]["lambda_token"]),
                float(row["configuration"]["lambda_logit"]),
            )
            for row in expert_rows
            if row["configuration"]["mode"] == "HE_DUAL_OBJECTIVE"
        }
    )
    controls = sorted(
        {str(row["configuration"]["control_id"]) for row in control_rows}
    )
    variants = sorted(
        {
            str(row["configuration"]["fusion_variant"])
            for row in fusions
        }
    )
    if experts != sorted(EXPERT_ORDER):
        raise ValueError("Stage-D expert coverage differs")
    if modes != sorted(HLT_MODES):
        raise ValueError("Stage-D evidence-mode coverage differs")
    if policies != sorted(REALIZATION_POLICIES):
        raise ValueError("Stage-D realization-policy coverage differs")
    if shapes != sorted(STAGE_D_SHAPES):
        raise ValueError("Stage-D shape coverage differs")
    if measurement != [False, True]:
        raise ValueError("Stage-D measurement-embedding coverage differs")
    if dual_weights != sorted(tuple(map(float, row)) for row in DUAL_WEIGHTS):
        raise ValueError("Stage-D dual-weight coverage differs")
    if controls != ["H_BASE", "H_WIDE"]:
        raise ValueError("Stage-D matched-control coverage differs")
    if variants != sorted(NATIVE_FUSION_VARIANTS):
        raise ValueError("Stage-D native-fusion coverage differs")
    return {
        "static_matrix_counts": counts,
        "native_hlt_run_count": len(native),
        "native_hlt_run_ids_sha256": require_sha256(
            hashlib.sha256(
                "\n".join(str(row["run_id"]) for row in native).encode("utf-8")
            ).hexdigest(),
            name="native_hlt_run_ids_sha256",
        ),
        "expert_ids": experts,
        "evidence_modes": modes,
        "realization_policies": policies,
        "shape_ids": shapes,
        "measurement_embedding_values": measurement,
        "dual_objective_weights": [list(row) for row in dual_weights],
        "matched_controls": controls,
        "native_fusion_variants": variants,
        "all_rows_non_performance_gated": True,
    }


def assert_finite_json(value: Any, *, path: str = "root") -> None:
    """Reject NaN/inf in any JSON-derived evidence tree."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite Stage-D evidence at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite_json(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            assert_finite_json(child, path=f"{path}[{index}]")
        return
    raise TypeError(f"unsupported Stage-D JSON evidence at {path}")


def build_stage_d_matrix_smoke_ledger(
    *,
    production_graph: Mapping[str, Any],
    jobs: Mapping[str, str],
    report_job_id: str,
) -> dict[str, Any]:
    scope = build_stage_d_matrix_smoke_scope(production_graph=production_graph)
    expected = set(scope["submitted_node_ids"])
    if set(jobs) != expected:
        raise ValueError("Stage-D matrix smoke job bindings differ")
    normalized = {str(key): str(value) for key, value in sorted(jobs.items())}
    if any(not value.isdigit() or int(value) <= 0 for value in normalized.values()):
        raise ValueError("Stage-D matrix smoke job ID differs")
    report = str(report_job_id)
    if not report.isdigit() or int(report) <= 0:
        raise ValueError("Stage-D matrix smoke report job ID differs")
    return with_content_hash(
        {
            "contract": STAGE_D_MATRIX_SMOKE_LEDGER_CONTRACT,
            "schema_version": 1,
            "campaign_id": str(production_graph["campaign_id"]),
            "production_graph_sha256": production_graph["content_hash"],
            "scope_sha256": scope["content_hash"],
            "jobs": normalized,
            "report_job_id": report,
            "terminal_dependency_job_id": normalized[
                STAGE_D_MATRIX_SMOKE_TERMINAL_NODE
            ],
            "submitted_node_count": len(normalized),
            "scientific_underperformance_cancels_jobs": False,
            "source": _source(production_graph),
        }
    )


def publish_stage_d_matrix_smoke_scope(
    *, campaign_root: str | Path, production_graph: Mapping[str, Any]
) -> dict[str, Any]:
    artifact = build_stage_d_matrix_smoke_scope(
        production_graph=production_graph
    )
    path = (
        Path(campaign_root)
        / "job_ledgers"
        / "stage_d_matrix_smoke_scope.json"
    )
    return {
        "artifact": artifact,
        "path": str(path),
        "publication": write_immutable_json(path, artifact),
    }


__all__ = [
    "STAGE_D_MATRIX_SMOKE_COUNTS",
    "STAGE_D_MATRIX_SMOKE_LEDGER_CONTRACT",
    "STAGE_D_MATRIX_SMOKE_REPORT_CONTRACT",
    "STAGE_D_MATRIX_SMOKE_SCOPE_CONTRACT",
    "STAGE_D_MATRIX_SMOKE_STAGES",
    "STAGE_D_MATRIX_SMOKE_TERMINAL_NODE",
    "assert_finite_json",
    "build_stage_d_matrix_smoke_ledger",
    "build_stage_d_matrix_smoke_scope",
    "publish_stage_d_matrix_smoke_scope",
    "stage_d_matrix_smoke_node_ids",
    "summarize_stage_d_matrix",
    "validate_stage_d_matrix_smoke_scope",
]
