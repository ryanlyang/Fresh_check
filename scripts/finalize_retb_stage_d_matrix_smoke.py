#!/usr/bin/env python3
"""Fail-closed aggregation of the complete RETB Stage-D miniature matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_controls import (  # noqa: E402
    HLT_CONTROL_REGISTRATION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    HLT_EXPERT_REGISTRATION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.native_fusion import (  # noqa: E402
    NATIVE_FUSION_REGISTRATION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
    TASK_MANIFEST_CONTRACT,
    task_manifest_path_for_graph,
    validate_production_graph,
    validate_task_manifest_for_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_d_matrix_smoke import (  # noqa: E402
    STAGE_D_MATRIX_SMOKE_REPORT_CONTRACT,
    STAGE_D_MATRIX_SMOKE_SCOPE_CONTRACT,
    STAGE_D_MATRIX_SMOKE_STAGES,
    assert_finite_json,
    summarize_stage_d_matrix,
    validate_stage_d_matrix_smoke_scope,
)
from teacher_logit_reco.relation_expert_token_bridge.task_completion import (  # noqa: E402
    TASK_MANIFEST_COMPLETION_CONTRACT,
    task_manifest_completion_path,
    validate_task_manifest_completion,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _registration_budget(payload: dict[str, Any]) -> str | None:
    contract = payload.get("contract")
    if contract in {
        HLT_EXPERT_REGISTRATION_CONTRACT,
        HLT_CONTROL_REGISTRATION_CONTRACT,
    }:
        if (
            int(payload.get("epochs_completed", -1)) != 2
            or payload.get("fixed_epoch_budget_completed") is not True
            or payload.get("performance_based_termination") is not False
        ):
            raise ValueError("Stage-D deep-sentinel epoch budget differs")
        if payload.get("evaluation_realization_policy") != "R_FIXED":
            raise ValueError("Stage-D evaluation realization policy differs")
        return "native_expert_or_control"
    if contract == NATIVE_FUSION_REGISTRATION_CONTRACT:
        expected_epochs = (
            0
            if payload.get("variant")
            in {"HF_LOGIT_MEAN", "HF_7X_UNBIASED_LOGIT_MEAN"}
            else 2
        )
        if (
            int(payload.get("epochs_completed", -1)) != expected_epochs
            or payload.get("fixed_epoch_budget_completed") is not True
            or payload.get("performance_based_termination") is not False
            or payload.get("realization_policy") != "R_MULTI"
        ):
            raise ValueError("Stage-D native-fusion miniature budget differs")
        return "native_fusion"
    return None


def _assert_finite_loaded(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"nonfinite checkpoint tensor at {path}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(
            np.isfinite(value).all()
        ):
            raise ValueError(f"nonfinite array at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_loaded(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_loaded(child, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"nonfinite scalar at {path}")


def _reload_semantic_output(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_content_hash(payload)
        assert_finite_json(payload, path=str(path))
        return "json"
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _assert_finite_loaded(payload, path=str(path))
        return "checkpoint"
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            for name in payload.files:
                _assert_finite_loaded(
                    np.asarray(payload[name]), path=f"{path}:{name}"
                )
        return "npz"
    raise ValueError(f"unsupported Stage-D expected-output type: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    graph = load_hashed_json(
        root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    graph_sha = validate_production_graph(graph)
    scope = load_hashed_json(
        root / "job_ledgers" / "stage_d_matrix_smoke_scope.json",
        expected_contract=STAGE_D_MATRIX_SMOKE_SCOPE_CONTRACT,
    )
    scope_sha = validate_stage_d_matrix_smoke_scope(
        scope, production_graph=graph
    )
    if campaign.get("source") != scope.get("source"):
        raise ValueError("Stage-D matrix smoke campaign source differs")
    plan = load_hashed_json(
        root / "registry" / "retb_static_experiment_plan.json"
    )
    if (
        plan.get("campaign_spec_sha256") != campaign["content_hash"]
        or plan.get("production_graph_sha256") != graph_sha
        or plan.get("source") != campaign.get("source")
        or plan.get("campaign_profile") != "miniature_test"
        or plan.get("miniature_policy")
        != "complete_scientific_matrix_on_miniature_populations"
    ):
        raise ValueError("Stage-D matrix static-plan lineage differs")
    matrix = summarize_stage_d_matrix(plan)

    manifest_completions: dict[str, str] = {}
    manifest_task_counts: dict[str, int] = {}
    json_evidence_count = 0
    checkpoint_reload_count = 0
    npz_reload_count = 0
    deep_sentinel_count = 0
    native_fusion_registration_count = 0
    native_registration_run_ids: set[str] = set()
    native_fusion_registration_run_ids: set[str] = set()
    deep_sentinel_contract_counts: dict[str, int] = {}
    for node in graph["nodes"]:
        node_id = str(node["node_id"])
        if str(node["stage"]) not in STAGE_D_MATRIX_SMOKE_STAGES:
            continue
        if node["array"] is None:
            continue
        manifest_path = task_manifest_path_for_graph(
            graph, node_id=node_id, campaign_root=root
        )
        manifest = load_hashed_json(
            manifest_path, expected_contract=TASK_MANIFEST_CONTRACT
        )
        validate_task_manifest_for_graph(
            manifest,
            production_graph=graph,
            campaign_root=root,
            repo_root=REPO_ROOT,
        )
        completion_path = task_manifest_completion_path(
            root, node_id=node_id
        )
        completion = load_hashed_json(
            completion_path,
            expected_contract=TASK_MANIFEST_COMPLETION_CONTRACT,
        )
        completion_sha = validate_task_manifest_completion(
            completion,
            campaign_root=root,
            campaign=campaign,
            task_manifest=manifest,
        )
        manifest_completions[node_id] = completion_sha
        manifest_task_counts[node_id] = int(manifest["task_count"])
        for row in completion["rows"]:
            for raw_path in row["output_hashes"]:
                path = Path(raw_path)
                output_kind = _reload_semantic_output(path)
                if output_kind == "checkpoint":
                    checkpoint_reload_count += 1
                elif output_kind == "npz":
                    npz_reload_count += 1
                else:
                    json_evidence_count += 1
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    registration_kind = _registration_budget(payload)
                    if registration_kind == "native_expert_or_control":
                        deep_sentinel_count += 1
                        run_id = str(payload["run_id"])
                        if run_id in native_registration_run_ids:
                            raise ValueError(
                                "Stage-D native registration is duplicated"
                            )
                        native_registration_run_ids.add(run_id)
                        contract = str(payload["contract"])
                        deep_sentinel_contract_counts[contract] = (
                            deep_sentinel_contract_counts.get(contract, 0) + 1
                        )
                    elif registration_kind == "native_fusion":
                        native_fusion_registration_count += 1
                        run_id = str(payload["run_id"])
                        if run_id in native_fusion_registration_run_ids:
                            raise ValueError(
                                "Stage-D fusion registration is duplicated"
                            )
                        native_fusion_registration_run_ids.add(run_id)

    for name, count in matrix["static_matrix_counts"].items():
        if manifest_task_counts.get(name) != int(count):
            raise ValueError(
                f"Stage-D matrix manifest count differs for {name}"
            )
    if deep_sentinel_count != 541:
        raise ValueError("Stage-D deep-sentinel registration coverage differs")
    if deep_sentinel_contract_counts != {
        HLT_CONTROL_REGISTRATION_CONTRACT: 2,
        HLT_EXPERT_REGISTRATION_CONTRACT: 539,
    }:
        raise ValueError("Stage-D native registration contract coverage differs")
    if native_fusion_registration_count != 30:
        raise ValueError("Stage-D native-fusion registration coverage differs")
    expected_native_run_ids = {
        str(row["run_id"])
        for row in plan["groups"]["native_hlt_expert_training"]
    }
    expected_fusion_run_ids = {
        str(row["run_id"])
        for row in plan["groups"]["native_hlt_fusion_training"]
    }
    if native_registration_run_ids != expected_native_run_ids:
        raise ValueError("Stage-D native registration identities differ")
    if native_fusion_registration_run_ids != expected_fusion_run_ids:
        raise ValueError("Stage-D fusion registration identities differ")
    if checkpoint_reload_count <= 0:
        raise ValueError("Stage-D matrix reloaded no checkpoints")
    expected_array_nodes = {
        str(node["node_id"])
        for node in graph["nodes"]
        if str(node["stage"]) in STAGE_D_MATRIX_SMOKE_STAGES
        and node["array"] is not None
    }
    if set(manifest_completions) != expected_array_nodes:
        raise ValueError("Stage-D matrix completion-node coverage differs")

    report = with_content_hash(
        {
            "contract": STAGE_D_MATRIX_SMOKE_REPORT_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "production_graph_sha256": graph_sha,
            "scope_sha256": scope_sha,
            "static_experiment_plan_sha256": plan["content_hash"],
            "matrix_coverage": matrix,
            "task_manifest_completion_hashes": dict(
                sorted(manifest_completions.items())
            ),
            "task_manifest_counts": dict(sorted(manifest_task_counts.items())),
            "completed_array_node_count": len(manifest_completions),
            "authenticated_json_evidence_count": json_evidence_count,
            "semantically_reloaded_checkpoint_count": checkpoint_reload_count,
            "semantically_reloaded_npz_count": npz_reload_count,
            "deep_two_epoch_native_registration_count": deep_sentinel_count,
            "native_fusion_registration_count": (
                native_fusion_registration_count
            ),
            "deep_sentinel_contract_counts": dict(
                sorted(deep_sentinel_contract_counts.items())
            ),
            "all_541_native_hlt_configurations_completed": True,
            "all_30_native_fusion_configurations_completed": True,
            "all_json_evidence_finite": True,
            "all_expected_outputs_reloaded_and_rehashed": True,
            "scientific_metric_used_as_gate": False,
            "scientific_performance_claimed": False,
            "production_evidence_eligible": False,
            "passed": True,
            "source": campaign["source"],
        }
    )
    output = args.output or (
        root / "evaluations" / "stage_d_matrix_smoke" / "report.json"
    )
    publication = write_immutable_json(output, report)
    print(
        json.dumps(
            {"report": report, "publication": publication},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
