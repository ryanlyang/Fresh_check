"""Post-final integrity gate for paired fusion bootstrap artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .fusion_campaign import stable_fusion_json_hash
from .fusion_final import LOCAL_RESIDUAL_FIELD_FUSION_FINAL_EVALUATION_CONTRACT, _read_json
from .fusion_seed_control import sha256_file
from .fusion_selection import FUSION_HEADLINE_SIGNALS, _atomic_json, load_selected_fusion_set


LOCAL_RESIDUAL_FIELD_FUSION_BOOTSTRAP_AUDIT_CONTRACT = "local_residual_field_fusion_bootstrap_audit_v1"


def _require_bootstrap(payload: Mapping[str, Any], *, label: str) -> None:
    if int(payload.get("replicates", 0)) < 1000:
        raise ValueError(f"{label} has fewer than 1000 bootstrap replicates")
    digest = payload.get("sampled_index_hash")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} lacks a sampled-index SHA-256")
    if payload.get("stratified_by_class") is not True:
        raise ValueError(f"{label} is not stratified by class")


def audit_selected_fusion_bootstraps(selected_fusion_json: str | Path) -> dict[str, Any]:
    selection_path = Path(selected_fusion_json).resolve()
    selection = load_selected_fusion_set(selection_path)
    final_root = selection_path.parent.parent / "final_evaluation" / selection["artifact_hash"][:16]
    final_path = final_root / "final_evaluation.json"
    final = _read_json(final_path)
    unsigned = dict(final)
    stored = unsigned.pop("artifact_hash", None)
    if final.get("contract") != LOCAL_RESIDUAL_FIELD_FUSION_FINAL_EVALUATION_CONTRACT or stored != stable_fusion_json_hash(unsigned):
        raise ValueError("final evaluation contract or logical hash mismatch")
    if final.get("selected_fusion_artifact_hash") != selection["artifact_hash"]:
        raise ValueError("final evaluation is not bound to selected_fusion.json")
    privileged = ("uses_true_fields", "uses_offline_particles", "uses_teacher_logits_at_runtime")
    if final.get("runtime_inputs") != "HLT_only" or final.get("deployable") is not True or any(
        final.get(key) is not False for key in privileged
    ):
        raise ValueError("final evaluation is not deployable HLT-only")
    rows: list[dict[str, Any]] = []
    for result in final["selected_results"]:
        if result.get("runtime_inputs") != "HLT_only" or result.get("deployable") is not True or any(
            result.get(key) is not False for key in privileged
        ):
            raise ValueError(f"final fusion row is not deployable HLT-only: {result.get('run_id')}")
        for key in ("run_id", "group_id", "candidate_id"):
            if not result.get(key):
                raise ValueError(f"final fusion row lacks {key}")
        multiclass = result["paired_bootstrap_vs_A0"]
        _require_bootstrap(multiclass, label=f"{result['run_id']}/multiclass")
        binary = result["paired_binary_bootstrap_vs_A0"]
        if set(binary) != set(FUSION_HEADLINE_SIGNALS):
            raise ValueError(f"{result['run_id']} lacks the locked headline binary bootstrap set")
        for signal, payload in binary.items():
            _require_bootstrap(payload, label=f"{result['run_id']}/QCD_vs_{signal}")
        rows.append({
            "run_id": result["run_id"], "group_id": result["group_id"], "candidate_id": result["candidate_id"],
            "multiclass_replicates": multiclass["replicates"], "multiclass_sampled_index_hash": multiclass["sampled_index_hash"],
            "binary_signals": sorted(binary), "minimum_binary_replicates": min(int(row["replicates"]) for row in binary.values()),
        })
    report: dict[str, Any] = {
        "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_BOOTSTRAP_AUDIT_CONTRACT,
        "campaign_id": selection["campaign_id"], "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_fusion_artifact_hash": selection["artifact_hash"], "final_evaluation_path": str(final_path.resolve()),
        "final_evaluation_sha256": sha256_file(final_path), "rows": rows,
        "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False, "final_test_reopened": False, "deployable": True,
    }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    _atomic_json(final_root / "bootstrap_audit.json", report)
    return report


__all__ = ["LOCAL_RESIDUAL_FIELD_FUSION_BOOTSTRAP_AUDIT_CONTRACT", "audit_selected_fusion_bootstraps"]
