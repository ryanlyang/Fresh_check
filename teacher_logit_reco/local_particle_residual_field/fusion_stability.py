"""Immutable representation-stability work plan derived from screening only."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jetclass_fresh.fusion import load_prediction_block
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .fusion_campaign import FUSION_GROUP_METHOD, FUSION_GROUP_SEED, FusionCampaignConfig, stable_fusion_json_hash
from .fusion_features import require_development_prediction_sources
from .fusion_metrics import local_residual_field_multiclass_metrics
from .fusion_seed_control import sha256_file
from .fusion_selection import _accuracy_choice, _atomic_json, _ranking_multiclass, _rejection_objective, _validate_candidate_report


LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN_CONTRACT = "local_residual_field_fusion_stability_plan_v1"


def write_representation_stability_plan(
    *, campaign_id: str, candidates_root: str | Path, prediction_sources: str | Path,
    source_artifact_audit: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    campaign = FusionCampaignConfig(campaign_id=campaign_id)
    registry = require_development_prediction_sources(prediction_sources, source_artifact_audit=source_artifact_audit)
    representation_ids = [candidate.candidate_id for candidate in campaign.candidates if candidate.family == "representation"]
    screening: dict[str, dict[str, Any]] = {}
    screening_bindings: list[dict[str, Any]] = []
    union: set[str] = set()
    trace: dict[str, Any] = {}
    for group in campaign.groups:
        rows = []
        for candidate_id in representation_ids:
            candidate_spec = next(candidate for candidate in campaign.candidates if candidate.candidate_id == candidate_id)
            path = Path(candidates_root) / group.group_id / candidate_id / "candidate_report.json"
            report = _validate_candidate_report(path)
            if (report["campaign_id"], report["group_id"], report["candidate_id"], report["phase"]) != (
                campaign_id, group.group_id, candidate_id, "screening",
            ):
                raise ValueError(f"representation screening identity mismatch: {path}")
            if report.get("candidate_spec_hash") != stable_fusion_json_hash(candidate_spec.to_dict()):
                raise ValueError(f"candidate registry drift in screening report: {path}")
            if report.get("prediction_sources_hash") != registry["manifest_hash"]:
                raise ValueError(f"screening report uses different prediction sources: {path}")
            rows.append(report)
            screening_bindings.append({
                "group_id": group.group_id, "candidate_id": candidate_id,
                "path": str(path.resolve()), "sha256": sha256_file(path),
                "artifact_hash": report["artifact_hash"],
            })
        screening[group.group_id] = {row["candidate_id"]: row["artifact_hash"] for row in rows}
        raw_rows = [
            load_prediction_block(registry["members"][member]["prediction_root"], member, "stack_val", verify_hash=True)
            for member in group.member_ids
        ]
        raw_accuracy = max(
            local_residual_field_multiclass_metrics(block.logits, block.labels, label_names=LABEL_NAMES)["accuracy"]
            for block in raw_rows
        )
        accuracy, accuracy_trace = _accuracy_choice(rows)
        union.add(accuracy["candidate_id"])
        eligible = [row for row in rows if _ranking_multiclass(row)["accuracy"] >= raw_accuracy - 0.001]
        rejection = min(
            eligible,
            key=lambda row: (_rejection_objective(row), _ranking_multiclass(row)["cross_entropy"],
                             int(row["trainable_parameter_count"]), row["candidate_id"]),
        ) if eligible else None
        if rejection is not None:
            union.add(rejection["candidate_id"])
        trace[group.group_id] = {
            "raw_accuracy_floor_reference": raw_accuracy, "accuracy_candidate": accuracy["candidate_id"],
            "accuracy_trace": accuracy_trace, "rejection_candidate": None if rejection is None else rejection["candidate_id"],
        }
    report: dict[str, Any] = {
        "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN_CONTRACT,
        "campaign_id": campaign_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "required_candidate_ids": sorted(union), "groups": [FUSION_GROUP_METHOD, FUSION_GROUP_SEED],
        "head_seeds": [5101, 5102, 5103], "screening_artifact_hashes": screening, "selection_trace": trace,
        "screening_bindings": screening_bindings,
        "prediction_sources_path": str(Path(prediction_sources).resolve()),
        "prediction_sources_hash": registry["manifest_hash"],
        "source_artifact_audit_path": str(Path(source_artifact_audit).resolve()),
        "source_artifact_audit_hash": registry["source_artifact_audit_hash"], "final_test_opened": False,
    }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    _atomic_json(Path(output_path), report)
    return report


def require_stability_candidate(path: str | Path, *, campaign_id: str, candidate_id: str) -> bool:
    import json
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    stored = unsigned.pop("artifact_hash", None)
    if payload.get("contract") != LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN_CONTRACT or payload.get("ok") is not True:
        raise ValueError("stability plan contract mismatch")
    if stored != stable_fusion_json_hash(unsigned) or payload.get("campaign_id") != campaign_id:
        raise ValueError("stability plan hash or campaign mismatch")
    return candidate_id in payload["required_candidate_ids"]


__all__ = ["LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN_CONTRACT", "write_representation_stability_plan", "require_stability_candidate"]
