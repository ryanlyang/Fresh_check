"""Frozen F_method recipe replay on the independent-seed group."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jetclass_fresh.fusion import load_prediction_block
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .fusion_campaign import (
    FUSION_CHAMPION_ACCURACY,
    FUSION_FIT_SPLIT,
    FUSION_GROUP_METHOD,
    FUSION_GROUP_SEED,
    FUSION_HEAD_SEEDS,
    FUSION_SELECTION_SPLIT,
    default_fusion_candidate_specs,
    stable_fusion_json_hash,
)
from .fusion_final import _validate_selection_dependencies
from .fusion_late import replay_late_fusion_recipe
from .fusion_metrics import local_residual_field_binary_projection_metrics, local_residual_field_multiclass_metrics
from .fusion_seed_control import sha256_file
from .fusion_selection import (
    FusionCandidateRunConfig,
    _atomic_json,
    _average_head_metrics,
    _representation_trial,
    _validate_candidate_report,
    load_selected_fusion_set,
)


LOCAL_RESIDUAL_FIELD_FUSION_RECIPE_REPLAY_CONTRACT = "local_residual_field_fusion_recipe_replay_v1"


def replay_selected_method_recipe(selected_fusion_json: str | Path) -> dict[str, Any]:
    """Refit the F_method accuracy recipe on F_seed without another search."""

    selected_path = Path(selected_fusion_json).resolve()
    selection = load_selected_fusion_set(selected_path)
    _audit, registry = _validate_selection_dependencies(selected_path, selection)
    source = next(
        row for row in selection["selections"]
        if row["group_id"] == FUSION_GROUP_METHOD and row["champion_role"] == FUSION_CHAMPION_ACCURACY
    )
    candidate_id = source["candidate_id"]
    spec = next(spec for spec in default_fusion_candidate_specs() if spec.candidate_id == candidate_id)
    source_binding = next(
        row for row in selection["selection_bindings"]
        if row["group_id"] == FUSION_GROUP_METHOD and row["champion_role"] == FUSION_CHAMPION_ACCURACY
    )
    source_report = _validate_candidate_report(source_binding["candidate_report_path"])
    replay_root = selected_path.parent / "recipe_replay" / candidate_id
    output_path = selected_path.parent / "recipe_replay" / "recipe_replay.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable recipe replay: {output_path}")
    if spec.family == "late":
        member_a, member_b = ("A0", "A0_seed1")
        def block(member: str, split: str) -> Any:
            return load_prediction_block(registry["members"][member]["prediction_root"], member, split, verify_hash=True)
        result = replay_late_fusion_recipe(
            candidate_id, source["hyperparameters"],
            block(member_a, FUSION_FIT_SPLIT), block(member_b, FUSION_FIT_SPLIT),
            block(member_a, FUSION_SELECTION_SPLIT), block(member_b, FUSION_SELECTION_SPLIT),
        )
        logits = {FUSION_FIT_SPLIT: result.train_logits, FUSION_SELECTION_SPLIT: result.validation_logits}
        labels = {
            split: block(member_a, split).labels for split in (FUSION_FIT_SPLIT, FUSION_SELECTION_SPLIT)
        }
        metrics = {
            split: {
                "multiclass": local_residual_field_multiclass_metrics(logits[split], labels[split], label_names=LABEL_NAMES),
                "binary_projection": local_residual_field_binary_projection_metrics(logits[split], labels[split], label_names=LABEL_NAMES),
            }
            for split in (FUSION_FIT_SPLIT, FUSION_SELECTION_SPLIT)
        }
        replay_parameters = dict(result.parameters)
        head_artifacts: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {}
    else:
        config = FusionCandidateRunConfig(
            campaign_id=selection["campaign_id"], group_id=FUSION_GROUP_SEED,
            candidate_id=candidate_id, output_dir=str(replay_root),
            prediction_sources=selection["prediction_sources_path"],
            source_artifact_audit=selection["source_artifact_audit_path"],
            feature_root=selection["feature_root"], phase="stability",
        )
        head_artifacts = [
            _representation_trial(config, seed=seed, hyperparameters=source["hyperparameters"])
            for seed in FUSION_HEAD_SEEDS
        ]
        metrics, diagnostics = _average_head_metrics(config, head_artifacts)
        replay_parameters = dict(source["hyperparameters"])
    report: dict[str, Any] = {
        "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_RECIPE_REPLAY_CONTRACT,
        "campaign_id": selection["campaign_id"], "created_at": datetime.now(timezone.utc).isoformat(),
        "source_group_id": FUSION_GROUP_METHOD, "replay_group_id": FUSION_GROUP_SEED,
        "candidate_id": candidate_id, "family": spec.family,
        "source_champion_role": FUSION_CHAMPION_ACCURACY,
        "source_selected_fusion_path": str(selected_path),
        "source_selected_fusion_hash": selection["artifact_hash"],
        "source_selected_fusion_artifact_hash": selection["artifact_hash"],
        "source_candidate_report_path": source_binding["candidate_report_path"],
        "source_candidate_report_sha256": sha256_file(source_binding["candidate_report_path"]),
        "source_hyperparameters": source["hyperparameters"],
        "replay_parameters": replay_parameters,
        "hyperparameter_search_performed": False,
        "head_seeds": [row["seed"] for row in head_artifacts],
        "head_artifacts": head_artifacts, "head_diagnostics": diagnostics,
        "head_stability": diagnostics.get("stability"),
        "metrics": metrics, "development_splits": [FUSION_FIT_SPLIT, FUSION_SELECTION_SPLIT],
        "final_test_opened": False,
        "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False, "deployable": True,
    }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    _atomic_json(output_path, report)
    return report


__all__ = ["LOCAL_RESIDUAL_FIELD_FUSION_RECIPE_REPLAY_CONTRACT", "replay_selected_method_recipe"]
