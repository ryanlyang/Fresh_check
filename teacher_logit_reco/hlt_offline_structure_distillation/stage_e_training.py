"""Fixed-budget Stage-E training and HLT-only export validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .baselines import HOSDTrainingProtocol
from .contracts import (
    FEEDBACK_CHECKPOINT_CONTRACT,
    FEEDBACK_COMPLETION_CONTRACT,
    FEEDBACK_RESULT_CONTRACT,
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
from .capacity import exact_trainable_parameter_count, feedback_model_flop_ledger
from .stage_d_training import evaluate_auxiliary, train_stage_d_auxiliary

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


FEEDBACK_EXPORT_CONTRACT = "hosd_feedback_deployable_export_v1"


def train_stage_e_feedback(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    design_select_loader: Any,
    output_dir: str | Path,
    row: Mapping[str, Any],
    component_group_ids: Sequence[str],
    stage_e_plan_sha256: str,
    campaign_spec_sha256: str,
    lineage_hashes: Mapping[str, str],
    protocol: HOSDTrainingProtocol,
    source: Mapping[str, Any],
    deployed_analytical_flops: float,
    deployed_parameter_count: int | None = None,
    device: str | Any = "cpu",
    resume: bool = True,
    training_gpu_hours_override: float | None = None,
) -> dict[str, Any]:
    if row.get("control") in {
        "SHUFFLED_PREDICTION",
        "SHUFFLED",
        "ORACLE_SUB",
        "ORACLE_TRAINED",
        "EXACT_HLT",
    } and not bool(getattr(train_loader.dataset, "feedback_intervention_ready", False)):
        raise ValueError(
            "Stage-E intervention row requires its identity-bound feedback source"
        )
    return train_stage_d_auxiliary(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_stop_loader,
        design_select_loader=design_select_loader,
        output_dir=output_dir,
        row=row,
        component_group_ids=component_group_ids,
        stage_d_plan_sha256=stage_e_plan_sha256,
        campaign_spec_sha256=campaign_spec_sha256,
        lineage_hashes=lineage_hashes,
        protocol=protocol,
        source=source,
        deployed_analytical_flops=deployed_analytical_flops,
        deployed_parameter_count=deployed_parameter_count,
        device=device,
        resume=resume,
        training_gpu_hours_override=training_gpu_hours_override,
        checkpoint_contract=FEEDBACK_CHECKPOINT_CONTRACT,
        completion_contract=FEEDBACK_COMPLETION_CONTRACT,
        prediction_contract=FEEDBACK_RESULT_CONTRACT,
        prediction_schema_version=3,
        completion_schema_version=3,
        plan_hash_field="stage_e_plan_sha256",
        stage_label="Stage-E",
        completion_filename="feedback_completion.json",
        curves_contract="hosd_feedback_training_curves_v1",
    )


def export_feedback_model(
    *,
    model: Any,
    checkpoint_path: str | Path,
    output_path: str | Path,
    row: Mapping[str, Any],
    stage_e_plan_sha256: str,
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish an HLT-only state dict; privileged target readers are never stored."""

    if torch is None:
        raise RuntimeError("PyTorch is required for HOSD export")
    if not bool(row.get("deployable")):
        raise ValueError("oracle feedback rows cannot be exported")
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("contract") != FEEDBACK_CHECKPOINT_CONTRACT
        or checkpoint.get("row_id") != row["row_id"]
        or checkpoint.get("stage_e_plan_sha256") != stage_e_plan_sha256
        or checkpoint.get("campaign_spec_sha256") != campaign_spec_sha256
        or checkpoint.get("source") != dict(source)
    ):
        raise ValueError("feedback export checkpoint lineage differs")
    state = dict(checkpoint["model_state_dict"])
    forbidden_fragments = (
        "oracle",
        "offline_target",
        "target_cache",
        "label",
    )
    if any(
        any(fragment in name.lower() for fragment in forbidden_fragments)
        for name in state
    ):
        raise ValueError("feedback export state contains a forbidden dependency")
    model.load_state_dict(state, strict=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": FEEDBACK_EXPORT_CONTRACT,
        "schema_version": 1,
        "row_id": row["row_id"],
        "target_id": row["target_id"],
        "interface": row["interface"],
        "gradient_path": row["gradient_path"],
        "source": dict(source),
        "stage_e_plan_sha256": require_sha256(
            stage_e_plan_sha256, name="stage_e_plan_sha256"
        ),
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="campaign_spec_sha256"
        ),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "runtime_inputs": [
            "hlt_features",
            "hlt_points",
            "hlt_lorentz_vectors",
            "hlt_mask",
        ],
        "retained_modules": [
            "hlt_classifier",
            "hlt_side_predictor",
            "feedback_consumer",
        ],
        "forbidden_runtime_dependencies": [],
        "state_dict": state,
    }
    torch.save(payload, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = with_content_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "state_dict"
        }
        | {
            "export_file": output.name,
            "export_sha256": digest,
            "state_key_count": len(state),
            "hlt_only": True,
        }
    )
    write_immutable_json(output.with_suffix(output.suffix + ".json"), manifest)
    return manifest


def evaluate_posthoc_feedback_control(
    *,
    source_model: Any,
    design_select_loader: Any,
    output_dir: str | Path,
    control_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    component_group_ids: Sequence[str],
    source_checkpoint_path: str | Path,
    source_completion: Mapping[str, Any],
    stage_e_plan_sha256: str,
    campaign_spec_sha256: str,
    lineage_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    device: str | Any,
) -> dict[str, Any]:
    """Attest evaluation-only oracle or wrong-event substitution."""

    if control_row.get("control") not in {
        "ORACLE_SUB",
        "SHUFFLED_PREDICTION",
        "SHUFFLED",
    }:
        raise ValueError("feedback control is not post-hoc")
    checkpoint = Path(source_checkpoint_path)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (
        source_completion.get("row_id") != source_row["row_id"]
        or source_completion.get("checkpoint_sha256") != checkpoint_sha
        or source_completion.get("stage_e_plan_sha256")
        != stage_e_plan_sha256
        or source_completion.get("source") != dict(source)
    ):
        raise ValueError("post-hoc feedback source completion lineage differs")
    if control_row["control"] == "ORACLE_SUB":
        source_model.allow_oracle = True
    evaluation = evaluate_auxiliary(
        source_model,
        design_select_loader,
        row=control_row,
        component_group_ids=component_group_ids,
        split="design_select",
        device=torch.device(device),
    )
    ledger = feedback_model_flop_ledger(source_model)
    parameters = exact_trainable_parameter_count(source_model)
    root = Path(output_dir)
    result = with_content_hash(
        {
            "contract": FEEDBACK_RESULT_CONTRACT,
            "schema_version": 3,
            "source": dict(source),
            "stage_e_plan_sha256": require_sha256(
                stage_e_plan_sha256, name="stage_e_plan_sha256"
            ),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "row_id": control_row["row_id"],
            "target_id": control_row["target_id"],
            "parameterization": control_row["parameterization"],
            "auxiliary_weight": control_row["auxiliary_weight"],
            "row_kind": control_row["row_kind"],
            "interface": control_row["interface"],
            "gradient_path": control_row["gradient_path"],
            "control": control_row["control"],
            "deployable": control_row["control"] != "ORACLE_SUB",
            "selection_eligible": False,
            "checkpoint_sha256": checkpoint_sha,
            "source_scientific_row_id": source_row["row_id"],
            "source_feedback_frozen": True,
            "posthoc_substitution_only": True,
            "design_select": evaluation,
            "deployed_analytical_flops": float(
                ledger["deployed_total_flops"]
            ),
            "deployed_parameter_count": parameters,
            "training_gpu_hours": 0.0,
            "pipeline_seed": int(control_row["pipeline_seed"]),
            "encoder_component_seed": int(
                source_row["encoder_component_seed"]
            ),
            "task_component_seed": int(
                source_row["feedback_component_seed"]
            ),
            "target_error_cross_family_tie_breaker": False,
        }
    )
    write_immutable_json(root / "design_select_result.json", result)
    checked_lineage = {
        key: require_sha256(value, name=f"lineage.{key}")
        for key, value in sorted(lineage_hashes.items())
    }
    checked_lineage["source_feedback_completion"] = source_completion[
        "content_hash"
    ]
    completion = with_content_hash(
        {
            "contract": FEEDBACK_COMPLETION_CONTRACT,
            "schema_version": 3,
            "source": dict(source),
            "row_id": control_row["row_id"],
            "target_id": control_row["target_id"],
            "row_kind": control_row["row_kind"],
            "interface": control_row["interface"],
            "gradient_path": control_row["gradient_path"],
            "control": control_row["control"],
            "deployable": control_row["control"] != "ORACLE_SUB",
            "stage_e_plan_sha256": stage_e_plan_sha256,
            "campaign_spec_sha256": campaign_spec_sha256,
            "lineage_hashes": checked_lineage,
            "pipeline_seed": int(control_row["pipeline_seed"]),
            "encoder_component_seed": int(
                source_row["encoder_component_seed"]
            ),
            "task_component_seed": int(
                source_row["feedback_component_seed"]
            ),
            "checkpoint_file": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "source_scientific_row_id": source_row["row_id"],
            "selected_epoch": int(source_completion["selected_epoch"]),
            "selected_val_stop": dict(
                source_completion["selected_val_stop"]
            ),
            "design_select_result_sha256": result["content_hash"],
            "epochs_completed": 0,
            "optimizer_updates_completed": 0,
            "early_stopping": False,
            "performance_based_termination": False,
            "future_rows_cancelled_for_performance": False,
            "classification_isolated": True,
            "auxiliary_head_removed_from_deployed_counts": False,
            "hlt_only_inference": control_row["control"] != "ORACLE_SUB",
            "source_feedback_frozen": True,
            "posthoc_substitution_only": True,
        }
    )
    write_immutable_json(root / "feedback_completion.json", completion)
    return completion


__all__ = [
    "FEEDBACK_EXPORT_CONTRACT",
    "evaluate_posthoc_feedback_control",
    "export_feedback_model",
    "train_stage_e_feedback",
]
