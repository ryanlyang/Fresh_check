"""Contracts and math for relation-matched supplemental KD controls."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    bind_source,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
)
from .expert_model import expert_relation_family
from .supplemental_offline_fusion import SUPPLEMENTAL_BANK_RESULT_CONTRACT


SPECIALIST_KD_PLAN_CONTRACT = "retb_specialist_kd_plan_v1"
SPECIALIST_TEACHER_CONTRACT = "retb_specialist_teacher_v1"
SPECIALIST_STUDENT_CONTRACT = "retb_specialist_kd_student_v1"
SPECIALIST_REPORT_CONTRACT = "retb_specialist_kd_report_v1"

SPECIALIST_EXPERTS = ("BASE4", "PT", "TRACK", "REGION")
NEW_SPECIALIST_TEACHERS = ("PT", "TRACK", "REGION")
SPECIALIST_CONDITIONS = ("MATCHED_KD", "HYBRID_KD")
STUDENT_COORDINATES = tuple(
    (condition, expert)
    for condition in SPECIALIST_CONDITIONS
    for expert in SPECIALIST_EXPERTS
)
TEMPERATURE = 2.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def specialist_teacher_configuration(expert: str) -> dict[str, Any]:
    if expert not in SPECIALIST_EXPERTS:
        raise ValueError("unknown specialist teacher")
    return {
        "expert_id": expert,
        "relation_family": expert_relation_family(expert),
        "all_particle_fields": True,
        "base4_present": True,
        "shape_id": "S8_128",
        "token_count": 8,
        "token_dimension": 128,
        "topology": "B_CONCAT",
        "tokenizer_mode": "TOK_WEAVER_CLASS",
        "loss_id": "ELOSS_CE",
        "initialization": "INIT_SCRATCH",
        "learning_rate": 1.0e-3,
        "particle_dropout": 0.0,
        "measurement_embedding": False,
        "ordinary_weaver_head": True,
        "summary_token_bottleneck_present": False,
        "token_target_eligible": False,
    }


def specialist_student_configuration(
    expert: str, condition: str
) -> dict[str, Any]:
    if expert not in SPECIALIST_EXPERTS:
        raise ValueError("unknown specialist student")
    if condition not in SPECIALIST_CONDITIONS:
        raise ValueError("unknown specialist KD condition")
    return {
        "expert_id": expert,
        "relation_family": expert_relation_family(expert),
        "all_particle_fields": True,
        "base4_present": True,
        "shape_id": "S8_128",
        "token_count": 8,
        "token_dimension": 128,
        "topology": "B_CONCAT",
        "tokenizer_mode": "TOK_CANONICAL",
        "loss_id": condition,
        "initialization": "INIT_SCRATCH",
        "learning_rate": 1.0e-3,
        "particle_dropout": 0.0,
        "measurement_embedding": False,
        "ordinary_weaver_head": False,
        "summary_token_bottleneck_present": True,
    }


def specialist_kd_objective(
    logits: Any,
    labels: Any,
    *,
    condition: str,
    common_teacher_logits: Any,
    specialist_teacher_logits: Any,
) -> tuple[Any, dict[str, Any]]:
    """Exact matched/hybrid loss with globally frozen weights."""

    if condition not in SPECIALIST_CONDITIONS:
        raise ValueError("unknown specialist KD objective")
    import torch

    expected = tuple(logits.shape)
    if (
        logits.ndim != 2
        or int(logits.shape[1]) != 10
        or tuple(common_teacher_logits.shape) != expected
        or tuple(specialist_teacher_logits.shape) != expected
        or tuple(labels.shape) != (int(logits.shape[0]),)
    ):
        raise ValueError("specialist KD tensor shapes differ")
    ce = torch.nn.functional.cross_entropy(logits, labels.long())

    def kd(target: Any) -> Any:
        probability = torch.softmax(target.detach() / TEMPERATURE, dim=-1)
        return torch.nn.functional.kl_div(
            torch.log_softmax(logits / TEMPERATURE, dim=-1),
            probability,
            reduction="batchmean",
        ) * (TEMPERATURE**2)

    common = kd(common_teacher_logits)
    specialist = kd(specialist_teacher_logits)
    if condition == "MATCHED_KD":
        total = 0.25 * ce + specialist
        weights = {"common": 0.0, "specialist": 1.0}
    else:
        total = 0.25 * ce + 0.5 * common + 0.5 * specialist
        weights = {"common": 0.5, "specialist": 0.5}
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("specialist KD objective is nonfinite")
    return total, {
        "cross_entropy": ce.detach(),
        "common_kd": common.detach(),
        "specialist_kd": specialist.detach(),
        "total": total.detach(),
        "weights": weights,
    }


def _json_record(path: Path) -> dict[str, Any]:
    payload = load_hashed_json(path)
    return {
        "path": str(path.resolve()),
        "content_hash": payload["content_hash"],
        "file_sha256": file_sha256(path),
        "source": payload.get("source"),
    }


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"specialist KD parent is absent or unsafe: {path}")
    return {"path": str(path.resolve()), "file_sha256": file_sha256(path)}


def build_specialist_kd_plan(
    *,
    parent_root: str | Path,
    common_fusion_root: str | Path,
    supplemental_id: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    parent = Path(parent_root).resolve()
    fusion = Path(common_fusion_root).resolve()
    campaign = load_hashed_json(parent / "campaign_spec.json")
    if campaign.get("campaign_profile") != "production_500k_scale3m":
        raise ValueError("specialist KD requires a production RETB parent")
    selection = load_hashed_json(
        parent / "selection/stage_a_strongest_teacher.json"
    )
    if selection.get("selected_teacher") != "O_FULLREL":
        raise ValueError("specialist KD common teacher differs")
    common_result_path = fusion / "runs/fusion_banks/KD4/MEAN_LOGITS/result.json"
    ce_result_path = fusion / "runs/fusion_banks/CE4/MEAN_LOGITS/result.json"
    common_result = load_hashed_json(
        common_result_path, expected_contract=SUPPLEMENTAL_BANK_RESULT_CONTRACT
    )
    ce_result = load_hashed_json(
        ce_result_path, expected_contract=SUPPLEMENTAL_BANK_RESULT_CONTRACT
    )
    common_manifest = load_hashed_json(
        parent / "inputs/teacher_logits/ELOSS_KD_DOMINANT.json"
    )
    base4_manifest = load_hashed_json(
        parent / "inputs/teacher_logits/ELOSS_BASE.json"
    )
    if common_manifest.get("teacher_fields") != ["SELECTED_STRONGEST"]:
        raise ValueError("specialist KD common teacher cache differs")
    if base4_manifest.get("teacher_fields") != ["O_BASE"]:
        raise ValueError("specialist KD BASE4 teacher cache differs")
    for bank, result in (("KD4", common_result), ("CE4", ce_result)):
        if (
            result.get("bank_id") != bank
            or result.get("variant") != "MEAN_LOGITS"
            or result.get("fixed_budget_completed") is not True
            or result.get("performance_based_termination") is not False
        ):
            raise ValueError(f"specialist KD {bank} control differs")
    artifacts = {
        "campaign_spec": _json_record(parent / "campaign_spec.json"),
        "step3_bundle": _json_record(
            parent / "registry/retb_step3_architecture_bundle.json"
        ),
        "expert_loss_registry": _json_record(
            parent / "registry/retb_expert_losses.json"
        ),
        "teacher_selection": _json_record(
            parent / "selection/stage_a_strongest_teacher.json"
        ),
        "common_teacher_registration": _json_record(
            parent
            / "runs/stage_a/offline_controls/O_FULLREL/seed_101/checkpoint_registration.json"
        ),
        "common_teacher_checkpoint": _file_record(
            parent
            / "runs/stage_a/offline_controls/SELECTED_STRONGEST/seed_101/best_model_val.pt"
        ),
        "base4_teacher_registration": _json_record(
            parent
            / "runs/stage_a/offline_controls/O_BASE/seed_101/checkpoint_registration.json"
        ),
        "base4_teacher_checkpoint": _file_record(
            parent
            / "runs/stage_a/offline_controls/O_BASE/seed_101/best_model_val.pt"
        ),
        "common_teacher_manifest": _json_record(
            parent / "inputs/teacher_logits/ELOSS_KD_DOMINANT.json"
        ),
        "common_teacher_train": _file_record(
            parent / "inputs/teacher_logits/ELOSS_KD_DOMINANT/model_train.npz"
        ),
        "common_teacher_val_stop": _file_record(
            parent / "inputs/teacher_logits/ELOSS_KD_DOMINANT/val_stop.npz"
        ),
        "base4_teacher_manifest": _json_record(
            parent / "inputs/teacher_logits/ELOSS_BASE.json"
        ),
        "base4_teacher_train": _file_record(
            parent / "inputs/teacher_logits/ELOSS_BASE/model_train.npz"
        ),
        "base4_teacher_val_stop": _file_record(
            parent / "inputs/teacher_logits/ELOSS_BASE/val_stop.npz"
        ),
        "val_design_manifest": _json_record(
            parent / "inputs/offline/val_design/offline_input_manifest.json"
        ),
        "val_design_inputs": _file_record(
            parent / "inputs/offline/val_design/offline_inputs.npz"
        ),
        "relation_normalization": _json_record(
            parent / "inputs/normalization/offline_500k/relation.json"
        ),
        "region_normalization": _json_record(
            parent / "inputs/normalization/offline_500k/region.json"
        ),
        "common_kd4_result": _json_record(common_result_path),
        "common_kd4_predictions": _file_record(
            common_result_path.parent / "val_design_predictions.npz"
        ),
        "ce4_result": _json_record(ce_result_path),
        "ce4_predictions": _file_record(
            ce_result_path.parent / "val_design_predictions.npz"
        ),
        **{
            f"region_tree_{split}_manifest": _json_record(
                parent
                / "inputs/region_tree/offline"
                / f"{split}_exclusive_ca_v1/manifest.json"
            )
            for split in ("model_train", "val_stop", "val_design")
        },
    }
    plan = with_content_hash(
        {
            "contract": SPECIALIST_KD_PLAN_CONTRACT,
            "schema_version": 1,
            "supplemental_id": str(supplemental_id),
            "parent_campaign_root": str(parent),
            "common_fusion_root": str(fusion),
            "parent_campaign_spec_sha256": campaign["content_hash"],
            "parent_campaign_source": campaign.get("source"),
            "parent_artifacts": artifacts,
            "experts": list(SPECIALIST_EXPERTS),
            "new_teacher_experts": list(NEW_SPECIALIST_TEACHERS),
            "conditions": list(SPECIALIST_CONDITIONS),
            "student_coordinates": [
                {"condition": condition, "expert": expert, "seed": 101}
                for condition, expert in STUDENT_COORDINATES
            ],
            "teacher_configurations": {
                expert: specialist_teacher_configuration(expert)
                for expert in SPECIALIST_EXPERTS
            },
            "student_configurations": {
                f"{condition}:{expert}": specialist_student_configuration(
                    expert, condition
                )
                for condition, expert in STUDENT_COORDINATES
            },
            "objective": {
                "temperature": TEMPERATURE,
                "cross_entropy_weight": 0.25,
                "MATCHED_KD": {
                    "common_teacher_weight": 0.0,
                    "specialist_teacher_weight": 1.0,
                },
                "HYBRID_KD": {
                    "common_teacher_weight": 0.5,
                    "specialist_teacher_weight": 0.5,
                },
            },
            "training_protocol": {
                "maximum_epochs": 40,
                "microbatch_size": 64,
                "gradient_accumulation_steps": 2,
                "effective_batch_size": 128,
                "checkpoint_selection": "val_stop",
                "comparison_split": "val_design",
                "early_stopping": False,
                "performance_based_termination": False,
            },
            "fusion": {
                "method": "MEAN_LOGITS",
                "expert_order": list(SPECIALIST_EXPERTS),
                "learned_parameters": 0,
            },
            "existing_controls": {
                "COMMON_KD4": common_result["content_hash"],
                "CE4": ce_result["content_hash"],
            },
            "final_test_access": False,
            "scientific_underperformance_blocks_execution": False,
        }
    )
    return bind_source(plan, source_snapshot=source_snapshot)


def validate_specialist_kd_plan(
    payload: Mapping[str, Any], *, check_parent_bytes: bool = True
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SPECIALIST_KD_PLAN_CONTRACT
    )
    if (
        int(payload.get("schema_version", -1)) != 1
        or payload.get("experts") != list(SPECIALIST_EXPERTS)
        or payload.get("new_teacher_experts")
        != list(NEW_SPECIALIST_TEACHERS)
        or payload.get("conditions") != list(SPECIALIST_CONDITIONS)
        or payload.get("student_coordinates")
        != [
            {"condition": condition, "expert": expert, "seed": 101}
            for condition, expert in STUDENT_COORDINATES
        ]
        or payload.get("final_test_access") is not False
        or payload.get("scientific_underperformance_blocks_execution") is not False
        or payload.get("teacher_configurations")
        != {
            expert: specialist_teacher_configuration(expert)
            for expert in SPECIALIST_EXPERTS
        }
        or payload.get("student_configurations")
        != {
            f"{condition}:{expert}": specialist_student_configuration(
                expert, condition
            )
            for condition, expert in STUDENT_COORDINATES
        }
        or payload.get("training_protocol")
        != {
            "maximum_epochs": 40,
            "microbatch_size": 64,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 128,
            "checkpoint_selection": "val_stop",
            "comparison_split": "val_design",
            "early_stopping": False,
            "performance_based_termination": False,
        }
        or payload.get("fusion")
        != {
            "method": "MEAN_LOGITS",
            "expert_order": list(SPECIALIST_EXPERTS),
            "learned_parameters": 0,
        }
    ):
        raise ValueError("specialist KD plan semantics differ")
    if payload.get("objective") != {
        "temperature": TEMPERATURE,
        "cross_entropy_weight": 0.25,
        "MATCHED_KD": {
            "common_teacher_weight": 0.0,
            "specialist_teacher_weight": 1.0,
        },
        "HYBRID_KD": {
            "common_teacher_weight": 0.5,
            "specialist_teacher_weight": 0.5,
        },
    }:
        raise ValueError("specialist KD objective differs")
    if check_parent_bytes:
        for name, row in payload["parent_artifacts"].items():
            path = Path(row["path"])
            if file_sha256(path) != row["file_sha256"]:
                raise ValueError(f"specialist KD parent {name} bytes drifted")
            if "content_hash" in row:
                artifact = load_hashed_json(path)
                if (
                    artifact["content_hash"] != row["content_hash"]
                    or artifact.get("source") != row.get("source")
                ):
                    raise ValueError(
                        f"specialist KD parent {name} lineage drifted"
                    )
    return digest


def pairwise_diversity(
    logits_by_expert: Mapping[str, Any], labels: Any
) -> list[dict[str, Any]]:
    import numpy as np

    names = list(SPECIALIST_EXPERTS)
    truth = np.asarray(labels, dtype=np.int64)
    if truth.ndim != 1 or truth.size <= 0:
        raise ValueError("specialist KD diversity labels differ")
    rows = []
    for left_index, left in enumerate(names):
        left_logits = np.asarray(logits_by_expert[left], dtype=np.float64)
        if left_logits.ndim != 2 or left_logits.shape[0] != truth.shape[0]:
            raise ValueError("specialist KD diversity logits differ")
        left_prediction = left_logits.argmax(axis=1)
        left_correct = left_prediction == truth
        left_centered = left_logits - left_logits.mean(axis=1, keepdims=True)
        for right in names[left_index + 1 :]:
            right_logits = np.asarray(logits_by_expert[right], dtype=np.float64)
            if right_logits.shape != left_logits.shape:
                raise ValueError("specialist KD diversity member shapes differ")
            right_prediction = right_logits.argmax(axis=1)
            right_correct = right_prediction == truth
            right_centered = right_logits - right_logits.mean(axis=1, keepdims=True)
            correlation = float(
                np.corrcoef(left_centered.ravel(), right_centered.ravel())[0, 1]
            )
            values = {
                "left": left,
                "right": right,
                "prediction_disagreement": float(
                    np.mean(left_prediction != right_prediction)
                ),
                "correctness_disagreement": float(np.mean(left_correct != right_correct)),
                "double_fault": float(np.mean(~left_correct & ~right_correct)),
                "centered_logit_correlation": correlation,
            }
            if not all(
                math.isfinite(float(value))
                for name, value in values.items()
                if name not in {"left", "right"}
            ):
                raise FloatingPointError("specialist KD diversity is nonfinite")
            rows.append(values)
    return rows


__all__ = [
    "NEW_SPECIALIST_TEACHERS",
    "SPECIALIST_CONDITIONS",
    "SPECIALIST_EXPERTS",
    "SPECIALIST_KD_PLAN_CONTRACT",
    "SPECIALIST_REPORT_CONTRACT",
    "SPECIALIST_STUDENT_CONTRACT",
    "SPECIALIST_TEACHER_CONTRACT",
    "STUDENT_COORDINATES",
    "build_specialist_kd_plan",
    "file_sha256",
    "pairwise_diversity",
    "specialist_kd_objective",
    "specialist_student_configuration",
    "specialist_teacher_configuration",
    "validate_specialist_kd_plan",
]
