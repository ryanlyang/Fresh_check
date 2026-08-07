"""Contracts for ordinary-head specialist self-distillation controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    bind_source,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .supplemental_specialist_kd import (
    SPECIALIST_CONDITIONS,
    SPECIALIST_EXPERTS,
    SPECIALIST_KD_PLAN_CONTRACT,
    SPECIALIST_REPORT_CONTRACT,
    SPECIALIST_STUDENT_CONTRACT,
    SPECIALIST_TEACHER_CONTRACT,
    STUDENT_COORDINATES,
    file_sha256,
    specialist_teacher_configuration,
    validate_specialist_kd_plan,
)


ORDINARY_SPECIALIST_PLAN_CONTRACT = "retb_ordinary_specialist_kd_plan_v1"
ORDINARY_SPECIALIST_STUDENT_CONTRACT = (
    "retb_ordinary_specialist_kd_student_v1"
)
ORDINARY_SPECIALIST_REPORT_CONTRACT = "retb_ordinary_specialist_kd_report_v2"
ORDINARY_SPECIALIST_CHECKPOINT_CONTRACT = (
    "retb_ordinary_specialist_kd_checkpoint_v1"
)
ORDINARY_SPECIALIST_RESUME_CONTRACT = "retb_ordinary_specialist_kd_resume_v1"


def ordinary_student_configuration(expert: str, condition: str) -> dict[str, Any]:
    """Return the exact no-bottleneck counterpart of one compact student."""

    if expert not in SPECIALIST_EXPERTS:
        raise ValueError("unknown ordinary specialist student")
    if condition not in SPECIALIST_CONDITIONS:
        raise ValueError("unknown ordinary specialist KD condition")
    configuration = specialist_teacher_configuration(expert)
    configuration.update(
        {
            "loss_id": condition,
            "ordinary_weaver_head": True,
            "summary_token_bottleneck_present": False,
            "token_target_eligible": False,
            "student_architecture": "ORDINARY_UNCOMPRESSED",
        }
    )
    return configuration


def _json_record(path: Path, *, expected_contract: str | None = None) -> dict[str, Any]:
    payload = load_hashed_json(path, expected_contract=expected_contract)
    return {
        "path": str(path.resolve()),
        "content_hash": payload["content_hash"],
        "file_sha256": file_sha256(path),
        "source": payload.get("source"),
    }


def _file_record(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"ordinary specialist KD parent is absent or unsafe: {path}"
        )
    return {"path": str(path.resolve()), "file_sha256": file_sha256(path)}


def build_ordinary_specialist_kd_plan(
    *,
    compact_specialist_root: str | Path,
    supplemental_id: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the completed compact wave and its teachers as fixed parents."""

    compact_root = Path(compact_specialist_root).resolve()
    compact_plan_path = compact_root / "registry/specialist_kd_plan.json"
    compact_report_path = compact_root / "reports/specialist_kd_report.json"
    compact_plan = load_hashed_json(
        compact_plan_path, expected_contract=SPECIALIST_KD_PLAN_CONTRACT
    )
    validate_specialist_kd_plan(compact_plan)
    compact_report = load_hashed_json(
        compact_report_path, expected_contract=SPECIALIST_REPORT_CONTRACT
    )
    if (
        compact_report.get("plan_sha256") != compact_plan["content_hash"]
        or compact_report.get("all_eight_students_complete") is not True
        or compact_report.get("final_test_accessed") is not False
    ):
        raise ValueError("ordinary specialist KD compact parent is incomplete")

    artifacts: dict[str, dict[str, Any]] = {
        "compact_plan": _json_record(
            compact_plan_path, expected_contract=SPECIALIST_KD_PLAN_CONTRACT
        ),
        "compact_report": _json_record(
            compact_report_path, expected_contract=SPECIALIST_REPORT_CONTRACT
        ),
    }
    for expert in ("PT", "TRACK", "REGION"):
        manifest_path = compact_root / "runs/teachers" / expert / "teacher_manifest.json"
        manifest = load_hashed_json(
            manifest_path, expected_contract=SPECIALIST_TEACHER_CONTRACT
        )
        if (
            manifest.get("plan_sha256") != compact_plan["content_hash"]
            or manifest.get("expert") != expert
            or manifest.get("fixed_budget_completed") is not True
        ):
            raise ValueError(f"ordinary specialist KD {expert} teacher differs")
        artifacts[f"teacher_{expert}"] = _json_record(
            manifest_path, expected_contract=SPECIALIST_TEACHER_CONTRACT
        )
        for split in ("model_train", "val_stop"):
            record = manifest["prediction_caches"][split]
            path = Path(record["path"])
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(
                    f"ordinary specialist KD {expert} {split} cache drifted"
                )
            artifacts[f"teacher_{expert}_{split}"] = _file_record(path)

    for condition, expert in STUDENT_COORDINATES:
        result_path = compact_root / "runs/students" / condition / expert / "result.json"
        result = load_hashed_json(
            result_path, expected_contract=SPECIALIST_STUDENT_CONTRACT
        )
        coordinate = {"condition": condition, "expert": expert, "seed": 101}
        if (
            result.get("plan_sha256") != compact_plan["content_hash"]
            or result.get("coordinate") != coordinate
            or result.get("fixed_budget_completed") is not True
        ):
            raise ValueError("ordinary specialist KD compact student differs")
        key = f"compact_{condition}_{expert}"
        artifacts[key] = _json_record(
            result_path, expected_contract=SPECIALIST_STUDENT_CONTRACT
        )
        for split in ("val_stop", "val_design"):
            record = result["predictions"][split]
            path = Path(record["path"])
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(
                    f"ordinary specialist KD compact {condition} {expert} "
                    f"{split} prediction drifted"
                )
            artifacts[f"{key}_{split}"] = _file_record(path)

    plan = with_content_hash(
        {
            "contract": ORDINARY_SPECIALIST_PLAN_CONTRACT,
            "schema_version": 1,
            "supplemental_id": str(supplemental_id),
            "compact_specialist_root": str(compact_root),
            "parent_campaign_root": compact_plan["parent_campaign_root"],
            "compact_plan_sha256": compact_plan["content_hash"],
            "compact_report_sha256": compact_report["content_hash"],
            "parent_artifacts": artifacts,
            "experts": list(SPECIALIST_EXPERTS),
            "conditions": list(SPECIALIST_CONDITIONS),
            "student_coordinates": [
                {"condition": condition, "expert": expert, "seed": 101}
                for condition, expert in STUDENT_COORDINATES
            ],
            "student_configurations": {
                f"{condition}:{expert}": ordinary_student_configuration(
                    expert, condition
                )
                for condition, expert in STUDENT_COORDINATES
            },
            "objective": compact_plan["objective"],
            "training_protocol": compact_plan["training_protocol"],
            "fusion": compact_plan["fusion"],
            "comparison_design": {
                "kd_effect": "ordinary_KD_minus_ordinary_CE_teacher",
                "compression_effect": "compact_KD_minus_ordinary_KD",
                "architectures_differ_only_by_summary_token_bottleneck": True,
                "primary_split": "val_design",
            },
            "final_test_access": False,
            "scientific_underperformance_blocks_execution": False,
        }
    )
    return bind_source(plan, source_snapshot=source_snapshot)


def validate_ordinary_specialist_kd_plan(
    payload: Mapping[str, Any], *, check_parent_bytes: bool = True
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=ORDINARY_SPECIALIST_PLAN_CONTRACT
    )
    if (
        int(payload.get("schema_version", -1)) != 1
        or payload.get("experts") != list(SPECIALIST_EXPERTS)
        or payload.get("conditions") != list(SPECIALIST_CONDITIONS)
        or payload.get("student_coordinates")
        != [
            {"condition": condition, "expert": expert, "seed": 101}
            for condition, expert in STUDENT_COORDINATES
        ]
        or payload.get("student_configurations")
        != {
            f"{condition}:{expert}": ordinary_student_configuration(
                expert, condition
            )
            for condition, expert in STUDENT_COORDINATES
        }
        or payload.get("comparison_design")
        != {
            "kd_effect": "ordinary_KD_minus_ordinary_CE_teacher",
            "compression_effect": "compact_KD_minus_ordinary_KD",
            "architectures_differ_only_by_summary_token_bottleneck": True,
            "primary_split": "val_design",
        }
        or payload.get("final_test_access") is not False
        or payload.get("scientific_underperformance_blocks_execution") is not False
        or payload.get("objective")
        != {
            "temperature": 2.0,
            "cross_entropy_weight": 0.25,
            "MATCHED_KD": {
                "common_teacher_weight": 0.0,
                "specialist_teacher_weight": 1.0,
            },
            "HYBRID_KD": {
                "common_teacher_weight": 0.5,
                "specialist_teacher_weight": 0.5,
            },
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
        raise ValueError("ordinary specialist KD plan semantics differ")
    require_sha256(
        payload.get("compact_plan_sha256"), name="compact_plan_sha256"
    )
    require_sha256(
        payload.get("compact_report_sha256"), name="compact_report_sha256"
    )
    required_records = {"compact_plan", "compact_report"}
    required_records.update(
        f"teacher_{expert}" for expert in ("PT", "TRACK", "REGION")
    )
    required_records.update(
        f"teacher_{expert}_{split}"
        for expert in ("PT", "TRACK", "REGION")
        for split in ("model_train", "val_stop")
    )
    required_records.update(
        f"compact_{condition}_{expert}"
        for condition, expert in STUDENT_COORDINATES
    )
    required_records.update(
        f"compact_{condition}_{expert}_{split}"
        for condition, expert in STUDENT_COORDINATES
        for split in ("val_stop", "val_design")
    )
    if not required_records.issubset(payload.get("parent_artifacts", {})):
        raise ValueError("ordinary specialist KD parent index is incomplete")
    for name, record in payload["parent_artifacts"].items():
        require_sha256(record.get("file_sha256"), name=f"{name}.file_sha256")
        if "content_hash" in record:
            require_sha256(
                record.get("content_hash"), name=f"{name}.content_hash"
            )
    if check_parent_bytes:
        for name, record in payload["parent_artifacts"].items():
            path = Path(record["path"])
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(
                    f"ordinary specialist KD parent {name} bytes drifted"
                )
            if "content_hash" in record:
                artifact = load_hashed_json(path)
                if (
                    artifact["content_hash"] != record["content_hash"]
                    or artifact.get("source") != record.get("source")
                ):
                    raise ValueError(
                        f"ordinary specialist KD parent {name} lineage drifted"
                    )
    return digest


__all__ = [
    "ORDINARY_SPECIALIST_CHECKPOINT_CONTRACT",
    "ORDINARY_SPECIALIST_PLAN_CONTRACT",
    "ORDINARY_SPECIALIST_REPORT_CONTRACT",
    "ORDINARY_SPECIALIST_RESUME_CONTRACT",
    "ORDINARY_SPECIALIST_STUDENT_CONTRACT",
    "build_ordinary_specialist_kd_plan",
    "ordinary_student_configuration",
    "validate_ordinary_specialist_kd_plan",
]
