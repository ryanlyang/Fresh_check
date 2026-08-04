"""Authenticated conventional-model KD controls for the RETB campaign."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .contracts import bind_source, load_hashed_json, validate_content_hash, with_content_hash
from .expert_training import validate_expert_loss_registry, validate_teacher_logits_manifest
from .offline_capacity_models import FAMILIES


KD_BASELINE_PLAN_CONTRACT = "retb_supplemental_kd_baseline_plan_v1"
KD_BASELINE_RESULT_CONTRACT = "retb_supplemental_kd_baseline_result_v1"
KD_BASELINE_REPORT_CONTRACT = "retb_supplemental_kd_baseline_report_v1"
KD_BASELINE_ARCHITECTURES = ("O_BASE", "O_FULLREL")
KD_BASELINE_SEEDS = (101, 202, 303)
KD_BASELINE_COORDINATES = tuple(
    (architecture, seed)
    for architecture in KD_BASELINE_ARCHITECTURES
    for seed in KD_BASELINE_SEEDS
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conventional_kd_configuration(architecture: str) -> dict[str, Any]:
    if architecture not in KD_BASELINE_ARCHITECTURES:
        raise ValueError("unknown supplemental KD baseline architecture")
    return {
        "expert_id": architecture,
        "relation_family": None if architecture == "O_BASE" else "ALL",
        "relation_families": [] if architecture == "O_BASE" else list(FAMILIES),
        "all_particle_fields": True,
        "base4_present": True,
        "shape_id": "S8_128",
        "token_count": 8,
        "token_dimension": 128,
        "topology": "B_CONCAT",
        "tokenizer_mode": "TOK_WEAVER_CLASS",
        "loss_id": "ELOSS_KD_DOMINANT",
        "initialization": "INIT_SCRATCH",
        "learning_rate": 1.0e-3,
        "particle_dropout": 0.0,
        "measurement_embedding": False,
        "ordinary_weaver_head": True,
        "token_target_eligible": False,
        "summary_token_bottleneck_present": False,
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
        raise FileNotFoundError(f"supplemental KD parent is absent or unsafe: {path}")
    return {"path": str(path.resolve()), "file_sha256": file_sha256(path)}


def build_kd_baseline_plan(
    *, parent_root: str | Path, supplemental_id: str, source_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(parent_root).resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    if campaign.get("campaign_profile") != "production_500k_scale3m":
        raise ValueError("supplemental KD baselines require a production parent")
    loss_registry_path = root / "registry/retb_expert_losses.json"
    loss_registry = load_hashed_json(loss_registry_path)
    validate_expert_loss_registry(loss_registry)
    candidate = loss_registry["candidates"]["ELOSS_KD_DOMINANT"]
    if candidate != {
        "cross_entropy_weight": 0.25,
        "kd_weight": 1.0,
        "teacher": "SELECTED_STRONGEST",
    } or float(loss_registry["temperature"]) != 2.0:
        raise ValueError("dominant KD objective differs")
    selection_path = root / "selection/stage_a_strongest_teacher.json"
    selection = load_hashed_json(selection_path)
    teacher_id = str(selection["selected_teacher"])
    if teacher_id != "O_FULLREL":
        raise ValueError("supplemental KD teacher is not the locked O_FULLREL")
    teacher_registration_path = (
        root / "runs/stage_a/offline_controls" / teacher_id
        / "seed_101/checkpoint_registration.json"
    )
    teacher_registration = load_hashed_json(teacher_registration_path)
    if (
        teacher_registration.get("expert_id") != teacher_id
        or int(teacher_registration.get("seed", -1)) != 101
        or teacher_registration.get("loss_id") != "ELOSS_CE"
        or teacher_registration.get("fixed_epoch_budget_completed") is not True
        or teacher_registration.get("performance_based_termination") is not False
    ):
        raise ValueError("supplemental KD teacher registration semantics differ")
    baseline_registrations = {}
    for control_id in KD_BASELINE_ARCHITECTURES:
        path = (
            root / "runs/stage_a/offline_controls" / control_id
            / "seed_101/checkpoint_registration.json"
        )
        registration = load_hashed_json(path)
        if (
            registration.get("expert_id") != control_id
            or int(registration.get("seed", -1)) != 101
            or registration.get("loss_id") != "ELOSS_CE"
            or registration.get("fixed_epoch_budget_completed") is not True
            or registration.get("performance_based_termination") is not False
        ):
            raise ValueError(
                f"supplemental KD baseline {control_id} semantics differ"
            )
        baseline_registrations[control_id] = {
            "path": str(path.resolve()),
            "content_hash": registration["content_hash"],
            "checkpoint_sha256": registration["checkpoint_sha256"],
            "selected_val_stop": registration["selected_val_stop"],
            "loss_id": registration["loss_id"],
            "fixed_epoch_budget_completed": registration[
                "fixed_epoch_budget_completed"
            ],
            "source": registration.get("source"),
        }
    teacher_checkpoint = (
        root / "runs/stage_a/offline_controls/SELECTED_STRONGEST/seed_101/best_model_val.pt"
    )
    teacher_checkpoint_sha = file_sha256(teacher_checkpoint)
    manifest_path = root / "inputs/teacher_logits/ELOSS_KD_DOMINANT.json"
    manifest = load_hashed_json(manifest_path)
    validate_teacher_logits_manifest(manifest)
    if (
        manifest["teacher_checkpoint_hashes"].get("SELECTED_STRONGEST")
        != teacher_checkpoint_sha
        or teacher_registration.get("checkpoint_sha256") != teacher_checkpoint_sha
    ):
        raise ValueError("supplemental KD teacher checkpoint lineage differs")
    train_npz = root / "inputs/teacher_logits/ELOSS_KD_DOMINANT/model_train.npz"
    stop_npz = root / "inputs/teacher_logits/ELOSS_KD_DOMINANT/val_stop.npz"
    if (
        manifest["model_train_npz_sha256"] != file_sha256(train_npz)
        or manifest["val_stop_npz_sha256"] != file_sha256(stop_npz)
    ):
        raise ValueError("supplemental KD teacher-logit cache bytes differ")
    design_manifest_path = root / "inputs/offline/val_design/offline_input_manifest.json"
    design_manifest = load_hashed_json(design_manifest_path)
    design_npz = design_manifest_path.parent / design_manifest["npz_filename"]
    if design_manifest["npz_sha256"] != file_sha256(design_npz):
        raise ValueError("supplemental KD val_design bytes differ")
    artifacts = {
        "campaign_spec": _json_record(root / "campaign_spec.json"),
        "expert_loss_registry": _json_record(loss_registry_path),
        "step3_bundle": _json_record(root / "registry/retb_step3_architecture_bundle.json"),
        "teacher_selection": _json_record(selection_path),
        "teacher_registration": _json_record(teacher_registration_path),
        "teacher_logits_manifest": _json_record(manifest_path),
        "teacher_checkpoint": _file_record(teacher_checkpoint),
        "model_train_teacher_inputs": _file_record(train_npz),
        "val_stop_teacher_inputs": _file_record(stop_npz),
        "val_design_manifest": _json_record(design_manifest_path),
        "val_design_inputs": _file_record(design_npz),
        "relation_normalization": _json_record(
            root / "inputs/normalization/offline_500k/relation.json"
        ),
        "region_normalization": _json_record(
            root / "inputs/normalization/offline_500k/region.json"
        ),
    }
    payload = with_content_hash(
        {
            "contract": KD_BASELINE_PLAN_CONTRACT,
            "schema_version": 1,
            "supplemental_id": str(supplemental_id),
            "parent_campaign_root": str(root),
            "parent_campaign_spec_sha256": campaign["content_hash"],
            "parent_campaign_source": campaign.get("source"),
            "parent_artifacts": artifacts,
            "teacher_id": teacher_id,
            "teacher_field": "SELECTED_STRONGEST",
            "teacher_checkpoint_sha256": teacher_checkpoint_sha,
            "baseline_registrations": baseline_registrations,
            "loss_id": "ELOSS_KD_DOMINANT",
            "objective": {
                "cross_entropy_weight": 0.25,
                "kd_weight": 1.0,
                "temperature": 2.0,
                "kl_direction": "teacher_probability_to_student_probability",
            },
            "architectures": {
                name: conventional_kd_configuration(name)
                for name in KD_BASELINE_ARCHITECTURES
            },
            "seeds": list(KD_BASELINE_SEEDS),
            "coordinates": [
                {"architecture": architecture, "seed": seed}
                for architecture, seed in KD_BASELINE_COORDINATES
            ],
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
            "final_test_access": False,
            "scientific_underperformance_blocks_execution": False,
        }
    )
    return bind_source(payload, source_snapshot=source_snapshot)


def validate_kd_baseline_plan(
    payload: Mapping[str, Any], *, check_parent_bytes: bool = True
) -> str:
    digest = validate_content_hash(payload, expected_contract=KD_BASELINE_PLAN_CONTRACT)
    if (
        payload.get("loss_id") != "ELOSS_KD_DOMINANT"
        or payload.get("teacher_id") != "O_FULLREL"
        or payload.get("teacher_field") != "SELECTED_STRONGEST"
        or payload.get("seeds") != list(KD_BASELINE_SEEDS)
        or payload.get("coordinates")
        != [
            {"architecture": architecture, "seed": seed}
            for architecture, seed in KD_BASELINE_COORDINATES
        ]
        or payload.get("final_test_access") is not False
        or set(payload.get("baseline_registrations", {}))
        != set(KD_BASELINE_ARCHITECTURES)
    ):
        raise ValueError("supplemental KD plan semantics differ")
    if payload.get("architectures") != {
        name: conventional_kd_configuration(name)
        for name in KD_BASELINE_ARCHITECTURES
    }:
        raise ValueError("supplemental KD architectures differ")
    if check_parent_bytes:
        for name, row in payload["parent_artifacts"].items():
            path = Path(row["path"])
            if file_sha256(path) != row["file_sha256"]:
                raise ValueError(f"supplemental KD parent {name} bytes drifted")
            if "content_hash" in row:
                artifact = load_hashed_json(path)
                if (
                    artifact["content_hash"] != row["content_hash"]
                    or artifact.get("source") != row.get("source")
                ):
                    raise ValueError(f"supplemental KD parent {name} lineage drifted")
        for name, row in payload["baseline_registrations"].items():
            registration = load_hashed_json(row["path"])
            if (
                registration["content_hash"] != row["content_hash"]
                or registration.get("checkpoint_sha256")
                != row["checkpoint_sha256"]
                or registration.get("selected_val_stop")
                != row["selected_val_stop"]
                or registration.get("loss_id") != row["loss_id"]
                or registration.get("fixed_epoch_budget_completed")
                != row["fixed_epoch_budget_completed"]
                or registration.get("source") != row.get("source")
            ):
                raise ValueError(
                    f"supplemental KD baseline {name} lineage drifted"
                )
    return digest


__all__ = [
    "KD_BASELINE_ARCHITECTURES",
    "KD_BASELINE_COORDINATES",
    "KD_BASELINE_PLAN_CONTRACT",
    "KD_BASELINE_REPORT_CONTRACT",
    "KD_BASELINE_RESULT_CONTRACT",
    "KD_BASELINE_SEEDS",
    "build_kd_baseline_plan",
    "conventional_kd_configuration",
    "file_sha256",
    "validate_kd_baseline_plan",
]
