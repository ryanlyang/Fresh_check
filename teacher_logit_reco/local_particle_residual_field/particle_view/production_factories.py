"""Built-in scientific factories backed by authenticated production caches."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .registry import validate_particle_view_registry
from .runtime_data import (
    PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT,
    audit_runtime_data_sources,
    load_aligned_logical_jet_view,
    make_logical_data_loader,
    resolve_parent_task_artifacts,
    validate_runtime_data_config,
)
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .offline_teacher import (
    build_existing_teacher_source_registration,
    build_teacher_recipe,
)
from .teacher_train import ParticleViewTeacherTrainConfig
from .direct_control import (
    DirectControlTrainConfig,
    PARTICLE_VIEW_STAGE_A_RESOURCE_PLAN_CONTRACT,
    STAGE_A_DIRECT_CONTROL_RUNS,
    build_direct_control_recipe,
    validate_stage_a_direct_resource_plan,
)


PARTICLE_VIEW_BASELINE_FACTORY_CONFIG_CONTRACT = (
    "particle_view_baseline_factory_config_v1"
)
PARTICLE_VIEW_DIRECT_FACTORY_CONFIG_CONTRACT = (
    "particle_view_direct_control_factory_config_v1"
)
PARTICLE_VIEW_SOURCE_PREFLIGHT_CONFIG_CONTRACT = (
    "particle_view_source_preflight_config_v1"
)
PARTICLE_VIEW_EXISTING_TEACHER_RUNTIME_LINEAGE_CONTRACT = (
    "particle_view_existing_teacher_runtime_lineage_v1"
)
PARTICLE_VIEW_EXISTING_TEACHER_UNAVAILABLE_CONTRACT = (
    "particle_view_existing_teacher_unavailable_v1"
)

_TRAINED_TEACHER_ROLES = {
    "A0_VIEW": ("A0_view", "base", "fixed_hlt"),
    "TOFF_VIEW_BASE": ("Toff_view", "base", "offline"),
    "TOFF_VIEW_LARGE": ("Toff_view", "large", "offline"),
}
_EXISTING_TEACHER_ROLE = "TOFF_VIEW_EXISTING"


def _optional_positive(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def build_baseline_factory_config(
    *,
    runtime_data_config: Mapping[str, Any],
    device: str = "auto",
    num_workers: int = 0,
    amp: bool = True,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    existing_checkpoint_path: str | Path | None = None,
    existing_observed_train_identity_sha256: str | None = None,
    existing_serialized_recipe_path: str | Path | None = None,
    existing_recipe_reproduced_exactly: bool = False,
    existing_provenance_metadata_sha256: str | None = None,
    existing_description: str = "pre-existing offline particle teacher",
) -> dict[str, Any]:
    """Build one shared factory config for the four teacher Stage-A roles."""

    validate_runtime_data_config(
        runtime_data_config, verify_cache_files=True
    )
    if not isinstance(device, str) or not device:
        raise ValueError("device must be nonempty")
    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    existing = None
    if existing_checkpoint_path is not None:
        checkpoint = Path(existing_checkpoint_path).resolve()
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError("existing teacher checkpoint is absent or unsafe")
        serialized_recipe = None
        serialized_recipe_file = None
        if existing_serialized_recipe_path is not None:
            recipe_path = Path(existing_serialized_recipe_path).resolve()
            serialized_recipe = json.loads(
                recipe_path.read_text(encoding="utf-8")
            )
            if not isinstance(serialized_recipe, Mapping):
                raise ValueError("existing serialized recipe must be an object")
            serialized_recipe_file = {
                "path": str(recipe_path),
                "sha256": sha256_file(recipe_path),
            }
        if existing_provenance_metadata_sha256 is None:
            raise ValueError(
                "existing teacher requires provenance_metadata_sha256"
            )
        require_sha256(
            "existing_provenance_metadata_sha256",
            existing_provenance_metadata_sha256,
        )
        if existing_observed_train_identity_sha256 is not None:
            require_sha256(
                "existing_observed_train_identity_sha256",
                existing_observed_train_identity_sha256,
            )
        existing = {
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            },
            "observed_train_identity_sha256": (
                existing_observed_train_identity_sha256
            ),
            "serialized_recipe": serialized_recipe,
            "serialized_recipe_file": serialized_recipe_file,
            "recipe_reproduced_exactly": bool(
                existing_recipe_reproduced_exactly
            ),
            "provenance_metadata_sha256": (
                existing_provenance_metadata_sha256
            ),
            "description": str(existing_description),
        }
    elif any(
        value is not None
        for value in (
            existing_observed_train_identity_sha256,
            existing_serialized_recipe_path,
            existing_provenance_metadata_sha256,
        )
    ) or existing_recipe_reproduced_exactly:
        raise ValueError("existing-teacher evidence requires a checkpoint")
    payload = with_content_hash(
        {
            "contract": PARTICLE_VIEW_BASELINE_FACTORY_CONFIG_CONTRACT,
            "runtime_data_config": dict(runtime_data_config),
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "training": {
                "device": device,
                "num_workers": workers,
                "amp": bool(amp),
                "max_train_batches": _optional_positive(
                    max_train_batches, name="max_train_batches"
                ),
                "max_val_batches": _optional_positive(
                    max_val_batches, name="max_val_batches"
                ),
            },
            "existing_teacher": existing,
            "supported_run_ids": [
                *_TRAINED_TEACHER_ROLES,
                _EXISTING_TEACHER_ROLE,
            ],
            "direct_control_run_ids": [
                "STAGE_A_PARAMETER_MATCH",
                "STAGE_A_FLOP_MATCH",
            ],
            "direct_controls_require_resource_match_factory": True,
        }
    )
    validate_baseline_factory_config(payload)
    return payload


def build_direct_control_factory_config(
    *,
    runtime_data_config: Mapping[str, Any],
    resource_plan: Mapping[str, Any],
    device: str = "auto",
    num_workers: int = 0,
    amp: bool = True,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Bind the two matched-control jobs to one immutable resource plan."""

    validate_runtime_data_config(
        runtime_data_config, verify_cache_files=True
    )
    validate_stage_a_direct_resource_plan(resource_plan)
    if not isinstance(device, str) or not device:
        raise ValueError("device must be nonempty")
    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DIRECT_FACTORY_CONFIG_CONTRACT,
            "runtime_data_config": dict(runtime_data_config),
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "resource_plan": dict(resource_plan),
            "resource_plan_sha256": resource_plan["content_hash"],
            "training": {
                "device": device,
                "num_workers": workers,
                "amp": bool(amp),
                "max_train_batches": _optional_positive(
                    max_train_batches, name="max_train_batches"
                ),
                "max_val_batches": _optional_positive(
                    max_val_batches, name="max_val_batches"
                ),
            },
            "run_ids": list(STAGE_A_DIRECT_CONTROL_RUNS),
            "quality_warnings_non_gating": True,
        }
    )
    validate_direct_control_factory_config(artifact)
    return artifact


def validate_direct_control_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_DIRECT_FACTORY_CONFIG_CONTRACT
    )
    if set(payload) != {
        "contract",
        "runtime_data_config",
        "runtime_data_config_sha256",
        "resource_plan",
        "resource_plan_sha256",
        "training",
        "run_ids",
        "quality_warnings_non_gating",
        "content_hash",
    }:
        raise ValueError("direct factory config field inventory mismatch")
    if (
        payload["runtime_data_config"].get("content_hash")
        != payload["runtime_data_config_sha256"]
    ):
        raise ValueError("direct factory runtime-data hash mismatch")
    validate_runtime_data_config(
        payload["runtime_data_config"], verify_cache_files=False
    )
    if (
        payload["resource_plan"].get("content_hash")
        != payload["resource_plan_sha256"]
    ):
        raise ValueError("direct factory resource-plan hash mismatch")
    validate_stage_a_direct_resource_plan(payload["resource_plan"])
    runtime = payload["training"]
    if set(runtime) != {
        "device",
        "num_workers",
        "amp",
        "max_train_batches",
        "max_val_batches",
    }:
        raise ValueError("direct factory training inventory mismatch")
    if (
        not isinstance(runtime["device"], str)
        or not runtime["device"]
        or not isinstance(runtime["num_workers"], int)
        or isinstance(runtime["num_workers"], bool)
        or runtime["num_workers"] < 0
        or not isinstance(runtime["amp"], bool)
    ):
        raise ValueError("direct factory runtime settings are invalid")
    for name in ("max_train_batches", "max_val_batches"):
        _optional_positive(runtime[name], name=name)
    if (
        payload["run_ids"] != list(STAGE_A_DIRECT_CONTROL_RUNS)
        or payload["quality_warnings_non_gating"] is not True
    ):
        raise ValueError("direct factory run/warning policy changed")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "run_count": len(payload["run_ids"]),
    }


def validate_baseline_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_BASELINE_FACTORY_CONFIG_CONTRACT
    )
    if set(payload) != {
        "contract",
        "runtime_data_config",
        "runtime_data_config_sha256",
        "training",
        "existing_teacher",
        "supported_run_ids",
        "direct_control_run_ids",
        "direct_controls_require_resource_match_factory",
        "content_hash",
    }:
        raise ValueError("baseline factory config field inventory mismatch")
    data = payload["runtime_data_config"]
    if data.get("content_hash") != payload["runtime_data_config_sha256"]:
        raise ValueError("baseline factory runtime-data hash mismatch")
    validate_runtime_data_config(data, verify_cache_files=False)
    training = payload["training"]
    if set(training) != {
        "device",
        "num_workers",
        "amp",
        "max_train_batches",
        "max_val_batches",
    }:
        raise ValueError("baseline training config field inventory mismatch")
    if (
        not isinstance(training["device"], str)
        or not training["device"]
        or not isinstance(training["num_workers"], int)
        or isinstance(training["num_workers"], bool)
        or training["num_workers"] < 0
        or not isinstance(training["amp"], bool)
    ):
        raise ValueError("baseline training runtime settings are invalid")
    for name in ("max_train_batches", "max_val_batches"):
        _optional_positive(training[name], name=name)
    if payload["supported_run_ids"] != [
        *_TRAINED_TEACHER_ROLES,
        _EXISTING_TEACHER_ROLE,
    ]:
        raise ValueError("baseline supported-role inventory changed")
    if (
        payload["direct_control_run_ids"]
        != ["STAGE_A_PARAMETER_MATCH", "STAGE_A_FLOP_MATCH"]
        or payload["direct_controls_require_resource_match_factory"] is not True
    ):
        raise ValueError("direct-control factory boundary changed")
    existing = payload["existing_teacher"]
    if existing is not None:
        if set(existing) != {
            "checkpoint",
            "observed_train_identity_sha256",
            "serialized_recipe",
            "serialized_recipe_file",
            "recipe_reproduced_exactly",
            "provenance_metadata_sha256",
            "description",
        }:
            raise ValueError("existing teacher evidence inventory mismatch")
        checkpoint = existing["checkpoint"]
        if (
            set(checkpoint) != {"path", "sha256"}
            or sha256_file(checkpoint["path"]) != checkpoint["sha256"]
        ):
            raise ValueError("existing teacher checkpoint changed")
        if existing["serialized_recipe_file"] is not None:
            binding = existing["serialized_recipe_file"]
            if (
                set(binding) != {"path", "sha256"}
                or sha256_file(binding["path"]) != binding["sha256"]
                or json.loads(
                    Path(binding["path"]).read_text(encoding="utf-8")
                ) != existing["serialized_recipe"]
            ):
                raise ValueError("existing serialized recipe changed")
        require_sha256(
            "provenance_metadata_sha256",
            existing["provenance_metadata_sha256"],
        )
        if existing["observed_train_identity_sha256"] is not None:
            require_sha256(
                "observed_train_identity_sha256",
                existing["observed_train_identity_sha256"],
            )
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "trained_teacher_count": len(_TRAINED_TEACHER_ROLES),
        "existing_teacher_configured": existing is not None,
    }


def _library_versions_sha256() -> str:
    torch = require_torch()
    try:
        weaver = importlib.metadata.version("weaver-core")
    except importlib.metadata.PackageNotFoundError:
        weaver = "unknown"
    return canonical_sha256(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "weaver_core": weaver,
        }
    )


def _source_preflight_binding(
    *,
    registry: Mapping[str, Any],
    output_dir: str,
    run_id: str,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    root = Path(output_dir).resolve().parent.parent
    parents = resolve_parent_task_artifacts(
        registry=registry,
        artifact_root=root,
        run_id=run_id,
        seed=seed,
    )
    if set(parents) != {"PV_SOURCE_PREFLIGHT"}:
        raise ValueError("Stage-A teacher must have only source preflight as parent")
    source = parents["PV_SOURCE_PREFLIGHT"]
    try:
        audit = source["artifacts"]["source_cache_audit.json"]
    except KeyError as exc:
        raise ValueError("source parent omitted source_cache_audit.json") from exc
    return str(audit["sha256"]), source


def _view_source_sha256(
    *,
    runtime_data_config: Mapping[str, Any],
    source_view: str,
    source_audit_sha256: str,
) -> str:
    kinds = (
        ("hlt_array", "hlt_metadata")
        if source_view == "fixed_hlt"
        else ("offline_array", "offline_metadata")
    )
    required = {"model_train", "model_val"}
    records = [
        {
            "parent_split": row["parent_split"],
            **{kind: row[kind] for kind in kinds},
        }
        for row in runtime_data_config["parent_cache_records"]
        if row["parent_split"] in required
    ]
    if {row["parent_split"] for row in records} != required:
        raise ValueError("teacher source cache inventory is incomplete")
    return canonical_sha256(
        {
            "contract": "particle_view_teacher_source_binding_v1",
            "source_view": source_view,
            "source_preflight_artifact_sha256": source_audit_sha256,
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "cache_records": records,
        }
    )


def _teacher_preprocessing_sha256(source_view: str) -> str:
    return canonical_sha256(
        {
            "contract": "particle_view_teacher_preprocessing_v1",
            "source_view": source_view,
            "feature_names": list(PF_FEATURE_NAMES),
            "builder": (
                "jetclass_fresh.part_inputs."
                "build_particle_transformer_inputs_from_tokens"
            ),
            "mask": "bool_valid_particle_mask_v1",
            "max_particles": 128,
        }
    )


def _validate_factory_run(
    registry: Mapping[str, Any], *, run_id: str, seed: int
) -> Mapping[str, Any]:
    validate_particle_view_registry(registry)
    rows = {row["run_id"]: row for row in registry["runs"]}
    if run_id not in rows:
        raise KeyError(f"unknown particle-view run {run_id!r}")
    row = rows[run_id]
    if int(seed) not in row["seed_ids"]:
        raise ValueError("baseline seed is not registered")
    if not str(row["scientific_role"]).startswith("baseline:"):
        raise ValueError("baseline factory received a non-baseline run")
    return row


def register_existing_teacher_source(
    *,
    checkpoint_path: str | None,
    expected_checkpoint_sha256: str | None,
    canonical_train_identity_sha256: str,
    observed_train_identity_sha256: str | None,
    serialized_recipe: Mapping[str, Any] | None,
    recipe_reproduced_exactly: bool,
    provenance_metadata_sha256: str,
    description: str,
    source_preflight_artifact_sha256: str,
    output_path: str,
    lineage_output_path: str,
) -> None:
    """Locked runtime operation for classifying an existing checkpoint."""

    if checkpoint_path is None:
        if expected_checkpoint_sha256 is not None:
            raise ValueError("unavailable existing teacher has a checkpoint hash")
        artifact = with_content_hash(
            {
                "contract": PARTICLE_VIEW_EXISTING_TEACHER_UNAVAILABLE_CONTRACT,
                "availability": "not_configured",
                "selection_status": "diagnostic_unavailable",
                "selectable": False,
                "canonical_train_identity_sha256": (
                    canonical_train_identity_sha256
                ),
                "source_preflight_artifact_sha256": (
                    source_preflight_artifact_sha256
                ),
                "scientific_warning": "WARN_EXISTING_TEACHER_UNAVAILABLE",
                "warning_is_non_gating": True,
            }
        )
        write_immutable_json(output_path, artifact)
        lineage = with_content_hash(
            {
                "contract": (
                    PARTICLE_VIEW_EXISTING_TEACHER_RUNTIME_LINEAGE_CONTRACT
                ),
                "existing_teacher_source_registration_sha256": artifact[
                    "content_hash"
                ],
                "source_preflight_artifact_sha256": (
                    source_preflight_artifact_sha256
                ),
                "checkpoint_sha256": None,
                "canonical_train_identity_sha256": (
                    canonical_train_identity_sha256
                ),
                "selection_status": "diagnostic_unavailable",
                "selectable": False,
            }
        )
        write_immutable_json(lineage_output_path, lineage)
        return
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("existing teacher checkpoint changed before registration")
    artifact = build_existing_teacher_source_registration(
        checkpoint_path=checkpoint_path,
        canonical_train_identity_sha256=canonical_train_identity_sha256,
        observed_train_identity_sha256=observed_train_identity_sha256,
        serialized_recipe=serialized_recipe,
        recipe_reproduced_exactly=recipe_reproduced_exactly,
        provenance_metadata_sha256=provenance_metadata_sha256,
        description=description,
    )
    write_immutable_json(output_path, artifact)
    lineage = with_content_hash(
        {
            "contract": (
                PARTICLE_VIEW_EXISTING_TEACHER_RUNTIME_LINEAGE_CONTRACT
            ),
            "existing_teacher_source_registration_sha256": artifact[
                "content_hash"
            ],
            "source_preflight_artifact_sha256": (
                source_preflight_artifact_sha256
            ),
            "checkpoint_sha256": artifact["checkpoint_sha256"],
            "canonical_train_identity_sha256": artifact[
                "canonical_train_identity_sha256"
            ],
            "selection_status": artifact["selection_status"],
            "selectable": artifact["selectable"],
        }
    )
    write_immutable_json(lineage_output_path, lineage)


def build_baseline_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    """Prepare a matched teacher or existing-teacher provenance task."""

    del task_id
    validate_baseline_factory_config(config)
    row = _validate_factory_run(
        registry, run_id=run_id, seed=int(seed)
    )
    source_audit_sha256, _ = _source_preflight_binding(
        registry=registry,
        output_dir=output_dir,
        run_id=run_id,
        seed=int(seed),
    )
    output = Path(output_dir).resolve()
    if run_id in _TRAINED_TEACHER_ROLES:
        if operation != "teacher_training":
            raise ValueError("trained teacher role has the wrong operation")
        role, architecture, source_view = _TRAINED_TEACHER_ROLES[run_id]
        data = config["runtime_data_config"]
        train = load_aligned_logical_jet_view(data, "train")
        stop = load_aligned_logical_jet_view(data, "model_val_stop")
        unified = load_hashed_json(data["unified_manifest"]["path"])
        recipe = build_teacher_recipe(
            role=role,
            architecture=architecture,
            seed=int(seed),
            unified_split_manifest=unified,
            preprocessing_sha256=_teacher_preprocessing_sha256(source_view),
            source_sha256=_view_source_sha256(
                runtime_data_config=data,
                source_view=source_view,
                source_audit_sha256=source_audit_sha256,
            ),
            initialization_implementation_sha256=canonical_sha256(
                {
                    "builder": "build_particle_transformer_classifier",
                    "constructor": "weaver.nn.model.ParticleTransformer",
                    "from_scratch": True,
                }
            ),
            library_versions_sha256=_library_versions_sha256(),
        )
        physical_batch = recipe.to_payload()["physical_batch_size"]
        runtime = config["training"]
        train_loader = make_logical_data_loader(
            train,
            mode=source_view,
            batch_size=physical_batch,
            shuffle=True,
            num_workers=runtime["num_workers"],
            seed=int(seed),
        )
        stop_loader = make_logical_data_loader(
            stop,
            mode=source_view,
            batch_size=physical_batch,
            shuffle=False,
            num_workers=runtime["num_workers"],
            seed=int(seed) + 1,
        )
        return {
            "kwargs": {
                "recipe": recipe,
                "train_loader": train_loader,
                "model_val_stop_loader": stop_loader,
                "config": ParticleViewTeacherTrainConfig(
                    output_dir=str(output),
                    device=runtime["device"],
                    max_train_batches=runtime["max_train_batches"],
                    max_val_batches=runtime["max_val_batches"],
                    amp=runtime["amp"],
                ),
            },
            "artifact_paths": [
                str(output / "best_model_val_stop.pt"),
                str(output / "teacher_registration.json"),
                str(output / "training_curves.json"),
                str(output / "teacher_report.json"),
            ],
            "action": None,
        }
    if run_id == _EXISTING_TEACHER_ROLE:
        if operation != "existing_teacher_registration":
            raise ValueError("existing teacher role has the wrong operation")
        existing = config["existing_teacher"]
        destination = output / "existing_teacher_source_registration.json"
        lineage_destination = output / "existing_teacher_runtime_lineage.json"
        absent = existing is None
        return {
            "kwargs": {
                "checkpoint_path": (
                    None if absent else existing["checkpoint"]["path"]
                ),
                "expected_checkpoint_sha256": (
                    None if absent else existing["checkpoint"]["sha256"]
                ),
                "canonical_train_identity_sha256": registry[
                    "train_identity_sha256"
                ],
                "observed_train_identity_sha256": (
                    None
                    if absent
                    else existing["observed_train_identity_sha256"]
                ),
                "serialized_recipe": (
                    None if absent else existing["serialized_recipe"]
                ),
                "recipe_reproduced_exactly": (
                    False if absent else existing["recipe_reproduced_exactly"]
                ),
                "provenance_metadata_sha256": (
                    "0" * 64
                    if absent
                    else existing["provenance_metadata_sha256"]
                ),
                "description": (
                    "existing offline teacher was not configured"
                    if absent
                    else existing["description"]
                ),
                "source_preflight_artifact_sha256": source_audit_sha256,
                "output_path": str(destination),
                "lineage_output_path": str(lineage_destination),
            },
            "artifact_paths": [str(destination), str(lineage_destination)],
            "action": None,
        }
    if run_id in {"STAGE_A_PARAMETER_MATCH", "STAGE_A_FLOP_MATCH"}:
        raise ValueError(
            "direct-control roles require the resource-match factory"
        )
    raise ValueError(f"unsupported Stage-A baseline role {row['run_id']}")


def build_direct_control_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    """Prepare one selected direct HLT control for the locked trainer."""

    del task_id
    if operation != "direct_control_training":
        raise ValueError("direct-control factory received another operation")
    validate_direct_control_factory_config(config)
    _validate_factory_run(registry, run_id=run_id, seed=int(seed))
    if run_id not in STAGE_A_DIRECT_CONTROL_RUNS:
        raise ValueError("direct-control factory received another baseline role")
    source_audit_sha256, _ = _source_preflight_binding(
        registry=registry,
        output_dir=output_dir,
        run_id=run_id,
        seed=int(seed),
    )
    data = config["runtime_data_config"]
    unified = load_hashed_json(data["unified_manifest"]["path"])
    recipe = build_direct_control_recipe(
        run_id=run_id,
        seed=int(seed),
        resource_plan=config["resource_plan"],
        unified_split_manifest=unified,
        preprocessing_sha256=_teacher_preprocessing_sha256("fixed_hlt"),
        source_sha256=_view_source_sha256(
            runtime_data_config=data,
            source_view="fixed_hlt",
            source_audit_sha256=source_audit_sha256,
        ),
        library_versions_sha256=_library_versions_sha256(),
    )
    train = load_aligned_logical_jet_view(data, "train")
    stop = load_aligned_logical_jet_view(data, "model_val_stop")
    runtime = config["training"]
    train_loader = make_logical_data_loader(
        train,
        mode="fixed_hlt",
        batch_size=128,
        shuffle=True,
        num_workers=runtime["num_workers"],
        seed=int(seed),
    )
    stop_loader = make_logical_data_loader(
        stop,
        mode="fixed_hlt",
        batch_size=128,
        shuffle=False,
        num_workers=runtime["num_workers"],
        seed=int(seed) + 1,
    )
    output = Path(output_dir).resolve()
    return {
        "kwargs": {
            "recipe": recipe,
            "train_loader": train_loader,
            "model_val_stop_loader": stop_loader,
            "config": DirectControlTrainConfig(
                output_dir=str(output),
                device=runtime["device"],
                max_train_batches=runtime["max_train_batches"],
                max_val_batches=runtime["max_val_batches"],
                amp=runtime["amp"],
            ),
        },
        "artifact_paths": [
            str(output / "best_model_val_stop.pt"),
            str(output / "direct_control_registration.json"),
            str(output / "training_curves.json"),
            str(output / "direct_control_report.json"),
        ],
        "action": None,
    }


def build_stage_a_teacher_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    """Return catalog fragments for the four now-runnable teacher roles."""

    path = Path(factory_config_path).resolve()
    config = load_hashed_json(path)
    validate_baseline_factory_config(config)
    common = {
        "factory": (
            "teacher_logit_reco.local_particle_residual_field.particle_view."
            "production_factories:build_baseline_factory"
        ),
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    return {
        run_id: {
            **common,
            "operation": (
                "existing_teacher_registration"
                if run_id == _EXISTING_TEACHER_ROLE
                else "teacher_training"
            ),
        }
        for run_id in [*_TRAINED_TEACHER_ROLES, _EXISTING_TEACHER_ROLE]
    }


def build_stage_a_direct_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    """Return catalog fragments for both three-seed direct controls."""

    path = Path(factory_config_path).resolve()
    config = load_hashed_json(path)
    validate_direct_control_factory_config(config)
    common = {
        "operation": "direct_control_training",
        "factory": (
            "teacher_logit_reco.local_particle_residual_field.particle_view."
            "production_factories:build_direct_control_factory"
        ),
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    return {run_id: dict(common) for run_id in STAGE_A_DIRECT_CONTROL_RUNS}


def _publish_source_cache_audit(
    *,
    runtime_data_config: Mapping[str, Any],
    output_path: str,
    storage_reservation: Mapping[str, Any] | None = None,
    storage_output_path: str | None = None,
) -> None:
    audit = audit_runtime_data_sources(runtime_data_config)
    write_immutable_json(output_path, audit)
    if storage_reservation is not None:
        validate_content_hash(
            storage_reservation,
            expected_contract="particle_view_storage_reservation_v1",
        )
        if storage_reservation.get("preflight_passed") is not True:
            raise ValueError("storage reservation did not pass preflight")
        if storage_output_path is None:
            raise ValueError("storage reservation output path is absent")
        write_immutable_json(storage_output_path, storage_reservation)


def build_source_preflight_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    """Prepare the non-training source task used by the runtime executor."""

    del task_id
    if operation != "source_preflight":
        raise ValueError("source-preflight factory received another operation")
    storage_reservation = None
    if config.get("contract") == PARTICLE_VIEW_SOURCE_PREFLIGHT_CONFIG_CONTRACT:
        validate_content_hash(
            config,
            expected_contract=PARTICLE_VIEW_SOURCE_PREFLIGHT_CONFIG_CONTRACT,
        )
        runtime_data_config = config["runtime_data_config"]
        storage_reservation = config["storage_reservation"]
        validate_runtime_data_config(
            runtime_data_config, verify_cache_files=False
        )
        validate_content_hash(
            storage_reservation,
            expected_contract="particle_view_storage_reservation_v1",
        )
        if (
            config.get("runtime_data_config_sha256")
            != runtime_data_config["content_hash"]
            or config.get("storage_reservation_sha256")
            != storage_reservation["content_hash"]
            or config.get("storage_is_execution_gating") is not True
            or config.get("scientific_metrics_are_execution_gating") is not False
        ):
            raise ValueError("source-preflight config lineage/policy changed")
    else:
        validate_content_hash(
            config, expected_contract=PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT
        )
        runtime_data_config = config
    validate_particle_view_registry(registry)
    rows = {row["run_id"]: row for row in registry["runs"]}
    if run_id not in rows or not str(rows[run_id]["scientific_role"]).startswith(
        "source:"
    ):
        raise ValueError("source-preflight factory received a non-source run")
    if int(seed) not in rows[run_id]["seed_ids"]:
        raise ValueError("source-preflight seed is not registered")
    destination = Path(output_dir).resolve() / "source_cache_audit.json"
    storage_destination = (
        Path(output_dir).resolve() / "storage_reservation.json"
    )
    return {
        "kwargs": {
            "runtime_data_config": runtime_data_config,
            "output_path": str(destination),
            "storage_reservation": storage_reservation,
            "storage_output_path": (
                str(storage_destination)
                if storage_reservation is not None
                else None
            ),
        },
        "artifact_paths": [
            str(destination),
            *(
                [str(storage_destination)]
                if storage_reservation is not None
                else []
            ),
        ],
        "action": _publish_source_cache_audit,
    }


def build_source_preflight_task_specs(
    *,
    runtime_data_config_path: str | Path | None = None,
    source_preflight_config_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """Bind the registry's source node to the authenticated runtime sources."""

    if (runtime_data_config_path is None) == (
        source_preflight_config_path is None
    ):
        raise ValueError("supply exactly one source-preflight config path")
    path = Path(
        source_preflight_config_path
        if source_preflight_config_path is not None
        else runtime_data_config_path
    ).resolve()
    payload = load_hashed_json(path)
    if source_preflight_config_path is None:
        validate_runtime_data_config(payload, verify_cache_files=False)
    else:
        validate_content_hash(
            payload,
            expected_contract=PARTICLE_VIEW_SOURCE_PREFLIGHT_CONFIG_CONTRACT,
        )
        validate_runtime_data_config(
            payload["runtime_data_config"], verify_cache_files=False
        )
        validate_content_hash(
            payload["storage_reservation"],
            expected_contract="particle_view_storage_reservation_v1",
        )
    return {
        "PV_SOURCE_PREFLIGHT": {
            "operation": "source_preflight",
            "factory": (
                "teacher_logit_reco.local_particle_residual_field."
                "particle_view.production_factories:"
                "build_source_preflight_factory"
            ),
            "factory_config_path": str(path),
            "factory_config_sha256": sha256_file(path),
        }
    }


__all__ = [
    "build_source_preflight_factory",
    "build_source_preflight_task_specs",
]
