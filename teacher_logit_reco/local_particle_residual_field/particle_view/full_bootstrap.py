"""One-command scientific bootstrap for the complete particle-view pilot.

This module closes the gap between the individually authenticated production
factories and the Slurm graph: it publishes every factory config, merges exact
run coverage, builds the locked scientific catalog, and emits handler commands
that the existing runtime-manifest/submission machinery consumes directly.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .confirmation_runtime import (
    build_confirmation_factory_config,
    build_confirmation_task_specs,
)
from .consumer_interface_runtime import (
    build_consumer_screen_factory_config,
    build_consumer_screen_task_specs,
)
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .direct_control import (
    build_stage_a_direct_resource_plan,
    validate_stage_a_direct_resource_plan,
)
from .distillation_runtime import (
    build_distillation_factory_config,
    build_distillation_task_specs,
)
from .fairness_runtime import (
    build_fairness_factory_config,
    build_fairness_task_specs,
)
from .final_runtime import (
    build_final_factory_config,
    build_final_task_specs,
)
from .focused_control_runtime import (
    build_focused_control_factory_config,
    build_focused_control_task_specs,
)
from .post_target_runtime import (
    build_post_target_factory_config,
    build_post_target_task_specs,
)
from .production_factories import (
    PARTICLE_VIEW_SOURCE_PREFLIGHT_CONFIG_CONTRACT,
    build_baseline_factory_config,
    build_direct_control_factory_config,
    build_source_preflight_task_specs,
    build_stage_a_direct_task_specs,
    build_stage_a_teacher_task_specs,
)
from .storage import build_diagnostic_budget, build_storage_reservation
from .tap_staging import build_tap_stage_reservation
from .registry import validate_particle_view_registry
from .report_runtime import (
    build_report_factory_config,
    build_report_task_specs,
)
from .runtime_data import validate_runtime_data_config
from .scientific_tasks import (
    build_scientific_handler_commands,
    build_scientific_task_catalog,
    validate_scientific_task_catalog,
)
from .stack_runtime import (
    build_stack_factory_config,
    build_stack_task_specs,
)
from .target_runtime import (
    build_target_discovery_factory_config,
    build_target_discovery_task_specs,
)
from .target_selection_runtime import (
    build_target_selection_factory_config,
    build_target_selection_task_specs,
)


PARTICLE_VIEW_FULL_BOOTSTRAP_CONTRACT = (
    "particle_view_full_scientific_bootstrap_v1"
)

_CONFIG_FILENAMES = {
    "runtime_data": "runtime_data_config.json",
    "source_preflight": "source_preflight_factory_config.json",
    "direct_resource_plan": "stage_a_direct_resource_plan.json",
    "baseline": "baseline_factory_config.json",
    "direct_control": "direct_control_factory_config.json",
    "target_discovery": "target_discovery_factory_config.json",
    "consumer_screen": "consumer_screen_factory_config.json",
    "target_selection": "target_selection_factory_config.json",
    "post_target": "post_target_factory_config.json",
    "distillation": "distillation_factory_config.json",
    "focused_control": "focused_control_factory_config.json",
    "confirmation": "confirmation_factory_config.json",
    "fairness": "fairness_factory_config.json",
    "stack": "stack_factory_config.json",
    "report": "report_factory_config.json",
    "final": "final_factory_config.json",
}


def _write_plain_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace bootstrap file {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _merge_task_specs(
    fragments: Sequence[Mapping[str, Mapping[str, str]]],
    *,
    registry: Mapping[str, Any],
    expected_run_ids: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for fragment in fragments:
        for run_id, spec in fragment.items():
            if run_id in merged:
                raise ValueError(f"duplicate full-bootstrap run {run_id}")
            merged[run_id] = dict(spec)
    registered = (
        {row["run_id"] for row in registry["runs"]}
        if expected_run_ids is None
        else set(expected_run_ids)
    )
    if set(merged) != registered:
        raise ValueError(
            "full-bootstrap task coverage mismatch: "
            f"missing={sorted(registered - set(merged))}, "
            f"extra={sorted(set(merged) - registered)}"
        )
    return {run_id: merged[run_id] for run_id in sorted(merged)}


def _npy_header(handle: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
    else:
        shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _cache_shape_inventory(
    runtime_data_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    inventory = []
    total_bytes = 0
    for record in runtime_data_config["parent_cache_records"]:
        for source_kind in ("hlt_array", "offline_array"):
            binding = record[source_kind]
            path = Path(binding["path"])
            arrays = []
            if path.suffix == ".npz":
                with zipfile.ZipFile(path) as archive:
                    for member in sorted(archive.namelist()):
                        if not member.endswith(".npy"):
                            continue
                        with archive.open(member) as handle:
                            shape, dtype = _npy_header(handle)
                        arrays.append(
                            {
                                "name": Path(member).stem,
                                "shape": list(shape),
                                "dtype": dtype.str,
                                "logical_bytes": int(
                                    np.prod(shape, dtype=np.int64)
                                    * dtype.itemsize
                                ),
                            }
                        )
            elif path.suffix == ".npy":
                with path.open("rb") as handle:
                    shape, dtype = _npy_header(handle)
                arrays.append(
                    {
                        "name": path.stem,
                        "shape": list(shape),
                        "dtype": dtype.str,
                        "logical_bytes": int(
                            np.prod(shape, dtype=np.int64) * dtype.itemsize
                        ),
                    }
                )
            else:
                raise ValueError(f"unsupported bound cache format: {path}")
            serialized_bytes = path.stat().st_size
            total_bytes += serialized_bytes
            inventory.append(
                {
                    "parent_split": record["parent_split"],
                    "source_kind": source_kind,
                    "path_sha256": binding["sha256"],
                    "serialized_bytes": serialized_bytes,
                    "arrays": arrays,
                }
            )
    return inventory, total_bytes


def _representative_serialized_model_sizes(
    resource_plan: Mapping[str, Any],
) -> dict[str, int]:
    """Measure state-dict serialization; retain an exact formula fallback."""

    target = resource_plan["canonical_target"]
    from jetclass_fresh.hlt_baseline import default_part_config
    from jetclass_fresh.jetclass_data import LABEL_NAMES
    from .direct_control import particle_transformer_parameter_count

    measured = {
        "canonical_deployable_float32_parameter_bytes": (
            int(target["deployed_parameters"]) * 4
        )
    }
    for model_size in ("base", "large"):
        config = default_part_config(
            num_classes=len(LABEL_NAMES), model_size=model_size
        )
        measured[f"teacher_{model_size}_state_dict_bytes"] = int(
            particle_transformer_parameter_count(config) * 4 * 1.01
        )
    try:
        import torch
        from jetclass_fresh.hlt_baseline import (
            build_particle_transformer_classifier,
        )
        from jetclass_fresh.jetclass_data import LABEL_NAMES

        for model_size in ("base", "large"):
            model = build_particle_transformer_classifier(
                num_classes=len(LABEL_NAMES),
                model_size=model_size,
            )
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            measured[f"teacher_{model_size}_state_dict_bytes"] = len(
                buffer.getbuffer()
            )
            del model
    except (ImportError, RuntimeError):
        pass
    return measured


def _checkpoint_multiplicity(run_id: str, operation: str) -> int:
    fixed = {
        "source_preflight": 0,
        "teacher_training": 1,
        "existing_teacher_registration": 0,
        "direct_control_training": 1,
        "consumer_interface_screen": 1,
        "configuration_selection": 0,
        "selected_view_publication": 0,
        "consumer_training": 1,
        "pview0_training": 3,
        "residual_sampler_fit": 0,
        "robust_consumer_training": 1,
        "frozen_distillation": 1,
        "joint_finetuning": 1,
        "trained_control_training": 1,
        "structural_control_evaluation": 0,
        "confirmation_training": 2,
        "stack_evaluation": 0,
        "fusion": 0,
        "reporting": 0,
        "bundle_export": 1,
        "bundle_reload": 0,
        "final_test": 0,
    }
    if operation == "target_discovery":
        return 8 if run_id == "VGEN_RECODESIGN" else 4
    if operation == "focused_composite_training":
        return 9
    if operation == "fairness_closure":
        if run_id == "SELECTED_PATH_FAIRNESS_LEDGER":
            return 0
        if run_id.endswith(
            ("A0_VIEW_LONG_DEPLOY", "A0_VIEW_TOTAL_LABEL_BUDGET")
        ):
            return 2
        # A winner-family alias creates no new checkpoint when both families
        # resolve identically. That fact is unknowable before selection, so
        # preflight safely reserves the distinct-family one-checkpoint case.
        return 1
    if operation not in fixed:
        raise ValueError(f"checkpoint inventory lacks operation {operation!r}")
    return fixed[operation]


def _derived_storage_reservation_inputs(
    *,
    registry: Mapping[str, Any],
    runtime_data_config: Mapping[str, Any],
    resource_plan: Mapping[str, Any],
    task_specs: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, int], list[dict[str, Any]], int, dict[str, Any]]:
    from jetclass_fresh.hlt_baseline import default_part_config
    from jetclass_fresh.jetclass_data import LABEL_NAMES

    unified = load_hashed_json(runtime_data_config["unified_manifest"]["path"])
    logical = unified["logical_splits"]
    max_particles = int(unified["split_config"]["max_particles"])
    cache_inventory, source_cache_bytes = _cache_shape_inventory(
        runtime_data_config
    )
    representative_sizes = _representative_serialized_model_sizes(
        resource_plan
    )
    runs = {row["run_id"]: row for row in registry["runs"]}
    if set(task_specs) != set(runs) - {"PV_SOURCE_PREFLIGHT"}:
        raise ValueError("storage task-operation inventory is incomplete")
    representative_checkpoint_bytes = max(
        int(
            representative_sizes.get(
                "teacher_base_state_dict_bytes",
                resource_plan["canonical_target"]["deployed_parameters"] * 4,
            )
        ),
        int(resource_plan["canonical_target"]["deployed_parameters"]) * 4,
    )
    checkpoint_inventory = []
    checkpoint_count = 0
    retained_checkpoint_bytes = 0
    for run_id in sorted(task_specs):
        operation = task_specs[run_id]["operation"]
        multiplicity = _checkpoint_multiplicity(run_id, operation)
        seed_count = len(runs[run_id]["seed_ids"])
        task_checkpoint_count = multiplicity * seed_count
        size_role = "canonical_deployable"
        unit_bytes = representative_checkpoint_bytes
        if operation == "teacher_training":
            architecture = "large" if "LARGE" in run_id.upper() else "base"
            size_role = f"teacher_{architecture}"
            unit_bytes = int(
                representative_sizes[f"teacher_{architecture}_state_dict_bytes"]
            )
        checkpoint_count += task_checkpoint_count
        retained_checkpoint_bytes += task_checkpoint_count * unit_bytes
        checkpoint_inventory.append(
            {
                "run_id": run_id,
                "operation": operation,
                "seed_count": seed_count,
                "checkpoint_multiplicity_per_task": multiplicity,
                "reserved_checkpoint_count": task_checkpoint_count,
                "representative_size_role": size_role,
                "representative_bytes": unit_bytes,
                "reserved_bytes": task_checkpoint_count * unit_bytes,
                "conditional_alias_creates_no_checkpoint": (
                    operation == "fairness_closure"
                    and run_id.startswith("FAIR_PRIVILEGED_SCIENTIFIC_")
                ),
            }
        )
    retained_checkpoint_bytes = int(retained_checkpoint_bytes * 1.10)
    seed_expanded_tasks = sum(
        len(row["seed_ids"]) for row in registry["runs"]
    )
    json_bytes = max(64 * 1024**2, seed_expanded_tasks * 2 * 1024**2)
    selected_count = sum(
        int(logical[name]["count"])
        for name in ("train", "model_val_stop", "model_val_select")
    )
    selected_view_bytes = selected_count * max_particles * 8 * 4
    selected_mask_bytes = selected_count * max_particles
    target_logit_npz_count_per_task = 3
    target_logit_npz_header_allowance_bytes = 4096
    per_task_target_logit_bytes = (
        selected_count * (len(LABEL_NAMES) * 4 + 8)
        + target_logit_npz_count_per_task
        * target_logit_npz_header_allowance_bytes
    )
    logit_operations = {
        "frozen_distillation",
        "joint_finetuning",
        "focused_composite_training",
        "trained_control_training",
        "confirmation_training",
    }
    target_logit_cache_task_count = sum(
        len(runs[run_id]["seed_ids"])
        for run_id, spec in task_specs.items()
        if spec["operation"] in logit_operations
    )
    selected_logit_bytes = (
        per_task_target_logit_bytes * target_logit_cache_task_count
    )
    selected_product_bytes = int(
        (selected_view_bytes + selected_mask_bytes + selected_logit_bytes)
        * 1.10
    )

    train = logical["train"]
    tap_rows = []
    for role, architecture, source_role in (
        ("A0_VIEW", "base", "hlt_memory_control"),
        ("TOFF_VIEW_BASE", "base", "offline_teacher"),
        ("TOFF_VIEW_LARGE", "large", "offline_teacher"),
    ):
        architecture_config = default_part_config(
            num_classes=len(LABEL_NAMES),
            model_size=architecture,
        )
        token_width = int(architecture_config["embed_dims"][-1])
        planned_teacher_identity = canonical_sha256(
            {
                "contract": "particle_view_planned_teacher_reservation_v1",
                "role": role,
                "architecture": architecture_config,
                "registry_sha256": registry["content_hash"],
            }
        )
        tap_spec_sha256 = canonical_sha256(
            {
                "contract": "particle_view_planned_tap_spec_v1",
                "role": role,
                "tap_choice": "penultimate",
                "token_width": token_width,
            }
        )
        tap_rows.append(
            build_tap_stage_reservation(
                source_role=source_role,
                source_manifest_sha256=runtime_data_config["content_hash"],
                logical_split_sha256=train["content_hash"],
                ordered_identity_sha256=train["ordered_identity_sha256"],
                teacher_checkpoint_sha256=planned_teacher_identity,
                tap_spec_sha256=tap_spec_sha256,
                jets=int(train["count"]),
                max_particles=max_particles,
                token_width=token_width,
                identity_columns=1,
            )
        )
    transient_ram_bytes = source_cache_bytes + 8 * 1024**3
    planned = {
        "retained_checkpoints_and_bundles": max(
            retained_checkpoint_bytes, 128 * 1024**2
        ),
        "json_metrics_registries_and_reports": json_bytes,
        "selected_view_and_logit_products": max(
            selected_product_bytes, 32 * 1024**2
        ),
    }
    evidence = {
        "contract": "particle_view_storage_derivation_evidence_v1",
        "cache_shape_dtype_inventory": cache_inventory,
        "source_cache_serialized_bytes": source_cache_bytes,
        "teacher_tap_split": "train",
        "teacher_tap_split_count": int(train["count"]),
        "teacher_tap_max_particles": max_particles,
        "seed_expanded_checkpoint_count": checkpoint_count,
        "checkpoint_inventory_by_operation": checkpoint_inventory,
        "checkpoint_retention_basis": (
            "every task-result-bound checkpoint remains restart-authenticated"
        ),
        "conditional_stage_g_alias_policy": (
            "reserve distinct-family checkpoint; identical-family alias adds zero"
        ),
        "seed_expanded_task_count": seed_expanded_tasks,
        "representative_serialized_model_sizes": representative_sizes,
        "selected_product_split_counts": {
            name: int(logical[name]["count"])
            for name in ("train", "model_val_stop", "model_val_select")
        },
        "selected_product_max_view_dim": 8,
        "selected_product_class_count": len(LABEL_NAMES),
        "target_logit_cache_task_count": target_logit_cache_task_count,
        "per_task_target_logit_bytes": per_task_target_logit_bytes,
        "target_logit_dtype": "float32",
        "target_logit_event_id_dtype": "int64",
        "target_logit_npz_count_per_task": target_logit_npz_count_per_task,
        "target_logit_npz_header_allowance_bytes": (
            target_logit_npz_header_allowance_bytes
        ),
        "transient_ram_formula": "serialized_source_caches_plus_8GiB_workspace",
    }
    return planned, tap_rows, transient_ram_bytes, evidence


def publish_full_pilot_scientific_bootstrap(
    *,
    output_dir: str | Path,
    registry: Mapping[str, Any],
    runtime_data_config: Mapping[str, Any],
    source_commit: str,
    device: str = "auto",
    num_workers: int = 0,
    amp: bool = True,
    batch_size: int = 128,
    bootstrap_replicates: int = 10_000,
    linear_fusion_steps: int = 300,
    optional_p7b_resource: Mapping[str, Any] | None = None,
    existing_checkpoint_path: str | Path | None = None,
    existing_observed_train_identity_sha256: str | None = None,
    existing_serialized_recipe_path: str | Path | None = None,
    existing_recipe_reproduced_exactly: bool = False,
    existing_provenance_metadata_sha256: str | None = None,
    existing_description: str = "pre-existing offline particle teacher",
    existing_teacher_compatible: bool = False,
    teacher_mix_compatible: bool = False,
    python_executable: str | Path = sys.executable,
    handler_python_executable: str = "python",
    reload_fixture_batch_size: int = 8,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_stack_batches: int | None = None,
    production: bool = True,
    persistent_storage_budget_bytes: int = 32 * 1024**3,
    allocation_ram_bytes: int = 128 * 1024**3,
) -> dict[str, Any]:
    """Publish the complete, exactly covered scientific execution bootstrap."""

    registry_audit = validate_particle_view_registry(registry)
    validate_runtime_data_config(runtime_data_config, verify_cache_files=True)
    if (
        registry["unified_split_manifest_sha256"]
        != runtime_data_config["unified_manifest"]["manifest_sha256"]
    ):
        raise ValueError("registry and runtime data use different split manifests")
    if production and any(
        value is not None
        for value in (
            max_train_batches,
            max_val_batches,
            max_stack_batches,
        )
    ):
        raise ValueError(
            "production bootstrap forbids partial-batch rehearsal limits"
        )
    if not isinstance(source_commit, str) or len(source_commit) not in {40, 64}:
        raise ValueError("source_commit must be a full Git commit digest")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    resource_plan = build_stage_a_direct_resource_plan()
    validate_stage_a_direct_resource_plan(resource_plan)
    baseline = build_baseline_factory_config(
        runtime_data_config=runtime_data_config,
        device=device,
        num_workers=num_workers,
        amp=amp,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        existing_checkpoint_path=existing_checkpoint_path,
        existing_observed_train_identity_sha256=(
            existing_observed_train_identity_sha256
        ),
        existing_serialized_recipe_path=existing_serialized_recipe_path,
        existing_recipe_reproduced_exactly=(
            existing_recipe_reproduced_exactly
        ),
        existing_provenance_metadata_sha256=(
            existing_provenance_metadata_sha256
        ),
        existing_description=existing_description,
    )
    direct = build_direct_control_factory_config(
        runtime_data_config=runtime_data_config,
        resource_plan=resource_plan,
        device=device,
        num_workers=num_workers,
        amp=amp,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    target = build_target_discovery_factory_config(
        runtime_data_config=runtime_data_config,
        device=device,
        num_workers=num_workers,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        existing_teacher_compatible=existing_teacher_compatible,
        teacher_mix_compatible=teacher_mix_compatible,
        baseline_factory_config=baseline,
    )
    consumer = build_consumer_screen_factory_config(
        runtime_data_config=runtime_data_config,
        device=device,
        num_workers=num_workers,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    target_selection = build_target_selection_factory_config(
        source_commit=source_commit
    )
    post_target = build_post_target_factory_config(
        runtime_data_config=runtime_data_config,
        device=device,
        num_workers=num_workers,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    distillation = build_distillation_factory_config(
        runtime_data_config=runtime_data_config,
        device=device,
        num_workers=num_workers,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    focused = build_focused_control_factory_config(
        distillation_factory_config=distillation
    )
    confirmation = build_confirmation_factory_config(
        distillation_factory_config=distillation
    )
    fairness = build_fairness_factory_config(
        runtime_data_config=runtime_data_config,
        device=device,
        num_workers=num_workers,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    stack = build_stack_factory_config(
        fairness_factory_config=fairness,
        device=device,
        num_workers=num_workers,
        batch_size=batch_size,
        max_stack_batches=max_stack_batches,
        bootstrap_replicates=bootstrap_replicates,
        linear_fusion_steps=linear_fusion_steps,
        optional_p7b_resource=optional_p7b_resource,
    )
    report = build_report_factory_config(
        stack_factory_config=stack,
        source_commit=source_commit,
        python_executable=python_executable,
        reload_fixture_batch_size=reload_fixture_batch_size,
    )
    final = build_final_factory_config(
        report_factory_config=report,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        bootstrap_replicates=bootstrap_replicates,
    )
    configs = {
        "runtime_data": dict(runtime_data_config),
        "direct_resource_plan": resource_plan,
        "baseline": baseline,
        "direct_control": direct,
        "target_discovery": target,
        "consumer_screen": consumer,
        "target_selection": target_selection,
        "post_target": post_target,
        "distillation": distillation,
        "focused_control": focused,
        "confirmation": confirmation,
        "fairness": fairness,
        "stack": stack,
        "report": report,
        "final": final,
    }
    config_paths = {
        name: root / filename for name, filename in _CONFIG_FILENAMES.items()
    }
    # Publish all non-source configs first so checkpoint retention can be
    # derived from the same exact operation inventory the runtime will use.
    for name, payload in configs.items():
        write_immutable_json(config_paths[name], payload)
    non_source_fragments = [
        build_stage_a_teacher_task_specs(
            factory_config_path=config_paths["baseline"]
        ),
        build_stage_a_direct_task_specs(
            factory_config_path=config_paths["direct_control"]
        ),
        build_target_discovery_task_specs(
            factory_config_path=config_paths["target_discovery"]
        ),
        build_consumer_screen_task_specs(
            factory_config_path=config_paths["consumer_screen"]
        ),
        build_target_selection_task_specs(
            factory_config_path=config_paths["target_selection"]
        ),
        build_post_target_task_specs(
            factory_config_path=config_paths["post_target"]
        ),
        build_distillation_task_specs(
            factory_config_path=config_paths["distillation"]
        ),
        build_focused_control_task_specs(
            factory_config_path=config_paths["focused_control"]
        ),
        build_confirmation_task_specs(
            factory_config_path=config_paths["confirmation"]
        ),
        build_fairness_task_specs(
            factory_config_path=config_paths["fairness"]
        ),
        build_stack_task_specs(factory_config_path=config_paths["stack"]),
        build_report_task_specs(factory_config_path=config_paths["report"]),
        build_final_task_specs(factory_config_path=config_paths["final"]),
    ]
    registered_run_ids = {row["run_id"] for row in registry["runs"]}
    non_source_specs = _merge_task_specs(
        non_source_fragments,
        registry=registry,
        expected_run_ids=registered_run_ids - {"PV_SOURCE_PREFLIGHT"},
    )
    # Measure an actual published campaign artifact, reserve every planned
    # persistent class, and bind the reservation into PV00 before submission.
    campaign_root = root.parent.parent
    campaign_root.mkdir(parents=True, exist_ok=True)
    diagnostic_budget = build_diagnostic_budget()
    (
        planned_persistent_bytes,
        tap_stage_reservations,
        transient_ram_bytes,
        storage_derivation_evidence,
    ) = _derived_storage_reservation_inputs(
        registry=registry,
        runtime_data_config=runtime_data_config,
        resource_plan=resource_plan,
        task_specs=non_source_specs,
    )
    storage_reservation = build_storage_reservation(
        campaign_root=campaign_root,
        measured_artifacts={
            "runtime_data_config": config_paths["runtime_data"],
        },
        planned_persistent_bytes=planned_persistent_bytes,
        tap_stage_reservations=tap_stage_reservations,
        persistent_budget_bytes=int(persistent_storage_budget_bytes),
        filesystem_available_bytes=int(
            shutil.disk_usage(campaign_root).free
        ),
        allocation_ram_bytes=int(allocation_ram_bytes),
        transient_ram_bytes=transient_ram_bytes,
        diagnostic_budget=diagnostic_budget,
        derivation_evidence=storage_derivation_evidence,
    )
    source_preflight = with_content_hash(
        {
            "contract": PARTICLE_VIEW_SOURCE_PREFLIGHT_CONFIG_CONTRACT,
            "runtime_data_config": dict(runtime_data_config),
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "storage_reservation": storage_reservation,
            "storage_reservation_sha256": storage_reservation["content_hash"],
            "diagnostic_budget": diagnostic_budget,
            "diagnostic_budget_sha256": diagnostic_budget["content_hash"],
            "storage_is_execution_gating": True,
            "scientific_metrics_are_execution_gating": False,
        }
    )
    configs["source_preflight"] = source_preflight
    for name, payload in configs.items():
        write_immutable_json(config_paths[name], payload)

    fragments = [
        build_source_preflight_task_specs(
            source_preflight_config_path=config_paths["source_preflight"]
        ),
        *non_source_fragments,
    ]
    specs = _merge_task_specs(fragments, registry=registry)
    specs_path = root / "scientific_task_specs.json"
    _write_plain_json(specs_path, specs)
    catalog_path = root / "scientific_task_catalog.json"
    catalog = build_scientific_task_catalog(
        registry=registry, task_specs=specs
    )
    validate_scientific_task_catalog(catalog, registry=registry)
    write_immutable_json(catalog_path, catalog)
    handlers = build_scientific_handler_commands(
        catalog=catalog,
        catalog_path=str(catalog_path),
        python_executable=handler_python_executable,
    )
    handler_path = root / "scientific_handler_commands.json"
    _write_plain_json(handler_path, handlers)
    index = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FULL_BOOTSTRAP_CONTRACT,
            "registry_sha256": registry["content_hash"],
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "source_commit": source_commit,
            "production": bool(production),
            "partial_batch_limits": {
                "max_train_batches": max_train_batches,
                "max_val_batches": max_val_batches,
                "max_stack_batches": max_stack_batches,
            },
            "factory_config_files": {
                name: {
                    "path": str(config_paths[name]),
                    "sha256": sha256_file(config_paths[name]),
                    "content_hash": configs[name]["content_hash"],
                }
                for name in sorted(configs)
            },
            "scientific_task_specs": {
                "path": str(specs_path),
                "sha256": sha256_file(specs_path),
            },
            "scientific_task_catalog": {
                "path": str(catalog_path),
                "sha256": sha256_file(catalog_path),
                "content_hash": catalog["content_hash"],
            },
            "scientific_handler_commands": {
                "path": str(handler_path),
                "sha256": sha256_file(handler_path),
            },
            "registered_run_count": registry_audit["run_count"],
            "covered_run_count": len(specs),
            "seed_expanded_task_count": sum(
                len(row["seed_ids"]) for row in registry["runs"]
            ),
            "category_count": len(handlers),
            "exact_registry_coverage": True,
            "source_preflight_included": True,
            "report_and_final_included": True,
            "performance_gates": False,
            "quality_warnings_stop_execution": False,
        }
    )
    write_immutable_json(root / "full_scientific_bootstrap.json", index)
    return index


def validate_full_pilot_scientific_bootstrap(
    payload: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    verify_files: bool = True,
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_FULL_BOOTSTRAP_CONTRACT
    )
    audit = validate_particle_view_registry(registry)
    if (
        payload["registry_sha256"] != registry["content_hash"]
        or payload["registered_run_count"] != audit["run_count"]
        or payload["covered_run_count"] != audit["run_count"]
        or payload["exact_registry_coverage"] is not True
        or payload["source_preflight_included"] is not True
        or payload["report_and_final_included"] is not True
        or payload["performance_gates"] is not False
        or payload["quality_warnings_stop_execution"] is not False
    ):
        raise ValueError("full scientific bootstrap policy/coverage changed")
    if payload["production"] and any(
        value is not None for value in payload["partial_batch_limits"].values()
    ):
        raise ValueError("production bootstrap contains rehearsal limits")
    if verify_files:
        bindings = [
            *payload["factory_config_files"].values(),
            payload["scientific_task_specs"],
            payload["scientific_task_catalog"],
            payload["scientific_handler_commands"],
        ]
        for binding in bindings:
            path = Path(binding["path"])
            if (
                path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != binding["sha256"]
            ):
                raise ValueError("full bootstrap artifact changed")
        catalog = load_hashed_json(
            payload["scientific_task_catalog"]["path"]
        )
        validate_scientific_task_catalog(catalog, registry=registry)
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "run_count": payload["covered_run_count"],
        "task_count": payload["seed_expanded_task_count"],
    }


__all__ = [
    "PARTICLE_VIEW_FULL_BOOTSTRAP_CONTRACT",
    "publish_full_pilot_scientific_bootstrap",
    "validate_full_pilot_scientific_bootstrap",
]
