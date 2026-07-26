"""One-command scientific bootstrap for the complete particle-view pilot.

This module closes the gap between the individually authenticated production
factories and the Slurm graph: it publishes every factory config, merges exact
run coverage, builds the locked scientific catalog, and emits handler commands
that the existing runtime-manifest/submission machinery consumes directly.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .confirmation_runtime import (
    build_confirmation_factory_config,
    build_confirmation_task_specs,
)
from .consumer_interface_runtime import (
    build_consumer_screen_factory_config,
    build_consumer_screen_task_specs,
)
from .contracts import (
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
    build_baseline_factory_config,
    build_direct_control_factory_config,
    build_source_preflight_task_specs,
    build_stage_a_direct_task_specs,
    build_stage_a_teacher_task_specs,
)
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
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for fragment in fragments:
        for run_id, spec in fragment.items():
            if run_id in merged:
                raise ValueError(f"duplicate full-bootstrap run {run_id}")
            merged[run_id] = dict(spec)
    registered = {row["run_id"] for row in registry["runs"]}
    if set(merged) != registered:
        raise ValueError(
            "full-bootstrap task coverage mismatch: "
            f"missing={sorted(registered - set(merged))}, "
            f"extra={sorted(set(merged) - registered)}"
        )
    return {run_id: merged[run_id] for run_id in sorted(merged)}


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
    for name, payload in configs.items():
        write_immutable_json(config_paths[name], payload)

    fragments = [
        build_source_preflight_task_specs(
            runtime_data_config_path=config_paths["runtime_data"]
        ),
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
