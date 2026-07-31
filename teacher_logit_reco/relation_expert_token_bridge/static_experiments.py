"""Compile the immutable, selector-independent RETB experiment surface.

The scientific registries deliberately retain duplicate *memberships* when one
physical run participates in several screens.  Slurm task manifests must not
train those runs twice, so this module deduplicates by run ID while preserving
every registry membership in the authenticated plan.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .contracts import (
    bind_source,
    canonical_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .production import (
    build_task_manifest,
    validate_production_campaign_binding,
    validate_production_graph,
    validate_task_manifest_for_graph,
    task_manifest_path_for_graph,
)
from .step4 import (
    build_stage_b_run_registry,
    resolve_stage_b_run,
    validate_stage_b_run_registry,
)
from .step5 import (
    build_stage_c_run_registry,
    resolve_expert_confirmation_training_run,
    resolve_stage_c_run,
    validate_stage_c_run_registry,
)
from .step6 import (
    build_stage_d_run_registry,
    resolve_stage_d_run,
    validate_stage_d_run_registry,
)
from .step7 import (
    build_stage_e_template_registry,
    validate_stage_e_template_registry,
)


STATIC_EXPERIMENT_PLAN_CONTRACT = "retb_static_experiment_plan_v5"
STATIC_EXPERIMENT_BUNDLE_CONTRACT = "retb_static_experiment_bundle_v5"

STATIC_MANIFEST_NODES = (
    "offline_expert_training",
    "offline_expert_confirmation",
    "offline_fusion_cache",
    "offline_fusion_training",
    "native_hlt_expert_training",
    "native_hlt_fusion_training",
    "bridge_pilot_training",
)

_STAGE_B_SECTIONS = (
    "primary_shape_screen",
    "tokenizer_controls",
    "dual_topology_controls",
    "representative_expert_loss_rows",
    "full_optimization_grid",
    "fixed_followup_references",
)
_STAGE_C_FUSION_SECTIONS = (
    "canonical_fusion_rows",
    "uniform_control_rows",
)
_STAGE_D_EXPERT_SECTIONS = (
    "scratch_expert_rows",
    "encoder_screen_rows",
    "bridge_parent_expert_rows",
)


def _source_snapshot(campaign: Mapping[str, Any]) -> dict[str, Any]:
    source = campaign["source"]
    return {
        "source_commit": source["commit"],
        "source_status_sha256": source["status_sha256"],
        "source_dirty": source["dirty"],
    }


def _source_bound_registries(
    campaign: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    snapshot = _source_snapshot(campaign)
    return {
        "stage_b": bind_source(
            build_stage_b_run_registry(), source_snapshot=snapshot
        ),
        "stage_c": bind_source(
            build_stage_c_run_registry(), source_snapshot=snapshot
        ),
        "stage_d": bind_source(
            build_stage_d_run_registry(), source_snapshot=snapshot
        ),
        "stage_e": bind_source(
            build_stage_e_template_registry(), source_snapshot=snapshot
        ),
    }


def _physical_rows(
    registry: Mapping[str, Any],
    sections: Sequence[str],
    *,
    resolver: Any,
) -> list[dict[str, Any]]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for section in sections:
        for row in registry[section]:
            run_id = str(row["run_id"])
            if run_id not in seen:
                seen.add(run_id)
                ordered_ids.append(run_id)
    return [resolver(registry, run_id=run_id) for run_id in ordered_ids]


def _path(root: Path, *parts: str) -> str:
    return str(root.joinpath(*parts))


def _deferred(
    path: str,
    *,
    contract: str,
    producer: str,
    role: str,
) -> dict[str, str]:
    return {
        "path": path,
        "expected_contract": contract,
        "producer": producer,
        "role": role,
    }


def _record(
    *,
    node_id: str,
    static_id: str,
    stage: str,
    component: str,
    seed: int,
    run_id: str | None,
    configuration: Mapping[str, Any],
    registry_sha256: str,
    argv: Sequence[str],
    expected_artifacts: Sequence[str],
    deferred_inputs: Sequence[Mapping[str, str]],
    registry_memberships: Sequence[Mapping[str, Any]] = (),
    run_id_resolution: str = "already_resolved_from_static_registry",
) -> dict[str, Any]:
    config = dict(configuration)
    configuration_sha = canonical_sha256(config)
    command = [str(value) for value in argv]
    outputs = [str(value) for value in expected_artifacts]
    if (
        not static_id
        or int(seed) < 0
        or not command
        or not outputs
        or len(outputs) != len(set(outputs))
    ):
        raise ValueError("static experiment row dimensions differ")
    payload = {
        "static_id": str(static_id),
        "node_id": str(node_id),
        "stage": str(stage),
        "component": str(component),
        "seed": int(seed),
        "run_id": None if run_id is None else str(run_id),
        "run_id_resolution": str(run_id_resolution),
        "configuration": config,
        "configuration_sha256": configuration_sha,
        "registry_sha256": str(registry_sha256),
        "argv": command,
        "command_sha256": canonical_sha256(command),
        "environment": {
            "RETB_CONFIGURATION_SHA256": configuration_sha,
            "RETB_RUN_SEED": str(int(seed)),
            "RETB_STATIC_ID": str(static_id),
            **(
                {}
                if run_id is None
                else {"RETB_RUN_ID": str(run_id)}
            ),
        },
        "deferred_inputs": [dict(value) for value in deferred_inputs],
        "expected_artifacts": outputs,
        "registry_memberships": [dict(value) for value in registry_memberships],
        "selector_result_consumed_to_define_row": False,
        "scientific_underperformance_skips_row": False,
    }
    payload["row_sha256"] = canonical_sha256(payload)
    return payload


def _stage_b_records(
    *,
    root: Path,
    python: str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _physical_rows(
        registry, _STAGE_B_SECTIONS, resolver=resolve_stage_b_run
    )
    output = []
    for row in rows:
        run_id = str(row["run_id"])
        seed = int(row["seed"])
        config = row["configuration"]
        run_root = root / "runs" / "stage_b" / run_id / f"seed_{seed}"
        argv = [
            python,
            "scripts/train_retb_offline_expert.py",
            "--campaign-root",
            str(root),
            "--run-id",
            run_id,
            "--train-npz",
            _path(root, "inputs", "offline", "model_train", "offline_inputs.npz"),
            "--val-stop-npz",
            _path(root, "inputs", "offline", "val_stop", "offline_inputs.npz"),
            "--relation-normalization",
            _path(
                root,
                "inputs",
                "normalization",
                "offline_500k",
                "relation.json",
            ),
            "--region-normalization",
            _path(
                root,
                "inputs",
                "normalization",
                "offline_500k",
                "region.json",
            ),
            "--region-tree-root",
            _path(root, "inputs", "region_tree", "offline"),
            "--resource-profile",
            _path(root, "job_ledgers", "resource_probes", "gpu.json"),
            "--output-dir",
            str(run_root),
        ]
        teacher = config.get("loss_id")
        if teacher not in {None, "ELOSS_CE"}:
            argv.extend(
                [
                    "--teacher-logits-manifest",
                    _path(
                        root,
                        "inputs",
                        "teacher_logits",
                        f"{teacher}.json",
                    ),
                ]
            )
            teacher_names = {
                "ELOSS_BASE_LOW": ("O_BASE",),
                "ELOSS_BASE": ("O_BASE",),
                "ELOSS_FULLREL": ("O_FULLREL",),
                "ELOSS_ENSEMBLE": ("O_BASE", "O_FULLREL"),
                "ELOSS_KD_DOMINANT": ("SELECTED_STRONGEST",),
            }[str(teacher)]
            for teacher_name in teacher_names:
                argv.extend(
                    [
                        "--teacher-checkpoint",
                        f"{teacher_name}="
                        + _path(
                            root,
                            "runs",
                            "stage_a",
                            "offline_controls",
                            teacher_name,
                            f"seed_{seed}",
                            "best_model_val.pt",
                        ),
                    ]
                )
        if config.get("initialization") != "INIT_SCRATCH":
            argv.extend(
                [
                    "--initialization-checkpoint",
                    _path(
                        root,
                        "runs",
                        "stage_a",
                        "offline_controls",
                        "O_BASE",
                        f"seed_{seed}",
                        "best_model_val.pt",
                    )
                ]
            )
        if config.get("initialization") == "INIT_ATTACH_AFTER_PRETRAIN":
            argv.extend(
                [
                    "--attachment-pretraining-record",
                    _path(
                        root,
                        "runs",
                        "stage_a",
                        "offline_controls",
                        "O_BASE",
                        f"seed_{seed}",
                        "attachment_pretraining_record.json",
                    ),
                ]
            )
        output.append(
            _record(
                node_id="offline_expert_training",
                static_id=run_id,
                stage="B",
                component=str(row["component"]),
                seed=seed,
                run_id=run_id,
                configuration=config,
                registry_sha256=registry["content_hash"],
                argv=argv,
                expected_artifacts=[
                    str(run_root / "checkpoint_registration.json"),
                    str(run_root / "best_model_val.pt"),
                ],
                deferred_inputs=[
                    _deferred(
                        _path(
                            root,
                            "inputs",
                            "offline",
                            "model_train",
                            "offline_input_manifest.json",
                        ),
                        contract="retb_offline_input_cache_v1",
                        producer="offline_input_cache",
                        role="model_train_features_and_labels",
                    ),
                    _deferred(
                        _path(
                            root,
                            "inputs",
                            "normalization",
                            "stage_a_normalizer_bundle.json",
                        ),
                        contract="retb_stage_a_normalizer_bundle_v1",
                        producer="normalizers_500k",
                        role="offline_relation_and_REGION_normalizers",
                    ),
                ],
                registry_memberships=row.get("registry_memberships", ()),
            )
        )
    return output


def _fusion_cache(root: Path, *, domain: str, shape: str, seed: int, split: str) -> str:
    filename = (
        f"{split}_frozen_tokens.json"
        if domain == "offline"
        else f"{split}_native_hlt_tokens.json"
    )
    return _path(
        root,
        "inputs",
        "fusion_cache",
        domain,
        shape,
        f"seed_{seed}",
        split,
        filename,
    )


def _stage_c_expert_records(
    *,
    root: Path,
    python: str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for member in registry["expert_confirmation_rows"]:
        run = resolve_expert_confirmation_training_run(
            registry,
            run_id=str(member["run_id"]),
            _registry_validated=True,
        )
        run_id = str(run["run_id"])
        seed = int(run["seed"])
        config = run["configuration"]
        run_root = (
            root
            / "runs"
            / "stage_c"
            / "offline_experts"
            / run_id
            / f"seed_{seed}"
        )
        argv = [
            python,
            "scripts/train_retb_offline_expert.py",
            "--campaign-root",
            str(root),
            "--registry-stage",
            "C",
            "--run-id",
            run_id,
            "--train-npz",
            _path(
                root,
                "inputs",
                "offline",
                "model_train",
                "offline_inputs.npz",
            ),
            "--val-stop-npz",
            _path(
                root,
                "inputs",
                "offline",
                "val_stop",
                "offline_inputs.npz",
            ),
            "--relation-normalization",
            _path(
                root,
                "inputs",
                "normalization",
                "offline_500k",
                "relation.json",
            ),
            "--region-normalization",
            _path(
                root,
                "inputs",
                "normalization",
                "offline_500k",
                "region.json",
            ),
            "--region-tree-root",
            _path(root, "inputs", "region_tree", "offline"),
            "--resource-profile",
            _path(root, "job_ledgers", "resource_probes", "gpu.json"),
            "--output-dir",
            str(run_root),
        ]
        output.append(
            _record(
                node_id="offline_expert_confirmation",
                static_id=run_id,
                stage="C",
                component="OFFLINE_EXPERT_CONFIRMATION",
                seed=seed,
                run_id=run_id,
                configuration=config,
                registry_sha256=registry["content_hash"],
                argv=argv,
                expected_artifacts=[
                    str(run_root / "checkpoint_registration.json"),
                    str(run_root / "best_model_val.pt"),
                ],
                deferred_inputs=[
                    _deferred(
                        _path(
                            root,
                            "inputs",
                            "offline",
                            "model_train",
                            "offline_input_manifest.json",
                        ),
                        contract="retb_offline_input_cache_v1",
                        producer="offline_input_cache",
                        role="model_train_features_and_labels",
                    ),
                    _deferred(
                        _path(
                            root,
                            "inputs",
                            "normalization",
                            "stage_a_normalizer_bundle.json",
                        ),
                        contract="retb_stage_a_normalizer_bundle_v1",
                        producer="normalizers_500k",
                        role="offline_relation_and_REGION_normalizers",
                    ),
                ],
                registry_memberships=run["registry_memberships"],
            )
        )
    if len(output) != 147:
        raise RuntimeError(
            "Stage-C expert confirmation execution rows differ"
        )
    return output


def _stage_c_cache_records(
    *,
    root: Path,
    python: str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["configuration"]["shape_id"]),
            int(row["seed"]),
            str(row["configuration"]["expert_id"]),
        ): str(row["run_id"])
        for row in registry["expert_confirmation_rows"]
    }
    output = []
    for shape_id in registry["uniform_shape_order"]:
        for seed in registry["pipeline_seeds"]:
            for split in ("model_train", "val_stop", "val_design"):
                output_dir = (
                    root
                    / "inputs"
                    / "fusion_cache"
                    / "offline"
                    / shape_id
                    / f"seed_{seed}"
                    / split
                )
                argv = [
                    python,
                    "scripts/build_retb_frozen_token_cache.py",
                    "--campaign-root",
                    str(root),
                    "--split",
                    split,
                    "--pipeline-seed",
                    str(seed),
                    "--shape-id",
                    shape_id,
                    "--input-npz",
                    _path(
                        root,
                        "inputs",
                        "offline",
                        split,
                        "offline_inputs.npz",
                    ),
                    "--input-manifest",
                    _path(
                        root,
                        "inputs",
                        "offline",
                        split,
                        "offline_input_manifest.json",
                    ),
                    "--relation-normalization",
                    _path(
                        root,
                        "inputs",
                        "normalization",
                        "offline_500k",
                        "relation.json",
                    ),
                    "--region-normalization",
                    _path(
                        root,
                        "inputs",
                        "normalization",
                        "offline_500k",
                        "region.json",
                    ),
                    "--region-tree-root",
                    _path(root, "inputs", "region_tree", "offline"),
                    "--output-dir",
                    str(output_dir),
                ]
                parent_paths = []
                for expert in registry["expert_order"]:
                    run_id = lookup[(shape_id, int(seed), expert)]
                    parent = (
                        root
                        / "runs"
                        / "stage_c"
                        / "offline_experts"
                        / run_id
                        / f"seed_{seed}"
                    )
                    registration = parent / "checkpoint_registration.json"
                    checkpoint = parent / "best_model_val.pt"
                    argv.extend(
                        [
                            "--expert-registration",
                            f"{expert}={registration}",
                            "--expert-checkpoint",
                            f"{expert}={checkpoint}",
                        ]
                    )
                    parent_paths.extend(
                        [str(registration), str(checkpoint)]
                    )
                identity = f"offline_cache:{shape_id}:seed_{seed}:{split}"
                output.append(
                    _record(
                        node_id="offline_fusion_cache",
                        static_id=identity,
                        stage="C",
                        component="OFFLINE_FROZEN_TOKEN_CACHE",
                        seed=int(seed),
                        run_id=None,
                        configuration={
                            "shape_id": shape_id,
                            "pipeline_seed": int(seed),
                            "split": split,
                            "expert_order": list(registry["expert_order"]),
                            "expert_parent_paths": parent_paths,
                        },
                        registry_sha256=registry["content_hash"],
                        argv=argv,
                        expected_artifacts=[
                            str(
                                output_dir
                                / f"{split}_frozen_tokens.json"
                            ),
                            str(
                                output_dir
                                / f"{split}_frozen_tokens.npz"
                            ),
                            *(
                                [
                                    str(
                                        output_dir
                                        / "val_design_relation_sensitivity.npz"
                                    )
                                ]
                                if split == "val_design"
                                else []
                            ),
                        ],
                        deferred_inputs=[
                            _deferred(
                                _path(
                                    root,
                                    "inputs",
                                    "offline",
                                    split,
                                    "offline_input_manifest.json",
                                ),
                                contract="retb_offline_input_cache_v1",
                                producer="offline_input_cache",
                                role=f"{split}_features_labels_identities",
                            ),
                            *[
                                _deferred(
                                    path,
                                    contract=(
                                        "retb_offline_expert_registration_v1"
                                        if path.endswith(".json")
                                        else "retb_expert_checkpoint_v1"
                                    ),
                                    producer="offline_expert_confirmation",
                                    role="frozen_expert_parent",
                                )
                                for path in parent_paths
                            ],
                        ],
                    )
                )
    if len(output) != 63:
        raise RuntimeError("offline frozen-token cache matrix differs")
    return output


def _stage_c_records(
    *,
    root: Path,
    python: str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _physical_rows(
        registry, _STAGE_C_FUSION_SECTIONS, resolver=resolve_stage_c_run
    )
    output = []
    for row in rows:
        run_id = str(row["run_id"])
        seed = int(row["seed"])
        config = row["configuration"]
        shape = str(config["shape_id"])
        variant = str(config["fusion_variant"])
        run_root = root / "runs" / "stage_c" / run_id
        if variant in {"F_BEST_SINGLE", "F_UNIFORM_LOGIT_MEAN"}:
            argv = [
                python,
                "scripts/evaluate_retb_offline_fusion_control.py",
                "--campaign-root",
                str(root),
                "--run-id",
                run_id,
                "--cache",
                _fusion_cache(
                    root, domain="offline", shape=shape, seed=seed, split="val_stop"
                ),
                "--val-stop-cache",
                _fusion_cache(
                    root, domain="offline", shape=shape, seed=seed, split="val_stop"
                ),
                "--val-design-cache",
                _fusion_cache(
                    root,
                    domain="offline",
                    shape=shape,
                    seed=seed,
                    split="val_design",
                ),
                "--output-dir",
                str(run_root),
            ]
            expected = [
                str(run_root / "val_stop_parameter_free_evaluation.json")
            ]
        else:
            argv = [
                python,
                "scripts/train_retb_offline_fusion.py",
                "--campaign-root",
                str(root),
                "--run-id",
                run_id,
                "--model-train-cache",
                _fusion_cache(
                    root,
                    domain="offline",
                    shape=shape,
                    seed=seed,
                    split="model_train",
                ),
                "--val-stop-cache",
                _fusion_cache(
                    root, domain="offline", shape=shape, seed=seed, split="val_stop"
                ),
                "--val-design-cache",
                _fusion_cache(
                    root,
                    domain="offline",
                    shape=shape,
                    seed=seed,
                    split="val_design",
                ),
                "--output-dir",
                str(run_root),
            ]
            expected = [
                str(run_root / "fusion_registration.json"),
                str(run_root / "best_model_val.pt"),
                str(run_root / "val_design_inference.json"),
                str(run_root / "val_design_predictions.npz"),
            ]
        output.append(
            _record(
                node_id="offline_fusion_training",
                static_id=run_id,
                stage="C",
                component=str(row["component"]),
                seed=seed,
                run_id=run_id,
                configuration=config,
                registry_sha256=registry["content_hash"],
                argv=argv,
                expected_artifacts=expected,
                deferred_inputs=[
                    _deferred(
                        _fusion_cache(
                            root,
                            domain="offline",
                            shape=shape,
                            seed=seed,
                            split="model_train",
                        ),
                        contract="retb_frozen_offline_token_cache_v1",
                        producer="offline_expert_training",
                        role="seven_frozen_offline_token_banks",
                    )
                ],
                registry_memberships=row.get("registry_memberships", ()),
            )
        )
    return output


def _hlt_cache(root: Path, role: str, replica: int) -> str:
    policy = "R_MULTI" if role in {"model_train", "scale_train"} else "R_FIXED"
    return _path(
        root,
        "inputs",
        "hlt_v3",
        role,
        f"replica_{replica}",
        policy,
        "D_NOMINAL",
        "hlt_v3_metadata.json",
    )


def _shape_arguments(root: Path, alias: str) -> list[str]:
    arguments: list[str] = []
    if alias in {"SHAPE_COMPACT", "SHAPE_HIGH"}:
        arguments.extend(
            [
                "--uniform-shapes",
                _path(root, "selection", "retb_offline_shapes.json"),
            ]
        )
    if alias in {"HET_SELECTED", "HET_BEAM"}:
        arguments.extend(
            [
                "--heterogeneous-shapes",
                _path(root, "selection", "retb_heterogeneous_shapes.json"),
            ]
        )
    return arguments


def _stage_d_expert_records(
    *,
    root: Path,
    python: str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _physical_rows(
        registry, _STAGE_D_EXPERT_SECTIONS, resolver=resolve_stage_d_run
    )
    for baseline in registry["baseline_rows"]:
        if baseline["configuration"]["control_id"] in {"H_BASE", "H_WIDE"}:
            rows.append(resolve_stage_d_run(registry, run_id=baseline["run_id"]))
    output = []
    for row in rows:
        run_id = str(row["run_id"])
        seed = int(row["seed"])
        config = row["configuration"]
        kind = str(config["kind"])
        if kind == "NATIVE_HLT_EXPERT":
            expert = str(config["expert_id"])
            alias = str(config["shape_id"])
            run_root = (
                root
                / "runs"
                / "stage_d"
                / "hlt_experts"
                / run_id
                / f"seed_{seed}"
            )
            argv = [
                python,
                "scripts/train_retb_native_hlt_expert.py",
                "--campaign-root",
                str(root),
                "--run-id",
                run_id,
            ]
            for replica in range(4):
                argv.extend(
                    [
                        "--train-cache",
                        f"{replica}="
                        f"{_hlt_cache(root, 'model_train', replica)}",
                    ]
                )
            argv.extend(
                [
                    "--val-stop-cache",
                    f"0={_hlt_cache(root, 'val_stop', 0)}",
                    "--val-design-cache",
                    f"0={_hlt_cache(root, 'val_design', 0)}",
                    "--train-labels",
                    _path(
                        root,
                        "inputs",
                        "offline",
                        "model_train",
                        "offline_inputs.npz",
                    ),
                    "--val-stop-labels",
                    _path(
                        root,
                        "inputs",
                        "offline",
                        "val_stop",
                        "offline_inputs.npz",
                    ),
                    "--val-design-labels",
                    _path(
                        root,
                        "inputs",
                        "offline",
                        "val_design",
                        "offline_inputs.npz",
                    ),
                    "--offline-registration",
                    _path(
                        root,
                        "selection",
                        "offline_experts",
                        alias,
                        expert,
                        f"seed_{seed}",
                        "checkpoint_registration.json",
                    ),
                    "--offline-checkpoint",
                    _path(
                        root,
                        "selection",
                        "offline_experts",
                        alias,
                        expert,
                        f"seed_{seed}",
                        "best_model_val.pt",
                    ),
                    "--relation-normalization",
                    _path(
                        root,
                        "inputs",
                        "normalization",
                        "hlt_shared_500k",
                        "relation.json",
                    ),
                    "--region-normalization",
                    _path(
                        root,
                        "inputs",
                        "normalization",
                        "hlt_shared_500k",
                        "region.json",
                    ),
                    "--region-tree-root",
                    _path(root, "inputs", "region_tree", "hlt"),
                    "--output-dir",
                    str(run_root),
                    *_shape_arguments(root, alias),
                ]
            )
            if config["offline_targets_consumed"]:
                argv.extend(
                    [
                        "--offline-train-targets",
                        _path(
                            root,
                            "inputs",
                            "stage_d_offline_targets",
                            alias,
                            expert,
                            f"seed_{seed}.npz",
                        ),
                    ]
                )
            expected = [
                str(run_root / "checkpoint_registration.json"),
                str(run_root / "best_model_val.pt"),
                str(run_root / "native_output_manifest.json"),
            ]
        else:
            control = str(config["control_id"])
            run_root = (
                root
                / "runs"
                / "stage_d"
                / "matched_controls"
                / run_id
                / f"seed_{seed}"
            )
            argv = [
                python,
                "scripts/train_retb_native_hlt_control.py",
                "--campaign-root",
                str(root),
                "--run-id",
                run_id,
            ]
            for replica in range(4):
                argv.extend(
                    [
                        "--train-cache",
                        f"{replica}="
                        f"{_hlt_cache(root, 'model_train', replica)}",
                    ]
                )
            argv.extend(
                [
                    "--val-stop-cache",
                    f"0={_hlt_cache(root, 'val_stop', 0)}",
                    "--train-labels",
                    _path(
                        root,
                        "inputs",
                        "offline",
                        "model_train",
                        "offline_inputs.npz",
                    ),
                    "--val-stop-labels",
                    _path(
                        root,
                        "inputs",
                        "offline",
                        "val_stop",
                        "offline_inputs.npz",
                    ),
                    "--output-dir",
                    str(run_root),
                ]
            )
            expected = [
                str(run_root / "checkpoint_registration.json"),
                str(run_root / "best_model_val.pt"),
                *(
                    [str(run_root / "locked_wide_capacity.json")]
                    if control == "H_WIDE"
                    else []
                ),
            ]
        output.append(
            _record(
                node_id="native_hlt_expert_training",
                static_id=run_id,
                stage="D",
                component=str(row["component"]),
                seed=seed,
                run_id=run_id,
                configuration=config,
                registry_sha256=registry["content_hash"],
                argv=argv,
                expected_artifacts=expected,
                deferred_inputs=[
                    _deferred(
                        _hlt_cache(root, "model_train", 0),
                        contract="retb_hlt_v3_cache_v1",
                        producer="hlt_v3_cache",
                        role="nominal_HLT_training_replica_zero",
                    )
                ],
                registry_memberships=row.get("registry_memberships", ()),
            )
        )
    return output


def _stage_d_fusion_records(
    *,
    root: Path,
    python: str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _physical_rows(
        registry, ("native_fusion_rows",), resolver=resolve_stage_d_run
    )
    output = []
    for row in rows:
        run_id = str(row["run_id"])
        seed = int(row["seed"])
        config = row["configuration"]
        shape = str(config["shape_id"])
        run_root = (
            root
            / "runs"
            / "stage_d"
            / "native_fusions"
            / run_id
            / f"seed_{seed}"
        )
        output.append(
            _record(
                node_id="native_hlt_fusion_training",
                static_id=run_id,
                stage="D",
                component=str(row["component"]),
                seed=seed,
                run_id=run_id,
                configuration=config,
                registry_sha256=registry["content_hash"],
                argv=[
                    python,
                    "scripts/train_retb_native_hlt_fusion.py",
                    "--campaign-root",
                    str(root),
                    "--run-id",
                    run_id,
                    "--model-train-cache",
                    _fusion_cache(
                        root,
                        domain="native_hlt",
                        shape=shape,
                        seed=seed,
                        split="model_train",
                    ),
                    "--val-stop-cache",
                    _fusion_cache(
                        root,
                        domain="native_hlt",
                        shape=shape,
                        seed=seed,
                        split="val_stop",
                    ),
                    "--output-dir",
                    str(run_root),
                ],
                expected_artifacts=(
                    [str(run_root / "fusion_registration.json")]
                    if int(config["fixed_epochs"]) == 0
                    else [
                        str(run_root / "fusion_registration.json"),
                        str(run_root / "best_model_val.pt"),
                    ]
                ),
                deferred_inputs=[
                    _deferred(
                        _fusion_cache(
                            root,
                            domain="native_hlt",
                            shape=shape,
                            seed=seed,
                            split="model_train",
                        ),
                        contract="retb_native_hlt_fusion_cache_v2",
                        producer="native_hlt_expert_training",
                        role="seven_native_HLT_expert_banks",
                    )
                ],
            )
        )
    return output


def _stage_e_pilot_records(
    *,
    root: Path,
    python: str,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for shape in registry["shapes"]:
        for expert in registry["expert_order"]:
            for seed in registry["pipeline_seeds"]:
                identity = f"pilot_t0:{shape}:{expert}:seed_{seed}"
                output_path = (
                    root
                    / "runs"
                    / "stage_e"
                    / "pilots"
                    / canonical_sha256(identity)[:20]
                )
                parent_root = (
                    root
                    / "selection"
                    / "stage_e_parents"
                    / shape
                    / expert
                    / f"seed_{seed}"
                )
                configuration = {
                    "pilot_id": "PILOT_T0",
                    "target_mode": "T0_PURE",
                    "expert_id": expert,
                    "shape_id": shape,
                    "template": registry["pilot_template"],
                    "parent_slots": {
                        "T0_registration": str(
                            parent_root / "t0_registration.json"
                        ),
                        "HLT_encoder_registration": str(
                            parent_root / "hlt_encoder_registration.json"
                        ),
                        "unbiased_particle_encoder_registration": str(
                            parent_root
                            / "unbiased_particle_encoder_registration.json"
                        ),
                        "T0_fusion_registration": str(
                            parent_root / "t0_fusion_registration.json"
                        ),
                        "target_normalizer": str(
                            parent_root / "target_normalizer.json"
                        ),
                        "model_train_pilot_dataset": str(
                            parent_root / "model_train_pilot_dataset.npz"
                        ),
                        "val_stop_pilot_dataset": str(
                            parent_root / "val_stop_pilot_dataset.npz"
                        ),
                        "val_design_pilot_dataset": str(
                            parent_root / "val_design_pilot_dataset.npz"
                        ),
                    },
                }
                argv = [
                    python,
                    "scripts/train_retb_bridge_pilot.py",
                    "--campaign-root",
                    str(root),
                    "--pipeline-seed",
                    str(seed),
                    "--expert-id",
                    expert,
                    "--shape-id",
                    shape,
                    "--t0-registration",
                    str(parent_root / "t0_registration.json"),
                    "--t0-checkpoint",
                    str(parent_root / "t0_best_model_val.pt"),
                    "--hlt-encoder-registration",
                    str(parent_root / "hlt_encoder_registration.json"),
                    "--hlt-encoder-checkpoint",
                    str(parent_root / "hlt_encoder_best_model_val.pt"),
                    "--unbiased-particle-encoder-registration",
                    str(parent_root / "unbiased_particle_encoder_registration.json"),
                    "--unbiased-particle-encoder-checkpoint",
                    str(
                        parent_root
                        / "unbiased_particle_encoder_best_model_val.pt"
                    ),
                    "--t0-fusion-registration",
                    str(parent_root / "t0_fusion_registration.json"),
                    "--t0-fusion-checkpoint",
                    str(parent_root / "t0_fusion_best_model_val.pt"),
                    "--target-normalizer",
                    str(parent_root / "target_normalizer.json"),
                    "--train-dataset",
                    str(parent_root / "model_train_pilot_dataset.npz"),
                    "--val-stop-dataset",
                    str(parent_root / "val_stop_pilot_dataset.npz"),
                    "--val-design-dataset",
                    str(parent_root / "val_design_pilot_dataset.npz"),
                    "--output-dir",
                    str(output_path),
                ]
                output.append(
                    _record(
                        node_id="bridge_pilot_training",
                        static_id=identity,
                        stage="E",
                        component="BRIDGE_PILOT",
                        seed=int(seed),
                        run_id=None,
                        configuration=configuration,
                        registry_sha256=registry["content_hash"],
                        argv=argv,
                        expected_artifacts=[
                            str(output_path / "checkpoint_registration.json"),
                            str(output_path / "best_model_val.pt"),
                            str(output_path / "model_train_coordinate_arrays.npz"),
                            str(output_path / "val_stop_coordinate_arrays.npz"),
                            str(output_path / "val_design_coordinate_arrays.npz"),
                        ],
                        deferred_inputs=[
                            _deferred(
                                str(parent_root / "t0_registration.json"),
                                contract="retb_offline_expert_registration_v1",
                                producer="offline_fusion_training",
                                role="seed_matched_T0_PURE_parent",
                            ),
                            _deferred(
                                str(parent_root / "hlt_encoder_registration.json"),
                                contract="retb_native_hlt_expert_registration_v1",
                                producer="native_hlt_expert_training",
                                role="seed_matched_HLT_encoder_parent",
                            ),
                            _deferred(
                                str(parent_root / "target_normalizer.json"),
                                contract="retb_bridge_token_normalizer_v1",
                                producer="offline_fusion_training",
                                role="seed_matched_T0_target_normalizer",
                            ),
                            _deferred(
                                str(parent_root / "model_train_pilot_dataset.npz"),
                                contract="retb_bridge_pilot_dataset_v1",
                                producer="native_hlt_fusion_training",
                                role="model_train_pilot_evidence",
                            ),
                        ],
                        run_id_resolution=(
                            "materialize_after_parent_checkpoint_hashes_are_immutable"
                        ),
                    )
                )
    return output


def _task_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    node_id: str,
    campaign_sha: str,
    graph_sha: str,
    plan_sha: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "task_id": f"{node_id}:{index}",
                "argv": record["argv"],
                "environment": record["environment"],
                "expected_outputs": record["expected_artifacts"],
                "input_artifact_hashes": {
                    "campaign_spec": campaign_sha,
                    "production_graph": graph_sha,
                    "static_experiment_plan": plan_sha,
                    "static_registry": record["registry_sha256"],
                    "configuration": record["configuration_sha256"],
                    "static_row": record["row_sha256"],
                },
            }
        )
    return rows


def build_static_experiment_bundle(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Build the full production matrix or the declared miniature prefix."""

    validate_content_hash(campaign)
    graph_sha = validate_production_graph(production_graph)
    validate_production_campaign_binding(production_graph, campaign)
    root = Path(campaign_root)
    if root.resolve() != Path(production_graph["campaign_root"]).resolve():
        raise ValueError("static experiment campaign root differs")
    python = str(python_executable or sys.executable)
    registries = _source_bound_registries(campaign)
    validate_stage_b_run_registry(registries["stage_b"])
    validate_stage_c_run_registry(registries["stage_c"])
    validate_stage_d_run_registry(registries["stage_d"])
    validate_stage_e_template_registry(registries["stage_e"])

    groups = {
        "offline_expert_training": _stage_b_records(
            root=root, python=python, registry=registries["stage_b"]
        ),
        "offline_expert_confirmation": _stage_c_expert_records(
            root=root, python=python, registry=registries["stage_c"]
        ),
        "offline_fusion_cache": _stage_c_cache_records(
            root=root, python=python, registry=registries["stage_c"]
        ),
        "offline_fusion_training": _stage_c_records(
            root=root, python=python, registry=registries["stage_c"]
        ),
        "native_hlt_expert_training": _stage_d_expert_records(
            root=root, python=python, registry=registries["stage_d"]
        ),
        "native_hlt_fusion_training": _stage_d_fusion_records(
            root=root, python=python, registry=registries["stage_d"]
        ),
        "bridge_pilot_training": _stage_e_pilot_records(
            root=root, python=python, registry=registries["stage_e"]
        ),
    }
    if tuple(groups) != STATIC_MANIFEST_NODES:
        raise RuntimeError("static experiment group order differs")
    full_counts = {name: len(rows) for name, rows in groups.items()}
    expected_counts = {
        "offline_expert_training": 147,
        "offline_expert_confirmation": 147,
        "offline_fusion_cache": 63,
        "offline_fusion_training": 49,
            "native_hlt_expert_training": 541,
        "native_hlt_fusion_training": 30,
        "bridge_pilot_training": 105,
    }
    if full_counts != expected_counts:
        raise RuntimeError(
            f"static experiment matrix counts differ: {full_counts}"
        )
    miniature = campaign["campaign_profile"] == "miniature_test"
    nodes = {
        str(node["node_id"]): node for node in production_graph["nodes"]
    }
    execution_groups = {}
    for node_id, rows in groups.items():
        execution_groups[node_id] = rows
        if not execution_groups[node_id]:
            raise RuntimeError("static experiment execution group is empty")

    plan = with_content_hash(
        {
            "contract": STATIC_EXPERIMENT_PLAN_CONTRACT,
            "schema_version": 5,
            "campaign_spec_sha256": campaign["content_hash"],
            "production_graph_sha256": graph_sha,
            "campaign_profile": campaign["campaign_profile"],
            "source": campaign["source"],
            "registry_hashes": {
                name: artifact["content_hash"]
                for name, artifact in sorted(registries.items())
            },
            "full_matrix_counts": full_counts,
            "execution_counts": {
                name: len(rows) for name, rows in execution_groups.items()
            },
            "groups": groups,
            "physical_run_deduplication": "first_registry_membership_order",
            "miniature_policy": (
                "complete_scientific_matrix_on_miniature_populations"
                if miniature
                else "complete_static_matrix"
            ),
            "selector_dependent_rows_included": False,
            "scientific_underperformance_skips_rows": False,
        }
    )
    manifests = {}
    for node_id, records in execution_groups.items():
        declaration = nodes[node_id]["array"]
        manifest = build_task_manifest(
            campaign_spec_sha256=campaign["content_hash"],
            production_graph_sha256=graph_sha,
            node_id=node_id,
            rows=_task_rows(
                records,
                node_id=node_id,
                campaign_sha=campaign["content_hash"],
                graph_sha=graph_sha,
                plan_sha=plan["content_hash"],
            ),
            maximum_concurrent_tasks=int(
                declaration["maximum_concurrent_tasks"]
            ),
        )
        validate_task_manifest_for_graph(
            manifest,
            production_graph=production_graph,
            campaign_root=root,
            repo_root=Path(__file__).resolve().parents[2],
        )
        manifests[node_id] = manifest
    bundle = with_content_hash(
        {
            "contract": STATIC_EXPERIMENT_BUNDLE_CONTRACT,
            "schema_version": 5,
            "campaign_spec_sha256": campaign["content_hash"],
            "production_graph_sha256": graph_sha,
            "static_experiment_plan_sha256": plan["content_hash"],
            "registry_hashes": plan["registry_hashes"],
            "task_manifest_hashes": {
                name: artifact["content_hash"]
                for name, artifact in manifests.items()
            },
            "static_manifest_nodes": list(STATIC_MANIFEST_NODES),
            "all_commands_repository_entrypoints": True,
            "all_rows_bind_configuration_and_lineage": True,
            "performance_based_row_skipping": False,
            "source": campaign["source"],
        }
    )
    return {
        "registries": registries,
        "static_experiment_plan": plan,
        "task_manifests": manifests,
        "static_experiment_bundle": bundle,
    }


def validate_static_experiment_bundle(
    payload: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    python_executable: str | None = None,
) -> str:
    expected = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=production_graph,
        campaign_root=campaign_root,
        python_executable=python_executable,
    )
    if dict(payload) != expected:
        raise ValueError("static experiment bundle semantics differ")
    return validate_content_hash(
        payload["static_experiment_bundle"],
        expected_contract=STATIC_EXPERIMENT_BUNDLE_CONTRACT,
    )


def publish_static_experiment_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Any],
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    python_executable: str | None = None,
) -> dict[str, Any]:
    validate_static_experiment_bundle(
        bundle,
        campaign=campaign,
        production_graph=production_graph,
        campaign_root=campaign_root,
        python_executable=python_executable,
    )
    root = Path(campaign_root)
    registry_paths = {
        "stage_b": root / "registry" / "retb_stage_b_runs.json",
        "stage_c": root / "registry" / "retb_stage_c_runs.json",
        "stage_d": root / "registry" / "retb_stage_d_runs.json",
        "stage_e": root / "registry" / "retb_stage_e_templates.json",
    }
    publications = {
        f"registry/{name}": write_immutable_json(
            registry_paths[name], artifact
        )
        for name, artifact in bundle["registries"].items()
    }
    publications["static_experiment_plan"] = write_immutable_json(
        root / "registry" / "retb_static_experiment_plan.json",
        bundle["static_experiment_plan"],
    )
    publications["static_experiment_bundle"] = write_immutable_json(
        root / "registry" / "retb_static_experiment_bundle.json",
        bundle["static_experiment_bundle"],
    )
    for node_id, manifest in bundle["task_manifests"].items():
        publications[f"task_manifest/{node_id}"] = write_immutable_json(
            task_manifest_path_for_graph(
                production_graph,
                node_id=node_id,
                campaign_root=root,
            ),
            manifest,
        )
    return publications


__all__ = [
    "STATIC_EXPERIMENT_BUNDLE_CONTRACT",
    "STATIC_EXPERIMENT_PLAN_CONTRACT",
    "STATIC_MANIFEST_NODES",
    "build_static_experiment_bundle",
    "publish_static_experiment_bundle",
    "validate_static_experiment_bundle",
]
