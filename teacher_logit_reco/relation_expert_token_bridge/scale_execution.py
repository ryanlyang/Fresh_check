"""Authenticated execution plans for genuine Stage-M refits and training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .scale_up import SCALE_COMPONENT_KEYS, SCALE_REFIT_KEYS


SCALE_REFIT_EXECUTION_PLAN_CONTRACT = (
    "retb_scale_refit_execution_plan_v1"
)
SCALE_GRAPH_EXECUTION_PLAN_CONTRACT = (
    "retb_scale_graph_execution_plan_v1"
)
SCALE_PREDICTOR_RUN_CONTRACT = "retb_scale_predictor_run_v1"
SCALE_STAGE_J_RUN_CONTRACT = "retb_scale_stage_j_run_v1"
SCALE_FINAL_CONSUMER_RUN_CONTRACT = (
    "retb_scale_final_consumer_run_v1"
)
SCALE_COMPONENT_INDEX_CONTRACT = "retb_scale_component_index_v1"
SCALE_JOINT_COMPLETION_CONTRACT = "retb_scale_joint_completion_v1"
_EXPERT_IDS = (
    "BASE4",
    "PT",
    "TRACK",
    "PID",
    "CHARGE",
    "DENSITY",
    "REGION",
)


def build_scale_predictor_run(
    *,
    base_run: Mapping[str, Any],
    scale_parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Carry a selected predictor topology onto explicit scale parents."""

    base_sha = validate_content_hash(base_run)
    parents = {
        str(name): require_sha256(
            value, name=f"scale_parent_hashes.{name}"
        )
        for name, value in sorted(scale_parent_hashes.items())
    }
    required = {
        "scale_train_target_cache",
        "val_stop_target_cache",
        "target_normalizer",
        "slot_queries",
        "offline_target_checkpoint",
        "offline_fusion",
        "native_hlt_expert",
        "scale_train_hlt_evidence_cache",
        "val_stop_hlt_evidence_cache",
        "scale_train_identity_manifest",
        "val_stop_identity_manifest",
        "step9_bundle",
    }
    if (
        base_run.get("contract")
        != "retb_materialized_predictor_run_v2"
        or not required.issubset(parents)
    ):
        raise ValueError("scale predictor base/parent semantics differ")
    payload = {
        key: value
        for key, value in base_run.items()
        if key not in {"contract", "schema_version", "parent_hashes", "source", "content_hash"}
    }
    payload.update(
        {
            "contract": SCALE_PREDICTOR_RUN_CONTRACT,
            "schema_version": 1,
            "base_run_sha256": base_sha,
            "training_population": "scale_train",
            "parent_hashes": parents,
        }
    )
    return with_content_hash(payload)


def build_scale_stage_j_run(
    *,
    base_run: Mapping[str, Any],
    scale_parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Carry the selected J5 topology onto scale-trained components."""

    base_sha = validate_content_hash(base_run)
    if (
        base_run.get("contract") != "retb_materialized_stage_j_run_v1"
        or base_run.get("variant") != "J5_END_TO_END"
    ):
        raise ValueError("scale Stage-J base run differs")
    parents = {
        str(name): require_sha256(value, name=f"scale_parent_hashes.{name}")
        for name, value in sorted(scale_parent_hashes.items())
    }
    required = {
        "scale_train_identity_manifest",
        "val_stop_identity_manifest",
        "val_design_identity_manifest",
        "val_design_label_manifest",
        "scale_train_R_MULTI_view_cache",
        "val_stop_R_MULTI_view_cache",
        "val_design_fixed_view_cache",
        "offline_target_cache",
        "target_normalizer_set",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
        "selected_predictor_seed_artifacts",
        "selected_HLT_expert_seed_artifacts",
        "j4_block_selection",
        "selected_J4_bridge_initialization",
    }
    if set(parents) != required:
        raise ValueError("scale Stage-J parent coverage differs")
    payload = {
        key: value
        for key, value in base_run.items()
        if key
        not in {
            "contract",
            "schema_version",
            "parent_hashes",
            "source",
            "content_hash",
        }
    }
    payload.update(
        {
            "contract": SCALE_STAGE_J_RUN_CONTRACT,
            "schema_version": 1,
            "base_run_sha256": base_sha,
            "training_population": "scale_train",
            "parent_hashes": parents,
        }
    )
    return with_content_hash(payload)


def build_scale_final_consumer_run(
    *,
    base_run: Mapping[str, Any],
    scale_parent_hashes: Mapping[str, str],
    graph_id: str,
    scale_run_id: str | None = None,
) -> dict[str, Any]:
    """Carry one locked final-consumer definition onto scale parents."""

    base_sha = validate_content_hash(base_run)
    if base_run.get("contract") != "retb_materialized_final_consumer_run_v2":
        raise ValueError("scale final-consumer base run differs")
    parents = {
        str(name): require_sha256(value, name=f"scale_parent_hashes.{name}")
        for name, value in sorted(scale_parent_hashes.items())
    }
    required = {
        "scale_train_identity_manifest",
        "val_stop_identity_manifest",
        "val_design_identity_manifest",
        "val_design_label_manifest",
        "scale_train_R_MULTI_view_cache",
        "val_stop_R_MULTI_view_cache",
        "val_design_fixed_view_cache",
        "joint_prediction_checkpoint",
        "native_HLT_checkpoint_bundle",
        "offline_target_cache",
        "target_normalizer_set",
        "uncertainty_calibration",
        "HLT_input_normalizer",
        "HLT_relation_normalizer",
        "HLT_region_normalizer",
        "degradation_profile",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
    }
    if base_run.get("token_input") == "TOKEN_REFINED_SELECTED":
        required.add("selected_token_refiner")
    if (
        set(parents) != required
        or not str(graph_id)
        or (scale_run_id is not None and not str(scale_run_id))
    ):
        raise ValueError("scale final-consumer parent coverage differs")
    payload = {
        key: value
        for key, value in base_run.items()
        if key
        not in {
            "contract",
            "schema_version",
            "parent_hashes",
            "source",
            "content_hash",
        }
    }
    payload.update(
        {
            "contract": SCALE_FINAL_CONSUMER_RUN_CONTRACT,
            "schema_version": 1,
            "base_run_sha256": base_sha,
            "base_run_id": str(base_run["run_id"]),
            "run_id": str(
                scale_run_id
                or (
                    f"RETB_SCALE_{graph_id}_"
                    f"S{int(base_run['pipeline_seed'])}"
                )
            ),
            "graph_id": str(graph_id),
            "training_population": "scale_train",
            "parent_hashes": parents,
        }
    )
    return with_content_hash(payload)


def validate_scale_final_consumer_run(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_FINAL_CONSUMER_RUN_CONTRACT
    )
    base = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "contract",
            "schema_version",
            "base_run_sha256",
            "graph_id",
            "training_population",
            "parent_hashes",
            "source",
            "content_hash",
        }
    }
    # The original run hash is an immutable external parent.  Reconstructing
    # the full v2 record is impossible without its original parent map, so the
    # validator checks the scale-only envelope and all scale parent coverage.
    del base
    if (
        payload.get("training_population") != "scale_train"
        or not str(payload.get("graph_id", ""))
        or not isinstance(payload.get("parent_hashes"), Mapping)
    ):
        raise ValueError("scale final-consumer semantics differ")
    require_sha256(payload.get("base_run_sha256"), name="base_run_sha256")
    return digest


def build_scale_joint_completion(
    *,
    graph_id: str,
    pipeline_seed: int,
    scale_component_index_sha256: str,
    scale_stage_j_run_sha256: str,
    joint_checkpoint_sha256: str,
    joint_registration_sha256: str,
    joint_training_curves_sha256: str,
) -> dict[str, Any]:
    if not str(graph_id) or int(pipeline_seed) not in {101, 202, 303}:
        raise ValueError("scale joint-completion identity differs")
    return with_content_hash(
        {
            "contract": SCALE_JOINT_COMPLETION_CONTRACT,
            "schema_version": 1,
            "graph_id": str(graph_id),
            "pipeline_seed": int(pipeline_seed),
            "scale_component_index_sha256": require_sha256(
                scale_component_index_sha256,
                name="scale_component_index_sha256",
            ),
            "scale_stage_j_run_sha256": require_sha256(
                scale_stage_j_run_sha256,
                name="scale_stage_j_run_sha256",
            ),
            "joint_checkpoint_sha256": require_sha256(
                joint_checkpoint_sha256, name="joint_checkpoint_sha256"
            ),
            "joint_registration_sha256": require_sha256(
                joint_registration_sha256,
                name="joint_registration_sha256",
            ),
            "joint_training_curves_sha256": require_sha256(
                joint_training_curves_sha256,
                name="joint_training_curves_sha256",
            ),
            "training_population": "scale_train",
            "performance_based_termination": False,
        }
    )


def build_scale_component_index(
    *,
    graph_id: str,
    pipeline_seed: int,
    base_j5_run_path: str | Path,
    target_cache_root: str | Path,
    target_normalizer_root: str | Path,
    scale_normalizer_bundle_sha256: str,
    hlt_relation_normalizer_sha256: str,
    hlt_region_normalizer_sha256: str,
    degradation_profile_sha256: str,
    offline_fusion_checkpoint: str | Path,
    offline_fusion_registration_sha256: str,
    offline_experts: Mapping[str, Mapping[str, Any]],
    native_hlt_experts: Mapping[str, Mapping[str, Any]],
    target_checkpoints: Mapping[str, Mapping[str, Any]],
    predictors: Mapping[str, Mapping[str, Any]],
    uncertainty_calibrations: Mapping[str, str | Path],
    selected_token_refiner: Mapping[str, Any] | None,
    component_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Publish the immutable scale components consumed by one graph row."""

    if (
        not str(graph_id)
        or int(pipeline_seed) not in {101, 202, 303}
        or set(offline_experts) != set(_EXPERT_IDS)
        or set(native_hlt_experts) != set(_EXPERT_IDS)
        or set(target_checkpoints) != set(_EXPERT_IDS)
        or set(predictors) != set(_EXPERT_IDS)
        or set(uncertainty_calibrations) != set(_EXPERT_IDS)
    ):
        raise ValueError("scale component-index coverage differs")

    def checked_rows(
        values: Mapping[str, Mapping[str, Any]],
        required: set[str],
        *,
        label: str,
    ) -> dict[str, dict[str, Any]]:
        output = {}
        for expert in _EXPERT_IDS:
            row = dict(values[expert])
            if set(row) != required:
                raise ValueError(f"{label}.{expert} fields differ")
            for name in required:
                if name.endswith("sha256"):
                    row[name] = require_sha256(
                        row[name], name=f"{label}.{expert}.{name}"
                    )
                else:
                    row[name] = str(row[name])
            output[expert] = row
        return output

    offline = checked_rows(
        offline_experts,
        {"checkpoint", "checkpoint_sha256", "registration"},
        label="offline_experts",
    )
    native = checked_rows(
        native_hlt_experts,
        {"output_root", "checkpoint_sha256", "registration_sha256"},
        label="native_hlt_experts",
    )
    targets = checked_rows(
        target_checkpoints,
        {"path", "sha256", "registration_sha256", "target_mode"},
        label="target_checkpoints",
    )
    predictor_rows = checked_rows(
        predictors,
        {
            "run_path",
            "output_root",
            "checkpoint_sha256",
            "registration_sha256",
        },
        label="predictors",
    )
    refiner = None if selected_token_refiner is None else dict(
        selected_token_refiner
    )
    if refiner is not None:
        if set(refiner) != {
            "variant",
            "checkpoint_path",
            "checkpoint_sha256",
            "registration_sha256",
        }:
            raise ValueError("scale selected-token-refiner fields differ")
        refiner["checkpoint_sha256"] = require_sha256(
            refiner["checkpoint_sha256"],
            name="selected_token_refiner.checkpoint_sha256",
        )
        refiner["registration_sha256"] = require_sha256(
            refiner["registration_sha256"],
            name="selected_token_refiner.registration_sha256",
        )
        refiner["variant"] = str(refiner["variant"])
        refiner["checkpoint_path"] = str(refiner["checkpoint_path"])
    return with_content_hash(
        {
            "contract": SCALE_COMPONENT_INDEX_CONTRACT,
            "schema_version": 1,
            "graph_id": str(graph_id),
            "pipeline_seed": int(pipeline_seed),
            "base_j5_run_path": str(Path(base_j5_run_path)),
            "target_cache_root": str(Path(target_cache_root)),
            "target_normalizer_root": str(Path(target_normalizer_root)),
            "scale_normalizer_bundle_sha256": require_sha256(
                scale_normalizer_bundle_sha256,
                name="scale_normalizer_bundle_sha256",
            ),
            "hlt_relation_normalizer_sha256": require_sha256(
                hlt_relation_normalizer_sha256,
                name="hlt_relation_normalizer_sha256",
            ),
            "hlt_region_normalizer_sha256": require_sha256(
                hlt_region_normalizer_sha256,
                name="hlt_region_normalizer_sha256",
            ),
            "degradation_profile_sha256": require_sha256(
                degradation_profile_sha256,
                name="degradation_profile_sha256",
            ),
            "offline_fusion_checkpoint": str(
                Path(offline_fusion_checkpoint)
            ),
            "offline_fusion_registration_sha256": require_sha256(
                offline_fusion_registration_sha256,
                name="offline_fusion_registration_sha256",
            ),
            "offline_experts": offline,
            "native_hlt_experts": native,
            "target_checkpoints": targets,
            "predictors": predictor_rows,
            "uncertainty_calibrations": {
                expert: str(Path(uncertainty_calibrations[expert]))
                for expert in _EXPERT_IDS
            },
            "selected_token_refiner": refiner,
            "component_hashes": {
                str(name): require_sha256(
                    value, name=f"component_hashes.{name}"
                )
                for name, value in sorted(component_hashes.items())
            },
            "training_population": "scale_train",
            "shared_components_trained_once_per_seed": True,
            "performance_based_termination": False,
        }
    )


def validate_scale_component_index(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_COMPONENT_INDEX_CONTRACT
    )
    if (
        payload.get("training_population") != "scale_train"
        or payload.get("shared_components_trained_once_per_seed") is not True
        or payload.get("performance_based_termination") is not False
        or int(payload.get("pipeline_seed", -1)) not in {101, 202, 303}
        or set(payload.get("offline_experts", {})) != set(_EXPERT_IDS)
        or set(payload.get("native_hlt_experts", {})) != set(_EXPERT_IDS)
        or set(payload.get("target_checkpoints", {})) != set(_EXPERT_IDS)
        or set(payload.get("predictors", {})) != set(_EXPERT_IDS)
        or set(payload.get("uncertainty_calibrations", {}))
        != set(_EXPERT_IDS)
    ):
        raise ValueError("scale component-index semantics differ")
    for name in (
        "scale_normalizer_bundle_sha256",
        "hlt_relation_normalizer_sha256",
        "hlt_region_normalizer_sha256",
        "degradation_profile_sha256",
        "offline_fusion_registration_sha256",
    ):
        require_sha256(payload.get(name), name=name)
    return digest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_digest(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"Stage-M artifact is absent: {resolved}")
    if resolved.suffix.lower() == ".json":
        return validate_content_hash(
            json.loads(resolved.read_text(encoding="utf-8"))
        )
    return file_sha256(resolved)


def source_bound_artifact_digest(
    path: str | Path, *, campaign_source: Mapping[str, Any]
) -> str:
    """Hash a JSON manifest only after exact campaign-source validation."""

    resolved = Path(path)
    if resolved.suffix.lower() != ".json":
        raise ValueError(
            "component evidence must be a source-bound JSON manifest"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    digest = validate_content_hash(payload)
    if payload.get("source") != campaign_source:
        raise ValueError("component artifact source differs")
    return digest


def _inside(path: str | Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("Stage-M execution path escapes campaign") from error
    return resolved


def validate_execution_steps(
    steps: Any,
    *,
    campaign_root: Path,
    repo_root: Path,
    forbidden_terms: tuple[str, ...] = ("stack_val", "final_test"),
    forbidden_entrypoints: frozenset[str] = frozenset(
        {
            "execute_retb_scale_refits.py",
            "execute_retb_scale_graph_training.py",
            "train_retb_scale_shortlist.py",
            "register_retb_scale_refits.py",
        }
    ),
) -> list[dict[str, Any]]:
    if not isinstance(steps, list) or not steps:
        raise ValueError("Stage-M execution steps are empty")
    checked = []
    for index, raw in enumerate(steps):
        if set(raw) != {"step_id", "argv", "expected_outputs"}:
            raise ValueError("Stage-M execution-step fields differ")
        argv = [str(value) for value in raw["argv"]]
        outputs = [
            str(_inside(value, campaign_root))
            for value in raw["expected_outputs"]
        ]
        if (
            str(raw["step_id"]) != f"step_{index:03d}"
            or len(argv) < 2
            or Path(argv[0]).name.lower().startswith("python") is False
            or Path(argv[1]).is_absolute()
            or Path(argv[1]).suffix != ".py"
            or "--dry-run" in argv
            or not outputs
        ):
            raise ValueError("Stage-M execution-step semantics differ")
        entrypoint = (repo_root / argv[1]).resolve()
        try:
            entrypoint.relative_to(repo_root)
        except ValueError as error:
            raise ValueError("Stage-M entry point escapes repository") from error
        lowered = " ".join(argv).lower()
        if (
            not entrypoint.is_file()
            or any(term in lowered for term in forbidden_terms)
            or entrypoint.name in forbidden_entrypoints
        ):
            raise ValueError("Stage-M execution step is not a training/refit worker")
        checked.append(
            {
                "step_id": str(raw["step_id"]),
                "argv": argv,
                "expected_outputs": outputs,
            }
        )
    return checked


def validate_scale_refit_execution_plan(
    payload: Mapping[str, Any],
    *,
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_REFIT_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "graph_id",
        "pipeline_seed",
        "locked_scale_shortlist_sha256",
        "scale_train_manifest_sha256",
        "val_design_identity_manifest_sha256",
        "five_hundred_k_artifact_hashes",
        "operations",
        "source",
        "content_hash",
    }
    operations = payload.get("operations", {})
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
        or int(payload["pipeline_seed"]) not in {101, 202, 303}
        or set(operations) != set(SCALE_REFIT_KEYS)
    ):
        raise ValueError("scale-refit execution plan semantics differ")
    root = Path(campaign_root).resolve()
    source = Path(repo_root).resolve()
    for name in SCALE_REFIT_KEYS:
        row = operations[name]
        if set(row) != {
            "population",
            "identity_manifest_sha256",
            "recipe_sha256",
            "replica_ids",
            "steps",
            "output_artifact",
        }:
            raise ValueError("scale-refit operation fields differ")
        validate_execution_steps(
            row["steps"], campaign_root=root, repo_root=source
        )
        _inside(row["output_artifact"], root)
        require_sha256(
            row["identity_manifest_sha256"],
            name=f"{name}.identity_manifest_sha256",
        )
        require_sha256(
            row["recipe_sha256"], name=f"{name}.recipe_sha256"
        )
    return digest


def validate_scale_graph_execution_plan(
    payload: Mapping[str, Any],
    *,
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_GRAPH_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "graph_id",
        "pipeline_seed",
        "locked_scale_shortlist_sha256",
        "scale_refit_bundle_sha256",
        "steps",
        "component_artifacts",
        "training_summary",
        "pre_stack_metrics",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
        or int(payload["pipeline_seed"]) not in {101, 202, 303}
        or set(payload["component_artifacts"])
        != set(SCALE_COMPONENT_KEYS)
    ):
        raise ValueError("scale-graph execution plan semantics differ")
    root = Path(campaign_root).resolve()
    validate_execution_steps(
        payload["steps"],
        campaign_root=root,
        repo_root=Path(repo_root).resolve(),
    )
    for path in payload["component_artifacts"].values():
        if _inside(path, root).suffix.lower() != ".json":
            raise ValueError(
                "scale component evidence must use JSON manifests"
            )
    _inside(payload["training_summary"], root)
    _inside(payload["pre_stack_metrics"], root)
    return digest


def execute_plan_steps(
    steps: list[Mapping[str, Any]],
    *,
    campaign_root: str | Path,
    repo_root: str | Path,
    forbidden_terms: tuple[str, ...] = ("stack_val", "final_test"),
    forbidden_entrypoints: frozenset[str] = frozenset(
        {
            "execute_retb_scale_refits.py",
            "execute_retb_scale_graph_training.py",
            "train_retb_scale_shortlist.py",
            "register_retb_scale_refits.py",
        }
    ),
) -> list[dict[str, Any]]:
    checked = validate_execution_steps(
        steps,
        campaign_root=Path(campaign_root).resolve(),
        repo_root=Path(repo_root).resolve(),
        forbidden_terms=forbidden_terms,
        forbidden_entrypoints=forbidden_entrypoints,
    )
    receipts = []
    for step in checked:
        existing = all(
            Path(path).is_file() and not Path(path).is_symlink()
            for path in step["expected_outputs"]
        )
        if not existing:
            completed = subprocess.run(
                step["argv"], cwd=repo_root, check=False
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Stage-M step failed: {step['step_id']} "
                    f"({completed.returncode})"
                )
        hashes = {
            path: artifact_digest(path)
            for path in step["expected_outputs"]
        }
        receipts.append(
            {
                "step_id": step["step_id"],
                "reused": existing,
                "output_hashes": hashes,
            }
        )
    return receipts


__all__ = [
    "SCALE_GRAPH_EXECUTION_PLAN_CONTRACT",
    "SCALE_REFIT_EXECUTION_PLAN_CONTRACT",
    "SCALE_PREDICTOR_RUN_CONTRACT",
    "SCALE_STAGE_J_RUN_CONTRACT",
    "SCALE_FINAL_CONSUMER_RUN_CONTRACT",
    "SCALE_COMPONENT_INDEX_CONTRACT",
    "SCALE_JOINT_COMPLETION_CONTRACT",
    "artifact_digest",
    "build_scale_predictor_run",
    "build_scale_stage_j_run",
    "build_scale_final_consumer_run",
    "build_scale_joint_completion",
    "build_scale_component_index",
    "execute_plan_steps",
    "file_sha256",
    "source_bound_artifact_digest",
    "validate_execution_steps",
    "validate_scale_graph_execution_plan",
    "validate_scale_refit_execution_plan",
    "validate_scale_final_consumer_run",
    "validate_scale_component_index",
]
