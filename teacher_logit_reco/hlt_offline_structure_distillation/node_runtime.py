"""Late-bind registered HOSD array coordinates from immutable upstream plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import string
from typing import Any, Mapping

from .contracts import (
    COMBINATION_BEAM_COMPLETION_CONTRACT,
    COMBINATION_SELECTION_CONTRACT,
    CONFIRMATION_PLAN_CONTRACT,
    GRAPH_REGISTRY_CONTRACT,
    FEEDBACK_SELECTION_CONTRACT,
    MECHANISM_SUMMARY_CONTRACT,
    ROBUSTNESS_PLAN_CONTRACT,
    RUNTIME_MANIFEST_CONTRACT,
    SCALE_EXECUTION_PLAN_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    STAGE_D_PLAN_CONTRACT,
    STAGE_E_PLAN_CONTRACT,
    STAGE_F_PLAN_CONTRACT,
    load_hashed_json,
    validate_content_hash,
    canonical_sha256,
)
from .baselines import BASELINE_IDS
from .auxiliary import resolve_stage_d_phase_two
from .graph_registry import build_locked_graph_registry
from .metrics import build_robustness_plan
from .metrics import DEGRADATION_PROFILES
from .stage_b_runtime import stage_b_wave_rows


FIXED_ARGUMENTS = {
    "campaign_bootstrap": ["--scope", "campaign"],
    "inherited_parent_audit": ["--scope", "parent-audit"],
    "shared_hlt_parent_rebuild": ["--submit"],
    "tree_parent_rebuild": ["--submit"],
    "relation_normalizer_rebuild": ["--submit"],
    "single_family_phase_lock": ["--mode", "phase-lock"],
    "combination_beam": ["--mode", "beam-run"],
    "combination_select": ["--mode", "full-select"],
    "mechanism_controls": ["--mode", "execute"],
    "mechanism_confirm": ["--mode", "finalize"],
    "robustness_plan_compile": ["--mode", "compile"],
    "robustness_evaluation": ["--mode", "execute"],
    "robustness_report": ["--mode", "finalize"],
    "confirmation_compile": ["--mode", "compile"],
    "confirmation_aggregate": ["--mode", "aggregate"],
    "scale_plan_compile": ["--mode", "compile"],
    "scale_teacher_lock": ["--mode", "teacher-lock"],
    "scale_finalize": ["--mode", "finalize"],
    "stack_selector": ["--mode", "select"],
    "finalist_lock": ["--mode", "lock"],
    "final_input_preparation": ["--mode", "prepare"],
    "execution_lock": ["--mode", "lock"],
}

ROW_SCOPED_RUNTIME_OPTIONS = {
    "canonical_target_build": frozenset(
        {"--input-npz", "--tree-cache-dir"}
    ),
    "hlt_analogue_target_build": frozenset(
        {"--input-npz", "--tree-cache-dir"}
    ),
}


def _shared_root(root: Path) -> Path:
    return root / "inputs" / "shared_retb_parent_campaign"


def _hlt_cache(root: Path, role: str, replica: int) -> Path:
    policy = "R_MULTI" if role in {"model_train", "scale_train"} else "R_FIXED"
    return (
        _shared_root(root)
        / "inputs"
        / "hlt_v3"
        / role
        / f"replica_{replica}"
        / policy
        / "D_NOMINAL"
    )


def _hlt_view(root: Path, role: str, replica: int) -> Path:
    return root / "inputs" / "hosd_views" / "hlt" / role / f"replica_{replica}.npz"


def _offline_view(root: Path, role: str) -> Path:
    return root / "inputs" / "hosd_views" / "offline" / f"{role}.npz"


def _labels_npz(root: Path, role: str) -> Path:
    if role in {"design_select", "design_confirm"}:
        return root / "registry" / "runtime_support" / f"{role}_identity_labels.npz"
    return _shared_root(root) / "inputs" / "offline" / role / "offline_inputs.npz"


def _tree(root: Path, view: str, role: str, replica: int | None = None) -> Path:
    policy = "R_MULTI" if role in {"model_train", "scale_train"} else "R_FIXED"
    name = (
        f"{role}_exclusive_ca_v1"
        if replica is None
        else f"{role}_r{replica}_{policy}_exclusive_ca_v1"
    )
    return _shared_root(root) / "inputs" / "region_tree" / view / name


def _support(root: Path, name: str) -> Path:
    return root / "registry" / "runtime_support" / name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _teacher_adapter_config(root: Path, row: Mapping[str, Any]) -> Path:
    """Publish the exact teacher/split adapter only after its row is known."""

    teacher_id = str(row["teacher_id"])
    split = str(row["split"])
    input_path = _offline_view(root, split)
    campaign = load_hashed_json(root / "campaign_spec.json")
    if split == "stack_val" and not input_path.is_file():
        # This label-blind offline view is deliberately materialized only
        # after finalist locking, when this function is reached by the
        # post-lock oracle diagnostic.
        from .input_views import materialize_retb_offline_input_view

        parent_lock = load_hashed_json(
            root / "inputs" / "resolved_inherited_parent_lock.json"
        )
        materialize_retb_offline_input_view(
            offline_cache_dir=(
                _shared_root(root) / "inputs" / "offline" / split
            ),
            split=split,
            output=input_path,
            parent_hashes={
                "campaign_spec": campaign["content_hash"],
                "resolved_parent_lock": parent_lock["content_hash"],
            },
            source=campaign["source"],
        )
    config: dict[str, Any] = {
        "contract": "hosd_teacher_adapter_config_v1",
        "source": campaign["source"],
        "campaign_spec_sha256": campaign["content_hash"],
        "teacher_id": teacher_id,
        "split": split,
        "input_npz": str(input_path.resolve()),
        "input_npz_sha256": _sha256_file(input_path),
        "screening_registry": str(_support(root, "screening_registry.json").resolve()),
        "relation_normalizer": str(
            (root / "inputs" / "normalization" / "offline_500k" / "relation.json").resolve()
        ),
        "device": "auto",
    }
    if teacher_id == "O_FULLREL":
        tree_path = _tree(root, "offline", split)
        tree_manifest = load_hashed_json(tree_path / "manifest.json")
        config.update(
            {
                "region_normalizer": str(
                    (root / "inputs" / "normalization" / "offline_500k" / "region.json").resolve()
                ),
                "tree_cache_dir": str(tree_path.resolve()),
                "tree_expected_parents": dict(tree_manifest["parents"]),
                "tree_resource_path": str(
                    (root / "inputs" / "inherited_angular_tree_resource.json").resolve()
                ),
                "tree_backend_manifest_path": str(
                    (root / "inputs" / "region_tree" / "backend_manifest.json").resolve()
                ),
            }
        )
    path = _support(root, "teacher_adapters") / split / f"{teacher_id}.json"
    encoded = json.dumps(config, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"immutable teacher adapter differs: {path}")
    else:
        path.write_text(encoded, encoding="utf-8")
    return path


def _repeated(option: str, values: Mapping[int | str, Path]) -> list[str]:
    return [
        item
        for key, path in values.items()
        for item in (option, f"{key}={path}")
    ]


def _derived_infrastructure(
    root: Path, node_id: str, row: Mapping[str, Any] | None
) -> list[str]:
    """Resolve campaign-owned paths without pre-binding future DAG outputs."""

    relation = root / "inputs" / "normalization" / "offline_500k" / "relation.json"
    region = root / "inputs" / "normalization" / "offline_500k" / "region.json"
    hlt_relation = root / "inputs" / "normalization" / "hlt_shared_500k" / "relation.json"
    backend = root / "inputs" / "region_tree" / "backend_manifest.json"
    tree_resource = root / "inputs" / "inherited_angular_tree_resource.json"
    support = root / "registry" / "runtime_support"
    if node_id == "storage_measurement":
        return ["--probe-input", str(_offline_view(root, "model_train")), "--relation-normalizer", str(relation)]
    if node_id == "offline_teacher_train":
        values = [
            "--model-contract-o-base", str(support / "model_contract_O_BASE.json"),
            "--model-contract-o-fullrel", str(support / "model_contract_O_FULLREL.json"),
            "--normalizer-hashes", str(support / "teacher_normalizer_hashes.json"),
            "--offline-manifest", str(root / "inputs" / "split_manifest.json.gz"),
            "--validation-partition", str(root / "inputs" / "validation_partition_manifest.json.gz"),
            "--screening-registry", str(support / "screening_registry.json"),
            "--relation-registry", str(support / "relation_family_registry.json"),
            "--relation-normalizer", str(relation),
            "--global-determinism", str(root / "registry" / "global_determinism.json"),
        ]
        if row is not None and row.get("teacher_id") == "O_FULLREL":
            values.extend(
                [
                    "--region-normalizer", str(region),
                    "--tree-root", str(
                        _shared_root(root) / "inputs" / "region_tree" / "offline"
                    ),
                ]
            )
        return values
    if node_id in {"canonical_target_build", "hlt_analogue_target_build"}:
        if row is None:
            raise ValueError(f"{node_id} requires a scientific row")
        split = str(row["split"])
        if node_id == "canonical_target_build":
            input_path = _offline_view(root, split)
            tree_path = _tree(root, "offline", split)
            active_normalizer = relation
        else:
            replica = int(row["replica"])
            input_path = _hlt_view(root, split, replica)
            tree_path = _tree(root, "hlt", split, replica)
            active_normalizer = hlt_relation
        return [
            "--input-npz", str(input_path),
            "--relation-normalizer", str(active_normalizer),
            "--tree-backend-manifest", str(backend),
            "--tree-cache-dir", str(tree_path),
        ]
    if node_id == "teacher_target_inference":
        if row is None:
            raise ValueError("teacher inference requires a scientific row")
        return ["--adapter-config", str(_teacher_adapter_config(root, row))]
    if node_id == "target_normalization":
        return [
            *_repeated("--model-train-hlt", {r: _hlt_view(root, "model_train", r) for r in range(4)}),
            *_repeated("--model-train-tree", {r: _tree(root, "hlt", "model_train", r) for r in range(4)}),
            "--relation-normalizer", str(hlt_relation),
        ]
    if node_id in {"target_controls", "target_audit"}:
        label_roles = (
            (
                "model_train",
                "val_stop",
                "val_design",
                "design_select",
                "design_confirm",
            )
            if node_id == "target_controls"
            else ("model_train", "val_stop", "val_design")
        )
        return [
            item
            for split in label_roles
            for item in ("--label-manifest", f"{split}={support / 'labels' / f'{split}.json'}")
        ]
    if node_id in {"baseline_train", "probe_tap_capture"}:
        # NativeHLTExpertDataset authenticates the original RETB HLT-v3
        # metadata contract.  HOSD's materialized views are independently
        # authenticated target-extraction inputs, not RETB cache artifacts.
        cache = _hlt_cache
        return [
            *_repeated("--train-cache", {r: cache(root, "model_train", r) for r in range(4)}),
            "--val-stop-cache", f"0={cache(root, 'val_stop', 0)}",
            # The beam-budget baseline evaluates the design_select identity
            # subset.  Its native inputs come from the authenticated parent
            # val_design cache; _dataset resolves the exact subrole indices.
            "--design-select-cache", f"0={cache(root, 'val_design', 0)}",
            "--train-labels", str(_labels_npz(root, "model_train")),
            "--val-stop-labels", str(_labels_npz(root, "val_stop")),
            "--design-select-labels", str(_labels_npz(root, "design_select")),
        ]
    if node_id == "probe_input_materialization":
        roles = {"model_train": "model_train", "val_stop": "val_stop", "design_select": "design_select"}
        values = [
            item
            for role, physical in roles.items()
            for item in (
                "--labels", f"{role}={_labels_npz(root, physical)}",
                "--raw-input", f"{role}={_offline_view(root, physical)}",
            )
        ]
        for role, physical in roles.items():
            replicas = range(4) if role == "model_train" else (0,)
            for replica in replicas:
                values.extend(["--hlt-input", f"{role}:{replica}={_hlt_view(root, physical, replica)}"])
                tree_role = "val_design" if role == "design_select" else physical
                values.extend(["--tree-cache", f"{role}:{replica}={_tree(root, 'hlt', tree_role, replica)}"])
        values.extend(["--relation-normalizer", str(hlt_relation)])
        return values
    if node_id in {
        "auxiliary_train", "relation_het_auxiliary_train", "hlt_self_auxiliary_train",
        "auxiliary_controls", "feedback_train", "feedback_controls",
    }:
        return ["--base-roles-json", str(support / "base_roles_design_select.json")]
    if node_id == "mechanism_controls":
        return ["--base-roles-json", str(support / "base_roles_design_confirm.json")]
    if node_id == "robustness_cache_build":
        return ["--offline-input", str(_offline_view(root, "design_confirm"))]
    if node_id == "robustness_evaluation":
        return [
            "--labels", str(_labels_npz(root, "design_confirm")),
            "--covariates", str(support / "robustness_covariates.npz"),
            "--subgroup-edges", str(support / "robustness_subgroup_edges.json"),
        ]
    if node_id == "confirmation_native_relation_build":
        return [
            "--input-npz", str(_hlt_view(root, "design_confirm", 0)),
            "--relation-normalizer", str(hlt_relation),
            "--tree-backend-manifest", str(backend),
            "--tree-cache-dir", str(_tree(root, "hlt", "val_design", 0)),
        ]
    if node_id in {"confirmation_train", "capacity_controls"}:
        return [
            *_repeated("--train-cache", {r: _hlt_view(root, "model_train", r) for r in range(4)}),
            "--val-stop-cache", f"0={_hlt_view(root, 'val_stop', 0)}",
            "--design-confirm-cache", f"0={_hlt_view(root, 'design_confirm', 0)}",
            "--train-labels", str(_labels_npz(root, "model_train")),
            "--val-stop-labels", str(_labels_npz(root, "val_stop")),
            "--design-confirm-labels", str(_labels_npz(root, "design_confirm")),
        ]
    if node_id == "scale_input_prepare":
        return [
            "--scale-offline-cache", str(
                _shared_root(root) / "inputs" / "offline" / "scale_train"
            ),
            *_repeated("--scale-hlt-cache", {r: _hlt_cache(root, "scale_train", r) for r in range(4)}),
        ]
    if node_id == "scale_tree_build":
        backend_artifact = load_hashed_json(backend)
        binary = _shared_root(root) / "backend" / str(backend_artifact["binary_filename"])
        return ["--tree-resource", str(tree_resource), "--backend-manifest", str(backend), "--backend-binary", str(binary)]
    if node_id == "scale_normalization":
        return [
            "--normalization-contract", str(_shared_root(root) / "inputs" / "inherited_normalization_contract.json"),
            "--relation-registry", str(support / "relation_family_registry.json"),
            "--raw-input-schema", str(root / "inputs" / "raw_input_schema.json"),
            "--hlt-profile", str(root / "inputs" / "hlt_v3_profile.json"),
            "--tree-resource", str(tree_resource),
        ]
    if node_id == "scale_teacher_train":
        return [
            "--model-contract-o-base", str(support / "model_contract_O_BASE.json"),
            "--model-contract-o-fullrel", str(support / "model_contract_O_FULLREL.json"),
            "--offline-manifest", str(root / "inputs" / "split_manifest.json.gz"),
            "--validation-partition", str(root / "inputs" / "validation_partition_manifest.json.gz"),
            "--screening-registry", str(support / "screening_registry.json"),
            "--relation-registry", str(support / "relation_family_registry.json"),
            "--global-determinism", str(root / "registry" / "global_determinism.json"),
            "--validation-tree-root", str(_shared_root(root) / "inputs" / "region_tree" / "offline"),
            "--scale-train-labels", str(_labels_npz(root, "scale_train")),
        ]
    if node_id == "scale_teacher_adapter_compile":
        return ["--screening-registry", str(support / "screening_registry.json")]
    if node_id == "scale_graph_train":
        return [
            "--val-stop-cache", f"0={_hlt_view(root, 'val_stop', 0)}",
            "--design-confirm-cache", f"0={_hlt_view(root, 'design_confirm', 0)}",
            "--scale-train-labels", str(_labels_npz(root, "scale_train")),
            "--val-stop-labels", str(_labels_npz(root, "val_stop")),
            "--design-confirm-labels", str(_labels_npz(root, "design_confirm")),
        ]
    if node_id == "scale_efficiency":
        return [
            "--cache",
            str(root / "scale_up" / "inputs" / "hlt" / "replica_0.npz"),
        ]
    if node_id == "stack_inference":
        return [
            "--identities-npz", str(support / "stack_val_identities.npz"),
            "--cache", f"0={_hlt_cache(root, 'stack_val', 0)}",
        ]
    if node_id == "stack_selector":
        label_manifest = load_hashed_json(
            root / "inputs" / "final_select_label_manifest.json.gz"
        )
        return [
            "--labels-npz", str(support / "stack_val_identity_labels.npz"),
            "--label-manifest-sha256", str(label_manifest["content_hash"]),
        ]
    if node_id == "postlock_oracle":
        base = _teacher_adapter_config(
            root, {"teacher_id": "O_BASE", "split": "stack_val"}
        )
        full = _teacher_adapter_config(
            root, {"teacher_id": "O_FULLREL", "split": "stack_val"}
        )
        return [
            "--teacher-lock", str(root / "teachers" / "teacher_lock.json"),
            "--adapter-config-o-base", str(base),
            "--adapter-config-o-fullrel", str(full),
        ]
    if node_id == "final_input_preparation":
        return [
            "--input", f"hlt_cache_metadata={_hlt_cache(root, 'final_test', 0) / 'hlt_v3_metadata.json'}",
            "--input", f"identity_labels={support / 'final_test_identity_labels.npz'}",
        ]
    if node_id == "final_test":
        return [
            "--identities-labels-npz", str(support / "final_test_identity_labels.npz"),
            "--cache", f"0={_hlt_cache(root, 'final_test', 0)}",
        ]
    return []


def _row_scoped_arguments(
    node_id: str,
    values: list[str],
    row: Mapping[str, Any] | None,
) -> list[str]:
    scoped = ROW_SCOPED_RUNTIME_OPTIONS.get(node_id, frozenset())
    if not scoped:
        return list(values)
    # Current runtime-manifest v4 derives these paths from the authenticated
    # campaign root.  Retain keyed selection only for reading legacy v3 test
    # fixtures and migration diagnostics.
    if not values:
        return []
    if row is None:
        raise ValueError(f"{node_id} requires a resolved scientific row")
    key = str(row["split"])
    if node_id == "hlt_analogue_target_build":
        key += f":{row['replica']}"
    output = []
    index = 0
    observed = set()
    while index < len(values):
        option = str(values[index])
        if option not in scoped:
            output.append(option)
            index += 1
            continue
        if index + 1 >= len(values):
            raise ValueError(f"{node_id} has an incomplete {option} binding")
        payload = str(values[index + 1])
        binding_key, separator, selected_value = payload.partition("=")
        if not separator:
            raise ValueError(f"{node_id} {option} binding is not keyed")
        if binding_key == key:
            if option in observed:
                raise ValueError(
                    f"{node_id} has duplicate {option} binding for {key}"
                )
            output.extend([option, selected_value])
            observed.add(option)
        index += 2
    if observed != scoped:
        raise ValueError(
            f"{node_id} lacks row-scoped bindings for {key}: "
            f"{sorted(scoped - observed)}"
        )
    return output


def _rows(root: Path, node_id: str) -> list[dict[str, Any]] | None:
    if node_id == "offline_teacher_train":
        return [{"teacher_id": value} for value in ("O_BASE", "O_FULLREL")]
    if node_id in {
        "canonical_target_build",
        "hlt_analogue_target_build",
        "teacher_target_inference",
        "residual_target_build",
    }:
        target_registry = load_hashed_json(
            root / "registry" / "structure_target_registry.json",
            expected_contract="hosd_structure_target_registry_v1",
        )
        kind = {
            "canonical_target_build": "canonical",
            "hlt_analogue_target_build": "hlt_analogue",
            "teacher_target_inference": "teacher_output",
            "residual_target_build": "residual",
        }[node_id]
        return stage_b_wave_rows(
            wave_kind=kind, target_registry=target_registry
        )
    if node_id == "baseline_train":
        return [{"baseline_id": value, "seed": 101} for value in BASELINE_IDS]
    if node_id == "robustness_cache_build":
        return [
            {"profile": profile, "replica": replica}
            for profile in DEGRADATION_PROFILES
            for replica in range(4)
        ]
    if node_id in {"probe_input_materialization", "probe_train"}:
        plan = load_hashed_json(
            root / "job_ledgers" / "stage_c_execution_plan.json"
        )
        return [dict(row) for row in plan["probe_rows"]]
    if node_id in {
        "auxiliary_train",
        "relation_het_auxiliary_train",
        "hlt_self_auxiliary_train",
        "auxiliary_controls",
    }:
        plan = load_hashed_json(
            root / "job_ledgers" / "stage_d_execution_plan.json",
            expected_contract=STAGE_D_PLAN_CONTRACT,
        )
        rows = list(plan["all_rows"])
        if node_id in {
            "relation_het_auxiliary_train",
            "hlt_self_auxiliary_train",
        }:
            phase_lock = load_hashed_json(
                root / "auxiliary" / "single_family_phase_lock.json",
                expected_contract="hosd_single_family_phase_lock_v1",
            )
            resolved = resolve_stage_d_phase_two(
                stage_d_plan=plan, phase_lock=phase_lock
            )
            rows = [
                *[
                    row
                    for row in rows
                    if row["phase"]
                    not in {"LOCKED_RELATION_HET", "MATCHED_HLT_SELF"}
                ],
                *resolved,
            ]
        if node_id == "auxiliary_train":
            rows = [
                row
                for row in rows
                if row["row_kind"] == "SCIENTIFIC"
                and row["phase"] == "PRIMARY"
            ]
        elif node_id == "relation_het_auxiliary_train":
            rows = [row for row in rows if row["phase"] == "LOCKED_RELATION_HET"]
        elif node_id == "hlt_self_auxiliary_train":
            rows = [row for row in rows if row["row_kind"] == "HLT_SELF"]
        else:
            rows = [row for row in rows if row["row_kind"] not in {"SCIENTIFIC", "HLT_SELF"}]
        return [dict(row) for row in rows]
    if node_id in {"feedback_train", "feedback_controls"}:
        plan = load_hashed_json(
            root / "job_ledgers" / "stage_e_execution_plan.json",
            expected_contract=STAGE_E_PLAN_CONTRACT,
        )
        kinds = (
            {"SCIENTIFIC", "REFERENCE_BASELINE"}
            if node_id == "feedback_train"
            else {"CONTROL"}
        )
        return [
            dict(row) for row in plan["all_rows"] if row["row_kind"] in kinds
        ]
    if node_id in {"combination_train", "pcgrad_control"}:
        plan = load_hashed_json(
            root / "job_ledgers" / "stage_f_execution_plan.json",
            expected_contract=STAGE_F_PLAN_CONTRACT,
        )
        if node_id == "pcgrad_control":
            return [dict(plan["pcgrad_control"])]
        beam = load_hashed_json(
            root / "combinations" / "beam_completion.json",
            expected_contract=COMBINATION_BEAM_COMPLETION_CONTRACT,
        )
        if beam.get("stage_f_plan_sha256") != plan["content_hash"]:
            raise ValueError("combination beam completion lineage differs")
        return [
            *[dict(row) for row in plan["mandatory_combinations"]],
            *[
                dict(row)
                for row in beam["promotion"]["promoted_graphs"]
            ],
        ]
    if node_id == "robustness_evaluation":
        plan_path = root / "robustness" / "evaluation_plan.json"
        plan = load_hashed_json(
            plan_path, expected_contract=ROBUSTNESS_PLAN_CONTRACT
        )
        return [dict(row) for row in plan["rows"]]
    if node_id == "discovery_export":
        registry = load_hashed_json(
            root / "registry" / "locked_graph_registry.json",
            expected_contract=GRAPH_REGISTRY_CONTRACT,
        )
        unique = {}
        for definition in registry["definitions_by_role"].values():
            unique.setdefault(str(definition["graph_id"]), dict(definition))
        return [
            {"row_id": f"DISCOVERY_EXPORT_{index:03d}", **definition}
            for index, definition in enumerate(
                (unique[key] for key in sorted(unique))
            )
        ]
    if node_id == "confirmation_train":
        plan = load_hashed_json(
            root / "confirmation_500k" / "execution_plan.json",
            expected_contract=CONFIRMATION_PLAN_CONTRACT,
        )
        return [dict(row) for row in plan["training_rows"]]
    if node_id == "capacity_controls":
        plan = load_hashed_json(
            root / "confirmation_500k" / "capacity_execution_plan.json",
            expected_contract="hosd_capacity_control_execution_plan_v1",
        )
        return [dict(row) for row in plan["rows"]]
    if node_id in {"scale_input_prepare", "scale_tree_build"}:
        return [
            {"coordinate": coordinate}
            for coordinate in ("offline", "0", "1", "2", "3")
        ]
    if node_id == "scale_native_relation_build":
        plan = load_hashed_json(
            root / "scale_up" / "execution_plan.json",
            expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
        )
        required = any(
            (
                row["graph_definition"].get("baseline_id")
                == "H_NATIVE_REL_AUX"
                or row["graph_definition"].get("graph", {}).get(
                    "native_relation_auxiliary"
                )
                is not None
            )
            for row in plan["graph_rows"]
        )
        return (
            [{"replica": str(replica)} for replica in range(4)]
            if required
            else [{"replica": "none"}]
        )
    if node_id in {
        "scale_teacher_train",
        "scale_teacher_target_inference",
        "scale_target_build",
        "scale_graph_train",
        "scale_efficiency",
        "stack_inference",
    }:
        plan = load_hashed_json(
            root / "scale_up" / "execution_plan.json",
            expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
        )
        key = {
            "scale_teacher_train": "teacher_rows",
            "scale_teacher_target_inference": "teacher_rows",
            "scale_target_build": "target_refit_rows",
            "scale_graph_train": "graph_rows",
            "scale_efficiency": "graph_rows",
            "stack_inference": "graph_rows",
        }[node_id]
        return [dict(row) for row in plan[key]]
    return None


def resolve_node_argv(
    *,
    node: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    campaign_root: str | Path,
    coordinate: int,
) -> tuple[list[str] | None, Mapping[str, Any] | None]:
    validate_content_hash(
        runtime_manifest, expected_contract=RUNTIME_MANIFEST_CONTRACT
    )
    support_sha = runtime_manifest.get("runtime_support_sha256")
    if support_sha is not None:
        support_manifest = load_hashed_json(
            Path(campaign_root)
            / "registry"
            / "runtime_support"
            / "manifest.json",
            expected_contract="hosd_runtime_support_manifest_v3",
        )
        if (
            support_manifest["content_hash"] != support_sha
            or support_manifest.get("source") != runtime_manifest.get("source")
        ):
            raise ValueError("runtime support artifacts drifted after preparation")
    node_id = str(node["node_id"])
    infrastructure = runtime_manifest[
        "infrastructure_arguments_by_node"
    ].get(node_id, [])
    referenced_files: set[str] = set()
    referenced_directories: set[str] = set()
    formatter = string.Formatter()
    for value in infrastructure:
        for _, field_name, _, _ in formatter.parse(str(value)):
            if field_name is None:
                continue
            if field_name.startswith("file_"):
                referenced_files.add(field_name.removeprefix("file_"))
            elif field_name.startswith("directory_"):
                referenced_directories.add(
                    field_name.removeprefix("directory_")
                )
    for key in sorted(referenced_files):
        record = runtime_manifest.get("files", {}).get(key)
        if record is None:
            raise ValueError(f"runtime node references unbound file: {key}")
        path = Path(record["path"]).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"runtime input is absent or unsafe at execution: {key}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        observed = digest.hexdigest()
        if observed != record["sha256"]:
            raise ValueError(
                f"runtime input drifted after manifest publication: {key}"
            )
    for key in sorted(referenced_directories):
        value = runtime_manifest.get("directories", {}).get(key)
        if value is None:
            raise ValueError(
                f"runtime node references unbound directory: {key}"
            )
        path = Path(value).resolve()
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(
                f"runtime directory is absent or unsafe at execution: {key}"
            )
    root = Path(campaign_root).resolve()
    rows = _rows(root, str(node["node_id"]))
    row = None
    if rows is not None:
        if int(coordinate) >= len(rows):
            return None, None
        row = rows[int(coordinate)]
    elif int(coordinate) != 0:
        return None, None
    context = {
        "campaign_root": str(root),
        "coordinate": str(int(coordinate)),
        **{
            f"file_{key}": str(value["path"])
            for key, value in runtime_manifest.get("files", {}).items()
        },
        **{
            f"directory_{key}": str(value)
            for key, value in runtime_manifest.get("directories", {}).items()
        },
        **({key: str(value) for key, value in row.items()} if row else {}),
    }
    infrastructure = _row_scoped_arguments(
        node_id, list(infrastructure), row
    )
    formatted = [str(value).format_map(context) for value in infrastructure]
    formatted.extend(_derived_infrastructure(root, node_id, row))
    if support_sha is not None:
        by_path = {
            str(Path(path).resolve()): key
            for key, path in support_manifest.get("paths", {}).items()
        }
        for raw in formatted:
            candidate = str(raw).split("=", 1)[-1]
            resolved = str(Path(candidate).resolve())
            key = by_path.get(resolved)
            if key is not None and (
                _sha256_file(Path(resolved))
                != support_manifest["file_sha256"].get(key)
            ):
                raise ValueError(
                    f"runtime support artifact drifted after preparation: {key}"
                )
    argv = [
        "python",
        str(node["entrypoint"]),
        "--campaign-root",
        str(root),
        *FIXED_ARGUMENTS.get(str(node["node_id"]), []),
        *formatted,
    ]
    if node["node_id"] == "storage_measurement":
        argv.extend(
            [
                "--evidence-output",
                str(root / "job_ledgers" / "storage_probe_evidence.json"),
                "--output",
                str(root / "job_ledgers" / "runtime_storage_measurements.json"),
            ]
        )
    # Scientific coordinate fields always come from authenticated plans.
    if row is not None:
        if node["node_id"] == "offline_teacher_train":
            argv.extend(
                [
                    "--execute-training",
                    "--teacher-id",
                    str(row["teacher_id"]),
                ]
            )
        elif node["node_id"] in {
            "canonical_target_build",
            "hlt_analogue_target_build",
        }:
            argv.extend(
                [
                    "--split",
                    str(row["split"]),
                    "--artifact-kind",
                    str(row["artifact_kind"]),
                    "--cache-id",
                    str(row["row_id"]),
                    *[
                        item
                        for target_id in row["target_ids"]
                        for item in ("--target-id", str(target_id))
                    ],
                ]
            )
            if "replica" in row:
                argv.extend(
                    ["--hlt-replica-id", str(row["replica"])]
                )
            argv.extend(
                [
                    "--output-dir",
                    str(
                        root
                        / "targets"
                        / (
                            "canonical"
                            if node["node_id"] == "canonical_target_build"
                            else "hlt_analogues"
                        )
                        / row["split"]
                        / (
                            Path()
                            if node["node_id"] == "canonical_target_build"
                            else Path(f"replica_{row['replica']}")
                        )
                    ),
                ]
            )
        elif node["node_id"] == "teacher_target_inference":
            argv.extend(
                [
                    "--teacher-lock",
                    str(root / "teachers" / "teacher_lock.json"),
                    "--teacher-id",
                    str(row["teacher_id"]),
                    "--split",
                    str(row["split"]),
                    "--output-dir",
                    str(
                        root
                        / "teachers"
                        / "outputs"
                        / row["split"]
                        / row["teacher_id"]
                    ),
                ]
            )
        elif node["node_id"] == "residual_target_build":
            argv.extend(
                [
                    "--kind",
                    "residual",
                    "--cache-id",
                    str(row["row_id"]),
                    "--canonical-cache",
                    str(root / "targets" / "canonical" / row["split"]),
                    "--hlt-cache",
                    str(
                        root
                        / "targets"
                        / "hlt_analogues"
                        / row["split"]
                        / f"replica_{row['replica']}"
                    ),
                    "--output-dir",
                    str(
                        root
                        / "targets"
                        / "residuals"
                        / row["split"]
                        / f"replica_{row['replica']}"
                    ),
                    "--auto-target-pairs",
                ]
            )
        elif node["node_id"] == "baseline_train":
            argv.extend(
                ["--baseline-id", row["baseline_id"], "--seed", str(row["seed"])]
            )
            if row["baseline_id"] in {
                "H_KD_LOGIT_O_BASE",
                "H_KD_LOGIT_O_FULLREL",
            }:
                teacher_id = row["baseline_id"].removeprefix(
                    "H_KD_LOGIT_"
                )
                argv.extend(
                    [
                        "--train-teacher-logits",
                        str(
                            root
                            / "teachers"
                            / "outputs"
                            / "model_train"
                            / teacher_id
                        ),
                    ]
                )
        elif node["node_id"] in {
            "auxiliary_train",
            "relation_het_auxiliary_train",
            "hlt_self_auxiliary_train",
            "auxiliary_controls",
            "feedback_train",
            "feedback_controls",
        }:
            argv.extend(["--row-id", str(row["row_id"])])
            if node["node_id"] in {
                "relation_het_auxiliary_train",
                "hlt_self_auxiliary_train",
            }:
                argv.extend(
                    [
                        "--phase-lock",
                        str(
                            root
                            / "auxiliary"
                            / "single_family_phase_lock.json"
                        ),
                    ]
                )
        elif node["node_id"] == "probe_input_materialization":
            argv.extend(
                [
                    "--row-id",
                    str(row["row_id"]),
                    "--output-dir",
                    str(root / "probes" / "inputs" / row["row_id"]),
                ]
            )
            if row["target_id"].startswith("T_OFFLINE_LOGITS_"):
                teacher_id = row["target_id"].removeprefix(
                    "T_OFFLINE_LOGITS_"
                )
                target_roots = {
                    role: root
                    / "teachers"
                    / "outputs"
                    / source_role
                    / teacher_id
                    for role, source_role in (
                        ("model_train", "model_train"),
                        ("val_stop", "val_stop"),
                        ("design_select", "val_design"),
                    )
                }
            elif row["target_id"] == "T_OFFLINE_POOLED_LATENT":
                target_roots = {
                    role: root
                    / "teachers"
                    / "outputs"
                    / source_role
                    / "O_BASE"
                    for role, source_role in (
                        ("model_train", "model_train"),
                        ("val_stop", "val_stop"),
                        ("design_select", "val_design"),
                    )
                }
            else:
                target_roots = {
                    role: root / "targets" / "canonical" / source_role
                    for role, source_role in (
                        ("model_train", "model_train"),
                        ("val_stop", "val_stop"),
                        ("design_select", "val_design"),
                    )
                }
            if row["target_id"] not in {
                "T_HLT_TRACK_PAIR_13",
                "T_HLT_REGION_PAIR_8",
            }:
                argv.extend(
                    [
                        item
                        for role, path in target_roots.items()
                        for item in (
                            "--target-cache",
                            f"{role}={path}",
                        )
                    ]
                )
        elif node["node_id"] == "probe_train":
            probe_root = root / "probes" / "inputs" / row["row_id"]
            argv.extend(
                [
                    "--row-id",
                    str(row["row_id"]),
                    "--train-npz",
                    str(probe_root / "model_train.npz"),
                    "--val-stop-npz",
                    str(probe_root / "val_stop.npz"),
                    "--design-select-npz",
                    str(probe_root / "design_select.npz"),
                    "--input-lineage",
                    str(probe_root / "input_lineage.json"),
                    "--output-dir",
                    str(root / "probes" / row["row_id"]),
                ]
            )
            if row["probe_kind"] in {"P_LINEAR", "P_SHALLOW"}:
                run = root / "baselines" / "H_BASE" / "seed_101"
                argv.extend(
                    [
                        "--probe-encoder-lock",
                        str(
                            root
                            / "probes"
                            / "frozen_taps"
                            / "probe_encoder_lock.json"
                        ),
                        "--baseline-checkpoint",
                        str(run / "best_model_val.pt"),
                    ]
                )
                argv.extend(
                    _derived_infrastructure(root, "probe_tap_capture", row)
                )
        elif node["node_id"] in {"combination_train", "pcgrad_control"}:
            argv.extend(
                ["--mode", "execute", "--graph-id", str(row["graph_id"])]
            )
        elif node["node_id"] == "robustness_evaluation":
            replicas = (
                (0,)
                if row["replica_policy"] == "R_FIXED"
                else (0, 1, 2, 3)
            )
            argv.extend(
                [
                    "--row-id",
                    str(row["row_id"]),
                    "--export",
                    str(
                        root
                        / "confirmation_500k"
                        / "discovery_exports"
                        / f"{row['graph_id']}.pt"
                    ),
                    "--output",
                    str(
                        root
                        / "robustness"
                        / "rows"
                        / f"{row['row_id']}.json"
                    ),
                    *[
                        item
                        for replica in replicas
                        for item in (
                            "--cache",
                            str(
                                replica
                            )
                            + "="
                            + str(
                                root
                                / "robustness"
                                / "caches"
                                / row["degradation_profile"]
                                / f"replica_{replica}"
                            ),
                        )
                    ],
                ]
            )
        elif node["node_id"] == "robustness_cache_build":
            argv.extend(
                [
                    "--profile",
                    str(row["profile"]),
                    "--replica",
                    str(row["replica"]),
                    "--output-dir",
                    str(
                        root
                        / "robustness"
                        / "caches"
                        / row["profile"]
                        / f"replica_{row['replica']}"
                    ),
                ]
            )
        elif node["node_id"] == "discovery_export":
            argv.extend(["--graph-id", str(row["graph_id"])])
        elif node["node_id"] == "confirmation_train":
            argv.extend(["--row-id", str(row["row_id"])])
        elif node["node_id"] == "capacity_controls":
            argv.extend(["--row-id", str(row["row_id"])])
        elif node["node_id"] in {
            "scale_input_prepare",
            "scale_tree_build",
        }:
            argv.extend(["--coordinate", str(row["coordinate"])])
        elif node["node_id"] == "scale_teacher_train":
            argv.extend(
                ["--mode", "teacher", "--teacher-id", str(row["teacher_id"])]
            )
        elif node["node_id"] == "scale_teacher_target_inference":
            argv.extend(
                [
                    "--mode",
                    "teacher-inference",
                    "--teacher-id",
                    str(row["teacher_id"]),
                ]
            )
        elif node["node_id"] == "scale_target_build":
            argv.extend(
                ["--mode", "target", "--target-id", str(row["target_id"])]
            )
        elif node["node_id"] == "scale_native_relation_build":
            argv.extend(["--replica", str(row["replica"])])
        elif node["node_id"] == "scale_graph_train":
            argv.extend(
                [
                    "--mode",
                    "graph",
                    "--graph-id",
                    str(row["graph_id"]),
                    "--seed",
                    str(row["seed"]),
                ]
            )
        elif node["node_id"] == "scale_efficiency":
            argv.extend(
                [
                    "--graph-id",
                    str(row["graph_id"]),
                    "--seed",
                    str(row["seed"]),
                    "--coordinate",
                    str(int(coordinate)),
                    "--output",
                    str(
                        root
                        / "scale_up"
                        / "efficiency"
                        / (
                            f"{row['graph_id']}__seed_"
                            f"{row['seed']}.json"
                        )
                    ),
                ]
            )
        elif node["node_id"] == "stack_inference":
            result = load_hashed_json(
                root
                / "scale_up"
                / "results"
                / f"{row['graph_id']}__seed_{row['seed']}.json"
            )
            argv.extend(
                [
                    "--graph-id",
                    str(row["graph_id"]),
                    "--seed",
                    str(row["seed"]),
                    "--export",
                    str(
                        root
                        / "scale_up"
                        / "exports"
                        / f"{row['graph_id']}__seed_{row['seed']}.pt"
                    ),
                    "--checkpoint-sha256",
                    str(result["checkpoint_sha256"]),
                    "--export-sha256",
                    str(result["deployable_export_sha256"]),
                    "--lineage",
                    f"scale_row_result={result['content_hash']}",
                    "--output",
                    str(
                        root
                        / "selection_predictions"
                        / "stack_val"
                        / "rows"
                        / f"{row['graph_id']}__seed_{row['seed']}.json"
                    ),
                ]
            )
    node_id = str(node["node_id"])
    if node_id == "single_family_phase_lock":
        argv.extend(
            [
                item
                for row in _rows(root, "auxiliary_train") or []
                for item in (
                    "--result",
                    str(
                        root
                        / "auxiliary"
                        / row["row_id"]
                        / "seed_101"
                        / "design_select_result.json"
                    ),
                )
            ]
        )
    elif node_id == "offline_teacher_lock":
        argv.extend(
            [
                "--training-manifest",
                str(root / "teachers" / "training_manifest.json"),
                "--o-base-completion",
                str(
                    root
                    / "teachers"
                    / "target_500k"
                    / "O_BASE"
                    / "training_completion.json"
                ),
                "--o-fullrel-completion",
                str(
                    root
                    / "teachers"
                    / "target_500k"
                    / "O_FULLREL"
                    / "training_completion.json"
                ),
            ]
        )
    elif node_id == "probe_tap_capture":
        run = root / "baselines" / "H_BASE" / "seed_101"
        argv.extend(
            [
                "--baseline-completion",
                str(run / "baseline_completion.json"),
                "--checkpoint",
                str(run / "best_model_val.pt"),
            ]
        )
    elif node_id == "teacher_target_finalize":
        argv.extend(
            [
                "--teacher-lock",
                str(root / "teachers" / "teacher_lock.json"),
                "--wave-completion",
                str(
                    root
                    / "teachers"
                    / "teacher_output_cache_completion.json"
                ),
            ]
        )
    elif node_id == "single_family_select":
        argv.extend(["--mode", "final-selection"])
        argv.extend(
            [
                "--phase-lock",
                str(root / "auxiliary" / "single_family_phase_lock.json"),
            ]
        )
        rows = []
        for upstream in (
            "auxiliary_train",
            "relation_het_auxiliary_train",
            "hlt_self_auxiliary_train",
            "auxiliary_controls",
        ):
            rows.extend(_rows(root, upstream) or [])
        argv.extend(
            [
                item
                for row in rows
                for item in (
                    "--result",
                    str(
                        root
                        / "auxiliary"
                        / row["row_id"]
                        / "seed_101"
                        / "design_select_result.json"
                    ),
                )
            ]
        )
    elif node_id == "feedback_select":
        rows = [
            *(_rows(root, "feedback_train") or []),
            *(_rows(root, "feedback_controls") or []),
        ]
        argv.extend(
            [
                item
                for row in rows
                for item in (
                    "--result",
                    str(
                        root
                        / "feedback"
                        / row["row_id"]
                        / "seed_101"
                        / "design_select_result.json"
                    ),
                )
            ]
        )
    elif node_id == "combination_select":
        full_rows = _rows(root, "combination_train") or []
        argv.extend(
            [
                item
                for row in full_rows
                for item in (
                    "--result",
                    str(
                        root
                        / "combinations"
                        / row["graph_id"]
                        / "seed_101"
                        / "design_select_result.json"
                    ),
                )
            ]
        )
        beam = load_hashed_json(
            root / "combinations" / "beam_completion.json",
            expected_contract=COMBINATION_BEAM_COMPLETION_CONTRACT,
        )
        argv.extend(
            [
                item
                for graph_id in beam["promotion"]["promoted_graph_ids"]
                for item in ("--beam-winner", str(graph_id))
            ]
        )
    elif node_id == "robustness_report":
        rows = _rows(root, "robustness_evaluation") or []
        argv.extend(
            [
                item
                for row in rows
                for item in (
                    "--result",
                    str(root / "robustness" / "rows" / f"{row['row_id']}.json"),
                )
            ]
        )
    elif node_id == "robustness_plan_compile":
        registry = load_hashed_json(
            root / "registry" / "locked_graph_registry.json",
            expected_contract=GRAPH_REGISTRY_CONTRACT,
        )
        graph_ids = sorted(
            {
                str(definition["graph_id"])
                for definition in registry["definitions_by_role"].values()
            }
        )
        for graph_id in graph_ids:
            argv.extend(["--graph-id", graph_id])
    elif node_id == "confirmation_compile":
        argv.extend(
            [
                "--graph-registry",
                str(root / "registry" / "locked_graph_registry.json"),
            ]
        )
    elif node_id == "confirmation_train":
        argv.extend(
            [
                "--stage-d-loader-root",
                str(root / "loaders" / "stage_d"),
                "--native-relation-design-confirm",
                str(
                    root
                    / "targets"
                    / "native_relations"
                    / "design_confirm"
                    / "replica_0.npz"
                ),
            ]
        )
    elif node_id == "scale_plan_compile":
        campaign = load_hashed_json(root / "campaign_spec.json")
        digest = campaign.get("shared_parent_hashes", {}).get(
            "scale_train_manifest"
        )
        if digest is None:
            raise ValueError("campaign lacks the scale-train manifest hash")
        argv.extend(["--scale-train-manifest-sha256", str(digest)])
    elif node_id == "scale_teacher_train":
        normalizers = load_hashed_json(
            root / "scale_up" / "normalization" / "completion.json",
            expected_contract="hosd_scale_normalizer_completion_v1",
        )
        argv.extend(
            [
                "--scale-offline-relation-normalizer",
                str(normalizers["artifact_paths"]["offline_relation"]),
            ]
        )
        if row is not None and row.get("teacher_id") == "O_FULLREL":
            argv.extend(
                [
                    "--scale-offline-region-normalizer",
                    str(normalizers["artifact_paths"]["offline_region"]),
                ]
            )
    elif node_id == "scale_target_build":
        trees = load_hashed_json(
            root / "scale_up" / "trees" / "completion.json",
            expected_contract="hosd_scale_tree_wave_completion_v1",
        )
        normalizers = load_hashed_json(
            root / "scale_up" / "normalization" / "completion.json",
            expected_contract="hosd_scale_normalizer_completion_v1",
        )
        argv.extend(
            [
                "--scale-offline-input",
                str(
                    root
                    / "scale_up"
                    / "inputs"
                    / "offline"
                    / "scale_train.npz"
                ),
                "--scale-offline-tree",
                str(
                    root
                    / "scale_up"
                    / "trees"
                    / "offline"
                    / "scale_train_exclusive_ca_v1"
                ),
                "--tree-backend-manifest",
                str(trees["backend_manifest_path"]),
                "--scale-offline-relation-normalizer",
                str(normalizers["artifact_paths"]["offline_relation"]),
                "--scale-hlt-relation-normalizer",
                str(normalizers["artifact_paths"]["shared_hlt_relation"]),
                "--scale-offline-region-normalizer",
                str(normalizers["artifact_paths"]["offline_region"]),
                "--scale-hlt-region-normalizer",
                str(normalizers["artifact_paths"]["shared_hlt_region"]),
            ]
        )
        for replica in range(4):
            argv.extend(
                [
                    "--scale-hlt-input",
                    f"{replica}={root / 'scale_up' / 'inputs' / 'hlt' / f'replica_{replica}.npz'}",
                    "--scale-hlt-tree",
                    f"{replica}={root / 'scale_up' / 'trees' / 'hlt' / f'replica_{replica}'}",
                ]
            )
    elif node_id == "scale_graph_train":
        argv.extend(
            [
                "--stage-d-loader-root",
                str(root / "loaders" / "stage_d"),
            ]
        )
        for replica in range(4):
            argv.extend(
                [
                    "--scale-train-cache",
                    f"{replica}={root / 'scale_up' / 'inputs' / 'hlt' / f'replica_{replica}.npz'}",
                ]
            )
    elif node_id == "confirmation_aggregate":
        confirmation_rows = _rows(root, "confirmation_train") or []
        capacity_rows = _rows(root, "capacity_controls") or []
        argv.extend(
            [
                item
                for row in confirmation_rows
                for item in (
                    "--training-result",
                    str(
                        root
                        / "confirmation_500k"
                        / "results"
                        / f"{row['row_id']}.json"
                    ),
                )
            ]
        )
        argv.extend(
            [
                item
                for row in capacity_rows
                for item in (
                    "--capacity-result",
                    str(
                        root
                        / "confirmation_500k"
                        / "capacity_results"
                        / f"{row['row_id']}.json"
                    ),
                )
            ]
        )
        argv.extend(
            [
                "--capacity-execution-plan",
                str(
                    root
                    / "confirmation_500k"
                    / "capacity_execution_plan.json"
                ),
            ]
        )
    elif node_id == "scale_finalize":
        plan = load_hashed_json(
            root / "scale_up" / "execution_plan.json",
            expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
        )
        for plan_row in plan["teacher_rows"]:
            completion = load_hashed_json(
                root
                / "scale_up"
                / "teachers"
                / plan_row["teacher_id"]
                / "completion.json"
            )
            argv.extend(
                [
                    "--teacher-completion",
                    f"{plan_row['teacher_id']}={completion['content_hash']}",
                ]
            )
        for plan_row in plan["target_refit_rows"]:
            completion = load_hashed_json(
                root
                / "scale_up"
                / "targets"
                / plan_row["target_id"]
                / "completion.json"
            )
            argv.extend(
                [
                    "--target-completion",
                    f"{plan_row['target_id']}={completion['content_hash']}",
                ]
            )
        for plan_row in plan["graph_rows"]:
            argv.extend(
                [
                    "--graph-result",
                    str(
                        root
                        / "scale_up"
                        / "results"
                        / (
                            f"{plan_row['graph_id']}__seed_"
                            f"{plan_row['seed']}.json"
                        )
                    ),
                ]
            )
        for key, relative in (
            ("scale_inputs", "scale_up/inputs/completion.json"),
            ("scale_trees", "scale_up/trees/completion.json"),
            (
                "scale_normalizers",
                "scale_up/normalization/completion.json",
            ),
            ("teacher_lock", "scale_up/teachers/teacher_lock.json"),
            (
                "teacher_adapters",
                "scale_up/teacher_outputs/adapter_configs/completion.json",
            ),
            ("teacher_outputs", "scale_up/teacher_outputs/completion.json"),
            ("target_wave", "scale_up/target_completion.json"),
            (
                "scale_native_relations",
                "scale_up/targets/native_relations/completion.json",
            ),
            ("graph_wave", "scale_up/graph_completion.json"),
        ):
            artifact = load_hashed_json(root / relative)
            argv.extend(
                [
                    "--pre-student-artifact",
                    f"{key}={artifact['content_hash']}",
            ]
        )
    elif node_id == "scale_export_audit":
        plan = load_hashed_json(
            root / "scale_up" / "execution_plan.json",
            expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
        )
        for plan_row in plan["graph_rows"]:
            argv.extend(
                [
                    "--export",
                    str(
                        root
                        / "scale_up"
                        / "exports"
                        / (
                            f"{plan_row['graph_id']}__seed_"
                            f"{plan_row['seed']}.pt.json"
                        )
                    ),
                ]
            )
    elif node_id == "stack_selector":
        argv.extend(
            [
                "--capacity-json",
                str(
                    root
                    / "selection_predictions"
                    / "stack_val"
                    / "capacity.json"
                ),
            ]
        )
        for plan_row in _rows(root, "stack_inference") or []:
            argv.extend(
                [
                    "--prediction",
                    str(
                        root
                        / "selection_predictions"
                        / "stack_val"
                        / "rows"
                        / (
                            f"{plan_row['graph_id']}__seed_"
                            f"{plan_row['seed']}.json"
                        )
                    ),
                ]
            )
    elif node_id == "finalist_lock":
        campaign = load_hashed_json(root / "campaign_spec.json")
        parents = load_hashed_json(
            root / "inputs" / "resolved_inherited_parent_lock.json"
        )
        parent_hashes = {
            row["parent_id"]: row["content_hash"]
            for row in parents["requirements"]
            if row.get("content_hash") is not None
        }
        trace = load_hashed_json(
            root / "selection" / "stack_selector_trace.json"
        )
        artifacts = {
            "split_manifest": campaign.get("split_manifest_hash")
            or campaign.get("shared_parent_hashes", {}).get(
                "split_manifest"
            )
            or parent_hashes["split_audit"],
            "validation_partition": parent_hashes[
                "validation_partition_manifest"
            ],
            "scale_pool": parent_hashes["scale_train_manifest"],
            "hlt_profile": parent_hashes["hlt_v3_profile"],
            "hlt_replica_manifest": parent_hashes[
                "hlt_replica_manifest"
            ],
            "hlt_cache_audit": parent_hashes[
                "hlt_v3_degradation_audit"
            ],
            "target_capability_audit": load_hashed_json(
                root / "capability" / "target_capability_audit.json"
            )["content_hash"],
            "target_registry": load_hashed_json(
                root / "registry" / "structure_target_registry.json"
            )["content_hash"],
            "confirmation_summary": load_hashed_json(
                root / "confirmation_500k" / "summary.json"
            )["content_hash"],
            "scale_shortlist": load_hashed_json(
                root / "selection" / "locked_scale_shortlist.json"
            )["content_hash"],
            "scale_export_audit": load_hashed_json(
                root / "scale_up" / "export_audit.json"
            )["content_hash"],
            "label_manifest": trace["label_manifest_sha256"],
            "selector_metrics": canonical_sha256(
                [
                    row["metrics_hashes"]
                    for row in trace["eligible_graphs"]
                ]
            ),
            "selector_trace": trace["content_hash"],
        }
        for key, digest in sorted(artifacts.items()):
            argv.extend(["--lineage", f"{key}={digest}"])
    elif node_id == "execution_lock":
        controls = load_hashed_json(
            root / "selection" / "finalist_control_completion.json"
        )
        for key, digest in sorted(controls["control_hashes"].items()):
            argv.extend(["--control", f"{key}={digest}"])
    elif node_id == "final_report":
        artifact_paths = {
            "final_evaluation": root
            / "final_test"
            / "final_evaluation.json",
            "final_execution_lock": root
            / "selection"
            / "final_test_execution_lock.json",
            "finalist_lock": root
            / "selection"
            / "locked_hosd_finalists.json",
            "finalist_controls": root
            / "selection"
            / "finalist_control_completion.json",
            "postlock_oracle": root
            / "postlock_oracle_diagnostics"
            / "completion.json",
            "robustness_summary": root
            / "robustness"
            / "summary.json",
            "scale_efficiency": root
            / "scale_up"
            / "efficiency"
            / "completion.json",
        }
        argv.extend(
            [
                "--rows-json",
                str(root / "final_test" / "final_evaluation.json"),
            ]
        )
        for name, path in sorted(artifact_paths.items()):
            argv.extend(
                [
                    "--artifact",
                    f"{name}={load_hashed_json(path)['content_hash']}",
                ]
            )
    elif node_id == "final_plots":
        argv.extend(
            [
                "--report",
                str(root / "reports" / "final_report.json"),
                "--output-dir",
                str(root / "reports" / "plots"),
                "--manifest",
                str(root / "reports" / "plots" / "manifest.json"),
            ]
        )
    if node_id in {
        "combination_beam",
        "combination_train",
        "pcgrad_control",
    } and "--member-loader-root" not in argv:
        argv.extend(
            [
                "--member-loader-root",
                str(root / "loaders" / "stage_d"),
            ]
        )
    return argv, row


__all__ = [
    "BASELINE_IDS",
    "FIXED_ARGUMENTS",
    "ROW_SCOPED_RUNTIME_OPTIONS",
    "resolve_node_argv",
]
