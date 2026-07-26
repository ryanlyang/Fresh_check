"""Two-stage raw and constructed-input audits for the Step-8 campaign."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
from jetclass_fresh.jetclass_data import LABEL_NAMES, SPLIT_ORDER, SplitManifest

from .ca_tree import (
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    ANGULAR_TREE_PROBE_CONTRACT,
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
)
from .contracts import validate_content_hash, with_content_hash
from .normalization import validate_relation_normalization_artifact
from .provenance import validate_raw_input_schema_contract
from .region_normalization import validate_region_normalization
from .splits import (
    RELATIONAL_HLT_BINDING_CONTRACT,
    RelationalSplitConfig,
    validate_relational_split_manifest,
)


PRECONSTRUCTION_AUDIT_CONTRACT = "relational_part_preconstruction_input_audit_v1"
POSTCONSTRUCTION_AUDIT_CONTRACT = "relational_part_postconstruction_input_audit_v1"
RAW_AUDIT_SALT = "rpt_raw_audit_v1"


def _identity_key(identity: Any) -> str:
    return identity.key() if hasattr(identity, "key") else str(identity)


def select_raw_audit_identities(
    manifest: SplitManifest,
    *,
    miniature: bool = False,
) -> dict[str, Any]:
    quotas = {
        split: (1_000 if split == "model_train" else 100)
        for split in ("model_train", "model_val", "stack_val", "final_test")
    }
    selected: dict[str, list[int]] = {}
    identity_keys: dict[str, list[str]] = {}
    for split, quota in quotas.items():
        indices = []
        keys = []
        for label in range(len(LABEL_NAMES)):
            candidates = [
                index
                for index, identity in enumerate(manifest.splits[split])
                if int(identity.label) == label
            ]
            candidates.sort(
                key=lambda index: (
                    hashlib.sha256(
                        RAW_AUDIT_SALT.encode("utf-8")
                        + _identity_key(
                            manifest.splits[split][index]
                        ).encode("utf-8")
                    ).digest(),
                    _identity_key(manifest.splits[split][index]),
                )
            )
            take = min(quota, len(candidates)) if miniature else quota
            if len(candidates) < take or (not miniature and len(candidates) < quota):
                raise ValueError(
                    f"{split} class {label} is too small for the raw audit"
                )
            indices.extend(candidates[:take])
        indices.sort()
        keys = [
            _identity_key(manifest.splits[split][index])
            for index in indices
        ]
        selected[split] = indices
        identity_keys[split] = keys
    return {
        "salt": RAW_AUDIT_SALT,
        "quota_per_class": quotas,
        "selected_indices": selected,
        "selected_counts": {
            split: len(indices) for split, indices in selected.items()
        },
        "selected_identity_sha256": {
            split: hashlib.sha256(
                "\n".join(identity_keys[split]).encode("utf-8")
            ).hexdigest()
            for split in selected
        },
    }


def audit_raw_sample(
    tokens: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
) -> dict[str, Any]:
    values = np.asarray(tokens)
    valid = np.asarray(mask)
    truth = np.asarray(labels)
    if values.dtype != np.float32:
        raise ValueError("raw audit tokens must be float32")
    if valid.dtype != np.bool_:
        raise ValueError("raw audit mask must be boolean")
    if truth.dtype != np.int64:
        raise ValueError("raw audit labels must be int64")
    if (
        values.ndim != 3
        or values.shape[-1] != 14
        or valid.shape != values.shape[:2]
        or truth.shape != (values.shape[0],)
    ):
        raise ValueError("raw audit array shapes differ from the locked schema")
    if bool(((truth < 0) | (truth >= len(LABEL_NAMES))).any()):
        raise ValueError("raw audit labels lie outside the ten-class domain")
    selected = values[valid]
    if not np.isfinite(selected).all():
        raise FloatingPointError("raw audit contains nonfinite valid values")
    if bool((values[~valid] != 0).any()):
        raise ValueError("raw audit padding is not exactly zero")
    counts = valid.sum(axis=1)
    if bool((counts < 1).any()) or bool((counts > 128).any()):
        raise ValueError("raw audit constituent count is outside 1..128")
    pid = selected[:, 5:10]
    nearest_pid = np.rint(pid)
    if bool((np.abs(pid - nearest_pid) > 1.0e-6).any()) or bool(
        ((nearest_pid != 0) & (nearest_pid != 1)).any()
    ):
        raise ValueError("raw audit PID flags are not binary")
    hot_count = nearest_pid.sum(axis=1)
    if bool((hot_count > 1).any()):
        raise ValueError("raw audit contains multi-hot PID particles")
    charge = selected[:, 4]
    locked_charge = np.asarray([-1.0, 0.0, 1.0])
    charge_distance = np.min(
        np.abs(charge[:, None] - locked_charge[None, :]), axis=1
    )
    if bool((charge_distance > 1.0e-6).any()):
        raise ValueError("raw audit charge lies outside -1/0/+1")
    pt = selected[:, 0].astype(np.float64)
    eta = selected[:, 1].astype(np.float64)
    energy = selected[:, 3].astype(np.float64)
    momentum = pt * np.cosh(eta)
    if bool((pt < 0).any()) or bool((energy < 0).any()):
        raise ValueError("raw audit contains negative pT or energy")
    mass_squared = energy * energy - momentum * momentum
    tolerance = 1.0e-5 * np.maximum(energy * energy, 1.0)
    if bool((mass_squared < -tolerance).any()):
        raise ValueError("raw audit four-vectors are not reconstructable")
    track = selected[:, 10:14]
    if not np.isfinite(track).all():
        raise FloatingPointError("raw audit track fields are nonfinite")
    class_counts = np.bincount(truth, minlength=len(LABEL_NAMES))
    return {
        "split": str(split),
        "event_count": int(len(values)),
        "valid_particle_count": int(valid.sum()),
        "constituent_count_minimum": int(counts.min()),
        "constituent_count_maximum": int(counts.max()),
        "class_counts": {
            LABEL_NAMES[index]: int(class_counts[index])
            for index in range(len(LABEL_NAMES))
        },
        "pid_zero_hot_particle_count": int((hot_count == 0).sum()),
        "pid_multi_hot_particle_count": 0,
        "charge_domain_exact": True,
        "finite_valid_values": True,
        "padding_exact_zero": True,
        "four_vector_reconstructable": True,
        "track_numeric_sentinel_inferred": False,
    }


def build_preconstruction_audit(
    *,
    manifest: SplitManifest,
    raw_input_schema: Mapping[str, Any],
    branch_inventory: Sequence[Mapping[str, Any]],
    sampled_arrays: Mapping[str, Mapping[str, np.ndarray]],
    miniature: bool = False,
) -> dict[str, Any]:
    schema_sha = validate_raw_input_schema_contract(raw_input_schema)
    config = (
        RelationalSplitConfig.miniature()
        if miniature
        else RelationalSplitConfig.production()
    )
    manifest_audit = validate_relational_split_manifest(manifest, config=config)
    selection = select_raw_audit_identities(
        manifest, miniature=miniature
    )
    if len(branch_inventory) != len(manifest.file_records):
        raise ValueError("raw branch inventory does not cover every source file")
    required = set(raw_input_schema["required_particle_branches"])
    inventory_paths = set()
    expected_records = {
        str(record.path): record for record in manifest.file_records
    }
    for row in branch_inventory:
        path = str(row["path"])
        if path in inventory_paths:
            raise ValueError("raw branch inventory contains duplicate files")
        inventory_paths.add(path)
        record = expected_records.get(path)
        if record is None:
            raise ValueError(f"raw branch inventory contains unknown file {path}")
        if set(row.get("required_branches", ())) != required:
            raise ValueError(f"raw branch inventory differs for {path}")
        if int(row.get("num_entries", -1)) != int(record.num_entries):
            raise ValueError(
                f"raw branch inventory entry count differs for {path}"
            )
        if row.get("shape_policy") != "jagged_particle_axis":
            raise ValueError(f"raw branch shape policy differs for {path}")
        if row.get("dtype_policy") != "numeric":
            raise ValueError(f"raw branch dtype policy differs for {path}")
    if inventory_paths != set(expected_records):
        raise ValueError("raw branch inventory does not cover exact source paths")
    reports = {}
    for split in selection["selected_indices"]:
        arrays = sampled_arrays.get(split)
        if not isinstance(arrays, Mapping):
            raise ValueError(f"raw audit lacks sampled arrays for {split}")
        reports[split] = audit_raw_sample(
            arrays["tokens"],
            arrays["mask"],
            arrays["labels"],
            split=split,
        )
        if reports[split]["event_count"] != selection["selected_counts"][split]:
            raise ValueError(f"raw audit sample count differs for {split}")
    return with_content_hash(
        {
            "contract": PRECONSTRUCTION_AUDIT_CONTRACT,
            "schema_version": 1,
            "raw_input_schema_sha256": schema_sha,
            "source_manifest_sha256": manifest_audit["manifest_hash"],
            "campaign_profile": (
                "nonproduction_miniature_test"
                if miniature
                else "production_1m_125k_0_125k_500k"
            ),
            "manifest_audit": manifest_audit,
            "branch_inventory": list(branch_inventory),
            "sample_selection": selection,
            "sample_reports": reports,
            "final_test_access": "sealed_preparation_input_validation_only",
            "checkpoint_accessed": False,
            "inference_performed": False,
            "normalizer_fitted": False,
            "label_dependent_metric_computed": False,
            "event_level_output_persisted": False,
            "ok": True,
        }
    )


def build_postconstruction_audit(
    *,
    preconstruction_audit: Mapping[str, Any],
    hlt_binding: Mapping[str, Any],
    relation_normalization: Mapping[str, Any],
    region_normalization: Mapping[str, Any],
    backend_manifest: Mapping[str, Any],
    throughput_probe: Mapping[str, Any],
    tree_manifests: Mapping[str, Mapping[str, Any]],
    tree_identity_audits: Mapping[str, Mapping[str, Any]],
    storage_projection: Mapping[str, Any],
) -> dict[str, Any]:
    pre_sha = validate_content_hash(
        preconstruction_audit,
        expected_contract=PRECONSTRUCTION_AUDIT_CONTRACT,
    )
    hlt_sha = validate_content_hash(
        hlt_binding, expected_contract=RELATIONAL_HLT_BINDING_CONTRACT
    )
    if preconstruction_audit.get(
        "source_manifest_sha256"
    ) != hlt_binding.get("source_manifest_hash"):
        raise ValueError("raw audit and HLT binding use different manifests")
    relation_sha = validate_relation_normalization_artifact(
        relation_normalization
    )
    region_sha = validate_region_normalization(
        region_normalization,
        relation_normalization_sha256=relation_sha,
    )
    backend_sha = validate_content_hash(
        backend_manifest,
        expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    )
    probe_sha = validate_content_hash(
        throughput_probe, expected_contract=ANGULAR_TREE_PROBE_CONTRACT
    )
    probe_parents = throughput_probe.get("parents", {})
    model_train_hlt = hlt_binding["split_reports"]["model_train"][
        "hlt_content_hash"
    ]
    if (
        throughput_probe.get("scientific_provenance_complete") is not True
        or probe_parents.get("hlt_content_sha256") != model_train_hlt
        or probe_parents.get("backend_manifest_sha256") != backend_sha
        or throughput_probe.get("limits", {}).get("passed") is not True
        or (
            throughput_probe.get("storage_check", {}).get("passed") is not True
            and throughput_probe.get("operational_override") is None
        )
        or throughput_probe.get("parity", {}).get("topology_exact") is not True
        or float(
            throughput_probe.get("parity", {}).get(
                "max_continuous_absolute_error", float("inf")
            )
        )
        > float(
            throughput_probe.get("parity", {}).get(
                "continuous_absolute_tolerance", -1.0
            )
        )
    ):
        raise ValueError("tree probe did not pass its operational limits")
    if region_normalization.get(
        "angular_tree_resource_sha256"
    ) != probe_parents.get("tree_resource_sha256"):
        raise ValueError("REGION normalizer uses another tree resource")
    required_splits = {"model_train", "model_val", "stack_val", "final_test"}
    if set(tree_manifests) != required_splits:
        raise ValueError("postconstruction audit lacks a tree split")
    if set(tree_identity_audits) != required_splits:
        raise ValueError("postconstruction audit lacks a tree identity audit")
    tree_hashes = {}
    identity_reports = {}
    for split in sorted(required_splits):
        tree = tree_manifests[split]
        tree_hash = validate_content_hash(
            tree, expected_contract=ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT
        )
        hlt_report = hlt_binding["split_reports"][split]
        if (
            tree.get("split") != split
            or int(tree.get("jet_count", -1)) != int(hlt_report["n_jets"])
            or tree.get("parents", {}).get("hlt_content_sha256")
            != hlt_report["hlt_content_hash"]
            or tree.get("parents", {}).get("backend_manifest_sha256")
            != backend_sha
            or tree.get("parents", {}).get("tree_resource_sha256")
            != probe_parents.get("tree_resource_sha256")
        ):
            raise ValueError(f"{split} tree provenance differs from HLT/backend")
        identity_report = dict(tree_identity_audits[split])
        if (
            identity_report.get("split") != split
            or identity_report.get("ordered_identity_sha256")
            != hlt_report["jet_identity_hash"]
            or int(identity_report.get("jet_count", -1))
            != int(hlt_report["n_jets"])
            or identity_report.get("complete") is not True
            or identity_report.get("duplicate_identity_count") != 0
        ):
            raise ValueError(
                f"{split} tree identities differ from the authenticated HLT split"
            )
        tree_hashes[split] = tree_hash
        identity_reports[split] = identity_report
    storage_sha = validate_content_hash(storage_projection)
    if (
        storage_projection.get("ok") is not True
        or not all(storage_projection.get("checks", {}).values())
    ):
        raise ValueError("authenticated storage projection is unsafe")
    if throughput_probe.get("storage_projection_sha256") != storage_sha:
        raise ValueError("tree probe used another storage projection")
    if relation_normalization.get("hlt_binding_sha256") != hlt_sha:
        raise ValueError("relation normalizer belongs to another HLT binding")
    if relation_normalization.get(
        "source_manifest_sha256"
    ) != hlt_binding.get("source_manifest_hash"):
        raise ValueError("relation normalizer uses another split manifest")
    return with_content_hash(
        {
            "contract": POSTCONSTRUCTION_AUDIT_CONTRACT,
            "schema_version": 1,
            "preconstruction_audit_sha256": pre_sha,
            "hlt_binding_sha256": hlt_sha,
            "relation_normalization_sha256": relation_sha,
            "region_normalization_sha256": region_sha,
            "backend_manifest_sha256": backend_sha,
            "throughput_probe_sha256": probe_sha,
            "tree_manifest_sha256": tree_hashes,
            "tree_identity_audits": identity_reports,
            "storage_projection_sha256": storage_sha,
            "complete_identity_hash_coverage": True,
            "exact_class_balance": True,
            "tree_runtime_parity": throughput_probe["parity"],
            "hlt_only_inputs": True,
            "offline_arrays_persisted": False,
            "final_test_metrics_computed": False,
            "ok": True,
        }
    )


__all__ = [
    "POSTCONSTRUCTION_AUDIT_CONTRACT",
    "PRECONSTRUCTION_AUDIT_CONTRACT",
    "RAW_AUDIT_SALT",
    "audit_raw_sample",
    "build_postconstruction_audit",
    "build_preconstruction_audit",
    "select_raw_audit_identities",
]
