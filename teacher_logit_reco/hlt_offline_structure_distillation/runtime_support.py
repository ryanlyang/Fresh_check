"""Deterministic, source-bound support files for HOSD execution.

These files are control-plane inputs derived exclusively from authenticated
campaign parents.  They are deliberately produced before the scientific DAG;
model outputs and other dependency-owned artifacts are never admitted here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from jetclass_fresh.jetclass_data import JetIdentity

from teacher_logit_reco.relational_part import (
    RELATION_FAMILY_REGISTRY_CONTRACT,
    bind_source_provenance,
    build_density_relation_contract,
    build_pair_base_contract,
    build_pid_charge_relation_contract,
    build_pt_relation_contract,
    build_region_relation_contract,
    build_rpt_base_model_contract,
    build_screening_registry,
    build_step5_model_contract,
    build_track_relation_contract,
    inspect_weaver_runtime,
)

from .contracts import (
    RUNTIME_LABEL_MANIFEST_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .input_views import load_materialized_input_view
from .normalization import build_hlt_conditional_context
from .target_cache import identity_order_sha256


RUNTIME_SUPPORT_CONTRACT = "hosd_runtime_support_manifest_v3"


def _source_snapshot(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_commit": source["commit"],
        "source_status_sha256": source["status_sha256"],
        "source_dirty": bool(source["dirty"]),
    }


def _bind(payload: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    return bind_source_provenance(
        payload, source_snapshot=_source_snapshot(source)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_plain_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    write_immutable_bytes(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"immutable runtime support is unsafe: {path}")
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(normalized) or any(
                not np.array_equal(existing[key], value)
                for key, value in normalized.items()
            ):
                raise FileExistsError(f"immutable runtime support differs: {path}")
    else:
        np.savez_compressed(path, **normalized)
    return _sha256(path)


def _shared_hlt_cache(root: Path, role: str, replica: int) -> Path:
    policy = "R_MULTI" if role in {"model_train", "scale_train"} else "R_FIXED"
    return (
        root
        / "inputs"
        / "shared_retb_parent_campaign"
        / "inputs"
        / "hlt_v3"
        / role
        / f"replica_{replica}"
        / policy
        / "D_NOMINAL"
    )


def _shared_tree(root: Path, view: str, role: str, replica: int | None = None) -> Path:
    base = (
        root
        / "inputs"
        / "shared_retb_parent_campaign"
        / "inputs"
        / "region_tree"
        / view
    )
    policy = "R_MULTI" if role in {"model_train", "scale_train"} else "R_FIXED"
    name = (
        f"{role}_exclusive_ca_v1"
        if replica is None
        else f"{role}_r{replica}_{policy}_exclusive_ca_v1"
    )
    return base / name


def _labels_npz(root: Path, role: str) -> Path:
    if role in {"design_select", "design_confirm"}:
        return root / "registry" / "runtime_support" / f"{role}_identity_labels.npz"
    return (
        root
        / "inputs"
        / "shared_retb_parent_campaign"
        / "inputs"
        / "offline"
        / role
        / "offline_inputs.npz"
    )


def _base_roles(root: Path, evaluation_role: str) -> dict[str, Any]:
    definitions: dict[str, Any] = {}
    for role, source_role, replicas in (
        ("model_train", "model_train", range(4)),
        ("val_stop", "val_stop", (0,)),
        (evaluation_role, evaluation_role, (0,)),
    ):
        definitions[role] = {
            "labels": str(_labels_npz(root, source_role).resolve()),
            "hlt_caches": {
                str(replica): str(
                    (
                        root / "inputs" / "hosd_views" / "hlt" / source_role
                        / f"replica_{replica}.npz"
                        if source_role in {"design_select", "design_confirm"}
                        else _shared_hlt_cache(root, source_role, replica)
                    ).resolve()
                )
                for replica in replicas
            },
            "tree_caches": {
                str(replica): str(
                    _shared_tree(
                        root,
                        "hlt",
                        "val_design" if source_role in {"design_select", "design_confirm"} else source_role,
                        replica,
                    ).resolve()
                )
                for replica in replicas
            },
        }
    return definitions


def _strict_edges(values: np.ndarray) -> list[float]:
    raw = np.quantile(
        np.asarray(values, dtype=np.float64),
        (0.0, 0.25, 0.5, 0.75, 1.0),
        method="linear",
    )
    unique = np.unique(raw)
    if len(unique) == 1:
        scale = max(abs(float(unique[0])), 1.0)
        unique = np.asarray(
            [float(unique[0]) - scale * 1.0e-6, float(unique[0]) + scale * 1.0e-6]
        )
    unique[0] = np.nextafter(unique[0], -np.inf)
    unique[-1] = np.nextafter(unique[-1], np.inf)
    return [float(value) for value in unique]


def _publish_registries_and_contracts(
    root: Path, campaign: Mapping[str, Any], support: Path
) -> dict[str, str]:
    source = campaign["source"]
    inherited_registry = load_hashed_json(
        root
        / "inputs"
        / "shared_retb_parent_campaign"
        / "registry"
        / "inherited_relation_family_registry.json",
        expected_contract=RELATION_FAMILY_REGISTRY_CONTRACT,
    )
    relation = load_hashed_json(
        root / "inputs" / "normalization" / "offline_500k" / "relation.json"
    )
    region = load_hashed_json(
        root / "inputs" / "normalization" / "offline_500k" / "region.json"
    )
    determinism = load_hashed_json(root / "registry" / "global_determinism.json")
    raw_schema = load_hashed_json(root / "inputs" / "raw_input_schema.json")
    tree_resource = load_hashed_json(
        root / "inputs" / "inherited_angular_tree_resource.json"
    )
    for name, artifact in (
        ("relation registry", inherited_registry),
        ("relation normalizer", relation),
        ("region normalizer", region),
        ("global determinism", determinism),
        ("raw input schema", raw_schema),
        ("tree resource", tree_resource),
    ):
        if artifact.get("source") != source:
            raise ValueError(f"{name} source differs from the HOSD campaign")

    screening = _bind(
        build_screening_registry(
            relation_registry_sha256=inherited_registry["content_hash"]
        ),
        source,
    )
    weaver = _bind(inspect_weaver_runtime(), source)
    pair = _bind(
        build_pair_base_contract(
            relation_registry_sha256=inherited_registry["content_hash"],
            global_determinism_sha256=determinism["content_hash"],
        ),
        source,
    )
    pt = _bind(
        build_pt_relation_contract(
            relation_registry_sha256=inherited_registry["content_hash"],
            relation_normalization_sha256=relation["content_hash"],
        ),
        source,
    )
    pid = _bind(
        build_pid_charge_relation_contract(
            relation_registry_sha256=inherited_registry["content_hash"],
            relation_normalization_sha256=relation["content_hash"],
        ),
        source,
    )
    track = _bind(
        build_track_relation_contract(
            relation_registry_sha256=inherited_registry["content_hash"],
            relation_normalization_sha256=relation["content_hash"],
            raw_input_schema_sha256=raw_schema["content_hash"],
        ),
        source,
    )
    density = _bind(
        build_density_relation_contract(
            relation_registry_sha256=inherited_registry["content_hash"],
            relation_normalization_sha256=relation["content_hash"],
            track_relation_sha256=track["content_hash"],
        ),
        source,
    )
    region_contract = _bind(
        build_region_relation_contract(
            relation_registry_sha256=inherited_registry["content_hash"],
            region_normalization_sha256=region["content_hash"],
            angular_tree_resource_sha256=tree_resource["content_hash"],
        ),
        source,
    )
    families = {
        "PT": pt["content_hash"],
        "PID": pid["content_hash"],
        "CHARGE": pid["content_hash"],
        "TRACK": track["content_hash"],
        "DENSITY": density["content_hash"],
        "REGION": region_contract["content_hash"],
    }
    base = _bind(
        build_rpt_base_model_contract(
            pair_base_sha256=pair["content_hash"],
            weaver_runtime_sha256=weaver["content_hash"],
            global_determinism_sha256=determinism["content_hash"],
        ),
        source,
    )
    full = _bind(
        build_step5_model_contract(
            "RPT_FULL_ALL",
            normalization_artifact=relation,
            screening_registry=screening,
            relation_registry_sha256=inherited_registry["content_hash"],
            pair_base_sha256=pair["content_hash"],
            family_contract_sha256=families,
            weaver_runtime_sha256=weaver["content_hash"],
            global_determinism_sha256=determinism["content_hash"],
            region_normalization_artifact=region,
        ),
        source,
    )
    artifacts = {
        "relation_family_registry": inherited_registry,
        "screening_registry": screening,
        "weaver_runtime": weaver,
        "pair_base": pair,
        "family_PT": pt,
        "family_PID_CHARGE": pid,
        "family_TRACK": track,
        "family_DENSITY": density,
        "family_REGION": region_contract,
        "model_contract_O_BASE": base,
        "model_contract_O_FULLREL": full,
    }
    paths: dict[str, str] = {}
    for name, artifact in artifacts.items():
        path = support / f"{name}.json"
        write_immutable_json(path, artifact)
        paths[name] = str(path.resolve())
    normalizer_path = support / "teacher_normalizer_hashes.json"
    _write_plain_json(
        normalizer_path,
        {
            "O_BASE": {"relation": relation["content_hash"]},
            "O_FULLREL": {
                "relation": relation["content_hash"],
                "region": region["content_hash"],
            },
        },
    )
    paths["teacher_normalizer_hashes"] = str(normalizer_path.resolve())
    return paths


def publish_runtime_support(
    *, campaign_root: str | Path, campaign: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish every static support artifact needed by the executable DAG."""

    root = Path(campaign_root).resolve()
    support = root / "registry" / "runtime_support"
    support.mkdir(parents=True, exist_ok=True)
    paths = _publish_registries_and_contracts(root, campaign, support)
    parity_path = support / "hosd_weaver_parity.json"
    parity = load_hashed_json(
        parity_path, expected_contract="hosd_weaver_split_forward_parity_v4"
    )
    if (
        parity.get("source") != campaign["source"]
        or parity.get("campaign_spec_sha256") != campaign["content_hash"]
        or parity.get("passed") is not True
    ):
        raise ValueError("authoritative HOSD Weaver parity lineage differs")
    paths["hosd_weaver_parity"] = str(parity_path.resolve())

    label_paths: dict[str, str] = {}
    design = load_hashed_json(
        root / "inputs" / "design_partition_manifest.json.gz",
        expected_contract="hosd_design_partition_manifest_v1",
    )
    if design.get("source") != campaign["source"]:
        raise ValueError("design partition source differs from campaign")
    val_design_path = _labels_npz(root, "val_design")
    with np.load(val_design_path, allow_pickle=False) as payload:
        val_ids = tuple(str(value) for value in payload["identities"])
        val_labels = np.asarray(payload["labels"], dtype=np.int64)
    val_lookup = {
        identity: int(label)
        for identity, label in zip(val_ids, val_labels, strict=True)
    }
    for role in ("design_select", "design_confirm"):
        identities = tuple(
            JetIdentity.from_dict(row).key() for row in design["roles"][role]
        )
        if not set(identities).issubset(val_lookup):
            raise ValueError(f"{role} identities are absent from val_design")
        _write_npz(
            _labels_npz(root, role),
            {
                "identities": np.asarray(identities, dtype="U"),
                "labels": np.asarray([val_lookup[value] for value in identities], dtype=np.int64),
            },
        )
        paths[f"{role}_identity_labels"] = str(_labels_npz(root, role).resolve())

    for role in ("model_train", "val_stop", "val_design", "design_select", "design_confirm"):
        path = _labels_npz(root, role)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"authenticated labels are absent: {path}")
        with np.load(path, allow_pickle=False) as payload:
            identities = tuple(str(value) for value in payload["identities"].tolist())
            labels = np.asarray(payload["labels"], dtype=np.int64)
        if labels.shape != (len(identities),) or len(set(identities)) != len(identities):
            raise ValueError(f"label population differs for {role}")
        artifact = with_content_hash(
            {
                "contract": RUNTIME_LABEL_MANIFEST_CONTRACT,
                "schema_version": 2,
                "source": dict(campaign["source"]),
                "campaign_spec_sha256": campaign["content_hash"],
                "split": role,
                "source_npz_sha256": _sha256(path),
                # Shuffle mappings are positional.  Keep the authenticated
                # NPZ order explicit because canonical JSON sorts mapping
                # keys and therefore cannot preserve identity_to_label order.
                "identity_order": list(identities),
                "identity_order_sha256": identity_order_sha256(identities),
                "identity_to_label": {
                    identity: int(label)
                    for identity, label in zip(identities, labels, strict=True)
                },
            }
        )
        output = support / "labels" / f"{role}.json"
        write_immutable_json(output, artifact)
        label_paths[role] = str(output.resolve())
    paths.update({f"label_manifest_{key}": value for key, value in label_paths.items()})

    for role in ("design_select", "design_confirm"):
        path = support / f"base_roles_{role}.json"
        _write_plain_json(path, _base_roles(root, role))
        paths[f"base_roles_{role}"] = str(path.resolve())

    for role in ("stack_val", "final_test"):
        source_path = _labels_npz(root, role)
        with np.load(source_path, allow_pickle=False) as payload:
            identities = np.asarray(payload["identities"])
            labels = np.asarray(payload["labels"], dtype=np.int64)
        if labels.shape != (len(identities),):
            raise ValueError(f"{role} identity/label population differs")
        labels_path = support / f"{role}_identity_labels.npz"
        _write_npz(
            labels_path,
            {"identities": identities, "labels": labels},
        )
        paths[f"{role}_identity_labels"] = str(labels_path.resolve())
        if role == "stack_val":
            identity_path = support / "stack_val_identities.npz"
            _write_npz(identity_path, {"identities": identities})
            paths["stack_val_identities"] = str(identity_path.resolve())

    hlt_view = root / "inputs" / "hosd_views" / "hlt" / "val_design" / "replica_0.npz"
    hlt_relation = load_hashed_json(
        root / "inputs" / "normalization" / "hlt_shared_500k" / "relation.json"
    )
    floors = hlt_relation.get("track_uncertainty_floors", {})
    if not {"d0", "dz"}.issubset(floors):
        raise ValueError("HLT relation normalizer lacks track uncertainty floors")
    arrays, _ = load_materialized_input_view(
        hlt_view,
        expected_view_kind="hlt_analogue",
        expected_source=campaign["source"],
    )
    confirm_path = _labels_npz(root, "design_confirm")
    with np.load(confirm_path, allow_pickle=False) as payload:
        confirm_ids = tuple(str(value) for value in payload["identities"])
    hlt_ids = tuple(str(value) for value in arrays["identities"])
    hlt_positions = {value: index for index, value in enumerate(hlt_ids)}
    if not set(confirm_ids).issubset(hlt_positions):
        raise ValueError("design-confirm covariate identities differ")
    confirm_indices = np.asarray(
        [hlt_positions[value] for value in confirm_ids], dtype=np.int64
    )
    context_rows = []
    for start in range(0, len(confirm_indices), 2048):
        stop = min(start + 2048, len(confirm_indices))
        indices = confirm_indices[start:stop]
        context_rows.append(
            build_hlt_conditional_context(
                arrays["tokens"][indices],
                arrays["mask"][indices],
                d0_uncertainty_floor=float(floors["d0"]["floor"]),
                dz_uncertainty_floor=float(floors["dz"]["floor"]),
                sentinel_policy=hlt_relation.get("track_sentinel_policy"),
            )
        )
    context = np.concatenate(context_rows, axis=0)
    covariates = {
        "identities": np.asarray(confirm_ids, dtype="U"),
        "jet_pt": np.exp(context[:, 0]),
        "abs_jet_eta": context[:, 1],
        "valid_multiplicity": context[:, 2],
        "valid_track_fraction": context[:, 3],
    }
    covariate_path = support / "robustness_covariates.npz"
    covariate_path.parent.mkdir(parents=True, exist_ok=True)
    if covariate_path.exists():
        with np.load(covariate_path, allow_pickle=False) as existing:
            if set(existing.files) != set(covariates) or any(
                not np.array_equal(existing[key], value)
                for key, value in covariates.items()
            ):
                raise FileExistsError("immutable robustness covariates differ")
    else:
        np.savez_compressed(covariate_path, **covariates)
    paths["robustness_covariates"] = str(covariate_path.resolve())
    edge_artifact = with_content_hash(
        {
            "contract": "hosd_robustness_subgroup_edges_v2",
            "schema_version": 2,
            "source": dict(campaign["source"]),
            "campaign_spec_sha256": campaign["content_hash"],
            "population": "design_confirm_hlt_nominal_replica_0_label_blind",
            "design_partition_sha256": design["content_hash"],
            "quantile_method": "numpy_linear",
            "probabilities": [0.0, 0.25, 0.5, 0.75, 1.0],
            "edges": {
                key: _strict_edges(value)
                for key, value in covariates.items()
                if key != "identities"
            },
        }
    )
    edge_path = support / "robustness_subgroup_edges.json"
    write_immutable_json(edge_path, edge_artifact)
    paths["robustness_subgroup_edges"] = str(edge_path.resolve())

    manifest = with_content_hash(
        {
            "contract": RUNTIME_SUPPORT_CONTRACT,
            "schema_version": 2,
            "source": dict(campaign["source"]),
            "campaign_spec_sha256": campaign["content_hash"],
            "paths": dict(sorted(paths.items())),
            "file_sha256": {
                key: _sha256(Path(value)) for key, value in sorted(paths.items())
            },
            "contains_model_outputs": False,
            "contains_performance_metrics": False,
        }
    )
    write_immutable_json(support / "manifest.json", manifest)
    return manifest


__all__ = ["RUNTIME_SUPPORT_CONTRACT", "publish_runtime_support"]
