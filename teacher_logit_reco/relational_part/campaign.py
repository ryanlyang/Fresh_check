"""Step-1 campaign bundle construction and immutable publication."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import SplitManifest, load_split_manifest, manifest_hash

from .ca_tree import build_angular_tree_resource_contract
from .contracts import (
    STEP1_REPORT_CONTRACT,
    bind_source_provenance,
    build_campaign_spec,
    canonical_json_bytes,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import (
    build_global_determinism_contract,
    validate_global_determinism_contract,
)
from .normalization import (
    build_normalization_contract,
    validate_normalization_contract,
)
from .provenance import (
    build_artifact_layout_contract,
    build_raw_input_schema_contract,
    initialize_artifact_layout,
    source_snapshot,
)
from .registry import (
    build_confirmation_architecture_registry,
    build_relation_family_registry,
    build_screening_registry,
    build_semantic_control_registry,
    validate_relation_family_registry,
)
from .splits import (
    RelationalSplitConfig,
    build_hlt_binding,
    build_hlt_expectation,
    build_split_binding,
)
from .storage import StorageMeasurements, build_storage_projection
from .storage import RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT


def _manifest_gzip_bytes(manifest: SplitManifest) -> bytes:
    raw = canonical_json_bytes(manifest.to_dict()) + b"\n"
    return gzip.compress(raw, compresslevel=9, mtime=0)


def _write_immutable_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"immutable destination is unsafe: {path}")
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace immutable artifact: {path}")
        return {"path": str(path.resolve()), "status": "already_present"}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"immutable destination appeared during publication: {path}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(path.resolve()), "status": "published"}


def build_step1_bundle(
    *,
    campaign_id: str,
    manifest: SplitManifest,
    measurements: StorageMeasurements | Mapping[str, Any],
    available_bytes: int,
    source_commit: str,
    source_status_sha256: str,
    source_dirty: bool = True,
    split_config: RelationalSplitConfig | None = None,
    hlt_cache_dir: str | Path | None = None,
    require_hlt_cache: bool = False,
    budget_gib: int = 20,
    minimum_free_reserve_gib: int = 20,
) -> dict[str, dict[str, Any]]:
    """Build the complete Step-1 object graph without writing any files."""

    production_profile = (
        split_config is None
        or (
            split_config.normalized_sizes()
            == RelationalSplitConfig.production().normalized_sizes()
            and split_config.normalized_seeds()
            == RelationalSplitConfig.production().normalized_seeds()
        )
    )
    if (
        not isinstance(measurements, Mapping)
        or measurements.get("contract") != RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT
    ):
        raise ValueError(
            "Step-1 campaign construction requires a source-bound "
            "relational_part_storage_measurements_v1 artifact"
        )
    if isinstance(measurements, Mapping):
        validate_content_hash(
            measurements,
            expected_contract=RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT,
        )
    split_config = split_config or RelationalSplitConfig.production()
    source = {
        "source_commit": source_commit,
        "source_status_sha256": source_status_sha256,
        "source_dirty": bool(source_dirty),
    }

    def bind(artifact: Mapping[str, Any]) -> dict[str, Any]:
        return bind_source_provenance(artifact, source_snapshot=source)

    measurements = bind(measurements)
    split_binding = bind(build_split_binding(manifest, config=split_config))
    hlt_expectation = bind(
        build_hlt_expectation(split_binding_sha256=split_binding["content_hash"])
    )
    hlt_binding = None
    if hlt_cache_dir is not None:
        hlt_binding = bind(
            build_hlt_binding(
                cache_dir=hlt_cache_dir,
                manifest=manifest,
                split_binding=split_binding,
                hlt_expectation=hlt_expectation,
            )
        )
    elif require_hlt_cache:
        raise ValueError("require_hlt_cache=True but no HLT cache directory was supplied")

    relation_registry = bind(build_relation_family_registry())
    screening_registry = bind(
        build_screening_registry(
            relation_registry_sha256=relation_registry["content_hash"]
        )
    )
    confirmation_registry = bind(
        build_confirmation_architecture_registry(
            relation_registry_sha256=relation_registry["content_hash"],
            screening_registry_sha256=screening_registry["content_hash"],
        )
    )
    semantic_registry = bind(
        build_semantic_control_registry(
            relation_registry_sha256=relation_registry["content_hash"],
            confirmation_registry_sha256=confirmation_registry["content_hash"],
        )
    )
    global_determinism = bind(build_global_determinism_contract())
    normalization = bind(
        build_normalization_contract(
            split_binding_sha256=split_binding["content_hash"]
        )
    )
    tree_resource = bind(
        build_angular_tree_resource_contract(
            split_binding_sha256=split_binding["content_hash"]
        )
    )
    layout = bind(build_artifact_layout_contract())
    raw_schema = bind(build_raw_input_schema_contract())
    storage = bind(
        build_storage_projection(
            measurements,
            available_bytes=available_bytes,
            budget_gib=budget_gib,
            minimum_free_reserve_gib=minimum_free_reserve_gib,
            total_hlt_jets=sum(split_config.normalized_sizes().values()),
            total_tree_jets=sum(split_config.normalized_sizes().values()),
            final_test_events=split_config.normalized_sizes()["final_test"],
        )
    )
    parents = {
        "split_binding_sha256": split_binding["content_hash"],
        "hlt_expectation_sha256": hlt_expectation["content_hash"],
        "hlt_binding_sha256": (
            None if hlt_binding is None else hlt_binding["content_hash"]
        ),
        "relation_family_registry_sha256": relation_registry["content_hash"],
        "screening_registry_sha256": screening_registry["content_hash"],
        "confirmation_architecture_registry_sha256": confirmation_registry[
            "content_hash"
        ],
        "semantic_control_registry_sha256": semantic_registry["content_hash"],
        "global_determinism_sha256": global_determinism["content_hash"],
        "normalization_contract_sha256": normalization["content_hash"],
        "angular_tree_resource_contract_sha256": tree_resource["content_hash"],
        "artifact_layout_sha256": layout["content_hash"],
        "raw_input_schema_sha256": raw_schema["content_hash"],
        "storage_projection_sha256": storage["content_hash"],
    }
    if (
        isinstance(measurements, Mapping)
        and measurements.get("contract") == RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT
    ):
        parents["storage_measurements_sha256"] = measurements["content_hash"]
    campaign_spec = build_campaign_spec(
        campaign_id=campaign_id,
        source_snapshot=source,
        artifact_hashes={
            key.removesuffix("_sha256"): value
            for key, value in parents.items()
            if value is not None
        },
        global_determinism=global_determinism,
        split_manifest_hash=split_binding["source_manifest_hash"],
        hlt_cache_status=(
            "authenticated" if hlt_binding is not None else "expected_not_built"
        ),
        campaign_profile=(
            "production_1m_125k_0_125k_500k"
            if production_profile
            else "nonproduction_miniature_test"
        ),
    )
    report = bind(with_content_hash(
        {
            "contract": STEP1_REPORT_CONTRACT,
            "schema_version": 3,
            "campaign_spec_sha256": campaign_spec["content_hash"],
            "campaign_id": campaign_id,
            "campaign_profile": (
                "production_1m_125k_0_125k_500k"
                if production_profile
                else "nonproduction_miniature_test"
            ),
            "split_manifest_ok": True,
            "split_counts": split_binding["split_sizes"],
            "screening_row_count": screening_registry["row_count"],
            "run_ids_training_independent": True,
            "storage_projection_ok": bool(storage["ok"]),
            "hlt_provenance_state": (
                "bound_and_validated"
                if hlt_binding is not None
                else "expected_contract_published_cache_not_yet_built"
            ),
            "hlt_cache_required_for_step1": bool(require_hlt_cache),
            "final_test_sealed": True,
            "global_determinism_fixed_before_results": True,
            "global_determinism_sha256": global_determinism["content_hash"],
            "offline_or_teacher_required_for_inference": False,
            "ready_for_step2": True,
            "scientific_results_allowed": bool(production_profile),
        }
    ))
    output = {
        "split_binding": split_binding,
        "hlt_expectation": hlt_expectation,
        "relation_family_registry": relation_registry,
        "screening_registry": screening_registry,
        "confirmation_architecture_registry": confirmation_registry,
        "semantic_control_registry": semantic_registry,
        "global_determinism": global_determinism,
        "normalization_contract": normalization,
        "angular_tree_resource_contract": tree_resource,
        "artifact_layout": layout,
        "raw_input_schema": raw_schema,
        "storage_projection": storage,
        "campaign_spec": campaign_spec,
        "step1_report": report,
    }
    if (
        isinstance(measurements, Mapping)
        and measurements.get("contract") == RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT
    ):
        output["storage_measurements"] = dict(measurements)
    if hlt_binding is not None:
        output["hlt_binding"] = hlt_binding
    validate_step1_bundle(manifest=manifest, bundle=output)
    return output


_ARTIFACT_PATHS = {
    "split_binding": "inputs/split_audit.json",
    "hlt_expectation": "inputs/hlt_expectation.json",
    "hlt_binding": "inputs/hlt_cache_audit.json",
    "relation_family_registry": "registry/relation_family_registry.json",
    "screening_registry": "registry/screening_registry.json",
    "confirmation_architecture_registry": (
        "registry/confirmation_architecture_registry.json"
    ),
    "semantic_control_registry": "registry/semantic_control_registry.json",
    "global_determinism": "registry/global_determinism.json",
    "normalization_contract": "inputs/normalization_contract.json",
    "angular_tree_resource_contract": "inputs/angular_tree_resource_contract.json",
    "artifact_layout": "registry/artifact_layout.json",
    "raw_input_schema": "inputs/raw_input_schema.json",
    "storage_projection": "storage_projection.json",
    "storage_measurements": "storage_measurements.json",
    "campaign_spec": "campaign_spec.json",
    "step1_report": "reports/relational_part_step1_report.json",
}


def validate_step1_bundle(
    *,
    manifest: SplitManifest,
    bundle: Mapping[str, Mapping[str, Any]],
) -> str:
    required = {
        "split_binding",
        "hlt_expectation",
        "relation_family_registry",
        "screening_registry",
        "confirmation_architecture_registry",
        "semantic_control_registry",
        "global_determinism",
        "normalization_contract",
        "angular_tree_resource_contract",
        "artifact_layout",
        "raw_input_schema",
        "storage_measurements",
        "storage_projection",
        "campaign_spec",
        "step1_report",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Step-1 bundle is incomplete: {missing}")
    hashes = {
        name: validate_content_hash(artifact)
        for name, artifact in bundle.items()
    }
    validate_global_determinism_contract(bundle["global_determinism"])
    validate_relation_family_registry(bundle["relation_family_registry"])
    validate_normalization_contract(bundle["normalization_contract"])
    spec = bundle["campaign_spec"]
    expected_parents = {
        "split_binding": hashes["split_binding"],
        "hlt_expectation": hashes["hlt_expectation"],
        "relation_family_registry": hashes["relation_family_registry"],
        "screening_registry": hashes["screening_registry"],
        "confirmation_architecture_registry": hashes[
            "confirmation_architecture_registry"
        ],
        "semantic_control_registry": hashes["semantic_control_registry"],
        "global_determinism": hashes["global_determinism"],
        "normalization_contract": hashes["normalization_contract"],
        "angular_tree_resource_contract": hashes[
            "angular_tree_resource_contract"
        ],
        "artifact_layout": hashes["artifact_layout"],
        "raw_input_schema": hashes["raw_input_schema"],
        "storage_measurements": hashes["storage_measurements"],
        "storage_projection": hashes["storage_projection"],
    }
    if "hlt_binding" in bundle:
        expected_parents["hlt_binding"] = hashes["hlt_binding"]
    if spec.get("parent_artifact_hashes") != dict(sorted(expected_parents.items())):
        raise ValueError("campaign spec parent hashes do not match the Step-1 bundle")
    if spec.get("split_manifest_hash") != manifest_hash(manifest):
        raise ValueError("campaign spec belongs to another split manifest")

    parent_checks = [
        (
            bundle["hlt_expectation"].get("split_binding_sha256"),
            hashes["split_binding"],
            "HLT expectation split parent",
        ),
        (
            bundle["screening_registry"].get("relation_registry_sha256"),
            hashes["relation_family_registry"],
            "screening relation parent",
        ),
        (
            bundle["confirmation_architecture_registry"].get(
                "screening_registry_sha256"
            ),
            hashes["screening_registry"],
            "confirmation screening parent",
        ),
        (
            bundle["confirmation_architecture_registry"].get(
                "relation_registry_sha256"
            ),
            hashes["relation_family_registry"],
            "confirmation relation parent",
        ),
        (
            bundle["semantic_control_registry"].get(
                "confirmation_registry_sha256"
            ),
            hashes["confirmation_architecture_registry"],
            "semantic confirmation parent",
        ),
        (
            bundle["semantic_control_registry"].get("relation_registry_sha256"),
            hashes["relation_family_registry"],
            "semantic relation parent",
        ),
        (
            bundle["campaign_spec"]
            .get("global_determinism", {})
            .get("content_hash"),
            hashes["global_determinism"],
            "campaign global-determinism parent",
        ),
        (
            bundle["normalization_contract"].get("split_binding_sha256"),
            hashes["split_binding"],
            "normalization split parent",
        ),
        (
            bundle["angular_tree_resource_contract"].get("split_binding_sha256"),
            hashes["split_binding"],
            "tree split parent",
        ),
        (
            bundle["storage_projection"].get("measurement_artifact_sha256"),
            hashes["storage_measurements"],
            "storage measurement parent",
        ),
        (
            bundle["step1_report"].get("campaign_spec_sha256"),
            hashes["campaign_spec"],
            "report campaign parent",
        ),
        (
            bundle["step1_report"].get("global_determinism_sha256"),
            hashes["global_determinism"],
            "report global-determinism parent",
        ),
    ]
    for actual, expected, label in parent_checks:
        if actual != expected:
            raise ValueError(f"{label} mismatch")

    source = spec.get("source")
    for name, artifact in bundle.items():
        if artifact.get("source") != source:
            raise ValueError(f"{name} source snapshot differs from campaign spec")
    return hashes["campaign_spec"]


def publish_step1_bundle(
    *,
    campaign_root: str | Path,
    manifest: SplitManifest,
    bundle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validate_step1_bundle(manifest=manifest, bundle=bundle)
    root = Path(campaign_root)
    layout_result = initialize_artifact_layout(root, bundle["artifact_layout"])
    publications: dict[str, Any] = {}
    for name, relative in _ARTIFACT_PATHS.items():
        if name in bundle:
            publications[name] = write_immutable_json(root / relative, bundle[name])
    manifest_result = _write_immutable_bytes(
        root / "inputs" / "split_manifest.json.gz",
        _manifest_gzip_bytes(manifest),
    )
    return {
        "campaign_root": str(root.resolve()),
        "campaign_spec_sha256": bundle["campaign_spec"]["content_hash"],
        "layout": layout_result,
        "manifest": {
            **manifest_result,
            "sha256": sha256_file(root / "inputs" / "split_manifest.json.gz"),
        },
        "publications": publications,
    }


def build_and_publish_step1(
    *,
    parent_manifest_path: str | Path,
    campaign_root: str | Path,
    campaign_id: str,
    measurements: StorageMeasurements | Mapping[str, Any],
    available_bytes: int,
    repo_root: str | Path,
    split_config: RelationalSplitConfig | None = None,
    hlt_cache_dir: str | Path | None = None,
    require_hlt_cache: bool = False,
    budget_gib: int = 20,
    minimum_free_reserve_gib: int = 20,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = load_split_manifest(parent_manifest_path)
    source = source_snapshot(repo_root)
    bundle = build_step1_bundle(
        campaign_id=campaign_id,
        manifest=manifest,
        measurements=measurements,
        available_bytes=available_bytes,
        source_commit=source["source_commit"],
        source_status_sha256=source["source_status_sha256"],
        source_dirty=bool(source["source_dirty"]),
        split_config=split_config,
        hlt_cache_dir=hlt_cache_dir,
        require_hlt_cache=require_hlt_cache,
        budget_gib=budget_gib,
        minimum_free_reserve_gib=minimum_free_reserve_gib,
    )
    if dry_run:
        return {
            "dry_run": True,
            "campaign_root": str(Path(campaign_root).resolve()),
            "campaign_spec": bundle["campaign_spec"],
            "step1_report": bundle["step1_report"],
        }
    return publish_step1_bundle(
        campaign_root=campaign_root,
        manifest=manifest,
        bundle=bundle,
    )


__all__ = [
    "build_and_publish_step1",
    "build_step1_bundle",
    "publish_step1_bundle",
    "validate_step1_bundle",
]
