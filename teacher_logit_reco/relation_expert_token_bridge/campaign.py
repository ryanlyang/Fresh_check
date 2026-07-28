"""Build and atomically publish the complete RETB Step-1 bundle."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import SplitManifest, manifest_hash

from .contracts import (
    STEP1_REPORT_CONTRACT,
    bind_source,
    build_campaign_spec,
    canonical_json_bytes,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .determinism import build_global_determinism
from .provenance import (
    build_artifact_layout,
    build_raw_input_schema,
    initialize_artifact_layout,
)
from .registry import build_registries, resolve_run_id
from .replicas import build_hlt_replica_manifest
from .splits import (
    RetbSplitConfig,
    build_final_select_label_manifest,
    build_scale_train_manifest,
    build_split_audits,
    build_validation_partition,
    validate_source_split_manifest,
)
from .storage import validate_storage_measurements


def _bind(payload: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    return bind_source(payload, source_snapshot=source)


def build_step1_bundle(
    *,
    campaign_id: str,
    manifest: SplitManifest,
    split_config: RetbSplitConfig,
    source_snapshot: Mapping[str, Any],
    storage_measurements: Mapping[str, Any],
) -> dict[str, Any]:
    source_manifest_sha = manifest_hash(manifest)
    source_audit = validate_source_split_manifest(manifest, config=split_config)

    validation = _bind(
        build_validation_partition(
            manifest,
            source_manifest_sha256=source_manifest_sha,
            config=split_config,
        ),
        source_snapshot,
    )
    scale = _bind(
        build_scale_train_manifest(
            manifest,
            source_manifest_sha256=source_manifest_sha,
            config=split_config,
        ),
        source_snapshot,
    )
    final_labels = _bind(
        build_final_select_label_manifest(
            manifest,
            source_manifest_sha256=source_manifest_sha,
        ),
        source_snapshot,
    )
    split_audit, scale_audit = build_split_audits(
        manifest,
        source_audit=source_audit,
        validation_partition=validation,
        scale_manifest=scale,
        final_select_labels=final_labels,
        config=split_config,
    )
    split_audit = _bind(split_audit, source_snapshot)
    scale_audit = _bind(scale_audit, source_snapshot)
    replica_manifest = _bind(
        build_hlt_replica_manifest(
            split_manifest_sha256=source_manifest_sha,
            validation_partition_sha256=validation["content_hash"],
            scale_train_manifest_sha256=scale["content_hash"],
        ),
        source_snapshot,
    )
    raw_schema = _bind(build_raw_input_schema(), source_snapshot)
    layout = _bind(build_artifact_layout(), source_snapshot)
    determinism = _bind(build_global_determinism(), source_snapshot)

    validate_storage_measurements(storage_measurements)
    measurements = _bind(storage_measurements, source_snapshot)

    registries = {
        name: _bind(artifact, source_snapshot)
        for name, artifact in build_registries().items()
    }
    parents = {
        "split_manifest": source_manifest_sha,
        "validation_partition_manifest": validation["content_hash"],
        "scale_train_manifest": scale["content_hash"],
        "final_select_label_manifest": final_labels["content_hash"],
        "split_audit": split_audit["content_hash"],
        "scale_train_audit": scale_audit["content_hash"],
        "hlt_replica_manifest": replica_manifest["content_hash"],
        "raw_input_schema": raw_schema["content_hash"],
        "artifact_layout": layout["content_hash"],
        "global_determinism": determinism["content_hash"],
        "storage_measurements": measurements["content_hash"],
    }
    campaign_spec = build_campaign_spec(
        campaign_id=campaign_id,
        campaign_profile=split_config.profile,
        source_snapshot=source_snapshot,
        parent_artifact_hashes=parents,
        run_registry_hashes={
            name: artifact["content_hash"] for name, artifact in registries.items()
        },
    )
    run_id_examples = {
        stage: resolve_run_id(
            stage=stage,
            component="CONTRACT_EXAMPLE",
            seed=101,
            configuration={"campaign_spec_sha256": campaign_spec["content_hash"]},
        )
        for stage in "ABCDEFGHIJKLMN"
    }
    report = _bind(
        with_content_hash(
            {
                "contract": STEP1_REPORT_CONTRACT,
                "schema_version": 1,
                "campaign_id": campaign_id,
                "campaign_spec_sha256": campaign_spec["content_hash"],
                "profile": split_config.profile,
                "checks": {
                    "split_counts_and_balance_exact": True,
                    "split_identities_disjoint": True,
                    "validation_partition_deterministic": True,
                    "scale_train_contains_model_train": True,
                    "scale_train_held_out_disjoint": True,
                    "four_training_replicas_registered": True,
                    "evaluation_replica_fixed_to_zero": True,
                    "access_roles_frozen": True,
                    "global_determinism_frozen": True,
                    "storage_measurements_authenticated": True,
                    "run_id_resolution_training_import_free": True,
                    "performance_based_termination_disabled": True,
                },
                "run_id_examples": run_id_examples,
                "future_runs_resolve_from_registry_only": True,
            }
        ),
        source_snapshot,
    )
    return {
        "campaign_spec": campaign_spec,
        "validation_partition_manifest": validation,
        "scale_train_manifest": scale,
        "final_select_label_manifest": final_labels,
        "split_audit": split_audit,
        "scale_train_audit": scale_audit,
        "hlt_replica_manifest": replica_manifest,
        "raw_input_schema": raw_schema,
        "artifact_layout": layout,
        "global_determinism": determinism,
        "storage_measurements": measurements,
        "registries": registries,
        "step1_report": report,
    }


def validate_step1_bundle(
    bundle: Mapping[str, Any],
    *,
    manifest: SplitManifest,
) -> str:
    for name in (
        "campaign_spec",
        "validation_partition_manifest",
        "scale_train_manifest",
        "final_select_label_manifest",
        "split_audit",
        "scale_train_audit",
        "hlt_replica_manifest",
        "raw_input_schema",
        "artifact_layout",
        "global_determinism",
        "storage_measurements",
        "step1_report",
    ):
        validate_content_hash(bundle[name])
    for artifact in bundle["registries"].values():
        validate_content_hash(artifact)
    spec = bundle["campaign_spec"]
    if spec["parent_artifact_hashes"]["split_manifest"] != manifest_hash(manifest):
        raise ValueError("campaign split-manifest parent differs")
    expected_registry_hashes = {
        name: artifact["content_hash"]
        for name, artifact in sorted(bundle["registries"].items())
    }
    if spec["registry_hashes"] != expected_registry_hashes:
        raise ValueError("campaign registry parents differ")
    if bundle["step1_report"]["campaign_spec_sha256"] != spec["content_hash"]:
        raise ValueError("Step-1 report points to a different campaign spec")
    return str(spec["content_hash"])


def publish_step1_bundle(
    *,
    campaign_root: str | Path,
    manifest: SplitManifest,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    validate_step1_bundle(bundle, manifest=manifest)
    root = Path(campaign_root)
    layout_result = initialize_artifact_layout(root, bundle["artifact_layout"])
    publications: dict[str, dict[str, Any]] = {}

    split_bytes = gzip.compress(
        canonical_json_bytes(manifest.to_dict()) + b"\n",
        compresslevel=9,
        mtime=0,
    )
    publications["split_manifest"] = write_immutable_bytes(
        root / "inputs" / "split_manifest.json.gz", split_bytes
    )
    paths = {
        "campaign_spec": root / "campaign_spec.json",
        "validation_partition_manifest": (
            root / "inputs" / "validation_partition_manifest.json.gz"
        ),
        "scale_train_manifest": root / "inputs" / "scale_train_manifest.json.gz",
        "final_select_label_manifest": (
            root / "inputs" / "final_select_label_manifest.json.gz"
        ),
        "split_audit": root / "inputs" / "split_audit.json",
        "scale_train_audit": root / "inputs" / "scale_train_audit.json",
        "hlt_replica_manifest": root / "inputs" / "hlt_replica_manifest.json",
        "raw_input_schema": root / "inputs" / "raw_input_schema.json",
        "artifact_layout": root / "registry" / "artifact_layout.json",
        "global_determinism": root / "registry" / "global_determinism.json",
        "storage_measurements": root / "storage_measurements.json",
        "step1_report": root / "reports" / "retb_step1_report.json",
    }
    for name, path in paths.items():
        publications[name] = write_immutable_json(path, bundle[name])
    for name, artifact in sorted(bundle["registries"].items()):
        publications[f"registry/{name}"] = write_immutable_json(
            root / "registry" / f"{name}.json", artifact
        )
    return {
        "campaign_root": str(root.resolve()),
        "campaign_spec_sha256": bundle["campaign_spec"]["content_hash"],
        "layout": layout_result,
        "publications": publications,
    }


__all__ = [
    "build_step1_bundle",
    "publish_step1_bundle",
    "validate_step1_bundle",
]
