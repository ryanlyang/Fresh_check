"""Build, validate, dry-run, and publish the HOSD Step-1 control plane."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import JetIdentity, SplitManifest, manifest_hash
from teacher_logit_reco.relation_expert_token_bridge import (
    RetbSplitConfig,
    build_step1_bundle as build_retb_step1_bundle,
    publish_step1_bundle as publish_retb_step1_bundle,
    validate_step1_bundle as validate_retb_step1_bundle,
)

from .contracts import (
    ARTIFACT_LAYOUT_CONTRACT,
    CAMPAIGN_SPEC_CONTRACT,
    DESIGN_PARTITION_CONTRACT,
    DRY_RUN_PLAN_CONTRACT,
    PARENT_REBUILD_PLAN_CONTRACT,
    PARENT_STATUS_CONTRACT,
    STEP1_REPORT_CONTRACT,
    canonical_json_bytes,
    require_safe_id,
    source_record,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .parents import build_parent_rebuild_plan, build_parent_status
from .registry import STAGE_ORDER, build_registries, validate_registry


REQUIRED_DIRECTORIES = (
    "inputs",
    "inputs/hlt_replicas",
    "inputs/region_tree",
    "inputs/shared_retb_parent_campaign",
    "capability",
    "registry",
    "teachers",
    "targets/canonical",
    "targets/hlt_analogues",
    "targets/residuals",
    "targets/controls",
    "normalization/relation_500k",
    "normalization/region_500k",
    "normalization/hlt_shared_500k",
    "normalization/target_500k",
    "normalization/residual_500k",
    "normalization/relation_scale",
    "normalization/region_scale",
    "normalization/hlt_shared_scale",
    "normalization/target_scale",
    "normalization/residual_scale",
    "baselines",
    "probes",
    "auxiliary",
    "feedback",
    "combinations",
    "mechanism_controls",
    "robustness",
    "confirmation_500k",
    "scale_up",
    "selection_predictions/stack_val",
    "postlock_oracle_diagnostics",
    "selection",
    "final_test",
    "reports",
    "job_ledgers",
)


def _identity_key(row: Mapping[str, Any]) -> str:
    return JetIdentity.from_dict(dict(row)).key()


def _partition_rank(row: Mapping[str, Any]) -> tuple[str, str]:
    identity = _identity_key(row)
    digest = hashlib.sha256(
        b"hosd_val_design_partition_v1" + identity.encode("utf-8")
    ).hexdigest()
    return digest, identity


def _identity_order_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"hosd_identity_order_v1\0")
    for row in rows:
        identity = JetIdentity.from_dict(row)
        digest.update(identity.key().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(identity.label)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_design_partition(
    validation_partition: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        validation_partition,
        expected_contract="retb_validation_partition_manifest_v1",
    )
    rows = list(validation_partition["roles"]["val_design"])
    by_label: dict[int, list[dict[str, Any]]] = {index: [] for index in range(10)}
    for raw in rows:
        row = dict(raw)
        identity = JetIdentity.from_dict(row)
        by_label[int(identity.label)].append(row)
    selected: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []
    per_class = None
    for label in range(10):
        ordered = sorted(by_label[label], key=_partition_rank)
        if len(ordered) < 2 or len(ordered) % 2:
            raise ValueError(
                "HOSD val_design must have a positive even count per class"
            )
        if per_class is None:
            per_class = len(ordered)
        elif len(ordered) != per_class:
            raise ValueError("HOSD val_design is not class balanced")
        midpoint = len(ordered) // 2
        selected.extend(ordered[:midpoint])
        confirmed.extend(ordered[midpoint:])
    return with_content_hash(
        {
            "contract": DESIGN_PARTITION_CONTRACT,
            "schema_version": 1,
            "retb_validation_partition_sha256": validation_partition[
                "content_hash"
            ],
            "partition_rule": (
                "per_class_sha256(hosd_val_design_partition_v1||identity),identity"
            ),
            "role_order": ["design_select", "design_confirm"],
            "roles": {
                "design_select": selected,
                "design_confirm": confirmed,
            },
            "counts": {
                "design_select": len(selected),
                "design_confirm": len(confirmed),
            },
            "per_class_counts": {
                "design_select": len(selected) // 10,
                "design_confirm": len(confirmed) // 10,
            },
            "identity_hashes": {
                "design_select": _identity_order_hash(selected),
                "design_confirm": _identity_order_hash(confirmed),
            },
            "selection_role": "architecture_target_loss_feedback_combination",
            "confirmation_role": "locked_mechanism_bundle_only",
            "source": dict(source),
        }
    )


def build_artifact_layout(*, source: Mapping[str, Any]) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": ARTIFACT_LAYOUT_CONTRACT,
            "schema_version": 1,
            "directories": list(REQUIRED_DIRECTORIES),
            "immutable_json_policy": "canonical_sha256_atomic_no_overwrite",
            "persistent_dense_pair_matrix_allowed": False,
            "source": dict(source),
        }
    )


def build_dry_run_plan(
    *,
    campaign_id: str,
    campaign_root: str | Path,
    stage_job_registry: Mapping[str, Any],
    producer_registry: Mapping[str, Any],
    parent_status: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_registry(stage_job_registry)
    validate_registry(producer_registry)
    jobs = []
    counts = Counter()
    for index, node in enumerate(stage_job_registry["nodes"]):
        counts[node["stage"]] += 1
        jobs.append(
            {
                "ordinal": index,
                "node_id": node["node_id"],
                "stage": node["stage"],
                "worker_role": node["worker_role"],
                "entrypoint": node["entrypoint"],
                "dependencies": list(node["dependencies"]),
                "expected_artifacts": list(node["outputs"]),
                "resource": node["resource"],
                "implementation_step": int(node["implementation_step"]),
                "implementation_status": (
                    "implemented_step1_control_plane"
                    if int(node["implementation_step"]) == 1
                    else "implemented_step2_capability_contract"
                    if int(node["implementation_step"]) == 2
                    and node["node_id"]
                    in {"capability_audit", "target_registry_compile"}
                    else f"implemented_step_{int(node['implementation_step'])}"
                    if int(node["implementation_step"])
                    in {3, 4, 5, 6, 7, 8, 9, 10, 11}
                    else f"reserved_for_step_{int(node['implementation_step'])}"
                ),
            }
        )
    return with_content_hash(
        {
            "contract": DRY_RUN_PLAN_CONTRACT,
            "schema_version": 1,
            "campaign_id": require_safe_id(campaign_id, name="campaign_id"),
            "campaign_root": str(Path(campaign_root).resolve()),
            "stage_order": list(STAGE_ORDER),
            "stage_job_counts": {
                stage: int(counts[stage]) for stage in STAGE_ORDER
            },
            "jobs": jobs,
            "job_count": len(jobs),
            "artifact_count": sum(len(job["expected_artifacts"]) for job in jobs),
            "producer_registry_sha256": producer_registry["content_hash"],
            "parent_status_sha256": parent_status["content_hash"],
            "matrix_bounds": {
                "stage_c_probe_rows_max": 162,
                "stage_d_rows_max": 213,
                "stage_e_rows_max": 46,
                "combination_beam_reduced_budget_fits_max": 96,
                "combination_full_fits_max": 10,
                "combination_pcgrad_controls_max": 1,
                "combination_total_executions_max": 11,
            },
            "execution_allowed": False,
            "scientific_outputs_created": False,
            "scientific_performance_used": False,
            "negative_result_can_prune_registered_job": False,
            "source": dict(source),
        }
    )


def _build_campaign_spec(
    *,
    campaign_id: str,
    campaign_profile: str,
    source: Mapping[str, Any],
    shared_parent_hashes: Mapping[str, str],
    registry_hashes: Mapping[str, str],
    design_partition: Mapping[str, Any],
    parent_status: Mapping[str, Any],
    parent_rebuild_plan: Mapping[str, Any],
    dry_run_plan: Mapping[str, Any],
    artifact_layout: Mapping[str, Any],
    split_role_counts: Mapping[str, int],
) -> dict[str, Any]:
    if campaign_profile not in {"production_500k_scale3m", "miniature_test"}:
        raise ValueError(f"unsupported HOSD campaign profile {campaign_profile!r}")
    return with_content_hash(
        {
            "contract": CAMPAIGN_SPEC_CONTRACT,
            "schema_version": 3,
            "campaign_id": require_safe_id(campaign_id, name="campaign_id"),
            "scientific_program": "hlt_offline_structure_distillation",
            "implemented_through_step": 12,
            "campaign_profile": campaign_profile,
            "campaign_stage": (
                "source_complete_real_tigris_miniature_acceptance_pending"
            ),
            # Bootstrap itself remains non-executing.  Step-5 workers obtain
            # role-scoped authorization only after all Stage-B parents and
            # their current-source lineage validate.
            "scientific_training_or_inference_authorized": False,
            "full_submission_requires_real_miniature_acceptance": True,
            "source": dict(source),
            "shared_parent_hashes": dict(sorted(shared_parent_hashes.items())),
            "registry_hashes": dict(sorted(registry_hashes.items())),
            "design_partition_sha256": design_partition["content_hash"],
            "inherited_parent_status_sha256": parent_status["content_hash"],
            "parent_rebuild_plan_sha256": parent_rebuild_plan["content_hash"],
            "dry_run_plan_sha256": dry_run_plan["content_hash"],
            "artifact_layout_sha256": artifact_layout["content_hash"],
            "split_roles": {
                str(role): int(count)
                for role, count in sorted(split_role_counts.items())
            },
            "seeds": {
                "discovery": 101,
                "confirmation": [202, 303, 404],
                "teacher": 101,
                "bootstrap": 917_301,
            },
            "access_policy": {
                "checkpoint_selection": "val_stop_only",
                "design_selection": "design_select_only",
                "mechanism_confirmation": "design_confirm_no_reselection",
                "finalist_selection": "final_select_stack_val_only",
                "final_test": "sealed_until_execution_lock",
                "performance_based_termination": False,
            },
            "continuation_source_snapshot_must_match_exactly": True,
            "missing_parent_requires_registered_rebuild": True,
            "source_drift_parent_reuse_allowed": False,
        }
    )


def build_step1_bundle(
    *,
    campaign_id: str,
    campaign_root: str | Path,
    manifest: SplitManifest,
    split_config: RetbSplitConfig,
    source_snapshot: Mapping[str, Any],
    storage_measurements: Mapping[str, Any],
    inherited_parent_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a non-executing HOSD Step-1 bundle entirely in memory."""

    source = source_record(source_snapshot)
    retb = build_retb_step1_bundle(
        campaign_id=f"{campaign_id}.shared_retb_parents",
        manifest=manifest,
        split_config=split_config,
        source_snapshot=source_snapshot,
        storage_measurements=storage_measurements,
    )
    shared = {
        "validation_partition_manifest": retb["validation_partition_manifest"],
        "scale_train_manifest": retb["scale_train_manifest"],
        "final_select_label_manifest": retb["final_select_label_manifest"],
        "split_audit": retb["split_audit"],
        "raw_input_schema": retb["raw_input_schema"],
        "hlt_replica_manifest": retb["hlt_replica_manifest"],
        "global_determinism": retb["global_determinism"],
    }
    parent_status = build_parent_status(
        source_snapshot=source_snapshot,
        in_memory_artifacts=shared,
        artifact_paths=inherited_parent_paths,
    )
    rebuild_plan = build_parent_rebuild_plan(parent_status)
    design = build_design_partition(
        retb["validation_partition_manifest"], source=source
    )
    registries = build_registries(source=source)
    layout = build_artifact_layout(source=source)
    dry_run = build_dry_run_plan(
        campaign_id=campaign_id,
        campaign_root=campaign_root,
        stage_job_registry=registries["stage_job_registry"],
        producer_registry=registries["producer_registry"],
        parent_status=parent_status,
        source=source,
    )
    shared_hashes = {
        "split_manifest": manifest_hash(manifest),
        "shared_retb_campaign_spec": retb["campaign_spec"]["content_hash"],
        **{name: artifact["content_hash"] for name, artifact in shared.items()},
        "storage_measurements": retb["storage_measurements"]["content_hash"],
    }
    spec = _build_campaign_spec(
        campaign_id=campaign_id,
        campaign_profile=split_config.profile,
        source=source,
        shared_parent_hashes=shared_hashes,
        registry_hashes={
            name: artifact["content_hash"] for name, artifact in registries.items()
        },
        design_partition=design,
        parent_status=parent_status,
        parent_rebuild_plan=rebuild_plan,
        dry_run_plan=dry_run,
        artifact_layout=layout,
        split_role_counts={
            "model_train": len(manifest.splits["model_train"]),
            "val_stop": int(
                retb["validation_partition_manifest"]["counts"]["val_stop"]
            ),
            "design_select": int(design["counts"]["design_select"]),
            "design_confirm": int(design["counts"]["design_confirm"]),
            "final_select": len(manifest.splits["stack_val"]),
            "final_test": len(manifest.splits["final_test"]),
            "scale_train": int(retb["scale_train_manifest"]["count"]),
        },
    )
    report = with_content_hash(
        {
            "contract": STEP1_REPORT_CONTRACT,
            "schema_version": 1,
            "campaign_id": campaign_id,
            "campaign_spec_sha256": spec["content_hash"],
            "implemented_step": 1,
            "checks": {
                "canonical_hashing": True,
                "source_bound": True,
                "retb_split_semantics_reused": True,
                "design_select_confirm_disjoint": True,
                "producer_registry_complete": True,
                "access_roles_fail_closed": True,
                "stage_a_to_k_enumerated": True,
                "parent_reuse_requires_exact_contract_and_source": True,
                "missing_parents_have_registered_rebuilds": True,
                "scientific_outputs_created": False,
                "performance_based_termination_disabled": True,
            },
            "stage_job_counts": dry_run["stage_job_counts"],
            "source": source,
        }
    )
    bundle = {
        "campaign_spec": spec,
        "artifact_layout": layout,
        "design_partition_manifest": design,
        "inherited_parent_status": parent_status,
        "parent_rebuild_plan": rebuild_plan,
        "dry_run_plan": dry_run,
        "registries": registries,
        "shared_parents": shared,
        "shared_retb_step1_bundle": retb,
        "storage_measurements": retb["storage_measurements"],
        "step1_report": report,
    }
    validate_step1_bundle(bundle, manifest=manifest)
    return bundle


def validate_step1_bundle(
    bundle: Mapping[str, Any],
    *,
    manifest: SplitManifest,
) -> str:
    contracts = {
        "campaign_spec": CAMPAIGN_SPEC_CONTRACT,
        "artifact_layout": ARTIFACT_LAYOUT_CONTRACT,
        "design_partition_manifest": DESIGN_PARTITION_CONTRACT,
        "inherited_parent_status": PARENT_STATUS_CONTRACT,
        "parent_rebuild_plan": PARENT_REBUILD_PLAN_CONTRACT,
        "dry_run_plan": DRY_RUN_PLAN_CONTRACT,
        "storage_measurements": None,
        "step1_report": STEP1_REPORT_CONTRACT,
    }
    for name, contract in contracts.items():
        validate_content_hash(bundle[name], expected_contract=contract)
        if bundle[name].get("source") != bundle["campaign_spec"]["source"]:
            raise ValueError(f"HOSD {name} source lineage differs")
    for artifact in bundle["shared_parents"].values():
        validate_content_hash(artifact)
    validate_retb_step1_bundle(
        bundle["shared_retb_step1_bundle"], manifest=manifest
    )
    if (
        bundle["campaign_spec"]["shared_parent_hashes"][
            "shared_retb_campaign_spec"
        ]
        != bundle["shared_retb_step1_bundle"]["campaign_spec"]["content_hash"]
    ):
        raise ValueError("shared RETB parent campaign lineage differs")
    for artifact in bundle["registries"].values():
        validate_registry(artifact)
        if artifact.get("source") != bundle["campaign_spec"]["source"]:
            raise ValueError("HOSD registry source lineage differs")
    spec = bundle["campaign_spec"]
    expected_registries = {
        name: artifact["content_hash"]
        for name, artifact in sorted(bundle["registries"].items())
    }
    if spec["registry_hashes"] != expected_registries:
        raise ValueError("HOSD campaign registry lineage differs")
    if spec["shared_parent_hashes"]["split_manifest"] != manifest_hash(manifest):
        raise ValueError("HOSD split-manifest lineage differs")
    links = {
        "design_partition_sha256": bundle["design_partition_manifest"][
            "content_hash"
        ],
        "inherited_parent_status_sha256": bundle["inherited_parent_status"][
            "content_hash"
        ],
        "parent_rebuild_plan_sha256": bundle["parent_rebuild_plan"][
            "content_hash"
        ],
        "dry_run_plan_sha256": bundle["dry_run_plan"]["content_hash"],
        "artifact_layout_sha256": bundle["artifact_layout"]["content_hash"],
    }
    for field, expected in links.items():
        if spec[field] != expected:
            raise ValueError(f"HOSD campaign {field} lineage differs")
    select = {
        _identity_key(row)
        for row in bundle["design_partition_manifest"]["roles"]["design_select"]
    }
    confirm = {
        _identity_key(row)
        for row in bundle["design_partition_manifest"]["roles"]["design_confirm"]
    }
    if select & confirm:
        raise ValueError("HOSD design roles overlap")
    inherited_design = {
        _identity_key(row)
        for row in bundle["shared_parents"]["validation_partition_manifest"][
            "roles"
        ]["val_design"]
    }
    if select | confirm != inherited_design:
        raise ValueError("HOSD design roles do not cover inherited val_design")
    if bundle["dry_run_plan"]["stage_order"] != list(STAGE_ORDER):
        raise ValueError("HOSD dry-run stage order differs")
    if bool(bundle["dry_run_plan"]["execution_allowed"]):
        raise ValueError("HOSD Step-1 dry run permits execution")
    if bundle["step1_report"]["campaign_spec_sha256"] != spec["content_hash"]:
        raise ValueError("HOSD Step-1 report points to another campaign")
    return str(spec["content_hash"])


def _initialize_layout(root: Path, layout: Mapping[str, Any]) -> list[str]:
    validate_content_hash(layout, expected_contract=ARTIFACT_LAYOUT_CONTRACT)
    if root.exists() and root.is_symlink():
        raise ValueError(f"campaign root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    created = []
    for relative in layout["directories"]:
        path = root / relative
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError(f"unsafe HOSD artifact directory: {path}")
        if not path.exists():
            path.mkdir(parents=True)
            created.append(relative)
    return created


def publish_step1_bundle(
    *,
    campaign_root: str | Path,
    manifest: SplitManifest,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    validate_step1_bundle(bundle, manifest=manifest)
    root = Path(campaign_root)
    created = _initialize_layout(root, bundle["artifact_layout"])
    publications: dict[str, Any] = {}
    shared_retb_root = root / "inputs" / "shared_retb_parent_campaign"
    publications["shared_retb_parent_campaign"] = publish_retb_step1_bundle(
        campaign_root=shared_retb_root,
        manifest=manifest,
        bundle=bundle["shared_retb_step1_bundle"],
    )
    encoded_manifest = gzip.compress(
        canonical_json_bytes(manifest.to_dict()) + b"\n",
        compresslevel=9,
        mtime=0,
    )
    publications["split_manifest"] = write_immutable_bytes(
        root / "inputs" / "split_manifest.json.gz", encoded_manifest
    )
    paths = {
        "campaign_spec": root / "campaign_spec.json",
        "artifact_layout": root / "registry" / "artifact_layout.json",
        "design_partition_manifest": (
            root / "inputs" / "design_partition_manifest.json.gz"
        ),
        "inherited_parent_status": root / "inputs" / "inherited_parent_status.json",
        "parent_rebuild_plan": root / "inputs" / "inherited_parent_rebuild_plan.json",
        "dry_run_plan": root / "job_ledgers" / "stage_a_to_k_dry_run_plan.json",
        "storage_measurements": root / "storage_measurements.json",
        "step1_report": root / "reports" / "hosd_step1_report.json",
    }
    for name, path in paths.items():
        publications[name] = write_immutable_json(path, bundle[name])
    parent_paths = {
        "validation_partition_manifest": (
            root / "inputs" / "validation_partition_manifest.json.gz"
        ),
        "scale_train_manifest": root / "inputs" / "scale_train_manifest.json.gz",
        "final_select_label_manifest": (
            root / "inputs" / "final_select_label_manifest.json.gz"
        ),
        "split_audit": root / "inputs" / "split_audit.json",
        "raw_input_schema": root / "inputs" / "raw_input_schema.json",
        "hlt_replica_manifest": root / "inputs" / "hlt_replica_manifest.json",
        "global_determinism": root / "registry" / "global_determinism.json",
    }
    for name, artifact in bundle["shared_parents"].items():
        publications[f"shared_parent/{name}"] = write_immutable_json(
            parent_paths[name], artifact
        )
    for name, artifact in sorted(bundle["registries"].items()):
        publications[f"registry/{name}"] = write_immutable_json(
            root / "registry" / f"{name}.json", artifact
        )
    return {
        "campaign_root": str(root.resolve()),
        "campaign_spec_sha256": bundle["campaign_spec"]["content_hash"],
        "created_directories": created,
        "publications": publications,
    }


__all__ = [
    "REQUIRED_DIRECTORIES",
    "build_artifact_layout",
    "build_design_partition",
    "build_dry_run_plan",
    "build_step1_bundle",
    "publish_step1_bundle",
    "validate_step1_bundle",
]
