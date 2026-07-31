"""Exact inherited-parent validation and deterministic rebuild planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    PARENT_REBUILD_PLAN_CONTRACT,
    PARENT_STATUS_CONTRACT,
    load_hashed_json,
    require_sha256,
    source_record,
    validate_content_hash,
    with_content_hash,
)


HLT_CACHE_SET_CONTRACT = "hosd_hlt_v3_cache_set_v1"


@dataclass(frozen=True)
class ParentRequirement:
    parent_id: str
    expected_contract: str
    required_before_stage: str
    canonical_path: str
    rebuild_entrypoint: str
    rebuild_group: str


PARENT_REQUIREMENTS: tuple[ParentRequirement, ...] = (
    ParentRequirement(
        "validation_partition_manifest",
        "retb_validation_partition_manifest_v1",
        "A",
        "inputs/validation_partition_manifest.json.gz",
        "scripts/build_hosd_campaign.py",
        "split",
    ),
    ParentRequirement(
        "scale_train_manifest",
        "retb_scale_train_manifest_v1",
        "A",
        "inputs/scale_train_manifest.json.gz",
        "scripts/build_hosd_campaign.py",
        "split",
    ),
    ParentRequirement(
        "final_select_label_manifest",
        "retb_final_select_label_manifest_v1",
        "A",
        "inputs/final_select_label_manifest.json.gz",
        "scripts/build_hosd_campaign.py",
        "split",
    ),
    ParentRequirement(
        "split_audit",
        "retb_split_audit_v1",
        "A",
        "inputs/split_audit.json",
        "scripts/build_hosd_campaign.py",
        "split",
    ),
    ParentRequirement(
        "raw_input_schema",
        "retb_raw_input_schema_v2",
        "A",
        "inputs/raw_input_schema.json",
        "scripts/build_hosd_campaign.py",
        "split",
    ),
    ParentRequirement(
        "hlt_replica_manifest",
        "retb_hlt_replica_manifest_v1",
        "A",
        "inputs/hlt_replica_manifest.json",
        "scripts/build_hosd_campaign.py",
        "split",
    ),
    ParentRequirement(
        "global_determinism",
        "retb_global_determinism_v1",
        "A",
        "registry/global_determinism.json",
        "scripts/build_hosd_campaign.py",
        "split",
    ),
    ParentRequirement(
        "hlt_v3_profile",
        "retb_hlt_v3_profile_v1",
        "A",
        "inputs/hlt_v3_profile.json",
        "scripts/build_hosd_shared_hlt_parents.py",
        "hlt",
    ),
    ParentRequirement(
        "hlt_v3_cache",
        HLT_CACHE_SET_CONTRACT,
        "A",
        "inputs/hlt_replicas/hlt_v3_cache_manifest.json",
        "scripts/build_hosd_shared_hlt_parents.py",
        "hlt",
    ),
    ParentRequirement(
        "hlt_v3_degradation_audit",
        "retb_hlt_v3_degradation_audit_v1",
        "A",
        "inputs/hlt_v3_degradation_audit.json",
        "scripts/build_hosd_shared_hlt_parents.py",
        "normalization",
    ),
    ParentRequirement(
        "angular_tree_backend",
        "relational_ca_tree_backend_manifest_v3",
        "A",
        "inputs/region_tree/backend_manifest.json",
        "scripts/build_hosd_tree_parents.py",
        "tree",
    ),
    ParentRequirement(
        "angular_tree_resource",
        "relational_part_angular_tree_resource_v3",
        "A",
        "inputs/inherited_angular_tree_resource.json",
        "scripts/build_hosd_tree_parents.py",
        "tree",
    ),
    ParentRequirement(
        "relation_500k_normalizer",
        "relational_part_relation_normalization_v2",
        "B",
        "inputs/normalization/offline_500k/relation.json",
        "scripts/fit_hosd_relation_normalizers.py",
        "normalization",
    ),
    ParentRequirement(
        "region_500k_normalizer",
        "relational_part_region_normalization_v1",
        "B",
        "inputs/normalization/offline_500k/region.json",
        "scripts/fit_hosd_relation_normalizers.py",
        "normalization",
    ),
    ParentRequirement(
        "hlt_shared_500k_normalizer",
        "relational_part_relation_normalization_v2",
        "B",
        "inputs/normalization/hlt_shared_500k/relation.json",
        "scripts/fit_hosd_relation_normalizers.py",
        "normalization",
    ),
    ParentRequirement(
        "hlt_shared_region_500k_normalizer",
        "relational_part_region_normalization_v1",
        "B",
        "inputs/normalization/hlt_shared_500k/region.json",
        "scripts/fit_hosd_relation_normalizers.py",
        "normalization",
    ),
)

_REBUILD_COMMANDS = {
    "hlt": [
        "scripts/build_retb_hlt_v3_cache.py",
        "scripts/audit_retb_hlt_v3.py",
    ],
    "tree": [
        "scripts/build_relational_part_tree_backend.py",
        "scripts/build_retb_region_tree_shard.py",
        "scripts/finalize_retb_region_tree_cache.py",
    ],
    "normalization": [
        "scripts/fit_relational_part_normalization.py",
        "scripts/fit_relational_part_region_normalization.py",
        "scripts/fit_retb_normalizers.py",
    ],
    "split": ["scripts/build_hosd_campaign.py"],
}


def _status_row(
    requirement: ParentRequirement,
    *,
    artifact: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    path: str | None,
) -> dict[str, Any]:
    if artifact is None:
        return {
            **asdict(requirement),
            "status": "rebuild_required_missing",
            "content_hash": None,
            "resolved_path": path,
            "reusable": False,
        }
    _validate_parent_semantics(requirement, artifact)
    if artifact.get("source") != source:
        return {
            **asdict(requirement),
            "status": "rebuild_required_source_drift",
            "content_hash": None,
            "resolved_path": path,
            "reusable": False,
        }
    return {
        **asdict(requirement),
        "status": "reusable_exact",
        "content_hash": str(artifact["content_hash"]),
        "resolved_path": path,
        "reusable": True,
    }


def _validate_parent_semantics(
    requirement: ParentRequirement,
    artifact: Mapping[str, Any],
) -> None:
    """Invoke the authoritative validator when the parent exposes one."""

    validate_content_hash(artifact, expected_contract=requirement.expected_contract)
    parent_id = requirement.parent_id
    if parent_id == "global_determinism":
        from teacher_logit_reco.relation_expert_token_bridge.determinism import (
            validate_global_determinism,
        )

        validate_global_determinism(artifact)
    elif parent_id == "hlt_replica_manifest":
        from teacher_logit_reco.relation_expert_token_bridge.replicas import (
            validate_hlt_replica_manifest,
        )

        validate_hlt_replica_manifest(artifact)
    elif parent_id == "hlt_v3_profile":
        from teacher_logit_reco.relation_expert_token_bridge.hlt_v3 import (
            validate_hlt_v3_profile_contract,
        )

        validate_hlt_v3_profile_contract(artifact)
    elif parent_id == "hlt_v3_degradation_audit":
        from teacher_logit_reco.relation_expert_token_bridge.hlt_audit import (
            validate_hlt_v3_degradation_audit,
        )

        validate_hlt_v3_degradation_audit(artifact)
    elif parent_id == "hlt_v3_cache":
        if artifact.get("contract") != HLT_CACHE_SET_CONTRACT:
            raise ValueError("HLT-v3 cache-set contract differs")
        rows = artifact.get("caches")
        if (
            not isinstance(rows, list)
            or not rows
            or int(artifact.get("cache_count", -1)) != len(rows)
        ):
            raise ValueError("HLT-v3 cache-set coverage differs")
        coordinates = set()
        from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
            HLT_V3_ARRAY_FILENAME,
            HLT_V3_METADATA_FILENAME,
        )

        for row in rows:
            coordinate = (
                str(row.get("logical_role")),
                int(row.get("replica_id", -1)),
                str(row.get("realization_policy")),
            )
            if coordinate in coordinates:
                raise ValueError("HLT-v3 cache-set coordinate is duplicated")
            coordinates.add(coordinate)
            path = Path(str(row.get("path", "")))
            if not path.is_dir() or path.is_symlink():
                raise FileNotFoundError(f"HLT-v3 cache is absent: {path}")
            array_path = path / HLT_V3_ARRAY_FILENAME
            metadata_path = path / HLT_V3_METADATA_FILENAME
            if (
                not array_path.is_file()
                or array_path.is_symlink()
                or not metadata_path.is_file()
                or metadata_path.is_symlink()
            ):
                raise FileNotFoundError(
                    f"HLT-v3 cache files are incomplete: {path}"
                )
            metadata = load_hashed_json(
                metadata_path, expected_contract="retb_hlt_v3_cache_v1"
            )
            if (
                metadata.get("content_hash") != row.get("metadata_sha256")
                or metadata.get("source") != artifact.get("source")
            ):
                raise ValueError("HLT-v3 cache-set nested lineage differs")
    elif parent_id == "angular_tree_backend":
        from teacher_logit_reco.relational_part.ca_tree import (
            validate_backend_manifest,
        )

        validate_backend_manifest(artifact)
    elif parent_id == "angular_tree_resource":
        from teacher_logit_reco.relational_part.ca_tree import (
            build_angular_tree_resource_contract,
        )

        split_hash = require_sha256(
            artifact.get("split_binding_sha256"),
            name="angular_tree_resource.split_binding_sha256",
        )
        actual = dict(artifact)
        actual.pop("content_hash", None)
        actual.pop("source", None)
        expected = build_angular_tree_resource_contract(
            split_binding_sha256=split_hash
        )
        expected.pop("content_hash")
        if actual != expected:
            raise ValueError("angular-tree resource semantics differ")
    elif parent_id in {
        "relation_500k_normalizer",
        "hlt_shared_500k_normalizer",
    }:
        from teacher_logit_reco.relational_part.normalization import (
            validate_relation_normalization_artifact,
        )

        validate_relation_normalization_artifact(artifact)
    elif parent_id in {
        "region_500k_normalizer",
        "hlt_shared_region_500k_normalizer",
    }:
        from teacher_logit_reco.relational_part.region_normalization import (
            validate_region_normalization,
        )

        validate_region_normalization(artifact)


def build_parent_status(
    *,
    source_snapshot: Mapping[str, Any],
    in_memory_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Inspect parents without treating path existence as compatibility."""

    source = source_record(source_snapshot)
    memory = dict(in_memory_artifacts or {})
    paths = {str(key): Path(value) for key, value in (artifact_paths or {}).items()}
    unknown = sorted((set(memory) | set(paths)) - {r.parent_id for r in PARENT_REQUIREMENTS})
    if unknown:
        raise ValueError(f"unknown inherited parent IDs: {unknown}")
    rows = []
    for requirement in PARENT_REQUIREMENTS:
        artifact = memory.get(requirement.parent_id)
        path = paths.get(requirement.parent_id)
        if artifact is not None and path is not None:
            raise ValueError(
                f"{requirement.parent_id} was supplied both in memory and by path"
            )
        if path is not None:
            if (
                requirement.parent_id == "hlt_v3_cache"
                and path.is_dir()
                and not path.is_symlink()
            ):
                from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
                    load_hlt_v3_cache,
                )

                _, artifact = load_hlt_v3_cache(path)
            elif path.is_file() and not path.is_symlink():
                artifact = load_hashed_json(
                    path, expected_contract=requirement.expected_contract
                )
            elif path.exists():
                raise ValueError(f"unsafe inherited parent path: {path}")
        rows.append(
            _status_row(
                requirement,
                artifact=artifact,
                source=source,
                path=None if path is None else str(path.resolve()),
            )
        )
    return with_content_hash(
        {
            "contract": PARENT_STATUS_CONTRACT,
            "schema_version": 1,
            "source": source,
            "requirements": rows,
            "all_stage_a_parents_reusable": all(
                row["reusable"] or row["required_before_stage"] != "A"
                for row in rows
            ),
            "all_stage_b_parents_reusable": all(row["reusable"] for row in rows),
            "path_existence_is_sufficient": False,
            "source_drift_reuse_allowed": False,
        }
    )


def build_parent_rebuild_plan(
    parent_status: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(parent_status, expected_contract=PARENT_STATUS_CONTRACT)
    unresolved = [
        row for row in parent_status["requirements"] if not bool(row["reusable"])
    ]
    groups = []
    for group in ("split", "hlt", "tree", "normalization"):
        parent_ids = sorted(
            row["parent_id"] for row in unresolved if row["rebuild_group"] == group
        )
        if not parent_ids:
            continue
        groups.append(
            {
                "group": group,
                "parent_ids": parent_ids,
                "wrapper_entrypoint": sorted(
                    {
                        row["rebuild_entrypoint"]
                        for row in unresolved
                        if row["rebuild_group"] == group
                    }
                ),
                "shared_producer_entrypoints": list(_REBUILD_COMMANDS[group]),
                "argv_template": [
                    "python",
                    (
                        "scripts/fit_hosd_relation_normalizers.py"
                        if group == "normalization"
                        else "scripts/build_hosd_shared_hlt_parents.py"
                        if group == "hlt"
                        else "scripts/build_hosd_tree_parents.py"
                        if group == "tree"
                        else "scripts/build_hosd_campaign.py"
                    ),
                    "--campaign-root",
                    "{campaign_root}",
                    "--submit",
                ],
            }
        )
    return with_content_hash(
        {
            "contract": PARENT_REBUILD_PLAN_CONTRACT,
            "schema_version": 1,
            "parent_status_sha256": require_sha256(
                parent_status["content_hash"], name="parent_status_sha256"
            ),
            "source": dict(parent_status["source"]),
            "groups": groups,
            "unresolved_parent_count": len(unresolved),
            "rebuild_uses_shared_contract_semantics": True,
            "rebuild_changes_parent_semantics": False,
            "scientific_results_consulted": False,
        }
    )


def require_parents_ready(
    parent_status: Mapping[str, Any],
    *,
    before_stage: str,
) -> None:
    validate_content_hash(parent_status, expected_contract=PARENT_STATUS_CONTRACT)
    if before_stage not in {"A", "B"}:
        raise ValueError("parent readiness stage must be A or B")
    allowed = {"A"} if before_stage == "A" else {"A", "B"}
    missing = [
        row["parent_id"]
        for row in parent_status["requirements"]
        if row["required_before_stage"] in allowed and not row["reusable"]
    ]
    if missing:
        raise RuntimeError(
            f"inherited parents are not ready before Stage {before_stage}: "
            f"{sorted(missing)}"
        )


__all__ = [
    "HLT_CACHE_SET_CONTRACT",
    "PARENT_REQUIREMENTS",
    "ParentRequirement",
    "build_parent_rebuild_plan",
    "build_parent_status",
    "require_parents_ready",
]
