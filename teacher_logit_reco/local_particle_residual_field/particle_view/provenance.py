"""Acyclic coordinate/cache/consumer/logit lineage validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import (
    PARTICLE_VIEW_COORDINATE_BINDING_CONTRACT,
    require_sha256,
    validate_content_hash,
    validate_view_coordinate_binding,
    with_content_hash,
)
from .distillation import (
    PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT,
    PARTICLE_VIEW_TARGET_LOGIT_CACHE_CONTRACT,
)
from .view_cache import validate_selected_view_cache_manifest


PARTICLE_VIEW_LINEAGE_GRAPH_CONTRACT = "particle_view_lineage_graph_v1"

LINEAGE_KIND_ORDER = {
    "source_split": 0,
    "a0_offline_teacher": 1,
    "tap_spec": 2,
    "generator": 3,
    "normalizer": 4,
    "coordinate": 5,
    "selected_view_cache": 6,
    "consumer": 7,
    "pview0_or_robust_consumer": 8,
    "target_logits": 9,
    "final_predictor": 10,
    "deployment": 11,
}

_REQUIRED_PARENT_KINDS = {
    "source_split": set(),
    "a0_offline_teacher": {"source_split"},
    "tap_spec": {"a0_offline_teacher"},
    "generator": {"tap_spec"},
    "normalizer": {"generator"},
    "coordinate": {
        "source_split",
        "a0_offline_teacher",
        "tap_spec",
        "generator",
        "normalizer",
    },
    "selected_view_cache": {"source_split", "coordinate"},
    "consumer": {"coordinate", "selected_view_cache"},
    "pview0_or_robust_consumer": {"coordinate", "consumer"},
    "target_logits": {"coordinate", "selected_view_cache", "consumer"},
    "final_predictor": {"coordinate", "consumer", "target_logits"},
    "deployment": {"coordinate", "consumer", "final_predictor"},
}


@dataclass(frozen=True)
class ParticleViewLineageNode:
    node_id: str
    kind: str
    artifact_sha256: str
    parent_node_ids: tuple[str, ...] = ()
    artifact_contract: str | None = None

    def to_payload(self) -> dict[str, Any]:
        if not self.node_id:
            raise ValueError("lineage node_id must be non-empty")
        if self.kind not in LINEAGE_KIND_ORDER:
            raise ValueError(f"unknown lineage kind {self.kind!r}")
        require_sha256("artifact_sha256", self.artifact_sha256)
        parents = tuple(self.parent_node_ids)
        if len(parents) != len(set(parents)) or self.node_id in parents:
            raise ValueError("lineage parents must be unique and exclude self")
        if any(not parent for parent in parents):
            raise ValueError("lineage parent IDs must be non-empty")
        if self.artifact_contract is not None and not self.artifact_contract:
            raise ValueError("artifact_contract cannot be empty")
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "artifact_sha256": self.artifact_sha256,
            "parent_node_ids": sorted(parents),
            "artifact_contract": self.artifact_contract,
        }


def build_particle_view_lineage_graph(
    nodes: Sequence[ParticleViewLineageNode],
) -> dict[str, Any]:
    """Build a hash-bound DAG in the plan's strict ancestor direction."""

    serialized = [node.to_payload() for node in nodes]
    by_id = {row["node_id"]: row for row in serialized}
    if len(by_id) != len(serialized) or not serialized:
        raise ValueError("lineage node IDs must be unique and non-empty")
    artifact_hashes = [row["artifact_sha256"] for row in serialized]
    if len(set(artifact_hashes)) != len(artifact_hashes):
        raise ValueError("one artifact cannot occupy multiple lineage nodes")
    for row in serialized:
        parents = []
        for parent_id in row["parent_node_ids"]:
            if parent_id not in by_id:
                raise ValueError(f"unknown lineage parent {parent_id!r}")
            parent = by_id[parent_id]
            if LINEAGE_KIND_ORDER[parent["kind"]] >= LINEAGE_KIND_ORDER[row["kind"]]:
                raise ValueError("lineage edge is circular or points to a descendant")
            parents.append(parent["kind"])
        missing = _REQUIRED_PARENT_KINDS[row["kind"]] - set(parents)
        if missing:
            raise ValueError(
                f"{row['kind']} is missing parent kinds {sorted(missing)}"
            )
    ordered = sorted(
        serialized,
        key=lambda row: (LINEAGE_KIND_ORDER[row["kind"]], row["node_id"]),
    )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_LINEAGE_GRAPH_CONTRACT,
            "kind_order": [
                kind
                for kind, _ in sorted(
                    LINEAGE_KIND_ORDER.items(), key=lambda item: item[1]
                )
            ],
            "nodes": ordered,
            "node_count": len(ordered),
            "acyclic": True,
            "coordinate_excludes_descendants": True,
        }
    )


def validate_particle_view_lineage_graph(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_LINEAGE_GRAPH_CONTRACT
    )
    expected_order = [
        kind
        for kind, _ in sorted(LINEAGE_KIND_ORDER.items(), key=lambda item: item[1])
    ]
    if payload.get("kind_order") != expected_order:
        raise ValueError("lineage kind order changed")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("lineage graph nodes must be a list")
    rebuilt = build_particle_view_lineage_graph(
        [
            ParticleViewLineageNode(
                node_id=row["node_id"],
                kind=row["kind"],
                artifact_sha256=row["artifact_sha256"],
                parent_node_ids=tuple(row["parent_node_ids"]),
                artifact_contract=row["artifact_contract"],
            )
            for row in raw_nodes
        ]
    )
    if rebuilt != dict(payload):
        raise ValueError("lineage graph is not canonical")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "node_count": payload["node_count"],
    }


def _coordinate_reference(payload: Mapping[str, Any]) -> str | None:
    if "coordinate_binding_sha256" in payload:
        return payload.get("coordinate_binding_sha256")
    lineage = payload.get("lineage")
    if isinstance(lineage, Mapping):
        return lineage.get("coordinate_binding_sha256")
    return None


def validate_coordinate_cache_consumer_logit_chain(
    *,
    coordinate_binding: Mapping[str, Any],
    selected_view_cache_manifests: Sequence[Mapping[str, Any]],
    consumer_registration: Mapping[str, Any],
    consumer_checkpoint_sha256: str,
    target_logit_cache_manifests: Sequence[Mapping[str, Any]],
    predictor_registration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the concrete coordinate→cache→consumer→logit chain."""

    coordinate_sha = validate_view_coordinate_binding(coordinate_binding)
    if coordinate_binding.get("contract") != PARTICLE_VIEW_COORDINATE_BINDING_CONTRACT:
        raise ValueError("coordinate binding contract mismatch")
    require_sha256("consumer_checkpoint_sha256", consumer_checkpoint_sha256)
    cache_hashes: set[str] = set()
    ordered_identity_hashes: set[str] = set()
    for cache in selected_view_cache_manifests:
        validate_selected_view_cache_manifest(cache)
        if cache.get("coordinate_binding_sha256") != coordinate_sha:
            raise ValueError("selected-view cache uses a different coordinate")
        if cache.get("split") == "final_test":
            raise ValueError("selected-view cache cannot exist for final_test")
        cache_hashes.add(cache["content_hash"])
        ordered_identity_hashes.add(cache["ordered_identity_sha256"])
    if not cache_hashes:
        raise ValueError("lineage validation requires selected-view caches")

    validate_content_hash(consumer_registration)
    consumer_registration_sha = consumer_registration["content_hash"]
    if _coordinate_reference(consumer_registration) != coordinate_sha:
        raise ValueError("consumer registration uses a different coordinate")
    registered_checkpoint = consumer_registration.get("checkpoint_sha256")
    if registered_checkpoint != consumer_checkpoint_sha256:
        raise ValueError("consumer checkpoint hash differs from its registration")
    consumer_lineage = consumer_registration.get("lineage", {})
    if isinstance(consumer_lineage, Mapping):
        referenced_cache_hashes = {
            value
            for name, value in consumer_lineage.items()
            if "view_cache_manifest_sha256" in str(name)
        }
        if referenced_cache_hashes and not referenced_cache_hashes.issubset(
            cache_hashes
        ):
            raise ValueError("consumer registration references a stale view cache")

    logit_hashes: set[str] = set()
    for logits in target_logit_cache_manifests:
        validate_content_hash(
            logits, expected_contract=PARTICLE_VIEW_TARGET_LOGIT_CACHE_CONTRACT
        )
        lineage = logits.get("lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError("target-logit cache lineage is missing")
        expected = {
            "coordinate_binding_sha256": coordinate_sha,
            "consumer_registration_sha256": consumer_registration_sha,
            "consumer_checkpoint_sha256": consumer_checkpoint_sha256,
        }
        for field, value in expected.items():
            if lineage.get(field) != value:
                raise ValueError(f"target-logit cache {field} mismatch")
        if lineage.get("selected_view_cache_sha256") not in cache_hashes:
            raise ValueError("target-logit cache references a stale selected view")
        if lineage.get("split_identity_sha256") not in ordered_identity_hashes:
            raise ValueError("target-logit cache references an unknown split")
        logit_hashes.add(logits["content_hash"])
    if not logit_hashes:
        raise ValueError("lineage validation requires target-logit caches")

    predictor_sha = None
    if predictor_registration is not None:
        validate_content_hash(
            predictor_registration,
            expected_contract=PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT,
        )
        lineage = predictor_registration.get("lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError("predictor registration lineage is missing")
        if lineage.get("coordinate_binding_sha256") != coordinate_sha:
            raise ValueError("predictor registration uses a stale coordinate")
        if lineage.get("consumer_registration_sha256") not in {
            None,
            consumer_registration_sha,
        }:
            raise ValueError("predictor registration uses a stale consumer")
        if lineage.get("consumer_checkpoint_sha256") not in {
            None,
            consumer_checkpoint_sha256,
        }:
            raise ValueError("predictor registration uses a stale consumer checkpoint")
        referenced_logits = {
            value
            for name, value in lineage.items()
            if str(name).endswith("target_logit_cache_sha256")
        }
        if referenced_logits != logit_hashes:
            raise ValueError("predictor registration target-logit inventory differs")
        predictor_sha = predictor_registration["content_hash"]
    return with_content_hash(
        {
            "contract": "particle_view_concrete_lineage_audit_v1",
            "coordinate_binding_sha256": coordinate_sha,
            "selected_view_cache_sha256": sorted(cache_hashes),
            "consumer_registration_sha256": consumer_registration_sha,
            "consumer_checkpoint_sha256": consumer_checkpoint_sha256,
            "target_logit_cache_sha256": sorted(logit_hashes),
            "predictor_registration_sha256": predictor_sha,
            "final_test_cache_present": False,
            "acyclic": True,
            "authenticated": True,
        }
    )


__all__ = [
    "LINEAGE_KIND_ORDER",
    "PARTICLE_VIEW_LINEAGE_GRAPH_CONTRACT",
    "ParticleViewLineageNode",
    "build_particle_view_lineage_graph",
    "validate_coordinate_cache_consumer_logit_chain",
    "validate_particle_view_lineage_graph",
]
