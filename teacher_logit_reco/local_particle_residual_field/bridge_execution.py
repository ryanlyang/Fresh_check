"""Immutable source and runtime binding for numerical bridge execution.

The Step 10 graph deliberately contains no dataset paths.  This contract is
the missing site-local binding: it resolves the four privileged development
parents, the HLT baseline checkpoint, and the exact Step 1 split lineage once,
then lets every numerical executor fail closed on the same hashes.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_cache import jet_identity_hash
from jetclass_fresh.jetclass_data import load_split_manifest, manifest_hash

from .bridge_contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .bridge import validate_bridge_recipe
from .bridge_consumer import ConsumerCampaignConfig
from .bridge_r0 import StreamedR0TrainConfig
from .bridge_splits import (
    PREDICTION_ANCHORED_SPLIT_CONTRACT,
    audit_child_split_manifest,
    prediction_anchored_split_config_from_payload,
)
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, local_particle_residual_field_layout


PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT = "prediction_anchored_execution_spec_v1"
EXECUTION_PARENT_SPLITS = ("model_train", "model_val", "stack_train", "stack_val")


def default_bridge_schema_hashes() -> dict[str, str]:
    names, groups, _ = local_particle_residual_field_layout(DEFAULT_LOCAL_RESIDUAL_RADII)
    return {
        "target_schema_sha256": canonical_sha256(
            {
                "radii": list(DEFAULT_LOCAL_RESIDUAL_RADII),
                "field_names": names,
                "field_groups": groups,
            }
        ),
        "preprocessing_sha256": canonical_sha256(
            {
                "source": "fixed_hlt_raw_tokens",
                "dtype": "float32",
                "mask": "bool",
                "units": "repository_native",
            }
        ),
    }


def _regular_file(path: str | Path, *, label: str) -> Path:
    value = Path(path).resolve()
    if value.is_symlink() or not value.is_file():
        raise FileNotFoundError(f"{label} is absent or unsafe: {value}")
    return value


def _metadata(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} metadata must be a JSON object")
    return payload


def _source_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _source_record(
    *,
    split: str,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    expected_count: int,
    expected_event_order_sha256: str,
    parent_order_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    hlt_root = Path(hlt_cache_dir)
    offline_root = Path(offline_cache_dir)
    files = {
        "hlt_npz": _regular_file(
            hlt_root / f"{split}_fixed_hlt.npz", label=f"{split} HLT NPZ"
        ),
        "hlt_metadata": _regular_file(
            hlt_root / f"{split}_fixed_hlt_metadata.json",
            label=f"{split} HLT metadata",
        ),
        "offline_npz": _regular_file(
            offline_root / f"{split}_offline.npz", label=f"{split} offline NPZ"
        ),
        "offline_metadata": _regular_file(
            offline_root / f"{split}_offline_metadata.json",
            label=f"{split} offline metadata",
        ),
    }
    hlt_metadata = _metadata(files["hlt_metadata"], label=f"{split} HLT")
    offline_metadata = _metadata(files["offline_metadata"], label=f"{split} offline")
    for name, metadata in (("hlt", hlt_metadata), ("offline", offline_metadata)):
        count = metadata.get("n_jets", metadata.get("n_events"))
        if count is not None and int(count) != int(expected_count):
            raise ValueError(f"{split} {name} event count does not match parent manifest")
        if metadata.get("jet_identity_hash") != expected_event_order_sha256:
            raise ValueError(f"{split} {name} event order does not match split manifest")
        declared_parent = metadata.get("source_manifest_hash")
        if declared_parent not in (None, expected_manifest_sha256):
            raise ValueError(f"{split} {name} source manifest hash is stale")
    if hlt_metadata["jet_identity_hash"] != offline_metadata["jet_identity_hash"]:
        raise ValueError(f"{split} HLT/offline metadata are not aligned")
    return {
        "split": split,
        "n_events": int(expected_count),
        "event_order_sha256": expected_event_order_sha256,
        "parent_order_sha256": parent_order_sha256,
        "hlt_npz": _source_file_record(files["hlt_npz"]),
        "hlt_metadata": _source_file_record(files["hlt_metadata"]),
        "offline_npz": _source_file_record(files["offline_npz"]),
        "offline_metadata": _source_file_record(files["offline_metadata"]),
        "hlt_content_hash": hlt_metadata.get("hlt_content_hash"),
        "offline_content_hash": offline_metadata.get("offline_content_hash")
        or offline_metadata.get("content_hash"),
    }


def build_prediction_anchored_execution_spec(
    *,
    parent_manifest_path: str | Path,
    child_manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    baseline_checkpoint_path: str | Path,
    r0_config: StreamedR0TrainConfig | None = None,
    consumer_config: ConsumerCampaignConfig | None = None,
    preprocessing_sha256: str | None = None,
    target_schema_sha256: str | None = None,
    parent_splits: Sequence[str] = EXECUTION_PARENT_SPLITS,
) -> dict[str, Any]:
    parent_path = _regular_file(parent_manifest_path, label="parent split manifest")
    child_path = _regular_file(child_manifest_path, label="child split manifest")
    baseline_path = _regular_file(baseline_checkpoint_path, label="HLT baseline checkpoint")
    parent = load_split_manifest(parent_path)
    parent_sha256 = manifest_hash(parent)
    child = load_hashed_json(
        child_path, expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    split_config = prediction_anchored_split_config_from_payload(child["split_config"])
    audit_child_split_manifest(child, parent=parent, config=split_config)
    requested = tuple(str(value) for value in parent_splits)
    if len(set(requested)) != len(requested) or set(requested) != set(EXECUTION_PARENT_SPLITS):
        raise ValueError(
            f"execution spec must bind exactly the development parents {EXECUTION_PARENT_SPLITS}"
        )
    sources = {}
    for split in EXECUTION_PARENT_SPLITS:
        sources[split] = _source_record(
            split=split,
            hlt_cache_dir=hlt_cache_dir,
            offline_cache_dir=offline_cache_dir,
            expected_count=len(parent.splits[split]),
            expected_event_order_sha256=jet_identity_hash(parent.splits[split]),
            parent_order_sha256=child["parent_split_order_sha256"][split],
            expected_manifest_sha256=parent_sha256,
        )
    defaults = default_bridge_schema_hashes()
    r0 = r0_config or StreamedR0TrainConfig(output_dir="__RUNTIME_OUTPUT_DIR__")
    r0_payload = asdict(r0)
    r0_payload.pop("output_dir", None)
    consumers = consumer_config or ConsumerCampaignConfig(
        baseline_steps=10_000,
        bridge_finetune_steps=2_000,
        batch_size=128,
        evaluation_interval_steps=200,
    )
    payload = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
            "parent_manifest": {
                **_source_file_record(parent_path),
                "manifest_sha256": parent_sha256,
            },
            "child_manifest": {
                **_source_file_record(child_path),
                "content_hash": child["content_hash"],
                "split_config_sha256": child["split_config_sha256"],
            },
            "sources": sources,
            "baseline_checkpoint": _source_file_record(baseline_path),
            "class_names": list(parent.class_names),
            "max_constits": int(parent.max_constits),
            "preprocessing_sha256": preprocessing_sha256
            or defaults["preprocessing_sha256"],
            "target_schema_sha256": target_schema_sha256
            or defaults["target_schema_sha256"],
            "r0_training": r0_payload,
            "consumer_training": asdict(consumers),
            "runtime": {
                "single_node_required": True,
                "one_open_per_compressed_source_per_allocation": True,
                "raw_sources_non_evictable": True,
                "derived_only_lru": True,
                "mandatory_ram_headroom_fraction": 0.20,
                "restart_scope": "whole_configuration",
                "persistent_dense_fields_allowed": False,
            },
            "final_test_policy": {
                "hlt_only": True,
                "offline_source_bound": False,
                "oracle_diagnostics_allowed": False,
            },
        }
    )
    validate_prediction_anchored_execution_spec(payload, verify_file_hashes=True)
    return payload


def validate_prediction_anchored_execution_spec(
    payload: Mapping[str, Any],
    *,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    if tuple(payload.get("sources", {})) != EXECUTION_PARENT_SPLITS:
        raise ValueError("execution spec has missing, extra, or reordered parent sources")
    for name in ("preprocessing_sha256", "target_schema_sha256"):
        value = payload.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"execution spec {name} is not a SHA-256 digest")
    runtime = payload.get("runtime", {})
    required_runtime = {
        "single_node_required": True,
        "one_open_per_compressed_source_per_allocation": True,
        "raw_sources_non_evictable": True,
        "derived_only_lru": True,
        "mandatory_ram_headroom_fraction": 0.20,
        "restart_scope": "whole_configuration",
        "persistent_dense_fields_allowed": False,
    }
    if runtime != required_runtime:
        raise ValueError("execution spec runtime safety policy changed")
    if payload.get("final_test_policy") != {
        "hlt_only": True,
        "offline_source_bound": False,
        "oracle_diagnostics_allowed": False,
    }:
        raise ValueError("execution spec final-test policy is not HLT-only")
    r0_payload = payload.get("r0_training")
    if not isinstance(r0_payload, Mapping):
        raise ValueError("execution spec is missing R0 training configuration")
    StreamedR0TrainConfig(output_dir="__VALIDATION__", **dict(r0_payload))
    consumer_payload = payload.get("consumer_training")
    if not isinstance(consumer_payload, Mapping):
        raise ValueError("execution spec is missing consumer training configuration")
    normalized_consumer = dict(consumer_payload)
    if "paired_seed_ids" in normalized_consumer:
        normalized_consumer["paired_seed_ids"] = tuple(normalized_consumer["paired_seed_ids"])
    ConsumerCampaignConfig(**normalized_consumer)
    parent_record = payload.get("parent_manifest", {})
    child_record = payload.get("child_manifest", {})
    if verify_file_hashes:
        records = [parent_record, child_record, payload.get("baseline_checkpoint", {})]
        for source in payload["sources"].values():
            records.extend(source[key] for key in (
                "hlt_npz", "hlt_metadata", "offline_npz", "offline_metadata"
            ))
        for record in records:
            path = _regular_file(record.get("path", ""), label="execution-bound file")
            if sha256_file(path) != record.get("sha256"):
                raise ValueError(f"execution-bound file hash changed: {path}")
    return {
        "ok": True,
        "contract": PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
        "content_hash": payload["content_hash"],
        "parent_manifest_sha256": parent_record.get("manifest_sha256"),
        "child_manifest_sha256": child_record.get("content_hash"),
        "source_splits": list(payload["sources"]),
        "file_hashes_verified": bool(verify_file_hashes),
        "final_test_hlt_only": True,
    }


def validate_bridge_recipe_execution_binding(
    recipe: Mapping[str, Any],
    *,
    execution_spec: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
    r0_checkpoint_sha256: str,
) -> dict[str, str]:
    """Bind a physical bridge recipe to one exact execution and distill child."""

    validate_bridge_recipe(recipe)
    validate_prediction_anchored_execution_spec(
        execution_spec, verify_file_hashes=False
    )
    validate_content_hash(
        child_manifest, expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    if (
        child_manifest.get("content_hash")
        != execution_spec.get("child_manifest", {}).get("content_hash")
    ):
        raise ValueError("execution spec and child manifest differ")
    expected = {
        "r0_checkpoint_sha256": str(r0_checkpoint_sha256),
        "hlt_source_sha256": str(
            execution_spec["sources"]["stack_train"]["hlt_npz"]["sha256"]
        ),
        "offline_source_sha256": str(
            execution_spec["sources"]["stack_train"]["offline_npz"]["sha256"]
        ),
        "split_manifest_sha256": str(
            child_manifest["children"]["stack_train_distill"]["content_hash"]
        ),
        "target_schema_sha256": str(execution_spec["target_schema_sha256"]),
        "preprocessing_sha256": str(execution_spec["preprocessing_sha256"]),
    }
    parents = recipe.get("parent_hashes", {})
    changed = [
        name for name, expected_value in expected.items()
        if parents.get(name) != expected_value
    ]
    if changed:
        raise ValueError(
            "bridge recipe belongs to a different execution: " + ", ".join(changed)
        )
    return expected


def write_prediction_anchored_execution_spec(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    validate_prediction_anchored_execution_spec(payload, verify_file_hashes=False)
    return write_immutable_json(path, payload)


__all__ = [
    "PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT",
    "EXECUTION_PARENT_SPLITS",
    "default_bridge_schema_hashes",
    "build_prediction_anchored_execution_spec",
    "validate_prediction_anchored_execution_spec",
    "validate_bridge_recipe_execution_binding",
    "write_prediction_anchored_execution_spec",
]
