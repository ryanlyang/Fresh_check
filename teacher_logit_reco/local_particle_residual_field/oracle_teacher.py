"""Provenance contracts for reusable local residual-field oracle teachers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_REUSE_CONTRACT = "local_residual_field_oracle_teacher_reuse_v1"
ORACLE_TEACHER_TRAIN_SPLITS = ("model_train", "model_val", "stack_val")
ORACLE_TEACHER_LOGIT_SPLITS = ("model_train", "model_val", "stack_train", "stack_val")


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def canonical_oracle_field_source(value: Any) -> str:
    source = str(value or "").strip().lower().replace("-", "_")
    if source in {"oracle", "target", "targets", "oracle_scaled", "scaled_oracle", "alpha_oracle"}:
        return "oracle_scaled"
    if source in {"oracle_field_subset", "oracle_subset", "subset_oracle"}:
        return "oracle_field_subset"
    if source in {"oracle_noisy", "noisy_oracle", "oracle_noise"}:
        return "oracle_noisy"
    if source in {"oracle_dropout", "dropout_oracle"}:
        return "oracle_dropout"
    if source in {"zero", "blank", "zero_augmented"}:
        return "zero"
    return source


def _dataset_split_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    alignment = _mapping(metadata.get("alignment_report"))
    hlt = _mapping(metadata.get("hlt_metadata"))
    target = _mapping(metadata.get("target_metadata"))
    return {
        "source_manifest_hash": _first(
            alignment.get("source_manifest_hash"),
            hlt.get("source_manifest_hash"),
            metadata.get("source_manifest_hash"),
        ),
        "hlt_content_hash": _first(
            alignment.get("hlt_content_hash"),
            hlt.get("hlt_content_hash"),
            metadata.get("hlt_content_hash"),
        ),
        "offline_content_hash": _first(
            alignment.get("offline_content_hash"),
            target.get("offline_content_hash"),
            metadata.get("offline_content_hash"),
        ),
        "target_content_hash": _first(
            alignment.get("target_content_hash"),
            target.get("target_content_hash"),
            metadata.get("target_content_hash"),
        ),
        "jet_identity_hash": _first(
            alignment.get("jet_identity_hash"),
            hlt.get("jet_identity_hash"),
            metadata.get("jet_identity_hash"),
        ),
    }


def _selected_field_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(payload.get("model_config"))
    names = payload.get("selected_field_names") or model.get("field_names") or payload.get("field_names") or ()
    groups = payload.get("selected_field_groups") or model.get("field_groups") or payload.get("field_groups") or {}
    indices = payload.get("selected_field_indices") or model.get("source_field_indices") or ()
    return {
        "field_names": [str(name) for name in names],
        "field_groups": {
            str(group): [int(index) for index in values]
            for group, values in sorted(dict(groups).items())
        },
        "source_field_indices": [int(index) for index in indices],
    }


def _label_order(payload: Mapping[str, Any]) -> list[str]:
    train = _mapping(payload.get("train_config"))
    model = _mapping(payload.get("model_config"))
    values = payload.get("label_names") or train.get("label_names") or model.get("label_names") or ()
    return [str(value) for value in values]


def _model_architecture(payload: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(payload.get("model_config"))
    train = _mapping(payload.get("train_config"))
    return {
        "model_contract": str(model.get("contract") or payload.get("model_contract") or ""),
        "num_classes": int(_first(model.get("num_classes"), train.get("num_classes"), 0) or 0),
        "field_dim": int(_first(model.get("field_dim"), len(_selected_field_schema(payload)["field_names"]), 0) or 0),
        "base_feature_dim": int(model.get("base_feature_dim") or 0),
        "augmented_feature_dim": int(model.get("augmented_feature_dim") or 0),
        "model_size": str(_first(model.get("model_size"), train.get("model_size"), "") or ""),
    }


def _field_recipe(payload: Mapping[str, Any]) -> dict[str, Any]:
    train = _mapping(payload.get("train_config"))
    model = _mapping(payload.get("model_config"))

    def value(name: str, default: Any) -> Any:
        return _first(payload.get(name), train.get(name), model.get(name), default)

    return {
        "field_source": canonical_oracle_field_source(value("field_source", "")),
        "oracle_field_alpha": float(value("oracle_field_alpha", 1.0)),
        "oracle_field_noise_std": float(value("oracle_field_noise_std", 0.0)),
        "oracle_field_dropout": float(value("oracle_field_dropout", 0.0)),
        "oracle_field_group_dropout": float(value("oracle_field_group_dropout", 0.0)),
        "field_subset": [str(item) for item in (payload.get("field_subset") or train.get("field_subset") or ())],
        "selected_field_names": list(_selected_field_schema(payload)["field_names"]),
    }


def build_oracle_teacher_reuse_contract(
    payload: Mapping[str, Any],
    *,
    required_splits: Sequence[str] = ORACLE_TEACHER_TRAIN_SPLITS,
) -> dict[str, Any]:
    dataset = _mapping(payload.get("dataset_metadata"))
    split_provenance = {
        str(split): _dataset_split_provenance(_mapping(dataset.get(str(split))))
        for split in required_splits
    }
    manifests = sorted(
        {
            str(values["source_manifest_hash"])
            for values in split_provenance.values()
            if values.get("source_manifest_hash") not in (None, "")
        }
    )
    field_schema = _selected_field_schema(payload)
    label_order = _label_order(payload)
    architecture = _model_architecture(payload)
    field_recipe = _field_recipe(payload)
    contract = {
        "contract": LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_REUSE_CONTRACT,
        "required_splits": [str(split) for split in required_splits],
        "source_manifest_hash": manifests[0] if len(manifests) == 1 else None,
        "source_manifest_hashes": manifests,
        "split_provenance": split_provenance,
        "split_provenance_hash": stable_json_hash(split_provenance),
        "field_schema": field_schema,
        "field_schema_hash": stable_json_hash(field_schema),
        "label_order": label_order,
        "label_order_hash": stable_json_hash(label_order),
        "field_recipe": field_recipe,
        "field_recipe_hash": stable_json_hash(field_recipe),
        "model_architecture": architecture,
        "model_architecture_hash": stable_json_hash(architecture),
    }
    contract["reuse_contract_hash"] = stable_json_hash(contract)
    return contract


def validate_oracle_teacher_reuse_contract(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "required_splits",
        "source_manifest_hash",
        "split_provenance_hash",
        "field_schema_hash",
        "label_order_hash",
        "field_recipe_hash",
        "model_architecture_hash",
    )
    mismatches: list[dict[str, Any]] = []
    for field in fields:
        actual_value = actual.get(field)
        expected_value = expected.get(field)
        if actual_value != expected_value:
            mismatches.append({"field": field, "actual": actual_value, "expected": expected_value})
    missing_split_fields: list[str] = []
    for split, values in _mapping(actual.get("split_provenance")).items():
        for name, value in _mapping(values).items():
            if value in (None, ""):
                missing_split_fields.append(f"{split}.{name}")
    if missing_split_fields:
        mismatches.append(
            {
                "field": "split_provenance_missing_values",
                "actual": sorted(missing_split_fields),
                "expected": [],
            }
        )
    return {
        "ok": not mismatches,
        "contract": LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_REUSE_CONTRACT,
        "actual_reuse_contract_hash": actual.get("reuse_contract_hash"),
        "expected_reuse_contract_hash": expected.get("reuse_contract_hash"),
        "mismatches": mismatches,
    }


__all__ = [
    "LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_REUSE_CONTRACT",
    "ORACLE_TEACHER_TRAIN_SPLITS",
    "ORACLE_TEACHER_LOGIT_SPLITS",
    "stable_json_hash",
    "canonical_oracle_field_source",
    "build_oracle_teacher_reuse_contract",
    "validate_oracle_teacher_reuse_contract",
]
