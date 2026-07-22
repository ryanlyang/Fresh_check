#!/usr/bin/env python3
"""Standard-library validation for immutable artifacts used by Slurm resume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


LOGICAL_HASH_KEYS = ("artifact_hash", "manifest_hash", "audit_hash", "metadata_hash")
FUSION_COMPLETION_CONTRACTS = {
    "local_residual_field_fusion_source_artifact_audit_v1",
    "local_residual_field_fusion_metric_reproduction_v1",
    "local_residual_field_a0_seed1_completion_v1",
    "local_residual_field_fusion_prediction_sources_v1",
    "local_residual_field_fusion_feature_manifest_v1",
    "local_residual_field_fusion_candidate_report_v1",
    "local_residual_field_fusion_stability_plan_v1",
    "local_residual_field_fusion_candidate_registry_v1",
    "local_residual_field_selected_fusion_set_v1",
    "local_residual_field_fusion_recipe_replay_v1",
    "local_residual_field_fusion_final_evaluation_v1",
    "local_residual_field_fusion_runtime_v1",
    "local_residual_field_fusion_bootstrap_audit_v1",
    "local_residual_field_fusion_campaign_report_v1",
}
EXPECTED_LOGICAL_PARENT_CONTRACTS = {
    "source_artifact_audit_path": "local_residual_field_fusion_source_artifact_audit_v1",
    "prediction_sources_path": "local_residual_field_fusion_prediction_sources_v1",
    "seed_control_completion_path": "local_residual_field_a0_seed1_completion_v1",
    "source_selected_fusion_path": "local_residual_field_selected_fusion_set_v1",
}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _referenced_files(value: Any) -> list[tuple[Path, str, str]]:
    output: list[tuple[Path, str, str]] = []
    if isinstance(value, Mapping):
        row = dict(value)
        pairs = [("path", "sha256")]
        for key in row:
            if key.endswith("_path"):
                pairs.append((key, f"{key[:-5]}_sha256"))
        if "checkpoint_path" in row and "checkpoint_hash" in row:
            pairs.append(("checkpoint_path", "checkpoint_hash"))
        for path_key, hash_key in pairs:
            path_value, hash_value = row.get(path_key), row.get(hash_key)
            if isinstance(path_value, str) and isinstance(hash_value, str):
                output.append((Path(path_value), hash_value, path_key))
        for item in row.values():
            output.extend(_referenced_files(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(_referenced_files(item))
    return output


def _logical_json_bindings(value: Any) -> list[tuple[Path, str, str]]:
    output: list[tuple[Path, str, str]] = []
    if isinstance(value, Mapping):
        row = dict(value)
        # Only these named fields carry this campaign's logical JSON hash
        # contract.  Other ``*_path``/``*_hash`` pairs (notably the source
        # audit's selected-consumer record) bind raw source bytes instead.
        for path_key in EXPECTED_LOGICAL_PARENT_CONTRACTS:
            path_value = row.get(path_key)
            if not isinstance(path_value, str):
                continue
            stem = path_key[:-5]
            expected = row.get(f"{stem}_hash")
            byte_hash = row.get(f"{stem}_sha256")
            if (
                isinstance(expected, str)
                and not isinstance(byte_hash, str)
                and Path(path_value).suffix.lower() == ".json"
            ):
                output.append((Path(path_value), expected, path_key))
        for item in row.values():
            output.extend(_logical_json_bindings(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(_logical_json_bindings(item))
    return output


def _logical_hash(payload: Mapping[str, Any]) -> tuple[str, str]:
    logical_hash_key = next((key for key in LOGICAL_HASH_KEYS if key in payload), None)
    if logical_hash_key is None:
        raise ValueError("completion JSON lacks a supported logical hash")
    unsigned = dict(payload)
    stored = unsigned.pop(logical_hash_key)
    if not isinstance(stored, str) or stored != _stable_hash(unsigned):
        raise ValueError(f"completion JSON {logical_hash_key} mismatch")
    return logical_hash_key, stored


def _require_contract_bindings(payload: Mapping[str, Any], contract: str) -> None:
    required: dict[str, tuple[str, ...]] = {
        "local_residual_field_fusion_metric_reproduction_v1": (
            "source_artifact_audit_path", "source_artifact_audit_hash",
        ),
        "local_residual_field_a0_seed1_completion_v1": (
            "source_artifact_audit_path", "source_artifact_audit_hash",
            "checkpoint_path", "checkpoint_sha256",
        ),
        "local_residual_field_fusion_prediction_sources_v1": (
            "source_artifact_audit_path", "source_artifact_audit_hash",
            "seed_control_completion_path", "seed_control_completion_hash",
            "checkpoint_path", "checkpoint_hash",
        ),
        "local_residual_field_fusion_feature_manifest_v1": (
            "source_artifact_audit_path", "source_artifact_audit_hash",
            "prediction_sources_path", "prediction_sources_hash",
            "checkpoint_path", "checkpoint_hash",
        ),
        "local_residual_field_fusion_candidate_report_v1": (
            "source_artifact_audit_path", "source_artifact_audit_hash",
            "prediction_sources_path", "prediction_sources_hash",
        ),
        "local_residual_field_fusion_stability_plan_v1": (
            "source_artifact_audit_path", "source_artifact_audit_hash",
            "prediction_sources_path", "prediction_sources_hash", "screening_bindings",
        ),
        "local_residual_field_selected_fusion_set_v1": (
            "source_artifact_audit_path", "source_artifact_audit_hash",
            "prediction_sources_path", "prediction_sources_hash", "selection_input_bindings",
        ),
        "local_residual_field_fusion_recipe_replay_v1": (
            "source_selected_fusion_path", "source_selected_fusion_hash",
        ),
        "local_residual_field_fusion_final_evaluation_v1": (
            "selected_fusion_path", "selected_fusion_sha256",
        ),
        "local_residual_field_fusion_runtime_v1": (
            "selected_fusion_path", "selected_fusion_sha256",
            "final_evaluation_path", "final_evaluation_sha256",
        ),
        "local_residual_field_fusion_bootstrap_audit_v1": (
            "final_evaluation_path", "final_evaluation_sha256",
        ),
        "local_residual_field_fusion_campaign_report_v1": (
            "input_artifacts", "output_artifacts",
        ),
    }
    missing = [key for key in required.get(contract, ()) if payload.get(key) in (None, "", [])]
    if missing:
        raise ValueError(f"completion artifact lacks required {contract} bindings: {missing}")
    if contract == "local_residual_field_fusion_candidate_report_v1":
        if payload.get("family") == "representation" and not payload.get("feature_manifest_bindings"):
            raise ValueError("representation candidate lacks feature-manifest bindings")
        if payload.get("phase") == "stability" and not payload.get("screening_candidate_binding"):
            raise ValueError("stability candidate lacks its screening-candidate binding")
        if payload.get("candidate_id") == "R0_linear_embeddings" and payload.get("phase") == "stability":
            rule = payload.get("head_stability", {}).get("stack_val", {}).get("deployment_rule")
            if rule != "single_fixed_seed_linear_head":
                raise ValueError("R0 stability artifact does not use the fixed-seed deployment contract")


def _resolve_reference(path: Path, *, owner: Path) -> Path:
    return path if path.is_absolute() else owner.parent / path


def _validate_payload_references(
    payload: Mapping[str, Any],
    *,
    owner: Path,
    visited_json: set[Path],
) -> list[str]:
    checked: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw_reference, expected, label in _referenced_files(payload):
        referenced = _resolve_reference(raw_reference, owner=owner)
        key = (str(referenced), expected)
        if key in seen:
            continue
        seen.add(key)
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected.lower()):
            raise ValueError(f"invalid referenced SHA-256 for {label}: {referenced}")
        if not referenced.is_file():
            raise FileNotFoundError(f"referenced completion artifact disappeared: {referenced}")
        if _file_hash(referenced) != expected:
            raise ValueError(f"referenced completion artifact changed: {referenced}")
        checked.append(str(referenced))
        resolved = referenced.resolve()
        if referenced.suffix.lower() == ".json" and resolved not in visited_json:
            nested = json.loads(referenced.read_text(encoding="utf-8"))
            if (
                isinstance(nested, Mapping)
                and nested.get("ok") is True
                and nested.get("contract") in FUSION_COMPLETION_CONTRACTS
                and any(key in nested for key in LOGICAL_HASH_KEYS)
            ):
                _logical_hash(nested)
                _require_contract_bindings(nested, str(nested["contract"]))
                visited_json.add(resolved)
                checked.extend(
                    _validate_payload_references(nested, owner=resolved, visited_json=visited_json)
                )
    for raw_reference, expected, label in _logical_json_bindings(payload):
        referenced = _resolve_reference(raw_reference, owner=owner).resolve()
        if not referenced.is_file():
            raise FileNotFoundError(f"logically bound parent artifact disappeared: {referenced}")
        parent = json.loads(referenced.read_text(encoding="utf-8"))
        if not isinstance(parent, Mapping) or parent.get("ok") is not True:
            raise ValueError(f"logically bound parent is not an ok artifact: {referenced}")
        parent_contract = parent.get("contract")
        if not isinstance(parent_contract, str) or not parent_contract:
            raise ValueError(f"logically bound parent lacks a contract: {referenced}")
        expected_parent_contract = EXPECTED_LOGICAL_PARENT_CONTRACTS.get(label)
        if expected_parent_contract is not None and parent_contract != expected_parent_contract:
            raise ValueError(
                f"logical parent contract mismatch for {label}: expected "
                f"{expected_parent_contract!r}, got {parent_contract!r}"
            )
        _key, observed = _logical_hash(parent)
        _require_contract_bindings(parent, parent_contract)
        if observed != expected:
            raise ValueError(f"logical parent hash changed for {label}: {referenced}")
        checked.append(str(referenced))
        if referenced not in visited_json:
            visited_json.add(referenced)
            checked.extend(
                _validate_payload_references(parent, owner=referenced, visited_json=visited_json)
            )
    return checked


def validate_completion(path: str | Path, *, expected_contract: str | None = None) -> dict[str, Any]:
    artifact_path = Path(path).resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise ValueError("completion JSON is not an ok artifact")
    contract = payload.get("contract")
    if not isinstance(contract, str) or not contract.strip():
        raise ValueError("completion JSON lacks a nonempty contract")
    if expected_contract is not None and contract != expected_contract:
        raise ValueError(f"completion contract mismatch: expected {expected_contract!r}, got {contract!r}")
    _require_contract_bindings(payload, contract)
    logical_hash_key, _stored = _logical_hash(payload)
    checked = _validate_payload_references(
        payload, owner=artifact_path, visited_json={artifact_path},
    )
    return {
        "ok": True, "path": str(artifact_path.resolve()), "contract": contract,
        "logical_hash_key": logical_hash_key, "referenced_files_checked": checked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-contract")
    args = parser.parse_args(argv)
    report = validate_completion(args.path, expected_contract=args.expected_contract)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
