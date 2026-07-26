"""Small fail-closed I/O helpers shared by Step-7 command-line workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    load_hashed_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .provenance import source_snapshot


RUN_RESULT_ENVELOPE_CONTRACT = "relational_part_run_results_v1"


def parse_named_hashes(values: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=SHA256, received {value!r}")
        name, digest = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"duplicate or empty hash name: {name!r}")
        output[name] = require_sha256(digest, name=name)
    if not output:
        raise ValueError("at least one named HLT-cache hash is required")
    return dict(sorted(output.items()))


def load_record_sequence(path: str | Path) -> list[Mapping[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, Mapping):
        if payload.get("contract") == RUN_RESULT_ENVELOPE_CONTRACT:
            validate_content_hash(
                payload, expected_contract=RUN_RESULT_ENVELOPE_CONTRACT
            )
        for name in ("rows", "results", "evaluations"):
            if isinstance(payload.get(name), list):
                payload = payload[name]
                break
    if not isinstance(payload, list) or not all(
        isinstance(row, Mapping) for row in payload
    ):
        raise ValueError(f"{source} must contain a JSON record sequence")
    return list(payload)


def validate_campaign_source(
    campaign: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Fail closed if a continuation uses code unlike the campaign source."""

    current = source_snapshot(repo_root)
    expected = campaign.get("source")
    if not isinstance(expected, Mapping):
        raise ValueError("campaign spec lacks its source snapshot")
    observed = {
        "commit": current["source_commit"],
        "status_sha256": current["source_status_sha256"],
        "dirty": bool(current["source_dirty"]),
    }
    locked = {
        "commit": expected.get("commit"),
        "status_sha256": expected.get("status_sha256"),
        "dirty": bool(expected.get("dirty")),
    }
    if observed != locked:
        raise ValueError(
            "active repository source snapshot differs from campaign_spec.json"
        )
    return current


def expected_training_lineage(
    campaign_root: str | Path,
    *,
    families: Sequence[str],
) -> dict[str, str]:
    root = Path(campaign_root)
    campaign = load_hashed_json(root / "campaign_spec.json")
    inputs = root / "inputs"
    binding = load_hashed_json(inputs / "hlt_cache_audit.json")
    expected = {
        "campaign_spec": campaign["content_hash"],
        "split_manifest": require_sha256(
            campaign["split_manifest_hash"], name="split_manifest_hash"
        ),
        "raw_input_schema": load_hashed_json(
            inputs / "raw_input_schema.json"
        )["content_hash"],
        "hlt_binding": binding["content_hash"],
        "postconstruction_input_audit": load_hashed_json(
            inputs / "postconstruction_input_audit.json"
        )["content_hash"],
        "relation_normalization": load_hashed_json(
            inputs / "relation_normalization.json"
        )["content_hash"],
    }
    for split in ("model_train", "model_val", "stack_val"):
        expected[f"hlt_{split}"] = require_sha256(
            binding["split_reports"][split]["hlt_content_hash"],
            name=f"hlt_{split}",
        )
    if "REGION" in set(map(str, families)):
        expected.update(
            {
                "region_normalization": load_hashed_json(
                    inputs / "region_normalization.json"
                )["content_hash"],
                "angular_tree_resource": load_hashed_json(
                    inputs / "angular_tree_resource_contract.json"
                )["content_hash"],
                "angular_tree_backend": load_hashed_json(
                    root / "backend" / "backend_manifest.json"
                )["content_hash"],
                "angular_tree_throughput_probe": load_hashed_json(
                    root / "backend" / "throughput_probe.json"
                )["content_hash"],
            }
        )
        tree_root = inputs / "relation_tree_cache"
        for split in ("model_train", "model_val", "stack_val"):
            expected[f"angular_tree_{split}"] = load_hashed_json(
                tree_root / f"{split}_exclusive_ca_v1" / "manifest.json"
            )["content_hash"]
    return dict(sorted(expected.items()))


def load_run_result(
    run_directory: str | Path,
    *,
    expected_lineage: Mapping[str, str] | None = None,
    expected_run_registry_sha256: str | None = None,
    expected_relation_registry_sha256: str | None = None,
    expected_model_contract_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(run_directory)
    registration = load_hashed_json(root / "checkpoint_registration.json")
    metrics = load_hashed_json(root / "val_select_metrics.json")
    if registration.get("val_select_metrics_sha256") != metrics["content_hash"]:
        raise ValueError(f"{root} registration and val_select metrics disagree")
    profile = registration.get("parameter_and_flop_profile")
    if not isinstance(profile, Mapping):
        raise ValueError(f"{root} lacks its registered parameter profile")
    checkpoint_path = root / str(registration.get("checkpoint_file", ""))
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != registration.get("checkpoint_sha256")
    ):
        raise ValueError(f"{root} checkpoint file hash differs")
    lineage = registration.get("lineage_hashes")
    if not isinstance(lineage, Mapping):
        raise ValueError(f"{root} lacks checkpoint lineage")
    normalized_lineage = {
        str(name): require_sha256(value, name=f"lineage_hashes.{name}")
        for name, value in sorted(lineage.items())
    }
    if expected_lineage is not None:
        normalized_expected = {
            str(name): require_sha256(value, name=f"expected_lineage.{name}")
            for name, value in sorted(expected_lineage.items())
        }
        if normalized_lineage != normalized_expected:
            raise ValueError(f"{root} checkpoint lineage differs from campaign")
    expected_bindings = {
        "run_registry_sha256": expected_run_registry_sha256,
        "relation_registry_sha256": expected_relation_registry_sha256,
        "model_contract_sha256": expected_model_contract_sha256,
    }
    for field, expected in expected_bindings.items():
        if expected is not None and registration.get(field) != require_sha256(
            expected, name=field
        ):
            raise ValueError(f"{root} registration {field} differs")
    return {
        "run_id": registration["run_id"],
        "seed": int(registration["seed"]),
        "configuration_role": registration["configuration_role"],
        "relational_selection_eligible": registration[
            "relational_selection_eligible"
        ],
        "checkpoint_sha256": registration["checkpoint_sha256"],
        "parameter_count": int(profile["trainable_parameters"]),
        "checkpoint_registration_sha256": registration["content_hash"],
        "val_select_metrics_sha256": metrics["content_hash"],
        "model_contract_sha256": registration["model_contract_sha256"],
        "training_contract_sha256": registration["training_contract_sha256"],
        "run_registry_sha256": registration["run_registry_sha256"],
        "relation_registry_sha256": registration[
            "relation_registry_sha256"
        ],
        "lineage_hashes": normalized_lineage,
        "lineage_authenticated": True,
        "val_select": metrics,
    }


def build_run_result_envelope(
    *,
    mode: str,
    registry_sha256: str,
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if mode not in {"screening", "confirmation", "unary"}:
        raise ValueError("unknown result-envelope mode")
    if not results:
        raise ValueError("result envelope may not be empty")
    registration_hashes = {}
    for row in results:
        run_seed = f"{row['run_id']}:seed_{int(row['seed'])}"
        if row.get("lineage_authenticated") is not True:
            raise ValueError(f"{run_seed} lineage is not authenticated")
        registration_hashes[run_seed] = require_sha256(
            row.get("checkpoint_registration_sha256"),
            name=f"{run_seed}.checkpoint_registration_sha256",
        )
    return with_content_hash(
        {
            "contract": RUN_RESULT_ENVELOPE_CONTRACT,
            "schema_version": 1,
            "mode": mode,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "registry_sha256": require_sha256(
                registry_sha256, name="registry_sha256"
            ),
            "source": {
                "commit": source["source_commit"],
                "status_sha256": require_sha256(
                    source["source_status_sha256"],
                    name="source.status_sha256",
                ),
                "dirty": bool(source["source_dirty"]),
            },
            "row_count": len(results),
            "checkpoint_registration_hashes": registration_hashes,
            "results": [dict(row) for row in results],
            "all_lineage_authenticated": True,
        }
    )


def reject_final_test_paths(paths: Sequence[str | Path]) -> None:
    for value in paths:
        parts = {part.lower() for part in Path(value).parts}
        if "final_test" in parts:
            raise ValueError(
                "selection and confirmation workers may not access final_test"
            )


__all__ = [
    "load_record_sequence",
    "build_run_result_envelope",
    "expected_training_lineage",
    "load_run_result",
    "parse_named_hashes",
    "reject_final_test_paths",
    "RUN_RESULT_ENVELOPE_CONTRACT",
    "validate_campaign_source",
]
