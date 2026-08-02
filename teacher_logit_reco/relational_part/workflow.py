"""Small fail-closed I/O helpers shared by Step-7 command-line workers."""

from __future__ import annotations

import json
import os
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
CHECKPOINT_REGISTRATION_CONTRACT = "relational_part_checkpoint_registration_v2"
SOURCE_RECOVERY_AUTHORIZATION_CONTRACT = (
    "relational_part_source_recovery_authorization_v1"
)
SOURCE_RECOVERY_AUTHORIZATION_ENV = "RPT_SOURCE_RECOVERY_AUTHORIZATION"
RECOVERED_ARCHITECTURE_RUN_IDS = (
    "RPT_BASE_LAYERWISE",
    "RPT_BASE_EDGEVALUE",
    "RPT_SELECTED_LAYERWISE",
    "RPT_SELECTED_EDGEVALUE",
)


def _source_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "commit": snapshot["source_commit"],
        "status_sha256": require_sha256(
            snapshot["source_status_sha256"], name="source.status_sha256"
        ),
        "dirty": bool(snapshot["source_dirty"]),
    }


def validate_source_recovery_authorization(
    authorization_path: str | Path,
    *,
    campaign: Mapping[str, Any],
    current_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the narrow source repair used by the failed Step-6 rows."""

    path = Path(authorization_path).resolve()
    authorization = load_hashed_json(
        path, expected_contract=SOURCE_RECOVERY_AUTHORIZATION_CONTRACT
    )
    campaign_sha = validate_content_hash(campaign)
    if authorization.get("campaign_spec_sha256") != campaign_sha:
        raise ValueError("source recovery belongs to another campaign")
    if authorization.get("campaign_root") != str(path.parents[2]):
        raise ValueError("source recovery path is outside its campaign root")
    expected_original = campaign.get("source")
    if not isinstance(expected_original, Mapping) or authorization.get(
        "original_campaign_source"
    ) != dict(expected_original):
        raise ValueError("source recovery original source differs from campaign")
    if authorization.get("recovery_source") != _source_identity(current_source):
        raise ValueError("active source differs from recovery authorization")
    if tuple(authorization.get("authorized_run_ids", ())) != tuple(
        sorted(RECOVERED_ARCHITECTURE_RUN_IDS)
    ):
        raise ValueError("source recovery run scope differs")
    if authorization.get("downstream_continuation_authorized") is not True:
        raise ValueError("source recovery does not authorize continuation")
    if authorization.get("final_test_still_requires_locked_finalists") is not True:
        raise ValueError("source recovery weakens the final-test seal")
    contracts = authorization.get("corrected_model_contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(
        RECOVERED_ARCHITECTURE_RUN_IDS
    ):
        raise ValueError("source recovery model-contract coverage differs")
    campaign_root = path.parents[2]
    for run_id, row in contracts.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"source recovery contract row differs for {run_id}")
        contract_path = (campaign_root / str(row.get("path", ""))).resolve()
        try:
            contract_path.relative_to(campaign_root)
        except ValueError as exc:
            raise ValueError("source recovery model contract escapes campaign") from exc
        contract = load_hashed_json(contract_path)
        if (
            contract.get("contract") != "relational_part_step6_model_v2"
            or contract.get("run_id") != run_id
            or contract.get("content_hash") != row.get("sha256")
        ):
            raise ValueError(f"source recovery model contract differs for {run_id}")
    task_registry = load_hashed_json(
        campaign_root
        / "selection"
        / "architecture_recovery_v1"
        / "architecture_tasks.json"
    )
    if (
        task_registry.get("contract")
        != "relational_part_confirmation_architecture_recovery_tasks_v1"
        or task_registry.get("content_hash")
        != authorization.get("recovery_task_registry_sha256")
        or int(task_registry.get("task_count", -1)) != 12
    ):
        raise ValueError("source recovery task registry differs")
    preflight = load_hashed_json(
        campaign_root
        / "selection"
        / "architecture_recovery_v1"
        / "real_weaver_construction_preflight.json"
    )
    if (
        preflight.get("contract")
        != "relational_part_real_weaver_architecture_recovery_preflight_v1"
        or preflight.get("content_hash")
        != authorization.get("real_weaver_construction_preflight_sha256")
        or preflight.get("real_weaver_import_and_construction_passed") is not True
        or preflight.get("trailing_BatchNorm1d_captured_per_layer") is not True
    ):
        raise ValueError("source recovery real-Weaver preflight differs")
    reused = authorization.get("reused_ordinary_checkpoint_registration_hashes")
    if (
        not isinstance(reused, Mapping)
        or len(reused) != 39
        or int(authorization.get("reused_ordinary_run_seed_count", -1)) != 39
        or authorization.get("retrain_ordinary_runs") is not False
    ):
        raise ValueError("source recovery ordinary-run reuse coverage differs")
    for name, digest in reused.items():
        require_sha256(digest, name=f"reused_ordinary.{name}")
    if authorization.get("performance_gate") is not False:
        raise ValueError("source recovery contains a performance gate")
    return authorization


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
    observed = _source_identity(current)
    locked = {
        "commit": expected.get("commit"),
        "status_sha256": expected.get("status_sha256"),
        "dirty": bool(expected.get("dirty")),
    }
    if observed != locked:
        authorization_path = os.environ.get(SOURCE_RECOVERY_AUTHORIZATION_ENV)
        if not authorization_path:
            raise ValueError(
                "active repository source snapshot differs from campaign_spec.json"
            )
        validate_source_recovery_authorization(
            authorization_path,
            campaign=campaign,
            current_source=current,
        )
    return current


def resolve_model_contract_path(
    campaign_root: str | Path,
    run_id: str,
) -> Path:
    """Resolve a model contract, honoring an authenticated Step-6 repair."""

    root = Path(campaign_root).resolve()
    authorization_path = os.environ.get(SOURCE_RECOVERY_AUTHORIZATION_ENV)
    if authorization_path and run_id in RECOVERED_ARCHITECTURE_RUN_IDS:
        campaign = load_hashed_json(root / "campaign_spec.json")
        current = source_snapshot(Path(__file__).resolve().parents[2])
        authorization = validate_source_recovery_authorization(
            authorization_path,
            campaign=campaign,
            current_source=current,
        )
        row = authorization["corrected_model_contracts"][run_id]
        path = root / str(row["path"])
        artifact = load_hashed_json(path)
        if artifact.get("run_id") != run_id or artifact.get(
            "content_hash"
        ) != row.get("sha256"):
            raise ValueError(f"recovered model contract differs for {run_id}")
        return path
    candidates = [
        root / "registry" / "model_contracts" / f"{run_id}.json",
        root / "registry" / "confirmation_model_contracts" / f"{run_id}.json",
        root / "selection" / "semantic_controls" / "unary_model_contract.json",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{run_id} must resolve to exactly one model contract; found={matches}"
        )
    artifact = load_hashed_json(matches[0])
    if artifact.get("run_id") != run_id:
        raise ValueError(f"{matches[0]} belongs to another run")
    return matches[0]


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
    registration = load_hashed_json(
        root / "checkpoint_registration.json",
        expected_contract=CHECKPOINT_REGISTRATION_CONTRACT,
    )
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
    "RECOVERED_ARCHITECTURE_RUN_IDS",
    "SOURCE_RECOVERY_AUTHORIZATION_CONTRACT",
    "SOURCE_RECOVERY_AUTHORIZATION_ENV",
    "resolve_model_contract_path",
    "validate_source_recovery_authorization",
    "validate_campaign_source",
]
