"""Immutable contracts for the prespecified offline RPT transfer experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    SOURCE_STATUS_HASH_POLICY,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)


OFFLINE_TRANSFER_CAMPAIGN_CONTRACT = "relational_part_offline_transfer_campaign_v1"
OFFLINE_TRANSFER_CACHE_BINDING_CONTRACT = (
    "relational_part_offline_transfer_cache_binding_v1"
)
OFFLINE_TRANSFER_MODEL_CONTRACT = "relational_part_offline_transfer_model_v1"
OFFLINE_TRANSFER_TASK_REGISTRY_CONTRACT = (
    "relational_part_offline_transfer_task_registry_v1"
)
OFFLINE_TRANSFER_VALIDATION_SUMMARY_CONTRACT = (
    "relational_part_offline_transfer_validation_summary_v1"
)
OFFLINE_TRANSFER_FINAL_LOCK_CONTRACT = (
    "relational_part_offline_transfer_final_lock_v1"
)
OFFLINE_TRANSFER_FINAL_TASK_REGISTRY_CONTRACT = (
    "relational_part_offline_transfer_final_task_registry_v1"
)
OFFLINE_TRANSFER_REPORT_CONTRACT = "relational_part_offline_transfer_report_v1"

OFFLINE_TRANSFER_SEEDS = (101, 202, 303)
OFFLINE_TRANSFER_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "OFF_RPT_BASE": {
        "architecture_source_run_id": "RPT_BASE",
        "attention_architecture": "standard_shared_pair_bias",
        "relation_families": [],
    },
    "OFF_RPT_BASE_EDGEVALUE": {
        "architecture_source_run_id": "RPT_BASE_EDGEVALUE",
        "attention_architecture": "edge_conditioned_values",
        "relation_families": [],
    },
    "OFF_RPT_SELECTED_LAYERWISE": {
        "architecture_source_run_id": "RPT_SELECTED_LAYERWISE",
        "attention_architecture": "layerwise_pair_bias",
        "relation_families": ["PT", "TRACK", "REGION"],
    },
    "OFF_RPT_SELECTED_EDGEVALUE": {
        "architecture_source_run_id": "RPT_SELECTED_EDGEVALUE",
        "attention_architecture": "edge_conditioned_values",
        "relation_families": ["PT", "TRACK", "REGION"],
    },
}


def build_offline_transfer_campaign(
    *,
    campaign_id: str,
    parent_campaign: Mapping[str, Any],
    parent_campaign_path: str | Path,
    split_manifest_path: str | Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the experiment before offline validation or test is opened."""

    parent_sha = validate_content_hash(parent_campaign)
    split_sha = sha256_file(split_manifest_path)
    source_row = {
        "commit": str(source["source_commit"]),
        "status_sha256": require_sha256(
            source["source_status_sha256"], name="source.status_sha256"
        ),
        "dirty": bool(source["source_dirty"]),
        "status_hash_policy": SOURCE_STATUS_HASH_POLICY,
    }
    return with_content_hash(
        {
            "contract": OFFLINE_TRANSFER_CAMPAIGN_CONTRACT,
            "schema_version": 1,
            "campaign_id": str(campaign_id),
            "source": source_row,
            "parent_hlt_campaign": {
                "path": str(Path(parent_campaign_path).resolve()),
                "campaign_spec_sha256": parent_sha,
                "campaign_id": str(parent_campaign.get("campaign_id", "")),
            },
            "split_manifest": {
                "path": str(Path(split_manifest_path).resolve()),
                "file_sha256": split_sha,
                "scientific_rule": "exact_parent_campaign_event_identities_and_order",
            },
            "input_view": "offline",
            "class_order": [
                "QCD", "Hbb", "Hcc", "Hgg", "H4q",
                "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl",
            ],
            "model_matrix": OFFLINE_TRANSFER_MODEL_SPECS,
            "seeds": list(OFFLINE_TRANSFER_SEEDS),
            "training_protocol": {
                "source": "relational_part_training_v1_production",
                "same_as_parent_hlt_confirmation": True,
                "normalization_fit_split": "model_train",
                "checkpoint_selection_split": "model_val",
                "validation_reporting_split": "stack_val",
                "final_evaluation_split": "final_test",
            },
            "hypotheses": {
                "primary": (
                    "OFF_RPT_SELECTED_EDGEVALUE_minus_OFF_RPT_BASE"
                ),
                "architecture_control": (
                    "OFF_RPT_BASE_EDGEVALUE_minus_OFF_RPT_BASE"
                ),
                "relation_path_control": (
                    "OFF_RPT_SELECTED_EDGEVALUE_minus_"
                    "OFF_RPT_SELECTED_LAYERWISE"
                ),
            },
            "selection_policy": {
                "all_four_models_and_all_three_seeds_are_predeclared": True,
                "validation_cannot_remove_a_predeclared_final_task": True,
                "final_test_open_requires_immutable_lock": True,
            },
            "failure_policy": {
                "performance_gate": False,
                "poor_metrics_never_cancel_future_tasks": True,
                "runtime_or_integrity_failure_still_fails_the_affected_job": True,
            },
            "scientific_interpretation": {
                "purpose": "offline_domain_replication_of_hlt_edgevalue_gain",
                "claim_is_transfer_not_global_test_set_novelty": True,
                "reason": (
                    "the same offline split identities may have appeared in "
                    "earlier repository experiments"
                ),
            },
        }
    )


def validate_offline_transfer_campaign(campaign: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        campaign, expected_contract=OFFLINE_TRANSFER_CAMPAIGN_CONTRACT
    )
    if campaign.get("input_view") != "offline":
        raise ValueError("offline transfer campaign input view differs")
    if campaign.get("model_matrix") != OFFLINE_TRANSFER_MODEL_SPECS:
        raise ValueError("offline transfer model matrix differs")
    if tuple(campaign.get("seeds", ())) != OFFLINE_TRANSFER_SEEDS:
        raise ValueError("offline transfer seeds differ")
    failure = campaign.get("failure_policy", {})
    if failure.get("performance_gate") is not False or failure.get(
        "poor_metrics_never_cancel_future_tasks"
    ) is not True:
        raise ValueError("offline transfer introduced a performance gate")
    require_sha256(
        campaign.get("split_manifest", {}).get("file_sha256"),
        name="split_manifest.file_sha256",
    )
    return digest


def build_offline_model_contract(
    run_id: str,
    *,
    campaign_sha256: str,
    relation_normalization_sha256: str,
    region_normalization_sha256: str,
) -> dict[str, Any]:
    if run_id not in OFFLINE_TRANSFER_MODEL_SPECS:
        raise ValueError(f"unknown offline transfer run {run_id!r}")
    spec = OFFLINE_TRANSFER_MODEL_SPECS[run_id]
    return with_content_hash(
        {
            "contract": OFFLINE_TRANSFER_MODEL_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            **spec,
            "input_view": "offline",
            "campaign_sha256": require_sha256(
                campaign_sha256, name="campaign_sha256"
            ),
            "relation_normalization_sha256": require_sha256(
                relation_normalization_sha256,
                name="relation_normalization_sha256",
            ),
            "region_normalization_sha256": require_sha256(
                region_normalization_sha256,
                name="region_normalization_sha256",
            ),
            "initialization": "from_scratch_at_each_seed",
            "checkpoint_selection": (
                "model_val_only_exact_parent_training_protocol"
            ),
        }
    )


def build_offline_task_registry(
    *,
    campaign_sha256: str,
    model_contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    tasks = []
    for run_id in OFFLINE_TRANSFER_MODEL_SPECS:
        contract = model_contracts[run_id]
        validate_content_hash(
            contract, expected_contract=OFFLINE_TRANSFER_MODEL_CONTRACT
        )
        for seed in OFFLINE_TRANSFER_SEEDS:
            tasks.append(
                {
                    "task_index": len(tasks),
                    "run_id": run_id,
                    "seed": seed,
                    "relation_families": list(
                        OFFLINE_TRANSFER_MODEL_SPECS[run_id]["relation_families"]
                    ),
                    "model_contract_sha256": contract["content_hash"],
                    "model_contract_path": (
                        f"registry/model_contracts/{run_id}.json"
                    ),
                }
            )
    return with_content_hash(
        {
            "contract": OFFLINE_TRANSFER_TASK_REGISTRY_CONTRACT,
            "schema_version": 1,
            "campaign_sha256": require_sha256(
                campaign_sha256, name="campaign_sha256"
            ),
            "task_count": len(tasks),
            "tasks": tasks,
            "performance_gate": False,
        }
    )


def validate_offline_task_registry(registry: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        registry, expected_contract=OFFLINE_TRANSFER_TASK_REGISTRY_CONTRACT
    )
    expected = len(OFFLINE_TRANSFER_MODEL_SPECS) * len(OFFLINE_TRANSFER_SEEDS)
    tasks = registry.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != expected:
        raise ValueError("offline task registry coverage differs")
    observed = [(row.get("run_id"), row.get("seed")) for row in tasks]
    wanted = [
        (run_id, seed)
        for run_id in OFFLINE_TRANSFER_MODEL_SPECS
        for seed in OFFLINE_TRANSFER_SEEDS
    ]
    if observed != wanted or registry.get("performance_gate") is not False:
        raise ValueError("offline task registry ordering or failure policy differs")
    return digest


__all__ = [
    "OFFLINE_TRANSFER_CACHE_BINDING_CONTRACT",
    "OFFLINE_TRANSFER_CAMPAIGN_CONTRACT",
    "OFFLINE_TRANSFER_FINAL_LOCK_CONTRACT",
    "OFFLINE_TRANSFER_FINAL_TASK_REGISTRY_CONTRACT",
    "OFFLINE_TRANSFER_MODEL_CONTRACT",
    "OFFLINE_TRANSFER_MODEL_SPECS",
    "OFFLINE_TRANSFER_REPORT_CONTRACT",
    "OFFLINE_TRANSFER_SEEDS",
    "OFFLINE_TRANSFER_TASK_REGISTRY_CONTRACT",
    "OFFLINE_TRANSFER_VALIDATION_SUMMARY_CONTRACT",
    "build_offline_model_contract",
    "build_offline_task_registry",
    "build_offline_transfer_campaign",
    "validate_offline_task_registry",
    "validate_offline_transfer_campaign",
]
