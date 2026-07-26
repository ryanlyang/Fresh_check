"""Three-seed selection, fairness closure, and sealed-split authorization."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import median
from typing import Any, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash


PARTICLE_VIEW_SEED_AGGREGATE_CONTRACT = "particle_view_seed_aggregate_v1"
PARTICLE_VIEW_WINNER_SELECTION_CONTRACT = "particle_view_winner_selection_v1"
PARTICLE_VIEW_FAIRNESS_LEDGER_CONTRACT = "particle_view_selected_path_fairness_ledger_v1"
PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT = "particle_view_split_authorization_v1"
PARTICLE_VIEW_STACK_REPORT_CONTRACT = "particle_view_stack_confirmation_v1"
PARTICLE_VIEW_SEED_EXPANSION_CONTRACT = "particle_view_seed_expansion_v1"
LOCKED_CONFIRMATION_SEEDS = (101, 202, 303)
_EXPECTED_STAGE_G_CONTROLS = {
    "A0_VIEW_LONG_DEPLOY",
    "A0_VIEW_TOTAL_LABEL_BUDGET",
    "SELECTED_PARAMETER_MATCH",
    "SELECTED_FLOP_MATCH",
}


def expand_three_seed_confirmation_rows(
    *,
    screen_rows: Sequence[Mapping[str, Any]],
    confirmation_roles: Mapping[str, str],
) -> dict[str, Any]:
    """Expand predeclared/numerically resolved roles without quality gating.

    ``confirmation_roles`` maps each Section-15.8 role to a configuration ID.
    Multiple roles resolving to one configuration are hash-deduplicated while
    preserving the complete role list.
    """

    if not confirmation_roles:
        raise ValueError("confirmation role mapping is empty")
    by_configuration: dict[str, Mapping[str, Any]] = {}
    for row in screen_rows:
        configuration_id = str(row.get("configuration_id", ""))
        if not configuration_id:
            raise ValueError("screen row has no configuration_id")
        if configuration_id in by_configuration:
            raise ValueError("screen configuration IDs must be unique")
        by_configuration[configuration_id] = row
    roles_by_configuration: dict[str, list[str]] = defaultdict(list)
    for role, configuration_id in confirmation_roles.items():
        if not role or configuration_id not in by_configuration:
            raise ValueError(f"confirmation role {role!r} has an unknown row")
        roles_by_configuration[configuration_id].append(str(role))
    expanded = []
    for configuration_id in sorted(roles_by_configuration):
        base = dict(by_configuration[configuration_id])
        base.pop("seed", None)
        base.pop("row_id", None)
        roles = sorted(roles_by_configuration[configuration_id])
        for seed in LOCKED_CONFIRMATION_SEEDS:
            expanded.append(
                {
                    **base,
                    "configuration_id": configuration_id,
                    "seed": seed,
                    "row_id": f"{configuration_id}__seed={seed}",
                    "confirmation_roles": roles,
                    "numerically_added_confirmation": any(
                        role.startswith("best_") for role in roles
                    ),
                    "minimum_quality_gate": None,
                }
            )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_SEED_EXPANSION_CONTRACT,
            "seeds": list(LOCKED_CONFIRMATION_SEEDS),
            "confirmation_roles": {
                key: confirmation_roles[key] for key in sorted(confirmation_roles)
            },
            "rows": expanded,
            "row_count": len(expanded),
            "deduplicated_configuration_count": len(roles_by_configuration),
            "warnings_are_non_gating": True,
        }
    )


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _required_replica(row: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "configuration_id",
        "run_id",
        "seed",
        "split",
        "deployable_accuracy",
        "deployable_cross_entropy",
        "recovery_status",
        "recovery_fraction",
        "oracle_gain",
        "deployed_parameters",
        "bundle_sha256",
        "privileged_claim_eligible",
        "pre_stage_g_deployable_eligible",
        "diagnostic",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"replica row is missing {sorted(missing)}")
    if row["split"] != "model_val_select":
        raise ValueError("winner selection may only read model_val_select")
    if int(row["seed"]) not in LOCKED_CONFIRMATION_SEEDS:
        raise ValueError("replica seed must be 101, 202, or 303")
    if not _finite(row["deployable_accuracy"]) or not _finite(
        row["deployable_cross_entropy"]
    ):
        raise FloatingPointError("selection metrics must be finite")
    if not 0.0 <= float(row["deployable_accuracy"]) <= 1.0:
        raise ValueError("deployable accuracy is outside [0,1]")
    if int(row["deployed_parameters"]) <= 0:
        raise ValueError("deployed parameter count must be positive")
    require_sha256("bundle_sha256", row["bundle_sha256"])
    for field in (
        "privileged_claim_eligible",
        "pre_stage_g_deployable_eligible",
        "diagnostic",
    ):
        if not isinstance(row[field], bool):
            raise ValueError(f"{field} must be boolean")
    return dict(row)


def aggregate_three_seed_configuration(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly seeds 101/202/303 and retain the median replica."""

    normalized = [_required_replica(row) for row in rows]
    if len(normalized) != 3:
        raise ValueError("configuration confirmation requires exactly three replicas")
    configuration_ids = {row["configuration_id"] for row in normalized}
    if len(configuration_ids) != 1:
        raise ValueError("cannot aggregate different configurations")
    seeds = {int(row["seed"]) for row in normalized}
    if seeds != set(LOCKED_CONFIRMATION_SEEDS):
        raise ValueError("confirmation seeds must be exactly 101, 202, and 303")
    for flag in (
        "privileged_claim_eligible",
        "pre_stage_g_deployable_eligible",
        "diagnostic",
    ):
        if len({row[flag] for row in normalized}) != 1:
            raise ValueError(f"configuration replicas disagree on {flag}")
    parameters = {int(row["deployed_parameters"]) for row in normalized}
    if len(parameters) != 1:
        raise ValueError("configuration replicas disagree on deployed parameters")
    ordered_replicas = sorted(normalized, key=lambda row: int(row["seed"]))
    accuracies = [float(row["deployable_accuracy"]) for row in ordered_replicas]
    cross_entropies = [
        float(row["deployable_cross_entropy"]) for row in ordered_replicas
    ]
    median_accuracy = float(median(accuracies))
    representative = min(
        ordered_replicas,
        key=lambda row: (
            abs(float(row["deployable_accuracy"]) - median_accuracy),
            float(row["deployable_cross_entropy"]),
            int(row["seed"]),
        ),
    )
    recoveries: list[float] = []
    recovery_defined = True
    for row in ordered_replicas:
        if (
            row["recovery_status"] != "finite"
            or not _finite(row["recovery_fraction"])
            or not _finite(row["oracle_gain"])
            or float(row["oracle_gain"]) <= 0.0
        ):
            recovery_defined = False
            break
        recoveries.append(float(row["recovery_fraction"]))
    aggregate = with_content_hash(
        {
            "contract": PARTICLE_VIEW_SEED_AGGREGATE_CONTRACT,
            "configuration_id": str(ordered_replicas[0]["configuration_id"]),
            "replicas": ordered_replicas,
            "seeds": list(LOCKED_CONFIRMATION_SEEDS),
            "mean_accuracy": sum(accuracies) / 3.0,
            "median_accuracy": median_accuracy,
            "sample_accuracy_std": (
                sum((value - sum(accuracies) / 3.0) ** 2 for value in accuracies)
                / 2.0
            )
            ** 0.5,
            "minimum_accuracy": min(accuracies),
            "mean_cross_entropy": sum(cross_entropies) / 3.0,
            "recovery_status": "finite" if recovery_defined else "undefined",
            "mean_recovery": (
                sum(recoveries) / 3.0 if recovery_defined else None
            ),
            "deployed_parameters": next(iter(parameters)),
            "registered_run_id": min(str(row["run_id"]) for row in normalized),
            "representative_seed": int(representative["seed"]),
            "representative_bundle_sha256": representative["bundle_sha256"],
            "privileged_claim_eligible": normalized[0][
                "privileged_claim_eligible"
            ],
            "pre_stage_g_deployable_eligible": normalized[0][
                "pre_stage_g_deployable_eligible"
            ],
            "diagnostic": normalized[0]["diagnostic"],
        }
    )
    return aggregate


def _aggregate_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    finite_recovery = (
        row.get("recovery_status") == "finite" and _finite(row.get("mean_recovery"))
    )
    return (
        -float(row["mean_accuracy"]),
        -float(row["median_accuracy"]),
        float(row["mean_cross_entropy"]),
        0 if finite_recovery else 1,
        -float(row["mean_recovery"]) if finite_recovery else 0.0,
        int(row["deployed_parameters"]),
        str(row["registered_run_id"]),
        str(row["configuration_id"]),
    )


def select_particle_view_winner_families(
    replica_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select distinct privileged, deployable, and diagnostic outcomes."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in replica_rows:
        groups[str(row.get("configuration_id"))].append(row)
    aggregates = [
        aggregate_three_seed_configuration(groups[configuration_id])
        for configuration_id in sorted(groups)
    ]
    privileged = [
        row
        for row in aggregates
        if row["privileged_claim_eligible"] and not row["diagnostic"]
    ]
    deployable = [
        row
        for row in aggregates
        if row["pre_stage_g_deployable_eligible"] and not row["diagnostic"]
    ]
    diagnostics = [row for row in aggregates if row["diagnostic"]]
    if not privileged or not deployable:
        raise ValueError("both privileged and pre-Stage-G pools must be nonempty")

    def select(pool: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        return dict(min(pool, key=_aggregate_order)) if pool else None

    privileged_winner = select(privileged)
    deployable_winner = select(deployable)
    diagnostic_winner = select(diagnostics)
    assert privileged_winner is not None and deployable_winner is not None
    winner_hashes = {
        privileged_winner["representative_bundle_sha256"],
        deployable_winner["representative_bundle_sha256"],
    }
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_WINNER_SELECTION_CONTRACT,
            "selection_split": "model_val_select",
            "selection_order": [
                "higher_three_seed_mean_accuracy",
                "higher_three_seed_median_accuracy",
                "lower_mean_cross_entropy",
                "finite_then_higher_recovery",
                "fewer_deployed_parameters",
                "lexicographically_smaller_run_id",
                "lexicographically_smaller_configuration_id",
            ],
            "selected_privileged_scientific_model": privileged_winner,
            "selected_pre_stage_g_hlt_deployable_model": deployable_winner,
            "best_diagnostic_control": diagnostic_winner,
            "aggregates": sorted(
                aggregates, key=lambda row: str(row["configuration_id"])
            ),
            "distinct_winner_bundle_count": len(winner_hashes),
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "minimum_quality_gate": None,
            "warnings_are_non_gating": True,
        }
    )


def _validate_selection(selection: Mapping[str, Any]) -> None:
    validate_content_hash(
        selection, expected_contract=PARTICLE_VIEW_WINNER_SELECTION_CONTRACT
    )
    if selection.get("selection_split") != "model_val_select":
        raise ValueError("selection artifact used the wrong split")
    if selection.get("stack_val_loaded") or selection.get("final_test_loaded"):
        raise ValueError("selection artifact leaked a sealed split")


def build_selected_path_fairness_ledger(
    *,
    selection: Mapping[str, Any],
    replica_training_ledgers: Mapping[str, Mapping[int, Mapping[str, Any]]],
    resource_profiles: Mapping[str, Mapping[int, Mapping[str, Any]]],
    train_identity_sha256: str,
    flop_fixture_sha256: str,
    flop_counter_sha256: str,
) -> dict[str, Any]:
    """Freeze exact post-selection budgets before any Stage-G control starts."""

    _validate_selection(selection)
    for name, value in (
        ("train_identity_sha256", train_identity_sha256),
        ("flop_fixture_sha256", flop_fixture_sha256),
        ("flop_counter_sha256", flop_counter_sha256),
    ):
        require_sha256(name, value)
    winner_keys = (
        "selected_privileged_scientific_model",
        "selected_pre_stage_g_hlt_deployable_model",
    )
    by_hash: dict[str, dict[str, Any]] = {}
    for family in winner_keys:
        winner = selection[family]
        configuration_id = str(winner["configuration_id"])
        family_ledgers = replica_training_ledgers.get(configuration_id)
        family_resources = resource_profiles.get(configuration_id)
        if family_ledgers is None or family_resources is None:
            raise ValueError(f"missing fairness inputs for {configuration_id}")
        entry_hash = str(winner["representative_bundle_sha256"])
        entry = by_hash.setdefault(
            entry_hash,
            {
                "configuration_id": configuration_id,
                "winner_families": [],
                "winner_bundle_sha256": entry_hash,
                "replicas": [],
            },
        )
        entry["winner_families"].append(family)
        if entry["replicas"]:
            continue
        for seed in LOCKED_CONFIRMATION_SEEDS:
            ledger = family_ledgers.get(seed)
            resource = family_resources.get(seed)
            if ledger is None or resource is None:
                raise ValueError("fairness closure requires all three seeds")
            validate_content_hash(ledger)
            validate_content_hash(resource)
            if ledger.get("train_identity_sha256") != train_identity_sha256:
                raise ValueError("fairness ledger uses the wrong train identities")
            retained = ledger.get("totals_retained_deployable_path")
            total = ledger.get("totals_all_training")
            if not isinstance(retained, Mapping) or not isinstance(total, Mapping):
                raise ValueError("training ledger has no auditable totals")
            parameters = int(resource["total_parameters"])
            flops_payload = resource.get("forward_flops")
            flops = (
                int(flops_payload["exact_integer_total"])
                if isinstance(flops_payload, Mapping)
                else int(resource["forward_flops"])
            )
            if parameters <= 0 or flops <= 0:
                raise ValueError("selected bundle resources must be positive")
            entry["replicas"].append(
                {
                    "seed": seed,
                    "training_ledger_sha256": ledger["content_hash"],
                    "resource_profile_sha256": resource["content_hash"],
                    "a0_view_long_deploy_exact_ce_updates": int(
                        retained["ce_bearing_steps"]
                    ),
                    "a0_view_total_label_budget_exact_updates": int(
                        total["label_bearing_steps"]
                    ),
                    "label_bearing_updates": int(total["label_bearing_steps"]),
                    "optimizer_updates_retained_path": int(
                        retained["optimizer_steps"]
                    ),
                    "training_flops_retained_path": int(retained["training_flops"]),
                    "deployed_parameters": parameters,
                    "forward_flops_128": flops,
                    "architecture_config_sha256": require_sha256(
                        "architecture_config_sha256",
                        resource["architecture_config_sha256"],
                    ),
                }
            )
    entries = []
    for bundle_hash in sorted(by_hash):
        entry = by_hash[bundle_hash]
        entry["winner_families"].sort()
        entry["replicas"].sort(key=lambda row: row["seed"])
        entry["fairness_entry_sha256"] = with_content_hash(
            {"contract": "particle_view_fairness_entry_v1", **entry}
        )["content_hash"]
        entries.append(entry)
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_FAIRNESS_LEDGER_CONTRACT,
            "selection_sha256": selection["content_hash"],
            "train_identity_sha256": train_identity_sha256,
            "flop_fixture_sha256": flop_fixture_sha256,
            "flop_counter_sha256": flop_counter_sha256,
            "entries": entries,
            "distinct_entry_count": len(entries),
            "replica_seeds": list(LOCKED_CONFIRMATION_SEEDS),
            "stage_g_controls_may_start": True,
            "published_before_stage_g": True,
        }
    )


def build_sealed_split_authorization(
    *,
    selection: Mapping[str, Any],
    fairness_ledger: Mapping[str, Any],
    stack_split_sha256: str,
    final_test_split_sha256: str,
    ce_only_comparator_bundles: Sequence[Mapping[str, Any]] = (),
    stage_g_control_bundles: Sequence[Mapping[str, Any]] = (),
    fusion_recipe_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    """Authorize stack rows broadly and final test only for median winners."""

    _validate_selection(selection)
    validate_content_hash(
        fairness_ledger, expected_contract=PARTICLE_VIEW_FAIRNESS_LEDGER_CONTRACT
    )
    if fairness_ledger.get("selection_sha256") != selection["content_hash"]:
        raise ValueError("fairness ledger belongs to a different selection")
    for name, value in (
        ("stack_split_sha256", stack_split_sha256),
        ("final_test_split_sha256", final_test_split_sha256),
    ):
        require_sha256(name, value)
    stack_rows: dict[str, dict[str, Any]] = {}
    final_rows: dict[str, dict[str, Any]] = {}
    for family in (
        "selected_privileged_scientific_model",
        "selected_pre_stage_g_hlt_deployable_model",
    ):
        winner = selection[family]
        for replica in winner["replicas"]:
            stack_rows.setdefault(
                replica["bundle_sha256"],
                {
                    "bundle_sha256": replica["bundle_sha256"],
                    "seed": replica["seed"],
                    "role": "preselected_winner_replica",
                    "winner_families": [],
                },
            )["winner_families"].append(family)
        final_rows.setdefault(
            winner["representative_bundle_sha256"],
            {
                "bundle_sha256": winner["representative_bundle_sha256"],
                "seed": winner["representative_seed"],
                "role": "preselected_median_winner",
                "winner_families": [],
            },
        )["winner_families"].append(family)
    for row in (*ce_only_comparator_bundles, *stage_g_control_bundles):
        bundle_hash = require_sha256("bundle_sha256", row.get("bundle_sha256"))
        seed = int(row.get("seed"))
        if seed not in LOCKED_CONFIRMATION_SEEDS:
            raise ValueError("control stack authorization uses an invalid seed")
        stack_rows.setdefault(
            bundle_hash,
            {
                "bundle_sha256": bundle_hash,
                "seed": seed,
                "role": str(row.get("role", "stage_g_control")),
                "winner_families": [],
            },
        )
    fusion_hashes = sorted(
        {require_sha256("fusion_recipe_sha256", value) for value in fusion_recipe_sha256}
    )
    for row in stack_rows.values():
        row["winner_families"].sort()
    for row in final_rows.values():
        row["winner_families"].sort()
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT,
            "selection_sha256": selection["content_hash"],
            "fairness_ledger_sha256": fairness_ledger["content_hash"],
            "stack_val": {
                "split_sha256": stack_split_sha256,
                "authorized_bundles": sorted(
                    stack_rows.values(),
                    key=lambda row: (row["bundle_sha256"], row["seed"]),
                ),
                "may_select_or_replace_winner": False,
            },
            "final_test": {
                "split_sha256": final_test_split_sha256,
                "authorized_bundles": sorted(
                    final_rows.values(), key=lambda row: row["bundle_sha256"]
                ),
                "authorized_fusion_recipes": fusion_hashes,
                "hlt_only_required": True,
                "stage_g_controls_forbidden": True,
            },
        }
    )


def assert_split_access(
    authorization: Mapping[str, Any],
    *,
    split: str,
    artifact_sha256: str,
    requires_offline: bool = False,
) -> None:
    validate_content_hash(
        authorization, expected_contract=PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT
    )
    if split not in {"stack_val", "final_test"}:
        raise ValueError("sealed authorization only governs stack_val/final_test")
    require_sha256("artifact_sha256", artifact_sha256)
    if split == "final_test" and requires_offline:
        raise PermissionError("final_test permits HLT-only inference")
    section = authorization[split]
    bundle_hashes = {
        row["bundle_sha256"] for row in section["authorized_bundles"]
    }
    recipe_hashes = set(section.get("authorized_fusion_recipes", []))
    if artifact_sha256 not in bundle_hashes | recipe_hashes:
        raise PermissionError(f"{artifact_sha256} is not authorized for {split}")


def build_stack_confirmation_report(
    *,
    selection: Mapping[str, Any],
    authorization: Mapping[str, Any],
    stack_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report Stage-G challenges without creating a second selection loop."""

    _validate_selection(selection)
    validate_content_hash(
        authorization, expected_contract=PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT
    )
    authorized = {
        row["bundle_sha256"]
        for row in authorization["stack_val"]["authorized_bundles"]
    }
    normalized: list[dict[str, Any]] = []
    for row in stack_rows:
        bundle_hash = require_sha256("bundle_sha256", row.get("bundle_sha256"))
        if bundle_hash not in authorized:
            raise PermissionError("stack metric supplied for an unauthorized bundle")
        if row.get("split") != "stack_val" or not _finite(row.get("accuracy")):
            raise ValueError("invalid stack-validation metric row")
        normalized.append(dict(row))
    summaries: dict[str, Any] = {}
    for family, registry_family in (
        (
            "selected_privileged_scientific_model",
            "PRIVILEGED_SCIENTIFIC",
        ),
        (
            "selected_pre_stage_g_hlt_deployable_model",
            "PRE_STAGE_G_DEPLOYABLE",
        ),
    ):
        winner = selection[family]
        winner_hashes = {row["bundle_sha256"] for row in winner["replicas"]}
        winner_values = [
            float(row["accuracy"])
            for row in normalized
            if row["bundle_sha256"] in winner_hashes
        ]
        if len(winner_values) != 3:
            raise ValueError("stack report requires all three winner replicas")
        winner_mean = sum(winner_values) / 3.0
        grouped_controls: dict[str, list[float]] = defaultdict(list)
        family_controls = [
            row
            for row in normalized
            if (
                row.get("role") == "stage_g_control"
                and row.get("winner_family") == registry_family
            )
        ]
        legacy_controls = [
            row
            for row in normalized
            if row.get("role") == "stage_g_control"
            and row.get("winner_family") is None
        ]
        if not family_controls and legacy_controls:
            # Backward-compatible report ingestion only. Production PV08
            # always writes winner_family and therefore takes the strict path.
            grouped_controls["legacy_stage_g_control"] = [
                float(row["accuracy"]) for row in legacy_controls
            ]
        for row in family_controls:
            parts = str(row["configuration_id"]).split("/", 2)
            if len(parts) != 3 or parts[0] != registry_family:
                raise ValueError("Stage-G control family identity changed")
            grouped_controls[parts[1]].append(
                float(row["accuracy"])
            )
        if family_controls and set(grouped_controls) != _EXPECTED_STAGE_G_CONTROLS:
            raise ValueError("stack report Stage-G control inventory is incomplete")
        if any(len(values) != 3 for values in grouped_controls.values()):
            raise ValueError(
                "stack report requires exactly three seeds per family/control"
            )
        control_means = {
            control_id: sum(values) / 3.0
            for control_id, values in grouped_controls.items()
        }
        strongest = (
            max(control_means, key=lambda key: (control_means[key], key))
            if control_means
            else None
        )
        delta = control_means[strongest] - winner_mean if strongest else None
        summaries[family] = {
            "winner_three_seed_mean_accuracy": winner_mean,
            "winner_sample_accuracy_std": (
                sum((value - winner_mean) ** 2 for value in winner_values) / 2.0
            )
            ** 0.5,
            "strongest_stage_g_control_id": strongest,
            "strongest_control_minus_winner_accuracy": delta,
            "post_stage_g_control_numerically_better": (
                bool(delta is not None and delta > 0.0)
            ),
        }
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_STACK_REPORT_CONTRACT,
            "selection_sha256": selection["content_hash"],
            "authorization_sha256": authorization["content_hash"],
            "split": "stack_val",
            "rows": normalized,
            "winner_summaries": summaries,
            "selection_changed": False,
            "warnings_are_non_gating": True,
        }
    )


__all__ = [
    "LOCKED_CONFIRMATION_SEEDS",
    "PARTICLE_VIEW_FAIRNESS_LEDGER_CONTRACT",
    "PARTICLE_VIEW_SEED_AGGREGATE_CONTRACT",
    "PARTICLE_VIEW_SEED_EXPANSION_CONTRACT",
    "PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT",
    "PARTICLE_VIEW_STACK_REPORT_CONTRACT",
    "PARTICLE_VIEW_WINNER_SELECTION_CONTRACT",
    "aggregate_three_seed_configuration",
    "expand_three_seed_confirmation_rows",
    "assert_split_access",
    "build_sealed_split_authorization",
    "build_selected_path_fairness_ledger",
    "build_stack_confirmation_report",
    "select_particle_view_winner_families",
]
