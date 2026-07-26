"""Production PV09 reporting, HLT-only export, and final-permit runtime.

PV09 is deliberately split into separate export, fresh-process reload,
aggregation, and permit tasks.  Every task resolves its inputs from authenticated
runtime-task results.  No task opens ``final_test`` and no performance warning
can stop publication of the diagnostic report.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .campaign import _STACK_STATIC_IDS, _WINNER_FAMILIES
from .confirmation_runtime import _candidate_records, resolve_confirmation_role
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .deployment import (
    PARTICLE_VIEW_BUNDLE_INPUT_NAMES,
    build_particle_view_deployment_manifest,
    export_hlt_only_particle_view_bundle,
    run_fresh_process_reload_audit,
    write_hlt_reload_fixture,
)
from .distillation import module_state_sha256
from .post_target_runtime import _artifact, _task_artifacts
from .production_factories import _teacher_preprocessing_sha256
from .registry import validate_particle_view_registry
from .reporting import (
    PARTICLE_VIEW_REPORT_SECTIONS,
    build_final_test_permit,
    build_separated_campaign_report,
)
from .selection import (
    build_stack_confirmation_report,
)
from .stack_runtime import (
    PARTICLE_VIEW_OPTIONAL_FUSION_STATUS_CONTRACT,
    PARTICLE_VIEW_STACK_EVALUATION_CONTRACT,
    PARTICLE_VIEW_STACK_FUSION_RESULT_CONTRACT,
    _FAMILY_SELECTION_KEYS,
    _load_confirmation_source,
    _load_selection_and_authorization,
    _source_logits,
    _stack_context,
    _winner_replica,
    validate_stack_factory_config,
)


PARTICLE_VIEW_REPORT_FACTORY_CONFIG_CONTRACT = (
    "particle_view_report_factory_config_v1"
)
PARTICLE_VIEW_DEPLOYMENT_LINEAGE_CONTRACT = (
    "particle_view_selected_deployment_lineage_v1"
)
PARTICLE_VIEW_REPORT_PUBLICATION_CONTRACT = (
    "particle_view_report_publication_v1"
)
PARTICLE_VIEW_EXPORT_PUBLICATION_CONTRACT = (
    "particle_view_export_publication_v1"
)
PARTICLE_VIEW_RELOAD_PUBLICATION_CONTRACT = (
    "particle_view_reload_publication_v1"
)
PARTICLE_VIEW_PERMIT_FAMILY_BINDING_CONTRACT = (
    "particle_view_final_permit_family_binding_v1"
)

_EXPORT_RUN_FAMILIES = {
    "REPORT_EXPORT_PRIVILEGED_WINNER": "PRIVILEGED_SCIENTIFIC",
    "REPORT_EXPORT_DEPLOYABLE_WINNER": "PRE_STAGE_G_DEPLOYABLE",
}
_RELOAD_EXPORT_RUNS = {
    "REPORT_RELOAD_PRIVILEGED_WINNER": (
        "REPORT_EXPORT_PRIVILEGED_WINNER"
    ),
    "REPORT_RELOAD_DEPLOYABLE_WINNER": (
        "REPORT_EXPORT_DEPLOYABLE_WINNER"
    ),
}
_PERMIT_FAMILIES = {
    "REPORT_FINAL_PERMIT_PRIVILEGED": "PRIVILEGED_SCIENTIFIC",
    "REPORT_FINAL_PERMIT_DEPLOYABLE": "PRE_STAGE_G_DEPLOYABLE",
}
_DEPLOYMENT_FAMILIES = {
    "PRIVILEGED_SCIENTIFIC": "privileged_scientific",
    "PRE_STAGE_G_DEPLOYABLE": "pre_stage_g_deployable",
}
_FINAL_FUSION_IDS = {
    "PRIVILEGED_LOGIT_AVERAGE",
    "PRIVILEGED_LINEAR_FUSION",
    "DEPLOYABLE_LOGIT_AVERAGE",
    "DEPLOYABLE_LINEAR_FUSION",
}


def build_report_factory_config(
    *,
    stack_factory_config: Mapping[str, Any],
    source_commit: str,
    python_executable: str | Path = sys.executable,
    reload_fixture_batch_size: int = 8,
) -> dict[str, Any]:
    """Bind PV09 to the exact PV08 configuration and executable."""

    validate_stack_factory_config(stack_factory_config)
    executable = Path(python_executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError("PV09 Python executable is absent")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) not in {40, 64}
        or any(value not in "0123456789abcdef" for value in source_commit)
    ):
        raise ValueError("source_commit must be a 40- or 64-character hex digest")
    if int(reload_fixture_batch_size) <= 0:
        raise ValueError("reload fixture batch size must be positive")
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_REPORT_FACTORY_CONFIG_CONTRACT,
            "stack_factory_config": dict(stack_factory_config),
            "stack_factory_config_sha256": stack_factory_config["content_hash"],
            "runtime_data_config_sha256": stack_factory_config[
                "runtime_data_config_sha256"
            ],
            "source_commit": source_commit,
            "python_executable": str(executable),
            "python_executable_sha256": sha256_file(executable),
            "reload_fixture_batch_size": int(reload_fixture_batch_size),
            "winner_families": list(_WINNER_FAMILIES),
            "final_test_loaded": False,
            "oracle_dependencies_in_export": False,
            "fresh_process_reload_required": True,
            "performance_gates": False,
            "quality_warnings_stop_execution": False,
        }
    )
    validate_report_factory_config(artifact, verify_executable=True)
    return artifact


def validate_report_factory_config(
    payload: Mapping[str, Any], *, verify_executable: bool = False
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_REPORT_FACTORY_CONFIG_CONTRACT
    )
    expected = {
        "contract",
        "stack_factory_config",
        "stack_factory_config_sha256",
        "runtime_data_config_sha256",
        "source_commit",
        "python_executable",
        "python_executable_sha256",
        "reload_fixture_batch_size",
        "winner_families",
        "final_test_loaded",
        "oracle_dependencies_in_export",
        "fresh_process_reload_required",
        "performance_gates",
        "quality_warnings_stop_execution",
        "content_hash",
    }
    stack = payload["stack_factory_config"]
    validate_stack_factory_config(stack)
    executable = Path(payload["python_executable"])
    if (
        set(payload) != expected
        or payload["stack_factory_config_sha256"] != stack["content_hash"]
        or payload["runtime_data_config_sha256"]
        != stack["runtime_data_config_sha256"]
        or payload["winner_families"] != list(_WINNER_FAMILIES)
        or int(payload["reload_fixture_batch_size"]) <= 0
        or payload["final_test_loaded"] is not False
        or payload["oracle_dependencies_in_export"] is not False
        or payload["fresh_process_reload_required"] is not True
        or payload["performance_gates"] is not False
        or payload["quality_warnings_stop_execution"] is not False
        or not isinstance(payload["source_commit"], str)
        or len(payload["source_commit"]) not in {40, 64}
        or any(
            value not in "0123456789abcdef"
            for value in payload["source_commit"]
        )
    ):
        raise ValueError("PV09 report factory policy changed")
    if verify_executable and (
        not executable.is_file()
        or sha256_file(executable) != payload["python_executable_sha256"]
    ):
        raise ValueError("PV09 Python executable changed")
    return {"ok": True, "content_hash": payload["content_hash"]}


def _registry_seed_ids(
    registry: Mapping[str, Any], run_id: str
) -> tuple[int, ...]:
    row = next(row for row in registry["runs"] if row["run_id"] == run_id)
    return tuple(int(value) for value in row["seed_ids"])


def _collect_pv08(
    *,
    root: Path,
    registry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and combine every PV08 sibling output."""

    selection, base_authorization, _ = _load_selection_and_authorization(
        root=root,
        registry=registry,
        config=config["stack_factory_config"],
    )
    evaluation_artifacts = []
    row_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    fusion_rows = []
    fusion_recipe_hashes = []
    optional_status = []
    for registry_row in registry["runs"]:
        if registry_row["stage"] != "stack":
            continue
        run_id = registry_row["run_id"]
        for seed in _registry_seed_ids(registry, run_id):
            artifacts = _task_artifacts(root, registry, run_id, seed)
            if "stack_evaluation.json" in artifacts:
                path = _artifact(artifacts, "stack_evaluation.json")
                payload = load_hashed_json(path)
                validate_content_hash(
                    payload,
                    expected_contract=PARTICLE_VIEW_STACK_EVALUATION_CONTRACT,
                )
                evaluation_artifacts.append(payload)
                for row in payload["rows"]:
                    identity = (
                        row["bundle_sha256"],
                        int(row["seed"]),
                        row["role"],
                        row["configuration_id"],
                    )
                    previous = row_by_identity.get(identity)
                    if previous is not None and previous != row:
                        raise ValueError("PV08 emitted conflicting stack rows")
                    row_by_identity[identity] = dict(row)
            elif "fusion_recipe.json" in artifacts:
                recipe = load_hashed_json(
                    _artifact(artifacts, "fusion_recipe.json")
                )
                report = load_hashed_json(
                    _artifact(artifacts, "fusion_report.json")
                )
                result = load_hashed_json(
                    _artifact(artifacts, "stack_fusion_result.json")
                )
                validate_content_hash(
                    result,
                    expected_contract=PARTICLE_VIEW_STACK_FUSION_RESULT_CONTRACT,
                )
                if (
                    result["fusion_recipe_sha256"] != recipe["content_hash"]
                    or result["fusion_report_sha256"] != report["content_hash"]
                ):
                    raise ValueError("PV08 fusion lineage changed")
                row = {
                    "run_id": run_id,
                    "fusion_id": recipe["fusion_id"],
                    "recipe": recipe,
                    "report": report,
                    "result": result,
                }
                fusion_rows.append(row)
                if recipe["fusion_id"] in _FINAL_FUSION_IDS:
                    fusion_recipe_hashes.append(recipe["content_hash"])
            else:
                status = load_hashed_json(
                    _artifact(artifacts, "fusion_status.json")
                )
                validate_content_hash(
                    status,
                    expected_contract=PARTICLE_VIEW_OPTIONAL_FUSION_STATUS_CONTRACT,
                )
                optional_status.append(status)
    expected_stack_runs = {
        row["run_id"] for row in registry["runs"] if row["stage"] == "stack"
    }
    observed_stack_runs = {
        row["run_id"] for row in fusion_rows
    } | {
        run_id
        for run_id in expected_stack_runs
        if any(
            artifact["rows"]
            for artifact in evaluation_artifacts
            if any(
                item["configuration_id"]
                for item in artifact["rows"]
            )
        )
    }
    # Exact task-result loading above is the authoritative coverage check.  The
    # set is retained only as a human-readable inventory in the publication.
    del observed_stack_runs
    authorization_payload = {
        key: value
        for key, value in base_authorization.items()
        if key != "content_hash"
    }
    authorization_payload["final_test"] = dict(
        authorization_payload["final_test"]
    )
    authorization_payload["final_test"]["authorized_fusion_recipes"] = sorted(
        set(fusion_recipe_hashes)
    )
    final_baselines = {}
    for family in _WINNER_FAMILIES:
        winner = selection[_FAMILY_SELECTION_KEYS[family]]
        representative_sha = winner["representative_bundle_sha256"]
        representative_seed = int(winner["representative_seed"])
        matches = [
            row
            for row in row_by_identity.values()
            if row["role"] == "preselected_winner_replica"
            and row["bundle_sha256"] == representative_sha
            and int(row["seed"]) == representative_seed
        ]
        if len(matches) != 1:
            raise ValueError(
                "PV08 does not contain one matched-A0 row for a median winner"
            )
        baseline_sha = matches[0]["matched_a0_bundle_sha256"]
        baseline = final_baselines.setdefault(
            baseline_sha,
            {
                "bundle_sha256": baseline_sha,
                "seed": representative_seed,
                "role": "matched_a0_final_baseline",
                "winner_families": [],
            },
        )
        if int(baseline["seed"]) != representative_seed:
            raise ValueError("one A0 checkpoint was assigned conflicting seeds")
        baseline["winner_families"].append(
            _FAMILY_SELECTION_KEYS[family]
        )
    for row in final_baselines.values():
        row["winner_families"].sort()
    authorization_payload["final_test"][
        "authorized_hlt_baselines"
    ] = sorted(
        final_baselines.values(),
        key=lambda row: (row["bundle_sha256"], row["seed"]),
    )
    final_authorization = with_content_hash(authorization_payload)
    validate_content_hash(
        final_authorization,
        expected_contract="particle_view_split_authorization_v1",
    )
    return {
        "selection": selection,
        "base_authorization": base_authorization,
        "final_authorization": final_authorization,
        "stack_rows": list(row_by_identity.values()),
        "evaluation_artifacts": evaluation_artifacts,
        "fusions": fusion_rows,
        "optional_status": optional_status,
        "stack_run_count": len(expected_stack_runs),
    }


def _winner_export_resources(
    *,
    root: Path,
    registry: Mapping[str, Any],
    config: Mapping[str, Any],
    family: str,
    final_authorization: Mapping[str, Any],
    exemplar: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    selection = _collect_pv08(
        root=root, registry=registry, config=config
    )["selection"]
    winner = selection[_FAMILY_SELECTION_KEYS[family]]
    seed = int(winner["representative_seed"])
    replica = _winner_replica(selection, family, seed)
    source = _load_confirmation_source(
        root=root, registry=registry, replica=replica
    )
    resolved = resolve_confirmation_role(
        str(replica["role_id"]), _candidate_records(root, registry)
    )
    if replica["source_registration_sha256"] != resolved[
        "registration_sha256"
    ]:
        raise ValueError("selected replica source registration changed")
    confirmation_artifacts = _task_artifacts(
        root, registry, f"CONFIRM_{replica['role_id']}", seed
    )
    confirmation = load_hashed_json(
        _artifact(confirmation_artifacts, "confirmation_replica.json")
    )
    resource = load_hashed_json(
        _artifact(confirmation_artifacts, "resource_profile.json")
    )
    training_ledger = load_hashed_json(
        _artifact(confirmation_artifacts, "training_ledger.json")
    )
    if (
        confirmation["bundle_sha256"] != replica["bundle_sha256"]
        or resource["content_hash"] != replica["resource_profile_sha256"]
        or training_ledger["content_hash"]
        != replica["training_ledger_sha256"]
    ):
        raise ValueError("selected confirmation lineage changed")
    coordinate_artifacts = _task_artifacts(
        root, registry, "SELECTED_COORDINATE_BINDING", seed
    )
    coordinate = load_hashed_json(
        _artifact(coordinate_artifacts, "selected_view_coordinate.json")
    )
    normalizer = load_hashed_json(
        _artifact(coordinate_artifacts, "selected_view_normalizer.json")
    )
    if coordinate["parents"]["normalizer_sha256"] != normalizer["content_hash"]:
        raise ValueError("selected coordinate normalizer changed")
    predictor = source["predictor"]
    consumer = source["consumer"]
    predictor_config = predictor.config.to_payload()
    consumer_config = consumer.config.to_payload()
    data = config["stack_factory_config"]["fairness_factory_config"][
        "runtime_data_config"
    ]
    unified = load_hashed_json(data["unified_manifest"]["path"])
    fairness_artifacts = _task_artifacts(
        root, registry, "SELECTED_PATH_FAIRNESS_LEDGER", 101
    )
    fairness = load_hashed_json(
        _artifact(
            fairness_artifacts, "selected_path_fairness_ledger.json"
        )
    )
    input_schema_sha = canonical_sha256(
        {
            "contract": "particle_view_hlt_export_schema_v1",
            "input_names": list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES),
            "shapes": {
                name: list(exemplar[name].shape[1:])
                for name in PARTICLE_VIEW_BUNDLE_INPUT_NAMES
            },
            "dtypes": {
                name: str(exemplar[name].dtype).removeprefix("torch.")
                for name in PARTICLE_VIEW_BUNDLE_INPUT_NAMES
            },
            "pf_feature_names": list(PF_FEATURE_NAMES),
            "max_particles": 128,
        }
    )
    lineage = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DEPLOYMENT_LINEAGE_CONTRACT,
            "winner_family": family,
            "selection_sha256": selection["content_hash"],
            "split_authorization_sha256": final_authorization["content_hash"],
            "fairness_ledger_sha256": fairness["content_hash"],
            "confirmation_replica_sha256": confirmation["content_hash"],
            "confirmation_training_ledger_sha256": training_ledger[
                "content_hash"
            ],
            "resource_profile_sha256": resource["content_hash"],
            "source_registration_sha256": resolved["registration_sha256"],
            "source_bundle_sha256": replica["bundle_sha256"],
            "coordinate_binding_sha256": coordinate["content_hash"],
            "view_normalizer_sha256": normalizer["content_hash"],
            "predictor_config_sha256": canonical_sha256(predictor_config),
            "predictor_state_sha256": module_state_sha256(predictor),
            "consumer_config_sha256": canonical_sha256(consumer_config),
            "consumer_state_sha256": module_state_sha256(consumer),
            "unified_split_manifest_sha256": unified["content_hash"],
            "train_identity_sha256": unified["logical_splits"]["train"][
                "ordered_identity_sha256"
            ],
            "final_test_loaded": False,
            "oracle_dependencies_in_deployment": False,
        }
    )
    mode = str(resolved["campaign_row"]["mode"])
    role_id = str(replica["role_id"])
    bundle_kind = (
        "hlt_memory_control"
        if "HLT_MEMORY" in role_id
        else (
            "ce_only_control"
            if "CE_ONLY" in role_id
            else ("frozen_consumer" if mode == "frozen" else "joint_hlt_only")
        )
    )
    manifest = build_particle_view_deployment_manifest(
        bundle_id=f"{family.lower()}__{replica['bundle_sha256'][:16]}",
        bundle_kind=bundle_kind,
        winner_family=_DEPLOYMENT_FAMILIES[family],
        unified_split_manifest_sha256=unified["content_hash"],
        train_identity_sha256=unified["logical_splits"]["train"][
            "ordered_identity_sha256"
        ],
        predictor_config_sha256=canonical_sha256(predictor_config),
        predictor_checkpoint_sha256=module_state_sha256(predictor),
        consumer_config_sha256=canonical_sha256(consumer_config),
        consumer_checkpoint_sha256=module_state_sha256(consumer),
        hlt_preprocessing_sha256=_teacher_preprocessing_sha256("fixed_hlt"),
        hlt_schema_sha256=input_schema_sha,
        view_normalizer_sha256=normalizer["content_hash"],
        coordinate_binding_sha256=coordinate["content_hash"],
        resource_profile_sha256=resource["content_hash"],
        source_commit=config["source_commit"],
        bottleneck_width=int(consumer.config.view_dim),
        class_names=config["stack_factory_config"]["class_names"],
    )
    return {
        "predictor": predictor,
        "consumer": consumer,
        "source": source,
        "replica": replica,
        "predictor_config": predictor_config,
        "consumer_config": consumer_config,
        "deployment_manifest": manifest,
        "lineage": lineage,
        "fairness": fairness,
        "training_ledger": training_ledger,
    }


def run_report_bundle_export(
    *,
    output_dir: str,
    predictor: torch.nn.Module,
    consumer: torch.nn.Module,
    source_bundle_sha256: str,
    predictor_config: Mapping[str, Any],
    consumer_config: Mapping[str, Any],
    deployment_manifest: Mapping[str, Any],
    lineage: Mapping[str, Any],
    fairness_ledger_sha256: str,
    split_authorization: Mapping[str, Any],
    exemplar_hlt_inputs: Mapping[str, torch.Tensor],
) -> None:
    """Export one winner and a label-free fresh-process reload fixture."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_immutable_json(output / "deployment_manifest.json", deployment_manifest)
    write_immutable_json(output / "deployment_lineage.json", lineage)
    write_immutable_json(
        output / "sealed_split_authorization.json", split_authorization
    )
    export = export_hlt_only_particle_view_bundle(
        output,
        predictor=predictor,
        consumer=consumer,
        exemplar_hlt_inputs=exemplar_hlt_inputs,
        deployment_manifest=deployment_manifest,
        source_bundle_sha256=source_bundle_sha256,
        predictor_config=predictor_config,
        consumer_config=consumer_config,
        lineage_graph_sha256=lineage["content_hash"],
        fairness_ledger_sha256=fairness_ledger_sha256,
        split_authorization_sha256=split_authorization["content_hash"],
    )
    source = {
        "kind": "particle_view",
        "predictor": predictor.cpu().eval(),
        "consumer": consumer.cpu().eval(),
    }
    with torch.no_grad():
        reference = _source_logits(source, exemplar_hlt_inputs).cpu().float()
    fixture = write_hlt_reload_fixture(
        output,
        exemplar_hlt_inputs=exemplar_hlt_inputs,
        reference_logits=reference,
        bundle_export_sha256=export["content_hash"],
    )
    publication = with_content_hash(
        {
            "contract": PARTICLE_VIEW_EXPORT_PUBLICATION_CONTRACT,
            "deployment_manifest_sha256": deployment_manifest["content_hash"],
            "deployment_lineage_sha256": lineage["content_hash"],
            "bundle_export_sha256": export["content_hash"],
            "reload_fixture_sha256": fixture["content_hash"],
            "source_bundle_sha256": source_bundle_sha256,
            "requires_oracle": False,
            "final_test_loaded": False,
        }
    )
    write_immutable_json(output / "export_publication.json", publication)


def run_report_bundle_reload(
    *,
    bundle_manifest_path: str,
    fixture_manifest_path: str,
    output_path: str,
    publication_path: str,
    python_executable: str,
) -> None:
    audit = run_fresh_process_reload_audit(
        bundle_manifest_path=bundle_manifest_path,
        fixture_manifest_path=fixture_manifest_path,
        output_path=output_path,
        python_executable=python_executable,
    )
    write_immutable_json(
        publication_path,
        with_content_hash(
            {
                "contract": PARTICLE_VIEW_RELOAD_PUBLICATION_CONTRACT,
                "bundle_export_sha256": audit["bundle_export_sha256"],
                "fresh_reload_audit_sha256": audit["content_hash"],
                "passed": audit["passed"],
                "only_hlt_inputs_visible": audit["only_hlt_inputs_visible"],
                "final_test_loaded": False,
            }
        ),
    )


def _report_sections(
    *,
    selection: Mapping[str, Any],
    stack_rows: Sequence[Mapping[str, Any]],
    fusions: Sequence[Mapping[str, Any]],
    deployment_exports: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    sections = {name: [] for name in PARTICLE_VIEW_REPORT_SECTIONS}
    selected_modes = {}
    family_by_deployment_name = {
        value: key for key, value in _DEPLOYMENT_FAMILIES.items()
    }
    for export in deployment_exports:
        family = family_by_deployment_name[
            export["deployment_manifest"]["winner_family"]
        ]
        for replica in selection[_FAMILY_SELECTION_KEYS[family]]["replicas"]:
            selected_modes[replica["bundle_sha256"]] = export[
                "deployment_manifest"
            ]["bundle_kind"]
    oracle_seen = set()
    for family in _WINNER_FAMILIES:
        winner = selection[_FAMILY_SELECTION_KEYS[family]]
        representative = next(
            row
            for row in winner["replicas"]
            if int(row["seed"]) == int(winner["representative_seed"])
        )
        if (
            family == "PRIVILEGED_SCIENTIFIC"
            and representative["bundle_sha256"] not in oracle_seen
        ):
            oracle_seen.add(representative["bundle_sha256"])
            sections["privileged_true_view_diagnostics"].append(
                {
                    "row_id": (
                        f"{family.lower()}__model_val_select_oracle_context"
                    ),
                    "artifact_sha256": representative["bundle_sha256"],
                    "split": "model_val_select",
                    "metrics": {
                        "deployable_accuracy": representative[
                            "deployable_accuracy"
                        ],
                        "oracle_gain": representative["oracle_gain"],
                        "recovery_status": representative["recovery_status"],
                        "recovery_fraction": representative[
                            "recovery_fraction"
                        ],
                    },
                    "requires_oracle": True,
                    "deployable": False,
                }
            )
    seen = set()
    for row in stack_rows:
        key = (
            row["bundle_sha256"],
            int(row["seed"]),
            row["role"],
            row["configuration_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        payload = {
            "row_id": (
                f"{row['role']}__{row['configuration_id']}__seed_{row['seed']}"
            ),
            "artifact_sha256": row["bundle_sha256"],
            "split": "stack_val",
            "metrics": row["metrics"],
            "requires_oracle": False,
            "deployable": True,
        }
        if row["role"] == "preselected_winner_replica":
            section = (
                "frozen_consumer_hlt_deployable"
                if selected_modes.get(row["bundle_sha256"])
                == "frozen_consumer"
                else "joint_hlt_deployable"
            )
        else:
            section = "pre_stage_g_hlt_performance_controls"
        sections[section].append(payload)
    for row in fusions:
        sections["fusion_ensemble_results"].append(
            {
                "row_id": f"fusion__{row['fusion_id']}",
                "artifact_sha256": row["recipe"]["content_hash"],
                "split": "stack_val_evaluation",
                "metrics": row["report"]["metrics"],
                "requires_oracle": False,
                "deployable": True,
            }
        )
    return sections


def run_report_aggregate(
    *,
    output_dir: str,
    campaign_root: str,
    selection: Mapping[str, Any],
    final_authorization: Mapping[str, Any],
    fairness: Mapping[str, Any],
    stack_rows: Sequence[Mapping[str, Any]],
    fusions: Sequence[Mapping[str, Any]],
    optional_status: Sequence[Mapping[str, Any]],
    deployment_lineages: Sequence[Mapping[str, Any]],
    deployment_exports: Sequence[Mapping[str, Any]],
    reload_audits: Sequence[Mapping[str, Any]],
    selected_training_ledgers: Sequence[Mapping[str, Any]],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stack_report = build_stack_confirmation_report(
        selection=selection,
        authorization=final_authorization,
        stack_rows=stack_rows,
    )
    label_index = with_content_hash(
        {
            "contract": "particle_view_selected_label_exposure_index_v1",
            "selection_sha256": selection["content_hash"],
            "ledger_sha256": sorted(
                {row["content_hash"] for row in selected_training_ledgers}
            ),
            "final_test_loaded": False,
        }
    )
    lineage_index = with_content_hash(
        {
            "contract": "particle_view_selected_deployment_lineage_index_v1",
            "selection_sha256": selection["content_hash"],
            "deployment_lineage_sha256": sorted(
                {row["content_hash"] for row in deployment_lineages}
            ),
            "deployment_export_sha256": sorted(
                {row["content_hash"] for row in deployment_exports}
            ),
            "fresh_reload_audit_sha256": sorted(
                {row["content_hash"] for row in reload_audits}
            ),
            "final_test_loaded": False,
        }
    )
    campaign = Path(campaign_root)
    task_sizes = {
        path.name: sum(
            child.stat().st_size
            for child in path.rglob("*")
            if child.is_file()
        )
        for path in sorted((campaign / "runtime_tasks").glob("*"))
        if path.is_dir()
    }
    storage = with_content_hash(
        {
            "contract": "particle_view_measured_storage_snapshot_v1",
            "campaign_root": str(campaign.resolve()),
            "runtime_task_bytes": task_sizes,
            "total_runtime_task_bytes": sum(task_sizes.values()),
            "measurement": "recursive_file_size_at_pre_final_report",
            "offline_context_tokens_persisted": False,
        }
    )
    warnings = []
    for status in optional_status:
        if status.get("status") != "run":
            warnings.append(str(status.get("reason")))
    for family, summary in stack_report["winner_summaries"].items():
        if summary["post_stage_g_control_numerically_better"]:
            warnings.append(
                f"WARN_STAGE_G_CONTROL_NUMERICALLY_BETTER::{family}"
            )
    warning_index = with_content_hash(
        {
            "contract": "particle_view_pv09_warning_index_v1",
            "warnings": sorted(set(warnings)),
            "warning_count": len(set(warnings)),
            "warnings_are_non_gating": True,
        }
    )
    sections = _report_sections(
        selection=selection,
        stack_rows=stack_rows,
        fusions=fusions,
        deployment_exports=deployment_exports,
    )
    report = build_separated_campaign_report(
        sections=sections,
        selection_sha256=selection["content_hash"],
        stack_report_sha256=stack_report["content_hash"],
        fairness_ledger_sha256=fairness["content_hash"],
        label_exposure_ledger_sha256=label_index["content_hash"],
        storage_reservation_sha256=storage["content_hash"],
        lineage_graph_sha256=lineage_index["content_hash"],
        deployment_export_sha256=[
            row["content_hash"] for row in deployment_exports
        ],
        aggregate_warning_summary_sha256=warning_index["content_hash"],
    )
    for name, payload in (
        ("sealed_split_authorization.json", final_authorization),
        ("stack_confirmation_report.json", stack_report),
        ("selected_label_exposure_index.json", label_index),
        ("selected_deployment_lineage_index.json", lineage_index),
        ("measured_storage_snapshot.json", storage),
        ("quality_warning_index.json", warning_index),
        ("pre_final_campaign_report.json", report),
    ):
        write_immutable_json(output / name, payload)
    write_immutable_json(
        output / "report_publication.json",
        with_content_hash(
            {
                "contract": PARTICLE_VIEW_REPORT_PUBLICATION_CONTRACT,
                "selection_sha256": selection["content_hash"],
                "authorization_sha256": final_authorization["content_hash"],
                "stack_report_sha256": stack_report["content_hash"],
                "campaign_report_sha256": report["content_hash"],
                "deployment_export_sha256": sorted(
                    row["content_hash"] for row in deployment_exports
                ),
                "fresh_reload_audit_sha256": sorted(
                    row["content_hash"] for row in reload_audits
                ),
                "final_test_loaded": False,
                "warnings_are_non_gating": True,
            }
        ),
    )


def run_report_final_permit(
    *,
    output_dir: str,
    permit_family: str,
    selection: Mapping[str, Any],
    split_authorization: Mapping[str, Any],
    stack_report: Mapping[str, Any],
    bundle_manifest_paths: Sequence[str],
    fresh_reload_audits: Sequence[Mapping[str, Any]],
) -> None:
    """Publish the common permit plus an explicit family binding."""

    permit = build_final_test_permit(
        selection=selection,
        split_authorization=split_authorization,
        stack_report=stack_report,
        bundle_export_manifests=bundle_manifest_paths,
        fresh_reload_audits=fresh_reload_audits,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_immutable_json(output / "final_test_permit.json", permit)
    write_immutable_json(
        output / "permit_family_binding.json",
        with_content_hash(
            {
                "contract": PARTICLE_VIEW_PERMIT_FAMILY_BINDING_CONTRACT,
                "permit_family": permit_family,
                "final_test_permit_sha256": permit["content_hash"],
                "authorized_export_count": permit[
                    "authorized_export_count"
                ],
                "hlt_only_required": True,
                "selection_changed": False,
            }
        ),
    )


def _export_and_reload_artifacts(
    root: Path, registry: Mapping[str, Any]
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_paths = []
    lineages = []
    audits = []
    for export_run, reload_run in (
        (
            "REPORT_EXPORT_PRIVILEGED_WINNER",
            "REPORT_RELOAD_PRIVILEGED_WINNER",
        ),
        (
            "REPORT_EXPORT_DEPLOYABLE_WINNER",
            "REPORT_RELOAD_DEPLOYABLE_WINNER",
        ),
    ):
        export_artifacts = _task_artifacts(root, registry, export_run, 101)
        manifest_paths.append(
            str(_artifact(export_artifacts, "particle_view_bundle_manifest.json"))
        )
        lineages.append(
            load_hashed_json(
                _artifact(export_artifacts, "deployment_lineage.json")
            )
        )
        reload_artifacts = _task_artifacts(root, registry, reload_run, 101)
        audits.append(
            load_hashed_json(
                _artifact(reload_artifacts, "fresh_reload_audit.json")
            )
        )
    return manifest_paths, lineages, audits


def _canonical_permit_exports(
    manifest_paths: Sequence[str],
    audits: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Deduplicate a shared winner source without losing its reload audit."""

    records = []
    audit_by_export = {
        row["bundle_export_sha256"]: dict(row) for row in audits
    }
    for path in manifest_paths:
        payload = load_hashed_json(path)
        records.append(
            {
                "path": path,
                "payload": payload,
                "source": payload["source_bundle_sha256"],
                "family": payload["deployment_manifest"]["winner_family"],
            }
        )
    selected = []
    for source in sorted({row["source"] for row in records}):
        candidates = [row for row in records if row["source"] == source]
        chosen = min(
            candidates,
            key=lambda row: (
                0
                if row["family"] == "privileged_scientific"
                else 1,
                row["payload"]["content_hash"],
            ),
        )
        audit = audit_by_export.get(chosen["payload"]["content_hash"])
        if audit is None:
            raise ValueError("canonical deployment export has no reload audit")
        selected.append((chosen["path"], audit))
    return [row[0] for row in selected], [row[1] for row in selected]


def build_report_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_report_factory_config(config, verify_executable=True)
    validate_particle_view_registry(registry)
    if task_id != f"{run_id}__seed_{int(seed)}" or int(seed) != 101:
        raise ValueError("PV09 task identity changed")
    output = Path(output_dir).resolve()
    root = output.parent.parent
    if run_id in _EXPORT_RUN_FAMILIES:
        if operation != "bundle_export":
            raise ValueError("PV09 export operation changed")
        pv08 = _collect_pv08(root=root, registry=registry, config=config)
        _, _, _, aligned, loader = _stack_context(
            config=config["stack_factory_config"],
            registry=registry,
            output=output,
        )
        del aligned
        batch = next(iter(loader))
        limit = min(
            int(config["reload_fixture_batch_size"]),
            int(batch["mask"].shape[0]),
        )
        exemplar = {
            name: batch[name][:limit].detach().cpu().contiguous()
            for name in PARTICLE_VIEW_BUNDLE_INPUT_NAMES
        }
        resources = _winner_export_resources(
            root=root,
            registry=registry,
            config=config,
            family=_EXPORT_RUN_FAMILIES[run_id],
            final_authorization=pv08["final_authorization"],
            exemplar=exemplar,
        )
        return {
            "kwargs": {
                "output_dir": str(output),
                "predictor": resources["predictor"],
                "consumer": resources["consumer"],
                "source_bundle_sha256": resources["replica"][
                    "bundle_sha256"
                ],
                "predictor_config": resources["predictor_config"],
                "consumer_config": resources["consumer_config"],
                "deployment_manifest": resources["deployment_manifest"],
                "lineage": resources["lineage"],
                "fairness_ledger_sha256": resources["fairness"][
                    "content_hash"
                ],
                "split_authorization": pv08["final_authorization"],
                "exemplar_hlt_inputs": exemplar,
            },
            "artifact_paths": [
                str(output / "deployment_manifest.json"),
                str(output / "deployment_lineage.json"),
                str(output / "sealed_split_authorization.json"),
                str(output / "particle_view_bundle.ts"),
                str(output / "particle_view_bundle_manifest.json"),
                str(output / "hlt_reload_fixture.npz"),
                str(output / "hlt_reload_fixture.json"),
                str(output / "export_publication.json"),
            ],
            "action": None,
        }
    if run_id in _RELOAD_EXPORT_RUNS:
        if operation != "bundle_reload":
            raise ValueError("PV09 reload operation changed")
        artifacts = _task_artifacts(
            root, registry, _RELOAD_EXPORT_RUNS[run_id], 101
        )
        return {
            "kwargs": {
                "bundle_manifest_path": str(
                    _artifact(
                        artifacts, "particle_view_bundle_manifest.json"
                    )
                ),
                "fixture_manifest_path": str(
                    _artifact(artifacts, "hlt_reload_fixture.json")
                ),
                "output_path": str(output / "fresh_reload_audit.json"),
                "publication_path": str(output / "reload_publication.json"),
                "python_executable": config["python_executable"],
            },
            "artifact_paths": [
                str(output / "fresh_reload_audit.json"),
                str(output / "reload_publication.json"),
            ],
            "action": None,
        }
    if run_id == "REPORT_AGGREGATE_REPORT":
        if operation != "reporting":
            raise ValueError("PV09 aggregate operation changed")
        pv08 = _collect_pv08(root=root, registry=registry, config=config)
        manifest_paths, lineages, audits = _export_and_reload_artifacts(
            root, registry
        )
        exports = [load_hashed_json(path) for path in manifest_paths]
        ledgers = []
        for family in _WINNER_FAMILIES:
            winner = pv08["selection"][_FAMILY_SELECTION_KEYS[family]]
            replica = _winner_replica(
                pv08["selection"], family, winner["representative_seed"]
            )
            artifacts = _task_artifacts(
                root,
                registry,
                f"CONFIRM_{replica['role_id']}",
                int(replica["seed"]),
            )
            ledgers.append(
                load_hashed_json(
                    _artifact(artifacts, "training_ledger.json")
                )
            )
        fairness_artifacts = _task_artifacts(
            root, registry, "SELECTED_PATH_FAIRNESS_LEDGER", 101
        )
        fairness = load_hashed_json(
            _artifact(
                fairness_artifacts, "selected_path_fairness_ledger.json"
            )
        )
        return {
            "kwargs": {
                "output_dir": str(output),
                "campaign_root": str(root),
                "selection": pv08["selection"],
                "final_authorization": pv08["final_authorization"],
                "fairness": fairness,
                "stack_rows": pv08["stack_rows"],
                "fusions": pv08["fusions"],
                "optional_status": pv08["optional_status"],
                "deployment_lineages": lineages,
                "deployment_exports": exports,
                "reload_audits": audits,
                "selected_training_ledgers": ledgers,
            },
            "artifact_paths": [
                str(output / "sealed_split_authorization.json"),
                str(output / "stack_confirmation_report.json"),
                str(output / "selected_label_exposure_index.json"),
                str(output / "selected_deployment_lineage_index.json"),
                str(output / "measured_storage_snapshot.json"),
                str(output / "quality_warning_index.json"),
                str(output / "pre_final_campaign_report.json"),
                str(output / "report_publication.json"),
            ],
            "action": None,
        }
    if run_id in _PERMIT_FAMILIES:
        if operation != "reporting":
            raise ValueError("PV09 permit operation changed")
        aggregate = _task_artifacts(
            root, registry, "REPORT_AGGREGATE_REPORT", 101
        )
        selection_artifacts = _task_artifacts(
            root, registry, "SELECT_WINNER_FAMILIES", 101
        )
        selection = load_hashed_json(
            _artifact(selection_artifacts, "winner_selection.json")
        )
        authorization = load_hashed_json(
            _artifact(aggregate, "sealed_split_authorization.json")
        )
        stack_report = load_hashed_json(
            _artifact(aggregate, "stack_confirmation_report.json")
        )
        manifests, _, audits = _export_and_reload_artifacts(root, registry)
        manifests, audits = _canonical_permit_exports(manifests, audits)
        return {
            "kwargs": {
                "output_dir": str(output),
                "permit_family": _PERMIT_FAMILIES[run_id],
                "selection": selection,
                "split_authorization": authorization,
                "stack_report": stack_report,
                "bundle_manifest_paths": manifests,
                "fresh_reload_audits": audits,
            },
            "artifact_paths": [
                str(output / "final_test_permit.json"),
                str(output / "permit_family_binding.json"),
            ],
            "action": None,
        }
    raise ValueError("unknown PV09 report/export run")


def build_report_task_specs(
    *, factory_config_path: str | Path
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_report_factory_config(
        load_hashed_json(path), verify_executable=False
    )
    common = {
        "factory": (
            "teacher_logit_reco.local_particle_residual_field."
            "particle_view.report_runtime:build_report_factory"
        ),
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    return {
        "REPORT_AGGREGATE_REPORT": {**common, "operation": "reporting"},
        "REPORT_EXPORT_PRIVILEGED_WINNER": {
            **common,
            "operation": "bundle_export",
        },
        "REPORT_EXPORT_DEPLOYABLE_WINNER": {
            **common,
            "operation": "bundle_export",
        },
        "REPORT_RELOAD_PRIVILEGED_WINNER": {
            **common,
            "operation": "bundle_reload",
        },
        "REPORT_RELOAD_DEPLOYABLE_WINNER": {
            **common,
            "operation": "bundle_reload",
        },
        "REPORT_FINAL_PERMIT_PRIVILEGED": {
            **common,
            "operation": "reporting",
        },
        "REPORT_FINAL_PERMIT_DEPLOYABLE": {
            **common,
            "operation": "reporting",
        },
    }


__all__ = [
    "PARTICLE_VIEW_REPORT_FACTORY_CONFIG_CONTRACT",
    "build_report_factory",
    "build_report_factory_config",
    "build_report_task_specs",
    "run_report_aggregate",
    "run_report_bundle_export",
    "run_report_bundle_reload",
    "run_report_final_permit",
    "validate_report_factory_config",
]
