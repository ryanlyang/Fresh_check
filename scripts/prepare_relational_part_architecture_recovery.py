#!/usr/bin/env python3
"""Authorize the source-versioned repair of the 12 failed Step-6 RPT tasks."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    CONFIRMATION_REGISTRY_CONTRACT,
    GLOBAL_DETERMINISM_CONTRACT,
    RECOVERED_ARCHITECTURE_RUN_IDS,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    SOURCE_RECOVERY_AUTHORIZATION_CONTRACT,
    bind_source_provenance,
    build_confirmation_architecture_registry,
    build_confirmation_architecture_model,
    build_step6_model_contract,
    expected_training_lineage,
    load_hashed_json,
    load_run_result,
    source_snapshot,
    with_content_hash,
    write_immutable_json,
)


RECOVERY_TASK_CONTRACT = (
    "relational_part_confirmation_architecture_recovery_tasks_v1"
)
REAL_WEAVER_RECOVERY_PREFLIGHT_CONTRACT = (
    "relational_part_real_weaver_architecture_recovery_preflight_v1"
)
ORIGINAL_CONFIRMATION_TASK_CONTRACT = (
    "relational_part_confirmation_task_registry_v1"
)


def _source_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "commit": snapshot["source_commit"],
        "status_sha256": snapshot["source_status_sha256"],
        "dirty": bool(snapshot["source_dirty"]),
    }


def _require_descendant(original_commit: str, recovery_commit: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", original_commit, recovery_commit],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("recovery source is not a descendant of campaign source")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    current = source_snapshot(REPO_ROOT)
    if current["source_dirty"]:
        raise ValueError("architecture recovery must be prepared from clean source")
    original_source = campaign.get("source")
    if not isinstance(original_source, dict):
        raise ValueError("campaign source snapshot is absent")
    _require_descendant(str(original_source["commit"]), current["source_commit"])

    confirmation = load_hashed_json(
        root / "selection" / "confirmation_registry.json",
        expected_contract=CONFIRMATION_REGISTRY_CONTRACT,
    )
    original_tasks = load_hashed_json(root / "selection" / "confirmation_tasks.json")
    if original_tasks.get("contract") != ORIGINAL_CONFIRMATION_TASK_CONTRACT:
        raise ValueError("original confirmation task registry differs")
    screening = load_hashed_json(root / "registry" / "screening_registry.json")
    relation_registry = load_hashed_json(
        root / "registry" / "relation_family_registry.json"
    )
    normalization = load_hashed_json(
        root / "inputs" / "relation_normalization.json",
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    determinism = load_hashed_json(
        root / "registry" / "global_determinism.json",
        expected_contract=GLOBAL_DETERMINISM_CONTRACT,
    )
    summary = load_hashed_json(root / "selection" / "screening_summary.json")

    recovery_root = root / "selection" / "architecture_recovery_v1"
    registry_path = recovery_root / "confirmation_architecture_registry.json"
    contract_root = recovery_root / "model_contracts"
    task_path = recovery_root / "architecture_tasks.json"
    preflight_path = recovery_root / "real_weaver_construction_preflight.json"
    authorization_path = recovery_root / "source_recovery_authorization.json"

    architecture_registry = bind_source_provenance(
        build_confirmation_architecture_registry(
            relation_registry_sha256=relation_registry["content_hash"],
            screening_registry_sha256=screening["content_hash"],
        ),
        source_snapshot=current,
    )
    selected_reference = load_hashed_json(
        root
        / "registry"
        / "model_contracts"
        / f"{summary['best_available_run_id']}.json"
    )
    base_reference = load_hashed_json(
        root / "registry" / "model_contracts" / "RPT_BASE.json"
    )
    corrected_contracts = {}
    for run_id in sorted(RECOVERED_ARCHITECTURE_RUN_IDS):
        selected = run_id.startswith("RPT_SELECTED_")
        corrected_contracts[run_id] = bind_source_provenance(
            build_step6_model_contract(
                run_id,
                selected_families=(
                    summary["selected_relation_set"] if selected else ()
                ),
                confirmation_architecture_registry=architecture_registry,
                relation_normalization_artifact=normalization,
                selected_shared_bias_model_contract_sha256=(
                    selected_reference["content_hash"]
                    if selected
                    else base_reference["content_hash"]
                ),
                global_determinism_sha256=determinism["content_hash"],
            ),
            source_snapshot=current,
        )

    region_normalization = load_hashed_json(
        root / "inputs" / "region_normalization.json"
    )
    construction_rows = []
    for run_id in sorted(RECOVERED_ARCHITECTURE_RUN_IDS):
        selected = run_id.startswith("RPT_SELECTED_")
        model = build_confirmation_architecture_model(
            run_id,
            selected_families=(
                summary["selected_relation_set"] if selected else ()
            ),
            normalization_artifact=normalization,
            region_normalization_artifact=region_normalization,
        )
        tails = []
        parameter_id_sets = []
        for projection in model.layer_bias.projections:
            module_names = [
                child.__class__.__name__ for child in projection.children()
            ]
            if not module_names:
                module_names = [projection.__class__.__name__]
            if module_names[0] not in {"Conv1d", "Linear"}:
                raise RuntimeError("real Weaver projection tail has no leading head")
            if "BatchNorm1d" not in module_names[1:]:
                raise RuntimeError(
                    "real Weaver recovery did not capture trailing BatchNorm1d"
                )
            tails.append(module_names)
            parameter_id_sets.append({id(value) for value in projection.parameters()})
        if any(
            left & right
            for index, left in enumerate(parameter_id_sets)
            for right in parameter_id_sets[index + 1 :]
        ):
            raise RuntimeError("layerwise projection tails share parameters")
        construction_rows.append(
            {
                "run_id": run_id,
                "model_contract_sha256": corrected_contracts[run_id][
                    "content_hash"
                ],
                "projection_count": len(model.layer_bias.projections),
                "projection_tail_module_types": tails,
                "all_projection_parameters_independent": True,
                "trainable_parameter_count": sum(
                    value.numel() for value in model.parameters() if value.requires_grad
                ),
            }
        )
        del model
        gc.collect()
    real_weaver_preflight = bind_source_provenance(
        with_content_hash(
            {
                "contract": REAL_WEAVER_RECOVERY_PREFLIGHT_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign["content_hash"],
                "corrected_architecture_registry_sha256": architecture_registry[
                    "content_hash"
                ],
                "rows": construction_rows,
                "row_count": 4,
                "real_weaver_import_and_construction_passed": True,
                "trailing_BatchNorm1d_captured_per_layer": True,
            }
        ),
        source_snapshot=current,
    )

    tasks = []
    confirmation_rows = {row["run_id"]: row for row in confirmation["rows"]}
    for run_id in sorted(RECOVERED_ARCHITECTURE_RUN_IDS):
        row = confirmation_rows[run_id]
        for seed in (101, 202, 303):
            tasks.append(
                {
                    "task_index": len(tasks),
                    "run_id": run_id,
                    "seed": seed,
                    "new_relation_families": list(row["new_relation_families"]),
                    "model_contract_path": str(
                        (contract_root / f"{run_id}.json").resolve()
                    ),
                    "mode": "train_from_scratch",
                }
            )
    task_registry = bind_source_provenance(
        with_content_hash(
            {
                "contract": RECOVERY_TASK_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign["content_hash"],
                "confirmation_registry_sha256": confirmation["content_hash"],
                "original_confirmation_tasks_sha256": original_tasks["content_hash"],
                "corrected_architecture_registry_sha256": architecture_registry[
                    "content_hash"
                ],
                "real_weaver_construction_preflight_sha256": real_weaver_preflight[
                    "content_hash"
                ],
                "tasks": tasks,
                "task_count": 12,
                "ordinary_confirmation_tasks_reused": True,
                "performance_gate": False,
            }
        ),
        source_snapshot=current,
    )

    ordinary_registrations = {}
    screening_registry_sha = screening["content_hash"]
    confirmation_registry_sha = confirmation["content_hash"]
    for row in confirmation["rows"]:
        run_id = str(row["run_id"])
        if run_id in RECOVERED_ARCHITECTURE_RUN_IDS:
            continue
        model_contract_candidates = [
            root / "registry" / "model_contracts" / f"{run_id}.json",
            root / "registry" / "confirmation_model_contracts" / f"{run_id}.json",
        ]
        matches = [path for path in model_contract_candidates if path.is_file()]
        if len(matches) != 1:
            raise ValueError(f"ordinary run {run_id} model contract is ambiguous")
        model_contract = load_hashed_json(matches[0])
        for seed in (101, 202, 303):
            expected_registry = confirmation_registry_sha
            if seed == 101 and row.get("seed_101", {}).get("mode") == "reuse_hash_exact":
                expected_registry = screening_registry_sha
            result = load_run_result(
                root / "runs" / run_id / f"seed_{seed}",
                expected_lineage=expected_training_lineage(
                    root, families=row["new_relation_families"]
                ),
                expected_run_registry_sha256=expected_registry,
                expected_relation_registry_sha256=relation_registry["content_hash"],
                expected_model_contract_sha256=model_contract["content_hash"],
            )
            ordinary_registrations[f"{run_id}:seed_{seed}"] = result[
                "checkpoint_registration_sha256"
            ]
    if len(ordinary_registrations) != 39:
        raise ValueError("recovery must authenticate exactly 39 reusable ordinary runs")

    corrected_mapping = {
        run_id: {
            "path": str(
                Path("selection")
                / "architecture_recovery_v1"
                / "model_contracts"
                / f"{run_id}.json"
            ),
            "sha256": corrected_contracts[run_id]["content_hash"],
        }
        for run_id in sorted(RECOVERED_ARCHITECTURE_RUN_IDS)
    }
    authorization = bind_source_provenance(
        with_content_hash(
            {
                "contract": SOURCE_RECOVERY_AUTHORIZATION_CONTRACT,
                "schema_version": 1,
                "campaign_root": str(root),
                "campaign_spec_sha256": campaign["content_hash"],
                "original_campaign_source": dict(original_source),
                "recovery_source": _source_identity(current),
                "reason": (
                    "Weaver_PairEmbed_final_head_projection_precedes_trailing_"
                    "BatchNorm1d"
                ),
                "semantic_change": (
                    "duplicate_the_complete_final_projection_tail_per_layer"
                ),
                "authorized_run_ids": sorted(RECOVERED_ARCHITECTURE_RUN_IDS),
                "authorized_seeds": [101, 202, 303],
                "corrected_architecture_registry_sha256": architecture_registry[
                    "content_hash"
                ],
                "corrected_model_contracts": corrected_mapping,
                "recovery_task_registry_sha256": task_registry["content_hash"],
                "real_weaver_construction_preflight_sha256": real_weaver_preflight[
                    "content_hash"
                ],
                "reused_ordinary_checkpoint_registration_hashes": dict(
                    sorted(ordinary_registrations.items())
                ),
                "reused_ordinary_run_seed_count": 39,
                "retrain_ordinary_runs": False,
                "downstream_continuation_authorized": True,
                "final_test_still_requires_locked_finalists": True,
                "performance_gate": False,
            }
        ),
        source_snapshot=current,
    )

    publications = {}
    if not args.dry_run:
        publications["architecture_registry"] = write_immutable_json(
            registry_path, architecture_registry
        )
        for run_id, artifact in corrected_contracts.items():
            publications[run_id] = write_immutable_json(
                contract_root / f"{run_id}.json", artifact
            )
        publications["tasks"] = write_immutable_json(task_path, task_registry)
        publications["real_weaver_preflight"] = write_immutable_json(
            preflight_path, real_weaver_preflight
        )
        publications["authorization"] = write_immutable_json(
            authorization_path, authorization
        )
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "recovery_task_count": len(tasks),
                "reused_ordinary_run_seed_count": len(ordinary_registrations),
                "authorization": authorization,
                "publications": publications,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
