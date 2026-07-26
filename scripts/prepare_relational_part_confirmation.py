#!/usr/bin/env python3
"""Resolve confirmation contracts and the exact post-screening training tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
    GLOBAL_DETERMINISM_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    SCREENING_SUMMARY_CONTRACT,
    bind_source_provenance,
    build_confirmation_registry,
    build_selected_union_model_contract,
    build_step6_model_contract,
    load_hashed_json,
    source_snapshot,
    validate_campaign_source,
    with_content_hash,
    write_immutable_json,
)


CONFIRMATION_TASK_REGISTRY = "relational_part_confirmation_task_registry_v1"
ARCHITECTURE_RUNS = {
    "RPT_BASE_LAYERWISE",
    "RPT_BASE_EDGEVALUE",
    "RPT_SELECTED_LAYERWISE",
    "RPT_SELECTED_EDGEVALUE",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.campaign_root
    campaign = load_hashed_json(root / "campaign_spec.json")
    validate_campaign_source(campaign, repo_root=REPO_ROOT)
    output_dir = args.output_dir or root / "selection"
    screening = load_hashed_json(
        root / "registry" / "screening_registry.json",
        expected_contract=SCREENING_REGISTRY_CONTRACT,
    )
    architecture = load_hashed_json(
        root / "registry" / "confirmation_architecture_registry.json",
        expected_contract=CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
    )
    summary = load_hashed_json(
        output_dir / "screening_summary.json",
        expected_contract=SCREENING_SUMMARY_CONTRACT,
    )
    normalization = load_hashed_json(
        root / "inputs" / "relation_normalization.json",
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    determinism = load_hashed_json(
        root / "registry" / "global_determinism.json",
        expected_contract=GLOBAL_DETERMINISM_CONTRACT,
    )
    contract_dir = root / "registry" / "model_contracts"
    index = load_hashed_json(contract_dir / "index.json")
    source = source_snapshot(REPO_ROOT)

    def bind(value: dict[str, Any]) -> dict[str, Any]:
        return bind_source_provenance(value, source_snapshot=source)

    confirmation = bind(
        build_confirmation_registry(
            screening_registry=screening,
            architecture_registry=architecture,
            screening_summary=summary,
        )
    )
    family_hashes = dict(index["family_contract_sha256"])
    union_contract = None
    if confirmation["selected_union"]["synthesized"]:
        region_hash = (
            load_hashed_json(root / "inputs" / "region_normalization.json")[
                "content_hash"
            ]
            if "REGION" in confirmation["selected_union"]["families"]
            else None
        )
        union_contract = bind(
            build_selected_union_model_contract(
                confirmation_registry=confirmation,
                relation_normalization_sha256=normalization["content_hash"],
                relation_registry_sha256=screening[
                    "relation_registry_sha256"
                ],
                pair_base_sha256=index["pair_base_sha256"],
                family_contract_sha256=family_hashes,
                weaver_runtime_sha256=index["weaver_runtime_sha256"],
                global_determinism_sha256=determinism["content_hash"],
                region_normalization_sha256=region_hash,
            )
        )
    selected_reference_id = summary["best_available_run_id"]
    selected_reference = load_hashed_json(
        contract_dir / f"{selected_reference_id}.json"
    )
    base_reference = load_hashed_json(contract_dir / "RPT_BASE.json")
    architecture_contracts = {}
    for run_id in sorted(ARCHITECTURE_RUNS):
        selected = run_id.startswith("RPT_SELECTED_")
        architecture_contracts[run_id] = bind(
            build_step6_model_contract(
                run_id,
                selected_families=(
                    summary["selected_relation_set"] if selected else ()
                ),
                confirmation_architecture_registry=architecture,
                relation_normalization_artifact=normalization,
                selected_shared_bias_model_contract_sha256=(
                    selected_reference["content_hash"]
                    if selected
                    else base_reference["content_hash"]
                ),
                global_determinism_sha256=determinism["content_hash"],
            )
        )
    tasks = []
    for row in confirmation["rows"]:
        run_id = row["run_id"]
        if run_id in architecture_contracts:
            contract_path = (
                root
                / "registry"
                / "confirmation_model_contracts"
                / f"{run_id}.json"
            )
        elif run_id == "RPT_SELECTED_UNION":
            if union_contract is None:
                raise AssertionError("synthesized union contract is absent")
            contract_path = (
                root
                / "registry"
                / "confirmation_model_contracts"
                / "RPT_SELECTED_UNION.json"
            )
        else:
            contract_path = contract_dir / f"{run_id}.json"
        for seed in row["seeds"]:
            if (
                int(seed) == 101
                and row["seed_101"]["mode"] == "reuse_hash_exact"
            ):
                continue
            tasks.append(
                {
                    "task_index": len(tasks),
                    "run_id": run_id,
                    "seed": int(seed),
                    "new_relation_families": list(
                        row["new_relation_families"]
                    ),
                    "model_contract_path": str(contract_path.resolve()),
                    "mode": "train_from_scratch",
                }
            )
    task_registry = bind(
        with_content_hash(
            {
                "contract": CONFIRMATION_TASK_REGISTRY,
                "schema_version": 1,
                "confirmation_registry_sha256": confirmation["content_hash"],
                "tasks": tasks,
                "task_count": len(tasks),
                "seed_101_reuse_requires_exact_hash_match": True,
                "all_six_singles_have_202_303": True,
                "all_architecture_rows_have_101_202_303": True,
                "performance_gate": False,
            }
        )
    )
    publications = {}
    if not args.dry_run:
        publications["confirmation_registry"] = write_immutable_json(
            output_dir / "confirmation_registry.json", confirmation
        )
        model_output = root / "registry" / "confirmation_model_contracts"
        for run_id, artifact in architecture_contracts.items():
            publications[run_id] = write_immutable_json(
                model_output / f"{run_id}.json", artifact
            )
        if union_contract is not None:
            publications["RPT_SELECTED_UNION"] = write_immutable_json(
                model_output / "RPT_SELECTED_UNION.json", union_contract
            )
        publications["tasks"] = write_immutable_json(
            output_dir / "confirmation_tasks.json", task_registry
        )
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "confirmation_registry": confirmation,
                "task_registry": task_registry,
                "publications": publications,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
