#!/usr/bin/env python3
"""Publish authenticated model contracts for all 21 screening rows."""

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
    ANGULAR_TREE_RESOURCE_CONTRACT,
    GLOBAL_DETERMINISM_CONTRACT,
    RAW_INPUT_SCHEMA_CONTRACT,
    REGION_NORMALIZATION_CONTRACT,
    RELATION_FAMILY_REGISTRY_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    bind_source_provenance,
    build_density_relation_contract,
    build_pair_base_contract,
    build_pid_charge_relation_contract,
    build_pt_relation_contract,
    build_region_relation_contract,
    build_rpt_base_model_contract,
    build_step5_model_contract,
    build_track_relation_contract,
    load_hashed_json,
    select_wide_widths,
    source_snapshot,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


MODEL_CONTRACT_INDEX = "relational_part_screening_model_contract_index_v1"
PARITY_REPORT_CONTRACT = "relational_part_weaver_parity_report_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.campaign_root
    output_dir = args.output_dir or root / "registry" / "model_contracts"
    registry = load_hashed_json(
        root / "registry" / "relation_family_registry.json",
        expected_contract=RELATION_FAMILY_REGISTRY_CONTRACT,
    )
    screening = load_hashed_json(
        root / "registry" / "screening_registry.json",
        expected_contract=SCREENING_REGISTRY_CONTRACT,
    )
    determinism = load_hashed_json(
        root / "registry" / "global_determinism.json",
        expected_contract=GLOBAL_DETERMINISM_CONTRACT,
    )
    raw_schema = load_hashed_json(
        root / "inputs" / "raw_input_schema.json",
        expected_contract=RAW_INPUT_SCHEMA_CONTRACT,
    )
    normalization = load_hashed_json(
        root / "inputs" / "relation_normalization.json",
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    region_normalization = load_hashed_json(
        root / "inputs" / "region_normalization.json",
        expected_contract=REGION_NORMALIZATION_CONTRACT,
    )
    tree_resource = load_hashed_json(
        root / "inputs" / "angular_tree_resource_contract.json",
        expected_contract=ANGULAR_TREE_RESOURCE_CONTRACT,
    )
    parity = load_hashed_json(
        args.parity_report, expected_contract=PARITY_REPORT_CONTRACT
    )
    if (
        parity.get("ok") is not True
        or parity.get("dtype") != "float32"
        or parity.get("autocast_enabled") is not False
    ):
        raise ValueError("authoritative Weaver parity did not pass in plain FP32")
    parity_runtime = parity["weaver_runtime"]
    validate_content_hash(parity_runtime)
    source = source_snapshot(REPO_ROOT)

    def bind(value: dict[str, Any]) -> dict[str, Any]:
        return bind_source_provenance(value, source_snapshot=source)

    weaver_runtime = bind(parity_runtime)
    pair_base = bind(
        build_pair_base_contract(
            relation_registry_sha256=registry["content_hash"],
            global_determinism_sha256=determinism["content_hash"],
        )
    )
    # The parity report was produced before source binding; authenticate the
    # semantic pair contract by rebuilding and checking its pre-bind identity.
    if parity["pair_base_contract_sha256"] != build_pair_base_contract(
        relation_registry_sha256=registry["content_hash"],
        global_determinism_sha256=determinism["content_hash"],
    )["content_hash"]:
        raise ValueError("Weaver parity used another pair-base contract")
    pt = bind(
        build_pt_relation_contract(
            relation_registry_sha256=registry["content_hash"],
            relation_normalization_sha256=normalization["content_hash"],
        )
    )
    pid_charge = bind(
        build_pid_charge_relation_contract(
            relation_registry_sha256=registry["content_hash"],
            relation_normalization_sha256=normalization["content_hash"],
        )
    )
    track = bind(
        build_track_relation_contract(
            relation_registry_sha256=registry["content_hash"],
            relation_normalization_sha256=normalization["content_hash"],
            raw_input_schema_sha256=raw_schema["content_hash"],
        )
    )
    density = bind(
        build_density_relation_contract(
            relation_registry_sha256=registry["content_hash"],
            relation_normalization_sha256=normalization["content_hash"],
            track_relation_sha256=track["content_hash"],
        )
    )
    region = bind(
        build_region_relation_contract(
            relation_registry_sha256=registry["content_hash"],
            region_normalization_sha256=region_normalization["content_hash"],
            angular_tree_resource_sha256=tree_resource["content_hash"],
        )
    )
    capacity = bind(select_wide_widths())
    family_artifacts = {
        "PT": pt,
        "TRACK": track,
        "PID": pid_charge,
        "CHARGE": pid_charge,
        "DENSITY": density,
        "REGION": region,
    }
    family_hashes = {
        name: artifact["content_hash"]
        for name, artifact in family_artifacts.items()
    }
    contracts = {}
    for row in screening["rows"]:
        run_id = row["run_id"]
        if run_id == "RPT_BASE":
            contract = build_rpt_base_model_contract(
                pair_base_sha256=pair_base["content_hash"],
                weaver_runtime_sha256=weaver_runtime["content_hash"],
                global_determinism_sha256=determinism["content_hash"],
            )
        else:
            contract = build_step5_model_contract(
                run_id,
                normalization_artifact=normalization,
                screening_registry=screening,
                relation_registry_sha256=registry["content_hash"],
                pair_base_sha256=pair_base["content_hash"],
                family_contract_sha256=family_hashes,
                weaver_runtime_sha256=weaver_runtime["content_hash"],
                global_determinism_sha256=determinism["content_hash"],
                region_normalization_artifact=region_normalization,
                wide_capacity_artifact=(
                    capacity if run_id == "RPT_BASE_WIDE_MAX" else None
                ),
            )
        contracts[run_id] = bind(contract)
    artifacts = {
        "pair_base.json": pair_base,
        "weaver_runtime.json": weaver_runtime,
        "family_PT.json": pt,
        "family_PID_CHARGE.json": pid_charge,
        "family_TRACK.json": track,
        "family_DENSITY.json": density,
        "family_REGION.json": region,
        "wide_capacity.json": capacity,
        **{
            f"{run_id}.json": artifact
            for run_id, artifact in contracts.items()
        },
    }
    index = bind(
        with_content_hash(
            {
                "contract": MODEL_CONTRACT_INDEX,
                "schema_version": 1,
                "screening_registry_sha256": screening["content_hash"],
                "authoritative_parity_report_sha256": parity["content_hash"],
                "pair_base_sha256": pair_base["content_hash"],
                "weaver_runtime_sha256": weaver_runtime["content_hash"],
                "family_contract_sha256": family_hashes,
                "wide_capacity_sha256": capacity["content_hash"],
                "rows": [
                    {
                        "registry_index": index_value,
                        "run_id": row["run_id"],
                        "model_contract_sha256": contracts[row["run_id"]][
                            "content_hash"
                        ],
                        "path": f"{row['run_id']}.json",
                    }
                    for index_value, row in enumerate(screening["rows"])
                ],
                "row_count": 21,
                "complete": True,
            }
        )
    )
    artifacts["index.json"] = index
    publications = {}
    if not args.dry_run:
        for name, artifact in artifacts.items():
            publications[name] = write_immutable_json(
                output_dir / name, artifact
            )
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "output_dir": str(output_dir.resolve()),
                "index": index,
                "publications": publications,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
