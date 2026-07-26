#!/usr/bin/env python3
"""Build the unary semantic-control registry from confirmation summary only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    CONFIRMATION_SUMMARY_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    bind_source_provenance,
    build_unary_control_registry,
    build_unary_model_contract,
    load_hashed_json,
    source_snapshot,
    write_immutable_json,
    validate_campaign_source,
)


_ARCHITECTURE = {
    "RPT_SELECTED_LAYERWISE",
    "RPT_SELECTED_EDGEVALUE",
    "RPT_BASE_LAYERWISE",
    "RPT_BASE_EDGEVALUE",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root
    campaign = load_hashed_json(root / "campaign_spec.json")
    validate_campaign_source(campaign, repo_root=REPO_ROOT)
    output = (
        args.output_dir
        or root / "selection" / "semantic_controls"
    )
    summary = load_hashed_json(
        root / "selection" / "confirmation_summary.json",
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    normalization = load_hashed_json(
        root / "inputs" / "relation_normalization.json",
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    rows = {row["run_id"]: row for row in summary["rows"]}
    winner_id = summary["nominal_relational_winner_id"]
    winner = rows[winner_id]
    families = tuple(winner["new_relation_families"])
    if winner_id in _ARCHITECTURE:
        references = [
            row
            for row in rows.values()
            if row["run_id"] not in _ARCHITECTURE
            and tuple(row["new_relation_families"]) == families
            and row["configuration_role"] == "scientific_finalist"
        ]
        if len(references) != 1:
            raise ValueError(
                "architecture winner does not resolve one ordinary shared-bias "
                "unary reference"
            )
        reference = references[0]
    else:
        reference = winner
    base = rows["RPT_BASE"]
    registry = build_unary_control_registry(
        nominal_winner_run_id=winner_id,
        unary_reference_run_id=reference["run_id"],
        families=families,
        reference_incremental_parameters=(
            int(reference["parameter_count"]) - int(base["parameter_count"])
        ),
        reference_total_parameters=int(reference["parameter_count"]),
        base_total_parameters=int(base["parameter_count"]),
        confirmation_summary_sha256=summary["content_hash"],
        relation_normalization_sha256=normalization["content_hash"],
    )
    base_contract = load_hashed_json(
        root / "registry" / "model_contracts" / "RPT_BASE.json"
    )
    source = source_snapshot(REPO_ROOT)
    registry = bind_source_provenance(registry, source_snapshot=source)
    # Rebuild after source binding because the model contract authenticates the
    # exact registry consumed by the trainer.
    model_contract = bind_source_provenance(
        build_unary_model_contract(
            unary_registry_sha256=registry["content_hash"],
            base_model_contract_sha256=base_contract["content_hash"],
            relation_normalization_sha256=normalization["content_hash"],
        ),
        source_snapshot=source,
    )
    publications = {}
    if not args.dry_run:
        publications["registry"] = write_immutable_json(
            output / "unary_control_registry.json", registry
        )
        publications["model_contract"] = write_immutable_json(
            output / "unary_model_contract.json", model_contract
        )
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "unary_registry": registry,
                "unary_model_contract": model_contract,
                "publications": publications,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
