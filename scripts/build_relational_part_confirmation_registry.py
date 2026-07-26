#!/usr/bin/env python3
"""Build the fixed matched-seed confirmation registry after screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    SCREENING_SUMMARY_CONTRACT,
    build_confirmation_registry,
    build_selected_union_model_contract,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.workflow import (  # noqa: E402
    parse_named_hashes,
    reject_final_test_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-registry", type=Path, required=True)
    parser.add_argument("--architecture-registry", type=Path, required=True)
    parser.add_argument("--screening-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-union-contract-output", type=Path)
    parser.add_argument("--relation-normalization-sha256")
    parser.add_argument("--relation-registry-sha256")
    parser.add_argument("--pair-base-sha256")
    parser.add_argument("--family-contract-sha256", action="append", default=[])
    parser.add_argument("--weaver-runtime-sha256")
    parser.add_argument("--global-determinism-sha256")
    parser.add_argument("--region-normalization-sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reject_final_test_paths(
        (
            args.screening_registry,
            args.architecture_registry,
            args.screening_summary,
            args.output,
        )
    )
    registry = build_confirmation_registry(
        screening_registry=load_hashed_json(
            args.screening_registry,
            expected_contract=SCREENING_REGISTRY_CONTRACT,
        ),
        architecture_registry=load_hashed_json(
            args.architecture_registry,
            expected_contract=CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
        ),
        screening_summary=load_hashed_json(
            args.screening_summary,
            expected_contract=SCREENING_SUMMARY_CONTRACT,
        ),
    )
    union_contract = None
    if (
        args.selected_union_contract_output is not None
        and registry["selected_union"]["synthesized"]
    ):
        required = (
            args.relation_normalization_sha256,
            args.relation_registry_sha256,
            args.pair_base_sha256,
            args.weaver_runtime_sha256,
            args.global_determinism_sha256,
        )
        if any(value is None for value in required):
            raise ValueError(
                "synthesized union contract output requires all parent hashes"
            )
        union_contract = build_selected_union_model_contract(
            confirmation_registry=registry,
            relation_normalization_sha256=args.relation_normalization_sha256,
            relation_registry_sha256=args.relation_registry_sha256,
            pair_base_sha256=args.pair_base_sha256,
            family_contract_sha256=parse_named_hashes(
                args.family_contract_sha256
            ),
            weaver_runtime_sha256=args.weaver_runtime_sha256,
            global_determinism_sha256=args.global_determinism_sha256,
            region_normalization_sha256=args.region_normalization_sha256,
        )
    print(json.dumps(
        {
            "confirmation_registry": registry,
            "selected_union_model_contract": union_contract,
        },
        indent=2,
        sort_keys=True,
    ))
    if not args.dry_run:
        write_immutable_json(args.output, registry)
        if union_contract is not None:
            write_immutable_json(
                args.selected_union_contract_output, union_contract
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
