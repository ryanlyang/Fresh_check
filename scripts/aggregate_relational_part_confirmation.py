#!/usr/bin/env python3
"""Aggregate confirmation and write the final-test seal after all controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    CONFIRMATION_REGISTRY_CONTRACT,
    RUN_RESULT_ENVELOPE_CONTRACT,
    SEMANTIC_PERTURBATION_CONTRACT,
    UNARY_CONTROL_REGISTRY_CONTRACT,
    aggregate_confirmation,
    load_hashed_json,
    write_immutable_json,
    validate_campaign_source,
    validate_semantic_diagnostics_bundle,
)
from teacher_logit_reco.relational_part.workflow import (  # noqa: E402
    expected_training_lineage,
    load_record_sequence,
    parse_named_hashes,
    reject_final_test_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-registry", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--unary-results", type=Path)
    parser.add_argument("--semantic-perturbations", type=Path)
    parser.add_argument("--unary-registry", type=Path)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--hlt-cache-hash", action="append", default=[])
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = [
            args.confirmation_registry,
            args.results,
            args.summary_output,
    ]
    paths.extend(
        value
        for value in (
            args.unary_results,
            args.semantic_perturbations,
            args.unary_registry,
            args.lock_output,
        )
        if value is not None
    )
    reject_final_test_paths(paths)
    confirmation = load_hashed_json(
        args.confirmation_registry,
        expected_contract=CONFIRMATION_REGISTRY_CONTRACT,
    )
    if args.summary_only:
        semantic = unary = None
    else:
        if None in (
            args.unary_results,
            args.semantic_perturbations,
            args.unary_registry,
            args.lock_output,
        ):
            raise ValueError(
                "final sealing requires unary results, both semantic artifacts, "
                "and --lock-output"
            )
        semantic = load_hashed_json(
            args.semantic_perturbations,
            expected_contract=SEMANTIC_PERTURBATION_CONTRACT,
        )
        unary = load_hashed_json(
            args.unary_registry,
            expected_contract=UNARY_CONTROL_REGISTRY_CONTRACT,
        )
    campaign = load_hashed_json(args.campaign_spec)
    validate_campaign_source(campaign, repo_root=REPO_ROOT)
    results_envelope = load_hashed_json(
        args.results, expected_contract=RUN_RESULT_ENVELOPE_CONTRACT
    )
    if (
        results_envelope.get("mode") != "confirmation"
        or results_envelope.get("registry_sha256")
        != confirmation["content_hash"]
        or results_envelope.get("campaign_spec_sha256")
        != campaign["content_hash"]
    ):
        raise ValueError("confirmation result envelope parent binding differs")
    unary_results_envelope = (
        None
        if args.summary_only
        else load_hashed_json(
            args.unary_results,
            expected_contract=RUN_RESULT_ENVELOPE_CONTRACT,
        )
    )
    if unary_results_envelope is not None and (
        unary_results_envelope.get("mode") != "unary"
        or unary_results_envelope.get("registry_sha256")
        != unary["content_hash"]
        or unary_results_envelope.get("campaign_spec_sha256")
        != campaign["content_hash"]
    ):
        raise ValueError("unary result envelope parent binding differs")
    summary, lock = aggregate_confirmation(
        confirmation_registry=confirmation,
        results=load_record_sequence(args.results),
        campaign_spec_sha256=campaign["content_hash"],
        split_manifest_sha256=args.split_manifest_sha256,
        hlt_cache_hashes=parse_named_hashes(args.hlt_cache_hash),
        results_envelope_sha256=results_envelope["content_hash"],
        expected_common_lineage_hashes=expected_training_lineage(
            args.campaign_spec.parent, families=()
        ),
        semantic_unary_results=(
            () if args.summary_only else load_record_sequence(args.unary_results)
        ),
        unary_results_envelope_sha256=(
            None
            if unary_results_envelope is None
            else unary_results_envelope["content_hash"]
        ),
        semantic_perturbation_sha256=(
            None if semantic is None else semantic["content_hash"]
        ),
        unary_control_registry_sha256=(
            None if unary is None else unary["content_hash"]
        ),
        seal_finalists=not args.summary_only,
    )
    if semantic is not None:
        validate_semantic_diagnostics_bundle(
            semantic, confirmation_summary=summary
        )
    if semantic is not None and semantic.get(
        "confirmation_summary_sha256"
    ) != summary["content_hash"]:
        raise ValueError("semantic controls belong to another confirmation")
    if unary is not None and unary.get(
        "confirmation_summary_sha256"
    ) != summary["content_hash"]:
        raise ValueError("unary registry belongs to another confirmation")
    resolved = {"confirmation_summary": summary, "locked_finalists": lock}
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if not args.dry_run:
        if args.summary_output.is_file():
            registered_summary = load_hashed_json(args.summary_output)
            if registered_summary["content_hash"] != summary["content_hash"]:
                raise ValueError("registered confirmation summary drifted")
        else:
            write_immutable_json(args.summary_output, summary)
        if lock is not None:
            write_immutable_json(args.lock_output, lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
