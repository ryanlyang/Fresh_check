#!/usr/bin/env python3
"""Select the immutable Step-7 screening ranking from val-select records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    SCREENING_REGISTRY_CONTRACT,
    RUN_RESULT_ENVELOPE_CONTRACT,
    build_screening_summary,
    load_hashed_json,
    write_immutable_json,
    validate_campaign_source,
)
from teacher_logit_reco.relational_part.workflow import (  # noqa: E402
    expected_training_lineage,
    load_record_sequence,
    parse_named_hashes,
    reject_final_test_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-registry", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--hlt-cache-hash", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reject_final_test_paths(
        (args.screening_registry, args.results, args.campaign_spec, args.output)
    )
    registry = load_hashed_json(
        args.screening_registry, expected_contract=SCREENING_REGISTRY_CONTRACT
    )
    campaign = load_hashed_json(args.campaign_spec)
    validate_campaign_source(campaign, repo_root=REPO_ROOT)
    envelope = load_hashed_json(
        args.results, expected_contract=RUN_RESULT_ENVELOPE_CONTRACT
    )
    if (
        envelope.get("mode") != "screening"
        or envelope.get("registry_sha256") != registry["content_hash"]
        or envelope.get("campaign_spec_sha256") != campaign["content_hash"]
    ):
        raise ValueError("screening result envelope parent binding differs")
    results = load_record_sequence(args.results)
    hlt_hashes = parse_named_hashes(args.hlt_cache_hash)
    summary = build_screening_summary(
        screening_registry=registry,
        results=results,
        campaign_spec_sha256=campaign["content_hash"],
        split_manifest_sha256=args.split_manifest_sha256,
        hlt_cache_hashes=hlt_hashes,
        results_envelope_sha256=envelope["content_hash"],
        expected_common_lineage_hashes=expected_training_lineage(
            args.campaign_spec.parent, families=()
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.dry_run:
        write_immutable_json(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
