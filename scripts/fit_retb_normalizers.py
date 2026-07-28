#!/usr/bin/env python3
"""Freeze RETB offline/shared-HLT normalizer populations and fit lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_v3 import (  # noqa: E402
    validate_hlt_v3_profile_contract,
)
from teacher_logit_reco.relation_expert_token_bridge.normalizer_lineage import (  # noqa: E402
    build_normalizer_population_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--model-train-identity-count", type=int, default=500_000)
    parser.add_argument("--scale-train-identity-count", type=int, default=3_000_000)
    parser.add_argument("--inherited-estimator-contract-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    profile = load_hashed_json(
        args.campaign_root / "inputs" / "hlt_v3_profile.json"
    )
    profile_sha = validate_hlt_v3_profile_contract(profile)
    parents = campaign["parent_artifact_hashes"]
    registry = build_normalizer_population_registry(
        model_train_manifest_sha256=parents["split_manifest"],
        model_train_identity_count=args.model_train_identity_count,
        scale_train_manifest_sha256=parents["scale_train_manifest"],
        scale_train_identity_count=args.scale_train_identity_count,
        raw_input_schema_sha256=parents["raw_input_schema"],
        hlt_v3_profile_sha256=profile_sha,
        inherited_estimator_contract_sha256=(
            args.inherited_estimator_contract_sha256
        ),
    )
    registry = bind_source(
        registry, source_snapshot=source_snapshot(REPO_ROOT)
    )
    output = args.output or (
        args.campaign_root
        / "inputs"
        / "normalizer_population_registry.json"
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "output": str(output.resolve()),
        "registry_sha256": registry["content_hash"],
        "recipe_hashes": registry["recipe_hashes"],
        "fit_execution": (
            "delegated_to_inherited_deterministic_streaming_estimator"
        ),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(output, registry)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
