#!/usr/bin/env python3
"""Build an authenticated RETB HLT-v3 cache from label-free offline tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    build_hlt_v3_cache,
    build_hlt_v3_cache_metadata,
    publish_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_v3 import (  # noqa: E402
    build_hlt_v3_profile_contract,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    select_normalization_jet_indices,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--offline-npz",
        type=Path,
        help="NPZ containing tokens[B,N,14], mask[B,N], identities[B].",
    )
    parser.add_argument("--logical-role", required=True)
    parser.add_argument("--replica-id", required=True, type=int)
    parser.add_argument(
        "--realization-policy",
        choices=("R_FIXED", "R_MULTI", "R_RANDOM"),
        default="R_MULTI",
    )
    parser.add_argument("--profile-id", default="D_NOMINAL")
    parser.add_argument("--identity-manifest-sha256", required=True)
    parser.add_argument("--raw-input-sha256", required=True)
    parser.add_argument(
        "--selected-jet-limit",
        type=int,
        help=(
            "Deterministically retain only the normalization population; "
            "used exclusively by the streamed A-C profile."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    parent = campaign["parent_artifact_hashes"]
    profile = bind_source(
        build_hlt_v3_profile_contract(
            raw_input_schema_sha256=parent["raw_input_schema"],
            hlt_replica_manifest_sha256=parent["hlt_replica_manifest"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    output_dir = args.output_dir or (
        args.campaign_root
        / "inputs"
        / f"replica_{args.replica_id}"
        / args.logical_role
        / args.realization_policy
        / args.profile_id
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "output_dir": str(output_dir.resolve()),
        "profile_contract_sha256": profile["content_hash"],
        "logical_role": args.logical_role,
        "replica_id": int(args.replica_id),
        "realization_policy": args.realization_policy,
        "profile_id": args.profile_id,
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.offline_npz is None:
        raise ValueError("--offline-npz is required unless --dry-run is used")
    with np.load(args.offline_npz, allow_pickle=False) as payload:
        required = {"tokens", "mask", "identities"}
        if not required.issubset(payload.files):
            raise ValueError(f"offline NPZ must contain {sorted(required)}")
        tokens = np.asarray(payload["tokens"])
        mask = np.asarray(payload["mask"], dtype=bool)
        identities = [str(value) for value in payload["identities"].tolist()]
    if args.selected_jet_limit is not None:
        if args.selected_jet_limit <= 0:
            raise ValueError("--selected-jet-limit must be positive")
        selected = select_normalization_jet_indices(
            identities,
            limit=min(args.selected_jet_limit, len(identities)),
        )
        tokens = np.asarray(tokens[selected])
        mask = np.asarray(mask[selected], dtype=bool)
        identities = [identities[int(index)] for index in selected]
    arrays, diagnostics = build_hlt_v3_cache(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role=args.logical_role,
        replica_id=args.replica_id,
        realization_policy=args.realization_policy,
        profile_id=args.profile_id,
    )
    metadata = build_hlt_v3_cache_metadata(
        arrays=arrays,
        diagnostics=diagnostics,
        logical_role=args.logical_role,
        replica_id=args.replica_id,
        realization_policy=args.realization_policy,
        degradation_profile_id=args.profile_id,
        profile_contract=profile,
        split_manifest_sha256=parent["split_manifest"],
        identity_manifest_sha256=args.identity_manifest_sha256,
        raw_input_sha256=args.raw_input_sha256,
    )
    metadata = bind_source(
        metadata, source_snapshot=source_snapshot(REPO_ROOT)
    )
    profile_publication = write_immutable_json(
        args.campaign_root / "inputs" / "hlt_v3_profile.json", profile
    )
    result["profile_publication"] = profile_publication
    result["cache_publication"] = publish_hlt_v3_cache(
        output_dir, arrays=arrays, metadata=metadata
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
