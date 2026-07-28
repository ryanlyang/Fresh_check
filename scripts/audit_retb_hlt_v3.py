#!/usr/bin/env python3
"""Audit an RETB HLT-v3 cache, transformed inputs, and inherited relations."""

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
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_audit import (  # noqa: E402
    REQUIRED_RELATION_FAMILIES,
    assert_layout_determinism,
    assert_train_scale_shared_identity,
    audit_strength_monotonicity,
    build_hlt_v3_degradation_audit,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_v3 import (  # noqa: E402
    build_hlt_v3_view,
    validate_hlt_v3_profile_contract,
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
    parser.add_argument("--offline-npz", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--relation-npz",
        type=Path,
        help=(
            "NPZ with FAMILY_offline/FAMILY_hlt arrays for standard_four, "
            "PT, TRACK, PID, CHARGE, DENSITY, REGION."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-boundary", action="append", type=int, default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    profile = load_hashed_json(
        args.campaign_root / "inputs" / "hlt_v3_profile.json"
    )
    profile_sha = validate_hlt_v3_profile_contract(profile)
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "profile_contract_sha256": profile_sha,
        "required_relation_families": list(REQUIRED_RELATION_FAMILIES),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.offline_npz is None or args.cache_dir is None or args.relation_npz is None:
        raise ValueError(
            "--offline-npz, --cache-dir, and --relation-npz are required"
        )
    if args.output is None:
        args.output = (
            args.campaign_root / "inputs" / "hlt_v3_degradation_audit.json"
        )
    with np.load(args.offline_npz, allow_pickle=False) as payload:
        offline_tokens = np.asarray(payload["tokens"])
        offline_mask = np.asarray(payload["mask"], dtype=bool)
        identities = [str(value) for value in payload["identities"].tolist()]
    arrays, metadata = load_hlt_v3_cache(
        args.cache_dir,
        expected_profile_contract_sha256=profile_sha,
    )
    if identities != [str(value) for value in arrays["identities"].tolist()]:
        raise ValueError("offline input identity order differs from HLT-v3 cache")
    generated = build_hlt_v3_view(
        offline_tokens,
        offline_mask,
        canonical_identities=identities,
        logical_role=metadata["logical_role"],
        replica_id=int(metadata["replica_id"]),
        realization_policy=metadata["realization_policy"],
        profile_id=metadata["degradation_profile_id"],
    )
    for expected, actual in zip(
        (arrays["tokens"], arrays["mask"], arrays["measurement_states"]),
        generated[:3],
    ):
        if not np.array_equal(expected, actual):
            raise ValueError("regenerated HLT-v3 view differs from cache")
    with np.load(args.relation_npz, allow_pickle=False) as payload:
        relation_views = {}
        for family in REQUIRED_RELATION_FAMILIES:
            before_name = f"{family}_offline"
            after_name = f"{family}_hlt"
            if before_name not in payload.files or after_name not in payload.files:
                raise ValueError(f"relation NPZ lacks {family} before/after tensors")
            relation_views[family] = (
                np.asarray(payload[before_name]),
                np.asarray(payload[after_name]),
            )
    boundaries = sorted(set(args.shard_boundary))
    layout = assert_layout_determinism(
        offline_tokens,
        offline_mask,
        canonical_identities=identities,
        logical_role=metadata["logical_role"],
        replica_id=int(metadata["replica_id"]),
        realization_policy=metadata["realization_policy"],
        profile_id=metadata["degradation_profile_id"],
        shard_boundaries=boundaries,
    )
    train_scale = assert_train_scale_shared_identity(
        offline_tokens[0],
        offline_mask[0],
        canonical_identity=identities[0],
        replica_id=int(metadata["replica_id"]),
        realization_policy=metadata["realization_policy"],
        profile_id=metadata["degradation_profile_id"],
    )
    monotonicity = audit_strength_monotonicity(
        offline_tokens,
        offline_mask,
        canonical_identities=identities,
        logical_role=metadata["logical_role"],
        replica_id=int(metadata["replica_id"]),
    )
    audit = build_hlt_v3_degradation_audit(
        offline_tokens=offline_tokens,
        offline_mask=offline_mask,
        hlt_tokens=arrays["tokens"],
        hlt_mask=arrays["mask"],
        measurement_states=arrays["measurement_states"],
        diagnostics=generated[3],
        relation_views=relation_views,
        profile_contract_sha256=profile_sha,
        cache_metadata_sha256=metadata["content_hash"],
        split_manifest_sha256=metadata["split_manifest_sha256"],
        identity_manifest_sha256=metadata["identity_manifest_sha256"],
        monotonicity=monotonicity,
        layout_determinism=layout,
        train_scale_equality=train_scale,
    )
    audit = bind_source(audit, source_snapshot=source_snapshot(REPO_ROOT))
    result["publication"] = write_immutable_json(args.output, audit)
    result["audit_sha256"] = audit["content_hash"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
