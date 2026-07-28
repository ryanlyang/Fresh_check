#!/usr/bin/env python3
"""Fit the immutable train-only per-slot bridge token normalizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.bridge_targets import (  # noqa: E402
    fit_bridge_token_normalizer,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (  # noqa: E402
    load_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--expert-id", required=True)
    parser.add_argument("--shape-id", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--token-cache-manifest", type=Path)
    inputs.add_argument("--tokens-npz", type=Path)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--target-registration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registration = load_hashed_json(args.target_registration)
    if (
        (
            registration.get("source") is not None
            and registration.get("source") != campaign.get("source")
        )
        or registration.get("expert_id") != args.expert_id
    ):
        raise ValueError("bridge normalizer source/target lineage differs")
    if args.token_cache_manifest is not None:
        cache, arrays = load_frozen_token_cache(args.token_cache_manifest)
        if (
            cache.get("source") != campaign.get("source")
            or cache["split"] != "model_train"
            or cache["shape_id"] != args.shape_id
            or cache["expert_checkpoint_hashes"][args.expert_id]
            != registration.get("checkpoint_sha256")
        ):
            raise ValueError("bridge normalizer frozen-cache lineage differs")
        tokens = arrays["token_banks"][args.expert_id]
        token_cache_sha256 = cache["content_hash"]
        identity_manifest_sha256 = cache["identity_manifest_sha256"]
    else:
        if args.identity_manifest is None:
            raise ValueError("--tokens-npz requires --identity-manifest")
        identity_manifest = load_hashed_json(args.identity_manifest)
        if identity_manifest.get("source") != campaign.get("source"):
            raise ValueError("bridge normalizer identity source differs")
        with np.load(args.tokens_npz, allow_pickle=False) as payload:
            if set(payload.files) != {"identities", "tokens"}:
                raise ValueError("bridge normalizer token NPZ fields differ")
            identities = tuple(
                str(value) for value in payload["identities"].tolist()
            )
            tokens = np.asarray(payload["tokens"], dtype=np.float32)
        if (
            not identities
            or len(identities) != len(set(identities))
            or len(identities) != len(tokens)
            or int(identity_manifest.get("event_count", -1)) != len(identities)
        ):
            raise ValueError("bridge normalizer token population differs")
        digest = hashlib.sha256()
        with args.tokens_npz.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        token_cache_sha256 = digest.hexdigest()
        identity_manifest_sha256 = identity_manifest["content_hash"]
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="model_train"
    )
    normalizer = bind_source(
        fit_bridge_token_normalizer(
            tokens,
            expert_id=args.expert_id,
            shape_id=args.shape_id,
            target_checkpoint_sha256=registration["checkpoint_sha256"],
            token_cache_sha256=token_cache_sha256,
            identity_manifest_sha256=identity_manifest_sha256,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": bool(args.dry_run),
        "normalizer": normalizer,
        "output": str(args.output.resolve()),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, normalizer)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
