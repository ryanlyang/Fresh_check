#!/usr/bin/env python3
"""Authenticate offline cache bytes and exact identity alignment to the HLT parent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import load_cached_hlt_view, jet_identity_hash  # noqa: E402
from teacher_logit_reco.architecture_view_part.train import load_cached_offline_view  # noqa: E402
from teacher_logit_reco.relational_part.contracts import load_hashed_json, with_content_hash, write_immutable_json  # noqa: E402
from teacher_logit_reco.relational_part.offline_transfer import (  # noqa: E402
    OFFLINE_TRANSFER_CACHE_BINDING_CONTRACT,
    validate_offline_transfer_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--parent-hlt-cache", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    validate_offline_transfer_campaign(campaign)
    reports = {}
    for split in ("model_train", "model_val", "stack_val", "final_test"):
        offline = load_cached_offline_view(root / "inputs" / "offline_cache", split, verify_hash=True)
        hlt = load_cached_hlt_view(args.parent_hlt_cache, split, verify_hash=True)
        offline_ids = [identity.key() for identity in offline.jet_ids]
        hlt_ids = [identity.key() for identity in hlt.jet_ids]
        if offline_ids != hlt_ids:
            raise ValueError(f"{split} offline identities/order differ from parent HLT")
        if not (offline.labels == hlt.labels).all():
            raise ValueError(f"{split} offline labels differ from parent HLT")
        reports[split] = {
            "event_count": len(offline_ids),
            "event_identity_sha256": jet_identity_hash(offline.jet_ids),
            "offline_content_sha256": str(offline.metadata["offline_content_hash"]),
            "parent_hlt_content_sha256": str(hlt.metadata["hlt_content_hash"]),
            "labels_exact": True,
            "identity_order_exact": True,
        }
    artifact = with_content_hash(
        {
            "contract": OFFLINE_TRANSFER_CACHE_BINDING_CONTRACT,
            "schema_version": 1,
            "campaign_sha256": campaign["content_hash"],
            "input_view": "offline",
            "parent_hlt_cache": str(args.parent_hlt_cache.resolve()),
            "splits": reports,
            "all_split_identities_and_labels_exact": True,
        }
    )
    write_immutable_json(root / "inputs" / "offline_cache_binding.json", artifact)
    print(artifact["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
