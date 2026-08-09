#!/usr/bin/env python3
"""Fit non-tree relation normalization from offline model_train only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.architecture_view_part.train import load_cached_offline_view  # noqa: E402
from teacher_logit_reco.relational_part.contracts import load_hashed_json, write_immutable_json  # noqa: E402
from teacher_logit_reco.relational_part.normalization import fit_relation_normalization  # noqa: E402
from teacher_logit_reco.relational_part.offline_transfer import OFFLINE_TRANSFER_CACHE_BINDING_CONTRACT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    binding = load_hashed_json(
        root / "inputs" / "offline_cache_binding.json",
        expected_contract=OFFLINE_TRANSFER_CACHE_BINDING_CONTRACT,
    )
    view = load_cached_offline_view(root / "inputs" / "offline_cache", "model_train", verify_hash=True)
    expected = binding["splits"]["model_train"]["offline_content_sha256"]
    if view.metadata["offline_content_hash"] != expected:
        raise ValueError("offline model_train differs from cache binding")
    artifact = fit_relation_normalization(
        view.tokens,
        view.mask,
        view.jet_ids,
        normalization_contract=load_hashed_json(root / "inputs" / "normalization_contract.json"),
        relation_registry=load_hashed_json(root / "registry" / "relation_family_registry.json"),
        raw_input_schema=load_hashed_json(root / "inputs" / "raw_input_schema.json"),
        source_manifest_sha256=campaign["split_manifest"]["file_sha256"],
        input_view="offline",
        input_binding_sha256=binding["content_hash"],
        model_train_content_sha256=expected,
    )
    write_immutable_json(root / "inputs" / "relation_normalization.json", artifact)
    print(artifact["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
