#!/usr/bin/env python3
"""Label-auditor-only deterministic HOSD target-shuffle plan producer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_target_shuffle_plan,
    load_and_validate_campaign,
    load_hashed_json,
    load_target_cache,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TARGET_CACHE_SPEC_CONTRACT,
    canonical_sha256,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--canonical-cache", required=True, type=Path)
    parser.add_argument("--label-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--shuffle-kind", required=True, choices=("global", "within_class")
    )
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    spec = load_hashed_json(
        args.canonical_cache / "cache_spec.json",
        expected_contract=TARGET_CACHE_SPEC_CONTRACT,
    )
    cache = load_target_cache(args.canonical_cache, cache_spec=spec)
    raw = json.loads(args.label_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("identity_to_label"), dict):
        raise ValueError("label manifest requires identity_to_label object")
    label_hash = raw.get("content_hash") or canonical_sha256(raw)
    labels = []
    for identity in cache.identities:
        if identity not in raw["identity_to_label"]:
            raise ValueError("label manifest lacks a target-cache identity")
        labels.append(int(raw["identity_to_label"][identity]))
    if set(raw["identity_to_label"]) != set(cache.identities):
        raise ValueError("label manifest identity population is not exact")
    outputs = {}
    for target_id in cache.manifest["persisted_target_ids"]:
        plan = build_target_shuffle_plan(
            cache.identities,
            labels=labels,
            target_id=target_id,
            split=cache.manifest["split"],
            shuffle_kind=args.shuffle_kind,
            label_manifest_sha256=label_hash,
            canonical_cache_manifest_sha256=cache.manifest["content_hash"],
            source=campaign["source"],
        )
        write_immutable_json(args.output_dir / f"{target_id}.json", plan)
        outputs[target_id] = plan["content_hash"]
    print(json.dumps({"shuffle_kind": args.shuffle_kind, "plans": outputs, "labels_stored_in_plans": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
