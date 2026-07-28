#!/usr/bin/env python3
"""Publish an audited RETB offline-target cache from generated target arrays."""

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
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    load_frozen_token_head_reproducer,
    publish_offline_target_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--specification", required=True, type=Path)
    parser.add_argument("--generated-targets", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-map",
        required=True,
        type=Path,
        help="JSON object mapping every expert ID to its checkpoint path.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    specification = load_hashed_json(args.specification)
    checkpoint_map = json.loads(args.checkpoint_map.read_text("utf-8"))
    if set(checkpoint_map) != set(EXPERT_ORDER):
        raise ValueError("checkpoint map expert coverage differs")
    payload = np.load(args.generated_targets, allow_pickle=False)
    required = {"identities", "labels"} | {
        f"{kind}_{expert}"
        for expert in EXPERT_ORDER
        for kind in ("tokens", "logits")
    }
    if set(payload.files) != required:
        raise ValueError("generated-target NPZ fields differ")
    identities = tuple(str(value) for value in payload["identities"].tolist())
    labels = np.asarray(payload["labels"], dtype=np.int64)
    reproducers = {}
    for index, expert in enumerate(EXPERT_ORDER):
        descriptor = specification["target_descriptors"][expert]
        reproducers[expert] = load_frozen_token_head_reproducer(
            checkpoint_path=checkpoint_map[expert],
            expected_checkpoint_sha256=descriptor["checkpoint_sha256"],
            target_mode=specification["target_tuple"][index],
            token_dimension=specification["allocation"][expert][1],
        )

    def generate(start: int, stop: int):
        return {
            "tokens": {
                expert: payload[f"tokens_{expert}"][start:stop]
                for expert in EXPERT_ORDER
            },
            "expert_logits": {
                expert: payload[f"logits_{expert}"][start:stop]
                for expert in EXPERT_ORDER
            },
        }

    manifest = publish_offline_target_cache(
        output_dir=args.output_dir,
        specification=specification,
        identities=identities,
        labels=labels,
        generator=generate,
        logit_reproducers=reproducers,
        source_snapshot=source_snapshot(REPO_ROOT),
        shard_size=args.shard_size,
    )
    print(
        json.dumps(
            {
                "target_cache_manifest_sha256": manifest["content_hash"],
                "event_count": manifest["event_count"],
                "shard_count": manifest["shard_count"],
                "storage_audit_hashes": manifest["storage_audit_hashes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
