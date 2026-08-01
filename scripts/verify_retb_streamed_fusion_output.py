#!/usr/bin/env python3
"""Fail closed when a streamed Stage-C fusion output is incomplete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.step5 import (  # noqa: E402
    resolve_stage_c_run,
    validate_stage_c_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_abc import (  # noqa: E402
    STREAMED_ABC_FUSION_RECEIPT_CONTRACT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--expected-output", action="append", type=Path, required=True
    )
    args = parser.parse_args()
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_c_runs.json"
    )
    validate_stage_c_run_registry(registry)
    run = resolve_stage_c_run(registry, run_id=args.run_id)
    missing = [str(path) for path in args.expected_output if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"streamed fusion outputs are absent: {missing}")
    receipt = load_hashed_json(
        args.campaign_root
        / "registry"
        / "streamed_fusion_waves"
        / str(run["configuration"]["shape_id"])
        / f"seed_{int(run['seed'])}.json",
        expected_contract=STREAMED_ABC_FUSION_RECEIPT_CONTRACT,
    )
    if (
        args.run_id not in receipt["run_ids"]
        or receipt.get("source") != campaign.get("source")
    ):
        raise ValueError("streamed fusion receipt lineage differs")
    for path in args.expected_output:
        relative = str(path.resolve().relative_to(args.campaign_root.resolve()))
        if receipt["persistent_output_hashes"].get(relative) != _sha256(path):
            raise ValueError("streamed fusion output hash differs from receipt")
        if path.suffix == ".json":
            payload = load_hashed_json(path)
            source = payload.get("source")
            if source is not None and source != campaign.get("source"):
                raise ValueError("streamed fusion output source differs")
    print(
        json.dumps(
            {
                "status": "streamed_fusion_output_verified",
                "run_id": run["run_id"],
                "output_count": len(args.expected_output),
                "receipt_sha256": receipt["content_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
