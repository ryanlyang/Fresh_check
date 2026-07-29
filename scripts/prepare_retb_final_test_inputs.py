#!/usr/bin/env python3
"""Attest checkpoint-free prelock final-test input preparation."""

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
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    build_prelock_final_test_inputs,
    validate_prelock_final_test_inputs,
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
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="final_test_input_preparation",
        requested_resource="final_test_inputs",
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    if set(configuration) != {
        "split_manifest_sha256",
        "degradation_profile_sha256",
        "input_hashes",
    }:
        raise ValueError("prelock final input configuration differs")
    artifact = bind_source(
        build_prelock_final_test_inputs(
            campaign_spec_sha256=campaign["content_hash"],
            **configuration,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_prelock_final_test_inputs(artifact)
    result = {
        "dry_run": args.dry_run,
        "prelock_final_inputs_sha256": artifact["content_hash"],
        "checkpoint_loading_allowed": False,
        "model_outputs_emitted": False,
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
