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
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    build_prelock_final_test_inputs,
    validate_prelock_final_test_inputs,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    publish_shared_deployable_inference_payload,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from scripts.prepare_retb_deployable_input import (  # noqa: E402
    build_hlt_only_inference_inputs,
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
    configuration = load_hashed_json(
        args.configuration,
        expected_contract="retb_prelock_final_input_configuration_v1",
    )
    values = {
        name: configuration[name]
        for name in (
            "split_manifest_sha256",
            "degradation_profile_sha256",
            "input_hashes",
        )
    }
    if set(values) != {
        "split_manifest_sha256",
        "degradation_profile_sha256",
        "input_hashes",
    } or configuration.get("source") != campaign.get("source"):
        raise ValueError("prelock final input configuration differs")
    shared_root = args.campaign_root / "inputs" / "stage_n" / "shared"
    shared = {}
    if not args.dry_run:
        for split in ("stack_val", "final_test"):
            identities, hlt_inputs = build_hlt_only_inference_inputs(
                campaign_root=args.campaign_root,
                split=split,
            )
            shared[split] = publish_shared_deployable_inference_payload(
                output_dir=shared_root,
                split=split,
                identities=identities,
                hlt_inputs=hlt_inputs,
                source_snapshot=source_snapshot(REPO_ROOT),
            )
        values["input_hashes"] = dict(values["input_hashes"])
        values["input_hashes"]["final_test_HLT_inputs"] = shared[
            "final_test"
        ]["manifest"]["content_hash"]
    artifact = bind_source(
        build_prelock_final_test_inputs(
            campaign_spec_sha256=campaign["content_hash"],
            **values,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_prelock_final_test_inputs(artifact)
    result = {
        "dry_run": args.dry_run,
        "prelock_final_inputs_sha256": artifact["content_hash"],
        "checkpoint_loading_allowed": False,
        "model_outputs_emitted": False,
        "shared_HLT_payloads": {
            split: publication["manifest"]["content_hash"]
            for split, publication in shared.items()
        },
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
