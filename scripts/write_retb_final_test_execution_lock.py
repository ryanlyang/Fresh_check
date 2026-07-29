#!/usr/bin/env python3
"""Write the second RETB lock authorizing scientific final-test inference."""

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
    FINAL_TEST_INPUT_PREPARATION_CONTRACT,
    FINALIST_CONTROLS_CONTRACT,
    POSTLOCK_ORACLE_TARGET_CONTRACT,
    build_final_test_execution_lock,
    validate_final_test_execution_lock,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    LOCKED_SCALE_FINALISTS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-finalists", required=True, type=Path)
    parser.add_argument("--finalist-controls", required=True, type=Path)
    parser.add_argument("--prelock-final-inputs", required=True, type=Path)
    parser.add_argument("--postlock-target", action="append", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    finalists = load_hashed_json(
        args.locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    controls = load_hashed_json(
        args.finalist_controls,
        expected_contract=FINALIST_CONTROLS_CONTRACT,
    )
    prelock_inputs = load_hashed_json(
        args.prelock_final_inputs,
        expected_contract=FINAL_TEST_INPUT_PREPARATION_CONTRACT,
    )
    targets = [
        load_hashed_json(
            path, expected_contract=POSTLOCK_ORACLE_TARGET_CONTRACT
        )
        for path in args.postlock_target
    ]
    configuration = json.loads(args.configuration.read_text("utf-8"))
    if (
        set(configuration) != {"parent_hashes", "final_input_hashes"}
        or any(
            row.get("source") != campaign.get("source")
            for row in (finalists, controls, prelock_inputs, *targets)
        )
        or configuration["parent_hashes"].get(
            "prelock_final_test_inputs"
        )
        != prelock_inputs["content_hash"]
        or configuration["final_input_hashes"]
        != prelock_inputs["input_hashes"]
    ):
        raise ValueError("final execution-lock source/configuration differs")
    artifact = bind_source(
        build_final_test_execution_lock(
            locked_scale_finalists=finalists,
            finalist_controls=controls,
            postlock_targets=targets,
            **configuration,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_final_test_execution_lock(
        artifact,
        locked_scale_finalists=finalists,
        finalist_controls=controls,
        postlock_targets=targets,
    )
    result = {
        "dry_run": args.dry_run,
        "final_test_execution_lock_sha256": artifact["content_hash"],
        "scientific_final_test_inference_authorized": True,
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
