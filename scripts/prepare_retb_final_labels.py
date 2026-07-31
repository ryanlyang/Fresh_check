#!/usr/bin/env python3
"""Publish canonical final-test labels after the immutable execution claim."""

from __future__ import annotations

import argparse
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
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    FINAL_TEST_EXECUTION_CLAIM_CONTRACT,
    FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    validate_final_test_execution_claim,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    publish_final_labels_manifest,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--execution-claim", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    lock = load_hashed_json(
        args.execution_lock,
        expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    )
    claim = load_hashed_json(
        args.execution_claim,
        expected_contract=FINAL_TEST_EXECUTION_CLAIM_CONTRACT,
    )
    validate_final_test_execution_claim(claim, execution_lock=lock)
    if (
        lock.get("source") != campaign["source"]
        or claim.get("source") != campaign["source"]
    ):
        raise ValueError("final label access lineage differs")
    authorize_dataset_access(
        worker_role="final_test_worker",
        requested_resource="final_test_targets",
    )
    with np.load(
        root / "inputs" / "offline" / "final_test" / "offline_inputs.npz",
        allow_pickle=False,
    ) as payload:
        identities = [
            str(value) for value in payload["identities"].tolist()
        ]
        labels = np.asarray(payload["labels"], dtype=np.int64)
    order = np.argsort(np.asarray(identities, dtype=np.str_))
    publish_final_labels_manifest(
        output_dir=args.output_dir,
        identities=[identities[int(index)] for index in order],
        labels=labels[order],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
