#!/usr/bin/env python3
"""Evaluate all and only execution-locked RETB rows on final_test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    build_sealed_final_test_evaluation,
    publish_final_test_evaluation,
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
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--final-labels-npz", required=True, type=Path)
    parser.add_argument("--final-labels-artifact-sha256", required=True)
    parser.add_argument("--prediction-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="final_test_worker",
        requested_resource="final_test_inputs",
    )
    authorize_dataset_access(
        worker_role="final_test_worker",
        requested_resource="final_test_targets",
    )
    execution_lock = load_hashed_json(
        args.execution_lock,
        expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    )
    if execution_lock.get("source") != campaign.get("source"):
        raise ValueError("final-test execution-lock source differs")
    with np.load(args.final_labels_npz, allow_pickle=False) as payload:
        if set(payload.files) != {"identities", "labels"}:
            raise ValueError("final labels NPZ fields differ")
        identities = np.asarray(payload["identities"]).tolist()
        labels = np.asarray(payload["labels"])
    index = json.loads(args.prediction_index.read_text("utf-8"))
    if not isinstance(index, list):
        raise ValueError("final prediction index must be a list")
    rows = []
    required = {
        "row_id",
        "graph_id",
        "pipeline_seed",
        "checkpoint_sha256",
        "degradation_profile_sha256",
        "prediction_artifact_sha256",
        "prediction_npz",
    }
    for item in index:
        if set(item) != required:
            raise ValueError("final prediction index row differs")
        path = Path(item["prediction_npz"])
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("final prediction NPZ is absent or unsafe")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "identities",
                "logits",
                "probabilities",
            }:
                raise ValueError("final prediction NPZ fields differ")
            row = {
                key: item[key] for key in required - {"prediction_npz"}
            }
            row.update(
                {
                    "identities": np.asarray(payload["identities"]).tolist(),
                    "logits": np.asarray(payload["logits"]),
                    "probabilities": np.asarray(payload["probabilities"]),
                }
            )
            rows.append(row)
    artifact = build_sealed_final_test_evaluation(
        execution_lock=execution_lock,
        identities=identities,
        labels=labels,
        final_labels_artifact_sha256=args.final_labels_artifact_sha256,
        prediction_rows=rows,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": args.dry_run,
        "sealed_final_test_evaluation_sha256": artifact["content_hash"],
        "evaluated_row_count": len(artifact["evaluated_row_ids"]),
        "test_result_selected_replacement": False,
    }
    if not args.dry_run:
        result["publication"] = publish_final_test_evaluation(
            output_dir=args.output_dir, artifact=artifact
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
