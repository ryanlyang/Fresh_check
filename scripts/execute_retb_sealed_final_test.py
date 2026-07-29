#!/usr/bin/env python3
"""Execute all locked final-test rows and publish one sealed evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    build_final_test_execution_claim,
    build_sealed_final_test_evaluation,
    publish_final_test_evaluation,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    execute_plan_steps,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    FINAL_TEST_LABEL_MANIFEST_CONTRACT,
    SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT,
    build_final_test_prediction,
    validate_final_labels_manifest,
    validate_sealed_final_test_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    lock = load_hashed_json(
        args.execution_lock,
        expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    )
    plan = load_hashed_json(
        args.execution_plan,
        expected_contract=SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT,
    )
    validate_sealed_final_test_execution_plan(
        plan,
        execution_lock=lock,
        campaign_source=campaign["source"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    if lock.get("source") != campaign["source"]:
        raise ValueError("sealed final-test source differs")
    preexisting = [
        path
        for step in plan["steps"]
        for path in step["expected_outputs"]
        if Path(path).exists()
    ]
    if preexisting:
        raise FileExistsError(
            "sealed final-test execution outputs predate the execution claim"
        )
    claim = bind_source(
        build_final_test_execution_claim(
            execution_lock=lock,
            execution_plan_sha256=plan["content_hash"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    claim_publication = write_immutable_json(
        args.output_dir / "retb_final_test_execution_claim.json", claim
    )
    if claim_publication["status"] != "published":
        raise RuntimeError(
            "final-test execution was already claimed; refusing repeat access"
        )
    receipts = execute_plan_steps(
        plan["steps"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
        forbidden_terms=(),
        forbidden_entrypoints=frozenset(
            {
                "evaluate_retb_final_test.py",
                "execute_retb_sealed_final_test.py",
            }
        ),
    )
    labels_manifest_path = Path(plan["final_labels_manifest"])
    labels_manifest = load_hashed_json(
        labels_manifest_path,
        expected_contract=FINAL_TEST_LABEL_MANIFEST_CONTRACT,
    )
    validate_final_labels_manifest(
        labels_manifest, manifest_path=labels_manifest_path
    )
    if labels_manifest.get("source") != campaign["source"]:
        raise ValueError("final-test labels source differs")
    labels_path = (
        labels_manifest_path.parent / labels_manifest["npz_filename"]
    )
    with np.load(labels_path, allow_pickle=False) as payload:
        identities = np.asarray(payload["identities"]).tolist()
        labels = np.asarray(payload["labels"])
    prediction_rows = []
    prediction_publications = []
    for record in plan["prediction_rows"]:
        npz_path = Path(record["inference_output_npz"])
        if not npz_path.is_file() or npz_path.is_symlink():
            raise FileNotFoundError("final-test inference output is absent")
        encoded = npz_path.read_bytes()
        with np.load(npz_path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "identities",
                "logits",
                "probabilities",
            }:
                raise ValueError("final-test inference NPZ fields differ")
            row_identities = np.asarray(payload["identities"]).tolist()
            logits = np.asarray(payload["logits"])
            probabilities = np.asarray(payload["probabilities"])
        if row_identities != identities:
            raise ValueError("final-test prediction identity order differs")
        prediction = bind_source(
            build_final_test_prediction(
                row=record,
                execution_lock=lock,
                identities=identities,
                logits=logits,
                probabilities=probabilities,
                npz_filename=npz_path.name,
                npz_sha256=hashlib.sha256(encoded).hexdigest(),
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        prediction_publications.append(
            write_immutable_json(
                record["prediction_manifest_output"], prediction
            )
        )
        prediction_rows.append(
            {
                "row_id": record["row_id"],
                "graph_id": record["graph_id"],
                "pipeline_seed": int(record["pipeline_seed"]),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "degradation_profile_sha256": lock[
                    "degradation_profile_sha256"
                ],
                "identities": identities,
                "logits": logits,
                "probabilities": probabilities,
                "prediction_artifact_sha256": prediction[
                    "content_hash"
                ],
            }
        )
    artifact = build_sealed_final_test_evaluation(
        execution_lock=lock,
        execution_claim=claim,
        identities=identities,
        labels=labels,
        final_labels_artifact_sha256=labels_manifest["content_hash"],
        prediction_rows=prediction_rows,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = publish_final_test_evaluation(
        output_dir=args.output_dir, artifact=artifact
    )
    print(
        json.dumps(
            {
                "sealed_final_test_evaluation_sha256": artifact[
                    "content_hash"
                ],
                "execution_receipts": receipts,
                "execution_claim_publication": claim_publication,
                "prediction_publications": prediction_publications,
                "evaluation_publication": publication,
                "evaluation_count": 1,
                "test_result_selected_replacement": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
