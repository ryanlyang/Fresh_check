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
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    build_final_test_execution_claim,
    build_sealed_final_test_evaluation,
    publish_final_test_evaluation,
    validate_final_test_execution_claim,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    execute_plan_steps,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    FINAL_TEST_INFERENCE_ATTESTATION_CONTRACT,
    FINAL_TEST_LABEL_MANIFEST_CONTRACT,
    SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT,
    build_final_test_prediction,
    validate_final_labels_manifest,
    validate_final_test_inference_attestation,
    validate_sealed_final_test_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _safe_row_id(value: object) -> str:
    return "".join(
        item if item.isalnum() or item in {"-", "_"} else "_"
        for item in str(value)
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
    claim = bind_source(
        build_final_test_execution_claim(
            execution_lock=lock,
            execution_plan_sha256=plan["content_hash"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    claim_path = args.output_dir / "retb_final_test_execution_claim.json"
    if claim_path.is_file():
        existing_claim = load_hashed_json(
            claim_path, expected_contract=claim["contract"]
        )
        validate_final_test_execution_claim(
            existing_claim, execution_lock=lock
        )
        if existing_claim != claim:
            raise ValueError("existing final-test execution claim differs")
        claim = existing_claim
        claim_publication = {
            "status": "already_present",
            "path": str(claim_path.resolve()),
            "content_hash": claim["content_hash"],
        }
    else:
        claim_publication = write_immutable_json(claim_path, claim)
        if claim_publication["status"] != "published":
            raise RuntimeError("final-test execution claim publication failed")
    for record in plan["prediction_rows"]:
        completion_path = (
            args.output_dir
            / "row_completions"
            / f"{_safe_row_id(record['row_id'])}.json"
        )
        if not completion_path.is_file():
            continue
        completion = load_hashed_json(
            completion_path,
            expected_contract="retb_final_test_row_completion_v3",
        )
        npz_path = Path(record["inference_output_npz"])
        attestation_path = Path(record["inference_attestation_output"])
        manifest_path = Path(record["prediction_manifest_output"])
        attestation = load_hashed_json(
            attestation_path,
            expected_contract=FINAL_TEST_INFERENCE_ATTESTATION_CONTRACT,
        )
        validate_final_test_inference_attestation(
            attestation,
            row=record,
            execution_lock_sha256=lock["content_hash"],
            execution_claim_sha256=claim["content_hash"],
            execution_plan_sha256=plan["content_hash"],
            locked_final_test_hlt_inputs_sha256=lock[
                "final_input_hashes"
            ]["final_test_HLT_inputs"],
            npz_path=npz_path,
            expected_source=campaign["source"],
        )
        if (
            completion.get("execution_claim_sha256")
            != claim["content_hash"]
            or completion.get("execution_plan_sha256")
            != plan["content_hash"]
            or completion.get("row_id") != record["row_id"]
            or not npz_path.is_file()
            or not manifest_path.is_file()
            or hashlib.sha256(npz_path.read_bytes()).hexdigest()
            != completion.get("inference_output_npz_sha256")
            or load_hashed_json(manifest_path)["content_hash"]
            != completion.get("prediction_manifest_sha256")
            or attestation["content_hash"]
            != completion.get("inference_attestation_sha256")
        ):
            raise ValueError(
                "completed final-test row cannot be reexecuted after drift"
            )
    # A row without a completion may still have finished inference before an
    # outer-worker interruption.  Only a valid claim/plan/checkpoint-bound
    # attestation allows execute_plan_steps to reuse that NPZ.  A bare NPZ
    # deliberately leaves the attestation output absent and therefore reruns
    # the inference command, whose immutable publication verifies the bytes.
    for record in plan["prediction_rows"]:
        npz_path = Path(record["inference_output_npz"])
        attestation_path = Path(record["inference_attestation_output"])
        if not attestation_path.is_file():
            continue
        attestation = load_hashed_json(
            attestation_path,
            expected_contract=FINAL_TEST_INFERENCE_ATTESTATION_CONTRACT,
        )
        validate_final_test_inference_attestation(
            attestation,
            row=record,
            execution_lock_sha256=lock["content_hash"],
            execution_claim_sha256=claim["content_hash"],
            execution_plan_sha256=plan["content_hash"],
            locked_final_test_hlt_inputs_sha256=lock[
                "final_input_hashes"
            ]["final_test_HLT_inputs"],
            npz_path=npz_path,
            expected_source=campaign["source"],
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
    row_completion_publications = []
    for record in plan["prediction_rows"]:
        npz_path = Path(record["inference_output_npz"])
        if not npz_path.is_file() or npz_path.is_symlink():
            raise FileNotFoundError("final-test inference output is absent")
        encoded = npz_path.read_bytes()
        attestation = load_hashed_json(
            record["inference_attestation_output"],
            expected_contract=FINAL_TEST_INFERENCE_ATTESTATION_CONTRACT,
        )
        validate_final_test_inference_attestation(
            attestation,
            row=record,
            execution_lock_sha256=lock["content_hash"],
            execution_claim_sha256=claim["content_hash"],
            execution_plan_sha256=plan["content_hash"],
            locked_final_test_hlt_inputs_sha256=lock[
                "final_input_hashes"
            ]["final_test_HLT_inputs"],
            npz_path=npz_path,
            expected_source=campaign["source"],
        )
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
        prediction_manifest_path = Path(record["prediction_manifest_output"])
        prediction_publications.append(
            write_immutable_json(prediction_manifest_path, prediction)
        )
        safe_row_id = _safe_row_id(record["row_id"])
        completion = bind_source(
            with_content_hash(
                {
                    "contract": "retb_final_test_row_completion_v3",
                    "schema_version": 3,
                    "row_id": str(record["row_id"]),
                    "graph_id": str(record["graph_id"]),
                    "pipeline_seed": int(record["pipeline_seed"]),
                    "final_test_execution_lock_sha256": lock[
                        "content_hash"
                    ],
                    "execution_claim_sha256": claim["content_hash"],
                    "execution_plan_sha256": plan["content_hash"],
                    "checkpoint_sha256": require_sha256(
                        record["checkpoint_sha256"],
                        name="checkpoint_sha256",
                    ),
                    "inference_output_npz_sha256": prediction[
                        "npz_sha256"
                    ],
                    "inference_attestation_sha256": attestation[
                        "content_hash"
                    ],
                    "prediction_manifest_sha256": prediction[
                        "content_hash"
                    ],
                    "completed_row_must_not_be_reexecuted": True,
                    "incomplete_other_rows_may_resume": True,
                }
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        completion_path = (
            args.output_dir / "row_completions" / f"{safe_row_id}.json"
        )
        if completion_path.is_file():
            existing_completion = load_hashed_json(
                completion_path,
                expected_contract="retb_final_test_row_completion_v3",
            )
            if existing_completion != completion:
                raise ValueError("final-test row completion differs")
        row_completion_publications.append(
            write_immutable_json(completion_path, completion)
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
                "row_completion_publications": row_completion_publications,
                "evaluation_publication": publication,
                "evaluation_count": 1,
                "completed_rows_reused_without_inference": sum(
                    1 for receipt in receipts if receipt["reused"]
                ),
                "incomplete_rows_resumable_under_same_claim": True,
                "test_result_selected_replacement": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
