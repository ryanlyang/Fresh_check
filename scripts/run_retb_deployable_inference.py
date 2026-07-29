#!/usr/bin/env python3
"""Run a selected RETB deployable graph on authenticated HLT-only inputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    canonical_sha256,
    load_hashed_json,
    write_immutable_bytes,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    DEPLOYABLE_EXPORT_CONTRACT,
    load_deployable_retb_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    FINAL_TEST_EXECUTION_CLAIM_CONTRACT,
    FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    validate_final_test_execution_claim,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    SCALE_COMPLETION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    DEPLOYABLE_INFERENCE_INPUT_CONTRACT,
    SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT,
    validate_deployable_inference_input,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("deployable inference identities differ")
    identities = [str(item) for item in value]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("deployable inference identities are not canonical")
    return identities


def _batch_value(value: Any, start: int, stop: int, total: int, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _batch_value(child, start, stop, total, device)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _batch_value(child, start, stop, total, device) for child in value
        )
    if isinstance(value, list):
        if len(value) != total:
            raise ValueError("HLT inference list leading dimension differs")
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or int(value.shape[0]) != total:
            raise ValueError("HLT inference array leading dimension differs")
        return torch.from_numpy(value[start:stop]).to(device)
    if torch.is_tensor(value):
        if value.ndim == 0 or int(value.shape[0]) != total:
            raise ValueError("HLT inference tensor leading dimension differs")
        return value[start:stop].to(device)
    raise TypeError("HLT inference inputs must contain only arrays or tensors")


def run_deployable_inference(
    *,
    graph: Any,
    hlt_inputs: Mapping[str, Any],
    identities: list[str],
    batch_size: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically score one canonical identity sequence."""

    if batch_size <= 0 or not identities:
        raise ValueError("deployable inference batch contract differs")
    graph.to(device).eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(identities), batch_size):
            stop = min(start + batch_size, len(identities))
            batch = _batch_value(
                hlt_inputs, start, stop, len(identities), device
            )
            logits = graph(hlt_inputs=batch)["logits"].float()
            if (
                logits.shape != (stop - start, 10)
                or not bool(torch.isfinite(logits).all())
            ):
                raise FloatingPointError("deployable inference logits differ")
            chunks.append(logits.cpu())
    logits = torch.cat(chunks, dim=0)
    probabilities = torch.softmax(logits, dim=1)
    return (
        logits.numpy().astype(np.float32, copy=False),
        probabilities.numpy().astype(np.float32, copy=False),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--deployable-export", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("stack_val", "final_test"))
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--scale-completion", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--execution-claim", type=Path)
    parser.add_argument("--execution-plan", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    if args.split == "stack_val":
        if (
            args.scale_completion is None
            or args.execution_lock is not None
            or args.execution_claim is not None
            or args.execution_plan is not None
        ):
            raise ValueError("stack-val deployable inference authorization differs")
        authorize_dataset_access(
            worker_role="stage_n_selection_inference",
            requested_resource="stack_val_features",
        )
        completion = load_hashed_json(
            args.scale_completion, expected_contract=SCALE_COMPLETION_CONTRACT
        )
        if completion.get("source") != campaign["source"]:
            raise ValueError("scale-completion source differs")
    else:
        if (
            args.scale_completion is not None
            or args.execution_lock is None
            or args.execution_claim is None
            or args.execution_plan is None
        ):
            raise ValueError("final-test deployable inference authorization differs")
        authorize_dataset_access(
            worker_role="final_test_worker",
            requested_resource="final_test_inputs",
        )
        lock = load_hashed_json(
            args.execution_lock, expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT
        )
        claim = load_hashed_json(
            args.execution_claim, expected_contract=FINAL_TEST_EXECUTION_CLAIM_CONTRACT
        )
        validate_final_test_execution_claim(
            claim, execution_lock=lock
        )
        plan = load_hashed_json(
            args.execution_plan,
            expected_contract=SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT,
        )
        if (
            lock.get("source") != campaign["source"]
            or claim.get("source") != campaign["source"]
            or plan.get("source") != campaign["source"]
            or claim["execution_plan_sha256"] != plan["content_hash"]
            or plan["final_test_execution_lock_sha256"]
            != lock["content_hash"]
        ):
            raise ValueError("final-test execution claim lineage differs")

    manifest = load_hashed_json(
        args.input_manifest,
        expected_contract=DEPLOYABLE_INFERENCE_INPUT_CONTRACT,
    )
    validate_deployable_inference_input(
        manifest, manifest_path=args.input_manifest
    )
    if (
        manifest["source"] != campaign["source"]
        or manifest["split"] != args.split
        or manifest["graph_id"] != args.graph_id
        or int(manifest["pipeline_seed"]) != args.pipeline_seed
    ):
        raise ValueError("deployable inference-input lineage differs")
    export = load_hashed_json(
        args.deployable_export, expected_contract=DEPLOYABLE_EXPORT_CONTRACT
    )
    if export.get("source") != campaign["source"]:
        raise ValueError("deployable export source differs")
    graph = load_deployable_retb_graph(
        args.deployable_export, expected_source=campaign["source"]
    )
    payload_path = args.input_manifest.parent / manifest["payload_filename"]
    if _file_sha256(payload_path) != manifest["payload_sha256"]:
        raise ValueError("deployable inference payload changed")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {"identities", "hlt_inputs"}:
        raise ValueError("deployable inference payload fields differ")
    identities = _identity_list(payload["identities"])
    if (
        len(identities) != int(manifest["identity_count"])
        or canonical_sha256(identities) != manifest["identity_order_sha256"]
        or not isinstance(payload["hlt_inputs"], Mapping)
    ):
        raise ValueError("deployable inference payload identities differ")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    logits, probabilities = run_deployable_inference(
        graph=graph,
        hlt_inputs=payload["hlt_inputs"],
        identities=identities,
        batch_size=args.batch_size,
        device=device,
    )
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        identities=np.asarray(identities, dtype=np.str_),
        logits=logits,
        probabilities=probabilities,
    )
    publication = write_immutable_bytes(args.output, stream.getvalue())
    print(
        json.dumps(
            {
                "split": args.split,
                "graph_id": args.graph_id,
                "pipeline_seed": args.pipeline_seed,
                "identity_count": len(identities),
                "deployable_export_sha256": export["content_hash"],
                "input_manifest_sha256": manifest["content_hash"],
                "output_sha256": publication["file_sha256"],
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
