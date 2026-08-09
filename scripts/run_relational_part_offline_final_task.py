#!/usr/bin/env python3
"""Evaluate one locked offline RPT checkpoint on offline final_test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part.contracts import load_hashed_json, sha256_file, with_content_hash, write_immutable_json  # noqa: E402
from teacher_logit_reco.relational_part.evaluation import collect_model_predictions, evaluate_logits  # noqa: E402
from teacher_logit_reco.relational_part.offline_transfer import OFFLINE_TRANSFER_FINAL_LOCK_CONTRACT, OFFLINE_TRANSFER_FINAL_TASK_REGISTRY_CONTRACT  # noqa: E402
from teacher_logit_reco.relational_part.runtime import build_final_test_loader, build_runtime_model  # noqa: E402


def _write_predictions(path: Path, predictions, metadata) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("offline final prediction artifact already exists")
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(
                stream,
                logits=np.asarray(predictions["logits"], dtype=np.float32),
                labels=np.asarray(predictions["labels"], dtype=np.int16),
                predictions=np.asarray(predictions["predictions"], dtype=np.int16),
                event_identities=np.asarray(predictions["event_identities"], dtype=np.str_),
                metadata=np.asarray([json.dumps(metadata, sort_keys=True, separators=(",", ":"))]),
            )
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    registry = load_hashed_json(root / "final_test" / "task_registry.json", expected_contract=OFFLINE_TRANSFER_FINAL_TASK_REGISTRY_CONTRACT)
    lock = load_hashed_json(root / "selection" / "locked_finalists.json", expected_contract=OFFLINE_TRANSFER_FINAL_LOCK_CONTRACT)
    if registry["lock_sha256"] != lock["content_hash"]:
        raise ValueError("offline final registry differs from lock")
    if args.task_index < 0 or args.task_index >= len(registry["tasks"]):
        raise IndexError("offline final task index differs")
    task = registry["tasks"][args.task_index]
    run_id, seed = task["run_id"], int(task["seed"])
    locked = next(row for row in lock["locked_rows"] if row["run_id"] == run_id and int(row["seed"]) == seed)
    registration = load_hashed_json(locked["checkpoint_registration_path"])
    if registration["content_hash"] != locked["checkpoint_registration_sha256"] or registration.get("offline_tagger_inference") is not True:
        raise ValueError("offline checkpoint registration differs from lock")
    checkpoint_path = Path(locked["checkpoint_path"])
    if sha256_file(checkpoint_path) != locked["checkpoint_sha256"]:
        raise ValueError("offline checkpoint bytes differ from lock")
    relation = load_hashed_json(root / "inputs" / "relation_normalization.json")
    region = load_hashed_json(root / "inputs" / "region_normalization.json")
    screening = load_hashed_json(root / "registry" / "screening_registry.json")
    model = build_runtime_model(run_id, screening_registry=screening, normalization_artifact=relation, region_normalization_artifact=region, selected_families=task["relation_families"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_contract_sha256") != locked["model_contract_sha256"]:
        raise ValueError("offline checkpoint model contract differs")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader, cache_sha = build_final_test_loader(
        cache_dir=root / "inputs" / "offline_cache",
        seed=seed,
        families=task["relation_families"],
        tree_root=root / "inputs" / "relation_tree_cache",
        input_view="offline",
    )
    if cache_sha != lock["final_test_offline_content_sha256"]:
        raise ValueError("opened offline final_test differs from lock")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model.to(device)
    predictions = collect_model_predictions(model, loader, device=device, required_split="final_test")
    expected_event_count = int(
        load_hashed_json(root / "inputs" / "offline_cache_binding.json")[
            "splits"
        ]["final_test"]["event_count"]
    )
    if len(predictions["labels"]) != expected_event_count:
        raise ValueError("offline final-test event count differs from lock binding")
    metrics = evaluate_logits(predictions["logits"], predictions["labels"], split="final_test")
    output = root / task["output_dir"]
    prediction_metadata = with_content_hash(
        {
            "contract": "relational_part_offline_final_predictions_v1",
            "schema_version": 1,
            "run_id": run_id,
            "seed": seed,
            "lock_sha256": lock["content_hash"],
            "checkpoint_sha256": locked["checkpoint_sha256"],
            "event_count": len(predictions["labels"]),
            "event_identity_sha256": predictions["event_identity_sha256"],
            "input_view": "offline",
        }
    )
    prediction_sha = _write_predictions(output / "predictions.npz", predictions, prediction_metadata)
    result = with_content_hash(
        {
            "contract": "relational_part_offline_final_evaluation_v1",
            "schema_version": 1,
            "run_id": run_id,
            "seed": seed,
            "lock_sha256": lock["content_hash"],
            "checkpoint_sha256": locked["checkpoint_sha256"],
            "offline_final_test_content_sha256": cache_sha,
            "input_view": "offline",
            "final_test_used_for_selection": False,
            "metrics": metrics,
            "prediction_file_sha256": prediction_sha,
            "prediction_metadata": prediction_metadata,
        }
    )
    write_immutable_json(output / "metrics.json", result)
    print(result["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
